import numpy as np, pandas as pd, tensorflow as tf
import  multiprocessing, html, shutil, re

from collections import OrderedDict
from sklearn.utils import shuffle as sk_shuffle
from sklearn import preprocessing
from datetime import datetime

from . import metadata, app_conf
from .utils import utils


def sep_fn(e):
    """Callback function for Input.fit, Input.transform function"""
    return e

class Input(object):
    """Handle all logic about data pipeline.

    Training period: Clean -> Split -> Prepare -> Fit -> Transform
    Serving period: Clean -> Prepare -> Transform

    In clean step do missing value imputing, maybe some data transformation to string features.
    In Split step simple split data to train and valid data, split rule is according to the data,
      usually random split to avoiding model overfitting, here we split by history logs of each user.
    In prepare step add features if needed and drop useless features.
    In fit step remember the statistical information about numeric data, label mapping about categorical data,
      and all other information to persistent for serving needed.
    In transform steps, transform all features to numeric, like normalize numeric features,
      embedding or one hot encoding categorical features.
    """
    instance = None
    logger = utils.logger(__name__)

    def __init__(self):
        self.p = app_conf.instance
        # self.feature = m.Feature.instance
        self.serving_fn = {
            'json': getattr(self, 'json_serving_input_fn')
            # 'csv': getattr(self, 'csv_serving_fn')
        }

    def clean(self, data, is_serving=False):
        """Missing value imputing, maybe some data transformation to string features.

        :param data: Input data, maybe DataFrame or simple file path string
        :param is_serving: True: train or eval period, False: serving period
        :return: Cleaned data
        """
        self.logger.info(f'Clean start, is_serving: {is_serving}')
        s = datetime.now()
        if isinstance(data, str):
            data = pd.read_csv(data)

        ret = None
        data['source_system_tab'] = data.source_system_tab.fillna('')
        data['source_screen_name'] = data.source_screen_name.fillna('')
        data['source_type'] = data.source_type.fillna('')

        if is_serving:
            ret = data
        else:
            members = pd.read_csv('./data/members.csv')
            songs = pd.read_csv('./data/songs.csv') \
                      .merge(pd.read_csv('./data/song_extra_info.csv'), how='left', on='song_id') \
                      .drop('name', 1)

            self.logger.info(f'Clean table members.')
            # Clean member table
            members['gender'] = members.gender.fillna('')
            date_fmt = '%Y%m%d'
            members['registration_init_time'] = pd.to_datetime(members.registration_init_time, format=date_fmt)\
                                                  .map(lambda e: e.timestamp() / 86400)
            members['expiration_date'] = pd.to_datetime(members.expiration_date, format=date_fmt)\
                                           .map(lambda e: e.timestamp() / 86400)
            members['registered_via'] = members.registered_via.astype(str)
            members['city'] = members.city.astype(str)

            self.logger.info(f'Clean table songs.')
            # Clean songs table
            def str2tuple(series):
                return series.fillna('') \
                    .map(html.unescape) \
                    .str.replace('[^.\w\|]+', ' ') \
                    .str.replace('^[\s\|]+|[\s\|]+$', '') \
                    .str.split('\s*\|+\s*') \
                    .map(lambda ary: tuple(sorted(ary)))
            songs['genre_ids'] = str2tuple(songs['genre_ids'])
            songs['artist_name'] = str2tuple(songs['artist_name'])
            songs['composer'] = str2tuple(songs['composer'])
            songs['lyricist'] = str2tuple(songs['lyricist'])
            # songs['language'] = songs.language.fillna(0).map(int, na_action='ignore').map(str)
            songs['language'] = songs.language.map(lambda e: str(int(float(e))), na_action='ignore').fillna('')
            members.to_pickle(f'{self.p.cleaned_path}/members.pkl')
            songs.to_pickle(f'{self.p.cleaned_path}/songs.pkl')
            ret = data

        self.logger.info(f'Clean take time {datetime.now() - s}')
        return ret

    def split(self, data):
        """Only necessary in training period, here we split by  history logs of each user,
          take latest 10 percent to valid data.

        Fist we calculate the statistical of user history logs counts
        for model generation, we will drop some user who's number of history logs less than 25% quantile,
        for huge amount history logs user, we take latest data of number of 75% quantile

        :param data: Train data for split
        :return: tuple of (train part, valid part)
        """
        self.logger.info('Split start')
        s = datetime.now()
        if isinstance(data, str):
            data = pd.read_csv(data)

        msno_describe = data.groupby('msno').size().describe()
        per25, per75 = int(msno_describe['25%']), int(msno_describe['75%'])

        def filter_fn(pipe):
            # At least 25 percentile
            if len(pipe) >= per25:
                # At most 75 percentile of latest data
                pipe = pipe[-per75:]
                len_ = len(pipe)
                is_train = np.zeros(len_)
                tr_size = int(len_ * (1 - self.p.valid_size))
                is_train[:tr_size] = 1.
                pipe['is_train'] = is_train
                return pipe

        self.logger.info(f'Msno data distribution \n{msno_describe}\n')
        self.logger.info(f'Filter training data')
        data = data.groupby('msno', as_index=False, sort=False).apply(filter_fn)
        vl = data.query('is_train == 0').drop('is_train', 1).reset_index(drop=True)
        tr = data.query('is_train == 1').drop('is_train', 1).reset_index(drop=True)

        tr.to_pickle(f'{self.p.cleaned_path}/tr.pkl')
        vl.to_pickle(f'{self.p.cleaned_path}/vl.pkl')

        self.logger.info(f'Split take time {datetime.now() - s}')
        return tr, vl

    def flatten(self, data, uni_cols:list, m_col, target):
        """For multivalent feature, which has replica value in a grid, like array,
          in order to calculate the statistical info we need to flatten the data by this feature

          For example, raw data like
          ```
          uni_col | multi_col
          --------|----------
           xxx      (1, 2, 3)
          ```
          which flatten to
          ```
          uni_col | multi_col
          --------|----------
           xxx    |     1
           xxx    |     2
           xxx    |     3
          ```

        :param data: Input data with DataFrame type
        :param uni_cols: Uni-variate features to be replicated
        :param m_col: Multivalent feature
        :param target: Label column
        :return:
        """
        from .utils import utils_nb

        return utils_nb.flatten(data, uni_cols, m_col, target)

    def msno_statis(self, data, col, to_calc, base_msno, is_multi=False):
        """Group by songs to calculate the statistical value with relevant features,
          e.g: count of favorite song_id, mean of favorite song_id, the statistical value usually could
          be weights of relevant feature, in Tensorflow song_id will embedding to vector and count, mean
          could be weights of song_id, so that we can do weighted sum or weighted average to song_id to
          represent a feature of an user, see `tf.embedding_lookup_sparse`.

        :param data: Input data with DataFrame type
        :param col: Feature to calculated in a user
        :param to_calc: Array list, specify which statistical values to compute, e.g: mean, count, max, min
        :param base_msno: To keep the order of user, because the data order may shuffle when calculate.
        :param is_multi: Indicate if col is a multivalent feature
        :return: Dictionary object, structure {key1: [...], key2: [...], ...}
        """
        s = datetime.now()
        label_name = f'msno_{col}_hist'
        calc_names = [f'msno_{col}_{calc}' for calc in to_calc]
        if is_multi:
            data = self.flatten(data, ['msno'], col, 'target')

        series = data.groupby(['msno', col]).target.agg(to_calc).reset_index()

        def map_fn(pipe):
            ret = {label_name: tuple(pipe[col])}
            ret.update({f'msno_{col}_{calc}': tuple(pipe[calc]) for calc in to_calc})
            return ret

        series = series.groupby('msno').apply(map_fn).reindex(base_msno)
        na_conds = series.isna()
        na_value = {label_name: ('',)}
        na_value.update({c: (0.,) for c in calc_names})
        series[na_conds] = [na_value] * len(na_conds)

        self.logger.info(f'{col} {",".join(to_calc)} done, take {datetime.now() - s}')
        return self.extract_col(series)

    def song_statis(self, data, col, to_calc, base_song, is_multi=False):
        """Group by members to calculate the statistical value with relevant features,
          e.g: count of favorite age(10-20, 20-30 ...), mean of favorite age, the statistical value usually could
          be weights of relevant feature, in Tensorflow song_id will embedding to vector and count, mean
          could be the weights of age, so that we can do weighted sum or weighted average to age to
          represent a feature of a song, see `tf.embedding_lookup_sparse`.

        :param data: Input data with DataFrame type
        :param col: Feature to calculated in a user
        :param to_calc: Array list, specify which statistical values to compute, e.g: mean, count, max, min
        :param base_msno: To keep the order of user, because the data order may shuffle when calculate.
        :param is_multi: Indicate if col is a multivalent feature
        :return: Dictionary object, structure {key1: [...], key2: [...], ...}
        """
        s = datetime.now()
        label_name = f'song_{col}_hist'
        calc_names = [f'song_{col}_{calc}' for calc in to_calc]
        if is_multi:
            data = self.flatten(data, ['song_id'], col, 'target')

        series = data.groupby(['song_id', col]).target.agg(to_calc).reset_index()

        def map_fn(pipe):
            ret = {label_name: tuple(pipe[col])}
            ret.update({f'song_{col}_{calc}': tuple(pipe[calc]) for calc in to_calc})
            return ret

        series = series.groupby('song_id').apply(map_fn).reindex(base_song)

        na_conds = series.isna()
        na_value = {label_name: ('',)}
        na_value.update({c: (0.,) for c in calc_names})
        series[na_conds] = [na_value] * len(na_conds)
        ret = self.extract_col(series)

        self.logger.info(f'{col} {",".join(to_calc)} done, take {datetime.now() - s}')
        return ret

    def extract_col(self, series):
        """Transform structure [{}, {}, {}...] to {key1: [], key2: [] ...}

        :param series: pandas.Series object structure [{}, {}, {}...]
        :return: Transformed dictionary object
        """
        ret = {k: [] for k in series.values[0].keys()}
        series.map(lambda dict_: [ret[k].append(v) for k, v in dict_.items()])
        return ret

    def prepare(self, data, is_serving=False):
        """Add features if needed and drop useless features.

        :param data: Cleaned data with DataFrame type
        :param is_serving: True: train or eval period, False: serving period
        :return: Prepared data with DataFrame type
        """
        self.logger.info('Prepare start')
        s = datetime.now()
        if isinstance(data, str):
            data = pd.read_pickle(data)

        ret = None
        if is_serving:
            # members = pd.read_pickle(f'{self.p.prepared_path}/members.pkl')
            # songs = pd.read_pickle(f'{self.p.prepared_path}/songs.pkl')
            ret = data
        else:
            members = pd.read_pickle(f'{self.p.cleaned_path}/members.pkl')
            songs = pd.read_pickle(f'{self.p.cleaned_path}/songs.pkl')

            self.logger.info(f'\nDo prepare_members')
            members = self.prepare_members(data, members, songs)

            self.logger.info(f'\nDo prepare_songs')
            songs = self.prepare_songs(data, members, songs)

            members.to_pickle(f'{self.p.prepared_path}/members.pkl')
            songs.to_pickle(f'{self.p.prepared_path}/songs.pkl')
            # We don't do anything about train, valid files, just copy them form cleaned dir to prepared dir
            shutil.copy2(f'{self.p.cleaned_path}/tr.pkl', f'{self.p.prepared_path}')
            shutil.copy2(f'{self.p.cleaned_path}/vl.pkl', f'{self.p.prepared_path}')
            ret = self
        self.logger.info(f'Prepare take time {datetime.now() - s}')
        return ret


    def prepare_members(self, data, members, songs):
        """Raw data transformation about members table.

          We will write back to members table after adding features, this is for the sake of performance,
          because train data merge profile table (such as members, songs) will replicate the feature in a members
          or in a songs.

        :param data: Cleaned train data with DataFrame type
        :param members: Cleaned members data with DataFrame type
        :param songs: Cleaned songs data with DataFrame type
        :return: Prepared members table with DataFrame type
        """
        data = data.merge(songs, how='left', on='song_id')

        self.logger.info('processing msno_age_catg done, msno_age_num, msno_tenure ...')
        bd = members.bd.copy()
        bins = np.array([6, 10, 20, 30, 40, 60, 80])
        age_map = {0: '', 1: '6-10', 2: '10-20', 3: '20-30', 4: '30-40', 5: '40-60', 6: '60-80', 7: ''}
        members['msno_age_catg'] = pd.Series(np.digitize(bd, bins)).map(age_map)

        reasonable_range = (6 <= bd) & (bd <= 80)
        median = bd[reasonable_range].describe()['50%']
        bd[~reasonable_range] = median
        members['msno_age_num'] = bd
        members = members.drop('bd', 1)
        # Tenure: the customer life time
        members['msno_tenure'] = members.expiration_date - members.registration_init_time

        msno_extra = {}
        base_msno = data.msno.unique()
        self.logger.info('processing msno_pos_query, msno_neg_query ...')
        # Positive query and negative query
        for key in ('pos', 'neg'):
            name = f'{key}_query'
            lable_name, w_name = f'msno_{name}_hist', f'msno_{name}_count'
            query = data.query(f"target == {1 if key == 'pos' else 0}") \
                        .groupby('msno') \
                        .apply(lambda e: {lable_name: tuple(e.song_id), w_name: (1.,) * len(e)}) \
                        .reindex(base_msno)
            na_conds = query.isna()
            query[na_conds] = [{lable_name: ('',), w_name: (0.,)}] * len(na_conds)
            msno_extra.update(self.extract_col(query))

        # Freq distribution of each member interaction with context
        for col in ('source_system_tab', 'source_screen_name', 'source_type'):
            msno_extra.update(self.msno_statis(data, col, ['count'], base_msno))

        # Preference of each member, calculate freq and mean
        msno_extra.update(self.msno_statis(data, 'language', ['count', 'mean'], base_msno))

        for col in ('artist_name', 'composer', 'genre_ids', 'composer', 'lyricist'):
            msno_extra.update(self.msno_statis(data, col, ['count', 'mean'], base_msno, is_multi=True))

        msno_extra = pd.DataFrame(msno_extra)
        msno_extra['msno'] = base_msno
        members = members.merge(msno_extra, how='left', on='msno')

        self.logger.info('prepare members: fill null values made by data merge.')
        # After merge, some members doesn't have any statistic values make the null, so fill default values
        for stats_feat in metadata.MSNO_EXTRA_COLS:  # msno_extra_cols
            if stats_feat in ('msno_age_catg', 'msno_age_num', 'msno_tenure'): continue

            na_conds = members[stats_feat].isna()
            if stats_feat.endswith('_hist'):
                na_value = ('',)
            elif stats_feat.endswith('_count') or stats_feat.endswith('_mean'):
                na_value = (0.,)
            members.loc[na_conds, stats_feat] = members[stats_feat][na_conds].map(lambda na: na_value)

        return members

    def prepare_songs(self, data, members, songs):
        """Raw data transformation about songs table.

          We will write back to songs table after adding features, this is for the sake of performance,
          because train data merge profile table (such as members, songs) will replicate the feature in a members
          or in a songs.

        :param data: Cleaned train data with DataFrame type
        :param members: Cleaned members data with DataFrame type
        :param songs: Cleaned songs data with DataFrame type
        :return: Prepared members table with DataFrame type
        """
        data = data.merge(members, how='left', on='msno')

        # Decode isrc
        songs['song_cc'] = songs.isrc.str.slice(0, 2).fillna('')
        songs['song_xxx'] = songs.isrc.str.slice(2, 5).fillna('')
        songs['song_yy'] = songs.isrc.str.slice(5, 7) \
                                .map(lambda e: 2000 + int(e) if int(e) < 18 else 1900 + int(e),
                                     na_action='ignore')
        songs['song_yy'] = songs.song_yy.fillna(songs.song_yy.median())
        songs['song_yy'] = songs.song_yy - 1900

        songs['song_artist_name_len'] = songs.artist_name.map(len)
        songs['song_composer_len'] = songs.composer.map(len)
        songs['song_lyricist_len'] = songs.lyricist.map(len)
        songs['song_genre_ids_len'] = songs.genre_ids.map(len)
        songs.drop('isrc', 1, inplace=True)

        self.logger.info(f'processing song_clicks, song_pplrty ...')
        base_song = data.song_id.unique()
        basic_stats = data.groupby('song_id').target.agg(['count', 'mean'])
        basic_stats['song_clicks'] = basic_stats['count']
        basic_stats['song_pplrty'] = basic_stats['mean']
        basic_stats = basic_stats.drop(['count', 'mean'], 1).reset_index()

        songs_extra = {}
        # Freq distribution of each song interaction with context
        for col in ('source_system_tab', 'source_screen_name', 'source_type', 'registered_via'):
            songs_extra.update(self.song_statis(data, col, ['count'], base_song))

        # Preference of each song, calculate freq and mean
        for col in ('city', 'gender', 'msno_age_catg'):
            songs_extra.update(self.song_statis(data, col, ['count', 'mean'], base_song))

        songs_extra = pd.DataFrame(songs_extra)
        songs_extra['song_id'] = base_song
        songs = songs.merge(songs_extra, how='left', on='song_id').merge(basic_stats, how='left', on='song_id')

        self.logger.info('prepare songs: fill null values made by data merge.')
        # After merge, some members doesn't have any statistic values make the null, so fill default values
        for stats_feat in metadata.SONG_EXTRA_COLS:
            if stats_feat in ('song_cc', 'song_xxx', 'song_yy', 'song_length',
                              'song_artist_name_len', 'song_composer_len', 'song_lyricist_len', 'song_genre_ids_len'):
                continue

            na_conds = songs[stats_feat].isna()
            if stats_feat.endswith('_hist'):
                songs.loc[na_conds, stats_feat] = songs[stats_feat][na_conds].map(lambda na: ('',))
            elif stats_feat.endswith('_count') or stats_feat.endswith('_mean'):
                songs.loc[na_conds, stats_feat] = songs[stats_feat][na_conds].map(lambda na: (0.,))
            elif stats_feat in ('song_pplrty', 'song_clicks'):
                songs[stats_feat] = songs[stats_feat].fillna(songs[stats_feat].median())
        return songs

    def fit(self, data):
        """Remember the statistical information about numeric data, label mapping about categorical data,
          and all other information to persistent for serving needed.

        :param data: Cleaned and prepared data with DataFrame type
        :return: self
        """
        self.logger.info('Fit start')

        s = datetime.now()
        if isinstance(data, str):
            data = pd.read_pickle(data)

        members = pd.read_pickle(f'{self.p.prepared_path}/members.pkl')
        songs = pd.read_pickle(f'{self.p.prepared_path}/songs.pkl')
        data = data.merge(members, on='msno', how='left').merge(songs, on='song_id', how='left')

        mapper_dict = {}
        # Categorical univariate features
        for uni_catg in ('msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'city',
                         'gender', 'registered_via', 'msno_age_catg', 'language', 'song_cc', 'song_xxx'):
            self.logger.info(f'fit {uni_catg} ...')
            mapper_dict[uni_catg] = utils.CountMapper(outlier='').fit(data[uni_catg])

        # Categorical multivalent features
        for multi_catg in ('genre_ids', 'artist_name', 'composer', 'lyricist'):
            self.logger.info(f'fit {multi_catg} ...')
            mapper_dict[multi_catg] = utils.CountMapper(is_multi=True, sep=sep_fn, outlier='').fit(
                data[multi_catg])

        # Numeric features
        for numeric in metadata.NUMERIC_COLS:
            self.logger.info(f'fit {numeric} ...')
            if numeric in ('registration_init_time', 'expiration_date', 'msno_age_num', 'msno_tenure'):
                mapper_dict[numeric] = utils.NumericMapper(scaler=preprocessing.StandardScaler()).fit(members[numeric])
            elif numeric in ('song_yy', 'song_length'):
                mapper_dict[numeric] = utils.NumericMapper(scaler=preprocessing.StandardScaler()).fit(songs[numeric])
            elif numeric in ('song_pplrty', 'song_clicks'):
                mapper_dict[numeric] = utils.NumericMapper(scaler=preprocessing.StandardScaler()).fit(data[numeric])

        # Persistent the statistic
        utils.write_pickle(f'{self.p.fitted_path}/stats.pkl', mapper_dict)

        self.logger.info(f'Fit take time {datetime.now() - s}')
        return self

    def transform(self, data, is_serving=False):
        """Transform all features to numeric, like normalize numeric features,
          embedding or one hot encoding categorical features.

        :param data: Cleaned, fitted and prepared data with DataFrame type
        :param is_serving: True: train or eval period, False: serving period
        :return: Transformed data with DataFrame type
        """
        self.logger.info('Transform start')

        s = datetime.now()
        if isinstance(data, str):
            data = pd.read_pickle(data)

        ret = None
        mapper_dict = utils.read_pickle(f'{self.p.fitted_path}/stats.pkl')
        # Serving
        if is_serving:
            members = pd.read_pickle(f'{self.p.transformed_path}/members.pkl')
            songs = pd.read_pickle(f'{self.p.transformed_path}/songs.pkl')
            data.insert(0, 'raw_msno', data.pop('msno'))
            data.insert(1, 'raw_song_id', data.song_id)
            for feat in ('song_id', 'source_system_tab', 'source_screen_name', 'source_type'):
                self.logger.info(f'transform {feat}, vocab_key: {feat} ...')
                data[feat] = self._transform_feature(data[feat], mapper_dict[feat])

            self.logger.info('Merge input data ...')
            ret = self.train_merge(data, members, songs, mapper_dict, is_serving=is_serving)

        # Train, eval period
        else:
            tr = data
            vl = pd.read_pickle(f'{self.p.prepared_path}/vl.pkl')
            members = pd.read_pickle(f'{self.p.prepared_path}/members.pkl')
            songs = pd.read_pickle(f'{self.p.prepared_path}/songs.pkl')

            members.insert(0, 'raw_msno', members.pop('msno'))
            for feat in metadata.MEMBER_FEATURES:
                # Univariate features
                if feat in ('city', 'gender', 'registered_via', 'registration_init_time', 'expiration_date',
                            'msno_age_catg', 'msno_age_num', 'msno_tenure'):
                    vocab_key = feat
                    self.logger.info(f'transform {feat}, vocab_key: {vocab_key} ...')
                    members[feat] = self._transform_feature(members[feat], mapper_dict[vocab_key])
                # Multivalent features
                elif feat in ('msno_artist_name_hist', 'msno_composer_hist', 'msno_genre_ids_hist', 'msno_language_hist',
                              'msno_lyricist_hist',
                              'msno_pos_query_hist', 'msno_neg_query_hist', 'msno_source_screen_name_hist',
                              'msno_source_system_tab_hist', 'msno_source_type_hist'):
                    if feat in ('msno_pos_query_hist', 'msno_neg_query_hist'):
                        vocab_key = 'song_id'
                    else:
                        vocab_key = re.sub('^(msno|song)_|_hist$', '', feat)
                    self.logger.info(f'transform {feat}, vocab_key: {vocab_key} ...')
                    # x = pd.Series( utils.transform(members[feat], mapper_dict[vocab_key], is_multi=True, sep=utils.sep_fn) )
                    # max_len = x.map(len).max()
                    # members[feat] = x.map(lambda tp: tp + (0,) * (max_len - len(tp)))
                    members[feat] = self._transform_feature(members[feat], mapper_dict[vocab_key], is_multi=True, sep=sep_fn)
                # elif feat.endswith('_count'):
                #     members[feat] = members[feat].map(lambda tp: ','.join(map(str, tp)) )
                # elif feat.endswith('_mean'):
                #     members[feat] = members[feat].map(lambda tp: ','.join(map('{:.4f}'.format, tp)) )
                # Padding weights Features, xxx_count, xxx_mean... etc.
                else:
                    pass
                    # print(f'transform {feat} (weighted columns) ...')
                    # max_len = int(members[feat].map(len).max())
                    # members[feat] = members[feat].map(lambda tp: tp + (0.,) * (max_len - len(tp)))

            # Songs table transform
            songs.insert(0, 'raw_song_id', songs.song_id)
            for feat in metadata.SONG_FEATURES:
                # if feat in ('song_artist_name_len', 'song_composer_len', 'song_lyricist_len', 'song_genre_ids_len'): continue
                # Univariate features
                if feat in (
                'song_id', 'language', 'song_cc', 'song_xxx', 'song_yy', 'song_length', 'song_pplrty', 'song_clicks'):
                    vocab_key = feat
                    self.logger.info(f'transform {feat}, vocab_key: {vocab_key} ...')
                    songs[feat] = self._transform_feature(songs[feat], mapper_dict[vocab_key])
                # Multivalent features
                elif feat in ('genre_ids', 'artist_name', 'composer', 'lyricist',
                              'song_city_hist', 'song_gender_hist', 'song_msno_age_catg_hist', 'song_registered_via_hist',
                              'song_source_screen_name_hist',
                              'song_source_system_tab_hist', 'song_source_type_hist'):
                    vocab_key = re.sub('^(msno|song)_|_hist$', '', feat)
                    self.logger.info(f'transform {feat}, vocab_key: {vocab_key} ...')
                    # x = pd.Series( utils.transform(songs[feat], mapper_dict[vocab_key], is_multi=True, sep=utils.sep_fn) )
                    # max_len = x.map(len).max()
                    # songs[feat] = x.map(lambda tp: tp + (0,) * (max_len - len(tp)) )
                    songs[feat] = self._transform_feature(songs[feat], mapper_dict[vocab_key], is_multi=True, sep=sep_fn)
                # Transform
                # elif feat.endswith('_count'):
                #     songs[feat] = songs[feat].map(lambda tp: ','.join(map(str, tp)) )
                # elif feat.endswith('_mean'):
                #     songs[feat] = songs[feat].map(lambda tp: ','.join(map('{:.4f}'.format, tp)) )
                else:
                    pass
                    # print(f'transform {feat} (weighted columns) ...')
                    # max_len = int(members[feat].map(len).max())
                    # songs[feat] = songs[feat].map(lambda tp: tp + (0.,) * (max_len - len(tp)))

            # Train, eval table transform
            tr.insert(0, 'raw_msno', tr.pop('msno'))
            tr.insert(1, 'raw_song_id', tr.song_id)
            vl.insert(0, 'raw_msno', vl.pop('msno'))
            vl.insert(1, 'raw_song_id', vl.song_id)
            for feat in ('song_id', 'source_system_tab', 'source_screen_name', 'source_type'):
                self.logger.info(f'transform {feat}, vocab_key: {feat} ...')
                tr[feat] = self._transform_feature(tr[feat], mapper_dict[feat])
                vl[feat] = self._transform_feature(vl[feat], mapper_dict[feat])

            members[['raw_msno'] + metadata.MEMBER_FEATURES].to_pickle(f'{self.p.transformed_path}/members.pkl')
            songs[['raw_song_id'] + metadata.SONG_FEATURES].to_pickle(f'{self.p.transformed_path}/songs.pkl')

            self.logger.info('Merge train data')
            tr = self.train_merge(tr, members, songs, mapper_dict)
            self.logger.info('Merge valid data')
            vl = self.train_merge(vl, members, songs, mapper_dict)

            self.logger.info('Persistent train valid data, maybe take a while ...')
            s_ = datetime.now()
            tr.to_pickle(f'{self.p.transformed_path}/tr.pkl')
            vl.to_pickle(f'{self.p.transformed_path}/vl.pkl')

            self.logger.info(f'Persistent train, valid take time {datetime.now() - s_}')
            ret = self

        self.logger.info(f'Transform take time {datetime.now() - s}')
        return ret

    def _transform_feature(self, y, mapper:utils.BaseMapper, is_multi=False, sep=None):
        """Transform specific features, include univariate and multivalent features.

          For multivalent feature:
          transform
          ```python
          [ ['label1', 'label2', 'label3'],
            ['label2', 'label3'],
            ... ]
          ```
          to
          ```
          [ [1, 2, 3],
            [2, 3],
            ... ]
          ```

        :param y: Input iterable array like data
        :param mapper: Fitted object to catch the transform needed information
        :param is_multi: True for multivalent feature, otherwise univariate
        :param sep: Separate symbol for string feature to split
        :return: Transformed feature
        """
        s = datetime.now()
        def split(inp):
            if callable(sep):
                return inp.map(sep, na_action='ignore')
            else:
                return inp.str.split(f'\s*{re.escape(sep)}\s*')

        y = pd.Series(y)
        ret = None
        if isinstance(mapper, utils.NumericMapper):
            ret = mapper.transform(y)
        else:
            # Transform to string splitted by comma,
            if is_multi:
                y = split(y)
                lens = y.map(lambda ary: len(ary) if type(ary) in (list, tuple) else 1)
                indices = np.cumsum(lens)

                concat = []
                y.map(lambda ary: concat.extend(ary) if type(ary) in (list, tuple) else
                                  concat.append(None))
                y = pd.Series(concat).map(mapper.enc_, na_action='ignore') \
                      .fillna(0).astype(int).values
                ret = pd.Series(np.split(y, indices)[:-1]).map(tuple).values

                # y = pd.Series(concat).map(mapper.enc_, na_action='ignore')\
                #                      .fillna(0).map(lambda e: str(int(e))).values
                # ret = pd.Series(np.split(y, indices)[:-1]).map(','.join).values
            else:
                ret = y.map(mapper.enc_, na_action='ignore').fillna(0).astype(int).values

        self.logger.info(f'transform take time {datetime.now() - s}')
        return ret

    def train_merge(self, inputs, members, songs, mapper_dict, is_serving=False):
        """Merge table with train table + all kind profile tables(members, songs).

          For the sake of convenient to track the data to the raw data, for members and songs table
          we add raw key feature, like msno -> raw_msno, song_id -> raw_song_id, the feature with prefix
          raw_ will not be transformed to numeric to model, just key for merge.

        :param inputs: Cleaned, fitted prepared transformed data with DataFrame type
        :param members: Prepared members table
        :param songs: Prepared songs table
        :param mapper_dict: Information object generated in fit step.
        :param is_serving: train or eval period, False: serving period
        :return: Merged data, aka **Fat Table**
        """
        columns = metadata.HEADER if not is_serving else metadata.SERVING_COLUMNS
        ret = inputs.merge(members, how='left', on='raw_msno') \
                    .merge(songs, how='left', on='raw_song_id', suffixes=('', '_y')) \
                    .drop(['raw_msno', 'raw_song_id', 'song_id_y'], 1)[columns]

        if not is_serving:
            defaults = dict(zip(columns, metadata.HEADER_DEFAULTS))
        else:
            defaults = dict(zip(columns, metadata.SERVING_DEFAULTS))

        def multi_fillna(df, colname):
            na_value = tuple(defaults[colname])
            na_conds = df[colname].isna().values
            df.loc[na_conds, colname] = df[na_conds][colname].map(lambda na: na_value)

        for colname in metadata.MEMBER_FEATURES + metadata.SONG_FEATURES:
            if colname.endswith('_hist') or colname.endswith('_count') or colname.endswith('_mean') or \
                    colname in ('genre_ids', 'artist_name', 'composer', 'lyricist'):
                multi_fillna(ret, colname)
            else:
                if colname in metadata.NUMERIC_COLS:
                    ret[colname] = ret[colname].fillna( mapper_dict[colname].median_ )
                else:
                    ret[colname] = ret[colname].fillna( defaults[colname][0] )
                    if type(defaults[colname][0]) == int:
                        ret[colname] = ret[colname].astype(int)
                # na_value = defaults[colname][0]
                # ret[colname] = ret[colname].fillna(defaults[colname][0])
                # if type(na_value) == int:
                #     ret[colname] = ret[colname].astype(int)
        return ret

    def json_serving_input_fn(self):
        """Declare the serving specification, what data format should receive and how to transform to
          put in model.

        :return: `tf.estimator.export.ServingInputReceiver` object
        """
        self.logger.info(f'use json_serving_input_fn !')

        columns = metadata.SERVING_COLUMNS
        shapes = self.get_shape(is_serving=True)
        dtypes = metadata.SERVING_DTYPES

        inputs = OrderedDict()
        for name, shape, typ in zip(columns, shapes, dtypes):
            # Remember add batch dimension to first position of shape
            inputs[name] = tf.placeholder(shape=[None, None] if len(shape) > 0 else [None], dtype=typ, name=name)

        return tf.estimator.export.ServingInputReceiver(
            features=inputs,
            receiver_tensors=inputs
        )

    def get_multi_cols(self, is_serving=False):
        """Return multivalent feature labels

        :param is_serving: True: train or eval period, False: serving period
        :return: Multivalent feature labels
        """
        columns = metadata.HEADER if not is_serving else metadata.SERVING_COLUMNS
        return list(filter(lambda col: (col.endswith('_hist') or
                                        col.endswith('_count') or col.endswith('_mean') or
                                        col in ('genre_ids', 'artist_name', 'composer', 'lyricist')),
                           columns))

    def get_uni_cols(self, is_serving=False):
        """Return univariate feature labels

        :param is_serving: True: train or eval period, False: serving period
        :return: Multivalent feature labels
        """
        columns = metadata.HEADER if not is_serving else metadata.SERVING_COLUMNS
        multi_cols = self.get_multi_cols()
        return list(filter(lambda col: col not in multi_cols, columns))

    def get_dtype(self, is_serving=False):
        """Return data type of each feature, include added feature.

        :param is_serving: True: train or eval period, False: serving period
        :return: Multivalent feature labels
        """
        columns = metadata.HEADER if not is_serving else metadata.SERVING_COLUMNS
        defaults = metadata.HEADER_DEFAULTS if not is_serving else metadata.SERVING_DEFAULTS
        return dict( map(lambda e: (e[0], type(e[1][0])), list(zip(columns, defaults))) )

    def get_shape(self, is_serving=False):
        """Return data shape of each feature, include added feature.

        :param is_serving: True: train or eval period, False: serving period
        :return: Data shape of each feature, include added feature.
        """
        cols = metadata.SERVING_COLUMNS if is_serving else metadata.HEADER
        shapes = []
        multi_cols = self.get_multi_cols(is_serving)
        for colname in cols:
            if colname in multi_cols:
                shapes.append([None])
            else:
                shapes.append([])
        return tuple(shapes)

    def generate_input_fn(self,
                          inputs,
                          mode=tf.estimator.ModeKeys.EVAL,
                          skip_header_lines=1,
                          num_epochs=1,
                          batch_size=200,
                          shuffle=False,
                          multi_threading=True,
                          hooks=None):
        """Generates an input function for reading training and evaluation data file(s).
        This uses the tf.data APIs.

        Here we abandon the dataset.shuffle function, Tensorflow suggest us to extend the eval frequency
        to avoid reset the data pipeline too early to see the whole train data. so we will shuffle data with
        sklearn.utils.shuffle

        Args:
            inputs: Data with DataFrametype.
            mode: tf.estimator.ModeKeys - either TRAIN or EVAL.
                Used to determine whether or not to randomize the order of data.
            skip_header_lines: int set to non-zero in order to skip header lines in CSV files.
            num_epochs: int - how many times through to read the data.
              If None will loop through data indefinitely
            batch_size: int - first dimension size of the Tensors returned by input_fn
            shuffle: whether to shuffle data
            multi_threading: boolean - indicator to use multi-threading or not
            hooks: Implementations of tf.train.SessionHook
        Returns:
            Tuple of (data_fn, hook), hook for do something in session initialize period
        """
        is_serving = True if mode == tf.estimator.ModeKeys.PREDICT else False
        # Train, Eval
        if not is_serving:
            output_key = tuple(metadata.HEADER)
            output_type = tuple(metadata.HEADER_DTYPES)
        # Prediction
        else:
            output_key = tuple(metadata.SERVING_COLUMNS)
            output_type = tuple(metadata.SERVING_DTYPES)
        output_shape = self.get_shape(is_serving)

        def generate_fn(inputs):
            def ret_fn():
                for row in inputs.itertuples(index=False):
                    yield row

            return ret_fn

        def zip_map(*row):
            ret = OrderedDict(zip(output_key, row))
            if is_serving:
                return ret
            else:
                target = ret.pop(metadata.TARGET_NAME)
                return ret, target

        hook = IteratorInitializerHook()

        def _input_fn():
            # shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
            num_threads = multiprocessing.cpu_count() if multi_threading else 1
            buffer_size = 2 * batch_size + 1

            self.logger.info("")
            self.logger.info("* data input_fn:")
            self.logger.info("================")
            self.logger.info("Mode: {}".format(mode))
            self.logger.info("Batch size: {}".format(batch_size))
            self.logger.info("Epoch count: {}".format(num_epochs))
            self.logger.info("Thread count: {}".format(num_threads))
            self.logger.info("Shuffle: {}".format(shuffle))
            self.logger.info("================")
            self.logger.info("")

            data = inputs
            if shuffle:
                self.logger.info('shuffle data manually.')
                data = inputs.iloc[ sk_shuffle(np.arange(len(inputs))) ]

            dataset = tf.data.Dataset.from_generator(generate_fn(data), output_type, output_shape)
            dataset = dataset.skip(skip_header_lines)
            dataset = dataset.map(zip_map, num_parallel_calls=num_threads)
            # if shuffle:
            #     dataset = dataset.shuffle(buffer_size)
            padded_shapes = OrderedDict(zip(output_key, output_shape))
            if not is_serving:
                padded_shapes = padded_shapes, padded_shapes.pop(metadata.TARGET_NAME)

            dataset = dataset.padded_batch(batch_size, padded_shapes) \
                             .prefetch(buffer_size=tf.contrib.data.AUTOTUNE) \
                             .repeat(num_epochs)

            iterator = dataset.make_initializable_iterator()
            hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer)
            if is_serving:
                # dataset.make_one_shot_iterator()
                features = iterator.get_next()
                return features, None
            else:
                features, target = iterator.get_next()
                return features, target

        return _input_fn, hook

Input.instance = Input()

class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)
