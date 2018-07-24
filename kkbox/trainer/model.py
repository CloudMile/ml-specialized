import tensorflow as tf, os, shutil, re
from tensorflow.python.feature_column import feature_column
from tensorflow.contrib.nn import alpha_dropout

from . import app_conf, input, metadata
from .utils import utils

class Model(object):

    def __init__(self, model_dir):
        """

        :param model_dir:
        """
        self.p = app_conf.instance
        self.model_dir = model_dir
        # self.feature = Feature.instance
        self.mapper = utils.read_pickle(f'{self.p.fitted_path}/stats.pkl')
        pass

    def model_fn(self, features, labels, mode):
        with tf.variable_scope("init") as scope:
            with tf.device('/cpu:0'):
                norm_init_fn = tf.glorot_normal_initializer()
                uniform_init_fn = tf.glorot_uniform_initializer()
                self.b_global = tf.Variable(uniform_init_fn(shape=[]), name="b_global")
                # Embedding init
                with tf.variable_scope("embedding"):
                    self.emb = {}
                    for colname, dim in metadata.EMB_COLS.items():
                        n_unique = len(self.mapper[colname].classes_)
                        self.emb[colname] = self.get_embedding_var(
                            [n_unique, dim], uniform_init_fn, f'emb_{colname}')

        with tf.variable_scope("members") as scope:
            with tf.device('/cpu:0'):
                self.city = tf.nn.embedding_lookup(self.emb['city'], features['city'])
                self.gender = tf.nn.embedding_lookup(self.emb['gender'], features['gender'])
                self.registered_via = tf.nn.embedding_lookup(self.emb['registered_via'], features['registered_via'])
                self.registration_init_time = features['registration_init_time'][:, tf.newaxis]
                self.expiration_date = features['expiration_date'][:, tf.newaxis]
                self.msno_age_catg = tf.nn.embedding_lookup(self.emb['msno_age_catg'], features['msno_age_catg'])
                self.msno_age_num = features['msno_age_num'][:, tf.newaxis]
                self.msno_tenure = features['msno_tenure'][:, tf.newaxis]

            with tf.device('/gpu:0'):
                self.weighted_sum(features, 'song_id', 'msno_pos_query_hist', ['msno_pos_query_count'])
                self.weighted_sum(features, 'song_id', 'msno_neg_query_hist', ['msno_neg_query_count'])
                self.weighted_sum(features, 'artist_name', 'msno_artist_name_hist', ['msno_artist_name_count', 'msno_artist_name_mean'])
                self.weighted_sum(features, 'composer', 'msno_composer_hist', ['msno_composer_count', 'msno_composer_mean'])
                self.weighted_sum(features, 'genre_ids', 'msno_genre_ids_hist', ['msno_genre_ids_count', 'msno_genre_ids_mean'])
                self.weighted_sum(features, 'language', 'msno_language_hist', ['msno_language_count', 'msno_language_mean'])
                self.weighted_sum(features, 'lyricist', 'msno_lyricist_hist', ['msno_lyricist_count', 'msno_lyricist_mean'])
                self.weighted_sum(features, 'source_screen_name', 'msno_source_screen_name_hist', ['msno_source_screen_name_count'])
                self.weighted_sum(features, 'source_system_tab', 'msno_source_system_tab_hist', ['msno_source_system_tab_count'])
                self.weighted_sum(features, 'source_type', 'msno_source_type_hist', ['msno_source_type_count'])
                members_concat_feats = [
                    self.city, self.gender, self.registered_via, self.registration_init_time, self.expiration_date,
                    self.msno_age_catg, self.msno_age_num, self.msno_tenure,
                    self.msno_pos_query_hist_count,
                    self.msno_neg_query_hist_count,
                    self.msno_artist_name_hist_count, self.msno_artist_name_hist_mean,
                    self.msno_composer_hist_count, self.msno_composer_hist_mean,
                    self.msno_genre_ids_hist_count, self.msno_genre_ids_hist_mean,
                    self.msno_language_hist_count, self.msno_language_hist_mean,
                    self.msno_lyricist_hist_count, self.msno_lyricist_hist_mean,
                    self.msno_source_screen_name_hist_count,
                    self.msno_source_system_tab_hist_count,
                    self.msno_source_type_hist_count
                ]
                self.members_feature = tf.concat(members_concat_feats, 1, name='members_feature')
                print(f'self.members_feature: {self.members_feature}')

        with tf.variable_scope("songs") as scope:
            with tf.device('/cpu:0'):
                self.song_id = tf.nn.embedding_lookup(self.emb['song_id'], features['song_id'])
                self.language = tf.nn.embedding_lookup(self.emb['language'], features['language'])
                self.song_cc = tf.nn.embedding_lookup(self.emb['song_cc'], features['song_cc'])
                self.song_xxx = tf.nn.embedding_lookup(self.emb['song_xxx'], features['song_xxx'])
                self.song_yy = features['song_yy'][:, tf.newaxis]
                self.song_length = features['song_length'][:, tf.newaxis]
                self.song_pplrty = features['song_pplrty'][:, tf.newaxis]
                self.song_clicks = features['song_clicks'][:, tf.newaxis]

            with tf.device('/gpu:0'):
                self.song_weighted_sum(features, 'genre_ids', 'genre_ids', ['song_genre_ids_len'], is_seq=True)
                self.song_weighted_sum(features, 'artist_name', 'artist_name', ['song_artist_name_len'], is_seq=True)
                self.song_weighted_sum(features, 'composer', 'composer', ['song_composer_len'], is_seq=True)
                self.song_weighted_sum(features, 'lyricist', 'lyricist', ['song_lyricist_len'], is_seq=True)
                self.song_weighted_sum(features, 'city', 'song_city_hist', ['song_city_count', 'song_city_mean'])
                self.song_weighted_sum(features, 'gender', 'song_gender_hist', ['song_gender_count', 'song_gender_mean'])
                self.song_weighted_sum(features, 'msno_age_catg', 'song_msno_age_catg_hist',
                                  ['song_msno_age_catg_count', 'song_msno_age_catg_mean'])
                self.song_weighted_sum(features, 'registered_via', 'song_registered_via_hist',
                                  ['song_registered_via_count'])
                self.song_weighted_sum(features, 'source_screen_name', 'song_source_screen_name_hist',
                                  ['song_source_screen_name_count'])
                self.song_weighted_sum(features, 'source_system_tab', 'song_source_system_tab_hist',
                                  ['song_source_system_tab_count'])
                self.song_weighted_sum(features, 'source_type', 'song_source_type_hist',
                                  ['song_source_type_count'])

                songs_concat_feats = [
                    self.song_id, self.language, self.song_cc, self.song_xxx, self.song_yy,
                    self.song_length, self.song_pplrty, self.song_clicks,
                    self.genre_ids, self.artist_name,self.composer, self.lyricist,
                    self.song_city_hist_count, self.song_city_hist_mean,
                    self.song_gender_hist_count, self.song_gender_hist_mean,
                    self.song_msno_age_catg_hist_count, self.song_msno_age_catg_hist_mean,
                    self.song_registered_via_hist_count,
                    self.song_source_screen_name_hist_count,
                    self.song_source_system_tab_hist_count,
                    self.song_source_type_hist_count
                ]
                self.songs_feature = tf.concat(songs_concat_feats, 1, name='songs_feature')
                print(f'self.songs_feature: {self.songs_feature}')

        """'source_system_tab', 'source_screen_name', 'source_type'"""
        with tf.variable_scope("context"):
            with tf.device('/cpu:0'):
                self.source_system_tab = tf.nn.embedding_lookup(self.emb['source_system_tab'], features['source_system_tab'])
                self.source_screen_name = tf.nn.embedding_lookup(self.emb['source_screen_name'], features['source_screen_name'])
                self.source_type = tf.nn.embedding_lookup(self.emb['source_type'], features['source_type'])
                self.context_features = tf.concat(
                    [self.source_system_tab, self.source_screen_name, self.source_type], 1, name='context_features')
                print(f'self.context_features: {self.context_features}')

        with tf.variable_scope("gmf"):
            pass

        with tf.variable_scope("dnn"):
            with tf.device('/gpu:0'):
                net = tf.concat([self.members_feature, self.songs_feature, self.context_features], 1)
                print(f'net: {net}')
                net = tf.layers.dense(net, 512, kernel_initializer=norm_init_fn, activation=tf.nn.selu)
                net = tf.layers.dense(net, 128, kernel_initializer=norm_init_fn, activation=tf.nn.selu)
                net = tf.layers.dense(net, 64, kernel_initializer=norm_init_fn, activation=tf.nn.selu)
                self.logits = tf.layers.dense(net, 1, kernel_initializer=norm_init_fn, activation=None)

                self.pred = tf.nn.sigmoid(self.logits, name='pred')

        # Provide an estimator spec for `ModeKeys.PREDICT`
        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                'outputs': tf.estimator.export.PredictOutput({
                    'predictions': self.pred
                })
            }
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=self.pred,
                                              export_outputs=export_outputs)

        with tf.variable_scope("loss"):
            self.labels = tf.to_float(labels[:, tf.newaxis])
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope("metrics") as scope:
            self.auc = tf.metrics.auc(tf.cast(self.labels, tf.bool), tf.reshape(self.pred, [-1]))
            tf.summary.scalar('auc', self.auc)

        self.train_op = None
        self.global_step = tf.train.get_or_create_global_step()
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope("train"):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_op = tf.train.AdamOptimizer().minimize(self.loss, self.global_step)
                    # self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops={'auc': self.auc},
            evaluation_hooks=[])

    def weighted_sum(self, features, dict_key, base_key, weighted_key: list):
        """Do weighted sum to multivariate column, just like tf.nn.embedding_lookup_sparse
         with combiner='sqrtn'

        :param features: data tensor from tf.data.Dataset api
        :param dict_key: embedding key in model.emb
        :param base_key: target variable to do weighted sum
        :param weighted_key: weight variable key list, at least on must be provided
        :return:
        """
        setattr(self, base_key, tf.nn.embedding_lookup(self.emb[dict_key], features[base_key]))
        # print(f'self.{base_key}: {getattr(self, base_key)}')
        for w_key in weighted_key:
            setattr(self, w_key, tf.nn.l2_normalize(features[w_key], 1)[:, :, tf.newaxis])
            # print(f'self.{w_key}: {getattr(self, w_key)}')
            tail = w_key.replace(base_key.replace('_hist', ''), '')
            name = f'{base_key}{tail}'
            val = tf.reduce_sum(getattr(self, base_key) * getattr(self, w_key), 1, name=name)
            setattr(self, name, val)
        #     print(f'self.{name}: {getattr(self, name)}')
        # print()

    def song_weighted_sum(self, features, dict_key, base_key, weighted_key: list, is_seq=False):
        """Do weighted sum to multivariate column, just like tf.nn.embedding_lookup_sparse
         with combiner='sqrtn'

        :param features: data tensor from tf.data.Dataset api
        :param dict_key: embedding key in model.emb
        :param base_key: target variable to do weighted sum
        :param weighted_key: weight variable key list, at least on must be provided
        :param is_seq: is weights an array length description, if so,
        :return:
        """
        if not is_seq:
            self.weighted_sum(features, dict_key, base_key, weighted_key)
        else:
            setattr(self, base_key, tf.nn.embedding_lookup(self.emb[dict_key], features[base_key]))
            # # todo hack
            # print(f'self.{base_key}: {getattr(self, base_key)}')

            w_key = weighted_key[0]
            setattr(self, w_key,
                    tf.nn.l2_normalize(tf.sequence_mask(features[w_key], dtype=tf.float32), 1)[:, :, tf.newaxis])
            # # todo hack
            # print(f'self.{w_key}: {getattr(self, w_key)}')

            name = base_key
            val = tf.reduce_sum(getattr(self, base_key) * getattr(self, w_key), 1, name=name)
            setattr(self, name, val)
            # # todo hack
            # print(f'self.{name}: {getattr(self, name)}')
            # print()

    def get_embedding_var(self, shape, init_fn, name, zero_first=True):
        if zero_first:
            return tf.concat([
                tf.Variable(tf.zeros([1, shape[1]]), trainable=False),
                tf.Variable(init_fn(shape=shape), name=name)
            ], 0)
        else:
            return tf.Variable(init_fn(shape=[shape[0] + 1, shape[1]]), name=name)


    def get_estimator(self, config:tf.estimator.RunConfig):
        est = tf.estimator.Estimator(model_fn=self.model_fn, model_dir=self.model_dir, config=config)

        # Create directory for export, it will raise error if in GCS environment
        try:
            os.makedirs(f'{self.p.model_dir}/export/{self.p.export_name}', exist_ok=True)
        except Exception as e:
            print(e)

        # def metric_fn(labels, predictions):
        #     """ Defines extra evaluation metrics to canned and custom estimators.
        #     By default, this returns an empty dictionary
        #
        #     Args:
        #         labels: A Tensor of the same shape as predictions
        #         predictions: A Tensor of arbitrary shape
        #     Returns:
        #         dictionary of string:metric
        #     """
        #     metrics = {}
        #     pred_values = predictions['predictions']
        #     metrics["auc"] = tf.metrics.auc(labels, pred_values)
        #     return metrics
        #
        # est = tf.contrib.estimator.add_metrics(est, metric_fn)
        print(f"creating a regression model: {est}")
        return est

class BestScoreExporter(tf.estimator.Exporter):
    logger = utils.logger('BestScoreExporter')

    def __init__(self,
                 name,
                 serving_input_receiver_fn,
                 assets_extra=None,
                 as_text=False):
        self._name = name
        self.serving_input_receiver_fn = serving_input_receiver_fn
        self.assets_extra = assets_extra
        self.as_text = as_text
        self.best = None
        self._exports_to_keep = 1
        self.export_result = None
        self.logger.info('BestScoreExporter init')

    @property
    def name(self):
        return self._name

    def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):

        self.logger.info(f'eval_result: {eval_result}')
        curloss = eval_result['loss']
        if self.best is None or self.best >= curloss:
            # Clean first, only keep the best weights
            self.logger.info(f'clean export_path: {export_path}')
            try:
                shutil.rmtree(export_path)
            except Exception as e:
                self.logger.warn(e)

            os.makedirs(export_path, exist_ok=True)

            self.best = curloss
            self.logger.info('nice eval loss: {}, export to pb'.format(curloss))
            self.export_result = estimator.export_savedmodel(
                export_path,
                self.serving_input_receiver_fn,
                assets_extra=self.assets_extra,
                as_text=self.as_text,
                checkpoint_path=checkpoint_path)
        else:
            self.logger.info('bad eval loss: {}'.format(curloss))

        return self.export_result

class Feature(object):
    instance = None

    def __init__(self):
        self.p = app_conf.instance

    def extend_feature_columns(self, feature_columns):
        """ Use to define additional feature columns, such as bucketized_column(s), crossed_column(s),
        and embedding_column(s). task.HYPER_PARAMS can be used to parameterise the creation
        of the extended columns (e.g., embedding dimensions, number of buckets, etc.).

        Default behaviour is to return the original feature_columns list as-is.

        Args:
            feature_columns: {column_name: tf.feature_column} - dictionary of base feature_column(s)
        Returns:
            {string: tf.feature_column}: extended feature_column(s) dictionary
        """
        # size = np.prod([len(values) for values in metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()])

        # interactions_feature = tf.feature_column.crossed_column(
        #     keys=metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys(),
        #     hash_bucket_size=10000
        # )
        #
        # interactions_feature_embedded = tf.feature_column.embedding_column(interactions_feature,
        #                                                                    dimension=task.HYPER_PARAMS.embedding_size)
        #
        # feature_columns['interactions_feature_embedded'] = interactions_feature_embedded
        dims = {
            # 'store_open': 16,
            'day_of_week': 8,
            # 'promo': 8,
            'state_holiday': 8,
            # 'school_holiday': 8,
            'month': 8,
            'day': 8,
            'state': 8,
            'store': 16,
            # 'year': 16,
            'assortment': 8,
            'store_type': 8
        }
        for name in metadata.INPUT_CATEGORICAL_FEATURE_NAMES:
            if name in ('promo', 'promo2', 'open', 'school_holiday'):
                feature_columns[name] = tf.feature_column.indicator_column(
                    feature_columns[name]
                )
            else:
                feature_columns[name] = tf.feature_column.embedding_column(
                    feature_columns[name],
                    dims[name]
                )

        return feature_columns


    def create_feature_columns(self):
        """Creates tensorFlow feature_column(s) based on the metadata of the input features.

            The tensorFlow feature_column objects are created based on the data types of the features
            defined in the metadata.py module.

            The feature_column(s) are created based on the input features,
            and the constructed features (process_features method in input.py), during reading data files.
            Both type of features (input and constructed) should be defined in metadata.py.

            Extended features (if any) are created, based on the base features, as the extend_feature_columns
            method is called, before the returning complete the feature_column dictionary.

            Returns:
              {string: tf.feature_column}: dictionary of name:feature_column .
            """

        # load the numeric feature stats (if exists)
        feature_stats = input.load_feature_stats()

        # all the numerical features including the input and constructed ones
        numeric_feature_names = set(metadata.INPUT_NUMERIC_FEATURE_NAMES + metadata.CONSTRUCTED_NUMERIC_FEATURE_NAMES)

        # create t.feature_column.numeric_column columns without scaling
        if feature_stats is None:
            numeric_columns = {feature_name: tf.feature_column.numeric_column(feature_name, normalizer_fn=None)
                               for feature_name in numeric_feature_names}

        # create t.feature_column.numeric_column columns with scaling
        else:
            numeric_columns = {}
            for feature_name in numeric_feature_names:
                try:
                    # standard scaling
                    mean = feature_stats[feature_name]['mean']
                    stdv = feature_stats[feature_name]['stdv']
                    normalizer_fn = lambda x: (x - mean) / stdv

                    # max_min scaling
                    # min_value = feature_stats[feature_name]['min']
                    # max_value = feature_stats[feature_name]['max']
                    # normalizer_fn = lambda x: (x-min_value)/(max_value-min_value)

                    numeric_columns[feature_name] = tf.feature_column.numeric_column(feature_name,
                                                                                     normalizer_fn=normalizer_fn)
                except:
                    numeric_columns[feature_name] = tf.feature_column.numeric_column(feature_name,
                                                                                     normalizer_fn=None)

        # all the categorical features with identity including the input and constructed ones
        categorical_feature_names_with_identity = metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY
        categorical_feature_names_with_identity.update(metadata.CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY)

        # create tf.feature_column.categorical_column_with_identity columns
        categorical_columns_with_identity = \
            {item[0]: tf.feature_column.categorical_column_with_identity(item[0], item[1])
             for item in categorical_feature_names_with_identity.items()}

        # create tf.feature_column.categorical_column_with_vocabulary_list columns
        categorical_columns_with_vocabulary = \
            {item[0]: tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1])
             for item in metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()}

        # create tf.feature_column.categorical_column_with_hash_bucket columns
        categorical_columns_with_hash_bucket = \
            {key: tf.feature_column.categorical_column_with_hash_bucket(
                key,
                val['bucket_size'],
                tf.as_dtype(val['dtype']) if 'dtype' in val else tf.string)
             for key, val in metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.items()}

        # this will include all the feature columns of various types
        feature_columns = {}

        if numeric_columns is not None:
            feature_columns.update(numeric_columns)

        if categorical_columns_with_identity is not None:
            feature_columns.update(categorical_columns_with_identity)

        if categorical_columns_with_vocabulary is not None:
            feature_columns.update(categorical_columns_with_vocabulary)

        if categorical_columns_with_hash_bucket is not None:
            feature_columns.update(categorical_columns_with_hash_bucket)

        # add extended feature_column(s) before returning the complete feature_column dictionary
        return self.extend_feature_columns(feature_columns)

    def get_deep_and_wide_columns(self, feature_columns):
        """Creates deep and wide feature_column lists.

        Given a list of feature_column(s), each feature_column is categorised as either:
        1) dense, if the column is tf.feature_column._NumericColumn or feature_column._EmbeddingColumn,
        2) categorical, if the column is tf.feature_column._VocabularyListCategoricalColumn or
        tf.feature_column._BucketizedColumn, or
        3) sparse, if the column is tf.feature_column._HashedCategoricalColumn or tf.feature_column._CrossedColumn.

        If use_indicators=True, then categorical_columns are converted into indicator_columns, and used as dense features
        in the deep part of the model. if use_wide_columns=True, then categorical_columns are used as sparse features
        in the wide part of the model.

        deep_columns = dense_columns + indicator_columns
        wide_columns = categorical_columns + sparse_columns

        Args:
            feature_columns: [tf.feature_column] - A list of tf.feature_column objects.
        Returns:
            [tf.feature_column],[tf.feature_column]: deep and wide feature_column lists.
        """
        dense_columns = list(
            filter(lambda column: isinstance(column, feature_column._NumericColumn) |
                                  isinstance(column, feature_column._EmbeddingColumn),
                   feature_columns)
        )

        categorical_columns = list(
            filter(lambda column: isinstance(column, feature_column._VocabularyListCategoricalColumn) |
                                  isinstance(column, feature_column._IdentityCategoricalColumn) |
                                  isinstance(column, feature_column._BucketizedColumn),
                   feature_columns)
        )

        sparse_columns = list(
            filter(lambda column: isinstance(column, feature_column._HashedCategoricalColumn) |
                                  isinstance(column, feature_column._CrossedColumn),
                   feature_columns)
        )

        indicator_columns = []

        encode_one_hot = self.p.encode_one_hot
        as_wide_columns = self.p.as_wide_columns

        # if encode_one_hot=True, then categorical_columns are converted into indicator_column(s),
        # and used as dense features in the deep part of the model.
        # if as_wide_columns=True, then categorical_columns are used as sparse features in the wide part of the model.

        if encode_one_hot:
            indicator_columns = list(
                map(lambda column: tf.feature_column.indicator_column(column),
                    categorical_columns)
            )

        deep_columns = dense_columns + indicator_columns
        wide_columns = sparse_columns + (categorical_columns if as_wide_columns else [])

        return deep_columns, wide_columns

Feature.instance = Feature()

