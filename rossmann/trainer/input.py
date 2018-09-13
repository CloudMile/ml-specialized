import numpy as np, pandas as pd, tensorflow as tf
import json, multiprocessing, shutil

from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta

from . import metadata, app_conf, utils,  model as m

class Input(object):
    """Handle all logic about data pipeline.

    Training period: Clean -> Prepare -> Fit -> Transform -> Split
    Serving period: Clean -> Prepare -> Transform

    In clean step do missing value imputing, maybe some data transformation to string features.
    In prepare step add features if needed and drop useless features.
    In fit step remember the statistical information about numeric data, label mapping about categorical data,
      and all other information to persistent for serving needed.
    In transform steps, transform all features to numeric, like normalize numeric features,
      embedding or one hot encoding categorical features.
    In Split step simple split data to train and valid data, split rule is according to the data,
      usually random split to avoiding model overfitting, here we split by history logs of each user.
    """
    instance = None
    logger = utils.logger(__name__)

    def __init__(self):
        self.p = None # app_conf.get_config()
        self.feature = m.Feature.get_instance()
        self.serving_fn = {
            'json': getattr(self, 'json_serving_input_fn')
            # 'csv': getattr(self, 'csv_serving_fn')
        }

    def clean(self, p, data, is_serving=False):
        """Missing value imputing, maybe some data transformation to string features.

        :param data: Input data, maybe DataFrame or simple file path string
        :param is_serving: True: train or eval period, False: serving period
        :return: Cleaned data
        """
        self.logger.info('Clean start, is_serving: {}'.format(is_serving))
        s = datetime.now()
        raw_dtype = dict(zip(metadata.RAW_HEADER, metadata.RAW_DTYPES))
        if isinstance(data, str):
            data = pd.read_csv(data, dtype=raw_dtype)

        def fill_empty(data):
            for col in ('day_of_week', 'date', 'open', 'promo', 'state_holiday', 'school_holiday'):
                data[col].fillna('', inplace=True)

        ret = None
        # Handle missing value and change column name style to PEP8 and resort columns order
        if not is_serving:
            store = pd.read_csv(p.store_data, dtype=raw_dtype)
            store['CompetitionDistance'].fillna(store.CompetitionDistance.median(), inplace=True)
            store = store.rename(index=str, columns=metadata.HEADER_MAPPING)
            store.to_csv('{}/store.csv'.format(p.cleaned_path), index=False)

            store_state = pd.read_csv(p.store_state, dtype=raw_dtype)
            store_state = store_state.rename(index=str, columns=metadata.HEADER_MAPPING)
            store_state.to_csv('{}/store_states.csv'.format(p.cleaned_path), index=False)

            ret = data.rename(index=str, columns=metadata.HEADER_MAPPING)
            fill_empty(ret)
            ret.to_csv('{}/tr.csv'.format(p.cleaned_path), index=False)
        else:
            ret = data.rename(index=str, columns=metadata.HEADER_MAPPING)
            fill_empty(ret)

        self.logger.info('Clean take time {}'.format(datetime.now() - s))
        return ret

    def prepare(self, p, data, is_serving=False):
        """Add features if needed and drop useless features.

        :param data: Cleaned data with DataFrame type
        :param is_serving: True: train or eval period, False: serving period
        :return: Prepared data with DataFrame type
        """
        self.logger.info('Prepare start, is_serving: {}'.format(is_serving))
        s = datetime.now()

        dtype = self.get_processed_dtype(is_serving=is_serving)
        if isinstance(data, str):
            data = pd.read_csv(data, dtype=dtype)

        # Train, eval
        if not is_serving:
            store = pd.read_csv('{}/store.csv'.format(p.cleaned_path), dtype=dtype)
            # CompetitionOpenSinceMonth, CompetitionOpenSinceYear need to be transform to days count from 1970/01/01
            def map_fn(e):
                y, m = e
                if pd.isna(y) or pd.isna(m): return np.nan
                y, m = int(y), int(m)
                return '{}-{}-1'.format(y, m)
            since_dt = pd.Series(list(zip(store.competition_open_since_year, store.competition_open_since_month)))\
                         .map(map_fn, na_action='ignore')
            store['competition_open_since'] = (pd.to_datetime(since_dt) - datetime(1970, 1, 1)).dt.days
            store['competition_open_since'].fillna(store['competition_open_since'].median(), inplace=True)
            store['competition_open_since_year'].fillna('', inplace=True)
            store['competition_open_since_month'].fillna('', inplace=True)

            # Promo2SinceYear + Promo2SinceWeek need to be transform to days count from 1970/01/01
            def promo2_fn(e):
                y, week = e
                if pd.isna(y) or pd.isna(week):
                    return np.nan
                return datetime.strptime('{}'.format(int(y)), '%Y')

            promo2_dt = pd.Series(list(zip(store.promo2since_year, store.promo2since_week))).map(promo2_fn)
            store['promo2since'] = (promo2_dt - datetime(1970, 1, 1)).dt.days
            store['promo2since'].fillna(store['promo2since'].median(), inplace=True)
            store['promo2since_year'].fillna('', inplace=True)
            store['promo2since_week'].fillna('', inplace=True)

            # Merge store_state to store and persistent
            self.logger.info('Persisten store to {}/store.csv'.format(p.prepared_path))
            store_states = pd.read_csv('{}/store_states.csv'.format(p.cleaned_path), dtype=dtype)
            store = store.merge(store_states, on='store', how='left')
            store.to_csv('{}/store.csv'.format(p.prepared_path), index=False)
        else:
            store = pd.read_csv('{}/store.csv'.format(p.prepared_path), dtype=dtype)

        # Construct year, month, day columns, maybe on specific day or period will has some trends.
        dt = pd.to_datetime(data['date'])
        data['year'] = dt.dt.year.map(str)
        data['month'] = dt.dt.month.map(str)
        data['day'] = dt.dt.day.map(str)
        merge = data.merge(store, how='left', on='store')

        # Calculate real promo2 happened timing
        merge['promo2'] = self.cal_promo2(merge)
        merge = merge.drop('promo_interval', 1)

        # Construct sales mean columns, at least we know whether this store is popular
        if not is_serving:
            sales_mean = merge.groupby('store').sales.mean()
            merge['sales_mean'] = sales_mean.reindex(merge.store).values

            # Add date for split train valid
            merge = merge.query('open == "1" and sales > 0')[['date'] + metadata.HEADER]
            merge.to_csv('{}/tr.csv'.format(p.prepared_path), index=False)
        else:
            with open(p.feature_stats_file, 'r') as fp:
                stats = json.load(fp)
            # Data type of column store is int, but unserialize from json in key will become string type,
            # so change the data type to int
            sales_mean = pd.Series(stats['sales_mean']['hist_mapping'])
            sales_mean = pd.Series(index=sales_mean.index.map(int), data=sales_mean.data)
            merge['sales_mean'] = sales_mean.reindex(merge.store).values

            # dtype.pop(metadata.TARGET_NAME)
            merge = merge[metadata.SERVING_COLUMNS]

        self.logger.info('Prepare take time {}'.format(datetime.now() - s))
        return merge

    def fit(self, p, data):
        """Remember the statistical information about numeric data, label mapping about categorical data,
          and all other information to persistent for serving needed.

        :param data: Cleaned and prepared data with DataFrame type
        :return: self
        """
        stats = defaultdict(defaultdict)
        numeric_feature_names = metadata.INPUT_NUMERIC_FEATURE_NAMES + metadata.CONSTRUCTED_NUMERIC_FEATURE_NAMES
        for name, col in data[numeric_feature_names].iteritems():
            self.logger.info('{}.mean: {}, {}.stdv: {}, '
                             '{}.median: {}'.format(name, col.mean(), name, col.std(), name, col.median()))
            stats[name] = {'mean': col.mean(), 'stdv': col.std(), 'median': col.median()}

        # Dump feature stats
        sales_mean = data.groupby('store').sales.mean()
        with open(p.feature_stats_file, 'w') as fp:
            stats['sales_mean']['hist_mapping'] = sales_mean.to_dict()
            json.dump(stats, fp)
        return self

    def transform(self, p, data, is_serving=False):
        """Transform all features to numeric, like normalize numeric features,
          embedding or one hot encoding categorical features,
          but in this case, we leave it to tf.feature_columns package, only do `np.log1p(target)` here
          for the sake of shrinking scale of sales

        :param data: Cleaned, fitted and prepared data with DataFrame type
        :param is_serving: True: train or eval period, False: serving period
        :return: Transformed data with DataFrame type
        """
        self.logger.info('Transform start, is_serving: {}'.format(is_serving))
        s = datetime.now()

        dtype = self.get_processed_dtype(is_serving=is_serving)
        if isinstance(data, str):
            data = pd.read_csv(data, dtype=dtype)

        if not is_serving:
            self.logger.info('Do np.log(data.{}) !'.format(metadata.TARGET_NAME))
            data[metadata.TARGET_NAME] = np.log1p(data[metadata.TARGET_NAME])
            tf.gfile.Copy('{}/store.csv'.format(p.prepared_path), '{}/store.csv'.format(p.transformed_path))
            # shutil.copy2('{}/store.csv'.format(p.prepared_path), '{}'.format(p.transformed_path))

        # Although we have fill missing value before write to csv, zero-length string values still
        # be treat as NaN value in pandas, dataset api will fill default value but not when serving
        # so here we will fill empty string to categorical feature
        data = self.fill_catg_na(data)

        self.logger.info('Transform take time {}'.format(datetime.now() - s))
        return data

    def fill_catg_na(self, data):
        """Fill empty string value to categorical feature."""
        for col in data.columns:
            if col in metadata.INPUT_CATEGORICAL_FEATURE_NAMES:
                data[col].fillna('', inplace=True)
        return data

    def split(self, p, data):
        """Only necessary in training period, here we split by history logs of each store,
          take latest 30 percent to valid data.

        :param data: Train data for split
        :return: tuple of (train part, valid part)
        """
        self.logger.info('Split start')
        s = datetime.now()
        if isinstance(data, str):
            data = pd.read_csv(data)

        tr, vl = [], []
        for st, df in data.groupby('store'):
            # Split order by date
            df = df.sort_values('date')
            total = len(df)
            cut_pos = total - int(total * p.valid_size)

            tr.append(df[:cut_pos])
            vl.append(df[cut_pos:])

        tr, vl = pd.concat(tr, 0), pd.concat(vl, 0)
        # In order to inspect time series figure, persistent date column of training data,
        tr.date.to_json(p.tr_dt_file)
        vl.date.to_json(p.vl_dt_file)

        tr.drop('date', 1).to_csv(p.train_files, index=False)
        vl.drop('date', 1).to_csv(p.valid_files, index=False)

        self.logger.info('Split take time {}'.format(datetime.now() - s))
        return self

    def cal_promo2(selfself, merge):
        """Calculate real promo2 happened timing, promo2 have periodicity per year,
          e.g: if PromoInterval = 'Jan,Apr,Jul,Oct', means month in 1, 4, 7, 10 in every year will
          have another promotion on some products, so it need to drop origin Promo2
          and recalculate if has any promotion.

        :param merge: Merged data(train tables + store tables)
        :return: `promo2` feature data with pandas.Series type
        """
        map_ = {'Jan': '1', 'Feb': '2', 'Mar': '3', 'Apr': '4',
                'May': '5', 'Jun': '6', 'Jul': '7', 'Aug': '8',
                'Sept': '9', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
        base = np.array(['0'] * len(merge))
        valid_cond = merge.promo2 == '1'
        merge = merge[valid_cond]
        df = pd.DataFrame({'month': merge.month.values,
                           'interval': merge.promo_interval.str.split(',').values})

        cut_idx = np.cumsum(df.interval.map(len).values)
        interval_concat = []
        df.interval.map(interval_concat.extend)
        df['interval'] = np.split(pd.Series(interval_concat).map(map_).values, cut_idx)[:-1]
        base[valid_cond] = pd.Series(list(zip(df.month, df.interval))) \
                             .map(lambda row: str(int(row[0] in row[1])))
        return base

    def get_processed_dtype(self, is_serving=False):
        """Get data type of processed data

        :param is_serving: True: train or eval period, False: serving period
        :return: Dictionary object (label -> data type)
        """
        header = metadata.HEADER if not is_serving else metadata.SERVING_COLUMNS
        columns = metadata.HEADER_DEFAULTS if not is_serving else metadata.SERVING_DEFAULTS
        return dict(zip(
            header,
            [type(e[0]) for e in columns]
        ))

    def load_feature_stats(self, p):
        """
        Load numeric column pre-computed statistics (mean, stdv, min, max, etc.)
        in order to be used for scaling/stretching numeric columns.

        In practice, the statistics of large datasets are computed prior to model training,
        using dataflow (beam), dataproc (spark), BigQuery, etc.

        The stats are then saved to gcs location. The location is passed to package
        in the --feature-stats-file argument. However, it can be a local path as well.

        Returns:
            json object with the following schema: stats['feature_name']['state_name']
        """

        feature_stats = None
        try:
            if p.feature_stats_file is not None and tf.gfile.Exists(p.feature_stats_file):
                with tf.gfile.Open(p.feature_stats_file) as file:
                    content = file.read()
                feature_stats = json.loads(content)
                self.logger.info("Feature stats were successfully loaded from local file...")
            else:
                self.logger.warn("Feature stats file not found. numerical columns will not be normalised...")
        except:
            self.logger.warn("Couldn't load feature stats. numerical columns will not be normalised...")

        return feature_stats

    def process_features(self, features):
        """ Use to implement custom feature engineering logic, e.g. polynomial expansion
        Default behaviour is to return the original feature tensors dictionary as is

        Args:
            features: {string:tensors} - dictionary of feature tensors
        Returns:
            {string:tensors}: extended feature tensors dictionary
        """
        return features

    def json_serving_input_fn(self, p):
        """Declare the serving specification, what data format should receive and how to transform to
          put in model.

        :return: `tf.estimator.export.ServingInputReceiver` object
        """
        self.logger.info('use json_serving_input_fn !')

        feat_cols = self.feature.create_feature_columns(p)
        input_feature_columns = [feat_cols[feature_name] for feature_name in metadata.INPUT_FEATURE_NAMES]

        inputs = {}
        for column in input_feature_columns:
            if column.name in metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY:
                inputs[column.name] = tf.placeholder(shape=[None], dtype=tf.int32, name=column.name)
            else:
                inputs[column.name] = tf.placeholder(shape=[None], dtype=column.dtype, name=column.name)

        features = {
            key: tf.expand_dims(tensor, -1)
            for key, tensor in inputs.items()
        }

        return tf.estimator.export.ServingInputReceiver(
            features=self.process_features(features),
            receiver_tensors=inputs
        )

    def get_features_target_tuple(self, features):
        """ Get a tuple of input feature tensors and target feature tensor.

        Args:
            features: {string:tensors} - dictionary of feature tensors
        Returns:
              {string:tensors}, {tensor} -  input feature tensor dictionary and target feature tensor
        """

        # unused_features = list(set(metadata.HEADER) -
        #                        set(metadata.INPUT_FEATURE_NAMES) -
        #                        {metadata.TARGET_NAME} -
        #                        {metadata.WEIGHT_COLUMN_NAME})
        #
        # # remove unused columns (if any)
        # for column in unused_features:
        #     features.pop(column, None)

        # get target feature
        target = features.pop(metadata.TARGET_NAME)

        return features, target

    def parse_csv(self, csv_row, is_serving=False):
        """Takes the string input tensor (csv) and returns a dict of rank-2 tensors.

        Takes a rank-1 tensor and converts it into rank-2 tensor, with respect to its data type
        (inferred from the metadata)

        :param csv_row: Rank-2 tensor of type string (csv)
        :param is_serving: True: train or eval period, False: serving period
        :return: rank-2 tensor of the correct data type
        """

        self.logger.info('is_serving: {}'.format(is_serving))
        if is_serving:
            column_names = metadata.SERVING_COLUMNS
            defaults = metadata.SERVING_DEFAULTS
        else:
            column_names = metadata.HEADER
            defaults = metadata.HEADER_DEFAULTS
        # tf.expand_dims(csv_row, -1)
        columns = tf.decode_csv(tf.expand_dims(csv_row, -1), record_defaults=defaults)

        features = OrderedDict(zip(column_names, columns))

        return features

    def generate_input_fn(self,
                          file_names_pattern,
                          file_encoding='csv',
                          mode=tf.estimator.ModeKeys.EVAL,
                          skip_header_lines=1,
                          num_epochs=None,
                          batch_size=200,
                          shuffle=False,
                          multi_threading=True,
                          hooks=None):
        """Generates an input function for reading training and evaluation data file(s).
        This uses the tf.data APIs.

        Args:
            file_names_pattern: [str] - file name or file name patterns from which to read the data.
            mode: tf.estimator.ModeKeys - either TRAIN or EVAL.
                Used to determine whether or not to randomize the order of data.
            file_encoding: type of the text files. Can be 'csv' or 'tfrecords'
            skip_header_lines: int set to non-zero in order to skip header lines in CSV files.
            num_epochs: int - how many times through to read the data.
              If None will loop through data indefinitely
            batch_size: int - first dimension size of the Tensors returned by input_fn
            shuffle: whether to shuffle data
            multi_threading: boolean - indicator to use multi-threading or not
            hooks: Implementations of tf.train.SessionHook
        Returns:
            A function () -> (features, indices) where features is a dictionary of
              Tensors, and indices is a single Tensor of label indices.
        """

        def _input_fn():
            # shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
            num_threads = multiprocessing.cpu_count() if multi_threading else 1
            buffer_size = 2 * batch_size + 1

            self.logger.info("")
            self.logger.info("* data input_fn:")
            self.logger.info("================")
            self.logger.info("Mode: {}".format(mode))
            self.logger.info("Input file(s): {}".format(file_names_pattern))
            self.logger.info("Files encoding: {}".format(file_encoding))
            self.logger.info("Batch size: {}".format(batch_size))
            self.logger.info("Epoch count: {}".format(num_epochs))
            self.logger.info("Thread count: {}".format(num_threads))
            self.logger.info("Shuffle: {}".format(shuffle))
            self.logger.info("================")
            self.logger.info("")

            file_names = tf.matching_files(file_names_pattern)

            is_serving = True if mode == tf.estimator.ModeKeys.PREDICT else False
            if file_encoding == 'csv':
                dataset = tf.data.TextLineDataset(filenames=file_names)
                dataset = dataset.skip(skip_header_lines)
                dataset = dataset.map(lambda csv_row: self.parse_csv(csv_row, is_serving=is_serving))
                # Some sales columns of rows equals to zero, filter this out
                if mode == tf.estimator.ModeKeys.EVAL:
                    def filter_fn(feat):
                        # Shape of feat[metadata.TARGET_NAME] is (1,),
                        # it's required to reduce the shape of returned bool value, use tf.squeeze
                        return tf.squeeze(feat[metadata.TARGET_NAME] > 0)
                    dataset = dataset.filter(filter_fn)
            else:
                # dataset = dt.TFRecordDataset(filenames=file_names)
                # dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example),
                #                       num_parallel_calls=num_threads)
                pass

            dataset = dataset.map(lambda features: self.get_features_target_tuple(features),
                                num_parallel_calls=num_threads)\
                             .map(lambda features, target: (self.process_features(features), target),
                                num_parallel_calls=num_threads)

            if shuffle:
                dataset = dataset.shuffle(buffer_size)

            dataset = dataset.batch(batch_size)\
                             .prefetch(buffer_size)\
                             .repeat(num_epochs)

            iterator = dataset.make_one_shot_iterator()
            features, target = iterator.get_next()
            return features, target

        return _input_fn

# Input.instance = Input()
