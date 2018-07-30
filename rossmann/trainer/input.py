import numpy as np, pandas as pd, tensorflow as tf
import json, multiprocessing, shutil

from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta

from . import metadata, model as m, app_conf, utils

class Input(object):
    instance = None
    logger = utils.logger(__name__)

    def __init__(self):
        self.p = app_conf.instance
        self.feature = m.Feature.instance
        self.serving_fn = {
            'json': getattr(self, 'json_serving_input_fn'),
            'csv': getattr(self, 'csv_serving_fn')
        }

    def clean(self, data, is_serving=False):
        self.logger.info(f'Clean start, is_serving: {is_serving}')
        s = datetime.now()
        if isinstance(data, str):
            data = pd.read_csv(data)

        ret = None
        # Handle missing value and change column name style to PEP8 and resort columns order
        if not is_serving:
            dtypes = dict(zip(metadata.RAW_HEADER, metadata.RAW_DTYPES))
            store = pd.read_csv(self.p.store_data, dtype=dtypes)
            store['CompetitionDistance'].fillna(store.CompetitionDistance.median(), inplace=True)
            store = store.rename(index=str, columns=metadata.HEADER_MAPPING)
            store.to_csv(f'{self.p.cleaned_path}/store.csv', index=False)

            store_state = pd.read_csv(self.p.store_state, dtype=dtypes)
            store_state = store_state.rename(index=str, columns=metadata.HEADER_MAPPING)
            store_state.to_csv(f'{self.p.cleaned_path}/store_state.csv', index=False)

            ret = data.rename(index=str, columns=metadata.HEADER_MAPPING)
            ret.to_csv(f'{self.p.cleaned_path}/tr.csv', index=False)
        else:
            ret = data.rename(index=str, columns=metadata.HEADER_MAPPING)

        self.logger.info(f'Clean take time {datetime.now() - s}')
        return ret

    def prepare(self, data, is_serving=False):
        self.logger.info(f'Prepare start, is_serving: {is_serving}')
        s = datetime.now()

        dtype = self.get_processed_dtype()
        if isinstance(data, str):
            data = pd.read_csv(data, dtype=dtype)

        store_states = pd.read_csv(f'{self.p.prepared_path}/store_state.csv', dtype=dtype)
        # Train, eval
        if not is_serving:
            store = pd.read_csv(f'{self.p.cleaned_path}/store.csv', dtype=dtype)
            # CompetitionOpenSinceMonth, CompetitionOpenSinceYear need to be transform to days count from 1970/01/01
            def map_fn(e):
                y, m = e
                if pd.isna(y) or pd.isna(m): return np.nan
                # y, m = int(float(y)), int(float(m))
                return f'{y}-{m}-1'

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
                return datetime.strptime(f'{y}', '%Y')
            # (datetime.strptime(f'{y}', '%Y') + timedelta(weeks=int(week)) - datetime(1970, 1, 1)).days

            promo2_dt = pd.Series(list(zip(store.promo2since_year, store.promo2since_week))).map(promo2_fn)
            store['promo2since'] = (promo2_dt - datetime(1970, 1, 1)).dt.days
            store['promo2since'].fillna(store['promo2since'].median(), inplace=True)
            store['promo2since_year'].fillna('', inplace=True)
            store['promo2since_week'].fillna('', inplace=True)

            # Merge store_state to store and persistent
            self.logger.info(f'Persisten store to {self.p.prepared_path}/store.csv')
            store = store.merge(store_states, on='store', how='left')
            store.to_csv(f'{self.p.prepared_path}/store.csv', index=False)
        else:
            store = pd.read_csv(f'{self.p.prepared_path}/store.csv', dtype=dtype)

        # Construct year, month, day columns, maybe on specific day or period will has some trends.
        dt = pd.to_datetime(data['date'])
        data['year'] = dt.dt.year
        data['month'] = dt.dt.month
        data['day'] = dt.dt.day

        merge = data.merge(store, how='left', on='store') # .merge(store_states, how='left', on='store')

        # Calculate real promo2 happened timing, promo2 have periodicity per year,
        # e.g: if PromoInterval = 'Jan,Apr,Jul,Oct', means month in 1, 4, 7, 10 in every year will
        # have another promotion on some products, so it need to drop origin Promo2
        # and recalculate if has any promotion
        merge['promo2'] = self.cal_promo2(merge)
        merge = merge.drop('promo_interval', 1)

        # Construct sales mean columns, at least we know whether this store is popular
        if not is_serving:
            sales_mean = merge.groupby('store').sales.mean()
            merge['sales_mean'] = sales_mean.reindex(merge.store).values

            # Add date for split train valid
            merge = merge.query('open == 1')[['date'] + metadata.HEADER]
            merge.to_csv(f'{self.p.prepared_path}/tr.csv', index=False)
        else:
            with open(self.p.feature_stats_file, 'r') as fp:
                stats = json.load(fp)
            # Data type of column store is int, but unserialize from json in key will become string type,
            # so change the data type to int
            sales_mean = pd.Series(stats['sales_mean']['hist_mapping'])
            sales_mean = pd.Series(index=sales_mean.index.map(int), data=sales_mean.data)
            merge['sales_mean'] = sales_mean.reindex(merge.store).values

            # dtype.pop(metadata.TARGET_NAME)
            merge = merge[metadata.SERVING_COLUMNS]

        self.logger.info(f'Prepare take time {datetime.now() - s}')
        return merge

    def fit(self, data):
        """

        :param data:
        :return:
        """
        stats = defaultdict(defaultdict)
        numeric_feature_names = metadata.INPUT_NUMERIC_FEATURE_NAMES + metadata.CONSTRUCTED_NUMERIC_FEATURE_NAMES
        for name, col in data[numeric_feature_names].iteritems():
            # scaler = preprocessing.StandardScaler()
            # scaler.fit(col.astype(float)[:, np.newaxis])
            self.logger.info(f'{name}.mean: {col.mean()}, {name}.stdv: {col.std()}')
            stats[name] = {'mean': col.mean(), 'stdv': col.std()}

        # Dump feature stats
        sales_mean = data.groupby('store').sales.mean()
        with open(self.p.feature_stats_file, 'w') as fp:
            stats['sales_mean']['hist_mapping'] = sales_mean.to_dict()
            json.dump(stats, fp)
        return self

    def transform(self, data, is_serving=False):
        """Transform columns value, maybe include categorical column to int, numeric column normalize...,
          but in this case, we leave it to tf.feature_columns package, only do `np.log1p(target)` here
          for the sake of shrinking scale of sales

        :param data:
        :param is_train:
        :return:
        """
        self.logger.info(f'Prepare start, is_serving: {is_serving}')
        s = datetime.now()

        dtype = self.get_processed_dtype()
        if isinstance(data, str):
            data = pd.read_csv(data, dtype=dtype)

        if not is_serving:
            self.logger.info(f'Do np.log(data.{metadata.TARGET_NAME}) !')
            data[metadata.TARGET_NAME] = np.log1p(data[metadata.TARGET_NAME])
            shutil.copy2(f'{self.p.prepared_path}/store.csv', f'{self.p.transformed_path}')
        # else:
        #     del dtype[metadata.TARGET_NAME]
        #     data['open'] = data.open.fillna(0)

        self.logger.info(f'Transform take time {datetime.now() - s}')
        return data.astype(dtype=dtype, errors='ignore')

    def split(self, data):
        """Merged training data

        :param data:
        :return:
        """
        self.logger.info(f'Split start')
        s = datetime.now()
        dtype = self.get_processed_dtype()
        if isinstance(data, str):
            data = pd.read_csv(data, dtype=dtype)

        tr, vl = [], []
        for st, df in data.groupby('store'):
            # Split order by date
            df = df.sort_values('date')
            total = len(df)
            cut_pos = total - int(total * self.p.valid_size)
            # self.logger.info(f'store {st}: len: {total}, '
            #                  f'{cut_pos} for train, {total - cut_pos} for valid')
            tr.append(df[:cut_pos])
            vl.append(df[cut_pos:])

        tr, vl = pd.concat(tr, 0), pd.concat(vl, 0)
        # In order to inspect time series figure, persistent date column of training data,
        tr.date.to_json(self.p.tr_dt_file)
        vl.date.to_json(self.p.vl_dt_file)

        tr.drop('date', 1).to_csv(self.p.train_files, index=False)
        vl.drop('date', 1).to_csv(self.p.valid_files, index=False)

        self.logger.info(f'Split take time {datetime.now() - s}')
        return self

    def cal_promo2(selfself, merge):
        """

        :param merge:
        :return:
        """
        map_ = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        base = np.zeros(len(merge))
        valid_cond = merge.promo2 == 1
        merge = merge[valid_cond]
        df = pd.DataFrame({'month': merge.month.values,
                           'interval': merge.promo_interval.str.split(',').values})

        cut_idx = np.cumsum(df.interval.map(len).values)
        interval_concat = []
        df.interval.map(interval_concat.extend)
        df['interval'] = np.split(pd.Series(interval_concat).map(map_).values, cut_idx)[:-1]

        base[valid_cond] = df.apply(lambda row: row.month in row.interval, 1).values
        return base.astype(int)

    def get_processed_dtype(self):
        return dict(zip(
            metadata.HEADER,
            [type(e[0]) for e in metadata.HEADER_DEFAULTS]
        ))

    def process_features(self, features):
        """ Use to implement custom feature engineering logic, e.g. polynomial expansion
        Default behaviour is to return the original feature tensors dictionary as is

        Args:
            features: {string:tensors} - dictionary of feature tensors
        Returns:
            {string:tensors}: extended feature tensors dictionary
        """
        return features

    def json_serving_input_fn(self):
        self.logger.info(f'use json_serving_input_fn !')

        feat_obj = m.Feature.instance
        feat_cols = feat_obj.create_feature_columns()
        dtype = self.get_processed_dtype()
        del dtype[metadata.TARGET_NAME]

        mappiing = {str: tf.string, int: tf.int32, float: tf.float32}

        inputs = {}
        for name, column in feat_cols.items():
            # print(f'name: {name}, column: {column}')
            # inputs[name] = tf.placeholder(shape=[None], dtype=column.dtype, name=name)
            inputs[name] = tf.placeholder(shape=[None], dtype=mappiing[dtype[name]], name=name)

        features = {
            key: tf.expand_dims(tensor, -1)
            for key, tensor in inputs.items()
        }

        return tf.estimator.export.ServingInputReceiver(
            features=self.process_features(features),
            receiver_tensors=inputs
        )

    def csv_serving_fn(self):
        """This function still have problem to solve, currently just use json_serving_input_fn will be great.

        :return:
        """
        self.logger.info(f'use csv_serving_fn !')

        file_name_pattern = tf.placeholder(shape=[None], dtype=tf.string, name='file_name_pattern')

        # feat_obj = m.Feature.instance
        # feat_cols = feat_obj.create_feature_columns()
        # dtype = self.get_dtype(is_train=False)
        # mappiing = {str: tf.string, int: tf.int32, float: tf.float32}
        # inputs = {}
        # for name, column in feat_cols.items():
        #     # print(f'name: {name}, column: {column}')
        #     # inputs[name] = tf.placeholder(shape=[None], dtype=column.dtype, name=name)
        #     inputs[name] = tf.placeholder(shape=[None], dtype=mappiing[dtype[name]], name=name)
        #
        # features = {
        #     key: tf.expand_dims(tensor, -1)
        #     for key, tensor in inputs.items()
        # }

        # serv_hook = IteratorInitializerHook()
        # inputs, _ = self.generate_input_fn(file_name_pattern, hooks=[serv_hook])()
        #
        # features = {
        #     key: tf.expand_dims(tensor, -1)
        #     for key, tensor in inputs.items()
        # }

        features = self.parse_csv(file_name_pattern, is_serving=True)

        unused_features = list(
            set(metadata.SERVING_COLUMNS) - set(metadata.INPUT_FEATURE_NAMES) - {metadata.TARGET_NAME})

        # Remove unused columns (if any)
        for column in unused_features:
            features.pop(column, None)

        return tf.estimator.export.ServingInputReceiver(
            features=self.process_features(features),
            receiver_tensors={'file_name_pattern': file_name_pattern}
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

        Args:
            csv_row: rank-2 tensor of type string (csv)
            is_serving: boolean to indicate whether this function is called during serving or training
            since the serving csv_row input is different than the training input (i.e., no target column)
        Returns:
            rank-2 tensor of the correct data type
        """
        self.logger.info(f'is_serving: {is_serving}')
        if is_serving:
            column_names = metadata.SERVING_COLUMNS
            defaults = metadata.SERVING_DEFAULTS
        else:
            column_names = metadata.HEADER
            defaults = metadata.HEADER_DEFAULTS
        # tf.expand_dims(csv_row, -1)
        columns = tf.decode_csv(tf.expand_dims(csv_row, -1), record_defaults=defaults)
        print(f'columns: {columns}')
        features = OrderedDict(zip(column_names, columns))

        return features

    def generate_input_fn(self,
                          file_names_pattern,
                          file_encoding='csv',
                          mode=tf.estimator.ModeKeys.EVAL,
                          skip_header_lines=1,
                          num_epochs=1,
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

            # return dataset, use make_one_shot_iterator will raise error
            # `Cannot capture a stateful node by value ...`
            iterator = dataset.make_one_shot_iterator()
            # iterator = dataset.make_initializable_iterator()

            # if hooks is not None:
            #     for hook in hooks:
            #         hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer)

            features, target = iterator.get_next()
            return features, target

        return _input_fn

Input.instance = Input()

def load_feature_stats():
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
    p = app_conf.instance
    try:
        if p.feature_stats_file is not None and tf.gfile.Exists(p.feature_stats_file):
            with tf.gfile.Open(p.feature_stats_file) as file:
                content = file.read()
            feature_stats = json.loads(content)
            print("INFO:Feature stats were successfully loaded from local file...")
        else:
            print("WARN:Feature stats file not found. numerical columns will not be normalised...")
    except:
        print("WARN:Couldn't load feature stats. numerical columns will not be normalised...")

    return feature_stats



class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)
