import numpy as np, pandas as pd, tensorflow as tf
import json, multiprocessing

from collections import OrderedDict, defaultdict
from sklearn.utils import shuffle as sk_shuffle

from . import metadata, model as m, app_conf
from .utils import utils

class Input(object):
    instance = None
    logger = utils.logger(__name__)

    def __init__(self):
        self.p = app_conf.instance
        # self.feature = m.Feature.instance
        self.serving_fn = {
            'json': getattr(self, 'json_serving_input_fn'),
            'csv': getattr(self, 'csv_serving_fn')
        }

    def prepare(self, fobj, dump=False, is_train=True):
        """

        :param fobj:
        :param dump:
        :param is_train:
        :return:
        """

        data = pd.read_csv(fobj)
        data['StateHoliday'] = data.StateHoliday.map(str)
        dt = pd.to_datetime(data.Date)
        # Side inputs
        st_drop_cols = [
            'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
            'Promo2SinceWeek', 'Promo2SinceYear'] # 'Promo2', 'PromoInterval']

        store = pd.read_csv(self.p.store_data).drop(st_drop_cols, 1)
        store_states = pd.read_csv(self.p.store_state)

        merge = data.merge(store, how='left', on='Store').merge(store_states, how='left', on='Store')
        del store, store_states

        # Construct year, month, day columns, maybe on specific day or period will has some trends.
        year, month, day = [], [], []
        dt.map(lambda e: [year.append(e.year), month.append(e.month), day.append(e.day)]).head()
        merge['year'] = year
        merge['month'] = month
        merge['day'] = day
        del year, month, day

        # Calculate real promo2 happened timing, promo2 have periodicity per year,
        # e.g: if PromoInterval = 'Jan,Apr,Jul,Oct', means month in 1, 4, 7, 10 in every year will
        # have another promotion on some products, so it need to drop origin Promo2
        # and recalculate if has any promotion
        merge['promo2'] = self.cal_promo2(merge)
        merge = merge.drop(['Promo2', 'PromoInterval'], 1)

        # Construct sales mean columns, at least we know whether this store is popular
        sales_mean = None
        if is_train:
            sales_mean = merge.groupby('Store').Sales.mean()
        else:
            with open(self.p.feature_stats_file, 'r') as fp:
                stats = json.load(fp)
            # Data type of column store is int, but unserialize from json in key will become string type,
            # so change the data type to int
            sales_mean = pd.Series(stats['sales_mean']['hist_mapping'])
            sales_mean = pd.Series(index=sales_mean.index.map(int), data=sales_mean.data)

        merge['sales_mean'] = sales_mean.reindex(merge.Store).values

        # Change column name style to PEP8 and resort columns order
        dtype = self.get_processed_dtype()
        if is_train:
            merge = merge.rename(index=str, columns=metadata.HEADER_MAPPING)
            # Add date for split train valid
            merge = merge.query('open == 1')[['date'] + metadata.HEADER]
        else:
            dtype.pop(metadata.TARGET_NAME)
            columns = metadata.HEADER_MAPPING.copy()
            columns.pop('Sales')
            merge = merge.rename(index=str, columns=columns)
            merge = merge[metadata.SERVING_COLUMNS]

        # Persistent in prepare stage
        if dump:
            merge.to_pickle(self.p.train_full_pr)
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

    def transform(self, data:pd.DataFrame, is_train=True):
        """Transform columns value, maybe include catg column to int, numeric column normalize...
        1. change column value
        2. handle missing value
        3. handle data type

        :param data:
        :param is_train:
        :return:
        """
        data = data.copy()
        dtype = self.get_processed_dtype()

        if is_train:
            self.logger.info(f'Do np.log(data.{metadata.TARGET_NAME}) !')
            data[metadata.TARGET_NAME] = np.log1p(data[metadata.TARGET_NAME])
        else:
            del dtype[metadata.TARGET_NAME]
            data['open'] = data.open.fillna(0)

        return data.astype(dtype=dtype)

    def split(self, data):
        """Merged training data

        :param data:
        :return:
        """
        tr, vl = [], []
        for st, df in data.groupby(['store']):
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
        valid_cond = merge.Promo2 == 1
        merge = merge[valid_cond]
        df = pd.DataFrame({'month': merge.month.values,
                           'interval': merge.PromoInterval.str.split(',').values})

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

    def csv2dicts(self, csvfile):
        data = []
        keys = []
        for row_index, row in enumerate(csvfile):
            if row_index == 0:
                keys = row
                continue
            data.append({key: value for key, value in zip(keys, row)})
        return data

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

    def get_shape(self, is_serving=False):
        cols = metadata.SERVING_COLUMNS if is_serving else metadata.HEADER
        shapes = []
        for colname in cols:
            if colname.endswith('_hist') or colname.endswith('_count') or colname.endswith('_mean') or \
                    colname in ('genre_ids', 'artist_name', 'composer', 'lyricist'):
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
