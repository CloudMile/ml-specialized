import numpy as np, pandas as pd, tensorflow as tf
import json, multiprocessing

from collections import OrderedDict, defaultdict

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

    # def prepare_feature(self, data, store_data, is_train=True):
    #     train_data_X = []
    #     train_data_y = []
    #
    #     for record in data:
    #         # if record['Sales'] != '0' and record['Open'] != '':
    #         fl = self.feature_list(record, store_data)
    #         train_data_X.append(fl)
    #         if is_train:
    #             train_data_y.append(float(record['Sales']))
    #
    #     if is_train:
    #         self.logger.info(f"min sales: {min(train_data_y)}, max sales: {max(train_data_y)}")
    #
    #     # Sort by first column: date
    #     ret = pd.DataFrame(data=train_data_X).sort_values(0)
    #     ret = ret.drop(0, 1)
    #     if is_train:
    #         ret.loc[:, -1] = np.array(train_data_y)
    #         ret.columns = metadata.HEADER
    #     else:
    #         ret.columns = metadata.SERVING_COLUMNS
    #     return ret

    # def feature_list(self, record, store_data):
    #     dt = datetime.strptime(record['Date'], '%Y-%m-%d')
    #     store_index = int(record['Store'])
    #     try:
    #         store_open = int(record['Open'])
    #     except:
    #         store_open = 1
    #
    #     return [
    #         record['Date'],
    #         store_open,
    #         store_index,
    #         int(record['DayOfWeek']),
    #         int(record['Promo']),
    #         record['StateHoliday'],
    #         record['SchoolHoliday'],
    #         dt.year,
    #         dt.month,
    #         dt.day,
    #         store_data[store_index - 1]['State'],
    #     ]

    # def do_raw(self, fpath):
    #     with open(fpath) as f:
    #         data = csv.reader(f, delimiter=',')
    #         # with open(p.train_pkl, 'wb') as f:
    #         data = self.csv2dicts(data)
    #         data = data[::-1]
    #         return data
    #
    # def do_store(self):
    #     with open(self.p.store_data) as st, open(self.p.store_state) as st_state:
    #         data = csv.reader(st, delimiter=',')
    #         state_data = csv.reader(st_state, delimiter=',')
    #         # with open(p.store_pkl, 'wb') as f:
    #         data = self.csv2dicts(data)
    #         state_data = self.csv2dicts(state_data)
    #         self.set_nan_as_string(data)
    #
    #         for index, val in enumerate(data):
    #             state = state_data[index]
    #             val['State'] = state['State']
    #             data[index] = val
    #         return data

    # def set_nan_as_string(self, data, replace_str='0'):
    #     for i, x in enumerate(data):
    #         for key, value in x.items():
    #             if value == '':
    #                 x[key] = replace_str
    #         data[i] = x

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
