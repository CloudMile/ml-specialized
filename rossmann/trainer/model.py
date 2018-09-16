import tensorflow as tf, os, shutil
from tensorflow.python.feature_column import feature_column
from tensorflow.python.ops import metrics_impl
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops

from . import utils, input, metadata

class Model(object):
    logger = utils.logger(__name__)

    def __init__(self, model_dir, name='deep'):
        """

        :param model_dir: Model checkpoint directory path
        """
        self.name = name
        self.model_dir = model_dir
        self.feature = utils.get_instance(Feature)

    def construct_hidden_units(self, p):
        """ Create the number of hidden units in each layer

        if the HYPER_PARAMS.layer_sizes_scale_factor > 0 then it will use a "decay" mechanism
        to define the number of units in each layer.

        Returns:
            list of int
        """

        hidden_units = [
            max(2, int(p.first_layer_size * p.scale_factor ** i))
            for i in range(p.num_layers)
        ]

        self.logger.info("Hidden units structure: {}".format(hidden_units))
        return hidden_units

    def get_estimator(self, p, config:tf.estimator.RunConfig):
        """In this case, return `tf.estimator.DNNRegressor` or `tf.estimator.DNNLinearCombinedRegressor`,

        :param config: received parameters
        :param config: see `tf.estimator.RunConfig`
        :return: `tf.estimator.Estimator`
        """
        feat_spec = list(self.feature.create_feature_columns(p).values())
        deep_columns, wide_columns = self.feature.get_deep_and_wide_columns(feat_spec)
        if self.name == 'deep':
            est = tf.estimator.DNNRegressor(
                hidden_units=self.construct_hidden_units(p),
                feature_columns=deep_columns,
                # model_dir=self.model_dir,
                label_dimension=1,
                weight_column=None,
                optimizer=tf.train.AdamOptimizer(p.learning_rate),
                # optimizer='Adagrad',
                # optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001),
                activation_fn=tf.nn.relu,
                dropout=p.drop_rate,
                input_layer_partitioner=None,
                config=config
            )
        else:
            est = tf.estimator.DNNLinearCombinedRegressor(
                # model_dir=self.model_dir,
                linear_feature_columns=wide_columns,
                linear_optimizer='Ftrl',
                # linear_optimizer=tf.train.GradientDescentOptimizer(p.learning_rate),
                dnn_feature_columns=deep_columns,
                # dnn_optimizer='Adagrad',
                dnn_optimizer=tf.train.AdamOptimizer(p.learning_rate),
                dnn_hidden_units=self.construct_hidden_units(p),
                dnn_activation_fn=tf.nn.relu,
                dnn_dropout=p.drop_rate,
                label_dimension=1,
                weight_column=None,
                input_layer_partitioner=None,
                config=config,
                warm_start_from=None,
                loss_reduction=tf.losses.Reduction.SUM
            )

        self.logger.info('Use {}'.format(est))
        # Create directory for export, it will raise error if in GCS environment
        try:
            tf.gfile.MakeDirs('{}/export/{}'.format(p.job_dir, p.export_name))
        except Exception as e:
            print(e)

        def metric_fn(labels, predictions):
            """ Defines extra evaluation metrics to canned and custom estimators.
            By default, this returns an empty dictionary

            Args:
                labels: A Tensor of the same shape as predictions
                predictions: A Tensor of arbitrary shape
            Returns:
                dictionary of string:metric
            """
            def rmspe(labels, predictions, weights=None):
                if context.executing_eagerly():
                    raise RuntimeError('rmspe is not supported '
                                       'when eager execution is enabled.')

                predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(
                    predictions=predictions, labels=labels, weights=weights)
                # The target has been take log1p, so take expm1 back
                labels, predictions = math_ops.expm1(labels), math_ops.expm1(predictions)
                mspe, update_op = metrics_impl.mean(
                    math_ops.square((labels - predictions) / labels), weights)
                rmspe = math_ops.sqrt(mspe)
                rmspe_update_op = math_ops.sqrt(update_op)
                return rmspe, rmspe_update_op

            metrics = {}
            pred_values = predictions['predictions']
            metrics["mae"] = tf.metrics.mean_absolute_error(labels, pred_values)
            metrics["rmse"] = tf.metrics.root_mean_squared_error(labels, pred_values)
            metrics["rmspe"] = rmspe(labels, pred_values)
            return metrics

        est = tf.contrib.estimator.add_metrics(est, metric_fn)
        self.logger.info("creating a regression model: {}".format(est))
        return est

class BestScoreExporter(tf.estimator.Exporter):
    """Only export model has the best score, like lowest loss or highest accuracy."""
    logger = utils.logger('BestScoreExporter')

    def __init__(self,
                 name,
                 serving_input_receiver_fn,
                 assets_extra=None,
                 as_text=False,
                 model_dir=None):
        self._name = name
        self.serving_input_receiver_fn = serving_input_receiver_fn
        self.assets_extra = assets_extra
        self.as_text = as_text
        self.model_dir = model_dir
        self.best = self.get_last_eval()
        self._exports_to_keep = 1
        self.export_result = None
        self.logger.info('BestScoreExporter init, last best eval is {}'.format(self.best))

    @property
    def name(self):
        return self._name

    def get_last_eval(self):
        """Get the latest best score."""
        path = '{}/best.eval'.format(self.model_dir)
        if tf.gfile.Exists(path):
            return utils.read_pickle(path)
        else:
            return None

    def save_last_eval(self, best:float):
        """Save the latest best score."""
        self.logger.info('Persistent best eval: {}'.format(best))
        path = '{}/best.eval'.format(self.model_dir)
        utils.write_pickle(path, best)

    def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
        """Call `tf.estimator.Estimator.export_savedmodel` to export model to protocol buffer.

        :param estimator: `tf.estimator.Estimator` object
        :param export_path: A string containing a directory where to write the export.
        :param checkpoint_path: The checkpoint path to export.
        :param eval_result: The output of `Estimator.evaluate` on this checkpoint.
        :param is_the_final_export: This boolean is True when this is an export in the
            end of training.  It is False for the intermediate exports during
            the training.
        :return:
        """
        self.logger.info('eval_result: {}'.format(eval_result))
        curloss = eval_result['rmspe']
        if self.best is None or self.best >= curloss:
            # Clean first, only keep the best weights
            export_path = export_path.replace('\\', '/')
            self.logger.info('clean export_path: {}'.format(export_path))
            try:
                tf.gfile.DeleteRecursively(export_path)
                # shutil.rmtree(export_path)
            except Exception as e:
                self.logger.warn(e)

            # os.makedirs(export_path, exist_ok=True)
            tf.gfile.MakeDirs(export_path)

            self.best = curloss
            self.save_last_eval(self.best)

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
        self.inp = None
        pass

    def create_feature_columns(self, p):
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
        feature_stats = self.inp.load_feature_stats(p)

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
        return self.extend_feature_columns(p, feature_columns)

    def extend_feature_columns(self, p, feature_columns):
        """ Use to define additional feature columns, such as bucketized_column(s), crossed_column(s),
        and embedding_column(s). task.HYPER_PARAMS can be used to parameterise the creation
        of the extended columns (e.g., embedding dimensions, number of buckets, etc.).

        Default behaviour is to return the original feature_columns list as-is.

        Args:
            feature_columns: {column_name: tf.feature_column} - dictionary of base feature_column(s)
        Returns:
            {string: tf.feature_column}: extended feature_column(s) dictionary
        """
        dims = {
            'state_holiday': 3,
            'month': 3,
            'day': 3,
            'state': 3,
            'store': p.embedding_size,
            'year': 3,
            'assortment': 3,
            'store_type': 3,
            'competition_open_since_month': 3,
            'competition_open_since_year': 3,
            'promo2since_week': 3,
            'promo2since_year': 3
        }
        # Deep columns
        for name in metadata.INPUT_CATEGORICAL_FEATURE_NAMES:
            if name not in dims:
                feature_columns['{}_indic'.format(name)] = tf.feature_column.indicator_column(
                    feature_columns[name]
                )
            else:
                feature_columns['{}_emb'.format(name)] = tf.feature_column.embedding_column(
                    feature_columns[name], dims[name]
                )
        # Wide columns
        feature_columns['promo_datetime_cross'] = tf.feature_column.crossed_column(
            ['promo', 'promo2', 'year', 'month', 'day', 'day_of_week'], hash_bucket_size=int(1e4))

        feature_columns['promo_holiday_cross'] = tf.feature_column.crossed_column(
            ['promo', 'promo2', 'state', 'state_holiday', 'school_holiday'], hash_bucket_size=int(1e4))

        feature_columns['store_type_cross'] = tf.feature_column.crossed_column(
            ['state', 'store_type', 'assortment', 'day_of_week'], hash_bucket_size=int(1e4))

        feature_columns['competition_cross'] = tf.feature_column.crossed_column(
            ['competition_open_since_month', 'competition_open_since_year'], hash_bucket_size=int(1e4))

        return feature_columns

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

        # indicator_columns = []

        # encode_one_hot = p.encode_one_hot
        # as_wide_columns = p.as_wide_columns
        #
        # # if encode_one_hot=True, then categorical_columns are converted into indicator_column(s),
        # # and used as dense features in the deep part of the model.
        # # if as_wide_columns=True, then categorical_columns are used as sparse features in the wide part of the model.
        # if encode_one_hot:
        #     indicator_columns = list(
        #         map(lambda column: tf.feature_column.indicator_column(column),
        #             categorical_columns)
        #     )

        indicator_columns = list(
            filter(lambda column: isinstance(column, feature_column._IndicatorColumn),
                   feature_columns)
        )

        deep_columns = dense_columns + indicator_columns
        wide_columns = sparse_columns # + (categorical_columns if as_wide_columns else [])

        return deep_columns, wide_columns

# Feature.instance = Feature()

