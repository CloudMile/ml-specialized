import tensorflow as tf, numpy as np, os
import pandas as pd

from . import app_conf, service, input
from . import model as m, utils

class Ctrl(object):
    instance = None
    app_dir = ''
    logger = utils.logger(__name__)

    def __init__(self):
        self.service:service.Service = service.Service.instance
        self.conf:app_conf.Config = app_conf.instance
        self.feature:m.Feature = m.Feature()
        self.input:input.Input = input.Input.instance

    def set_client_secret(self):
        from google.auth import environment_vars

        CREDENTIAL_NAME = environment_vars.CREDENTIALS
        self.logger.info(f"Set env variable [{CREDENTIAL_NAME}]")
        os.environ[CREDENTIAL_NAME] = self.conf.api_key_path
        return self

    def prepare(self, p):
        """
        1. Clean data: fill NaN value, drop unnecessary columns
        2. Prepare data: add, merge, drop columns ...
        3. Fit (if in training stage)
        4. Transform
        5. Split input data to train, valid
        :param p:
        :return:
        """
        data = self.input.clean(p.fpath, is_serving=False)
        data = self.input.prepare(data, is_serving=False)
        data = self.input.fit(data).transform(data, is_serving=False)
        self.input.split(data)
        return self

    def transform(self, p):
        """Transform serving input data

        :param p: config params
        :return:
        """
        data = self.input.clean(p.fpath, is_serving=True)
        data = self.input.prepare(data, is_serving=True)
        data = self.input.transform(data, is_serving=True)
        return data

    def train(self, p):
        self.service.train(model_name=p.model_name, reset=p.reset)
        return self

    def upload_model(self, p):
        """

        :param p:
        :return:
        """
        from google.cloud import storage

        # utils.find_latest_expdir(self.conf)
        bucket = storage.Client().get_bucket(p.bucket_name)
        # clean model dir
        for blob in bucket.list_blobs(prefix=p.prefix):
            self.logger.info(f'delete {p.bucket_name}/{blob.name}')
            blob.delete()
        # upload
        for local, blob_name in utils.deep_walk(p.model_path, prefix=p.prefix):
            self.logger.info(f'copy {local} to {blob_name}')
            bucket.blob(blob_name).upload_from_filename(local)


    def deploy(self, p):
        """

        :param p:
        :return:
        """
        ml = self.service.find_ml()
        self.service.create_model_rsc(ml, p.model_name)
        self.service.clear_model_ver(ml, p.model_name)
        self.service.create_model_ver(ml, p.model_name, p.deployment_uri)
        return self


    def local_predict(self, p):
        """Read local saved protocol buffer file and do prediction

        :param p: config params
        :return:
        """
        from tensorflow.contrib import predictor

        export_dir = self.service.find_latest_expdir(p.model_name)
        predict_fn = predictor.from_saved_model(export_dir, signature_def_key='predict')

        if p.is_src_file:
            datasource = self.service.read_transformed(p.datasource)
            datasource = self.input.fill_catg_na(datasource)
        else:
            datasource = p.datasource

        base = np.zeros(len(datasource))
        open_flag = datasource.open == '1'
        preds = predict_fn(datasource[open_flag]).get('predictions').ravel()

        # Column sales has been take np.log1p, so predict take np.expm
        preds = np.round(np.expm1(preds))
        base[open_flag] = preds
        return base

    def online_predict(self, p):
        """

        :param p: p.datasource must be structure as [{}, {} ...]
        :return:
        """
        datasource = p.datasource
        base = np.zeros(len(datasource))
        if not isinstance(datasource, pd.DataFrame):
            datasource = pd.DataFrame(datasource) # .to_dict('records')

        open_flag = datasource.open == '1'
        preds = self.service.online_predict(datasource[open_flag].to_dict('records'), p.model_name)
        base[open_flag] = np.round(np.expm1(preds))
        return base

    # TODO: alternative prediction way, use tf.saved_model.loader.load
    def local_predict_alt(self, p):
        """Alternative way for prediction, use tf.saved_model.loader.load to load pb file.
        but you should know what the output node, and the input, just for a memo

        :param p:
        :return:
        """
        if p.is_src_file:
            datasource = self.service.read_transformed(p.datasource)
        else:
            datasource = p.datasource
        export_dir = utils.find_latest_expdir(self.conf)
        self.logger.info(f'Found export dir: {export_dir}')

        tf.reset_default_graph()
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
            # Json serving input
            feed_dict = {
                f'{k}:0': v
                for k, v in datasource.to_dict('list').items()}
            predictions = sess.run(sess.graph.get_tensor_by_name('dnn/logits/BiasAdd:0'),
                                   feed_dict=feed_dict)

            # CSV serving input
            # feed_dict = { 'file_name_pattern:0': ['./data/test.csv'] }
            # predictions = sess.run(sess.graph.get_tensor_by_name('dnn/logits/BiasAdd:0'), feed_dict=feed_dict)

        return np.round(np.expm1(predictions.ravel()))

    # TODO hack: inspect data
    def inspect(self, key, encoded_key, typ='deep'):
        model = m.Model(model_dir=None)
        train_fn = self.input.generate_input_fn(
            file_names_pattern=self.conf.train_files,
            mode=tf.estimator.ModeKeys.TRAIN,
            num_epochs=self.conf.num_epochs,
            batch_size=5000,
            shuffle=False
        )

        feat_data, target = train_fn()
        feat_spec = model.feature.create_feature_columns()
        deep_spec, wide_spec = model.feature.get_deep_and_wide_columns(list(feat_spec.values()))
        deep_spec = dict(pd.Series(deep_spec).map(lambda e: (e.name, e)).values)
        wide_spec = dict(pd.Series(wide_spec).map(lambda e: (e.name, e)).values)
        print(f'deep_spec.keys: {deep_spec.keys()}')
        print(f'wide_spec.keys: {wide_spec.keys()}')
        origin = feat_data[key]
        if typ == 'deep':
            encoded = tf.feature_column.input_layer(feat_data, deep_spec[encoded_key])
            dense = tf.feature_column.input_layer(feat_data, list(deep_spec.values()))
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                sess.run(tf.tables_initializer())
                origin_, encoded_, all_ = sess.run([origin, encoded, dense])
            return origin_, encoded_, all_
        # Wide and deep can't use
        else:
            wrap = tf.feature_column.indicator_column(wide_spec[encoded_key])
            encoded = tf.feature_column.input_layer(feat_data, wrap)
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                sess.run(tf.tables_initializer())
                origin_, encoded_ = sess.run([origin, encoded])
            return origin_, encoded_, None

    # TODO hack:
    def get_from_dataset(self, p):
        model = m.Model(model_dir=self.conf.model_dir)
        feat_spec = model.feature.create_feature_columns()
        train_fn = self.input.generate_input_fn(
            file_names_pattern=self.conf.train_files,
            mode=tf.estimator.ModeKeys.TRAIN,
            num_epochs=1,
            batch_size=10000,
            shuffle=False
        )
        valid_fn = self.input.generate_input_fn(
            file_names_pattern=self.conf.valid_files,
            mode=tf.estimator.ModeKeys.EVAL,
            num_epochs=1,
            batch_size=10000,
            shuffle=False
        )

        def from_dataset(data_fn):
            ret = []
            x, y = data_fn()
            x = tf.feature_column.input_layer(x, list(feat_spec.values()))
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                sess.run(tf.tables_initializer())
                while True:
                    try:
                        x_, y_ = sess.run([x, y])
                        ret.append(np.c_[x_, y_])
                    except tf.errors.OutOfRangeError as e: break
            return np.concatenate(ret, 0)
        tr = from_dataset(train_fn)
        vl = from_dataset(valid_fn)

        tr_label, vl_label = tr[:, -1], vl[:, -1]
        return tr[:, :-1], tr_label, vl[:, :-1], vl_label

    # TODO any test !
    def test(self):
        model = m.Model(model_dir=self.conf.model_dir)
        with tf.Graph().as_default():
            train_fn = self.input.generate_input_fn(
                file_names_pattern=self.conf.train_files,
                mode=tf.estimator.ModeKeys.TRAIN,
                num_epochs=self.conf.num_epochs,
                batch_size=5000,
                shuffle=False
            )

            feat_data, target = train_fn()
            feat_spec = model.feature.create_feature_columns()
            dense_data = tf.feature_column.input_layer({'year': tf.constant([['2019'], ['2015']])}, feat_spec['year'])
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                sess.run(tf.tables_initializer())
                return sess.run(dense_data)

Ctrl.instance = Ctrl()

# @api_view(['GET', 'POST'])
# @permission_classes([CustomPermission])
# @authentication_classes([CustomTokenAuthentication])
# def entry(*args, **dicts):
#     req = args[0]
#     func = dicts.get('func')
#     return utils.dispatcher(Ctrl.instance, func, req)
