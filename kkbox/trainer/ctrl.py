import tensorflow as tf, numpy as np, os
import pandas as pd

from . import app_conf, service, input
from . import model as m
from .utils import utils

class Ctrl(object):
    instance = None
    app_dir = ''
    logger = utils.logger(__name__)

    def __init__(self):
        self.service:service.Service = service.Service.instance
        self.p:app_conf.Config = app_conf.instance
        # self.feature:m.Feature = m.Feature()
        self.input:input.Input = input.Input.instance

    def set_client_secret(self):
        from google.auth import environment_vars

        CREDENTIAL_NAME = environment_vars.CREDENTIALS
        os.environ[CREDENTIAL_NAME] = self.p.api_key_path
        return self

    def prepare(self, p):
        """

        :param p:
        :return:
        """
        data = self.input.clean(p.fpath, is_serving=False)
        self.input.split(data)
        del data

        self.input.prepare(f'{self.p.cleaned_path}/tr.pkl', is_serving=True)
        self.input.fit(f'{self.p.prepared_path}/tr.pkl')
        self.input.transform(f'{self.p.prepared_path}/tr.pkl', is_serving=True)
        return self

    def transform(self, p):
        """Transform future input data

        :param p: config params
        :return:
        """
        data = self.input.clean(p.fpath, is_serving=True)
        data = self.input.prepare(data, is_serving=True)
        data = self.input.transform(data, is_serving=True)
        return data

    def train(self, p):
        self.service.train()
        return self

    def upload_model(self, p):
        """

        :param p:
        :return:
        """
        from google.cloud import storage

        utils.find_latest_expdir(self.p)
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

        export_dir = utils.find_latest_expdir(self.p)
        predict_fn = predictor.from_saved_model(export_dir, signature_def_key='outputs')

        if p.is_src_file:
            datasource = self.service.read_transformed(p.datasource)
        else:
            datasource = p.datasource

        return self.service.batch_predict(predict_fn, datasource)
        # preds = predict_fn(datasource).get('predictions').ravel()
        # return preds

    def online_predict(self, p):
        """

        :param p: p.datasource must be structure as [{}, {} ...]
        :return:
        """
        datasource = p.datasource
        base = np.zeros(len(datasource))
        if not isinstance(datasource, pd.DataFrame):
            datasource = pd.DataFrame(datasource) # .to_dict('records')

        open_flag = datasource.open == 1
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
        export_dir = utils.find_latest_expdir(self.p)

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
    def inspect(self, key):
        model = m.Model(model_dir=self.p.model_dir)
        train_fn = self.input.generate_input_fn(
            file_names_pattern=self.p.train_files,
            mode=tf.estimator.ModeKeys.TRAIN,
            num_epochs=self.p.num_epochs,
            batch_size=5000,
            shuffle=False
        )

        feat_data, target = train_fn()
        feat_spec = model.feature.create_feature_columns()
        dense_data = tf.feature_column.input_layer(feat_data, feat_spec[key])
        dense_all = tf.feature_column.input_layer(feat_data, list(feat_spec.values()))

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            sess.run(tf.tables_initializer())
            encoded, origin, feat_data, target_, all_ = sess.run([
                dense_data, feat_data[key], feat_data, target, dense_all])
        return encoded, origin, feat_data, target_, all_

    # TODO any test !
    def test(self):
        serving_inp = self.input.csv_serving_fn()
        return serving_inp

Ctrl.instance = Ctrl()

# @api_view(['GET', 'POST'])
# @permission_classes([CustomPermission])
# @authentication_classes([CustomTokenAuthentication])
# def entry(*args, **dicts):
#     req = args[0]
#     func = dicts.get('func')
#     return utils.dispatcher(Ctrl.instance, func, req)
