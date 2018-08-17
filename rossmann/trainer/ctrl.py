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
        """Set environment variable to api key path in order to
          access GCP service

        :return: self
        """
        from google.auth import environment_vars

        CREDENTIAL_NAME = environment_vars.CREDENTIALS
        os.environ[CREDENTIAL_NAME] = self.conf.api_key_path
        self.logger.info(f"Set env variable [{CREDENTIAL_NAME}]")
        return self

    def prepare(self, p):
        """Do all data pipeline from raw data to the format model recognized
          - Clean:
            - Fill missing value, drop unnecessary features

          - Prepare:
            - Join store and store_states to make the **Fat table**
            - Add features we mentioned in data exploration, drop also.
            - Filter some records not appropriate, like open = 0
            - Maybe persistent some files

          - Fit:
            - Persistent the statistical information of numeric features
            - Persistent the unique count value of categorical features

          - Transform:
            - Numeric data normalization
            - Make all categorical variable to int, one hot encoding ... etc.
            - Because of the scale of sales is large and large standard deviation, **we take logarithm of the target column**

          - Split: split train data to train part and valid part to check metrics on valid data to avoid overfitting

        :param p: Parameters
          - fpath: training file path
        :return: self
        """
        data = self.input.clean(p.fpath, is_serving=False)
        data = self.input.prepare(data, is_serving=False)
        data = self.input.fit(data).transform(data, is_serving=False)
        self.input.split(data)
        return self

    def transform(self, p):
        """Transform future input data, just like training period, clean -> prepare -> transform
          but not fit

        :param p: Parameters
          - fpath: training file path
        :return: Transformed data
        """
        data = self.input.clean(p.fpath, is_serving=True)
        data = self.input.prepare(data, is_serving=True)
        data = self.input.transform(data, is_serving=True)
        return data

    def train(self, p):
        """Simple call service.train

        :param p: Parameters
          - reset: If True empty the training model directory
          - model_name: Specify which model to train
        :return: self
        """
        self.service.train(model_name=p.model_name, reset=p.reset)
        return self

    def upload_model(self, p):
        """Upload trained model to GCP ML-Engine

        :param p: Parameters
          - bucket_name: GCS unique bucket name
          - prefix: path behind GCS bucket
          - model_path: Exported model protocol buffer directory
        :return: self
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
        """Use restful api to deploy model(already uploaded to GCS) to ML-Engine
          1. First create model repository
          2. We will delete all model versions first, of course this is optional, in fact GCP provide multi versions on model repository but
            we need to specify a default model version for serving
          3. Create new model version on specific repository

        :param p: Parameters
          - model_name: ML-Engine repository name
        :return: self
        """
        ml = self.service.find_ml()
        self.service.create_model_rsc(ml, p.model_name)
        self.service.clear_model_ver(ml, p.model_name)
        self.service.create_model_ver(ml, p.model_name, p.deployment_uri)
        return self


    def local_predict(self, p):
        """Read local saved protocol buffer file and do prediction

        :param p: Parameters
          - datasource: DataFrame or file path to predict
          - is_src_file: Is datasource a DataFrame object or file path, True: DataFrame object, False: file path
          - model_name: Model name in `dnn` `neu_mf`
        :return: Prediction result
        """
        from tensorflow.contrib import predictor

        export_dir = self.service.find_latest_expdir(p.model_name)
        predict_fn = predictor.from_saved_model(export_dir, signature_def_key='predict')

        if p.is_src_file:
            datasource = p.datasource
        else:
            datasource = self.service.read_transformed(p.datasource)
            datasource = self.input.fill_catg_na(datasource)

        base = np.zeros(len(datasource))
        open_flag = datasource.open == '1'
        preds = predict_fn(datasource[open_flag]).get('predictions').ravel()

        # Column sales has been take np.log1p, so predict take np.expm
        preds = np.round(np.expm1(preds))
        base[open_flag] = preds
        return base

    def online_predict(self, p):
        """Call online deployed model through restful api by `googleapiclient`

        param p: Parameters
          - datasource: Records style json object, e.g:
            [
             {key: value, key2: value2, ...},
             {key: value, key2: value2, ...},
             ...
            ]
          - model_name: Model name in `dnn` `neu_mf`
        :return: Predicted result
        """
        datasource = p.datasource
        base = np.zeros(len(datasource))
        if not isinstance(datasource, pd.DataFrame):
            datasource = pd.DataFrame(datasource) # .to_dict('records')

        open_flag = datasource.open == '1'
        preds = self.service.online_predict(datasource[open_flag].to_dict('records'), p.model_name)
        base[open_flag] = np.round(np.expm1(preds))
        return base


Ctrl.instance = Ctrl()

# @api_view(['GET', 'POST'])
# @permission_classes([CustomPermission])
# @authentication_classes([CustomTokenAuthentication])
# def entry(*args, **dicts):
#     req = args[0]
#     func = dicts.get('func')
#     return utils.dispatcher(Ctrl.instance, func, req)
