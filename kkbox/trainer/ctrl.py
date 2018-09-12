import os, pandas as pd

from . import app_conf, service, input
from .utils import utils

class Ctrl(object):
    """High level controller object for restful style or local function call, exposed function always
      come up with one parameter, which is usually a dictionary object, this job of this kind object just
      do data receive and check, this is one of design patterns called MVC

    """
    instance = None
    logger = utils.logger(__name__)

    def __init__(self):
        self.service:service.Service = service.Service.instance
        self.p:app_conf.Config = app_conf.instance
        # self.feature:m.Feature = m.Feature()
        self.input:input.Input = input.Input.instance

    def set_client_secret(self):
        """Set environment variable to api key path in order to
          access GCP service

        :return: self
        """
        from google.auth import environment_vars

        CREDENTIAL_NAME = environment_vars.CREDENTIALS
        os.environ[CREDENTIAL_NAME] = self.p.api_key_path
        self.logger.info(f"Set env variable [{CREDENTIAL_NAME}]")
        return self

    def prepare(self, p):
        """Do all data pipeline from raw data to the format model recognized
          - Clean:
            - Fill missing value, drop unnecessary features

          - Split: split train data to train part and valid part to check metrics on valid data to avoid overfitting

          - Prepare:
            - Add features we mentioned in data exploration, drop also.
            - Maybe persistent some files

          - Fit:
            - Persistent the statistical information of numeric features
            - Persistent the unique count value of categorical features

          - Transform:
            - Numeric data normalization
            - Make all categorical variable to int, one hot encoding ... etc.

        :param p: Parameters
          - fpath: training file path
        :return: self
        """
        data = self.input.clean(p.fpath, is_serving=False)
        self.input.split(data)
        del data

        self.input.prepare(f'{self.p.cleaned_path}/tr.pkl', is_serving=False)
        self.input.fit(f'{self.p.prepared_path}/tr.pkl')
        self.input.transform(f'{self.p.prepared_path}/tr.pkl', is_serving=False)
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
        self.service.train(reset=p.reset, model_name=p.model_name)
        return self

    def upload_model(self, p):
        """Upload trained model to GCS

        :param p: Parameters
          - bucket_name: GCS unique bucket name
          - prefix: path behind GCS bucket
          - model_path: Exported model protocol buffer directory
        :return: self
        """
        from google.cloud import storage

        bucket = storage.Client().get_bucket(p.bucket_name)
        # clean model dir
        for blob in bucket.list_blobs(prefix=p.prefix):
            self.logger.info(f'delete {p.bucket_name}/{blob.name}')
            blob.delete()
        # upload
        for local, blob_name in utils.deep_walk(p.model_path, prefix=p.prefix):
            self.logger.info(f'copy {local} to {blob_name}')
            bucket.blob(blob_name).upload_from_filename(local)
        return self

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
        predict_fn = predictor.from_saved_model(export_dir, signature_def_key='outputs')

        if p.is_src_file:
            datasource = p.datasource
        else:
            datasource = self.service.read_transformed(p.datasource)

        # callback = lambda pipe: predict_fn(pipe.to_dict('list')).get('predictions')
        predictions = []
        count, n_total = 0, len(datasource)
        for pipe in self.service.padded_batch(datasource):
            predictions.extend(predict_fn(pipe.to_dict('list')).get('predictions').ravel())
            count += len(pipe)
            if count % 10000 == 0:
                self.logger.info(f"{count}/{n_total} ...")
        return predictions

    def online_predict(self, p):
        """Call online deployed model through restful api by `googleapiclient`

        param p: Parameters
          - datasource: Records style json object, e.g:
            [
             {key: value, key2: value2, ...},
             {key: value, key2: value2, ...},
             ...
            ]
          - is_src_file: Is datasource a DataFrame object or file path, True: DataFrame object, False: file path
          - model_name: Model name in `dnn` `neu_mf`
        :return: Predicted result
        """
        self.logger.info(f"Online prediction ...")
        datasource = p.datasource
        if not isinstance(datasource, pd.DataFrame):
            datasource = pd.DataFrame(datasource)

        datasource = list(self.service.padded_batch(datasource))[0]
        records = datasource.to_dict('records')
        return self.service.online_predict(records, p.model_name)

Ctrl.instance = Ctrl()

