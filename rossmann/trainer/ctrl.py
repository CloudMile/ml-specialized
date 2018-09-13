import numpy as np, os, argparse
import pandas as pd, re

from . import app_conf, service, input, model as m
from . import utils

class Ctrl(object):
    instance = None
    app_dir = ''
    logger = utils.logger(__name__)

    def __init__(self):
        self.input = None # input.Input.instance
        self.service = None # service.Service.instance

    def set_client_secret(self, p):
        """Set environment variable to api key path in order to
          access GCP service

        :return: self
        """
        from google.auth import environment_vars

        p = self.merge_params(p)
        CREDENTIAL_NAME = environment_vars.CREDENTIALS
        os.environ[CREDENTIAL_NAME] = p.api_key_path
        self.logger.info("Set env variable [{}]".format(CREDENTIAL_NAME))
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
        p = self.merge_params(p)
        data = self.input.clean(p, p.fpath, is_serving=False)
        data = self.input.prepare(p, data, is_serving=False)
        data = self.input.fit(p, data).transform(p, data, is_serving=False)
        self.input.split(p, data)
        return self

    def transform(self, p):
        """Transform future input data, just like training period, clean -> prepare -> transform
          but not fit

        :param p: Parameters
          - fpath: training file path
        :return: Transformed data
        """
        p = self.merge_params(p)
        data = self.input.clean(p, p.fpath, is_serving=True)
        data = self.input.prepare(p, data, is_serving=True)
        data = self.input.transform(p, data, is_serving=True)
        return data

    def submit(self, p):
        """Simple call service.train

        :param p: Parameters
          - reset: If True empty the training model directory
          - model_name: Specify which model to train
        :return: self
        """
        p = self.merge_params(p)

        commands = """
            gcloud ml-engine jobs submit training {job_name} \
                --job-dir={job_dir} \
                --runtime-version=1.10 \
                --region=asia-east1 \
                --module-name=trainer.ctrl \
                --package-path=trainer  \
                --config=config.yaml \
                -- \
                --method=train \
                --model-name={model_name} \
                --train-steps={train_steps} \
                --verbosity={verbosity} \
                --save-checkpoints-steps={save_checkpoints_steps} \
                --throttle-secs={throttle_secs} \
                --reset={reset}
        """.strip().format(**p.to_dict())

        self.logger.info('submit cmd:\n{commands}'.format(
            **{'commands': re.sub(r'\s{2,}', '\n  ', commands)}))
        self.logger.info( utils.cmd(commands) )
        # print( 'commands: {}'.format(commands) )
        return self

    def train(self, p):
        """Simple call service.train

        :param p: Parameters
          - reset: If True empty the training model directory
          - model_name: Specify which model to train
        :return: self
        """
        p = self.merge_params(p)
        self.service.train(p)
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

        p = self.merge_params(p)
        bucket = storage.Client().get_bucket(p.bucket_name)
        # clean model dir
        for blob in bucket.list_blobs(prefix=p.prefix):
            self.logger.info('delete {}/{}'.format(p.bucket_name, blob.name))
            blob.delete()
        # upload
        for local, blob_name in utils.deep_walk(p.model_path, prefix=p.prefix):
            self.logger.info('copy {} to {}'.format(local, blob_name))
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
        p = self.merge_params(p)
        ml = self.service.find_ml()
        self.service.create_model_rsc(p, ml, p.model_name)
        self.service.clear_model_ver(p, ml, p.model_name)
        self.service.create_model_ver(p, ml, p.model_name, p.deployment_uri)
        return self


    def local_predict(self, p):
        """Read local saved protocol buffer file and do prediction

        :param p: Parameters
          - datasource: DataFrame or file path to predict
          - is_src_file: Is datasource a DataFrame object or file path, True: DataFrame object, False: file path
          - model_name: Model name in `dnn` `neu_mf`
          - job_dir: model checkpoint directory
        :return: Prediction result
        """
        from tensorflow.contrib import predictor

        p = self.merge_params(p)
        export_dir = self.service.find_latest_expdir(p, p.model_name, p.job_dir)
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
        p = self.merge_params(p)
        datasource = p.datasource
        base = np.zeros(len(datasource))
        if not isinstance(datasource, pd.DataFrame):
            datasource = pd.DataFrame(datasource) # .to_dict('records')

        open_flag = datasource.open == '1'
        preds = self.service.online_predict(p, datasource[open_flag].to_dict('records'), p.model_name)
        base[open_flag] = np.round(np.expm1(preds))
        return base

    def merge_params(self, args=None):
        """Merge received parameters with default settings in app_conf.py

        :param args: Received parameters
        :return: Merged parameters
        """
        if args is not None:
            if not isinstance(args, dict):
                args = args.to_dict() if isinstance(args, pd.Series) else args.__dict__

            params = {}
            params.update(app_conf.get_config(args.get("env")).__dict__)
            params.update(args)
            params = pd.Series(params)
        else:
            params = pd.Series(app_conf.get_config().__dict__)

        return params

def arrange_instances():
    """Like a container, init all instance and set all dependencies

    :param args: Received parameters
    :return:
    """
    ctrl, svc, inp, feature = Ctrl(), service.Service(), input.Input(), m.Feature()

    # ctrl.p = params
    ctrl.service = svc
    ctrl.input = inp
    Ctrl.instance = ctrl

    svc.inp = inp
    # svc.p = params
    service.Service.instance = svc

    # inp.p = params
    inp.feature = feature
    input.Input.instance = inp

    # feature.p = params
    feature.inp = inp
    m.Feature.instance = feature

    return ctrl, svc, inp, feature


# @api_view(['GET', 'POST'])
# @permission_classes([CustomPermission])
# @authentication_classes([CustomTokenAuthentication])
# def entry(*args, **dicts):
#     req = args[0]
#     func = dicts.get('func')
#     return utils.dispatcher(Ctrl.instance, func, req)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env',
        default="cloud",
        help='string "cloud" for app_conf.CMLEConfig, else app_conf.Config',
    )
    parser.add_argument(
        '--reset',
        default=True,
        type=bool,
        help='whether to clear job dir',
    )
    parser.add_argument(
        '--model-name',
        default="deep",
        help='',
    )
    parser.add_argument(
        '--method',
        default='train',
        help='execution method in Controller object',
    )
    parser.add_argument(
        '--job-dir',
        help='where to put checkpoints',
    )
    # parser.add_argument(
    #     '--job-id',
    #     help='job id for training and deploy',
    # )
    parser.add_argument(
        '--train-steps',
        default=2308 * 8,
        type=int,
        help='max train steps',
    )
    parser.add_argument(
        '--valid-steps',
        default=989,
        type=int,
        help='max eval steps',
    )
    parser.add_argument(
        '--runtime-version',
        default='1.10',
        help='specific the runtime version',
    )
    parser.add_argument(
        '--batch-size',
        help='Batch size for each training step',
        type=int,
        default=200
    )
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )
    parser.add_argument(
        '--learning-rate',
        default=0.001,
        type=float,
        help='learning rate',
    )
    parser.add_argument(
        '--drop-rate',
        default=0.,
        type=float,
        help='drop out rate',
    )
    parser.add_argument(
        '--embedding-size',
        help='Number of embedding dimensions for store only',
        default=16,
        type=int
    )
    parser.add_argument(
        '--first-layer-size',
        help='Number of nodes in the first layer of the DNN',
        default=128,
        type=int
    )
    parser.add_argument(
        '--num-layers',
        help='Number of layers in the DNN',
        default=3,
        type=int
    )
    parser.add_argument(
        '--scale-factor',
        help='How quickly should the size of the layers in the DNN decay',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--throttle-secs',
        help='How long to wait before running the next evaluation',
        default=60,
        type=int
    )
    parser.add_argument(
        '--save-checkpoints-steps',
        help='save checkpoint every steps',
        default=2308,
        type=int
    )

    args = parser.parse_args()
    ctrl, svc, inp, feature = arrange_instances()

    # from pprint import pprint
    # pprint(params.to_dict())

    execution = getattr(ctrl, args.method)
    execution(args)
