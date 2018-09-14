import os, pandas as pd, argparse, re

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
        self.service = utils.get_instance(service.Service)
        self.input = utils.get_instance(input.Input)

    def set_client_secret(self, p):
        """Set environment variable to api key path in order to
          access GCP service

        :return: self
        """
        from google.auth import environment_vars

        CREDENTIAL_NAME = environment_vars.CREDENTIALS
        os.environ[CREDENTIAL_NAME] = p.api_key_path
        self.logger.info("Set env variable [{}]".format(CREDENTIAL_NAME))
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
        p = self.merge_params(p)
        data = self.input.clean(p, p.fpath, is_serving=False)
        self.input.split(p, data)
        del data

        self.input.prepare(p, '{}/tr.pkl'.format(p.cleaned_path), is_serving=False)
        self.input.fit(p, '{}/tr.pkl'.format(p.prepared_path))
        self.input.transform(p, '{}/tr.pkl'.format(p.prepared_path), is_serving=False)
        return self

    def transform(self, p):
        """Transform future input data, just like training period, clean -> prepare -> transform
          but not fit

        :param p: Parameters
          - fpath: training file path
        :return: Transformed data
        """
        p = self.merge_params(p)
        data = self.input.clean(p.fpath, is_serving=True)
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

        p = self.merge_params(p)
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

        p = self.merge_params(p)
        export_dir = self.service.find_latest_expdir(p, p.model_name)
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
                self.logger.info("{}/{} ...".format(count, n_total))
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
        self.logger.info("Online prediction ...")

        p = self.merge_params(p)
        datasource = p.datasource
        if not isinstance(datasource, pd.DataFrame):
            datasource = pd.DataFrame(datasource)

        datasource = list(self.service.padded_batch(datasource))[0]
        records = datasource.to_dict('records')
        return self.service.online_predict(p, records, p.model_name)

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
            # if `cos_decay_steps` is not specified, default the same as `train_steps`
            # if both `cos_decay_steps` and `train_steps` are not specified,
            # the default settings are in the `app_conf` module
            if args.get("train_steps") is not None and args.get("cos_decay_steps") is None:
                params['cos_decay_steps'] = params['train_steps']
            params = pd.Series(params)
        else:
            params = pd.Series(app_conf.get_config().__dict__)

        return params


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
        default="neu_mf",
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
    parser.add_argument(
        '--train-steps',
        default=4358,
        type=int,
        help='max train steps',
    )
    parser.add_argument(
        '--valid-steps',
        default=492,
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
        default=1000
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

    # hyper parameters
    parser.add_argument(
        '--learning-rate',
        default=0.001,
        type=float,
        help='learning rate',
    )
    parser.add_argument(
        '--drop-rate',
        default=0.3,
        type=float,
        help='drop out rate',
    )
    parser.add_argument(
        '--embedding-size',
        help='number of embedding dimensions for songs, artist_name, composer, lyricist ...',
        default=16,
        type=int
    )
    parser.add_argument(
        '--num-layers',
        help='number of layers in the DNN, here both size of DNN part and MF part are the same',
        default=3,
        type=int
    )
    parser.add_argument(
        '--scale-factor',
        help='how quickly should the size of the layers in the DNN decay',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--first-mlp-layer-size',
        help='recommendation dnn part first hidden layer size',
        default=512,
        type=int
    )
    parser.add_argument(
        '--first-factor-layer-size',
        help='recommendation mf part first hidden layer size (before features concatenate)',
        default=32,
        type=int
    )
    parser.add_argument(
        '--cos-decay-steps',
        help='cosine decay steps, usually same with train_steps',
        default=4358,
        type=int
    )


    parser.add_argument(
        '--throttle-secs',
        help='how long to wait before running the next evaluation',
        default=60,
        type=int
    )
    parser.add_argument(
        '--save-checkpoints-steps',
        help='save checkpoint every steps',
        default=500,
        type=int
    )

    args = parser.parse_args()
    ctrl = utils.get_instance(Ctrl)
    # from pprint import pprint
    # pprint(params.to_dict())
    execution = getattr(ctrl, args.method)
    execution(args)


