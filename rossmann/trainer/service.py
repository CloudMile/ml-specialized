import random, tensorflow as tf, shutil, os, pandas as pd
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from datetime import datetime
from pprint import pprint

from . import app_conf, input, model as m, utils
random.seed(42)

class Service(object):
    instance = None
    logger = utils.logger(__name__)

    def __init__(self):
        self.p :app_conf.Config = app_conf.instance
        self.inp :input.Input = input.Input.instance

    def train(self, reset=True):
        """Use tf.estimator.DNNRegressor to train model,
        eval at every epoch end, and export only when best (smallest) loss value occurs,

        :return: Self object
        """
        if reset and tf.gfile.Exists(self.p.model_dir):
            self.logger.info(f"Deleted job_dir {self.p.model_dir} to avoid re-use")
            shutil.rmtree(self.p.model_dir, ignore_errors=True)
            # tf.gfile.DeleteRecursively(self.p.model_dir)
        os.makedirs(self.p.model_dir, exist_ok=True)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        # Turn on XLA JIT compilation
        sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        run_config = tf.estimator.RunConfig(
            session_config=sess_config,
            # tf_random_seed=878787,
            log_step_count_steps=self.p.log_step_count_steps,
            # save_checkpoints_steps=self.p.save_checkpoints_steps,
            # save_checkpoints_secs=HYPER_PARAMS.eval_every_secs,
            keep_checkpoint_max=self.p.keep_checkpoint_max,
            model_dir=self.p.model_dir
        )

        model = m.Model(model_dir=self.p.model_dir)
        self.logger.info(f"Model Directory: {run_config.model_dir}")

        exporter = m.BestScoreExporter(
            self.p.export_name,
            self.inp.serving_fn[self.p.serving_format],
            as_text=False  # change to true if you want to export the model as readable text
        )
        # Train spec
        # tr_hook = input.IteratorInitializerHook()
        train_fn = self.inp.generate_input_fn(
            file_names_pattern=self.p.train_files,
            mode=tf.estimator.ModeKeys.TRAIN,
            # num_epochs=self.p.num_epochs,
            batch_size=self.p.batch_size,
            shuffle=True,
            # hooks=[]
        )
        train_spec = tf.estimator.TrainSpec(
            train_fn,
            max_steps=self.p.train_steps,
            # hooks=[]
        )
        # Valid spec
        # vl_hook = input.IteratorInitializerHook()
        valid_fn = self.inp.generate_input_fn(
            file_names_pattern=self.p.valid_files,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=self.p.batch_size,
            # hooks=[]
        )
        eval_spec = tf.estimator.EvalSpec(
            valid_fn,
            steps=self.p.valid_steps,
            exporters=[exporter],
            name='estimator-eval',
            # throttle_secs=self.p.eval_every_secs,
            # hooks=[]
        )
        # train and evaluate
        tf.estimator.train_and_evaluate(
            model.get_estimator(run_config),
            train_spec,
            eval_spec
        )
        return self

    def read_transformed(self, fpath):
        """Read transformed data for model prediction

        :param fpath:
        :return:
        """
        return pd.read_csv(fpath, dtype=self.inp.get_processed_dtype(is_serving=True))

    def find_ml(self):
        """GCP ML service

        :return: GCP ML service object
        """
        credentials = GoogleCredentials.get_application_default()
        return discovery.build('ml', 'v1', credentials=credentials)

    def create_model_rsc(self, ml, model_name):
        """

        :param ml: ML service
        :param model_name: model resource name
        :return:
        """
        proj_uri = f'projects/{self.p.project_id}'
        try:
            ml.projects().models().create(
                parent=proj_uri, body={'name': model_name, 'onlinePredictionLogging': True}
            ).execute()
        except Exception as e:
            self.logger.warn(e)
        return self

    def clear_model_ver(self, ml, model_name):
        """

        :param ml: ML service
        :param model_name: model resource name
        :return:
        """
        model_rsc = f'projects/{self.p.project_id}/models/{model_name}'
        vdict = ml.projects().models().versions().list(parent=model_rsc).execute()

        def delete(m):
            self.logger.info('delete model version [{name}]'.format(**m))
            ml.projects().models().versions().delete(name=m.get('name')).execute()

        if len(vdict) and 'versions' in vdict:
            if len(vdict['versions']) == 1:
                delete(vdict['versions'][0])
            else:
                for m in vdict['versions']:
                    if m['state'] != 'READY':
                        continue
                    # can't delete default version
                    if m.get('isDefault'):
                        continue
                    delete(m)
        return self

    def create_model_ver(self, ml, model_name, deployment_uri):
        """

        :param ml: ML service
        :param model_name: model resource name
        :param deployment_uri: GCS directory uri path
        :return:
        """
        model_uri = f'projects/{self.p.project_id}/models/{model_name}'
        now = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        version = f'v{now}' # f'v{utils.timestamp()}'

        self.logger.info(f'create model {model_name} from {deployment_uri}')
        res = ml.projects().models().versions().create(
            parent=model_uri,
            body={
                'name': version,
                'description': 'Regression model use tf.estimator.DNNRegressor',
                # 'isDefault': True,
                'deploymentUri': deployment_uri,
                'runtimeVersion': '1.8'
            }
        ).execute()
        return res

    def online_predict(self, datasource, model_name):
        """Online prediction with ML Engine

        :param datasource: Array list contains many dict objects
        :param model_name: Deployed model name for prediction, for no version provide,
            gcp will get default version
        :return:
        """
        model_uri = f'projects/{self.p.project_id}/models/{model_name}'
        ml = self.find_ml()
        # return data type must be in records mode
        result = ml.projects().predict(name=model_uri, body={'instances': datasource}).execute()
        return [rec.get('predictions')[0] for rec in result.get('predictions')]

Service.instance = Service()

