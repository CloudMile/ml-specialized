import tensorflow as tf, shutil, os, pandas as pd, numpy as np
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from datetime import datetime

from . import app_conf, input, model as m
from .utils import utils

class Service(object):
    """Business logic object, all kind of logic write down here to called by controller,
      maybe there are some other assistance object, like input.Input for data handle... etc.

    """
    instance = None
    logger = utils.logger(__name__)

    def __init__(self):
        self.p: app_conf.Config = app_conf.instance
        self.inp: input.Input = input.Input.instance

    def train(self, train_fn=None, tr_hook=None, valid_fn=None, vl_hook=None,
              reset=True, model_name='dnn'):
        """Train tensorflow model with tf.estimator.Estimator object

        Train spec: wrap train_fn or training hook(optional)
        Eval spec: wrap valid_fn or validation hook(optional)

        :param train_fn: Data source provider, usually with implicit iterator to pull from train data
        :param tr_hook: Callback object inherit from `tf.train.SessionRunHook` in training period
        :param valid_fn: Data source provider, usually with implicit iterator to pull from valid data
        :param vl_hook: Callback object inherit from `tf.train.SessionRunHook` in validation period
        :param reset: If True, empty model directory, otherwise not
        :param model_name: Model name in `dnn` `neu_mf`
        :return: self
        """
        self.check_model_name(model_name)
        model_dir = self.p.model_dir if model_name == 'dnn' else self.p.neu_mf_model_dir

        if reset and tf.gfile.Exists(model_dir):
            self.logger.info(f"Delete job_dir {model_dir} to avoid re-use")
            shutil.rmtree(model_dir, ignore_errors=True)
            # tf.gfile.DeleteRecursively(self.p.model_dir)
        os.makedirs(model_dir, exist_ok=True)


        self.logger.info(f"Model: {model_name}, model_dir: {model_dir}")
        model = m.Model(model_dir=self.p.model_dir) if model_name == 'dnn' else \
                m.NeuMFModel(model_dir=self.p.neu_mf_model_dir)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        # Turn on XLA JIT compilation
        sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        run_config = tf.estimator.RunConfig(
            session_config=sess_config,
            # tf_random_seed=878787,
            log_step_count_steps=self.p.log_step_count_steps,
            save_checkpoints_steps=self.p.save_checkpoints_steps,
            # save_checkpoints_secs=HYPER_PARAMS.eval_every_secs,
            keep_checkpoint_max=self.p.keep_checkpoint_max,
            model_dir=model_dir
        )

        self.logger.info(f'Use model {model_name}: {model}')
        self.logger.info(f"Model Directory: {run_config.model_dir}")

        exporter = m.BestScoreExporter(
            self.p.export_name,
            self.inp.serving_fn[self.p.serving_format],
            model_dir=model_dir,
            as_text=False  # change to true if you want to export the model as readable text
        )
        # Train spec
        # tr_hook = input.IteratorInitializerHook()
        if not train_fn:
            self.logger.info(f'read train file into memory')
            tr = pd.read_pickle(self.p.train_files)
            train_fn, tr_hook = self.inp.generate_input_fn(
                inputs=tr,
                mode=tf.estimator.ModeKeys.TRAIN,
                skip_header_lines=0,
                num_epochs=self.p.num_epochs,
                batch_size=self.p.batch_size,
                shuffle=True,
                multi_threading=True
            )
        train_spec = tf.estimator.TrainSpec(
            train_fn,
            max_steps=self.p.train_steps,
            hooks=[tr_hook]
        )
        # Valid spec
        # vl_hook = input.IteratorInitializerHook()
        if not valid_fn:
            self.logger.info(f'read valid file into memory')
            vl = pd.read_pickle(self.p.valid_files)
            valid_fn, vl_hook = self.inp.generate_input_fn(
                inputs=vl,
                mode=tf.estimator.ModeKeys.EVAL,
                skip_header_lines=0,
                num_epochs=self.p.num_epochs,
                batch_size=self.p.batch_size,
                shuffle=False,
                multi_threading=True
            )
        eval_spec = tf.estimator.EvalSpec(
            valid_fn,
            steps=self.p.valid_steps,
            exporters=[exporter],
            name='estimator-eval',
            throttle_secs=self.p.eval_every_secs,
            hooks=[vl_hook]
        )

        # train and evaluate
        tf.estimator.train_and_evaluate(
            model.get_estimator(run_config),
            train_spec,
            eval_spec
        )
        return self

    def find_latest_expdir(self, model_name):
        """Find latest exported directory by specified model name

        :param model_name: Model name in `dnn` `neu_mf`
        :return: Latest directory path
        """
        self.check_model_name(model_name)
        model_dir = self.p.neu_mf_model_dir if model_name == 'neu_mf' else self.p.model_dir
        # Found latest export dir
        export_dir = f'{model_dir}/export/{self.p.export_name}'
        return f'{export_dir}/{sorted(os.listdir(export_dir))[-1]}'

    def check_model_name(self, model_name):
        """Check if model name in (`dnn` `neu_mf`)

        :param model_name: Model name in `dnn` `neu_mf`
        :return:
        """
        assert model_name in ('dnn', 'neu_mf'), "model_name only support ('dnn', 'neu_mf')"

    def read_transformed(self, fpath):
        """Same as `Ctrl.instance.transform`, transform data to model recognized format in serving period

        :param fpath: Raw serving file path
        :return: Transformed data
        """
        data = self.inp.clean(fpath, is_serving=True)
        data = self.inp.prepare(data, is_serving=True)
        return self.inp.transform(data, is_serving=True)

    def find_ml(self):
        """Return GCP ML service

        :return: GCP ML service object
        """
        credentials = GoogleCredentials.get_application_default()
        return discovery.build('ml', 'v1', credentials=credentials)

    def create_model_rsc(self, ml, model_name):
        """Create model repository on GCP ML-Engine

        :param ml: GCP ML service object
        :param model_name: Model name to put on ML-Engine repository
        :return: self
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
        """Clear all model versions in repository

        :param ml: GCP ML service object
        :param model_name: Model name to put on ML-Engine repository
        :return: self
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
        """Create a new model version with current datetime as version name

        :param ml: GCP ML service object
        :param model_name: Model name to put on ML-Engine repository
        :param deployment_uri: GCS path to locate the saved model
        :return: self
        """
        model_uri = f'projects/{self.p.project_id}/models/{model_name}'
        now = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        version = f'v{now}' # f'v{utils.timestamp()}'

        self.logger.info(f'create model {model_name} from {deployment_uri}')
        ml.projects().models().versions().create(
            parent=model_uri,
            body={
                'name': version,
                'description': 'Regression model use tf.estimator.DNNRegressor',
                # 'isDefault': True,
                'deploymentUri': deployment_uri,
                'runtimeVersion': '1.8'
            }
        ).execute()
        return self

    def padded_batch(self, data, n_batch=1000):
        """Batch yeild data and pad the variable length feature to the same length in a batch,
         this job should done in `input.Input.generate_input_fn`, but exported model doesn't contains
         the dataset part, so here we do this by pandas.

        :param data: Not padded transformed data for prediction
        :param n_batch: Batch size
        :return: Yield padded batch data
        """
        assert n_batch <= 1000, 'Prediction batch size must less equal than 1000!'

        multi_cols = self.inp.get_multi_cols(is_serving=True)
        pad = tf.keras.preprocessing.sequence.pad_sequences
        dtype = self.inp.get_dtype(is_serving=True)
        lens = len(data)
        mod = lens % n_batch
        batch_count = int( lens // n_batch )
        indices = [0] + [n_batch] * batch_count
        if mod != 0:
            indices.append(mod)

        indices = np.cumsum(indices)
        for pos in np.arange(len(indices) - 1):
            s = datetime.now()
            start, nxt = indices[pos], indices[pos + 1]
            pipe = data[start:nxt].copy()

            for m_col in multi_cols:
                typ = 'int32' if dtype[m_col] == int else 'float32'
                pipe[m_col] = pad(pipe[m_col], padding='post', dtype=typ).tolist()

            # self.logger.info(f"padded_batch take time: {datetime.now() - s}")
            yield pipe

    def online_predict(self, datasource, model_name):
        """Online prediction with ML Engine

        :param datasource: Array list contains many dict objects
        :param model_name: Deployed model name for prediction, for no version provide,
            GCP will get default version
        :return: Prediction result
        """
        model_uri = f'projects/{self.p.project_id}/models/{model_name}'
        ml = self.find_ml()

        result = ml.projects().predict(name=model_uri, body={'instances': datasource}).execute()
        return [rec.get('predictions')[0] for rec in result.get('predictions')]

Service.instance = Service()

