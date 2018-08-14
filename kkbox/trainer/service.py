import tensorflow as tf, shutil, os, pandas as pd, numpy as np
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from datetime import datetime

from . import app_conf, input, model as m
from .utils import utils

class Service(object):
    instance = None
    logger = utils.logger(__name__)

    def __init__(self):
        self.p: app_conf.Config = app_conf.instance
        self.inp: input.Input = input.Input.instance

    def train(self, train_fn=None, tr_hook=None, valid_fn=None, vl_hook=None,
              reset=True, model_name='dnn'):
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
        # estimator = model.get_estimator(run_config)
        # for epoch in range(10):
        #     estimator.train(train_fn, steps=10)
        #     estimator.evaluate(valid_fn, steps=10)
        return self

    def find_latest_expdir(self, model_name):
        self.check_model_name(model_name)
        model_dir = self.p.neu_mf_model_dir if model_name == 'neu_mf' else self.p.model_dir
        # Found latest export dir
        export_dir = f'{model_dir}/export/{self.p.export_name}'
        return f'{export_dir}/{sorted(os.listdir(export_dir))[-1]}'

    def check_model_name(self, model_name):
        assert model_name in ('dnn', 'neu_mf'), "model_name only support ('dnn', 'neu_mf')"

    def read_transformed(self, fpath):
        data = self.inp.clean(fpath, is_serving=True)
        data = self.inp.prepare(data, is_serving=True)
        return self.inp.transform(data, is_serving=True)

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

    def padded_batch(self, data, n_batch=1000, callback=None):
        assert n_batch <= 1000, 'Prediction batch size must less equal than 1000!'

        multi_cols = self.inp.get_multi_cols(is_serving=True)
        pad = tf.keras.preprocessing.sequence.pad_sequences
        dtype = self.inp.get_dtype(is_serving=True)

        n_total = len(data)
        cache = {'idx': -1, 'count': 0}
        def map_pred_fn(pipe):
            ret = None
            if cache['idx'] >= 0:
                for m_col in multi_cols:
                    typ = 'int32' if dtype[m_col] == int else 'float32'
                    maxlen = pipe[m_col].map(len).max()
                    # pad(pipe[m_col], padding='post', dtype=typ).tolist()
                    def pad_fn(tp):
                        padlen = maxlen - len(tp)
                        if typ == 'int32':
                            return tuple(map(int, tp + (0,) * padlen))
                        else:
                            return tuple(map(float, tp + (0,) * padlen))
                    pipe[m_col] = pipe[m_col].map(pad_fn)

                if callback is not None:
                    ret = callback(pipe)
                else:
                    ret = pipe
                cache['count'] += len(pipe)
                self.logger.info(f'{cache.get("count")}/{n_total} ... ')
            else:
                self.logger.info('pandas apply testing ...')
            cache['idx'] += 1
            return ret

        self.logger.info(f'Total {n_total} to predict ...')
        result = data.groupby(np.arange(len(data)) // n_batch).apply(map_pred_fn)

        # todo hack
        print(result.shape)

        if isinstance(result, pd.DataFrame):
            return result
        else:
            return np.concatenate(result, 0)

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

