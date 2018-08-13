import random, tensorflow as tf, shutil, os, pandas as pd, numpy as np
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from datetime import datetime

from . import app_conf, input, model as m, utils, metadata

class Service(object):
    instance = None
    logger = utils.logger(__name__)

    def __init__(self):
        self.p :app_conf.Config = app_conf.instance
        self.inp :input.Input = input.Input.instance

    def train_ridge(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge

        tr = self.inp.fill_catg_na(pd.read_csv(app_conf.instance.train_files))
        vl = self.inp.fill_catg_na(pd.read_csv(app_conf.instance.valid_files))
        cut_pos = np.cumsum([len(tr), len(vl)]).tolist()

        # Numeric features
        merge = pd.concat([tr, vl], 0, ignore_index=True)
        num_feats = merge[metadata.INPUT_NUMERIC_FEATURE_NAMES]
        num_feats = StandardScaler().fit_transform(num_feats.values)
        # Categorical features
        catg_feats = merge[metadata.INPUT_CATEGORICAL_FEATURE_NAMES]
        catg_feats = pd.get_dummies(catg_feats, columns=metadata.INPUT_CATEGORICAL_FEATURE_NAMES).values

        # Split train, valid data
        self.logger.info("Split train valid")
        data = np.c_[num_feats, catg_feats, merge.sales.values[:, None]]
        tr_x = data[:cut_pos[0]]
        vl_x = data[cut_pos[0]:cut_pos[1]]
        tr_x, tr_y = tr_x[:, :-1], tr_x[:, -1]
        vl_x, vl_y = vl_x[:, :-1], vl_x[:, -1]

        ridge = Ridge(0.5)
        ridge.fit(tr_x, tr_y)

        def rmspe(label, pred):
            label, pred = np.expm1(label), np.expm1(pred)
            return np.sqrt((((label - pred) / label) ** 2).mean())

        def rmse(label, pred):
            # label, pred = np.expm1(label), np.expm1(pred)
            return np.sqrt(((label - pred) ** 2).mean())

        tr_pred, vl_pred = ridge.predict(tr_x), ridge.predict(vl_x)

        tr_rmpse = rmspe(tr_y, tr_pred)
        vl_rmpse = rmspe(vl_y, vl_pred)

        tr_rmse = rmse(tr_y, tr_pred)
        vl_rmse = rmse(vl_y, vl_pred)

        self.logger.info(f'RMSPE on train data: {tr_rmpse}, valid data: {vl_rmpse}')
        self.logger.info(f'RMSE on train data: {tr_rmse}, valid data: {vl_rmse}')

    def train(self, model_name='deep', reset=True):

        self.check_model_name(model_name)

        if model_name == 'ridge': return self.train_ridge()

        model_dir = self.p.dnn_model_dir if model_name == 'deep' else self.p.wnd_model_dir

        if reset and tf.gfile.Exists(model_dir):
            self.logger.info(f"Deleted job_dir {model_dir} to avoid re-use")
            shutil.rmtree(model_dir, ignore_errors=True)

        os.makedirs(model_dir, exist_ok=True)

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
            model_dir=model_dir
        )

        self.logger.info(f"Model_name: {model_name}")
        self.logger.info(f"Model directory: {run_config.model_dir}")
        model = m.Model(model_dir=model_dir, name=model_name)


        exporter = m.BestScoreExporter(
            self.p.export_name,
            self.inp.serving_fn[self.p.serving_format],
            model_dir=model_dir,
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

    def find_latest_expdir(self, model_name):
        self.check_model_name(model_name)
        model_dir = self.p.dnn_model_dir if model_name == 'deep' else self.p.wnd_model_dir
        # Found latest export dir
        export_dir = f'{model_dir}/export/{self.p.export_name}'
        return f'{export_dir}/{sorted(os.listdir(export_dir))[-1]}'

    def check_model_name(self, model_name):
        assert model_name in ('deep', 'wide_and_deep', 'ridge'), \
            "model_name only support ('deep', 'wide_and_deep', 'ridge')"

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

