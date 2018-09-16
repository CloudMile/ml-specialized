import tensorflow as tf, shutil, os, pandas as pd, numpy as np
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from datetime import datetime

from . import input, model as m, utils, metadata

class Service(object):
    """Business logic object, all kind of logic write down here to called by controller,
      maybe there are some other assistance object, like input.Input for data handle... etc.

    """
    instance = None
    logger = utils.logger(__name__)

    def __init__(self):
        self.inp = utils.get_instance(input.Input)

    def train_ridge(self, p):
        """Train model with scikit-learn package, see `sklearn.linear_model.Ridge` for
          model comparison.

        :return: None
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge

        tr = self.inp.fill_catg_na(pd.read_csv(p.train_files))
        vl = self.inp.fill_catg_na(pd.read_csv(p.valid_files))
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
            """Because we take log to target, so now take np.expm1 back."""
            label, pred = np.expm1(label), np.expm1(pred)
            return np.sqrt((((label - pred) / label) ** 2).mean())

        def rmse(label, pred):
            return np.sqrt(((label - pred) ** 2).mean())

        tr_pred, vl_pred = ridge.predict(tr_x), ridge.predict(vl_x)

        tr_rmpse = rmspe(tr_y, tr_pred)
        vl_rmpse = rmspe(vl_y, vl_pred)

        tr_rmse = rmse(tr_y, tr_pred)
        vl_rmse = rmse(vl_y, vl_pred)

        self.logger.info('RMSPE on train data: {}, valid data: {}'.format(tr_rmpse, vl_rmpse))
        self.logger.info('RMSE on train data: {}, valid data: {}'.format(tr_rmse, vl_rmse))

    def train(self, p):
        """Train tensorflow model with tf.estimator.Estimator object

        Train spec: wrap train_fn or training hook(optional)
        Eval spec: wrap valid_fn or validation hook(optional)

        :param p: Received parameters
        :return: self
        """
        if isinstance(p, pd.Series):
            for k, v in p.items():
                self.logger.info('{}: {}'.format(k, v))

        self.check_model_name(p.model_name)

        if p.model_name == 'ridge': return self.train_ridge(p)

        # model_dir = p.dnn_model_dir if model_name == 'deep' else p.wnd_model_dir
        model_dir = p.job_dir

        if p.reset and tf.gfile.Exists(model_dir):
            self.logger.info("Deleted job_dir {} to avoid re-use".format(model_dir))
            tf.gfile.DeleteRecursively(model_dir)
            # shutil.rmtree(model_dir, ignore_errors=True)

        tf.gfile.MakeDirs(model_dir)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        # Turn on XLA JIT compilation
        sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        run_config = tf.estimator.RunConfig(
            session_config=sess_config,
            # tf_random_seed=878787,
            log_step_count_steps=p.log_step_count_steps,
            save_checkpoints_steps=p.save_checkpoints_steps,
            # save_checkpoints_secs=HYPER_PARAMS.eval_every_secs,
            keep_checkpoint_max=p.keep_checkpoint_max,
            model_dir=model_dir
        )

        self.logger.info("Model_name: {}".format(p.model_name))
        self.logger.info("Model directory: {}".format(run_config.model_dir))
        model = m.Model(model_dir=model_dir, name=p.model_name)
        model.feature = utils.get_instance(m.Feature)
        model.feature.inp = self.inp

        exporter = m.BestScoreExporter(
            p.export_name,
            lambda: self.inp.serving_fn[p.serving_format](p),
            model_dir=model_dir,
            as_text=False  # change to true if you want to export the model as readable text
        )
        # Train spec
        # tr_hook = input.IteratorInitializerHook()
        train_fn = self.inp.generate_input_fn(
            file_names_pattern=p.train_files,
            mode=tf.estimator.ModeKeys.TRAIN,
            # num_epochs=p.num_epochs,
            batch_size=p.batch_size,
            shuffle=True,
            # hooks=[]
        )
        train_spec = tf.estimator.TrainSpec(
            train_fn,
            max_steps=p.train_steps,
            # hooks=[]
        )
        # Valid spec
        # vl_hook = input.IteratorInitializerHook()
        valid_fn = self.inp.generate_input_fn(
            file_names_pattern=p.valid_files,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=p.batch_size,
            # hooks=[]
        )
        eval_spec = tf.estimator.EvalSpec(
            valid_fn,
            steps=p.valid_steps,
            exporters=[exporter],
            name='estimator-eval',
            throttle_secs=p.throttle_secs,
            # hooks=[]
        )
        # train and evaluate
        tf.estimator.train_and_evaluate(
            model.get_estimator(p, run_config),
            train_spec,
            eval_spec
        )
        return self

    def read_transformed(self, fpath):
        """Read transformed data for model prediction

        :param fpath: File path
        :return: Data with DataFrame type
        """
        return pd.read_csv(fpath, dtype=self.inp.get_processed_dtype(is_serving=True))

    def find_latest_expdir(self, p):
        """Find latest exported directory by specified model name

        :param model_name: Model name in `dnn` `neu_mf`
        :param job_dir: Model checkpoint directory
        :return: Latest directory path
        """
        model_dir = p.job_dir
        # Found latest export dir
        export_dir = '{}/export/{}'.format(model_dir, p.export_name)

        return '{}/{}'.format(export_dir, sorted(tf.gfile.ListDirectory(export_dir))[-1])

    def check_model_name(self, model_name):
        """Check if model name in (`deep` `wide_and_deep` `ridge`)

        :param model_name: Model name in `deep` `wide_and_deep` `ridge`
        :return:
        """
        assert model_name in ('deep', 'wide_and_deep', 'ridge'), \
            "model_name only support ('deep', 'wide_and_deep', 'ridge')"

    def find_ml(self):
        """Return GCP ML service

        :return: GCP ML service object
        """
        credentials = GoogleCredentials.get_application_default()
        return discovery.build('ml', 'v1', credentials=credentials)

    def create_model_rsc(self, p, ml, model_name):
        """Create model repository on GCP ML-Engine

        :param ml: GCP ML service object
        :param model_name: Model name to put on ML-Engine repository
        :return: self
        """
        proj_uri = 'projects/{}'.format(p.project_id)
        try:
            ml.projects().models().create(
                parent=proj_uri, body={'name': model_name, 'onlinePredictionLogging': True}
            ).execute()
        except Exception as e:
            self.logger.warn(e)
        return self

    def clear_model_ver(self, p, ml, model_name):
        """Clear all model versions in repository

        :param ml: GCP ML service object
        :param model_name: Model name to put on ML-Engine repository
        :return: self
        """
        model_rsc = 'projects/{}/models/{}'.format(p.project_id, model_name)
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

    def create_model_ver(self, p, ml, model_name, deployment_uri):
        """Create a new model version with current datetime as version name

        :param ml: GCP ML service object
        :param model_name: Model name to put on ML-Engine repository
        :param deployment_uri: GCS path to locate the saved model
        :return: self
        """
        model_uri = 'projects/{}/models/{}'.format(p.project_id, model_name)
        now = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        version = 'v{}'.format(now)

        self.logger.info('create model {} from {}'.format(model_name, deployment_uri))
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

    def online_predict(self, p, datasource, model_name):
        """Online prediction with ML Engine

        :param datasource: Array list contains many dict objects
        :param model_name: Deployed model name for prediction, for no version provide,
            GCP will get default version
        :return: Prediction result
        """
        model_uri = 'projects/{}/models/{}'.format(p.project_id, model_name)
        ml = self.find_ml()
        # return data type must be in records mode
        result = ml.projects().predict(name=model_uri, body={'instances': datasource}).execute()
        return [rec.get('predictions')[0] for rec in result.get('predictions')]

# Service.instance = Service()

