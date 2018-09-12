import os

class Config(object):
    instance = None

    def __init__(self):
        # Base
        self.project_id = 'ml-team-cloudmile'
        self.api_key_path = 'C:/Users/gary/client_secret.json'
        self.base_dir = os.path.dirname(os.path.dirname(__file__)).replace('\\', '/')
        self.data_path = '{}/data'.format(self.base_dir)
        self.proc_path = '{}/processed'.format(self.data_path)
        self.model_path = '{}/models'.format(self.base_dir)

        # Original data path
        self.store_data = '{}/store.csv'.format(self.data_path)
        self.store_state = '{}/store_states.csv'.format(self.data_path)
        self.train_data = '{}/train.csv'.format(self.data_path)
        self.valid_data = '{}/valid.csv'.format(self.data_path)
        self.test_data = '{}/test.csv'.format(self.data_path)

        # Processed data path
        self.cleaned_path = '{}/cleaned'.format(self.proc_path)
        self.prepared_path = '{}/prepared'.format(self.proc_path)
        self.fitted_path = '{}/fitted'.format(self.proc_path)
        self.transformed_path = '{}/transformed'.format(self.proc_path)

        # train_full_pr = '{proc_path}/train_full_pr.pkl'
        # Support base match pattern, see tf.matching_files function
        self.train_files = '{}/tr.csv'.format(self.transformed_path)
        self.valid_files = '{}/vl.csv'.format(self.transformed_path)
        self.feature_stats_file = '{}/stats.json'.format(self.fitted_path)
        self.tr_dt_file = '{}/tr_date.json'.format(self.transformed_path)
        self.vl_dt_file = '{}/vl_date.json'.format(self.transformed_path)

        # Data prepare relevant parameter
        self.valid_size = 0.3
        # self.job_dir = '{self.model_path}/dnn_regressor'
        # self.dnn_model_dir = '{self.model_path}/dnn_regressor_128_64_32'
        # self.wnd_model_dir = '{self.model_path}/wide_and_deep'
        # Dir name in {model_dir}/export
        self.export_name = 'estimator'

        # Hyper parameter
        self.mlp_layers = [128, 64, 32]
        # mlp_layers = [1024, 512, 256]
        self.learning_rate = 0.005
        self.drop_rate = 0.3
        self.keep_checkpoint_max = 3
        self.log_step_count_steps = 500
        self.train_steps = 2308 * 8
        self.valid_steps = 989
        self.batch_size = 256
        self.num_epochs = 1
        # eval_every_secs = 600
        # encode_one_hot = False
        # as_wide_columns = False

        # Serving relevant
        self.serving_format = 'json'

# Config.instance = Config()

class CMLEConfig(Config):
    instance = None

    def __init__(self):
        super(CMLEConfig, self).__init__()
        self.base_dir = 'gs://ml-specialized/rossmann'
        self.data_path = '{}/data'.format(self.base_dir)
        self.proc_path = '{}/processed'.format(self.data_path)
        self.model_path = '{}/models'.format(self.base_dir)

        # Original data path
        self.store_data = '{}/store.csv'.format(self.data_path)
        self.store_state = '{}/store_states.csv'.format(self.data_path)
        self.train_data = '{}/train.csv'.format(self.data_path)
        self.valid_data = '{}/valid.csv'.format(self.data_path)
        self.test_data = '{}/test.csv'.format(self.data_path)

        # Processed data path
        self.cleaned_path = '{}/cleaned'.format(self.proc_path)
        self.prepared_path = '{}/prepared'.format(self.proc_path)
        self.fitted_path = '{}/fitted'.format(self.proc_path)
        self.transformed_path = '{}/transformed'.format(self.proc_path)

        # Support base match pattern, see tf.matching_files function
        self.train_files = '{}/tr.csv'.format(self.transformed_path)
        self.valid_files = '{}/vl.csv'.format(self.transformed_path)
        self.feature_stats_file = '{}/stats.json'.format(self.fitted_path)
        self.tr_dt_file = '{}/tr_date.json'.format(self.transformed_path)
        self.vl_dt_file = '{}/vl_date.json'.format(self.transformed_path)


# CMLEConfig.instance = CMLEConfig()

def get_config(env=None):
    if env == 'cloud':
        return CMLEConfig()
    else:
        return Config()