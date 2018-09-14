import os

class Config(object):
    instance = None

    def __init__(self):
        # Base
        self.project_id = 'ml-team-cloudmile'
        self.api_key_path = 'C:/Users/gary/client_secret.json'
        self.base_dir = os.path.dirname(os.path.dirname(__file__)).replace('\\', '/')
        self.set_path()

        # Data prepare relevant parameter
        self.valid_size = 0.3
        # Dir name in {model_dir}/export
        self.export_name = 'estimator'

        # Hyper parameter
        self.embedding_size = 16
        self.first_layer_size = 128
        self.num_layers = 3
        self.scale_factor = 0.7
        self.learning_rate = 0.001
        self.drop_rate = 0.3

        # Training config
        self.job_dir = '{}/dnn_regressor'.format(self.model_path)
        self.keep_checkpoint_max = 5
        self.log_step_count_steps = 500
        self.train_steps = 2308 * 8
        self.valid_steps = 989
        self.batch_size = 256
        self.num_epochs = None
        self.verbosity = 'INFO'
        self.throttle_secs = 60
        self.save_checkpoints_steps = 2308

        # Serving relevant
        self.serving_format = 'json'

    def set_path(self):
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
        self.test_files = '{}/te.csv'.format(self.transformed_path)
        self.feature_stats_file = '{}/stats.json'.format(self.fitted_path)
        self.tr_dt_file = '{}/tr_date.json'.format(self.transformed_path)
        self.vl_dt_file = '{}/vl_date.json'.format(self.transformed_path)

# Config.instance = Config()

class CMLEConfig(Config):
    instance = None

    def __init__(self):
        super(CMLEConfig, self).__init__()
        self.base_dir = 'gs://ml-specialized/rossmann'
        self.set_path()


# CMLEConfig.instance = CMLEConfig()

def get_config(env=None):
    if env == 'cloud':
        return CMLEConfig()
    else:
        return Config()