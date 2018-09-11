import os

class Config(object):
    def __init__(self):
        # Base
        self.project_id = 'ml-team-cloudmile'
        self.api_key_path = 'C:/Users/gary/client_secret.json'
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.data_path = f'{self.base_dir}/data'
        self.proc_path = f'{self.data_path}/processed'
        self.model_path = f'{self.base_dir}/models'

        # Original data path
        self.store_data = f'{self.data_path}/store.csv'
        self.store_state = f'{self.data_path}/store_states.csv'
        self.train_data = f'{self.data_path}/train.csv'
        self.valid_data = f'{self.data_path}/valid.csv'
        self.test_data = f'{self.data_path}/test.csv'

        # Processed data path
        self.cleaned_path = f'{self.proc_path}/cleaned'
        self.prepared_path = f'{self.proc_path}/prepared'
        self.fitted_path = f'{self.proc_path}/fitted'
        self.transformed_path = f'{self.proc_path}/transformed'

        # train_full_pr = f'{proc_path}/train_full_pr.pkl'
        # Support base match pattern, see tf.matching_files function
        self.train_files = f'{self.transformed_path}/tr.csv'
        self.valid_files = f'{self.transformed_path}/vl.csv'
        self.feature_stats_file = f'{self.fitted_path}/stats.json'
        self.tr_dt_file = f'{self.transformed_path}/tr_date.json'
        self.vl_dt_file = f'{self.transformed_path}/vl_date.json'

        # Data prepare relevant parameter
        self.valid_size = 0.3
        self.dnn_model_dir = f'{self.model_path}/dnn_regressor_128_64_32'
        self.wnd_model_dir = f'{self.model_path}/wide_and_deep'
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

instance = Config()

class CMLEConfig(Config):
    def __init__(self):
        super(CMLEConfig, self).__init__()
        self.base_dir = f'gs://ml-specialized/rossmann'
        self.data_path = f'{self.base_dir}/data'
        self.proc_path = f'{self.data_path}/processed'
        self.model_path = f'{self.base_dir}/models'

        # Original data path
        self.store_data = f'{self.data_path}/store.csv'
        self.store_state = f'{self.data_path}/store_states.csv'
        self.train_data = f'{self.data_path}/train.csv'
        self.valid_data = f'{self.data_path}/valid.csv'
        self.test_data = f'{self.data_path}/test.csv'

        # Processed data path
        self.cleaned_path = f'{self.proc_path}/cleaned'
        self.prepared_path = f'{self.proc_path}/prepared'
        self.fitted_path = f'{self.proc_path}/fitted'
        self.transformed_path = f'{self.proc_path}/transformed'

        # Support base match pattern, see tf.matching_files function
        self.train_files = f'{self.transformed_path}/tr.csv'
        self.valid_files = f'{self.transformed_path}/vl.csv'
        self.feature_stats_file = f'{self.fitted_path}/stats.json'
        self.tr_dt_file = f'{self.transformed_path}/tr_date.json'
        self.vl_dt_file = f'{self.transformed_path}/vl_date.json'

        #
        self.model_name = "wide_and_deep"

def get_config(env=None):
    if env == 'cloud':
        return CMLEConfig()
    else:
        return Config()