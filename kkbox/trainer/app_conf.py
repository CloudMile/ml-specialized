import os

class Config(object):
    def __init__(self):
        # Base
        self.project_id = 'ml-team-cloudmile'
        self.api_key_path = 'C:/Users/gary/client_secret.json'
        self.base_dir = os.path.dirname(os.path.dirname(__file__)).replace('\\', '/')
        self.data_path = '{}/data'.format(self.base_dir)
        self.proc_path = '{}/processed'.format(self.data_path)
        self.model_path = '{}/models'.format(self.base_dir)

        # Original data path
        self.raw_train = '{}/train.csv'.format(self.data_path)
        self.raw_test = '{}/test.csv'.format(self.data_path)
        self.raw_members = '{}/members.csv'.format(self.data_path)
        self.raw_songs = '{}/songs.csv'.format(self.data_path)
        self.raw_song_extra_info = '{}/song_extra_info.csv'.format(self.data_path)

        # Processed data path
        self.cleaned_path = '{}/cleaned'.format(self.proc_path)
        self.prepared_path = '{}/prepared'.format(self.proc_path)
        self.fitted_path = '{}/fitted'.format(self.proc_path)
        self.transformed_path = '{}/transformed'.format(self.proc_path)

        self.train_files = '{}/tr.pkl'.format(self.transformed_path)
        self.valid_files = '{}/vl.pkl'.format(self.transformed_path)
        self.test_files = '{}/te.pkl'.format(self.transformed_path)

        # Data prepare relevant parameter
        self.valid_size = 0.1
        self.job_dir = '{}/kkbox'.format(self.model_path)
        # Dir name in {model_dir}/export
        self.export_name = 'estimator'

        # Hyper parameter
        self.embedding_size = 16
        self.first_layer_size = 128
        self.num_layers = 3
        self.scale_factor = 0.7
        self.learning_rate = 0.001
        self.drop_rate = 0.3
        self.first_mlp_layer_size = 512
        self.first_factor_layer_size = 32

        # Training config
        self.job_dir = '{}/neu_mf'.format(self.model_path)
        self.keep_checkpoint_max = 5
        self.log_step_count_steps = 100
        self.train_steps = 4358 * 1
        self.valid_steps = 492
        # Recommend to assign to same as train_steps, for tf.train.cosine_decay,
        # and tune the alpha hyper param to control the lr won't down to zero.
        self.cos_decay_steps = self.train_steps
        self.batch_size = 1000
        self.num_epochs = None
        self.verbosity = 'INFO'
        self.throttle_secs = 600
        self.save_checkpoints_steps = 500

        # Serving relevant
        self.serving_format = 'json'

class CMLEConfig(Config):
    instance = None

    def __init__(self):
        super(CMLEConfig, self).__init__()
        self.base_dir = 'gs://ml-specialized/kkbox'
        self.data_path = '{}/data'.format(self.base_dir)
        self.proc_path = '{}/processed'.format(self.data_path)
        self.model_path = '{}/models'.format(self.base_dir)

        # Original data path
        self.raw_train = '{}/train.csv'.format(self.data_path)
        self.raw_test = '{}/test.csv'.format(self.data_path)
        self.raw_members = '{}/members.csv'.format(self.data_path)
        self.raw_songs = '{}/songs.csv'.format(self.data_path)
        self.raw_song_extra_info = '{}/song_extra_info.csv'.format(self.data_path)

        # Processed data path
        self.cleaned_path = '{}/cleaned'.format(self.proc_path)
        self.prepared_path = '{}/prepared'.format(self.proc_path)
        self.fitted_path = '{}/fitted'.format(self.proc_path)
        self.transformed_path = '{}/transformed'.format(self.proc_path)

        self.train_files = '{}/tr.pkl'.format(self.transformed_path)
        self.valid_files = '{}/vl.pkl'.format(self.transformed_path)
        self.test_files = '{}/te.pkl'.format(self.transformed_path)


def get_config(env=None):
    if env == 'cloud':
        return CMLEConfig()
    else:
        return Config()
