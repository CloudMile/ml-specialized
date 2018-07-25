import os

class Config(object):
    # Base
    project_id = 'ml-team-cloudmile'
    api_key_path = 'C:/Users/gary/client_secret.json'
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = f'{base_dir}/data'
    proc_path = f'{data_path}/processed'
    model_path = f'{base_dir}/models'

    # Original data path
    raw_train = f'{data_path}/train.csv'
    raw_test = f'{data_path}/test.csv'
    raw_members = f'{data_path}/members.csv'
    raw_songs = f'{data_path}/songs.csv'
    raw_song_extra_info = f'{data_path}/song_extra_info.csv'

    # Processed data path
    cleaned_path = f'{proc_path}/cleaned'
    prepared_path = f'{proc_path}/prepared'
    fitted_path = f'{proc_path}/fitted'
    transformed_path = f'{proc_path}/transformed'

    train_files = f'{transformed_path}/tr.pkl'
    valid_files = f'{transformed_path}/vl.pkl'
    test_files = f'{transformed_path}/te.pkl'

    # Data prepare relevant parameter
    valid_size = 0.2
    model_dir = f'{model_path}/kkbox'
    # Dir name in {model_dir}/export
    export_name = 'estimator'

    # Hyper parameter
    keep_checkpoint_max = 5
    log_step_count_steps = 100
    train_steps = 3874 * 5
    valid_steps = 977
    batch_size = 1000
    num_epochs = 1
    save_checkpoints_steps = 500
    eval_every_secs = 4800

    # Serving relevant
    serving_format = 'json' # [json]

instance = Config()

