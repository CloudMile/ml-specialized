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
    valid_size = 0.1
    model_dir = f'{model_path}/kkbox_dnn_adam_max_norm'
    neu_mf_model_dir = f'{model_path}/kkbox_neumf_no_share_embedding'
    # Dir name in {model_dir}/export
    export_name = 'estimator'

    # Hyper parameter
    keep_checkpoint_max = 5
    log_step_count_steps = 100
    train_steps = 4358 * 1
    valid_steps = 492
    batch_size = 1000
    # Number of loops for dataset
    num_epochs = 1
    save_checkpoints_steps = 500
    eval_every_secs = 1000
    # Recommend to assign to same as train_steps, for tf.train.cosine_decay,
    # and tune the alpha hyper param to control the lr won't down to zero.
    cos_decay_steps = train_steps
    initial_learning_rate = 0.001
    mlp_layers = [512, 128, 64]
    factor_layers = [32, 16]
    drop_rate = 0
    # momentum = 0.99
    reg_scale = 0

    # Serving relevant
    serving_format = 'json' # [json]

instance = Config()

