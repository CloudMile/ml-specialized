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
    store_data = f'{data_path}/store.csv'
    store_state = f'{data_path}/store_states.csv'
    train_data = f'{data_path}/train.csv'
    valid_data = f'{data_path}/valid.csv'
    test_data = f'{data_path}/test.csv'

    # Processed data path
    cleaned_path = f'{proc_path}/cleaned'
    prepared_path = f'{proc_path}/prepared'
    fitted_path = f'{proc_path}/fitted'
    transformed_path = f'{proc_path}/transformed'

    train_full_pr = f'{proc_path}/train_full_pr.pkl'
    # Support base match pattern, see tf.matching_files function
    train_files = f'{transformed_path}/tr.csv'
    valid_files = f'{transformed_path}/vl.csv'
    feature_stats_file = f'{fitted_path}/stats.json'
    tr_dt_file = f'{transformed_path}/tr_date.json'
    vl_dt_file = f'{transformed_path}/vl_date.json'

    # Data prepare relevant parameter
    valid_size = 0.3
    model_dir = f'{model_path}/dnn_regressor'
    # Dir name in {model_dir}/export
    export_name = 'estimator'

    # Hyper parameter
    mlp_layers = [128, 64, 32]
    learning_rate = 0.005
    keep_checkpoint_max = 3
    log_step_count_steps = 500
    train_steps = 2308 * 5
    valid_steps = 989
    batch_size = 256
    num_epochs = 10
    # eval_every_secs = 600
    encode_one_hot = False
    as_wide_columns = False

    # Serving relevant
    serving_format = 'json' # [json]

instance = Config()

