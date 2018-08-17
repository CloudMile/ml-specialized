import tensorflow as tf

# raw data feature
RAW_HEADER = [
    'msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type',
    'city', 'bd', 'gender', 'registered_via', 'registration_init_time',
    'expiration_date', 'song_length', 'genre_ids', 'artist_name', 'composer',
    'lyricist', 'language', 'name', 'isrc',
    'target'
]

# raw data type
RAW_DTYPES = [
    str, str, str, str, str,
    str, str, str, str, str,
    str, float, str, str, str,
    str, str, str, str,
    int
]

MSNO_EXTRA_COLS = [
    'msno_age_catg', 'msno_age_num', 'msno_tenure',
    'msno_pos_query_hist', 'msno_pos_query_count',
    'msno_neg_query_hist', 'msno_neg_query_count',
    'msno_artist_name_hist', 'msno_artist_name_count', 'msno_artist_name_mean',
    'msno_composer_hist', 'msno_composer_count', 'msno_composer_mean',
    'msno_genre_ids_hist', 'msno_genre_ids_count', 'msno_genre_ids_mean',
    'msno_language_hist', 'msno_language_count', 'msno_language_mean',
    'msno_lyricist_hist', 'msno_lyricist_count', 'msno_lyricist_mean',
    'msno_source_screen_name_hist', 'msno_source_screen_name_count',
    'msno_source_system_tab_hist', 'msno_source_system_tab_count',
    'msno_source_type_hist', 'msno_source_type_count'
]

MEMBER_FEATURES = [
    # 'msno',
    'city', 'gender', 'registered_via',
    'registration_init_time', 'expiration_date',
] + MSNO_EXTRA_COLS

SONG_EXTRA_COLS = [
    'song_cc', 'song_xxx', 'song_yy', 'song_length', 'song_pplrty', 'song_clicks',
    'song_artist_name_len', 'song_composer_len', 'song_lyricist_len', 'song_genre_ids_len',
    'song_city_hist', 'song_city_count', 'song_city_mean',
    'song_gender_hist', 'song_gender_count', 'song_gender_mean',
    'song_msno_age_catg_hist', 'song_msno_age_catg_count', 'song_msno_age_catg_mean',
    'song_registered_via_hist', 'song_registered_via_count',
    'song_source_screen_name_hist', 'song_source_screen_name_count',
    'song_source_system_tab_hist', 'song_source_system_tab_count',
    'song_source_type_hist', 'song_source_type_count'
]

SONG_FEATURES = [
    'song_id',
    'genre_ids', 'artist_name', 'composer', 'lyricist', 'language',
] + SONG_EXTRA_COLS

CONTEXT_FEATURES = ['source_system_tab', 'source_screen_name', 'source_type']
TARGET_NAME = ['target']
HEADER = MEMBER_FEATURES + SONG_FEATURES + CONTEXT_FEATURES + TARGET_NAME

# tensor data types
HEADER_DTYPES = [
    # Members
    tf.int32, tf.int32, tf.int32,
    tf.float32, tf.float32, tf.int32, tf.float32, tf.float32,
    tf.int32, tf.float32,
    tf.int32, tf.float32,
    tf.int32, tf.float32, tf.float32,
    tf.int32, tf.float32, tf.float32,
    tf.int32, tf.float32, tf.float32,
    tf.int32, tf.float32, tf.float32,
    tf.int32, tf.float32, tf.float32,
    tf.int32, tf.float32,
    tf.int32, tf.float32,
    tf.int32, tf.float32,
    # Songs
    tf.int32,
    tf.int32, tf.int32, tf.int32, tf.int32, tf.int32,
    tf.int32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32,
    tf.int32, tf.int32, tf.int32, tf.int32,
    tf.int32, tf.float32, tf.float32,
    tf.int32, tf.float32, tf.float32,
    tf.int32, tf.float32, tf.float32,
    tf.int32, tf.float32,
    tf.int32, tf.float32,
    tf.int32, tf.float32,
    tf.int32, tf.float32,
    # Context
    tf.int32, tf.int32, tf.int32,
    # Target
    tf.int32
]

# list of the default values of all the columns of the input data, to help decoding the data types of the columns
HEADER_DEFAULTS = [
    # Members
    # [0],
    [0], [0], [0],
    [0.], [0.], [0], [0.], [0.],
    [0], [0.],
    [0], [0.],
    [0], [0.], [0.],
    [0], [0.], [0.],
    [0], [0.], [0.],
    [0], [0.], [0.],
    [0], [0.], [0.],
    [0], [0.],
    [0], [0.],
    [0], [0.],
    # Songs
    [0],
    [0], [0], [0], [0], [0],
    [0], [0], [0.], [0.], [0.], [0.],
    [0], [0], [0], [0],
    [0], [0.], [0.],
    [0], [0.], [0.],
    [0], [0.], [0.],
    [0], [0.],
    [0], [0.],
    [0], [0.],
    [0], [0.],
    # Context
    [0], [0], [0],
    # Target
    [0]
]

# embedding feature label and dimension.
EMB_COLS = {
    'city': 3, 'gender': 3, 'registered_via': 3, 'msno_age_catg': 3,
    'song_id': 16, 'genre_ids': 8, 'artist_name': 16, 'composer': 16, 'lyricist': 16, 'language': 3,
    'song_cc': 3, 'song_xxx': 8, 'source_system_tab': 3, 'source_screen_name': 3, 'source_type': 3,
    # 'song_query': 16,
}

# numeric feature labels
NUMERIC_COLS = [
    'registration_init_time', 'expiration_date', 'msno_age_num', 'msno_tenure', 'song_yy',
    'song_length', 'song_pplrty', 'song_clicks'
]

# target feature name (response or class variable)
TARGET_NAME = 'target'

# list of the columns expected during serving (which probably different than the header of the training data)
SERVING_COLUMNS = HEADER.copy()
SERVING_COLUMNS.remove(TARGET_NAME)

# list of the default values of all the columns of the serving data, to help decoding the data types of the columns
SERVING_DEFAULTS = HEADER_DEFAULTS.copy()
SERVING_DEFAULTS.pop(-1)

SERVING_DTYPES = HEADER_DTYPES.copy()
SERVING_DTYPES.pop(-1)

