#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ************************************************************************
# YOU NEED TO MODIFY THE META DATA TO ADAPT THE TRAINER TEMPLATE YOUR DATA
# ************************************************************************

# task type can be either 'classification' or 'regression', based on the target feature in the dataset
TASK_TYPE = 'custom'

# RAW_HEADER = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers',
#               'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType',
#               'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
#               'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'State']
#
# RAW_DTYPES = [str, str, str, int, int,
#               int, int, str, str, str,
#               str, float, str, str, int,
#               str, str, str, str]
#
# HEADER_MAPPING = {
#     'Store': 'store',
#     'DayOfWeek': 'day_of_week',
#     'Date': 'date',
#     'Sales': 'sales',
#     # 'Customers': 'customers',
#     'Open': 'open',
#     'Promo': 'promo',
#     'StateHoliday': 'state_holiday',
#     'SchoolHoliday': 'school_holiday',
#     'StoreType': 'store_type',
#     'Assortment': 'assortment',
#     'State': 'state'
# }

MEMBER_FEATURES = [
    # 'msno',
    'city', 'gender', 'registered_via',
    'registration_init_time', 'expiration_date', 'msno_age_catg', 'msno_age_num', 'msno_tenure',
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
SONG_FEATURES = [
    'song_id',
    'genre_ids', 'artist_name', 'composer', 'lyricist', 'language',
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
CONTEXT_FEATURES = ['source_system_tab', 'source_screen_name', 'source_type']
TARGET_NAME = ['target']
HEADER = MEMBER_FEATURES + SONG_FEATURES + CONTEXT_FEATURES + TARGET_NAME

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

HEADER_DF_DTYPES = {
    'city': int,
    'gender': int,
    'registered_via': int,
    'msno_age_catg': int,
    'language': int,
    'song_artist_name_len': int,
    'song_composer_len': int,
    'song_lyricist_len': int,
    'song_genre_ids_len': int,
    'song_cc': int,
    'song_xxx': int
}

# target feature name (response or class variable)
TARGET_NAME = 'target'

# list of the columns expected during serving (which probably different than the header of the training data)
SERVING_COLUMNS = HEADER.copy()
SERVING_COLUMNS.remove(TARGET_NAME)

# list of the default values of all the columns of the serving data, to help decoding the data types of the columns
SERVING_DEFAULTS = HEADER_DEFAULTS.copy()
SERVING_DEFAULTS.pop(-1)


# # list of the feature names of type int or float
# INPUT_NUMERIC_FEATURE_NAMES = []
#
# # numeric features constructed, if any, in process_features function in input.py module,
# # as part of reading data
# CONSTRUCTED_NUMERIC_FEATURE_NAMES = []
#
# # a dictionary of feature names with int values, but to be treated as categorical features.
# # In the dictionary, the key is the feature name, and the value is the num_buckets (count of distinct values)
# INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}
#
# # categorical features with identity constructed, if any, in process_features function in input.py module,
# # as part of reading data. Usually include constructed boolean flags
# CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}
#
# # a dictionary of categorical features with few nominal values (to be encoded as one-hot indicators)
# #  In the dictionary, the key is the feature name, and the value is the list of feature vocabulary
# INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {}
#
# # a dictionary of categorical features with many values (sparse features)
# # In the dictionary, the key is the feature name, and the value is the bucket size + dtype
# INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {}
#
# # list of all the categorical feature names
# INPUT_CATEGORICAL_FEATURE_NAMES = list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY.keys()) \
#                                   + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) \
#                                   + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.keys())
#
# # list of all the input feature names to be used in the model
# INPUT_FEATURE_NAMES = INPUT_NUMERIC_FEATURE_NAMES + INPUT_CATEGORICAL_FEATURE_NAMES
#
# # the column include the weight of each record
# WEIGHT_COLUMN_NAME = None
#
# # list of the class values (labels) in a classification dataset
# TARGET_LABELS = []

