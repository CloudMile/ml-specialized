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

RAW_HEADER = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers',
              'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType',
              'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
              'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'State']

RAW_DTYPES = [str, str, str, int, int,
              int, int, str, str, str,
              str, float, str, str, int,
              str, str, str, str]

HEADER_MAPPING = {
    'Store': 'store',
    'DayOfWeek': 'day_of_week',
    'Date': 'date',
    'Sales': 'sales',
    # 'Customers': 'customers',
    'Open': 'open',
    'Promo': 'promo',
    'StateHoliday': 'state_holiday',
    'SchoolHoliday': 'school_holiday',
    'StoreType': 'store_type',
    'Assortment': 'assortment',
    'State': 'state'
}

# list of all the columns (header) of the input data file(s)
HEADER = ['store', 'day_of_week', 'open', 'promo2',
          'promo', 'state_holiday', 'school_holiday', 'store_type', 'assortment',
          'state', 'month', 'day', 'sales_mean',
          'sales']

# list of the default values of all the columns of the input data, to help decoding the data types of the columns
HEADER_DEFAULTS = [[0], [0], [0], [0],
                   [0], [''], [0], [''], [''],
                   [''], [0], [0], [0.0],
                   [0.0]]

# list of the feature names of type int or float
INPUT_NUMERIC_FEATURE_NAMES = ['sales_mean']

# numeric features constructed, if any, in process_features function in input.py module,
# as part of reading data
CONSTRUCTED_NUMERIC_FEATURE_NAMES = []

# a dictionary of feature names with int values, but to be treated as categorical features.
# In the dictionary, the key is the feature name, and the value is the num_buckets (count of distinct values)
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {
    'open': 2,
    'promo': 2,
    'promo2': 2,
    'school_holiday': 2,
    'day': 32,
    'month': 13,
    'day_of_week': 8,
}

# categorical features with identity constructed, if any, in process_features function in input.py module,
# as part of reading data. Usually include constructed boolean flags
CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}

# a dictionary of categorical features with few nominal values (to be encoded as one-hot indicators)
#  In the dictionary, the key is the feature name, and the value is the list of feature vocabulary
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {
    'state': ['HE', 'HH', 'SH', 'NW', 'BE', 'BY', 'SN', 'RP', 'TH', 'HB,NI', 'BW', 'ST'],
    'state_holiday': ['a', '0', 'b', 'c'],
    'assortment': ['a', 'b', 'c', '0'],
    'store_type': ['a', 'b', 'c', 'd']
}

# a dictionary of categorical features with many values (sparse features)
# In the dictionary, the key is the feature name, and the value is the bucket size + dtype
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {
    # 'date': 10000,
    'store': {'bucket_size': 2000, 'dtype': 'int32'},
    # 'year': {'bucket_size': 1000, 'dtype': 'int32'}
}

# list of all the categorical feature names
INPUT_CATEGORICAL_FEATURE_NAMES = list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.keys())

# list of all the input feature names to be used in the model
INPUT_FEATURE_NAMES = INPUT_NUMERIC_FEATURE_NAMES + INPUT_CATEGORICAL_FEATURE_NAMES

# the column include the weight of each record
WEIGHT_COLUMN_NAME = None

# target feature name (response or class variable)
TARGET_NAME = 'sales'

# list of the class values (labels) in a classification dataset
TARGET_LABELS = []

# list of the columns expected during serving (which probably different than the header of the training data)
SERVING_COLUMNS = HEADER.copy()
SERVING_COLUMNS.remove(TARGET_NAME)

# list of the default values of all the columns of the serving data, to help decoding the data types of the columns
SERVING_DEFAULTS = [[0], [0], [0], [0], [0],
                    [0], [''], [0], [''], [''],
                    [''], [0], [0], [0.0]]