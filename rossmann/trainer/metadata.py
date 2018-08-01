# task type can be either 'classification' or 'regression', based on the target feature in the dataset
RAW_HEADER = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
       'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
       'CompetitionDistance', 'CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
       'Promo2SinceYear', 'PromoInterval', 'State']

RAW_DTYPES = [int, str, str, float, float, str, str,
              str, str, str, str,
              float, str,
              str, str, str,
              str, str, str]

HEADER_MAPPING = {
    # train, valid, test
    'Store': 'store',
    'DayOfWeek': 'day_of_week',
    'Date': 'date',
    'Sales': 'sales',
    'Open': 'open',
    'Promo': 'promo',
    'StateHoliday': 'state_holiday',
    'SchoolHoliday': 'school_holiday',
    'Customers': 'customers',
    # store
    'StoreType': 'store_type',
    'Assortment': 'assortment',
    'CompetitionDistance': 'competition_distance',
    'CompetitionOpenSinceMonth': 'competition_open_since_month',
    'CompetitionOpenSinceYear': 'competition_open_since_year',
    'Promo2SinceWeek': 'promo2since_week',
    'Promo2SinceYear': 'promo2since_year',
    'Promo2': 'promo2',
    'PromoInterval': 'promo_interval',
    # store_states
    'State': 'state'
}


# list of all the columns (header) of the input data file(s)
HEADER = ['store', 'day_of_week', 'open', 'promo2',
          'promo', 'state_holiday', 'school_holiday', 'store_type', 'assortment',
          'state', 'year', 'month', 'day', 'sales_mean',
          'competition_distance',
          'competition_open_since', 'competition_open_since_month', 'competition_open_since_year',
          'promo2since', 'promo2since_week', 'promo2since_year',
          'sales']

# list of the default values of all the columns of the input data, to help decoding the data types of the columns
HEADER_DEFAULTS = [[0], [''], [''], [''],
                   [''], [''], [''], [''], [''],
                   [''], [''], [''], [''], [0.0],
                   [0.0],
                   [0.0], [''], [''],
                   [0.0], [''], [''],
                   [0.0]]

# list of the feature names of type int or float
INPUT_NUMERIC_FEATURE_NAMES = ['sales_mean']

# numeric features constructed, if any, in process_features function in input.py module,
# as part of reading data
CONSTRUCTED_NUMERIC_FEATURE_NAMES = ['competition_open_since', 'promo2since', 'competition_distance']


INPUT_NUMERIC_FEATURE_NAMES = INPUT_NUMERIC_FEATURE_NAMES + CONSTRUCTED_NUMERIC_FEATURE_NAMES

# a dictionary of feature names with int values, but to be treated as categorical features.
# In the dictionary, the key is the feature name, and the value is the num_buckets (count of distinct values)
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {
    # 'open': 2,
    # 'promo': 2,
    # 'promo2': 2,
    # 'school_holiday': 2
}

# categorical features with identity constructed, if any, in process_features function in input.py module,
# as part of reading data. Usually include constructed boolean flags
CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}

# a dictionary of categorical features with few nominal values (to be encoded as one-hot indicators)
#  In the dictionary, the key is the feature name, and the value is the list of feature vocabulary
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {
    # Embedding features
    'state': ['HE', 'HH', 'SH', 'NW', 'BE', 'BY', 'SN', 'RP', 'TH', 'HB,NI', 'BW', 'ST'],
    'state_holiday': ['a', 'b', 'c', '0'],
    'assortment': ['a', 'b', 'c', '0'],
    'store_type': ['a', 'b', 'c', 'd'],
    'competition_open_since_month': ['9', '11', '12', '4', '10', '8', '3', '6', '5', '1', '2', '7'],
    'competition_open_since_year': ['2008', '2007', '2006', '2009', '2015', '2013', '2014', '2000',
                                    '2011', '2010', '2005', '1999', '2003', '2012', '2004', '2002',
                                    '1961', '1995', '2001', '1990', '1994', '1900', '1998'],
    'promo2since_week': ['13', '14', '1', '45', '40', '26', '22', '5', '6', '10', '31',
                         '37', '9', '39', '27', '18', '35', '23', '48', '36', '50',
                         '44', '49', '28'],
    'promo2since_year': ['2010', '2011', '2012', '2009', '2014', '2015', '2013'],
    'year': ['2015', '2014', '2013'],
    'month': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    'day': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
            '31'],

    # OneHot features
    'day_of_week': ['1', '2', '3', '4', '5', '6', '7'],
    'open': ['0', '1'],
    'promo': ['0', '1'],
    'promo2': ['0', '1'],
    'school_holiday': ['0', '1']
}

# a dictionary of categorical features with many values (sparse features)
# In the dictionary, the key is the feature name, and the value is the bucket size + dtype
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {
    'store': {'bucket_size': 2000, 'dtype': 'int32'}
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
# TARGET_LABELS = []

# list of the columns expected during serving (which probably different than the header of the training data)
SERVING_COLUMNS = HEADER.copy()
SERVING_COLUMNS.remove(TARGET_NAME)

# list of the default values of all the columns of the serving data, to help decoding the data types of the columns
SERVING_DEFAULTS = HEADER_DEFAULTS.copy()
SERVING_DEFAULTS.pop(-1)
