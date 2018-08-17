import yaml, codecs, logging, os, pandas as pd, numpy as np, re, pickle

from logging import config
from datetime import datetime
from collections import Counter

class Logging(object):
    """Logging object"""
    instance = None

    @staticmethod
    def get_logger(name):
        """Initialize the logging object by name

        :param name: Logger name
        :return: Object from `logging.getLogger`
        """
        if Logging.instance is None:
            print(f'init logger instance ...')

            log_conf_path = f'{os.path.dirname(os.path.dirname(__file__))}/logging.yaml'
            with codecs.open(log_conf_path, 'r', 'utf-8') as r:
                logging.config.dictConfig(yaml.load(r))
            Logging.instance = logging

        return Logging.instance.getLogger(name)

def logger(name):
    """Short path of Logging.logger"""
    return Logging.get_logger(name)

def cmd(commands):
    """Execute command in python code"""
    import subprocess

    proc = subprocess.Popen(commands, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    outs = []
    while proc.poll() is None:
        output = proc.stdout.readline()
        decode = None
        for encode in ('utf-8', 'big5'):
            try:
                decode = output.decode(encode)
            except:
                pass
        assert decode is not None, 'decode failed!'
        outs.append(decode)
    return ''.join(outs)

def timestamp():
    return int(datetime.now().timestamp())

def deep_walk(path, prefix:str=None):
    """Deep walk directory, return all object in the directory with specified prefix(context) string."""
    path = os.path.abspath(path)
    if not prefix:
        prefix = ''
    for root, dirs, files in os.walk(path):
        sub_root = root.replace(path, '', 1).replace('\\', '/')
        # print(f'root: {root}, sub_root: {sub_root}')
        sub_root = f'{prefix}{sub_root}'
        for name in files:
            yield '/'.join([root, name]), '/'.join([sub_root, name])

def preview(fpath, heads=5):
    for chunk in pd.read_csv(fpath, chunksize=heads):
        return chunk

def read_pickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)

def write_pickle(path, obj):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp)

from sklearn.base import BaseEstimator, TransformerMixin
class BaseMapper(BaseEstimator, TransformerMixin):
    """Inherit scikit-learn class to custom our mapper object, for the purpose to transform all
      kind of feature to numeric to feed in model.

    """
    def init_check(self):
        return self

    def fit(self, y):
        return self.partial_fit(y)

    def partial_fit(self, y):
        return self

    def transform(self, y):
        return pd.Series(y).map(self.enc_).fillna(0).values

    def fit_transform(self, y, **fit_params):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        x = pd.Series(y).map(self.inv_enc_)
        return x.where(x.notnull(), None).values

    @staticmethod
    def unserialize(cls, fp):
        return cls().unserialize(fp)


class CatgMapper(BaseMapper):
    logger = logger('CatgMapper')

    """fit categorical feature"""
    def __init__(self, name=None, outlier=None,
                 is_multi=False, sep=None, default=None,
                 vocabs:list=None, vocabs_path:str=None):
        self.name = name
        self.default = default
        self.is_multi = is_multi
        self.outlier = outlier
        self.sep = sep
        self.vocabs = vocabs
        self.vocabs_path = vocabs_path

        self.classes_ = []
        self.freeze_ = False
        self.enc_ = None
        self.inv_enc_ = None

    @property
    def n_unique(self):
        return len(self.classes_)

    @property
    def emb_size(self):
        bins = np.array([8, 16, 32, 64])
        dim_map = dict(zip(range(len(bins)), bins))
        log_n_uniq = np.log(len(self.classes_))
        return dim_map[int(np.digitize(log_n_uniq, bins))]

    def init_check(self):
        if self.vocabs is not None and self.vocabs_path is not None:
            raise ValueError("[{}]: choose either vocab or vocab_path, can't specified both"
                             .format(self.name))
        if self.vocabs is not None:
            self.classes_ = list(pd.Series(self.vocabs).unique())
            self.gen_mapper()
            self.freeze_ = True

        if self.vocabs_path is not None:
            from . import flex
            blob = flex.io(self.vocabs_path)
            assert blob.exists(), "[{}]: can't find vocabs file [{}]"\
                                  .format(self.name, self.vocabs_path)
            self.logger.info('[{}] fetch vocab [{}] '.format(self.name, self.vocabs_path))
            with blob.as_reader('r') as f:
                clazz = pd.Series(f.stream.readlines()).map(str.strip).unique()
            self.classes_ = list(clazz[clazz != ''])
            self.gen_mapper()
            self.freeze_ = True
        return self

    def partial_fit(self, y):
        try:
            if isinstance(y, str):
                raise Exception()

            for _ in y: break
        except Exception as e:
            y = [y]

        # Remove outlier, prevent counting in classes
        y = pd.Series(list(y)).dropna()
        if self.outlier is not None:
            y = y[y != self.outlier]

        if self.is_multi:
            stack = set()
            self.split(y).map(stack.update)
            # Maybe outlier in the array of some row, e.g: ('', 'a', 'b', ...)
            if self.outlier is not None and self.outlier in stack:
                stack.remove(self.outlier)
            y = pd.Series(list(stack))

        if len(y):
            clazz = set(self.classes_)
            clazz.update(y)
            self.classes_ = sorted(clazz)
            self.gen_mapper()
        return self

    def gen_mapper(self):
        """According classes_ to generate enc_(encoder) and inv_enc_(decoder)

        :return:
        """
        idx = list(range(1, len(self.classes_) + 1))
        val = self.classes_

        self.enc_ = dict(zip(val, idx))
        self.inv_enc_ = dict(zip(idx, val))

    def split(self, y):
        if callable(self.sep):
            return y.map(self.sep, na_action='ignore')
        else:
            return y.str.split(f'\s*{re.escape(self.sep)}\s*')

    def transform(self, y):
        """Transform data(must fit first)

        :param y: string or string list
        :return: Tuple with integer elements
        """
        # is_multi = is_multi if is_multi is not None else self.is_multi

        y = pd.Series(y)
        if self.is_multi:
            y = self.split(y)
            lens = y.map(lambda ary: len(ary) if type(ary) in (list, tuple) else 1)
            indices = np.cumsum(lens)

            concat = []
            y.map(lambda ary: concat.extend(ary) if type(ary) in (list, tuple) else
            concat.append(None))

            y = pd.Series(concat).map(self.enc_, na_action='ignore') \
                  .fillna(0).astype(int).values
            return pd.Series(np.split(y, indices)[:-1]).values
        else:
            # y = do_default(y)
            return y.map(self.enc_, na_action='ignore').fillna(0).astype(int).values

    def serialize(self, fp):
        info = {
            'name': self.name,
            'classes_': self.classes_,
            'is_multi': self.is_multi,
            'sep': self.sep,
            'default': self.default,
            'freeze_': self.freeze_
        }
        yaml.dump(info, fp)
        return self

    def unserialize(self, fp):
        info = yaml.load(fp)
        self.is_multi = info['is_multi']
        self.sep = info['sep']
        self.name = info['name']
        self.classes_ = info['classes_']
        self.default = info['default']
        self.freeze_ = info['freeze_']
        self.gen_mapper()
        return self

class CountMapper(CatgMapper):
    """Mapping label to numeric order by occurrence frequency of label,
      the more frequency the lower index, vice versa
    """
    def __init__(self, lowest_freq_thres=0, **kw):
        super(CountMapper, self).__init__(**kw)
        self.lowest_freq_thres = lowest_freq_thres
        self.counter = Counter()
        self.freq_ = []

    def partial_fit(self, y):
        try:
            if isinstance(y, str):
                raise Exception()

            for _ in y: break
        except Exception as e:
            y = [y]

        # Remove outlier, prevent counting in classes
        y = pd.Series(list(y)).dropna()
        if self.outlier is not None:
            y = y[y != self.outlier]

        if self.is_multi:
            self.split(y).map(self.counter.update)
            # Maybe outlier in the array of some row, e.g: ('', 'a', 'b', ...)
            if self.outlier is not None and self.outlier in self.counter:
                self.counter.pop(self.outlier)
            # y.str.split(f'\s*{re.escape(self.sep)}\s*').map(self.counter.update)
        else:
            self.counter.update(y.values)

        if len(y):
            # Filter the low freq categories
            most_common = pd.DataFrame(columns=['clazz', 'freq'], data=np.array(self.counter.most_common()))
            most_common['freq'] = most_common.freq.astype(int)
            most_common = most_common[most_common.freq >= self.lowest_freq_thres]
            clazz, freq = most_common.clazz.tolist(), most_common.freq.tolist()
            self.counter = Counter(dict(zip(clazz, freq)))

            self.classes_ = clazz
            self.freq_ = freq
            self.gen_mapper()
        return self

class NumericMapper(BaseMapper):
    """Fit numerical to count common statistical values"""
    def __init__(self, name=None, default=None, scaler=None):
        self.default = default
        self.name = name
        self.scaler = scaler

    def init_check(self):
        if self.default is not None:
            try:
                float(self.default)
            except Exception as e:
                raise Exception('[{}]: default value must be numeric for NumericMapper'.format(self.name))
        return self

    @property
    def count_(self):
        return self.desc_['count']

    @property
    def mean_(self):
        return self.desc_['mean']

    @property
    def median_(self):
        return self.desc_['50%']

    @property
    def max_(self):
        return self.desc_['max']

    @property
    def min_(self):
        return self.desc_['min']

    def fit(self, y):
        try:
            if isinstance(y, str):
                raise Exception()

            y = list(y)
        except ValueError as e:
            y = list([y])

        y = pd.Series(y).dropna()
        self.desc_ = y.describe()
        # self.mean_ = self.desc_['mean']
        self.scaler.fit(y.values[:, np.newaxis])
        return self

    def transform(self, y):
        y = pd.Series(y).fillna(self.median_)[:, np.newaxis]
        return self.scaler.transform(y).reshape([-1])

    def inverse_transform(self, y):
        y = np.array(y)[:, np.newaxis]
        return self.scaler.inverse_transform(y).reshape([-1])
