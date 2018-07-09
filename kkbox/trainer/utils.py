import yaml, codecs, logging, os, pandas as pd, numpy as np, re, seaborn as sns

from logging import config
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from . import flex

class Logging(object):
    instance = None

    @staticmethod
    def get_logger(name):
        if Logging.instance is None:
            print(f'init logger instance ...')

            log_conf_path = f'{os.path.dirname(__file__)}/logging.yaml'
            with codecs.open(log_conf_path, 'r', 'utf-8') as r:
                logging.config.dictConfig(yaml.load(r))
            Logging.instance = logging

        return Logging.instance.getLogger(name)

# short path of Logging.logger
def logger(name):
    return Logging.get_logger(name)

def cmd(commands):
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

def find_latest_expdir(conf):
    # Found latest export dir
    export_dir = f'{conf.model_dir}/export/{conf.export_name}'
    return f'{export_dir}/{sorted(os.listdir(export_dir))[-1]}'

def timestamp():
    return int(datetime.now().timestamp())

def deep_walk(path, prefix:str=None):
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


from sklearn.base import BaseEstimator, TransformerMixin
class BaseMapper(BaseEstimator, TransformerMixin):
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
    def __init__(self, name=None, allow_null=True,
                 is_multi=False, sep=None, default=None,
                 vocabs:list=None, vocabs_path:str=None):
        self.name = name
        self.allow_null = allow_null
        self.default = default
        self.is_multi = is_multi
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

        y = pd.Series(list(set(y)))
        if not self.allow_null:
            assert not y.hasnans, '[{}]: null value detected'.format(self.name)

        y = y.dropna()
        if self.is_multi:
            stack = set()
            self.separate(y).map(stack.update)
            y = stack

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

    def separate(self, y:pd.Series):
        if callable(self.sep):
            return y.map(self.sep, na_action='ignore')
        else:
            return y.str.split(f'\s*{re.escape(self.sep)}\s*')

    def transform(self, y):
        """transform data(must fit first)

        :param y: string or string list
        :return: Tuple with integer elements
        """
        y = pd.Series(y)
        if not self.allow_null:
            assert not y.hasnans, '[{}]: null value detected'.format(self.name)

        # Handle outlier(regard None as outlier)
        # if default value not in self.classes_: mapping to index which the default value mapping to
        # else: mapping to 0
        # def do_default(data):
        #     if self.default is not None:
        #         data.loc[~data.isin(self.enc_)] = self.default
        #     return data

        if self.is_multi:
            x = self.separate(y)
            lens = np.cumsum(x.map(len, na_action='ignore').fillna(1).astype(int).values)
            na_conds = x.isna()
            na_indices = na_conds.nonzero()[0]
            na_len = sum(na_conds)
            x.loc[na_conds] = [[None]] * na_len
            concat = pd.Series(np.concatenate(x.values))
            # concat = do_default(concat)
            x = concat.map(self.enc_, na_action='ignore').fillna(0).astype(int).values
            x = pd.Series(np.split(x, lens)[:-1]) \
                    .map(sorted, na_action='ignore') \
                    .map(tuple, na_action='ignore')
            # Put empty tuple to represent missing value in vector space,
            # pandas.Series.fillna not work at array type, e.g: series.fillna(tuple()) will fail
            for idx in na_indices:
                x.at[idx] = tuple()
            return x.values
        else:
            # y = do_default(y)
            return y.map(self.enc_, na_action='ignore').fillna(0).astype(int).values


    def serialize(self, fp):
        info = {
            'name': self.name,
            'classes_': self.classes_,
            'is_multi': self.is_multi,
            'sep': self.sep,
            'allow_null': self.allow_null,
            'default': self.default,
            'freeze_': self.freeze_
        }
        yaml.dump(info, fp)
        return self

    def unserialize(self, fp):
        info = yaml.load(fp)
        self.is_multi = info['is_multi']
        self.allow_null = info['allow_null']
        self.sep = info['sep']
        self.name = info['name']
        self.classes_ = info['classes_']
        self.default = info['default']
        self.freeze_ = info['freeze_']
        self.gen_mapper()
        return self

class CounterEncoder(CatgMapper):
    """依照出現頻率進行編碼, 頻率由高到低的index = 0, 1, 2, 3 ..., 以此類推
     keep index = 0 for outlier.
    """
    def __init__(self, lowest_freq_thres=0, **kw):
        super().__init__(**kw)
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

        y = pd.Series(list(y))
        if not self.allow_null:
            assert not y.hasnans, '[{}]: null value detected'.format(self.name)

        y = y.dropna()
        if self.is_multi:
            self.separate(y).map(self.counter.update)
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


class MultiCatgToString(CounterEncoder):
    def __init__(self, **kw):
        super().__init__(**kw)

    def transform(self, y):
        """Transform data(must fit first)

        :param y: string or string list
        :return: string splited by comma sign
        """
        y = pd.Series(y)
        if not self.allow_null:
            assert not y.hasnans, '[{}]: null value detected'.format(self.name)

        assert self.is_multi

        if self.is_multi:
            # For performance issue, concat all sequence in a batch and do once mapping,
            # I've encountered super worst performance in processing once per row
            x = self.separate(y)
            lens = np.cumsum(x.map(len, na_action='ignore').fillna(1).astype(int).values)
            na_conds = x.isna()
            # na_indices = na_conds.nonzero()[0]
            na_len = sum(na_conds)
            x.loc[na_conds] = [[None]] * na_len
            concat = pd.Series(np.concatenate(x.values))
            # concat = do_default(concat)
            x = concat.map(self.enc_).map(lambda e: str(int(e)) if pd.notna(e) else '')
            x = pd.Series(np.split(x.values, lens)[:-1]).map(','.join)
            return x.values


class NumericMapper(BaseMapper):
    """fit numerical feature"""
    def __init__(self, name=None, default=None, scaler=None):
        self.default = default
        self.name = name
        self.scaler = scaler
        self.max_ = None
        self.min_ = None
        self.cumsum_ = 0
        self.n_total_ = 0

    def init_check(self):
        if self.default is not None:
            try:
                float(self.default)
            except Exception as e:
                raise Exception('[{}]: default value must be numeric for NumericMapper'.format(self.name))
        return self

    @property
    def mean(self):
        return self.cumsum_ / self.n_total_ if self.n_total_ > 0 else None

    def partial_fit(self, y):
        try:
            if isinstance(y, str):
                raise Exception()

            y = list(y)
        except ValueError as e:
            y = list([y])

        assert not isinstance(y[0], str), 'NumericMapper requires numeric data, got string!'

        y = pd.Series(y).dropna().values
        if len(y):
            self.cumsum_ += sum(y)
            self.n_total_ += len(y)
            self.scaler.partial_fit([[min(y)], [max(y)]])
            self.max_ = self.scaler.data_max_[0]
            self.min_ = self.scaler.data_min_[0]
        return self

    def transform(self, y):
        y = pd.Series(y).fillna(self.default if self.default is not None else self.mean)[:, np.newaxis]
        return self.scaler.transform(y).reshape([-1])

    def inverse_transform(self, y):
        y = np.array(y)[:, np.newaxis]
        return self.scaler.inverse_transform(y).reshape([-1])

    def _serialize(self):
        ret = {}
        ret['name'] = self.name
        for num_attr in ('max_', 'min_', 'cumsum_', 'n_total_', 'default'):
            val = getattr(self, num_attr)
            ret[num_attr] = float(val) if val is not None else None
        return ret

    def _unserialize(self, info):
        self.scaler = MinMaxScaler()
        self.max_ = info['max_']
        self.min_ = info['min_']
        self.scaler.partial_fit([[self.max_], [self.min_]])
        self.cumsum_ = info['cumsum_']
        self.n_total_ = info['n_total_']
        self.name = info['name']
        self.default = info['default']
        return self

    def serialize(self, fp):
        yaml.dump(self._serialize(), fp)
        return self

    def unserialize(self, fp):
        info = yaml.load(fp)
        return self._unserialize(info)

class DatetimeMapper(NumericMapper):
    def __init__(self, name=None, dt_fmt=None, default=None):
        super().__init__(name=name, default=default)
        self.dt_fmt = dt_fmt
        self.default_ = None
        self.default = default

    def init_check(self):
        if self.default is not None:
            default = self.default
            assert isinstance(default, str), 'datetime default value must be string!'
            self.default = default.strip()
            try:
                self.default_ = datetime.strptime(self.default, self.dt_fmt).timestamp()
            except Exception as e:
                raise ValueError('parse default datetime [{}] failed!\n\n{}'.format(self.default, e))
        return self

    def partial_fit(self, y):
        try:
            if isinstance(y, str):
                raise Exception()

            y = list(y)
        except ValueError as e:
            y = list([y])

        assert isinstance(y[0], str), 'DatetimeMapper requires string data for parsing, got {}!'.format(type(y[0]))

        y = pd.Series(y).dropna().map(lambda e: datetime.strptime(e, self.dt_fmt).timestamp()).values
        if len(y):
            self.cumsum_ += sum(y)
            self.n_total_ += len(y)
            self.scaler.partial_fit([[min(y)], [max(y)]])
            self.max_ = self.scaler.data_max_[0]
            self.min_ = self.scaler.data_min_[0]
        return self

    def transform(self, y):
        y = pd.Series(y).map(lambda e: datetime.strptime(e, self.dt_fmt).timestamp() if e is not None else None)\
                        .fillna(self.default_ if self.default_ is not None else self.mean)[:, np.newaxis]
        return self.scaler.transform(y).reshape([-1])

    def _serialize(self):
        info = super()._serialize()
        info['dt_fmt'] = self.dt_fmt
        info['default_'] = self.default_
        return info

    def _unserialize(self, info):
        super()._unserialize(info)
        self.dt_fmt = info['dt_fmt']
        self.default_ = info['default_']
        return self