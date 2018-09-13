import yaml, codecs, logging, os, pandas as pd, pickle, tensorflow as tf
from logging import config
from datetime import datetime

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
            print('init logger instance ...')

            log_conf_path = '{}/logging.yaml'.format(os.path.dirname(__file__))
            with codecs.open(log_conf_path, 'r', 'utf-8') as r:
                logging.config.dictConfig(yaml.load(r))
            Logging.instance = logging

        return Logging.instance.getLogger(name)


def logger(name):
    """Short path of Logging.logger"""
    return Logging.get_logger(name)

def get_instance(cls, *args, **kwargs):
    """Simulate the singleton pattern"""
    if cls.instance is None:
        cls.instance = cls(*args, **kwargs)
    return cls.instance

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

# def find_latest_expdir(conf):
#     """Found latest export dir"""
#     export_dir = '{}/export/{}'.format(conf.model_dir, conf.export_name)
#     return '{}/{}'.format(export_dir, sorted(os.listdir(export_dir))[-1])

def timestamp():
    return int(datetime.now().timestamp())

def deep_walk(path, prefix:str=None):
    """Deep walk directory, return all object in the directory with specified prefix(context) string."""
    path = os.path.abspath(path)
    if not prefix:
        prefix = ''
    for root, dirs, files in os.walk(path):
        sub_root = root.replace(path, '', 1).replace('\\', '/')

        sub_root = '{}{}'.format(prefix, sub_root)
        for name in files:
            yield '/'.join([root, name]), '/'.join([sub_root, name])

def preview(fpath, heads=5):
    for chunk in pd.read_csv(fpath, chunksize=heads):
        return chunk

def read_pickle(path):
    # with open(path, 'rb') as fp:
    with tf.gfile.Open(path, 'rb') as fp:
        return pickle.load(fp)

def write_pickle(path, obj):
    # with open(path, 'wb') as fp:
    with tf.gfile.Open(path, 'wb') as fp:
        pickle.dump(obj, fp)

def heatmap(data, *cols, xtick=None, ytick=None, annot=True, fmt='.2f', figsize=None, label='sales'):
    import seaborn as sns
    from matplotlib import pyplot as plt

    figsize = figsize or (16, 4)
    f, axs = plt.subplots(1, 2, figsize=figsize)

    def draw(chop, axis):
        pivot_params = list(cols) + [label]
        g = chop.groupby(list(cols))[label]
        mean_ = g.mean().reset_index().pivot(*pivot_params)
        count_ = g.size().reset_index().pivot(*pivot_params)
        if xtick is not None or ytick is not None:
            mean_ = mean_.reindex(index=ytick, columns=xtick)
            count_ = count_.reindex(index=ytick, columns=xtick)

        sns.heatmap(mean_.fillna(0), annot=annot, ax=axis[0], fmt=fmt)
        sns.heatmap(count_.fillna(0), annot=annot, ax=axis[1], fmt=fmt)
        axis[0].set_title('mean')
        axis[1].set_title('count')

    draw(data, axs)
    plt.show()

