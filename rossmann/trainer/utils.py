import yaml, codecs, logging, os, re
from logging import config
from datetime import datetime

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