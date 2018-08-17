import codecs, os, re, shutil
from io import StringIO, BytesIO

from . import utils

def rm_quiet(fpath):
    if fpath is not None and os.path.exists(fpath) and os.path.isfile(fpath):
        os.remove(fpath)

def io(path):
    """Portable for local io or GCS io, the purpose is to access object with the same api,
      although tensorflow tf.gfile.Gile could do this, but not on windows, so we still implement this module
    """
    m = re.search('(?i)^gs://', path)
    return LocalIO(path) if m is None else GCSIO(path)


class FlexIO(object):
    """Abstract class for IO interface to implement"""
    def __init__(self, path):
        m = re.search('(?i)^gs://', path)
        self.is_local = True if m is None else False
        self.path = path
        self.mode = None
        self.placeholder = None
        self.stream = None

    def exists(self):
        raise NotImplementedError()

    def rm(self):
        raise NotImplementedError()

    def mkdirs(self):
        raise NotImplementedError()

    def list(self):
        raise NotImplementedError()

    def read(self, mode='rb', encoding=None):
        raise NotImplementedError()

    def write(self, data, mode='wb', encoding=None):
        raise NotImplementedError()

    def as_reader(self, mode='rb', encoding=None):
        self._file_handler('read', mode=mode, encoding=encoding)
        return self

    def as_writer(self, mode='wb', encoding=None):
        self._file_handler('write', mode=mode, encoding=encoding)
        return self

    def _file_handler(self, tpe, mode, encoding):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LocalIO(FlexIO):
    """Local IO object"""
    logger = utils.logger('LocalIO')

    def __init__(self, path):
        self.path = path
        self.mode = None
        self.stream = None

    def exists(self):
        return os.path.exists(self.path)

    def rm(self):
        _ = rm_quiet(self.path) if os.path.isfile(self.path) else shutil.rmtree(self.path, ignore_errors=True)
        return self

    def mkdirs(self):
        os.makedirs(self.path, exist_ok=True)
        return self

    def list(self):
        if os.path.isdir(self.path):
            return [os.path.join(root, f).replace('\\', '/')
                    for root, ds, fs in os.walk(self.path) for f in fs]
        return []

    def read(self, mode='rb', encoding=None):
        return self.as_reader(mode=mode, encoding=encoding).stream.read()

    def write(self, data, mode='wb', encoding=None):
        self.as_writer(mode=mode, encoding=encoding).stream.write(data)
        return self

    def _file_handler(self, tpe, mode, encoding):
        if self.stream is None:
            self.mode = mode
            if not self.exists():
                dirpath = os.path.dirname(os.path.abspath(self.path))
                os.makedirs(dirpath, exist_ok=True)
            self.stream = codecs.open(self.path, mode=mode, encoding=encoding)
            self.stream.f = self
        return self.stream

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
            self.mode = None
        return self


class GCSIO(FlexIO):
    """IO object for access GCS object"""
    logger = utils.logger('GCSIO')

    def __init__(self, path):
        self.path = path
        self.mode = None
        self.placeholder = self.gcs_blob(path)
        self.stream = None

    def exists(self):
        return self.placeholder.exists()

    def rm(self):
        _ = [self.gcs_rm_quiet(e) for e in io(self.path).list()]
        return self

    def mkdirs(self):
        self.logger.info('mkdirs not support by GCS!')
        return self

    def list(self):
        return list(map(lambda blob: 'gs://' + os.path.join(blob.bucket.name, blob.name).replace('\\', '/'),
                        self.gcs_list(self.path)))

    def read(self, mode='rb', encoding=None):
        return self.as_reader(mode=mode, encoding=encoding).stream.getvalue()

    def write(self, data, mode='wb', encoding=None):
        self.as_writer(mode=mode, encoding=encoding).stream.write(data)
        return self

    def _file_handler(self, tpe, mode, encoding):
        if self.stream is None:
            self.mode = mode
            is_binary = 'b' in mode
            self.stream = BytesIO() if is_binary else StringIO()
            if tpe == 'read':
                # GCS only accept bytes download ...
                if is_binary:
                    self.placeholder.download_to_file(self.stream)
                else:
                    stream_ = BytesIO()
                    self.placeholder.download_to_file(stream_)
                    self.stream.write(stream_.getvalue().decode())
                self.stream.seek(0)
            self.stream.f = self
        return self.stream

    def close(self):
        if self.stream is not None:
            if 'w' in self.mode:
                self.logger.info('upload to [{}]'.format(self.path))
                self.stream.seek(0)
                self.placeholder.upload_from_string(self.stream.getvalue())
            self.stream.close()
            self.stream = None
            self.mode = None
        return self

    def get_bucket(self, bucket_name):
        from google.cloud import storage
        return storage.Client().get_bucket(bucket_name)

    def parse_gsc_uri(self, s):
        s = re.sub('(?i)^gs://', '', s)
        ary = s.split('/')
        # bucket, prefix
        return ary[0], '/'.join(ary[1:])

    def gcs_blob(self, s):
        bucket, prefix = self.parse_gsc_uri(s)
        bucket = self.get_bucket(bucket)
        return bucket.blob(prefix)

    def gcs_clear(self, dir_):
        dir_ = self.gcs_blob(dir_)
        for blob in dir_.bucket.list_blobs(prefix=dir_.name): blob.delete()

    def gcs_rm_quiet(self, fpath):
        if fpath is not None:
            blob = self.gcs_blob(fpath)
            _ = blob.delete() if blob.exists() else None

    def gcs_list(self, s):
        blob = self.gcs_blob(s)
        return list(blob.bucket.list_blobs(prefix=blob.name))