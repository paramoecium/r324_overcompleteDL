from datetime import datetime
import os
import errno
import cPickle

class Timer(object):
    def __init__(self, name=None, verbose=2):
        self.name = name
        self.verbose = verbose;

    def __enter__(self):
        if self.name and self.verbose >= 1:
            print '...', self.name
        self.start = datetime.now()
        return self

    def __exit__(self, type, value, traceback):
        if self.verbose >= 2:
            if self.name:
                print '...', self.name, "done in", datetime.now() - self.start
            else:
                print '... done in', datetime.now() - self.start

    def now_time(self):
        return datetime.now() - self.start

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def save_pkl(obj, filename):
    with open(filename, "wb") as fp, Timer("Pickling to %s" % (filename)):
        cPickle.dump(obj, fp, protocol = cPickle.HIGHEST_PROTOCOL);

def load_pkl(filename):
    with open(filename, "rb") as fp, Timer("Unpickling from %s" % (filename)):
        obj = cPickle.load(fp);
    return obj




