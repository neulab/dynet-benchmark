from __future__ import absolute_import

import cPickle


def serialize_to_file(obj, path, protocol=cPickle.HIGHEST_PROTOCOL):
    f = open(path, 'wb')
    cPickle.dump(obj, f, protocol=protocol)
    f.close()


def deserialize_from_file(path):
    f = open(path, 'rb')
    obj = cPickle.load(f)
    f.close()
    return obj