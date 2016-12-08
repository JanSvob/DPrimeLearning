from __future__ import print_function

import h5py
import cPickle
import zlib
import numpy as np


class Snapshotter(object):
    def __init__(self, fileName, readOnly=False):
        '''
        Key-Value pair snapshot utility for numpy arrays.
        Inspired by Jonathan Masci's snapshotter (learning_func_maps/learnfm/snapshotter.py).
        '''
        self.fileName = fileName
        print('Opening DB {}'.format(fileName))
        if readOnly:
            self.database = h5py.File(fileName, 'r')
        else:
            self.database = h5py.File(fileName)

    def store(self, key, value):
        print('Storing {}'.format(key))
        compressedVal = np.void(zlib.compress(cPickle.dumps(value,
                                                            protocol=cPickle.HIGHEST_PROTOCOL)))

        if key in self.database:
            print('Overwriting value for key '.format(key))
            del self.database[key]

        self.database[key] = [compressedVal]

    def load(self, key):
        return cPickle.loads(zlib.decompress(self.database[key][:][0].tostring()))

    def close(self):
        self.database.close()
