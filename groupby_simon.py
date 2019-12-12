import numpy as np
import pandas as pd
import numpy_indexed as npi
import sys
from time import time

class Groupby:
    def __init__(self, slices):
        self.slices = slices

    def groupby(self, printing=True):
        t = time()
        print('   ......estimating capacity: ', end='')
        minimums = np.min(self.slices, axis=0)
        maximums = np.max(self.slices, axis=0)
        ranges = maximums - minimums + 1
        capacity = int(ranges[0])*int(ranges[1])*int(ranges[2])
        if capacity <= 4294967295: nptype = np.uint32
        else: nptype = np.uint64
        ranges = ranges.astype(nptype)
        print(round(time() - t, 3), '['+str(round(100 * capacity/4294967295, 3))+'% of memory for np.uint32]')
        t = time()
        print('   ......slices: [memory =', round(self.slices.nbytes/1048576,3),'megabytes]', end='\n')
        print('   ......transforming: ', end='')
        transform = self.slices - minimums
        print(round(time()-t,3),'[memory =', round(transform.nbytes/1048576,3),'megabytes]')
        t = time()
        print('   ......creating hasharray: ', end='')
        hasharray = transform[:,0].astype(nptype)*ranges[1]*ranges[2] + transform[:,1].astype(nptype)*ranges[2] + transform[:,2].astype(nptype)
        del transform
        print(round(time() - t, 3), '[memory =', round(hasharray.nbytes / 1048576, 3), 'megabytes]; transform deleted')
        t = time()
        print('   ......grouping zipped column: ', end='')
        labels = self.label_of_pandas(hasharray)
        del hasharray
        print(round(time() - t, 3), '[memory =', round(sum(map(sys.getsizeof, labels.keys())) / 1048576, 3),
              'megabytes for keys and', round(sum(map(sys.getsizeof, labels.values())) / 1048576, 3),
              'for values]; hasharray deleted')
        t = time();
        print('   ......converting back to dict: ', end='')
        result = {(self.slices[n[0]][0], self.slices[n[0]][1], self.slices[n[0]][2]): n for n in labels.values()}
        # consider replacing labels.values() with labels when Groupby applies smth different than label_of_pandas method
        print(round(time() - t, 3), '[memory =', round(sum(map(sys.getsizeof, result.keys())) / 1048576, 3),
              'megabytes for keys and', round(sum([n.nbytes for n in result.values()]) / 1048576, 3),
              'for values]')
        return result

    @staticmethod
    def label_of_pandas(hasharray):
        #consumes memory, returns dict, takes 1.8s
        return pd.DataFrame(hasharray).groupby([0]).indices #returns dict
        # this describes memory issue: https://stackoverflow.com/questions/50051210/avoiding-memory-issues-for-groupby-on-large-pandas-dataframe

    @staticmethod
    def label_of_npi(hasharray):
        # may consume memory, returns components as ndarrays, takes 3.1s
        return npi.group_by(hasharray).split(hasharray)

    @staticmethod
    def label_of_simon(hasharray):
        # no extra memory, returns components as ndarrays, takes 2.65s
        hasharray.sort()
        starting_indices = np.cumsum(np.unique(hasharray, return_counts=True)[1])[:-1]
        return np.split(hasharray, starting_indices)
