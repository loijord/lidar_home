import numpy as np
import pandas as pd
from collections import defaultdict
from time import time

def groupby1(cubes):
    # proposal of mathfux
    return pd.DataFrame(cubes).groupby([0,1,2]).indices
    # takes 7.5sec

def groupby2(cubes):
    #thanks @abc
    result = defaultdict(list)
    for idx, elem in enumerate(cubes):
        result[elem.tobytes()].append(idx)
    return result
    # takes 19.5sec

def groupby2_1(cubes):
    #thanks @abc
    result = defaultdict(list)
    for idx, elem in enumerate(cubes):
        result[tuple(elem)].append(idx)
    return result
    # takes 20.5sec

def groupby2_2(cubes):
    #thanks @abc
    result = defaultdict(list)
    for idx, elem in enumerate(cubes):
        result[elem[0], elem[1], elem[2]].append(idx)
    return result
    # takes 19.5sec

def groupby3(cubes):
    #thanks @MykolaZotko
    u, idx = np.unique(cubes, return_inverse=True, axis=0)
    m = idx == np.arange(len(u))[:, None]
    result = {tuple(i): np.flatnonzero(j) for i, j in zip(u, m)}
    return result

print('reading file of cubes:', end=' ')
t = time()
cubes = np.load('cubes.npz')['array']
print(round(time()- t, 3))

def test(function):
    print('duration of {}:'.format(function.__name__), end=' ')
    times = []
    for i in range(10):
        t = time()
        result = function(cubes)
        duration = round(time()- t, 3)
        print(duration, end=' ')
        times.append(duration)
    print('\naverage time of 10 tests for {}: {}'.format(function.__name__, round(sum(times) / len(times), 3)))
    print('-'*100)

test(groupby1)
test(groupby2)
test(groupby2_1)
test(groupby2_2)
test(groupby3)
