import numpy as np
from time import time
t = time()
cubes = np.load('cubes.npz')['array']
print(round(time() - t, 3))