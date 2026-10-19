import h5py
import numpy as np

f = h5py.File('sift10K_byte.h5', 'r')

dataset = f['dataset'][:].astype(np.float32)   # (9000, 128)
query   = f['query'][:].astype(np.float32)     # (1000, 128)
match   = f['match'][:].astype(np.int32)       # (1000, 10)

dataset.tofile('dataset.fbin')   # raw floats, flattened row-major
query.tofile('query.fbin')
match.tofile('match.ibin')

print(dataset.shape, query.shape, match.shape)