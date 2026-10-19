import h5py
import numpy as np

with h5py.File("sift100K_byte.h5", "r") as f:
    dataset = f["dataset"][:].astype(np.float32)
    query = f["query"][:].astype(np.float32)
    match = f["match"][:].astype(np.int32)

dataset.tofile("dataset.fbin")   # (99000, 128) float32
query.tofile("query.fbin")       # (1000, 128) float32
match.tofile("match.ibin")       # (1000, 1) int32

print("Dataset:", dataset.shape, dataset.dtype)
print("Query:", query.shape, query.dtype)
print("Match:", match.shape, match.dtype)