import sys
import h5py
import numpy as np
from pathlib import Path

folder = Path(__file__).resolve().parent
h5_path = folder / "sift_1m_old_dist.h5"
train_key = "train_vectors"
test_key = "test_vectors"
batch = 50
check_queries = 200

with h5py.File(h5_path, "r") as f:
    names = []
    f.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
    print("datasets in file:")
    for name in names:
        print(" ", name, f[name].shape, f[name].dtype)
    if train_key not in names or test_key not in names:
        print("Set train_key and test_key at the top to match the names above.")
        sys.exit(1)
    train = np.ascontiguousarray(f[train_key][:], dtype=np.float32)
    test = np.ascontiguousarray(f[test_key][:], dtype=np.float32)
    candidates = {}
    for name in names:
        low = name.lower()
        if "neighbor" in low or "match" in low:
            candidates[name] = f[name][:].astype(np.int64)
    if "distances_0" in names:
        print("distances_0 first row, first 3:", f["distances_0"][0, :3])

n, dim = train.shape
nq = test.shape[0]
print("train:", train.shape, "test:", test.shape)

train.tofile(folder / "dataset.fbin")
test.tofile(folder / "query.fbin")

train_norm = np.einsum("ij,ij->i", train, train)
top1 = np.empty(nq, dtype=np.int32)

for start in range(0, nq, batch):
    q = test[start:start + batch]
    q_norm = np.einsum("ij,ij->i", q, q)
    approx = q_norm[:, None] + train_norm[None, :] - 2.0 * (q @ train.T)
    cand = np.argpartition(approx, 20, axis=1)[:, :20]
    for r in range(q.shape[0]):
        ids = cand[r]
        diff = train[ids].astype(np.float64) - q[r].astype(np.float64)
        exact = (diff * diff).sum(axis=1)
        top1[start + r] = ids[np.argmin(exact)]
    if (start // batch) % 20 == 0:
        print("brute force:", start + q.shape[0], "/", nq)

check = min(check_queries, nq)
for key, arr in candidates.items():
    if arr.ndim != 2 or arr.shape[0] != nq:
        print(key, arr.shape, "skipped (not one row per query)")
        continue
    col0 = float((arr[:check, 0] == top1[:check]).mean())
    in_row = float(np.mean([top1[i] in arr[i] for i in range(check)]))
    print(key, arr.shape, "| first column == brute top-1:", col0, "| brute top-1 appears in row:", in_row)

out_path = folder / "match_top1.ibin"
top1.tofile(out_path)
print("wrote", out_path.name, out_path.stat().st_size, "bytes for", nq, "queries")
print("wrote dataset.fbin and query.fbin in", folder)