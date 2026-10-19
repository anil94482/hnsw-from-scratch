# HNSW from scratch (C++17)

A from-scratch implementation of **Hierarchical Navigable Small World (HNSW)** approximate nearest-neighbour search in C++17, benchmarked against exact brute force and a single-layer ("flat") graph on SIFT descriptors (128-d) at three scales: **9K, 99K and 1M vectors**.

Only the C++ standard library is used (no FAISS, hnswlib, Eigen, etc.). The code is single-threaded and scalar.

> Status: baseline complete (brute force, flat graph, layered HNSW with closest-M neighbour selection). Planned: the paper's neighbour-selection heuristic, filtered search.

## Headline results

1M vectors, M=16, efConstruction=100, Recall@1 (tie-aware), average over 1000 queries. Brute force: 235.1 ms/query.

| ef | Recall@1 | ms/query | speedup vs brute force |
|---|---|---|---|
| 10 | 0.672 | 0.178 | 1323x |
| 50 | 0.897 | 0.468 | 503x |
| 200 | 0.979 | 1.352 | 174x |
| 1000 | 0.997 | 5.111 | 46x |

Speedups are against a **single-threaded, unoptimized (-O0), unvectorized** brute-force baseline built the same way as the graph code. They are not a comparison with optimized libraries.

Findings:

1. **Brute force scales linearly; HNSW barely grows.** Brute-force cost per stored vector is about 233 ns (9K), 241 ns (99K) and 235 ns (1M). Layered HNSW query time grows about 2x while the data grows 111x (about 2.0 to 2.1x at ef=10, 50 and 200). Note that at a fixed ef, recall falls as the data grows.
2. **Layers help at low ef, and the benefit grows with dataset size.** Layered vs flat query time at ef=10: about -4% (9K), -7.6% (99K), -13% to -16% (1M). Recall at ef=10 changes by -0.6, +0.4 and +3.4 points (layered minus flat). The benefit is gone by ef=200 (within noise). Why the upper layers help (a better starting point for the search) is a hypothesis; it has not been tested.
3. **Recall keeps rising with ef at 1M; there is no plateau up to ef=1000** (tie-aware recall 0.980 at ef=200, 0.990 at ef=500, 0.997 at ef=1000), but the cost is steep.
4. **The 1M brute-force "recall 0.994" is a counting artifact.** Six of 1000 queries had two bit-identical vectors at exactly the same distance, and brute force returned one while the ground-truth file listed the other (index gaps of 78,816 for five pairs and 64,278 for one). Counting a result as correct when its distance equals the ground-truth distance ("tie-aware" recall) gives 1.0. The dataset therefore contains exact duplicate vectors.
5. **Build time:** the layered build was about 20% faster than flat at 1M (780 s vs 990 s here; 903 s vs 1124 s in an earlier session). There is no reliable difference at 9K or 99K.

## Full results

Recall is Recall@1. Layered vs flat = change in ms/query (negative = layered faster).

### 9K (9,000 vectors)

Brute force: 2.097 ms/query (range 2.058 to 2.131 over 5 runs), recall 1.0. Build: flat 3.41 s, layered 3.50 s.
Method: flat and layered graphs in one process, 10 interleaved sweeps, medians.

| ef | recall flat / layered | flat ms | layered ms | layered vs flat | speedup flat / layered |
|---|---|---|---|---|---|
| 10 | 0.977 / 0.971 | 0.0882 | 0.0842 | -3.7% | 23.8x / 24.9x |
| 20 | 0.994 / 0.994 | 0.1324 | 0.1256 | -4.8% | 15.8x / 16.7x |
| 50 | 1.000 / 1.000 | 0.2433 | 0.2343 | -3.9% | 8.6x / 9.0x |
| 100 | 1.000 / 1.000 | 0.3930 | 0.3883 | -1.5% | 5.3x / 5.4x |
| 200 | 1.000 / 1.000 | 0.6523 | 0.6495 | -0.4% | 3.2x / 3.2x |

### 99K (99,000 vectors)

Brute force: 23.9 ms/query (median of the fast runs; all 10 runs ranged 23.1 to 30.4), recall 1.0. Build: flat median 61.7 s, layered about 64 s.
Method: separate processes, 5 runs per method, means. Flat and layered were measured in different sessions, so the "layered vs flat" column is only good to about +/-3 to 5 points.

| ef | recall flat / layered | flat ms | layered ms | layered vs flat | speedup flat / layered |
|---|---|---|---|---|---|
| 10 | 0.906 / 0.910 | 0.1460 | 0.1350 | -7.6% | 164x / 177x |
| 20 | 0.959 / 0.963 | 0.2220 | 0.2067 | -6.9% | 108x / 116x |
| 50 | 0.990 / 0.992 | 0.4047 | 0.3870 | -4.4% | 59x / 62x |
| 100 | 0.996 / 0.996 | 0.6831 | 0.6643 | -2.8% | 35x / 36x |
| 200 | 0.999 / 0.999 | 1.1558 | 1.1559 | 0.0% | 21x / 21x |

### 1M (1,000,000 vectors)

Brute force: 235.1 ms/query (median of the fast runs; all 21 runs ranged 234 to 347), Recall@1 0.994 exact-id and 1.0 tie-aware. Build: flat 989.8 s, layered 779.5 s.
Method: one process per graph, 5 sweeps, means. Machine speed differed by about 3% between the flat and layered processes (brute-force probe readings 235.5 vs 228.3 ms/query). After adjusting for that, the layered vs flat gaps are roughly -13% / -11% / -7% / -5% / -1% / +1% / +2%, so only ef up to about 100 should be treated as a real gap.

| ef | recall flat / layered | flat ms | layered ms | layered vs flat (raw) | speedup flat / layered |
|---|---|---|---|---|---|
| 10 | 0.638 / 0.672 | 0.2106 | 0.1778 | -15.6% | 1116x / 1323x |
| 20 | 0.777 / 0.792 | 0.2977 | 0.2576 | -13.5% | 790x / 913x |
| 50 | 0.899 / 0.897 | 0.5193 | 0.4676 | -10.0% | 453x / 503x |
| 100 | 0.963 / 0.961 | 0.8401 | 0.7733 | -8.0% | 280x / 304x |
| 200 | 0.980 / 0.979 | 1.4136 | 1.3520 | -4.4% | 166x / 174x |
| 500 | 0.990 / 0.989 | 2.9261 | 2.8706 | -1.9% | 80x / 82x |
| 1000 | 0.997 / 0.997 | 5.1635 | 5.1111 | -1.0% | 45.5x / 46x |

Exact-id recall at 1M is lower by 0.4 to 0.6 points in every row (for example layered at ef=200: 0.973 exact-id vs 0.979 tie-aware), because of the duplicate vectors described above.

## What is implemented

- **Distance:** squared Euclidean distance, plain scalar loop.
- **Level assignment:** level = floor(-ln(U) * mL) with mL = 1/ln(M), seeded Mersenne Twister (seed 42).
- **Insertion:** greedy descent (ef=1) from the top layer to one above the node's level; then, at each layer from the node's level down to 0, a beam search with efConstruction=100, links to the M closest candidates (bidirectional), and shrinking of an overfull neighbour list to Mmax by keeping the closest.
- **Parameters:** M=16; Mmax=32 at layer 0 and 16 above; efConstruction=100.
- **Search:** greedy descent with ef=1 through the upper layers, then a beam search with the given ef at layer 0. The search uses a candidate min-heap and a result max-heap, and stops when the closest candidate is worse than the furthest result.
- **Flat baseline:** the same insertion procedure and parameters on a single layer; every search starts at node 0.
- **Neighbour selection:** closest-M only. The paper's diversity heuristic is not implemented yet.
- **Resulting graphs:**

| size | nodes per layer | layer-0 average degree |
|---|---|---|
| 9K | 9000 / 532 / 38 / 1 | 25.2 |
| 99K | 99000 / 6253 / 387 / 23 | 25.0 |
| 1M | 1,000,000 / 62,478 / 3,957 / 227 / 11 / 1 | 24.6 |

## Data and metrics

- **Data sources:**
  - **9K:** `sift10K_byte.h5` from the FLANN repository (https://github.com/flann-lib/flann/blob/master/datasets/sift10K_byte.h5). Contains 10,000 vectors; 1,000 are held out as query vectors, leaving 9,000 base vectors.
  - **99K:** `sift100K_byte.h5` from the FLANN repository (https://github.com/flann-lib/flann/blob/master/datasets/sift100K_byte.h5). Contains 100,000 vectors; 1,000 are held out as query vectors, leaving 99,000 base vectors.
  - **1M:** SIFT1M (https://www.tensorflow.org/datasets/catalog/sift1m).
  The `.h5` files were converted to raw float32 binary (`dataset.fbin`, `query.fbin`) by a small conversion script.
- 1,000 queries evaluated per size. For 1M, `query.fbin` contains all 10,000 SIFT1M queries, but only the first 1,000 are evaluated by the benchmarks.
- Ground truth origin:
  - **9K and 99K**: Ground-truth nearest neighbours came precomputed and shipped within the respective `.h5` dataset packages.
  - **1M**: The nearest-neighbour lists shipped with the original 1M HDF5 file did not match the vectors, so `match_top1.ibin` was regenerated by exact brute-force search over all 1M base vectors (`1m/brute.cpp`).
- File format (raw binary, no header):
  - `dataset.fbin`: n x 128 float32 (9,000; 99,000; 1,000,000)
  - `query.fbin`: 1,000 x 128 float32 at 9K and 99K; 10,000 x 128 float32 at 1M
  - `9k/match.ibin`: 1,000 x 10 int32 (shipped with dataset; only the first column is used for top-1)
  - `99k/match.ibin`: 1,000 int32 (shipped with dataset)
  - `1m/match_top1.ibin`: 10,000 int32 (regenerated via brute force; benchmarks evaluate the first 1,000 queries)
- **Recall@1 (exact-id):** fraction of queries where the returned id equals the ground-truth id.
- **Recall@1 (tie-aware):** also counts a result as correct if its squared distance equals the ground-truth point's squared distance. This is exact for SIFT because all values are integers, so float sums are exact. A "closer than ground truth" counter is also printed and was 0 everywhere, so the ground truth is correct.
- The 9K set has no duplicate vectors; exact-id is used there, and brute force returns recall 1.0.
- Raw dataset vectors (`dataset.fbin` and `query.fbin`) are not included due to file size. Obtain the `.h5` files from the sources above and convert them to raw row-major float32 binary format.
- Precomputed ground-truth files (`9k/match.ibin`, `99k/match.ibin`, `1m/match_top1.ibin`) are committed directly in the repository so benchmarks can run immediately without having to recompute ground truth.

## Benchmark methodology

- **Machine:** 13th Gen Intel Core i5-13450HX, 24 GB RAM, Windows 11 Home 64-bit. Compiler: g++ 16.2.0 (Rev3, MSYS2), C++17, **no optimization flags** (-O0) for every number reported here.
- **Timing:** wall clock (`std::chrono::high_resolution_clock`), average ms/query over 1000 queries. Recall is computed after the timer stops.
- **Slow-state problem:** on the benchmark machine the same binary ran 1.3x to 2x slower in some stretches (one cause was Windows "Balanced" power mode; other slow stretches were unexplained). Brute-force timings are bimodal for the same reason.
- **Probes:** each sweep is bracketed by a short brute-force probe (20 queries at 99K and 1M with reference 235.1 ms/query at 1M; 100 queries at 9K with reference 2.097 ms/query). A sweep counts only if both probes are within 5% (8% at 9K) of the reference. Brute-force results are reported as the median of the fast cluster, with the full range given.
- **Build once, sweep many times:** at 9K and 1M the graph is built once per process and the ef sweep is repeated inside it. Sweeps inside one process agree more closely than separate processes would, so the error bars understate process-to-process variation (about +/-3% at 99K).
- **Same-state comparison:** only the 9K layered-vs-flat comparison is interleaved (flat pass, layered pass, alternating order each sweep). The 99K and 1M comparisons come from separate processes and sessions.

## Limitations

- One machine, one dataset (SIFT), one random seed (42), Recall@1 only.
- The brute-force baseline is single-threaded, unvectorized and unoptimized; it is not a comparison with FAISS, hnswlib or other optimized libraries.
- Absolute timings are for -O0 builds; optimized builds will be faster and could change the speedups.
- Layered-vs-flat gaps at 99K and 1M carry about +/-3 to 5 points of cross-session uncertainty, and gaps at ef >= 200 are within noise.
- Closest-M neighbour selection only (no diversity heuristic), no deletions or updates, no persistence, no multithreading.

## Repository layout

```
9k/    brute.cpp  flat.cpp  hnsw.cpp  compare.cpp  match.ibin
99k/   brute.cpp  flat.cpp  hnsw.cpp  match.ibin
1m/    brute.cpp  flat.cpp  hnsw.cpp  match_top1.ibin
```

Each program is a single self-contained file. Each size folder is also where that size's data files go (they are not committed). At 9K, `compare.cpp` builds the flat and layered graphs in one process and interleaves their query sweeps; the 9K layered-vs-flat table comes from it.

## Build and run

Requires a C++17 compiler with the GCC standard library (the code includes `<bits/stdc++.h>`; MSVC is not supported).

```
cd 1m
g++ -std=c++17 hnsw.cpp -o hnsw
./hnsw               # 5 sweeps by default (Windows: .\hnsw)
./hnsw 10            # optional first argument: number of sweeps
```

Run each program from inside its size folder, which must contain `dataset.fbin`, `query.fbin` and the match file. The 1M graph builds take about 13 to 17 minutes each, and the 1M brute force takes about 4 minutes per run. Keep the machine plugged in, in a high-performance power mode, and idle while benchmarking.

## Planned

- The neighbour-selection heuristic from the HNSW paper (diverse neighbours), compared against closest-M at equal recall.
- Filtered search (pre-, post- and in-graph filtering).

## References

- Y. A. Malkov and D. A. Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, IEEE TPAMI, 2020 (arXiv:1603.09320).
- H. Jegou, M. Douze and C. Schmid, *Product quantization for nearest neighbor search*, IEEE TPAMI, 2011 (source of the SIFT1M dataset).

## License

MIT. See `LICENSE`.
