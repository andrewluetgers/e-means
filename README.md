# emeans

Rust implementation of **e-means** clustering (Bessa et al., Meta FAIR,
[arXiv:2611.09501](https://arxiv.org/abs/2611.09501)): mini-batch k-means with
EMA centroid updates, a cosine step schedule, automatic batch/step scaling
laws, dead-cluster revival and an exact last-epoch closing mean. The only
knobs you normally set are `k` and the number of epochs.

It is a port of the FP32 reference in faiss (`faiss.contrib.e_means.NumpyEmeans`,
MIT) and uses the constants that reference ships.

## Use

```rust
use emeans::{Config, train_cpu, assign, cpu::CpuAssigner};

// x: row-major n x d f32
let out = train_cpu(&x, d, k, &Config::default())?;      // 5 epochs
let centroids = out.centroids;                            // k x d
let labels = out.labels;                                  // last-pass labels
let (exact_labels, sq_dist) = assign(&x, &centroids, d, &mut CpuAssigner::new())?;
```

GPU (build with `--features cuda`):

```rust
let mut gpu = emeans::cuda::CudaAssigner::new(0)?;        // GPU 0
let out = emeans::train(&x, d, k, &Config::default(), &mut gpu)?;
```

CLI example on a raw f32 file:

```
cargo run --release --example cluster -- vectors.f32 <d> <k> --epochs 5 [--gpu]
```

## Backends

| Backend | How nearest centroids are found |
|---|---|
| CPU | 4x4 inner-product micro-kernel in AVX-512 or AVX2+FMA intrinsics (portable fallback), picked at runtime, cache-blocked, rayon-parallel. Threads: `RAYON_NUM_THREADS`. |
| CUDA | cuBLAS `sgemm` plus an argmin kernel compiled at startup with NVRTC, via `cudarc`. CUDA libraries are loaded at runtime, so building needs no toolkit. Data stays on the host and is uploaded a chunk at a time. |

Anything else (another GPU API, an ANN index for very large `k`) can be
plugged in by implementing the `Assigner` trait.

## Status

- **CPU: tested.** Unit tests check every kernel against a brute-force f64
  search, the recipe against values from faiss 1.16.0, and the closing-mean
  invariant.
- **CUDA: compiles, not yet run on a GPU.** On a GPU machine run
  `cargo test --release --features cuda -- --ignored`, which checks it
  against the CPU kernel.

Measured on 2 cores of a Xeon with AVX-512, 200,000 x 96-D synthetic mixture,
k = 1024 (final MSE from an exact re-assignment):

| | Time | Final MSE |
|---|---|---|
| This crate, 2 epochs | 1.1 s | 307.5 |
| This crate, 5 epochs | 2.7 s | 298.9 |
| faiss `NumpyEmeans`, 2 epochs | 2.3 s | 308.2 |
| faiss `NumpyEmeans`, 5 epochs | 5.7 s | 299.8 |
| faiss `Kmeans` (Lloyd), 25 iterations | 22.1 s | 328.3 |

## int8 rows

`train`, `train_cpu` and `assign` accept `&[i8]` or `&[u8]` rows as well as
`&[f32]` (a quarter of the memory). Centroids, sums and outputs stay `f32`.

| Search | Use | Notes |
|---|---|---|
| `cpu::CpuAssigner` | `train_cpu(&codes, ..)` | Rows widened to f32 a tile at a time: identical to f32 input, no speedup |
| `cpu_i8::CpuI8Assigner::new(CentroidBits::I8)` | `train(&codes, .., &mut a)` | Each batch's centroids rounded to int8 for the search, as faiss's INT8 kernels do. AVX-512 VNNI `vpdpbusd` |
| `cpu_i8::CpuI8Assigner::new(CentroidBits::I16)` | same | Centroids rounded to 1/128 unit (coarser above d = 512). `vpdpwssd` |

Without VNNI the integer assigner falls back to a portable kernel. On CUDA,
int8 rows are widened to f32 on the host; there is no integer GPU kernel.

`quantize::Int8Quantizer` converts f32 data to int8 codes and maps centroids
back. `Calibration::faiss()` reproduces faiss's `Int8Quantizer`: per-dimension
shift (sample mean), one global power-of-two scale putting the largest
centered magnitude at or below 112. `Calibration::clipped(q)` scales a
quantile to 127 instead.

`cargo run --release --example int8 -- vectors.f32 <d> <k>` measures all of
this on your data. 2 cores, AVX-512 VNNI, k = 1024, 5 epochs, 3 seeds; MSE in
original units against the original f32 vectors:

| | 100k x 512-D unit-norm: time | MSE vs f32 | 200k x 64-D heavy-tailed: time | MSE vs f32 |
|---|---|---|---|---|
| f32 rows, f32 search | 4.4 s | baseline | 2.3 s | baseline |
| int8 codes, f32 search | 4.6 s | +0.2% | 2.2 s | +0.02% |
| int8 codes, int16-centroid search | 2.7 s | -0.1% | 2.2 s | +0.05% |
| int8 codes, int8-centroid search | 2.1 s | -0.3% | 2.0 s | +0.10% |

(faiss calibration; the clipped calibration was within 0.13% everywhere.)
The MSE differences are within seed-to-seed variation. The integer search is
about 2x faster at 512-D and gives nothing at 64-D, where the search is no
longer the dominant cost.

## Cluster report and outliers

`emeans::report::report(&labels, &sqdist, &centroids, d, &cfg)` takes what
`assign` returns and produces, without touching the vectors again:

| Looking for | Signal | Where |
|---|---|---|
| Fringe points of a cluster | distance far above the cluster's own median (robust z) | `local_z` |
| Points in catch-all clusters | distance far above the corpus median | `global_z` |
| Catch-all clusters (unrelated rare things lumped together) | spread well above typical | `clusters[j].looseness` |
| Rare prototypes | small cluster far from every other centroid | `rare_prototypes()`, `isolation` |
| Prevalence of a prototype family | share of rows in a fixed-radius group of centroids | `families[f].share` |

`report.top_outliers(n)` ranks rows by the larger of the two z-scores.
Per-cluster `p99` and `max` are covering radii, usable for list pruning in an
IVF index. `emeans::balance_stats(&labels, k)` gives the imbalance factor
(`k * sum n_j^2 / n^2`, 1.0 = equal sizes). `cargo run --release --example outliers -- vectors.f32 <d> <k>`
prints a summary and writes per-cluster and per-outlier TSV files.

Check on synthetic data (200,000 heavy-tailed 64-D rows, k = 1024) with 200
scattered off-distribution rows and one 40-row rare tight group planted:

| | Result |
|---|---|
| Scattered rows in the top 240 scores (0.12%) | 151 / 200 |
| Scattered rows in the top 1200 scores (0.6%) | 200 / 200 |
| Rare group | own cluster, the most isolated (6.4x) |

The rare group is found at cluster level, not by point scores: its rows sit
close to their own centroid. Family radius and score thresholds are heuristics
to tune on real data.

## Differences from the reference

- GPU search is FP32 only. The reference's GPU path defaults to BF16 distances with fused
  Triton/CuTe kernels, which will be faster than `sgemm` + argmin here.
- No spherical mode, multi-GPU, or `max_points_per_centroid` subsampling.
- A different random generator, so runs are reproducible per seed but not
  bit-identical to faiss.
- At most 2^32 - 1 rows per call.
