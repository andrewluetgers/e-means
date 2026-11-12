//! CPU nearest-centroid search: AVX-512, AVX2+FMA or portable, chosen at runtime.
//!
//! The search minimizes `|c|^2 - 2 x.c` (and adds `|x|^2` back for the returned
//! distance). The inner products come from a 4-query x 4-centroid micro-kernel
//! with one SIMD accumulator per pair, so every loaded vector is used four
//! times and the FMA units stay saturated. Centroids are walked in blocks
//! sized for L2, query tiles are spread over threads with rayon
//! (`RAYON_NUM_THREADS` controls the thread count).

use crate::{AssignRows, Assigner, Result, Scalar};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Simd {
    Avx512,
    Avx2,
    Portable,
}

impl Simd {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Simd::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Simd::Avx2;
            }
        }
        Simd::Portable
    }
}

/// Inner products of 4 queries with 4 centroids: `out[4 * q + c]`.
type Kernel = unsafe fn(q: &[*const f32; 4], c: &[*const f32; 4], d: usize, out: &mut [f32; 16]);

/// Bytes of centroids (and of queries) kept hot per block.
const BLOCK_BYTES: usize = 256 * 1024;

pub struct CpuAssigner {
    simd: Simd,
    k: usize,
    d: usize,
    centroids: Vec<f32>,
    norms: Vec<f32>,
}

impl Default for CpuAssigner {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuAssigner {
    pub fn new() -> Self {
        Self::with_simd(Simd::detect())
    }

    /// Force a kernel (for tests and benchmarks). Panics if the CPU lacks it.
    pub fn with_simd(simd: Simd) -> Self {
        let best = Simd::detect();
        let ok = match simd {
            Simd::Avx512 => best == Simd::Avx512,
            Simd::Avx2 => best != Simd::Portable,
            Simd::Portable => true,
        };
        assert!(ok, "{simd:?} is not supported by this CPU");
        CpuAssigner { simd, k: 0, d: 0, centroids: Vec::new(), norms: Vec::new() }
    }

    fn kernel(&self) -> Kernel {
        match self.simd {
            #[cfg(target_arch = "x86_64")]
            Simd::Avx512 => dots_avx512,
            #[cfg(target_arch = "x86_64")]
            Simd::Avx2 => dots_avx2,
            _ => dots_portable,
        }
    }
}

impl Assigner for CpuAssigner {
    fn set_centroids(&mut self, centroids: &[f32], k: usize, d: usize) -> Result<()> {
        assert_eq!(centroids.len(), k * d);
        self.k = k;
        self.d = d;
        self.centroids.clear();
        self.centroids.extend_from_slice(centroids);
        self.norms.resize(k, 0.0);
        self.norms.par_iter_mut().zip(self.centroids.par_chunks_exact(d)).for_each(|(n, c)| *n = sqnorm(c));
        Ok(())
    }

    fn name(&self) -> String {
        format!("CPU {:?}, {} threads", self.simd, rayon::current_num_threads())
    }
}

/// Non-`f32` rows are converted to `f32` one cache-sized tile at a time, inside
/// the worker threads, so results are identical to passing the same values as `f32`.
impl<T: Scalar> AssignRows<T> for CpuAssigner {
    fn assign(&mut self, x: &[T], labels: &mut [u32], dist: &mut [f32]) -> Result<()> {
        let (k, d) = (self.k, self.d);
        assert!(k > 0, "set_centroids must be called first");
        assert_eq!(x.len(), labels.len() * d);
        assert_eq!(labels.len(), dist.len());
        let kernel = self.kernel();
        let block = (BLOCK_BYTES / (4 * d)).clamp(4, 4096) & !3;
        let (centroids, norms) = (&self.centroids[..], &self.norms[..]);

        labels.par_chunks_mut(block).zip(dist.par_chunks_mut(block)).zip(x.par_chunks(block * d)).for_each(
            |((labels, best), x)| {
                let converted: Vec<f32>;
                let x: &[f32] = match T::as_f32(x) {
                    Some(rows) => rows,
                    None => {
                        converted = x.iter().map(|v| v.to_f32()).collect();
                        &converted
                    }
                };
                let nq = labels.len();
                best.fill(f32::INFINITY);
                labels.fill(0);
                let mut out = [0f32; 16];
                for c0 in (0..k).step_by(block) {
                    let c1 = (c0 + block).min(k);
                    for q0 in (0..nq).step_by(4) {
                        let qn = (nq - q0).min(4);
                        // Short groups repeat their last row; the extra results are ignored.
                        let qp: [*const f32; 4] = std::array::from_fn(|i| x[(q0 + i.min(qn - 1)) * d..].as_ptr());
                        for j0 in (c0..c1).step_by(4) {
                            let cn = (c1 - j0).min(4);
                            let cp: [*const f32; 4] =
                                std::array::from_fn(|i| centroids[(j0 + i.min(cn - 1)) * d..].as_ptr());
                            // SAFETY: every pointer addresses d readable floats, and
                            // the kernel was selected by runtime feature detection.
                            unsafe { kernel(&qp, &cp, d, &mut out) };
                            for qi in 0..qn {
                                let (mut b, mut l) = (best[q0 + qi], labels[q0 + qi]);
                                for ci in 0..cn {
                                    let s = norms[j0 + ci] - 2.0 * out[4 * qi + ci];
                                    if s < b {
                                        b = s;
                                        l = (j0 + ci) as u32;
                                    }
                                }
                                best[q0 + qi] = b;
                                labels[q0 + qi] = l;
                            }
                        }
                    }
                }
                for (b, row) in best.iter_mut().zip(x.chunks_exact(d)) {
                    *b = (*b + sqnorm(row)).max(0.0);
                }
            },
        );
        Ok(())
    }
}

/// Squared norm with 16 independent accumulators, which LLVM vectorizes.
pub(crate) fn sqnorm(v: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let chunks = v.chunks_exact(16);
    let tail = chunks.remainder();
    for ch in chunks {
        for i in 0..16 {
            acc[i] += ch[i] * ch[i];
        }
    }
    acc.iter().sum::<f32>() + tail.iter().map(|a| a * a).sum::<f32>()
}

unsafe fn dots_portable(q: &[*const f32; 4], c: &[*const f32; 4], d: usize, out: &mut [f32; 16]) {
    for qi in 0..4 {
        let a = std::slice::from_raw_parts(q[qi], d);
        for ci in 0..4 {
            let b = std::slice::from_raw_parts(c[ci], d);
            let mut acc = [0f32; 8];
            let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
            let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
            for (x, y) in ca.zip(cb) {
                for i in 0..8 {
                    acc[i] += x[i] * y[i];
                }
            }
            out[4 * qi + ci] = acc.iter().sum::<f32>() + tail;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dots_avx512(q: &[*const f32; 4], c: &[*const f32; 4], d: usize, out: &mut [f32; 16]) {
    use std::arch::x86_64::*;
    // 16 accumulators + 4 query + 4 centroid registers: 24 of the 32 zmm.
    let mut acc = [_mm512_setzero_ps(); 16];
    let mut i = 0;
    while i + 16 <= d {
        let qv: [__m512; 4] = std::array::from_fn(|t| _mm512_loadu_ps(q[t].add(i)));
        let cv: [__m512; 4] = std::array::from_fn(|t| _mm512_loadu_ps(c[t].add(i)));
        for a in 0..4 {
            for b in 0..4 {
                acc[4 * a + b] = _mm512_fmadd_ps(qv[a], cv[b], acc[4 * a + b]);
            }
        }
        i += 16;
    }
    if i < d {
        // Masked loads read only the remaining d - i floats.
        let mask: __mmask16 = (1u16 << (d - i)) - 1;
        let qv: [__m512; 4] = std::array::from_fn(|t| _mm512_maskz_loadu_ps(mask, q[t].add(i)));
        let cv: [__m512; 4] = std::array::from_fn(|t| _mm512_maskz_loadu_ps(mask, c[t].add(i)));
        for a in 0..4 {
            for b in 0..4 {
                acc[4 * a + b] = _mm512_fmadd_ps(qv[a], cv[b], acc[4 * a + b]);
            }
        }
    }
    for t in 0..16 {
        out[t] = _mm512_reduce_add_ps(acc[t]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dots_avx2(q: &[*const f32; 4], c: &[*const f32; 4], d: usize, out: &mut [f32; 16]) {
    use std::arch::x86_64::*;
    // Only 16 ymm registers: two passes of 2 queries x 4 centroids
    // (8 accumulators + 2 query + 4 centroid registers).
    let body = d & !7;
    for half in 0..2 {
        let (q0, q1) = (q[2 * half], q[2 * half + 1]);
        let mut acc = [_mm256_setzero_ps(); 8];
        let mut i = 0;
        while i < body {
            let (a0, a1) = (_mm256_loadu_ps(q0.add(i)), _mm256_loadu_ps(q1.add(i)));
            let cv: [__m256; 4] = std::array::from_fn(|t| _mm256_loadu_ps(c[t].add(i)));
            for b in 0..4 {
                acc[b] = _mm256_fmadd_ps(a0, cv[b], acc[b]);
                acc[4 + b] = _mm256_fmadd_ps(a1, cv[b], acc[4 + b]);
            }
            i += 8;
        }
        for t in 0..8 {
            let v = acc[t];
            let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            let qrow = if t < 4 { q0 } else { q1 };
            let mut r = _mm_cvtss_f32(s);
            for i in body..d {
                r += *qrow.add(i) * *c[t % 4].add(i);
            }
            out[8 * half + t] = r;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// Every kernel must agree with a brute-force f64 search, across odd
    /// dimensions and row/centroid counts that exercise all the tail paths.
    #[test]
    fn kernels_match_brute_force() {
        let levels: Vec<Simd> = [Simd::Portable, Simd::Avx2, Simd::Avx512]
            .into_iter()
            .filter(|&s| s == Simd::Portable || Simd::detect() == Simd::Avx512 || s == Simd::detect())
            .collect();
        for &(n, k, d) in &[(1, 1, 1), (5, 3, 7), (13, 9, 16), (37, 70, 33), (130, 257, 100), (64, 1030, 384), (9, 6, 1027)] {
            let (x, c) = (pseudo(n * d, 1), pseudo(k * d, 2));
            for &simd in &levels {
                let mut a = CpuAssigner::with_simd(simd);
                a.set_centroids(&c, k, d).unwrap();
                let (mut labels, mut dist) = (vec![0u32; n], vec![0f32; n]);
                AssignRows::<f32>::assign(&mut a, &x, &mut labels, &mut dist).unwrap();
                for i in 0..n {
                    let exact = |j: usize| -> f64 {
                        (0..d).map(|t| (x[i * d + t] as f64 - c[j * d + t] as f64).powi(2)).sum()
                    };
                    let best = (0..k).map(exact).fold(f64::INFINITY, f64::min);
                    let got = exact(labels[i] as usize);
                    let tol = 1e-4 * (1.0 + best);
                    assert!(got <= best + tol, "{simd:?} n={n} k={k} d={d} row {i}: {got} vs {best}");
                    assert!((dist[i] as f64 - best).abs() <= tol, "{simd:?} distance {} vs {best}", dist[i]);
                }
            }
        }
    }
}
