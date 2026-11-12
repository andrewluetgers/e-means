//! Integer nearest-centroid search for `i8` rows (AVX-512 VNNI, portable fallback).
//!
//! The centroids stay `f32` in the model; for the search only, each batch's
//! snapshot is rounded to integers and the ranking is done in exact integer
//! arithmetic. Sums, EMA updates and the final centroids are unaffected.
//!
//! Two centroid precisions:
//!
//! - [`CentroidBits::I8`]: centroids rounded to the nearest integer, as the
//!   faiss INT8 kernels do. `vpdpbusd`: 64 products per instruction.
//! - [`CentroidBits::I16`]: centroids rounded to 1/2^s of a unit (s up to 7,
//!   chosen from `d` so sums cannot overflow). `vpdpwssd`: 32 products per
//!   instruction, and close to exact ranking.
//!
//! Returned distances are to the rounded centroid. For exact distances to the
//! final `f32` centroids use [`crate::cpu::CpuAssigner`], which also accepts `i8` rows.

use crate::{AssignRows, Assigner, Result};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CentroidBits {
    I8,
    I16,
}

/// Largest dimension for which no intermediate sum can overflow an `i32`.
pub const MAX_D: usize = 32768;
const BLOCK_BYTES: usize = 256 * 1024;

type Kernel8 = unsafe fn(q: &[*const i8; 4], c: &[*const i8; 4], d: usize, out: &mut [i32; 16]);
type Kernel16 = unsafe fn(q: &[*const i8; 4], c: &[*const i16; 4], d: usize, out: &mut [i32; 16]);

pub struct CpuI8Assigner {
    bits: CentroidBits,
    vnni: bool,
    k: usize,
    d: usize,
    shift: u32,     // centroids are stored as round(c * 2^shift)
    c8: Vec<i8>,    // I8 mode
    c16: Vec<i16>,  // I16 mode
    /// Per centroid: `|c_q|^2` (plus the `vpdpbusd` offset correction and any
    /// balance bias), so that `score = offset - mult * dot` ranks by distance.
    offset: Vec<i64>,
    mult: i64,
}

fn has_vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vnni");
    }
    #[allow(unreachable_code)]
    false
}

impl CpuI8Assigner {
    pub fn new(bits: CentroidBits) -> Self {
        Self::with_vnni(bits, has_vnni())
    }

    /// `vnni = false` forces the portable kernel (for tests and benchmarks).
    pub fn with_vnni(bits: CentroidBits, vnni: bool) -> Self {
        assert!(!vnni || has_vnni(), "AVX-512 VNNI is not supported by this CPU");
        CpuI8Assigner { bits, vnni, k: 0, d: 0, shift: 0, c8: Vec::new(), c16: Vec::new(), offset: Vec::new(), mult: 2 }
    }
}

impl Assigner for CpuI8Assigner {
    fn set_centroids(&mut self, centroids: &[f32], k: usize, d: usize) -> Result<()> {
        assert_eq!(centroids.len(), k * d);
        if d > MAX_D {
            return Err(format!("integer assignment supports d <= {MAX_D}").into());
        }
        (self.k, self.d) = (k, d);
        match self.bits {
            CentroidBits::I8 => {
                self.shift = 0;
                self.c8.resize(k * d, 0);
                self.c8.par_iter_mut().zip(centroids).for_each(|(q, &c)| *q = c.round().clamp(-128.0, 127.0) as i8);
                // vpdpbusd needs an unsigned operand: rows are used as x + 128,
                // and (x + 128).c = x.c + 128 * sum(c). With dp the kernel's
                // result, |x - c|^2 - |x|^2 = |c|^2 + 256 * sum(c) - 2 * dp.
                // The portable kernel returns the same quantity.
                self.offset = self
                    .c8
                    .par_chunks_exact(d)
                    .map(|c| c.iter().map(|&v| (v as i64) * (v as i64) + 256 * v as i64).sum())
                    .collect();
                self.mult = 2;
            }
            CentroidBits::I16 => {
                // |x| <= 128 and |c16| <= 128 * 2^s: d * 128 * 128 * 2^s must stay below 2^31.
                self.shift = (0..=7u32).rev().find(|&s| (d as u64) << (14 + s) < 1 << 31).unwrap_or(0);
                let scale = (1u32 << self.shift) as f32;
                self.c16.resize(k * d, 0);
                self.c16
                    .par_iter_mut()
                    .zip(centroids)
                    .for_each(|(q, &c)| *q = (c * scale).round().clamp(-32768.0, 32767.0) as i16);
                // scale^2 * (|x - c|^2 - |x|^2) = |c16|^2 - 2 * scale * (x . c16)
                self.offset = self.c16.par_chunks_exact(d).map(|c| c.iter().map(|&v| (v as i64) * (v as i64)).sum()).collect();
                self.mult = 2 << self.shift;
            }
        }
        Ok(())
    }

    fn set_bias(&mut self, bias: &[f32]) -> Result<()> {
        assert_eq!(bias.len(), self.k);
        let unit = (1u64 << (2 * self.shift)) as f64; // score units per squared distance
        self.offset.iter_mut().zip(bias).for_each(|(o, &b)| *o += (b as f64 * unit).round() as i64);
        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "CPU int8 rows, {:?} centroids{}, {}, {} threads",
            self.bits,
            if self.bits == CentroidBits::I16 { format!(" (1/{} unit)", 1u32 << self.shift) } else { String::new() },
            if self.vnni { "AVX-512 VNNI" } else { "portable" },
            rayon::current_num_threads()
        )
    }
}

impl AssignRows<i8> for CpuI8Assigner {
    fn assign(&mut self, x: &[i8], labels: &mut [u32], dist: &mut [f32]) -> Result<()> {
        let (k, d) = (self.k, self.d);
        assert!(k > 0, "set_centroids must be called first");
        assert_eq!(x.len(), labels.len() * d);
        assert_eq!(labels.len(), dist.len());
        let wide = self.bits == CentroidBits::I16;
        let k8: Kernel8 = if self.vnni { vnni_kernel8() } else { dots8_portable };
        let k16: Kernel16 = if self.vnni { vnni_kernel16() } else { dots16_portable };
        let elem = if wide { 2 } else { 1 };
        let block = (BLOCK_BYTES / (elem * d)).clamp(4, 8192) & !3;
        let (c8, c16, offset, mult) = (&self.c8[..], &self.c16[..], &self.offset[..], self.mult);
        let inv_unit = 1.0 / (1u64 << (2 * self.shift)) as f64;

        labels.par_chunks_mut(block).zip(dist.par_chunks_mut(block)).zip(x.par_chunks(block * d)).for_each(
            |((labels, dist), x)| {
                let nq = labels.len();
                let mut best = vec![i64::MAX; nq];
                labels.fill(0);
                let mut out = [0i32; 16];
                for c0 in (0..k).step_by(block) {
                    let c1 = (c0 + block).min(k);
                    for q0 in (0..nq).step_by(4) {
                        let qn = (nq - q0).min(4);
                        let qp: [*const i8; 4] = std::array::from_fn(|i| x[(q0 + i.min(qn - 1)) * d..].as_ptr());
                        for j0 in (c0..c1).step_by(4) {
                            let cn = (c1 - j0).min(4);
                            // SAFETY: every pointer addresses d readable elements; the
                            // VNNI kernels are only selected after runtime detection.
                            unsafe {
                                if wide {
                                    let cp: [*const i16; 4] = std::array::from_fn(|i| c16[(j0 + i.min(cn - 1)) * d..].as_ptr());
                                    k16(&qp, &cp, d, &mut out);
                                } else {
                                    let cp: [*const i8; 4] = std::array::from_fn(|i| c8[(j0 + i.min(cn - 1)) * d..].as_ptr());
                                    k8(&qp, &cp, d, &mut out);
                                }
                            }
                            for qi in 0..qn {
                                let (mut b, mut l) = (best[q0 + qi], labels[q0 + qi]);
                                for ci in 0..cn {
                                    let s = offset[j0 + ci] - mult * out[4 * qi + ci] as i64;
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
                for ((v, &b), row) in dist.iter_mut().zip(&best).zip(x.chunks_exact(d)) {
                    let norm: i64 = row.iter().map(|&a| (a as i64) * (a as i64)).sum();
                    *v = (norm as f64 + b as f64 * inv_unit).max(0.0) as f32;
                }
            },
        );
        Ok(())
    }
}

/// `out[4q + c] = sum_i (x_q[i] + 128) * c_c[i]`, matching the VNNI kernel.
unsafe fn dots8_portable(q: &[*const i8; 4], c: &[*const i8; 4], d: usize, out: &mut [i32; 16]) {
    for qi in 0..4 {
        let a = std::slice::from_raw_parts(q[qi], d);
        for ci in 0..4 {
            let b = std::slice::from_raw_parts(c[ci], d);
            out[4 * qi + ci] = a.iter().zip(b).map(|(&x, &y)| (x as i32 + 128) * y as i32).sum();
        }
    }
}

unsafe fn dots16_portable(q: &[*const i8; 4], c: &[*const i16; 4], d: usize, out: &mut [i32; 16]) {
    for qi in 0..4 {
        let a = std::slice::from_raw_parts(q[qi], d);
        for ci in 0..4 {
            let b = std::slice::from_raw_parts(c[ci], d);
            out[4 * qi + ci] = a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum();
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn vnni_kernel8() -> Kernel8 {
    dots8_vnni
}
#[cfg(target_arch = "x86_64")]
fn vnni_kernel16() -> Kernel16 {
    dots16_vnni
}
#[cfg(not(target_arch = "x86_64"))]
fn vnni_kernel8() -> Kernel8 {
    dots8_portable
}
#[cfg(not(target_arch = "x86_64"))]
fn vnni_kernel16() -> Kernel16 {
    dots16_portable
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn dots8_vnni(q: &[*const i8; 4], c: &[*const i8; 4], d: usize, out: &mut [i32; 16]) {
    use std::arch::x86_64::*;
    let flip = _mm512_set1_epi8(-128); // x ^ 0x80 == x + 128 as u8
    let mut acc = [_mm512_setzero_si512(); 16];
    let mut i = 0;
    while i < d {
        let left = d - i;
        // The last step loads only the remaining bytes; masked-out centroid
        // lanes are zero, so they add nothing.
        let mask: __mmask64 = if left >= 64 { !0 } else { (1u64 << left) - 1 };
        let qv: [__m512i; 4] = std::array::from_fn(|t| _mm512_xor_si512(_mm512_maskz_loadu_epi8(mask, q[t].add(i)), flip));
        let cv: [__m512i; 4] = std::array::from_fn(|t| _mm512_maskz_loadu_epi8(mask, c[t].add(i)));
        for a in 0..4 {
            for b in 0..4 {
                acc[4 * a + b] = _mm512_dpbusd_epi32(acc[4 * a + b], qv[a], cv[b]);
            }
        }
        i += 64;
    }
    for t in 0..16 {
        out[t] = _mm512_reduce_add_epi32(acc[t]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
unsafe fn dots16_vnni(q: &[*const i8; 4], c: &[*const i16; 4], d: usize, out: &mut [i32; 16]) {
    use std::arch::x86_64::*;
    let mut acc = [_mm512_setzero_si512(); 16];
    let mut i = 0;
    while i < d {
        let left = d - i;
        let mask: __mmask32 = if left >= 32 { !0 } else { (1u32 << left) - 1 };
        let qv: [__m512i; 4] = std::array::from_fn(|t| _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, q[t].add(i))));
        let cv: [__m512i; 4] = std::array::from_fn(|t| _mm512_maskz_loadu_epi16(mask, c[t].add(i)));
        for a in 0..4 {
            for b in 0..4 {
                acc[4 * a + b] = _mm512_dpwssd_epi32(acc[4 * a + b], qv[a], cv[b]);
            }
        }
        i += 32;
    }
    for t in 0..16 {
        out[t] = _mm512_reduce_add_epi32(acc[t]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo(n: usize, seed: u32) -> Vec<i8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8 as i8
            })
            .collect()
    }

    /// With integer-valued centroids both precisions must reproduce an exact
    /// brute-force search, label for label and distance for distance, on every
    /// kernel and across dimensions that exercise the masked tails, including
    /// the extreme values -128 and 127.
    #[test]
    fn integer_kernels_are_exact() {
        for &(n, k, d) in &[(1, 1, 1), (5, 3, 7), (13, 9, 64), (37, 70, 33), (130, 257, 100), (20, 1030, 512), (9, 6, 1027)] {
            let (x, ci) = (pseudo(n * d, 1), pseudo(k * d, 2));
            let c: Vec<f32> = ci.iter().map(|&v| v as f32).collect();
            let exact: Vec<(u32, i64)> = (0..n)
                .map(|i| {
                    (0..k)
                        .map(|j| (j as u32, (0..d).map(|t| (x[i * d + t] as i64 - ci[j * d + t] as i64).pow(2)).sum::<i64>()))
                        .min_by_key(|&(j, s)| (s, j))
                        .unwrap()
                })
                .collect();
            for bits in [CentroidBits::I8, CentroidBits::I16] {
                for vnni in [false, true].into_iter().filter(|&v| !v || has_vnni()) {
                    let mut a = CpuI8Assigner::with_vnni(bits, vnni);
                    a.set_centroids(&c, k, d).unwrap();
                    let (mut labels, mut dist) = (vec![0u32; n], vec![0f32; n]);
                    a.assign(&x, &mut labels, &mut dist).unwrap();
                    for i in 0..n {
                        assert_eq!(labels[i], exact[i].0, "{bits:?} vnni={vnni} n={n} k={k} d={d} row {i}");
                        assert_eq!(dist[i], exact[i].1 as f32, "{bits:?} vnni={vnni} distance, row {i}");
                    }
                }
            }
        }
    }

    #[test]
    fn i16_resolves_fractional_centroids() {
        // Two centroids that round to the same integer: I16 still tells them apart.
        let (x, c) = ([1i8, 0], [0.26f32, 0.0, -0.26, 0.0]);
        let mut a = CpuI8Assigner::new(CentroidBits::I16);
        a.set_centroids(&c, 2, 2).unwrap();
        let (mut l, mut dist) = ([9u32], [0f32]);
        a.assign(&x, &mut l, &mut dist).unwrap();
        assert_eq!(l[0], 0);
        assert!((dist[0] - 0.74f32 * 0.74).abs() < 0.01);
    }
}
