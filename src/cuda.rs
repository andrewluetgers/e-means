//! NVIDIA GPU nearest-centroid search (feature `cuda`), FP32.
//!
//! Per chunk of rows: upload the rows, one cuBLAS `sgemm` for all inner
//! products `x.c`, then a small CUDA kernel takes the per-row argmin of
//! `|c|^2 - 2 x.c`. The kernel is compiled at startup with NVRTC, and the CUDA
//! libraries are loaded at runtime, so building needs no CUDA toolkit.
//!
//! The training data stays in host memory; only one chunk is on the GPU at a
//! time, so the dataset does not need to fit in GPU memory.
//!
//! NOTE: this backend compiles in CI without a GPU but has not been run on
//! one yet. Run `cargo test --features cuda -- --ignored` on a GPU machine:
//! it checks this backend against the CPU kernel.

use crate::{AssignRows, Assigner, Result, Scalar};
use cudarc::cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

const ARGMIN_SRC: &str = r#"
extern "C" __global__ void argmin_rows(
    const float* dots,   // r x k, row-major: dots[row * k + j] = x_row . c_j
    const float* cnorm,  // k: |c_j|^2
    unsigned int* labels, float* best, int r, int k)
{
    __shared__ float sv[256];
    __shared__ unsigned int si[256];
    const int row = blockIdx.x, t = threadIdx.x;
    if (row >= r) return;
    const float* g = dots + (size_t)row * (size_t)k;
    float bv = 3.402823466e+38f;
    unsigned int bi = 0xffffffffu;
    for (int j = t; j < k; j += blockDim.x) {
        const float s = cnorm[j] - 2.0f * g[j];
        if (s < bv) { bv = s; bi = (unsigned int)j; }
    }
    sv[t] = bv; si[t] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            const float ov = sv[t + s]; const unsigned int oi = si[t + s];
            if (ov < sv[t] || (ov == sv[t] && oi < si[t])) { sv[t] = ov; si[t] = oi; }
        }
        __syncthreads();
    }
    if (t == 0) { labels[row] = si[0]; best[row] = sv[0]; }
}
"#;

pub struct CudaAssigner {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    argmin: CudaFunction,
    k: usize,
    d: usize,
    rows_cap: usize,
    centroids: Option<CudaSlice<f32>>,
    cnorm: Option<CudaSlice<f32>>,
    x: Option<CudaSlice<f32>>,
    dots: Option<CudaSlice<f32>>,
    labels: Option<CudaSlice<u32>>,
    best: Option<CudaSlice<f32>>,
    norms_host: Vec<f32>,
}

impl CudaAssigner {
    /// Use GPU number `ordinal` (0 for the first one).
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;
        let module = ctx.load_module(cudarc::nvrtc::compile_ptx(ARGMIN_SRC)?)?;
        let argmin = module.load_function("argmin_rows")?;
        Ok(CudaAssigner {
            ctx,
            stream,
            blas,
            argmin,
            k: 0,
            d: 0,
            rows_cap: 0,
            centroids: None,
            cnorm: None,
            x: None,
            dots: None,
            labels: None,
            best: None,
            norms_host: Vec::new(),
        })
    }

    /// (Re)allocate device buffers when the problem shape changes.
    fn reshape(&mut self, k: usize, d: usize) -> Result<()> {
        if (k, d) == (self.k, self.d) {
            return Ok(());
        }
        if k > i32::MAX as usize || d > i32::MAX as usize {
            return Err("k and d must fit in a 32-bit int for cuBLAS".into());
        }
        // Free the old buffers first so the budget below sees that memory.
        (self.centroids, self.cnorm, self.x, self.dots, self.labels, self.best) = (None, None, None, None, None, None);
        let (free, _total) = self.ctx.mem_get_info()?;
        let fixed = 4 * (k * d + k);
        let per_row = 4 * (k + d) + 8;
        // Use at most 60% of the free memory, and at most 65536 rows per chunk.
        let budget = (free as f64 * 0.6) as usize;
        if budget < fixed + per_row {
            return Err(format!("not enough free GPU memory for k = {k}, d = {d}").into());
        }
        let rows = ((budget - fixed) / per_row).clamp(1, 65536);
        // SAFETY: every buffer is fully written before it is read.
        unsafe {
            self.centroids = Some(self.stream.alloc::<f32>(k * d)?);
            self.cnorm = Some(self.stream.alloc::<f32>(k)?);
            self.x = Some(self.stream.alloc::<f32>(rows * d)?);
            self.dots = Some(self.stream.alloc::<f32>(rows * k)?);
            self.labels = Some(self.stream.alloc::<u32>(rows)?);
            self.best = Some(self.stream.alloc::<f32>(rows)?);
        }
        (self.k, self.d, self.rows_cap) = (k, d, rows);
        Ok(())
    }
}

impl Assigner for CudaAssigner {
    fn set_centroids(&mut self, centroids: &[f32], k: usize, d: usize) -> Result<()> {
        assert_eq!(centroids.len(), k * d);
        self.reshape(k, d)?;
        self.norms_host.clear();
        self.norms_host.extend(centroids.chunks_exact(d).map(|c| c.iter().map(|v| v * v).sum::<f32>()));
        self.stream.memcpy_htod(centroids, self.centroids.as_mut().unwrap())?;
        self.stream.memcpy_htod(&self.norms_host, self.cnorm.as_mut().unwrap())?;
        Ok(())
    }

    fn set_bias(&mut self, bias: &[f32]) -> Result<()> {
        assert_eq!(bias.len(), self.k);
        self.norms_host.iter_mut().zip(bias).for_each(|(n, b)| *n += b);
        self.stream.memcpy_htod(&self.norms_host, self.cnorm.as_mut().unwrap())?;
        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "CUDA FP32 (cuBLAS sgemm + argmin), {}, {} rows per chunk",
            self.ctx.name().unwrap_or_else(|_| "unknown GPU".into()),
            self.rows_cap
        )
    }
}

/// Non-`f32` rows are converted to `f32` on the host before upload.
impl<T: Scalar> AssignRows<T> for CudaAssigner {
    fn assign(&mut self, x: &[T], labels: &mut [u32], dist: &mut [f32]) -> Result<()> {
        let converted: Vec<f32>;
        let x: &[f32] = match T::as_f32(x) {
            Some(rows) => rows,
            None => {
                converted = x.iter().map(|v| v.to_f32()).collect();
                &converted
            }
        };
        let (k, d) = (self.k, self.d);
        assert!(k > 0, "set_centroids must be called first");
        assert_eq!(x.len(), labels.len() * d);
        assert_eq!(labels.len(), dist.len());
        let (c_dev, cnorm_dev) = (self.centroids.as_ref().unwrap(), self.cnorm.as_ref().unwrap());
        let x_dev = self.x.as_mut().unwrap();
        let dots_dev = self.dots.as_mut().unwrap();
        let labels_dev = self.labels.as_mut().unwrap();
        let best_dev = self.best.as_mut().unwrap();

        for ((xc, lc), dc) in
            x.chunks(self.rows_cap * d).zip(labels.chunks_mut(self.rows_cap)).zip(dist.chunks_mut(self.rows_cap))
        {
            let r = lc.len();
            self.stream.memcpy_htod(xc, x_dev)?;
            // Row-major (r x d) rows are a column-major d x r matrix X, and the
            // centroids a column-major d x k matrix C. G = C^T X is k x r
            // column-major, i.e. G[j + i * k] = c_j . x_i: row-major r x k.
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: k as i32,
                n: r as i32,
                k: d as i32,
                alpha: 1.0f32,
                lda: d as i32,
                ldb: d as i32,
                beta: 0.0f32,
                ldc: k as i32,
            };
            // SAFETY: A is d x k, B is d x r (first r * d floats of x_dev), and
            // C has room for k x r; all three were allocated with those sizes.
            unsafe { self.blas.gemm(cfg, c_dev, &*x_dev, dots_dev)? };

            let (ri, ki) = (r as i32, k as i32);
            let launch = LaunchConfig { grid_dim: (r as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let mut args = self.stream.launch_builder(&self.argmin);
            args.arg(&*dots_dev).arg(cnorm_dev).arg(&mut *labels_dev).arg(&mut *best_dev).arg(&ri).arg(&ki);
            // SAFETY: argument order and types match argmin_rows; the kernel
            // reads r * k dots and k norms, and writes r labels and r values.
            unsafe { args.launch(launch)? };

            self.stream.memcpy_dtoh(&labels_dev.slice(..r), lc)?;
            self.stream.memcpy_dtoh(&best_dev.slice(..r), dc)?;
            for (b, row) in dc.iter_mut().zip(xc.chunks_exact(d)) {
                *b = (*b + row.iter().map(|v| v * v).sum::<f32>()).max(0.0);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuAssigner;

    /// Needs a GPU: `cargo test --features cuda -- --ignored`.
    #[test]
    #[ignore]
    fn matches_cpu() {
        let (n, k, d) = (3000, 1031, 100);
        let mut s = 1u32;
        let mut gen = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    (s >> 8) as f32 / (1 << 24) as f32 - 0.5
                })
                .collect()
        };
        let (x, c) = (gen(n * d), gen(k * d));
        let run = |a: &mut dyn AssignRows<f32>| {
            a.set_centroids(&c, k, d).unwrap();
            let (mut l, mut dist) = (vec![0u32; n], vec![0f32; n]);
            a.assign(&x, &mut l, &mut dist).unwrap();
            (l, dist)
        };
        let (cl, cd) = run(&mut CpuAssigner::new());
        let (gl, gd) = run(&mut CudaAssigner::new(0).unwrap());
        let same = cl.iter().zip(&gl).filter(|(a, b)| a == b).count();
        assert!(same as f64 >= 0.999 * n as f64, "only {same} of {n} labels agree");
        for (a, b) in cd.iter().zip(&gd) {
            assert!((a - b).abs() <= 1e-3 * (1.0 + a.abs()), "{a} vs {b}");
        }
    }
}
