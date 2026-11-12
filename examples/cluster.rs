//! Cluster a raw little-endian f32 file (n x d, row-major) and write the centroids.
//!
//!   cargo run --release --example cluster -- vectors.f32 <d> <k> [--epochs 5]
//!       [--out centroids.f32] [--simd avx512|avx2|portable] [--gpu [ordinal]]
//!
//! --gpu needs the crate's `cuda` feature.

use emeans::{cpu, AssignRows, Config};
use std::time::Instant;

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("cannot read input");
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn run<A: AssignRows<f32>>(x: &[f32], d: usize, k: usize, cfg: &Config, a: &mut A, out: &str) {
    let t = Instant::now();
    let res = emeans::train(x, d, k, cfg, a).expect("training failed");
    let train_s = t.elapsed().as_secs_f64();
    let (_, dist) = emeans::assign(x, &res.centroids, d, a).expect("assignment failed");
    let mse = dist.iter().map(|&v| v as f64).sum::<f64>() / dist.len() as f64;
    let bytes: Vec<u8> = res.centroids.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out, bytes).expect("cannot write output");
    println!(
        "{} | train {train_s:.2} s | batch {} | alpha0 {:.4} | revived {} | final MSE {mse:.4} | wrote {out}",
        a.name(),
        res.recipe.batch_size,
        res.recipe.alpha0,
        res.n_splits
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: cluster <vectors.f32> <d> <k> [--epochs T] [--out FILE] [--simd LEVEL] [--gpu [N]]");
        std::process::exit(2);
    }
    let (d, k): (usize, usize) = (args[2].parse().unwrap(), args[3].parse().unwrap());
    let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let cfg = Config { nepochs: opt("--epochs").map_or(5.0, |v| v.parse().unwrap()), verbose: true, ..Default::default() };
    let out = opt("--out").unwrap_or_else(|| "centroids.f32".into());
    let x = read_f32(&args[1]);

    if args.iter().any(|a| a == "--gpu") {
        #[cfg(feature = "cuda")]
        {
            let ordinal = opt("--gpu").and_then(|v| v.parse().ok()).unwrap_or(0);
            let mut a = emeans::cuda::CudaAssigner::new(ordinal).expect("CUDA init failed");
            return run(&x, d, k, &cfg, &mut a, &out);
        }
        #[cfg(not(feature = "cuda"))]
        panic!("rebuild with --features cuda");
    }
    let simd = match opt("--simd").as_deref() {
        Some("avx512") => cpu::Simd::Avx512,
        Some("avx2") => cpu::Simd::Avx2,
        Some("portable") => cpu::Simd::Portable,
        _ => cpu::Simd::detect(),
    };
    run(&x, d, k, &cfg, &mut cpu::CpuAssigner::with_simd(simd), &out);
}
