//! Characterize int8 clustering against f32 on the same data.
//!
//!   cargo run --release --example int8 -- vectors.f32 <d> <k> [--epochs 5] [--seeds 3]
//!
//! Every row reports MSE in the original units, from an exact f32 assignment
//! of the original vectors to the (decoded) centroids.

use emeans::cpu::CpuAssigner;
use emeans::cpu_i8::{CentroidBits, CpuI8Assigner};
use emeans::quantize::{Calibration, Int8Quantizer};
use emeans::Config;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (d, k): (usize, usize) = (args[2].parse().unwrap(), args[3].parse().unwrap());
    let num = |name: &str, default: f64| args.iter().position(|a| a == name).map_or(default, |i| args[i + 1].parse().unwrap());
    let (epochs, seeds) = (num("--epochs", 5.0), num("--seeds", 3.0) as u64);
    let bytes = std::fs::read(&args[1]).expect("cannot read input");
    let x: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let n = x.len() / d;
    let mse = |centroids: &[f32]| {
        let (_, dist) = emeans::assign(&x, centroids, d, &mut CpuAssigner::new()).unwrap();
        dist.iter().map(|&v| v as f64).sum::<f64>() / n as f64
    };
    println!("{n} x {d}, k = {k}, {epochs} epochs, mean of {seeds} seeds");
    println!("{:<46} {:>8} {:>12} {:>9}", "configuration", "train s", "MSE", "vs f32");

    let mut base = 0f64;
    let mut row = |name: &str, run: &mut dyn FnMut(&Config) -> Vec<f32>| {
        let (mut secs, mut err) = (0f64, 0f64);
        for seed in 0..seeds {
            let cfg = Config { nepochs: epochs, seed: 1234 + seed, ..Default::default() };
            let t = Instant::now();
            let centroids = run(&cfg);
            secs += t.elapsed().as_secs_f64() / seeds as f64;
            err += mse(&centroids) / seeds as f64;
        }
        if base == 0.0 {
            base = err;
        }
        println!("{name:<46} {secs:>8.2} {err:>12.6} {:>+8.3}%", 100.0 * (err / base - 1.0));
    };

    row("f32 rows, f32 search", &mut |cfg| emeans::train_cpu(&x, d, k, cfg).unwrap().centroids);
    for (label, cal) in [("faiss calibration", Calibration::faiss()), ("clipped at 99.99%", Calibration::clipped(0.9999))] {
        let q = Int8Quantizer::fit(&x, d, cal);
        let codes = q.encode(&x);
        println!("-- int8 codes, {label}: scale {:.4}, {:.4}% of values clipped", q.scale, 100.0 * q.clipped_fraction(&x));
        row("  f32 search (exact on the codes)", &mut |cfg| q.decode(&emeans::train_cpu(&codes, d, k, cfg).unwrap().centroids));
        row("  int16-centroid search (VNNI)", &mut |cfg| {
            q.decode(&emeans::train(&codes, d, k, cfg, &mut CpuI8Assigner::new(CentroidBits::I16)).unwrap().centroids)
        });
        row("  int8-centroid search (VNNI, as faiss)", &mut |cfg| {
            q.decode(&emeans::train(&codes, d, k, cfg, &mut CpuI8Assigner::new(CentroidBits::I8)).unwrap().centroids)
        });
    }
}
