//! Sweep the balance penalty and report the MSE / balance trade-off.
//!
//!   cargo run --release --example balance -- vectors.f32 <d> <k> [--epochs 5]
//!
//! "nearest" re-assigns every row to its plain nearest centroid, which is what
//! an IVF index does at build and query time. "penalized" uses the labels from
//! training's last pass, which honour the penalty.

use emeans::{balance_stats, cpu::CpuAssigner, Config};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (d, k): (usize, usize) = (args[2].parse().unwrap(), args[3].parse().unwrap());
    let epochs = args.iter().position(|a| a == "--epochs").map_or(5.0, |i| args[i + 1].parse().unwrap());
    let bytes = std::fs::read(&args[1]).expect("cannot read input");
    let x: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let n = x.len() / d;

    let seeds: u64 = args.iter().position(|a| a == "--seeds").map_or(5, |i| args[i + 1].parse().unwrap());
    println!("{n} x {d}, k = {k}, {epochs} epochs, mean over {seeds} seeds (sd of MSE in brackets)");
    println!("lambda | nearest: MSE [sd]    vs 0   imbal  max/mean | penalized: MSE   vs 0   imbal  max/mean  empty");
    let mut base = 0f64;
    for lambda in [0.0f32, 0.03, 0.1, 0.3, 1.0, 3.0] {
        let mut acc = [0f64; 8];
        let mut sq = 0f64;
        for seed in 0..seeds {
            let cfg = Config { nepochs: epochs, balance: lambda, seed: 1234 + seed, ..Default::default() };
            let out = emeans::train_cpu(&x, d, k, &cfg).unwrap();
            let (near, dist) = emeans::assign(&x, &out.centroids, d, &mut CpuAssigner::new()).unwrap();
            let mse_near = dist.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            let mse_pen = x
                .chunks_exact(d)
                .zip(&out.labels)
                .map(|(row, &l)| {
                    let c = &out.centroids[l as usize * d..][..d];
                    row.iter().zip(c).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>()
                })
                .sum::<f64>()
                / n as f64;
            let (ni, nm, _) = balance_stats(&near, k);
            let (pi, pm, pe) = balance_stats(&out.labels, k);
            for (a, v) in acc.iter_mut().zip([mse_near, ni, nm, mse_pen, pi, pm, pe as f64]) {
                *a += v / seeds as f64;
            }
            sq += mse_near * mse_near / seeds as f64;
        }
        let sd = (sq - acc[0] * acc[0]).max(0.0).sqrt();
        if lambda == 0.0 {
            base = acc[0];
        }
        println!(
            "{lambda:6.2} | {:10.3} [{sd:5.2}] {:+6.2}%  {:5.2}  {:8.1} | {:14.3} {:+6.2}%  {:5.2}  {:8.1}  {:5.1}",
            acc[0],
            100.0 * (acc[0] / base - 1.0),
            acc[1],
            acc[2],
            acc[3],
            100.0 * (acc[3] / base - 1.0),
            acc[4],
            acc[5],
            acc[6],
        );
    }
}
