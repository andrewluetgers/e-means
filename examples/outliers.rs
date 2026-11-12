//! Cluster a raw f32 file, then report clusters, families and outliers.
//!
//!   cargo run --release --example outliers -- vectors.f32 <d> <k> [--epochs 5]
//!       [--balance 0] [--top 1000] [--family-radius 1.25] [--prefix out]
//!
//! Writes <prefix>.clusters.tsv (one row per cluster) and <prefix>.outliers.tsv
//! (the top-scoring rows) and <prefix>.labels.u32 (each row's cluster), and
//! prints a summary.

use emeans::{cpu::CpuAssigner, report, Config};
use std::io::{BufWriter, Write};

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (d, k): (usize, usize) = (args[2].parse().unwrap(), args[3].parse().unwrap());
    let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let num = |name: &str, default: f64| opt(name).map_or(default, |v| v.parse().unwrap());
    let prefix = opt("--prefix").unwrap_or_else(|| "out".into());
    let top = num("--top", 1000.0) as usize;
    let bytes = std::fs::read(&args[1])?;
    let x: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let n = x.len() / d;

    let cfg = Config { nepochs: num("--epochs", 5.0), balance: num("--balance", 0.0) as f32, ..Default::default() };
    let out = emeans::train_cpu(&x, d, k, &cfg).expect("training failed");
    let (labels, sq) = emeans::assign(&x, &out.centroids, d, &mut CpuAssigner::new()).expect("assign failed");
    let rcfg = report::ReportConfig { family_radius: num("--family-radius", 1.25) as f32, ..Default::default() };
    let r = report::report(&labels, &sq, &out.centroids, d, &rcfg);

    println!("{n} rows, {k} clusters, {} families (radius {} x median centroid spacing {:.3})", r.families.len(), rcfg.family_radius, r.median_nearest_dist);
    println!("median distance to centroid {:.3}, median cluster rms {:.3}", r.median_dist, r.median_rms);

    println!("\nLargest families (prevalence):");
    for f in r.families.iter().take(5) {
        println!("  {:6.2}% of rows in {:4} clusters (leader {})", 100.0 * f.share, f.members.len(), f.leader);
    }
    let singles = r.families.iter().filter(|f| f.members.len() == 1).count();
    let rare = r.families.iter().filter(|f| f.share < 0.1 / k as f64).count();
    println!("  {singles} single-cluster families; {rare} families hold under a tenth of an average cluster");

    let mut loose: Vec<usize> = (0..k).filter(|&j| r.clusters[j].size > 0).collect();
    loose.sort_unstable_by(|&a, &b| r.clusters[b].looseness.total_cmp(&r.clusters[a].looseness));
    println!("\nLoosest clusters (catch-alls):");
    for &j in loose.iter().take(5) {
        let c = &r.clusters[j];
        println!("  cluster {j:6}: size {:6}, looseness {:.2}, rms {:.3}, max {:.3}", c.size, c.looseness, c.rms, c.max);
    }

    println!("\nSmall, isolated clusters (rare prototypes):");
    for (j, size, iso) in r.rare_prototypes(0.25, 1.5).into_iter().take(5) {
        println!("  cluster {j:6}: size {size:6}, isolation {iso:.2}");
    }

    let z5 = (0..n).filter(|&i| r.score(i) > 5.0).count();
    println!("\n{z5} rows score above 5 ({:.3}%). Top rows:", 100.0 * z5 as f64 / n as f64);
    let best = r.top_outliers(top);
    for &(i, s) in best.iter().take(5) {
        let i = i as usize;
        println!("  row {i:9}: score {s:7.2} (local {:.2}, global {:.2}), cluster {}", r.local_z[i], r.global_z[i], labels[i]);
    }

    r.write_clusters_tsv(BufWriter::new(std::fs::File::create(format!("{prefix}.clusters.tsv"))?))?;
    let mut w = BufWriter::new(std::fs::File::create(format!("{prefix}.outliers.tsv"))?);
    writeln!(w, "row\tscore\tlocal_z\tglobal_z\tdist\tcluster\tcluster_size\tfamily_share")?;
    for &(i, s) in &best {
        let i = i as usize;
        let c = &r.clusters[labels[i] as usize];
        writeln!(w, "{i}\t{s:.3}\t{:.3}\t{:.3}\t{:.5}\t{}\t{}\t{:.6}", r.local_z[i], r.global_z[i], sq[i].sqrt(), labels[i], c.size, r.families[c.family as usize].share)?;
    }
    let label_bytes: Vec<u8> = labels.iter().flat_map(|l| l.to_le_bytes()).collect();
    std::fs::write(format!("{prefix}.labels.u32"), label_bytes)?;
    println!("\nwrote {prefix}.clusters.tsv, {prefix}.outliers.tsv and {prefix}.labels.u32");
    Ok(())
}
