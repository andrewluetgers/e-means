//! Per-cluster statistics, outlier scores and prototype families, computed
//! from a finished clustering.
//!
//! Inputs are what [`crate::assign`] returns (each row's label and squared
//! distance) plus the centroids, so the vectors themselves are not needed.
//!
//! Where outliers show up, and which field finds them:
//!
//! | Kind | Looks like | Field |
//! |---|---|---|
//! | Fringe point | far from its centroid compared with its cluster-mates | `local_z` |
//! | Point in a catch-all cluster | far from its centroid compared with the corpus | `global_z` |
//! | Catch-all cluster | unrelated rare things lumped together: large spread | `looseness` |
//! | Rare prototype | a small cluster far from every other centroid | `size`, `isolation` |
//! | Rare family | a region of space holding little of the corpus | `Family::share` |

use rayon::prelude::*;
use std::io::Write;

#[derive(Clone, Debug)]
pub struct ReportConfig {
    /// Centroids within `family_radius` x (median nearest-centroid distance)
    /// of a family's leader join that family. Larger: fewer, broader families.
    pub family_radius: f32,
    /// Clusters smaller than this use corpus-wide distance statistics for
    /// `local_z`, since their own median and spread are unreliable.
    pub min_local_size: usize,
}

impl Default for ReportConfig {
    fn default() -> Self {
        ReportConfig { family_radius: 1.25, min_local_size: 30 }
    }
}

/// Distances here are plain L2 (not squared).
#[derive(Clone, Debug, Default)]
pub struct ClusterStats {
    pub size: u32,
    /// Root mean squared distance of members to the centroid.
    pub rms: f32,
    pub median: f32,
    /// Median absolute deviation of member distances.
    pub mad: f32,
    pub p90: f32,
    pub p99: f32,
    /// Farthest member: the cluster's covering radius.
    pub max: f32,
    /// `rms` / corpus median `rms`. Well above 1: a loose, catch-all cluster.
    pub looseness: f32,
    pub nearest_centroid: u32,
    pub nearest_dist: f32,
    /// `nearest_dist` / corpus median. Well above 1: an isolated prototype.
    pub isolation: f32,
    pub family: u32,
}

#[derive(Clone, Debug)]
pub struct Family {
    /// The largest member cluster, around which the family was formed.
    pub leader: u32,
    pub members: Vec<u32>,
    pub size: u64,
    /// Fraction of the corpus in this family: its prevalence.
    pub share: f64,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub clusters: Vec<ClusterStats>,
    /// Sorted by decreasing size.
    pub families: Vec<Family>,
    /// Robust z-score of each row's distance within its own cluster:
    /// `(dist - cluster median) / (1.4826 * cluster MAD)`.
    pub local_z: Vec<f32>,
    /// The same z-score against the whole corpus' distances.
    pub global_z: Vec<f32>,
    pub median_dist: f32,
    pub median_rms: f32,
    pub median_nearest_dist: f32,
}

fn quantile(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// Median and MAD of an ascending slice.
fn median_mad(sorted: &[f32]) -> (f32, f32) {
    let med = quantile(sorted, 0.5);
    let mut dev: Vec<f32> = sorted.iter().map(|v| (v - med).abs()).collect();
    dev.sort_unstable_by(f32::total_cmp);
    (med, quantile(&dev, 0.5))
}

fn median_of(values: impl Iterator<Item = f32>) -> f32 {
    let mut v: Vec<f32> = values.collect();
    v.sort_unstable_by(f32::total_cmp);
    quantile(&v, 0.5)
}

fn sqdist(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| (x - y) * (x - y)).sum();
    for (x, y) in ca.zip(cb) {
        for i in 0..16 {
            acc[i] += (x[i] - y[i]) * (x[i] - y[i]);
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// Build the report. `labels` and `sqdist_to_centroid` are per row, as returned
/// by [`crate::assign`]; `centroids` is `k x d` row-major.
///
/// Cost beyond sorting: `O(k^2 d)` for nearest-centroid distances, fine up to
/// tens of thousands of centroids.
pub fn report(labels: &[u32], sqdist_to_centroid: &[f32], centroids: &[f32], d: usize, cfg: &ReportConfig) -> Report {
    assert_eq!(labels.len(), sqdist_to_centroid.len());
    let (n, k) = (labels.len(), centroids.len() / d);
    let dist: Vec<f32> = sqdist_to_centroid.iter().map(|v| v.max(0.0).sqrt()).collect();

    // Group member distances by cluster (counting sort), then sort each group.
    let mut start = vec![0usize; k + 1];
    labels.iter().for_each(|&l| start[l as usize + 1] += 1);
    for j in 0..k {
        start[j + 1] += start[j];
    }
    let mut cursor = start.clone();
    let mut grouped = vec![0f32; n];
    for (&l, &v) in labels.iter().zip(&dist) {
        grouped[cursor[l as usize]] = v;
        cursor[l as usize] += 1;
    }

    let mut all = dist.clone();
    all.par_sort_unstable_by(f32::total_cmp);
    let (g_med, g_mad) = median_mad(&all);
    drop(all);
    let g_scale = (1.4826 * g_mad).max(f32::MIN_POSITIVE);

    let mut clusters: Vec<ClusterStats> = (0..k)
        .into_par_iter()
        .map(|j| {
            let mut m = grouped[start[j]..start[j + 1]].to_vec();
            if m.is_empty() {
                return ClusterStats::default();
            }
            m.sort_unstable_by(f32::total_cmp);
            let (median, mad) = median_mad(&m);
            let rms = (m.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / m.len() as f64).sqrt() as f32;
            ClusterStats {
                size: m.len() as u32,
                rms,
                median,
                mad,
                p90: quantile(&m, 0.9),
                p99: quantile(&m, 0.99),
                max: *m.last().unwrap(),
                ..Default::default()
            }
        })
        .collect();
    drop(grouped);

    // Nearest other centroid, among non-empty clusters.
    let nearest: Vec<(u32, f32)> = (0..k)
        .into_par_iter()
        .map(|i| {
            let ci = &centroids[i * d..(i + 1) * d];
            let mut best = (i as u32, f32::INFINITY);
            for j in (0..k).filter(|&j| j != i && clusters[j].size > 0) {
                let s = sqdist(ci, &centroids[j * d..(j + 1) * d]);
                if s < best.1 {
                    best = (j as u32, s);
                }
            }
            (best.0, best.1.sqrt())
        })
        .collect();
    let live = |c: &ClusterStats| c.size > 0;
    let median_rms = median_of(clusters.iter().filter(|c| live(c)).map(|c| c.rms));
    let median_nearest = median_of((0..k).filter(|&j| clusters[j].size > 0).map(|j| nearest[j].1));
    for (c, &(nj, nd)) in clusters.iter_mut().zip(&nearest) {
        c.nearest_centroid = nj;
        c.nearest_dist = nd;
        c.looseness = c.rms / median_rms.max(f32::MIN_POSITIVE);
        c.isolation = nd / median_nearest.max(f32::MIN_POSITIVE);
    }

    // Families: fixed-radius balls around leaders, largest clusters first. A
    // fixed radius makes a family a region of space rather than a share of the
    // data, so its size is a prevalence: a dense region split over many
    // centroids becomes one large family, a rare one stays small.
    let radius = cfg.family_radius * median_nearest;
    let mut order: Vec<usize> = (0..k).filter(|&j| clusters[j].size > 0).collect();
    order.sort_unstable_by_key(|&j| std::cmp::Reverse(clusters[j].size));
    let mut family_of = vec![u32::MAX; k];
    let mut families: Vec<Family> = Vec::new();
    for &lead in &order {
        if family_of[lead] != u32::MAX {
            continue;
        }
        let cl = &centroids[lead * d..(lead + 1) * d];
        let joins: Vec<u32> = order
            .par_iter()
            .filter(|&&j| family_of[j] == u32::MAX && (j == lead || sqdist(cl, &centroids[j * d..(j + 1) * d]).sqrt() <= radius))
            .map(|&j| j as u32)
            .collect();
        joins.iter().for_each(|&j| family_of[j as usize] = families.len() as u32);
        let size: u64 = joins.iter().map(|&j| clusters[j as usize].size as u64).sum();
        families.push(Family { leader: lead as u32, members: joins, size, share: size as f64 / n as f64 });
    }
    // Largest family first; renumber.
    let mut rank: Vec<usize> = (0..families.len()).collect();
    rank.sort_unstable_by_key(|&f| std::cmp::Reverse(families[f].size));
    let families: Vec<Family> = rank.iter().map(|&f| families[f].clone()).collect();
    for (f, fam) in families.iter().enumerate() {
        fam.members.iter().for_each(|&j| clusters[j as usize].family = f as u32);
    }

    let global_z: Vec<f32> = dist.par_iter().map(|&v| (v - g_med) / g_scale).collect();
    let local_z: Vec<f32> = dist
        .par_iter()
        .zip(labels)
        .zip(&global_z)
        .map(|((&v, &l), &gz)| {
            let c = &clusters[l as usize];
            if (c.size as usize) < cfg.min_local_size || c.mad <= 0.0 {
                gz
            } else {
                (v - c.median) / (1.4826 * c.mad)
            }
        })
        .collect();

    Report { clusters, families, local_z, global_z, median_dist: g_med, median_rms, median_nearest_dist: median_nearest }
}

impl Report {
    /// Each row's outlier score: the larger of its two z-scores, so both
    /// fringe points of tight clusters and members of catch-all clusters rank high.
    pub fn score(&self, row: usize) -> f32 {
        self.local_z[row].max(self.global_z[row])
    }

    /// The `count` highest-scoring rows, as `(row, score)`, best first.
    pub fn top_outliers(&self, count: usize) -> Vec<(u32, f32)> {
        let mut rows: Vec<(u32, f32)> = (0..self.local_z.len()).map(|i| (i as u32, self.score(i))).collect();
        let count = count.min(rows.len());
        if count < rows.len() {
            rows.select_nth_unstable_by(count, |a, b| b.1.total_cmp(&a.1));
            rows.truncate(count);
        }
        rows.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        rows
    }

    /// Clusters that are both small and far from every other centroid:
    /// candidate rare prototypes. `(cluster, size, isolation)`, most isolated first.
    pub fn rare_prototypes(&self, max_size_ratio: f32, min_isolation: f32) -> Vec<(u32, u32, f32)> {
        let live = self.clusters.iter().filter(|c| c.size > 0).count().max(1);
        let mean = self.local_z.len() as f32 / live as f32;
        let mut v: Vec<(u32, u32, f32)> = self
            .clusters
            .iter()
            .enumerate()
            .filter(|(_, c)| c.size > 0 && (c.size as f32) <= max_size_ratio * mean && c.isolation >= min_isolation)
            .map(|(j, c)| (j as u32, c.size, c.isolation))
            .collect();
        v.sort_unstable_by(|a, b| b.2.total_cmp(&a.2));
        v
    }

    /// One row per cluster, tab-separated, with a header line.
    pub fn write_clusters_tsv<W: Write>(&self, mut w: W) -> std::io::Result<()> {
        writeln!(w, "cluster\tsize\trms\tmedian\tmad\tp90\tp99\tmax\tlooseness\tnearest_centroid\tnearest_dist\tisolation\tfamily\tfamily_share")?;
        for (j, c) in self.clusters.iter().enumerate() {
            let share = if c.size > 0 { self.families[c.family as usize].share } else { 0.0 };
            writeln!(
                w,
                "{j}\t{}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.5}\t{:.3}\t{}\t{:.5}\t{:.3}\t{}\t{:.6}",
                c.size, c.rms, c.median, c.mad, c.p90, c.p99, c.max, c.looseness, c.nearest_centroid, c.nearest_dist, c.isolation, c.family, share
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_fringe_point_and_a_rare_prototype() {
        // 1-D: two big clusters at 0 and 1 (touching), one tiny cluster at 50.
        let centroids = [0.0f32, 1.0, 50.0];
        let (mut labels, mut sq) = (Vec::new(), Vec::new());
        for i in 0..200 {
            let off = 0.001 * (i % 100) as f32; // members within 0.1 of their centroid
            labels.push((i % 2) as u32);
            sq.push(off * off);
        }
        labels.push(0);
        sq.push(0.45 * 0.45); // a fringe point of cluster 0: row 200
        for _ in 0..3 {
            labels.push(2);
            sq.push(1e-4);
        }
        let r = report(&labels, &sq, &centroids, 1, &ReportConfig::default());
        assert_eq!(r.clusters.iter().map(|c| c.size).collect::<Vec<_>>(), vec![101, 100, 3]);
        assert_eq!(r.top_outliers(1)[0].0, 200);
        assert!(r.local_z[200] > 10.0);
        // Clusters 0 and 1 are one family; the far cluster is its own, and rare.
        assert_eq!(r.families.len(), 2);
        assert_eq!(r.families[0].size, 201);
        assert_eq!(r.clusters[0].family, r.clusters[1].family);
        assert_eq!(r.rare_prototypes(0.25, 2.0), vec![(2, 3, 49.0)]);
        let mut tsv = Vec::new();
        r.write_clusters_tsv(&mut tsv).unwrap();
        assert_eq!(tsv.iter().filter(|&&b| b == b'\n').count(), 4);
    }
}
