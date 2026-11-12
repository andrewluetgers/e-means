//! e-means clustering: mini-batch k-means with exponential-moving-average
//! centroid updates, automatic batch/step scaling laws, dead-cluster revival
//! and an exact last-epoch closing mean.
//!
//! Algorithm: Bessa et al., "e-means: outperforming k-means (in one-tenth the
//! epochs)", arXiv:2611.09501. This is a Rust port of the FP32 reference
//! (`faiss.contrib.e_means.NumpyEmeans`, MIT), using the constants that
//! reference ships.
//!
//! ```no_run
//! let (d, k) = (96, 1024);
//! let x: Vec<f32> = vec![0.0; 200_000 * d]; // row-major, n * d
//! let out = emeans::train_cpu(&x, d, k, &emeans::Config::default()).unwrap();
//! assert_eq!(out.centroids.len(), k * d);
//! ```
//!
//! The nearest-centroid search is behind the [`Assigner`] trait:
//! [`cpu::CpuAssigner`] (AVX-512 / AVX2+FMA / portable, picked at runtime) and,
//! with the `cuda` feature, [`cuda::CudaAssigner`].

pub mod cpu;
pub mod cpu_i8;
pub mod quantize;
pub mod report;
#[cfg(feature = "cuda")]
pub mod cuda;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

// Constants of the reference implementation (faiss/contrib/e_means/base.py).
const ALPHA_MIN_FACTOR: f64 = 0.1; // cosine floor: final step / initial step
const CHECK_UNUSED_EVERY: usize = 5; // batch updates between revival checks
const INIT_USAGE: f32 = 1.0 / 65536.0; // initial EMA mass N_j
const SPLIT_EPSILON: f32 = 1e-4; // relative perturbation of a split
const UNUSED_THRESHOLD: f32 = 0.1; // revive N_j < tau * mean(N)
const NO_REVIVAL_MAX_D: usize = 2; // revival is off at or below this dimension
const AUTO_QUALITY_MAX_D: usize = 16; // "auto" picks the quality law up to here
const BATCH_THROUGHPUT: [f64; 4] = [0.035134, 0.134749, 0.876473, 0.813344];
const BATCH_QUALITY: [f64; 4] = [0.00439175, 0.134749, 0.876473, 0.813344];
const ALPHA_LAW: [f64; 5] = [0.8244, -0.4617, 0.9003, -0.4498, -0.1057];
/// Rows gathered and searched at once (memory/speed only, not the batch size).
const ASSIGN_CHUNK: usize = 65536;

/// Element type of the input rows: `f32`, or `i8` / `u8` for quantized
/// embeddings (a quarter of the memory). Centroids are always `f32`.
pub trait Scalar: Copy + Send + Sync + 'static {
    fn to_f32(self) -> f32;
    /// The slice itself when the rows are already `f32` (avoids a copy).
    fn as_f32(_rows: &[Self]) -> Option<&[f32]> {
        None
    }
}
impl Scalar for f32 {
    fn to_f32(self) -> f32 {
        self
    }
    fn as_f32(rows: &[Self]) -> Option<&[f32]> {
        Some(rows)
    }
}
impl Scalar for i8 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl Scalar for u8 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

/// A nearest-centroid search backend: the centroid side. The row side is
/// [`AssignRows`], implemented per supported row type.
pub trait Assigner {
    /// Load a `k x d` row-major centroid table for the following `assign` calls.
    fn set_centroids(&mut self, centroids: &[f32], k: usize, d: usize) -> Result<()>;
    /// Add `bias[j]` to centroid j's score for the following `assign` calls:
    /// rows then go to the centroid minimizing `|x - c_j|^2 + bias[j]`, and the
    /// returned distance includes that bias. Reset by `set_centroids`.
    fn set_bias(&mut self, bias: &[f32]) -> Result<()>;
    /// Human-readable description of the kernel in use.
    fn name(&self) -> String;
}

/// Nearest-centroid search over rows of type `T`.
pub trait AssignRows<T: Scalar>: Assigner {
    /// For each row of `x` (row-major, `labels.len() x d`): the index of the
    /// nearest centroid and its squared L2 distance. Ties go to the lowest index.
    fn assign(&mut self, x: &[T], labels: &mut [u32], dist: &mut [f32]) -> Result<()>;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BatchLaw {
    /// Quality for `d <= 16`, throughput otherwise.
    Auto,
    /// Larger batches, fastest training.
    Throughput,
    /// About 8x smaller batches: lower final MSE, more updates.
    Quality,
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Passes over the data; may be fractional (e.g. 2.5).
    pub nepochs: f64,
    pub batch_law: BatchLaw,
    /// Overrides the batch law; the step law then adapts `alpha0` to it.
    pub batch_size: Option<usize>,
    /// Overrides the step law.
    pub alpha0: Option<f64>,
    /// Revival threshold tau. `None`: 0.1, or 0 (off) for `d <= 2`.
    pub unused_threshold: Option<f32>,
    /// Cluster-size balance penalty lambda (0 = off, plain e-means). During
    /// training a row goes to the centroid minimizing
    /// `|x - c_j|^2 + lambda * s * N_j / mean(N)`, where `N_j` is the cluster's
    /// running mass and `s` the current mean squared distance, so lambda is
    /// dimensionless: at 1.0 a cluster twice the mean size costs one typical
    /// squared distance more than an average one. This is the per-point form
    /// of adding `lambda * s * k/n * sum_j n_j^2` to the objective.
    pub balance: f32,
    pub seed: u64,
    pub verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            nepochs: 5.0,
            batch_law: BatchLaw::Auto,
            batch_size: None,
            alpha0: None,
            unused_threshold: None,
            balance: 0.0,
            seed: 1234,
            verbose: false,
        }
    }
}

/// The resolved training plan.
#[derive(Clone, Debug)]
pub struct Recipe {
    pub batch_size: usize,
    pub alpha0: f64,
    pub epoch_rows: Vec<usize>,
    pub total_steps: usize,
    pub unused_threshold: f32,
}

#[derive(Clone, Debug)]
pub struct Output {
    /// `k x d`, row-major.
    pub centroids: Vec<f32>,
    /// Each row's label from the last pass (the centroids were still moving, so
    /// call [`assign`] for exact nearest-centroid labels).
    pub labels: Vec<u32>,
    /// Sum of squared distances seen during each pass (moving centroids).
    pub objective: Vec<f64>,
    /// Number of dead clusters revived.
    pub n_splits: usize,
    pub recipe: Recipe,
}

fn round_power_of_two(v: f64) -> usize {
    if v <= 1.0 {
        return 1;
    }
    let low = 2f64.powi(v.log2().floor() as i32);
    let high = 2.0 * low;
    (if v / low < high / v { low } else { high }) as usize
}

fn step_law(n: usize, k: usize, nepochs: f64, batch: usize) -> f64 {
    let [c, ak, ab, an, at] = ALPHA_LAW;
    let a = c * (k as f64).powf(ak) * (batch as f64).powf(ab) * (n as f64).powf(an) * nepochs.powf(at);
    a.clamp(1e-4, 1.0)
}

fn epoch_rows(n: usize, nepochs: f64) -> Vec<usize> {
    let whole = nepochs.floor() as usize;
    let rem = ((nepochs - whole as f64) * n as f64).round_ties_even() as usize;
    if whole == 0 {
        return vec![rem];
    }
    let mut v = vec![n; whole - 1];
    if rem > 0 {
        v.push(rem);
    }
    v.push(n);
    v
}

fn scheduled_alpha(alpha0: f64, step: usize, total: usize) -> f64 {
    let frac = step.min(total) as f64 / total.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * frac).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

/// Resolve batch size, initial step and the pass plan for a problem size.
pub fn resolve_recipe(n: usize, d: usize, k: usize, cfg: &Config) -> Recipe {
    let law = match cfg.batch_law {
        BatchLaw::Auto if d <= AUTO_QUALITY_MAX_D => BatchLaw::Quality,
        BatchLaw::Auto => BatchLaw::Throughput,
        l => l,
    };
    let batch_size = match cfg.batch_size {
        Some(b) => b.clamp(1, n),
        None => {
            let [c, bk, bn, bt] = if law == BatchLaw::Throughput { BATCH_THROUGHPUT } else { BATCH_QUALITY };
            let raw = c * (k as f64).powf(bk) * (n as f64).powf(bn) * cfg.nepochs.powf(bt);
            let b = round_power_of_two(raw);
            // The two fitted recipes cap differently, as in the reference.
            let cap = if law == BatchLaw::Throughput { round_power_of_two(n as f64 / 2.0) } else { n / 2 };
            b.min(cap).min(n).max(1)
        }
    };
    let alpha0 = cfg.alpha0.unwrap_or_else(|| step_law(n, k, cfg.nepochs, batch_size));
    let epoch_rows = epoch_rows(n, cfg.nepochs);
    let total_steps = epoch_rows.iter().map(|r| r.div_ceil(batch_size)).sum();
    let unused_threshold = cfg
        .unused_threshold
        .unwrap_or(if d <= NO_REVIVAL_MAX_D { 0.0 } else { UNUSED_THRESHOLD });
    Recipe { batch_size, alpha0, epoch_rows, total_steps, unused_threshold }
}

/// xoshiro256++, seeded through splitmix64. Small, fast, dependency-free.
struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        Rng([next(), next(), next(), next()])
    }
    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let r = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }
    /// Uniform in [0, bound), unbiased (Lemire).
    fn below(&mut self, bound: u64) -> u64 {
        loop {
            let m = (self.next_u64() as u128) * (bound as u128);
            if (m as u64) >= bound.wrapping_neg() % bound {
                return (m >> 64) as u64;
            }
        }
    }
    /// Uniform in (0, 1).
    fn open01(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn permutation(&mut self, n: usize) -> Vec<u32> {
        let mut p: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            p.swap(i, self.below(i as u64 + 1) as usize);
        }
        p
    }
}

struct State {
    k: usize,
    d: usize,
    centroids: Vec<f32>, // c_j = S_j / N_j
    usage: Vec<f32>,     // N_j
    n_splits: usize,
}

impl State {
    /// EMA update (Algorithm 1, line 8) from one batch's counts and sums, in
    /// the mass/position form: N' = (1-a) N + a n,  c' = c + (a n / N') (s/n - c).
    fn apply(&mut self, touched: &[u32], counts: &[u32], sums: &[f32], alpha: f32) {
        let d = self.d;
        let replace = alpha == 1.0;
        if !replace {
            let decay = 1.0 - alpha;
            self.usage.iter_mut().for_each(|u| *u *= decay);
        }
        for &j in touched {
            let j = j as usize;
            let n = counts[j] as f32;
            let retained = if replace { 0.0 } else { self.usage[j] };
            let mass = retained + alpha * n;
            let weight = alpha * n / mass;
            let c = &mut self.centroids[j * d..(j + 1) * d];
            let s = &sums[j * d..(j + 1) * d];
            if weight == 1.0 {
                c.iter_mut().zip(s).for_each(|(c, s)| *c = s / n);
            } else {
                c.iter_mut().zip(s).for_each(|(c, s)| *c += weight * (s / n - *c));
            }
            self.usage[j] = mass;
        }
    }

    /// Revive clusters whose mass fell below tau * mean mass by splitting donor
    /// clusters, sampled without replacement in proportion to their mass
    /// (Gumbel top-k). Donor and child each keep half the donor's mass.
    fn revive(&mut self, tau: f32, rng: &mut Rng) {
        let (k, d) = (self.k, self.d);
        let total: f32 = self.usage.iter().sum();
        let threshold = total * tau / k as f32;
        let unused: Vec<usize> = (0..k).filter(|&j| self.usage[j] < threshold).collect();
        if unused.is_empty() {
            return;
        }
        let mut donors: Vec<(f64, usize)> = (0..k)
            .filter(|&j| self.usage[j] >= threshold && self.usage[j] > 0.0)
            .map(|j| ((self.usage[j] as f64).ln() - (-rng.open01().ln()).ln(), j))
            .collect();
        let nsplit = unused.len().min(donors.len());
        donors.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        for (&(_, src), &dst) in donors[..nsplit].iter().zip(&unused) {
            let half = self.usage[src] / 2.0;
            for i in 0..d {
                let offset = if i % 2 == 0 { SPLIT_EPSILON } else { -SPLIT_EPSILON };
                let c = self.centroids[src * d + i];
                self.centroids[src * d + i] = c * (1.0 - offset);
                self.centroids[dst * d + i] = c * (1.0 + offset);
            }
            // A zero donor would give an identical child: nudge one coordinate.
            if (0..d).all(|i| self.centroids[src * d + i] == self.centroids[dst * d + i]) {
                self.centroids[dst * d] = self.centroids[dst * d].next_up();
            }
            self.usage[src] = half;
            self.usage[dst] = half;
        }
        self.n_splits += nsplit;
    }
}

/// Train e-means on `x` (row-major `n x d`) with the given assignment backend.
pub fn train<T: Scalar, A: AssignRows<T>>(x: &[T], d: usize, k: usize, cfg: &Config, assigner: &mut A) -> Result<Output> {
    if d == 0 || k == 0 || x.len() % d != 0 {
        return Err("x must be a non-empty row-major n x d array".into());
    }
    let n = x.len() / d;
    if n < k {
        return Err(format!("need at least k = {k} rows, got {n}").into());
    }
    if n > u32::MAX as usize {
        return Err("more than 2^32 - 1 rows: subsample or shard the input".into());
    }
    if !(cfg.nepochs > 0.0) {
        return Err("nepochs must be positive".into());
    }
    let recipe = resolve_recipe(n, d, k, cfg);
    let batch = recipe.batch_size;
    let t0 = std::time::Instant::now();
    if cfg.verbose {
        eprintln!(
            "e-means: {n} points in {d}D to {k} clusters, {} epochs (batch {batch}, alpha0 {:.3e}), {}",
            cfg.nepochs,
            recipe.alpha0,
            assigner.name()
        );
    }

    // Random data vectors as starting centroids.
    let init = Rng::new(cfg.seed).permutation(n);
    let mut centroids = vec![0f32; k * d];
    for (c, &i) in centroids.chunks_exact_mut(d).zip(&init) {
        c.iter_mut().zip(&x[i as usize * d..(i as usize + 1) * d]).for_each(|(c, v)| *c = v.to_f32());
    }
    drop(init);
    let mut st = State { k, d, centroids, usage: vec![INIT_USAGE; k], n_splits: 0 };
    let mut split_rng = Rng::new(cfg.seed ^ 0x5EED_5EED_5EED_5EED);

    let chunk = ASSIGN_CHUNK.min(batch);
    let mut rows: Vec<T> = vec![x[0]; chunk * d];
    let mut chunk_labels = vec![0u32; chunk];
    let mut chunk_dist = vec![0f32; chunk];
    let mut counts = vec![0u32; k];
    let mut sums = vec![0f32; k * d];
    let mut touched: Vec<u32> = Vec::with_capacity(k.min(batch));
    let mut labels = vec![u32::MAX; n];
    let mut objective = Vec::new();
    let mut pass_counts = vec![0u64; k];
    let mut pass_sums: Vec<f32> = Vec::new();
    let mut bias = vec![0f32; if cfg.balance > 0.0 { k } else { 0 }];
    let mut mean_sqdist = 0f64; // s: mean squared distance of the previous batch
    let mut step = 0usize;
    let last = recipe.epoch_rows.len() - 1;

    for (epoch, &epoch_n) in recipe.epoch_rows.iter().enumerate() {
        let order = Rng::new(cfg.seed.wrapping_mul(1_000_003).wrapping_add(epoch as u64 + 1)).permutation(n);
        let is_last = epoch == last;
        if is_last {
            pass_counts.iter_mut().for_each(|c| *c = 0);
            pass_sums = vec![0f32; k * d];
        }
        let mut epoch_obj = 0f64;
        for ids in order[..epoch_n].chunks(batch) {
            // One batch: assign every row against a fixed centroid snapshot.
            assigner.set_centroids(&st.centroids, k, d)?;
            let penalized = cfg.balance > 0.0 && mean_sqdist > 0.0;
            if penalized {
                let mean_mass = st.usage.iter().map(|&u| u as f64).sum::<f64>() / k as f64;
                let scale = (cfg.balance as f64 * mean_sqdist / mean_mass) as f32;
                bias.iter_mut().zip(&st.usage).for_each(|(b, &u)| *b = scale * u);
                assigner.set_bias(&bias)?;
            }
            let mut batch_obj = 0f64;
            for part in ids.chunks(chunk) {
                let m = part.len();
                for (row, &i) in rows.chunks_exact_mut(d).zip(part) {
                    row.copy_from_slice(&x[i as usize * d..(i as usize + 1) * d]);
                }
                assigner.assign(&rows[..m * d], &mut chunk_labels[..m], &mut chunk_dist[..m])?;
                if penalized {
                    // Report true squared distances, without the penalty.
                    for (v, &l) in chunk_dist[..m].iter_mut().zip(&chunk_labels[..m]) {
                        *v = (*v - bias[l as usize]).max(0.0);
                    }
                }
                for ((row, &l), &i) in rows.chunks_exact(d).zip(&chunk_labels[..m]).zip(part) {
                    let l = l as usize;
                    if counts[l] == 0 {
                        touched.push(l as u32);
                    }
                    counts[l] += 1;
                    sums[l * d..(l + 1) * d].iter_mut().zip(row).for_each(|(s, v)| *s += v.to_f32());
                    if is_last {
                        labels[i as usize] = l as u32;
                    }
                }
                batch_obj += chunk_dist[..m].iter().map(|&v| v as f64).sum::<f64>();
            }
            epoch_obj += batch_obj;
            mean_sqdist = batch_obj / ids.len() as f64;
            if is_last {
                // Kept for the closing mean, before the EMA consumes them.
                for &j in &touched {
                    let j = j as usize;
                    pass_counts[j] += counts[j] as u64;
                    let (p, s) = (&mut pass_sums[j * d..(j + 1) * d], &sums[j * d..(j + 1) * d]);
                    p.iter_mut().zip(s).for_each(|(p, s)| *p += s);
                }
            }
            let alpha = scheduled_alpha(recipe.alpha0, step, recipe.total_steps) as f32;
            st.apply(&touched, &counts, &sums, alpha);
            for &j in &touched {
                let j = j as usize;
                counts[j] = 0;
                sums[j * d..(j + 1) * d].fill(0.0);
            }
            touched.clear();
            if (step + 1) % CHECK_UNUSED_EVERY == 0 && recipe.unused_threshold > 0.0 {
                st.revive(recipe.unused_threshold, &mut split_rng);
            }
            step += 1;
        }
        objective.push(epoch_obj);
        if cfg.verbose {
            eprintln!(
                "  epoch {}/{} ({:.2} s): objective {:.6e}, {} revived so far",
                epoch + 1,
                last + 1,
                t0.elapsed().as_secs_f64(),
                epoch_obj,
                st.n_splits
            );
        }
    }

    // Closing mean: each centroid becomes the exact mean of the points the last
    // pass assigned to it. Never-assigned centroids keep their position.
    for j in 0..k {
        if pass_counts[j] > 0 {
            let inv = 1.0 / pass_counts[j] as f32;
            let (c, p) = (&mut st.centroids[j * d..(j + 1) * d], &pass_sums[j * d..(j + 1) * d]);
            c.iter_mut().zip(p).for_each(|(c, p)| *c = p * inv);
        }
    }
    Ok(Output { centroids: st.centroids, labels, objective, n_splits: st.n_splits, recipe })
}

/// Balance statistics of a labelling: `(imbalance factor, largest / mean size,
/// empty clusters)`. The imbalance factor is `k * sum_j n_j^2 / n^2`: 1.0 when
/// all clusters are equal, and the factor by which an IVF scan of one random
/// row's own list is longer than with equal lists.
pub fn balance_stats(labels: &[u32], k: usize) -> (f64, f64, usize) {
    let mut counts = vec![0u64; k];
    labels.iter().for_each(|&l| counts[l as usize] += 1);
    let n = labels.len() as f64;
    let squares: f64 = counts.iter().map(|&c| (c * c) as f64).sum();
    let max = *counts.iter().max().unwrap_or(&0) as f64;
    (k as f64 * squares / (n * n), max * k as f64 / n, counts.iter().filter(|&&c| c == 0).count())
}

/// Train on the CPU with the fastest SIMD kernel this machine supports.
pub fn train_cpu<T: Scalar>(x: &[T], d: usize, k: usize, cfg: &Config) -> Result<Output> {
    train(x, d, k, cfg, &mut cpu::CpuAssigner::new())
}

/// Exact nearest-centroid labels and squared distances for `x` (row-major).
pub fn assign<T: Scalar, A: AssignRows<T>>(x: &[T], centroids: &[f32], d: usize, assigner: &mut A) -> Result<(Vec<u32>, Vec<f32>)> {
    let (n, k) = (x.len() / d, centroids.len() / d);
    assigner.set_centroids(centroids, k, d)?;
    let (mut labels, mut dist) = (vec![0u32; n], vec![0f32; n]);
    for ((xc, lc), dc) in x.chunks(ASSIGN_CHUNK * d).zip(labels.chunks_mut(ASSIGN_CHUNK)).zip(dist.chunks_mut(ASSIGN_CHUNK)) {
        assigner.assign(xc, lc, dc)?;
    }
    Ok((labels, dist))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_matches_reference() {
        // Values printed by faiss.contrib.e_means (faiss 1.16.0) for the same inputs.
        let r = resolve_recipe(200_000, 96, 1024, &Config::default());
        assert_eq!(r.batch_size, 16384);
        assert!((r.alpha0 - 0.728183).abs() < 1e-5, "{}", r.alpha0);
        let r = resolve_recipe(1_000_000, 8, 100, &Config { nepochs: 2.5, ..Default::default() });
        assert_eq!(r.epoch_rows, vec![1_000_000, 500_000, 1_000_000]);
        assert_eq!(r.batch_size, 4096);
        assert!((r.alpha0 - 0.319215).abs() < 1e-5, "{}", r.alpha0);
        let r = resolve_recipe(4000, 4, 8, &Config::default());
        assert_eq!(r.batch_size, 32);
        assert!((r.alpha0 - 0.144604).abs() < 1e-5, "{}", r.alpha0);
        assert_eq!(resolve_recipe(1000, 2, 10, &Config::default()).unused_threshold, 0.0);
    }

    #[test]
    fn schedule_endpoints() {
        assert!((scheduled_alpha(0.5, 0, 100) - 0.5).abs() < 1e-12);
        assert!((scheduled_alpha(0.5, 100, 100) - 0.05).abs() < 1e-12);
    }

    #[test]
    fn full_batch_step_one_is_lloyd() {
        // With the whole dataset as one batch and alpha = 1, one update is one
        // Lloyd iteration: every touched centroid becomes its points' mean.
        let x = [0.0f32, 0.0, 0.0, 2.0, 10.0, 0.0, 10.0, 2.0];
        let mut st = State { k: 2, d: 2, centroids: vec![0.0, 0.0, 10.0, 0.0], usage: vec![INIT_USAGE; 2], n_splits: 0 };
        let mut sums = vec![0f32; 4];
        for (i, row) in x.chunks(2).enumerate() {
            sums[(i / 2) * 2] += row[0];
            sums[(i / 2) * 2 + 1] += row[1];
        }
        st.apply(&[0, 1], &[2, 2], &sums, 1.0);
        assert_eq!(st.centroids, vec![0.0, 1.0, 10.0, 1.0]);
        assert_eq!(st.usage, vec![2.0, 2.0]);
    }

    #[test]
    fn revival_splits_a_donor() {
        let mut st = State { k: 3, d: 2, centroids: vec![1.0, 1.0, 5.0, 5.0, 9.0, 9.0], usage: vec![4.0, 0.0, 4.0], n_splits: 0 };
        st.revive(0.1, &mut Rng::new(1));
        assert_eq!(st.n_splits, 1);
        assert!(st.usage.iter().sum::<f32>() == 8.0 && st.usage[1] == 2.0);
        let src = if st.usage[0] == 2.0 { 0 } else { 2 };
        assert!((st.centroids[2] - st.centroids[src * 2]).abs() < 1e-2);
        assert_ne!(st.centroids[2], st.centroids[src * 2]);
    }

    #[test]
    fn closing_mean_and_loss() {
        // 20 overlapping Gaussian-ish blobs in 6-D, clustered into 32.
        let (n, d, k) = (20_000, 6, 32);
        let mut rng = Rng::new(7);
        let centers: Vec<f32> = (0..20 * d).map(|_| 8.0 * (rng.open01() as f32 - 0.5)).collect();
        let mut x = Vec::with_capacity(n * d);
        for i in 0..n {
            for t in 0..d {
                let noise: f32 = (0..4).map(|_| rng.open01() as f32 - 0.5).sum();
                x.push(centers[(i % 20) * d + t] + noise);
            }
        }
        let out = train_cpu(&x, d, k, &Config::default()).unwrap();

        // Each centroid is exactly the mean of the rows the last pass gave it.
        let (mut sums, mut counts) = (vec![0f64; k * d], vec![0usize; k]);
        for (row, &l) in x.chunks(d).zip(&out.labels) {
            counts[l as usize] += 1;
            row.iter().enumerate().for_each(|(t, v)| sums[l as usize * d + t] += *v as f64);
        }
        assert_eq!(counts.iter().sum::<usize>(), n);
        for j in (0..k).filter(|&j| counts[j] > 0) {
            for t in 0..d {
                let mean = sums[j * d + t] / counts[j] as f64;
                assert!((out.centroids[j * d + t] as f64 - mean).abs() < 1e-3);
            }
        }

        // Far better than the random data points training started from, and
        // close to the noise floor (d * 4/12 = 2.0 with one centroid per blob).
        let mse = |c: &[f32]| {
            let (_, dist) = assign(&x, c, d, &mut cpu::CpuAssigner::new()).unwrap();
            dist.iter().map(|&v| v as f64).sum::<f64>() / n as f64
        };
        let init: Vec<f32> = Rng::new(1234).permutation(n)[..k].iter().flat_map(|&i| x[i as usize * d..][..d].to_vec()).collect();
        let (trained, start) = (mse(&out.centroids), mse(&init));
        assert!(trained < 0.75 * start && trained < 2.0, "trained {trained}, start {start}");
        assert!(out.objective.windows(2).all(|w| w[1] < w[0]));
    }
}
