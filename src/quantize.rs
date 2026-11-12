//! Calibrated `f32` -> `i8` conversion that preserves L2 geometry.
//!
//! Same scheme as faiss's `Int8Quantizer`: a per-dimension shift (the sample
//! mean) and one global scale, `code = round((x - shift) * scale)` clamped to
//! `[-128, 127]`. A single scale for all dimensions keeps squared distances
//! proportional (`|a - b|^2 * scale^2` in code units); per-dimension scales
//! would reweight the coordinates.
//!
//! [`Calibration::faiss()`] reproduces faiss: the scale is the largest power
//! of two that maps the sample's largest centered magnitude to at most 112.
//! [`Calibration::clipped`] instead maps a high quantile of the magnitudes to
//! the target, clipping the few values beyond it; on heavy-tailed data that
//! spends the 8 bits on the bulk of the values rather than on one extreme.

use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    /// Code magnitude the calibrated peak is mapped to.
    pub target_peak: f32,
    /// Which centered magnitude counts as the peak: 1.0 is the maximum.
    pub quantile: f64,
    /// Round the scale down to a power of two (decode is then exact).
    pub power_of_two: bool,
}

impl Calibration {
    pub fn faiss() -> Self {
        Calibration { target_peak: 112.0, quantile: 1.0, power_of_two: true }
    }
    pub fn clipped(quantile: f64) -> Self {
        Calibration { target_peak: 127.0, quantile, power_of_two: false }
    }
}

#[derive(Clone, Debug)]
pub struct Int8Quantizer {
    pub d: usize,
    pub shift: Vec<f32>,
    pub scale: f32,
}

impl Int8Quantizer {
    /// Fit on a representative sample (row-major, `n x d`).
    pub fn fit(sample: &[f32], d: usize, cal: Calibration) -> Self {
        let n = sample.len() / d;
        assert!(n > 0 && sample.len() == n * d);
        let mut mean = vec![0f64; d];
        for row in sample.chunks_exact(d) {
            mean.iter_mut().zip(row).for_each(|(m, &v)| *m += v as f64);
        }
        let shift: Vec<f32> = mean.iter().map(|m| (m / n as f64) as f32).collect();
        let mut mags: Vec<f32> =
            sample.par_chunks_exact(d).flat_map_iter(|row| row.iter().zip(&shift).map(|(v, s)| (v - s).abs())).collect();
        let idx = (((mags.len() - 1) as f64) * cal.quantile.clamp(0.0, 1.0)).round() as usize;
        let peak = *mags.select_nth_unstable_by(idx, f32::total_cmp).1;
        let mut scale = if peak > 0.0 { cal.target_peak / peak } else { 1.0 };
        if cal.power_of_two {
            scale = 2f32.powi(scale.log2().floor() as i32);
        }
        Int8Quantizer { d, shift, scale }
    }

    /// Encode rows. Values beyond the calibrated range are clamped.
    pub fn encode(&self, x: &[f32]) -> Vec<i8> {
        let mut out = vec![0i8; x.len()];
        out.par_chunks_mut(self.d).zip(x.par_chunks(self.d)).for_each(|(o, row)| {
            for ((o, &v), &s) in o.iter_mut().zip(row).zip(&self.shift) {
                *o = ((v - s) * self.scale).round().clamp(-128.0, 127.0) as i8;
            }
        });
        out
    }

    /// Map codes, or centroids learned on codes, back to the original units.
    pub fn decode(&self, codes: &[f32]) -> Vec<f32> {
        let inv = 1.0 / self.scale;
        codes.chunks(self.d).flat_map(|row| row.iter().zip(&self.shift).map(move |(&v, &s)| v * inv + s)).collect()
    }

    /// Squared distances in code units -> original units.
    pub fn decode_sqdist(&self, sqdist: f32) -> f32 {
        sqdist / (self.scale * self.scale)
    }

    /// Fraction of values in `x` that clamp when encoded.
    pub fn clipped_fraction(&self, x: &[f32]) -> f64 {
        let clipped: usize = x
            .par_chunks(self.d)
            .map(|row| {
                row.iter().zip(&self.shift).filter(|&(&v, &s)| !(-128.5..127.5).contains(&((v - s) * self.scale))).count()
            })
            .sum();
        clipped as f64 / x.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faiss_calibration_and_round_trip() {
        // Dimension 0 centred on 10, dimension 1 on -3; largest centred magnitude 5.
        let sample = [15.0f32, -3.0, 5.0, -3.0, 10.0, -2.0, 10.0, -4.0];
        let q = Int8Quantizer::fit(&sample, 2, Calibration::faiss());
        assert_eq!(q.shift, vec![10.0, -3.0]);
        assert_eq!(q.scale, 16.0); // floor(log2(112 / 5)) = 4
        let codes = q.encode(&sample);
        assert_eq!(codes, vec![80, 0, -80, 0, 0, 16, 0, -16]);
        let back = q.decode(&codes.iter().map(|&c| c as f32).collect::<Vec<_>>());
        assert_eq!(back, sample);
        assert_eq!(q.decode_sqdist(256.0), 1.0);
        assert_eq!(q.encode(&[1000.0, -3.0]), vec![127, 0]); // outliers clamp
        assert_eq!(q.clipped_fraction(&[1000.0, -3.0]), 0.5);
    }
}
