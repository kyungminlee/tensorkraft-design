//! Spectral function types and extraction pipelines.
//!
//! Contains:
//! - `SpectralFunction` — real-frequency spectral function A(omega)
//! - `SpectralSolverMode` — adaptive TDVP/Chebyshev selection
//! - TDVP-based G(t) computation pipeline
//! - Chebyshev expansion of A(omega)
//! - Linear prediction (Levinson-Durbin) and FFT
//! - Spectral positivity restoration

pub mod chebyshev;
pub mod linear_predict;
pub mod positivity;
pub mod tdvp;

/// Real-frequency spectral function A(omega) = -Im[G(omega)] / pi.
///
/// Defined on a uniform frequency grid `omega`. The spectral sum rule
/// requires integral A(omega) d_omega = 1 for a single-orbital impurity.
///
/// Invariant maintained after `restore_positivity`:
///   A(omega) >= 0 for all omega
#[derive(Clone, Debug)]
pub struct SpectralFunction {
    /// Frequency grid points (uniform spacing).
    pub omega: Vec<f64>,
    /// Spectral weight at each grid point. Same length as `omega`.
    pub values: Vec<f64>,
    /// Frequency spacing d_omega (cached for integration).
    pub d_omega: f64,
}

impl SpectralFunction {
    /// Construct from frequency grid and spectral values.
    ///
    /// # Panics
    /// Panics if `omega.len() != values.len()` or if `omega` is empty.
    pub fn new(omega: Vec<f64>, values: Vec<f64>) -> Self {
        assert!(!omega.is_empty(), "omega must not be empty");
        assert_eq!(
            omega.len(),
            values.len(),
            "omega and values must have the same length"
        );
        let d_omega = if omega.len() > 1 {
            (omega[1] - omega[0]).abs()
        } else {
            1.0
        };
        Self {
            omega,
            values,
            d_omega,
        }
    }

    /// Spectral sum rule: integral A(omega) d_omega via the trapezoidal rule.
    pub fn sum_rule(&self) -> f64 {
        if self.values.len() < 2 {
            return self.values.first().copied().unwrap_or(0.0) * self.d_omega;
        }
        let n = self.values.len();
        let mut sum = 0.5 * (self.values[0] + self.values[n - 1]);
        for i in 1..n - 1 {
            sum += self.values[i];
        }
        sum * self.d_omega
    }

    /// Value at omega ~ 0 (Fermi level). Interpolates linearly between the
    /// two grid points bracketing omega = 0.
    ///
    /// # Panics
    /// Panics if `omega` does not span omega = 0.
    pub fn value_at_omega_zero(&self) -> f64 {
        // Find the bracket
        for i in 0..self.omega.len() - 1 {
            if self.omega[i] <= 0.0 && self.omega[i + 1] >= 0.0 {
                let w0 = self.omega[i];
                let w1 = self.omega[i + 1];
                let v0 = self.values[i];
                let v1 = self.values[i + 1];
                if (w1 - w0).abs() < f64::EPSILON {
                    return v0;
                }
                let t = (0.0 - w0) / (w1 - w0);
                return v0 + t * (v1 - v0);
            }
        }
        panic!("omega grid does not span omega = 0");
    }

    /// The nth spectral moment: integral omega^n A(omega) d_omega via the trapezoidal rule.
    pub fn moment(&self, n: usize) -> f64 {
        if self.values.len() < 2 {
            let w = self.omega.first().copied().unwrap_or(0.0);
            return w.powi(n as i32) * self.values.first().copied().unwrap_or(0.0) * self.d_omega;
        }
        let len = self.values.len();
        let mut sum = 0.5
            * (self.omega[0].powi(n as i32) * self.values[0]
                + self.omega[len - 1].powi(n as i32) * self.values[len - 1]);
        for i in 1..len - 1 {
            sum += self.omega[i].powi(n as i32) * self.values[i];
        }
        sum * self.d_omega
    }

    /// L-infinity distance ||self - other||_inf for convergence checks.
    ///
    /// # Panics
    /// Panics if `self.omega.len() != other.omega.len()`.
    pub fn max_distance(&self, other: &SpectralFunction) -> f64 {
        assert_eq!(
            self.omega.len(),
            other.omega.len(),
            "spectral functions must have the same grid size"
        );
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Number of grid points.
    pub fn len(&self) -> usize {
        self.omega.len()
    }

    /// Whether the spectral function has no grid points.
    pub fn is_empty(&self) -> bool {
        self.omega.is_empty()
    }
}

/// Controls which spectral function engine is designated as primary.
///
/// Both engines (TDVP + linear prediction, and Chebyshev) are always
/// computed at each DMFT iteration. Only their roles (primary vs.
/// cross-validation) change.
#[derive(Clone, Debug)]
pub enum SpectralSolverMode {
    /// TDVP + linear prediction is primary; Chebyshev is cross-validation.
    /// Appropriate for gapped/insulating phases.
    TdvpPrimary,
    /// Chebyshev expansion is primary; TDVP + linear prediction is cross-validation.
    /// Appropriate for gapless/metallic phases.
    ChebyshevPrimary,
    /// Automatically select based on the entanglement spectrum gap.
    ///
    /// gap >= gap_threshold => gapped/insulating => TdvpPrimary
    /// gap <  gap_threshold => gapless/metallic  => ChebyshevPrimary
    ///
    /// Default `gap_threshold`: 0.1 (design doc Section 8.4.1)
    Adaptive { gap_threshold: f64 },
}

impl Default for SpectralSolverMode {
    fn default() -> Self {
        SpectralSolverMode::Adaptive {
            gap_threshold: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_function_construction() {
        let omega: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        let values: Vec<f64> = omega.iter().map(|&w| {
            // Lorentzian: (eta/pi) / (w^2 + eta^2)
            let eta = 0.1;
            eta / (std::f64::consts::PI * (w * w + eta * eta))
        }).collect();
        let spec = SpectralFunction::new(omega, values);
        assert_eq!(spec.len(), 101);
        assert!((spec.d_omega - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_spectral_sum_rule_lorentzian() {
        // Lorentzian should integrate to 1.0
        let n = 10001;
        let omega: Vec<f64> = (0..n).map(|i| -50.0 + 100.0 * i as f64 / (n as f64 - 1.0)).collect();
        let eta = 0.5;
        let values: Vec<f64> = omega.iter().map(|&w| {
            eta / (std::f64::consts::PI * (w * w + eta * eta))
        }).collect();
        let spec = SpectralFunction::new(omega, values);
        let sr = spec.sum_rule();
        assert!((sr - 1.0).abs() < 0.01, "sum rule = {}", sr);
    }

    #[test]
    fn test_value_at_omega_zero() {
        let omega: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.5).collect();
        let values: Vec<f64> = omega.iter().map(|&w| {
            let eta = 0.5;
            eta / (std::f64::consts::PI * (w * w + eta * eta))
        }).collect();
        let spec = SpectralFunction::new(omega, values);
        let v0 = spec.value_at_omega_zero();
        let expected = 0.5 / (std::f64::consts::PI * 0.25); // eta/(pi*eta^2) = 1/(pi*eta)
        assert!((v0 - expected).abs() < 0.01, "v0 = {}, expected = {}", v0, expected);
    }

    #[test]
    fn test_max_distance() {
        let omega: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let v1: Vec<f64> = vec![1.0; 10];
        let v2: Vec<f64> = vec![1.5; 10];
        let s1 = SpectralFunction::new(omega.clone(), v1);
        let s2 = SpectralFunction::new(omega, v2);
        assert!((s1.max_distance(&s2) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_spectral_moments() {
        // For symmetric Lorentzian centered at 0, odd moments should be ~0
        let n = 10001;
        let omega: Vec<f64> = (0..n).map(|i| -50.0 + 100.0 * i as f64 / (n as f64 - 1.0)).collect();
        let eta = 0.5;
        let values: Vec<f64> = omega.iter().map(|&w| {
            eta / (std::f64::consts::PI * (w * w + eta * eta))
        }).collect();
        let spec = SpectralFunction::new(omega, values);
        let m0 = spec.moment(0);
        let m1 = spec.moment(1);
        assert!((m0 - 1.0).abs() < 0.01, "m0 = {}", m0);
        assert!(m1.abs() < 0.01, "m1 = {}", m1);
    }
}
