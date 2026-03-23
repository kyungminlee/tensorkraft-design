//! Chebyshev expansion of the spectral function.
//!
//! Computes A(omega) directly in frequency space by expanding the resolvent
//! in Chebyshev polynomials rescaled to the spectrum of H. This bypasses
//! time-domain linear prediction instabilities that afflict metallic phases.

use crate::error::{DmftError, DmftResult};
use crate::spectral::SpectralFunction;

/// Configuration for Chebyshev expansion of the spectral function.
#[derive(Clone, Debug)]
pub struct ChebyshevConfig {
    /// Number of Chebyshev moments to compute. Default: 1000.
    pub n_moments: usize,
    /// Lorentzian broadening eta (Jackson kernel damping). Default: 0.05.
    pub broadening_eta: f64,
    /// Lower spectral bound E_min. `None` = auto-detect.
    pub e_min: Option<f64>,
    /// Upper spectral bound E_max. `None` = auto-detect.
    pub e_max: Option<f64>,
    /// Apply Jackson kernel to reduce Gibbs oscillations. Default: true.
    pub jackson_kernel: bool,
}

impl Default for ChebyshevConfig {
    fn default() -> Self {
        Self {
            n_moments: 1000,
            broadening_eta: 0.05,
            e_min: None,
            e_max: None,
            jackson_kernel: true,
        }
    }
}

/// Compute Jackson kernel coefficients g_n for N Chebyshev moments.
///
/// g_n = [(N-n+1)*cos(pi*n/(N+1)) + sin(pi*n/(N+1))*cot(pi/(N+1))] / (N+1)
pub fn jackson_kernel(n_moments: usize) -> Vec<f64> {
    let big_n = n_moments as f64;
    let pi = std::f64::consts::PI;
    let cot_pi_over_np1 = (pi / (big_n + 1.0)).cos() / (pi / (big_n + 1.0)).sin();

    (0..n_moments)
        .map(|n| {
            let nf = n as f64;
            let cos_term = (big_n - nf + 1.0) * (pi * nf / (big_n + 1.0)).cos();
            let sin_term = (pi * nf / (big_n + 1.0)).sin() * cot_pi_over_np1;
            (cos_term + sin_term) / (big_n + 1.0)
        })
        .collect()
}

/// Reconstruct A(omega) from Chebyshev moments mu_n.
///
/// A(omega) = (1 / (pi * a * sqrt(1 - w_tilde^2))) * (mu_0 + 2 * Sum_{n>=1} g_n * mu_n * T_n(w_tilde))
///
/// where w_tilde = (omega - b) / a is the rescaled frequency.
///
/// # Parameters
/// - `moments`: Chebyshev moments mu_n
/// - `omega`: target frequency grid
/// - `e_min`: lower spectral bound
/// - `e_max`: upper spectral bound
/// - `config`: Chebyshev configuration
///
/// # Errors
/// Returns `DmftError::ChebyshevBandwidthError` if e_min >= e_max.
pub fn reconstruct_from_moments(
    moments: &[f64],
    omega: &[f64],
    e_min: f64,
    e_max: f64,
    config: &ChebyshevConfig,
) -> DmftResult<SpectralFunction> {
    if e_min >= e_max {
        return Err(DmftError::ChebyshevBandwidthError {
            e_min,
            e_max,
            e0: 0.0,
        });
    }

    let eps = 0.01; // small margin for numerical safety
    let a = (e_max - e_min) / (2.0 - eps);
    let b = (e_max + e_min) / 2.0;
    let pi = std::f64::consts::PI;

    let kernel = if config.jackson_kernel {
        jackson_kernel(moments.len())
    } else {
        vec![1.0; moments.len()]
    };

    let values: Vec<f64> = omega
        .iter()
        .map(|&w| {
            let w_tilde = (w - b) / a;
            if w_tilde.abs() >= 1.0 {
                return 0.0;
            }

            let sqrt_factor = (1.0 - w_tilde * w_tilde).sqrt();
            if sqrt_factor < f64::EPSILON {
                return 0.0;
            }

            // Evaluate Chebyshev sum using three-term recursion
            let mut t_prev = 1.0; // T_0
            let mut t_curr = w_tilde; // T_1
            let mut sum = kernel[0] * moments[0];

            for n in 1..moments.len() {
                sum += 2.0 * kernel[n] * moments[n] * t_curr;
                let t_next = 2.0 * w_tilde * t_curr - t_prev;
                t_prev = t_curr;
                t_curr = t_next;
            }

            sum / (pi * a * sqrt_factor)
        })
        .collect();

    Ok(SpectralFunction::new(omega.to_vec(), values))
}

/// Placeholder for full Chebyshev expansion using DMRG engine.
///
/// The actual implementation requires `DMRGEngine` to apply H to MPS states.
/// This is a design coordination point with tk-dmrg (spec Open Question #3).
///
/// For draft: provides `reconstruct_from_moments` for pre-computed moments.
/// Full `chebyshev_expand` that computes moments via H_eff matvec will be
/// implemented when `DMRGEngine::apply_hamiltonian_to_mps` is available.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ChebyshevConfig::default();
        assert_eq!(config.n_moments, 1000);
        assert!(config.jackson_kernel);
    }

    #[test]
    fn test_jackson_kernel_first_element() {
        let kernel = jackson_kernel(100);
        // g_0 should be close to 1.0
        assert!((kernel[0] - 1.0).abs() < 0.02, "g_0 = {}", kernel[0]);
    }

    #[test]
    fn test_jackson_kernel_monotone_decreasing() {
        let kernel = jackson_kernel(100);
        // Jackson kernel should generally decrease
        for n in 1..kernel.len() {
            assert!(
                kernel[n] <= kernel[n - 1] + 0.05,
                "kernel not approximately decreasing at n={}: {} > {}",
                n,
                kernel[n],
                kernel[n - 1]
            );
        }
    }

    #[test]
    fn test_reconstruct_bandwidth_error() {
        let result = reconstruct_from_moments(
            &[1.0],
            &[0.0],
            1.0,
            0.0, // e_min > e_max
            &ChebyshevConfig::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_reconstruct_trivial_moment() {
        // With mu_0 = 1, mu_n>0 = 0, we get a semicircular-like shape
        let n_moments = 50;
        let mut moments = vec![0.0; n_moments];
        moments[0] = 1.0;

        let omega: Vec<f64> = (-100..=100).map(|i| i as f64 * 0.05).collect();
        let config = ChebyshevConfig {
            n_moments,
            jackson_kernel: true,
            ..Default::default()
        };
        let result = reconstruct_from_moments(&moments, &omega, -5.0, 5.0, &config);
        assert!(result.is_ok());
        let spec = result.unwrap();
        // Values inside the band should be non-negative (approximately)
        // and values outside should be zero
        for (&w, &v) in spec.omega.iter().zip(spec.values.iter()) {
            if w.abs() > 5.0 {
                assert!(v.abs() < 1e-10, "nonzero outside band at w={}: {}", w, v);
            }
        }
    }
}
