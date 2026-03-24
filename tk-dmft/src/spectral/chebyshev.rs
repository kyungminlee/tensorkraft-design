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

/// Compute the impurity spectral function via Chebyshev expansion.
///
/// Algorithm (design doc Section 8.4.3):
///
/// 1. Rescale H to H_tilde = (H - b) / a where:
///      a = (E_max - E_min) / (2 - epsilon)   (small epsilon for numerical safety)
///      b = (E_max + E_min) / 2
///    This maps the spectrum into (-1, 1).
///
/// 2. Construct |alpha> = c^dag_{0,sigma}|psi_0> by applying the impurity
///    creation operator to the DMRG ground state.
///
/// 3. Compute Chebyshev moments via the three-term recursion:
///      |phi_0> = |alpha>
///      |phi_1> = H_tilde|alpha>
///      |phi_n> = 2 * H_tilde|phi_{n-1}> - |phi_{n-2}>
///      mu_n = <psi_0|c_{0,sigma}|phi_n>
///
///    Each step calls `DMRGEngine::apply_hamiltonian` (H_eff matvec reuse).
///
/// 4. Apply Jackson kernel (if enabled).
///
/// 5. Reconstruct A(omega) via `reconstruct_from_moments`.
///
/// # Complexity
/// O(n_moments * N * d * D^2 * w) — one H_eff matvec application per moment.
///
/// # Parameters
/// - `omega`: target frequency grid
/// - `e_min`: lower spectral bound
/// - `e_max`: upper spectral bound
/// - `config`: Chebyshev configuration
///
/// # Errors
/// Returns `DmftError::ChebyshevBandwidthError` if e_min >= e_max.
/// Returns `DmftError::Dmrg` if the Hamiltonian application fails.
pub fn chebyshev_expand(
    _omega: &[f64],
    e_min: f64,
    e_max: f64,
    _config: &ChebyshevConfig,
) -> DmftResult<SpectralFunction> {
    if e_min >= e_max {
        return Err(DmftError::ChebyshevBandwidthError {
            e_min,
            e_max,
            e0: 0.0,
        });
    }

    // The Chebyshev moment computation requires:
    //
    // 1. A converged ground state MPS |psi_0> from DMRGEngine
    // 2. Application of c†_{0,σ}|psi_0> = |α> (same as TDVP pipeline)
    // 3. Iterative H_eff matvec to build Chebyshev vectors:
    //    |φ_0> = |α>
    //    |φ_1> = H̃|α>    where H̃ = (H - b·I) / a
    //    |φ_n> = 2·H̃|φ_{n-1}> - |φ_{n-2}>
    // 4. Moments: μ_n = <ψ_0|c_{0,σ}|φ_n>
    //
    // When DMRGEngine::apply_hamiltonian_to_mps is available:
    //
    //   let eps = 0.01;
    //   let a = (e_max - e_min) / (2.0 - eps);
    //   let b = (e_max + e_min) / 2.0;
    //
    //   let alpha = apply_operator_to_mps(c_dag_up, site_0, &psi0);
    //   let bra = apply_operator_to_mps(c_up, site_0, &psi0);
    //
    //   let mut phi_prev = alpha.clone();
    //   let mut phi_curr = rescaled_apply_h(&phi_prev, a, b, engine);
    //   let mut moments = vec![0.0; config.n_moments];
    //   moments[0] = mps_overlap(&bra, &phi_prev).re;
    //   if config.n_moments > 1 {
    //       moments[1] = mps_overlap(&bra, &phi_curr).re;
    //   }
    //
    //   for n in 2..config.n_moments {
    //       let phi_next = 2 * rescaled_apply_h(&phi_curr, a, b, engine) - &phi_prev;
    //       moments[n] = mps_overlap(&bra, &phi_next).re;
    //       phi_prev = phi_curr;
    //       phi_curr = phi_next;
    //   }
    //
    //   reconstruct_from_moments(&moments, omega, e_min, e_max, config)

    Err(DmftError::Dmrg(tk_dmrg::DmrgError::NotImplemented(
        "Chebyshev moment computation requires DMRGEngine::apply_hamiltonian_to_mps".into(),
    )))
}

/// Compute a spectral function from pre-computed Chebyshev moments.
///
/// This is the usable entry point when moments have been computed externally
/// (e.g., from exact diagonalization or a future DMRGEngine integration).
///
/// Delegates to `reconstruct_from_moments` with the configured Jackson kernel.
pub fn chebyshev_from_precomputed_moments(
    moments: &[f64],
    omega: &[f64],
    e_min: f64,
    e_max: f64,
    config: &ChebyshevConfig,
) -> DmftResult<SpectralFunction> {
    reconstruct_from_moments(moments, omega, e_min, e_max, config)
}

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

    #[test]
    fn test_chebyshev_expand_bandwidth_error() {
        let omega = vec![0.0];
        let result = chebyshev_expand(&omega, 1.0, 0.0, &ChebyshevConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_chebyshev_expand_returns_not_implemented() {
        let omega: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.1).collect();
        let result = chebyshev_expand(&omega, -5.0, 5.0, &ChebyshevConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_chebyshev_from_precomputed_moments() {
        let mut moments = vec![0.0; 50];
        moments[0] = 1.0;
        let omega: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        let config = ChebyshevConfig {
            n_moments: 50,
            ..Default::default()
        };
        let result = chebyshev_from_precomputed_moments(&moments, &omega, -5.0, 5.0, &config);
        assert!(result.is_ok());
    }
}
