//! Bath discretization via Lanczos tridiagonalization.
//!
//! Converts the continuous hybridization function Delta(omega) into
//! `n_bath` discrete bath parameters (epsilon_k, V_k).

use num_traits::Zero;
use tk_core::Scalar;

use crate::error::{DmftError, DmftResult};
use crate::impurity::bath::BathParameters;

/// Configuration for Lanczos tridiagonalization bath discretization.
#[derive(Clone, Debug)]
pub struct BathDiscretizationConfig {
    /// Maximum number of Lanczos recursion steps.
    /// Default: 0 (auto: `n_bath * 10`).
    pub max_lanczos_steps: usize,
    /// Convergence threshold on tridiagonalization residual. Default: 1e-12.
    pub lanczos_tol: f64,
    /// Number of frequency grid points for hybridization function evaluation.
    /// Default: 2000.
    pub n_omega_points: usize,
    /// Half-bandwidth of the frequency grid. Default: 10.0.
    pub bandwidth: f64,
    /// Lorentzian broadening for hybridization function evaluation. Default: 0.05.
    pub broadening: f64,
}

impl Default for BathDiscretizationConfig {
    fn default() -> Self {
        Self {
            max_lanczos_steps: 0,
            lanczos_tol: 1e-12,
            n_omega_points: 2000,
            bandwidth: 10.0,
            broadening: 0.05,
        }
    }
}

/// Perform Lanczos tridiagonalization to discretize a hybridization function.
///
/// Algorithm (design doc Section 2.2 / Section 8.4):
///
/// 1. Frequency grid: omega_j in [-Omega, Omega] with `n_omega_points` points.
/// 2. Spectral weight: w_j = -Im[Delta(omega_j)] * d_omega / pi (must be >= 0).
/// 3. Lanczos start vector: phi_0(j) = 1 (normalized by total weight W).
///    V_1 = sqrt(W).
/// 4. Lanczos recursion: build tridiagonal T of dimension n_bath x n_bath.
/// 5. Bath parameters: epsilon_k = alpha_k, V_k = beta_k.
/// 6. Validation: check ||Delta_discretized - Delta_target||_inf / ||Delta_target||_inf < tol.
///
/// Complexity: O(n_bath * n_omega_points)
pub fn lanczos_tridiagonalize<T: Scalar>(
    delta_target: &[T],
    omega: &[T::Real],
    n_bath: usize,
    config: &BathDiscretizationConfig,
) -> DmftResult<BathParameters<T>>
where
    T::Real: Into<f64> + From<f64>,
{
    let n_omega = omega.len();
    assert_eq!(
        delta_target.len(),
        n_omega,
        "delta_target and omega must have the same length"
    );

    if n_bath == 0 {
        return Ok(BathParameters {
            epsilon: vec![],
            v: vec![],
            n_bath: 0,
        });
    }

    // Compute frequency spacing (assume uniform grid)
    let d_omega: f64 = if n_omega > 1 {
        let o0: f64 = omega[0].into();
        let o1: f64 = omega[1].into();
        (o1 - o0).abs()
    } else {
        1.0
    };

    // Step 2: Spectral weight w_j = -Im[Delta(omega_j)] * d_omega / pi
    let pi = std::f64::consts::PI;
    let mut weights: Vec<f64> = Vec::with_capacity(n_omega);
    let mut n_negative = 0usize;
    for &d in delta_target {
        // -Im[Delta] for complex; for real, the imaginary part is zero
        // We use abs_sq and from_real to be generic, but Delta is typically complex.
        // For real-valued T, the "imaginary part" is zero.
        let val_f64: f64 = if T::is_real() {
            // For real-valued input, we interpret delta_target as -Im[Delta] directly
            // (the caller passes the spectral weight, which is real and non-negative)
            let r: f64 = d.abs_sq().into();
            r.sqrt()
        } else {
            // For complex input: extract -Im[Delta] = Im(Delta.conj())
            // We use: d - d.conj() = 2i * Im(d), so |d - d.conj()|^2 = 4 * Im(d)^2
            let diff = d - d.conj();
            let im_sq: f64 = diff.abs_sq().into() / 4.0;
            let im = im_sq.sqrt();
            // -Im[Delta] should be positive for a valid hybridization function
            im
        };
        let w = val_f64 * d_omega / pi;
        if w < -1e-15 {
            n_negative += 1;
        }
        weights.push(w.max(0.0));
    }

    if n_negative > 0 {
        return Err(DmftError::InvalidHybridizationFunction { n_negative });
    }

    // Total weight W = sum_j w_j
    let total_weight: f64 = weights.iter().sum();
    if total_weight < f64::EPSILON {
        // Degenerate case: zero hybridization
        return Ok(BathParameters {
            epsilon: vec![T::Real::zero(); n_bath],
            v: vec![T::zero(); n_bath],
            n_bath,
        });
    }

    // Step 3: Start vector phi_0(j) = 1/sqrt(W) (normalized in the w-inner product)
    // V_1 = sqrt(W)
    let v1 = total_weight.sqrt();

    // Lanczos recursion vectors
    let mut phi_prev: Vec<f64> = vec![0.0; n_omega]; // phi_{k-1}
    let mut phi_curr: Vec<f64> = weights.iter().map(|_| 1.0 / v1).collect(); // phi_0 normalized

    let max_steps = if config.max_lanczos_steps == 0 {
        n_bath * 10
    } else {
        config.max_lanczos_steps
    };

    let actual_steps = n_bath.min(max_steps).min(n_omega);

    let mut alphas: Vec<f64> = Vec::with_capacity(actual_steps);
    let mut betas: Vec<f64> = Vec::with_capacity(actual_steps);
    betas.push(v1); // beta_0 = V_1 = sqrt(W)

    // Step 4: Lanczos recursion
    for k in 0..actual_steps {
        // alpha_k = <phi_k | omega | phi_k>_w = sum_j w_j * omega_j * |phi_k(j)|^2
        let alpha_k: f64 = (0..n_omega)
            .map(|j| {
                let oj: f64 = omega[j].into();
                weights[j] * oj * phi_curr[j] * phi_curr[j]
            })
            .sum();
        alphas.push(alpha_k);

        if k + 1 >= actual_steps {
            break;
        }

        // Compute residual: r(j) = omega_j * phi_k(j) - alpha_k * phi_k(j) - beta_k * phi_{k-1}(j)
        let beta_k = if k > 0 { betas[k] } else { 0.0 };
        let mut phi_next: Vec<f64> = Vec::with_capacity(n_omega);
        for j in 0..n_omega {
            let oj: f64 = omega[j].into();
            let r = oj * phi_curr[j] - alpha_k * phi_curr[j] - beta_k * phi_prev[j];
            phi_next.push(r);
        }

        // beta_{k+1} = ||r||_w = sqrt(sum_j w_j * r_j^2)
        let beta_next_sq: f64 = (0..n_omega)
            .map(|j| weights[j] * phi_next[j] * phi_next[j])
            .sum();
        let beta_next = beta_next_sq.sqrt();

        if beta_next < 1e-15 {
            // Lanczos breakdown: exact invariant subspace found
            betas.push(0.0);
            // Pad remaining alphas/betas with zeros
            for _ in (k + 1)..actual_steps {
                alphas.push(0.0);
                betas.push(0.0);
            }
            break;
        }

        betas.push(beta_next);

        // Normalize: phi_{k+1} = r / beta_{k+1}
        for j in 0..n_omega {
            phi_next[j] /= beta_next;
        }

        phi_prev = phi_curr;
        phi_curr = phi_next;
    }

    // Step 5: Extract bath parameters
    // epsilon_k = alpha_k (on-site energies)
    // V_1 = sqrt(W) (first hybridization), V_k = beta_k for k >= 1
    let n_actual = alphas.len().min(n_bath);
    let mut epsilon = Vec::with_capacity(n_bath);
    let mut v_params = Vec::with_capacity(n_bath);

    for k in 0..n_bath {
        if k < n_actual {
            epsilon.push(T::Real::from(alphas[k]));
        } else {
            epsilon.push(T::Real::zero());
        }
        if k < betas.len() {
            v_params.push(T::from_real(T::Real::from(betas[k])));
        } else {
            v_params.push(T::zero());
        }
    }

    // Step 6: Validation (optional — check residual if needed)
    // For draft: skip full validation, return the discretized parameters.
    // TODO: Implement full Delta_discretized vs Delta_target residual check.

    Ok(BathParameters {
        epsilon,
        v: v_params,
        n_bath,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BathDiscretizationConfig::default();
        assert_eq!(config.n_omega_points, 2000);
        assert!((config.lanczos_tol - 1e-12).abs() < 1e-15);
        assert!((config.bandwidth - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_lanczos_trivial_zero_bath() {
        let config = BathDiscretizationConfig::default();
        let delta = vec![0.0_f64; 10];
        let omega: Vec<f64> = (0..10).map(|i| -5.0 + i as f64).collect();
        let result = lanczos_tridiagonalize::<f64>(&delta, &omega, 0, &config);
        assert!(result.is_ok());
        let bath = result.unwrap();
        assert_eq!(bath.n_bath, 0);
    }

    #[test]
    fn test_lanczos_produces_correct_count() {
        // Use a simple positive spectral weight
        let n_omega = 100;
        let omega: Vec<f64> = (0..n_omega).map(|i| -5.0 + 10.0 * i as f64 / (n_omega as f64 - 1.0)).collect();
        // Semicircular density of states: w(omega) = (2/pi) * sqrt(1 - (omega/W)^2) for |omega| < W
        let half_bw = 2.0;
        let delta: Vec<f64> = omega.iter().map(|&w| {
            if w.abs() < half_bw {
                (2.0 / std::f64::consts::PI) * (1.0 - (w / half_bw).powi(2)).sqrt()
            } else {
                0.0
            }
        }).collect();

        let config = BathDiscretizationConfig {
            max_lanczos_steps: 0,
            lanczos_tol: 1e-12,
            n_omega_points: n_omega,
            bandwidth: 10.0,
            broadening: 0.05,
        };

        let n_bath = 4;
        let result = lanczos_tridiagonalize::<f64>(&delta, &omega, n_bath, &config);
        assert!(result.is_ok());
        let bath = result.unwrap();
        assert_eq!(bath.n_bath, n_bath);
        assert_eq!(bath.epsilon.len(), n_bath);
        assert_eq!(bath.v.len(), n_bath);
    }
}
