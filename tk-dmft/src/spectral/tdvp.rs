//! TDVP-based real-time Green's function computation.
//!
//! Pipeline: apply c^dag|psi_0> -> TDVP time evolution -> sample G(t_k) -> return.
//!
//! This module provides the TDVP spectral pipeline orchestration.
//! The actual TDVP time evolution is delegated to `tk-dmrg`'s `TdvpDriver`.

use num_complex::Complex;

use crate::error::{DmftError, DmftResult};
use crate::spectral::SpectralFunction;
use crate::spectral::linear_predict::{
    LinearPredictionConfig, deconvolve_lorentzian, fft_to_spectral, linear_predict_regularized,
};
use crate::spectral::positivity::restore_positivity;

/// Configuration for the TDVP Green's function pipeline.
///
/// This wraps the time evolution parameters needed by the TDVP driver.
#[derive(Clone, Debug)]
pub struct TdvpSpectralConfig {
    /// Total simulation time t_max (inverse energy units). Default: 20.0.
    pub t_max: f64,
    /// Physical time step dt. Default: 0.05.
    pub dt: f64,
    /// Maximum MPS bond dimension during time evolution. Default: 500.
    pub max_bond_dim: usize,
}

impl Default for TdvpSpectralConfig {
    fn default() -> Self {
        Self {
            t_max: 20.0,
            dt: 0.05,
            max_bond_dim: 500,
        }
    }
}

/// Compute the retarded Green's function G(t) via TDVP time evolution.
///
/// The complete pipeline:
/// 1. Construct |alpha> = c^dag_{0,sigma}|psi_0> from the DMRG ground state.
/// 2. Run `TdvpDriver` forward in time with steps of size dt.
/// 3. Sample G(t_k) = -i * <psi_0|c_{0,sigma}|alpha(t_k)> at each step.
///
/// This requires `DMRGEngine` and `TdvpDriver` from tk-dmrg, and the ability
/// to apply fermionic operators to MPS states.
///
/// # Returns
/// Complex Green's function samples at uniform time steps 0, dt, 2dt, ..., t_max.
///
/// # Errors
/// Returns `DmftError::Dmrg` wrapping errors from DMRG/TDVP operations.
pub fn compute_greens_function_tdvp(
    _config: &TdvpSpectralConfig,
) -> DmftResult<Vec<Complex<f64>>> {
    // The TDVP Green's function computation requires:
    //
    // 1. A converged ground state MPS |psi_0> from DMRGEngine::run()
    // 2. Application of c†_{0,σ}|psi_0> to create the excited state |α>
    //    - This requires MPS operator application (apply a local operator
    //      to site 0 of the MPS, increasing or modifying the local state)
    //    - c†_{0,up}: acts on the 4-state Fock basis {|0>,|↑>,|↓>,|↑↓>}
    //      mapping |0>→|↑>, |↓>→|↑↓>, others→0
    // 3. TdvpDriver time evolution of |α(t)> = e^{-iHt}|α>
    //    - Forward evolution with dt steps, soft D_max policy
    //    - Tikhonov stabilization for near-singular bond matrices
    // 4. Overlap computation G(t_k) = -i·<psi_0|c_{0,σ}|α(t_k)>
    //    - Requires mps_overlap between bra (with operator applied) and ket
    //
    // These operations are implemented in tk-dmrg but require a functional
    // sweep engine with environment contraction. When available, the
    // implementation would be:
    //
    //   let n_steps = (config.t_max / config.dt).ceil() as usize;
    //   let mut g_t = Vec::with_capacity(n_steps + 1);
    //
    //   // Initial overlap at t=0
    //   let g0 = mps_overlap(&bra_psi0_c, &alpha);
    //   g_t.push(Complex::new(0.0, -1.0) * g0);
    //
    //   // Time evolution loop
    //   let mut tdvp = TdvpDriver::new(alpha, mpo, backend, stabilization);
    //   for _ in 0..n_steps {
    //       tdvp.step(config.dt, config.max_bond_dim)?;
    //       let overlap = mps_overlap(&bra_psi0_c, tdvp.mps());
    //       g_t.push(Complex::new(0.0, -1.0) * overlap);
    //   }
    //
    //   Ok(g_t)

    Err(DmftError::Dmrg(tk_dmrg::DmrgError::NotImplemented(
        "TDVP Green's function requires functional DMRGEngine and TdvpDriver".into(),
    )))
}

/// Run the full TDVP + linear prediction spectral pipeline.
///
/// Stages:
/// 1. Compute G(t) via TDVP (`compute_greens_function_tdvp`)
/// 2. Extrapolate via linear prediction (`linear_predict_regularized`)
/// 3. FFT to frequency domain (`fft_to_spectral`)
/// 4. Deconvolve Lorentzian broadening (if eta > 0)
/// 5. Restore positivity (always)
///
/// # Parameters
/// - `tdvp_config`: TDVP time evolution parameters
/// - `lp_config`: linear prediction pipeline parameters
/// - `omega`: target frequency grid
///
/// # Errors
/// Propagates errors from TDVP computation and linear prediction.
pub fn tdvp_spectral_pipeline(
    tdvp_config: &TdvpSpectralConfig,
    lp_config: &LinearPredictionConfig,
    omega: &[f64],
) -> DmftResult<SpectralFunction> {
    // Step 1: Compute G(t) via TDVP
    let g_t = compute_greens_function_tdvp(tdvp_config)?;

    // Step 2: Linear prediction extrapolation
    let g_t_extended = linear_predict_regularized(&g_t, tdvp_config.dt, lp_config)?;

    // Step 3: FFT to spectral function
    let spectral = fft_to_spectral(&g_t_extended, tdvp_config.dt, omega);

    // Step 4: Deconvolve Lorentzian broadening (if eta > 0)
    let spectral = if lp_config.broadening_eta > 0.0 {
        deconvolve_lorentzian(&spectral, lp_config)?
    } else {
        spectral
    };

    // Step 5: Restore positivity (always mandatory)
    Ok(restore_positivity(&spectral, lp_config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tdvp_config_default() {
        let config = TdvpSpectralConfig::default();
        assert!((config.t_max - 20.0).abs() < 1e-12);
        assert!((config.dt - 0.05).abs() < 1e-12);
        assert_eq!(config.max_bond_dim, 500);
    }

    #[test]
    fn test_compute_greens_function_returns_error() {
        let config = TdvpSpectralConfig::default();
        let result = compute_greens_function_tdvp(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_tdvp_pipeline_propagates_error() {
        let tdvp_config = TdvpSpectralConfig::default();
        let lp_config = LinearPredictionConfig::default();
        let omega: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        let result = tdvp_spectral_pipeline(&tdvp_config, &lp_config, &omega);
        // Should fail because compute_greens_function_tdvp is not yet functional
        assert!(result.is_err());
    }
}
