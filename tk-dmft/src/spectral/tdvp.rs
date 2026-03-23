//! TDVP-based real-time Green's function computation.
//!
//! Pipeline: apply c^dag|psi_0> -> TDVP time evolution -> sample G(t_k) -> return.
//!
//! This module provides the TDVP spectral pipeline orchestration.
//! The actual TDVP time evolution is delegated to `tk-dmrg`'s `TdvpDriver`.

use num_complex::Complex;

use crate::error::DmftResult;

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

/// Placeholder for the full TDVP-based G(t) computation.
///
/// The complete pipeline:
/// 1. Construct |alpha> = c^dag_{0,sigma}|psi_0> from the DMRG ground state.
/// 2. Run `TdvpDriver::run(n_steps, dt, max_bond_dim, cancel)`.
/// 3. Sample G(t_k) = <psi_0|c_{0,sigma}|alpha(t_k)> at each step.
///
/// This requires `DMRGEngine` and `TdvpDriver` from tk-dmrg, and the ability
/// to apply fermionic operators to MPS states. These operations depend on
/// the full DMRG sweep engine being implemented.
///
/// For draft: returns a stub error.
///
/// # Returns
/// Complex Green's function samples at uniform time steps 0, dt, 2dt, ..., t_max.
pub fn compute_greens_function_tdvp(
    _config: &TdvpSpectralConfig,
) -> DmftResult<Vec<Complex<f64>>> {
    // TODO: Implement when DMRGEngine sweep, environment contraction,
    // and TdvpDriver time evolution are functional in tk-dmrg.
    //
    // Required from tk-dmrg:
    // - DMRGEngine::run() — DMRG ground state
    // - TdvpDriver::step() — single TDVP step
    // - MPS operator application (c^dag|psi>)
    // - MPS overlap computation (<psi|phi>)

    unimplemented!(
        "TDVP Green's function computation requires functional \
         DMRGEngine and TdvpDriver from tk-dmrg"
    )
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
}
