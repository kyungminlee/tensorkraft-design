//! Infinite DMRG (iDMRG) bootstrap for thermodynamic-limit initialization.
//!
//! Grows the system two sites at a time until thermodynamic-limit
//! energy-per-site converges, then maps to a finite system.

use tk_core::Scalar;
use tk_linalg::LinAlgBackend;
use tk_symmetry::BitPackable;

use crate::error::{DmrgError, DmrgResult};
use crate::mpo::MPO;
use crate::mps::{MPS, MixedCanonical};

/// Configuration for infinite DMRG bootstrap.
pub struct IDmrgConfig {
    /// Target bond dimension for the infinite system.
    pub target_bond_dim: usize,
    /// Energy-per-site convergence tolerance (default: 1e-10).
    pub energy_tol_per_site: f64,
    /// Maximum number of unit-cell extensions (default: 500).
    pub max_extensions: usize,
}

impl Default for IDmrgConfig {
    fn default() -> Self {
        IDmrgConfig {
            target_bond_dim: 200,
            energy_tol_per_site: 1e-10,
            max_extensions: 500,
        }
    }
}

/// Run infinite DMRG to bootstrap a finite MPS.
///
/// Grows the system by inserting 2-site unit cells until the
/// energy per site converges, then truncates to `n_target_sites`.
pub fn run_idmrg<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>(
    _unit_cell_mpo: &MPO<T, Q>,
    _n_target_sites: usize,
    _config: &IDmrgConfig,
    _backend: &B,
) -> DmrgResult<MPS<T, Q, MixedCanonical>> {
    // Skeleton: full implementation would iteratively grow the system
    Err(DmrgError::IDmrgConvergenceFailed { extensions: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idmrg_config_defaults() {
        let config = IDmrgConfig::default();
        assert_eq!(config.target_bond_dim, 200);
        assert_eq!(config.max_extensions, 500);
    }
}
