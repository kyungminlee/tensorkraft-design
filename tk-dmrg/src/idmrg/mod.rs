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
/// energy per site converges, then maps to a finite-system MPS
/// of length `n_target_sites`.
///
/// # Algorithm
/// 1. Start with a 2-site MPS (one unit cell)
/// 2. For each extension:
///    a. Insert 2 new sites at center
///    b. Optimize center 2 sites via eigensolve + SVD
///    c. Compute energy per site = E_total / n_sites
///    d. Check convergence: |ΔE_per_site| < energy_tol_per_site
/// 3. Tile the converged unit cell to fill `n_target_sites`
/// 4. Return as MixedCanonical MPS
///
/// # Current Limitations
/// Full iDMRG requires building environments for the growing system and
/// managing unit-cell structure. This implementation provides the loop
/// structure but returns an error since the internal DMRG steps require
/// a fully-constructed MPO for the growing system.
pub fn run_idmrg<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>(
    _unit_cell_mpo: &MPO<T, Q>,
    n_target_sites: usize,
    config: &IDmrgConfig,
    _backend: &B,
) -> DmrgResult<MPS<T, Q, MixedCanonical>> {
    // The full iDMRG algorithm requires:
    // 1. A way to extend the MPO by inserting unit cells
    // 2. Environment management for the growing system
    // 3. Two-site optimization at the center
    //
    // These depend on MPO construction (Phase 8) which is not yet complete.
    // For now, we implement the convergence loop structure and return
    // an error indicating the algorithm could not converge.

    let mut _prev_energy_per_site = f64::INFINITY;

    for extension in 0..config.max_extensions {
        let _n_sites = 2 + 2 * extension;

        // Would build environments, optimize center, compute energy
        // Check convergence: |e_new - e_old| < tol
        // If converged, tile to n_target_sites and return

        if extension > 0 {
            // Convergence check placeholder
            break;
        }
    }

    // Return error since full implementation needs MPO construction
    Err(DmrgError::IDmrgConvergenceFailed {
        extensions: config.max_extensions,
    })
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
