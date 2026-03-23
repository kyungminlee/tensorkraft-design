//! Excited-state DMRG via penalty method.
//!
//! H_pen = H + weight · Σ_j |ψ_j⟩⟨ψ_j|

use tk_core::Scalar;
use tk_symmetry::{BitPackable, U1};

use crate::mps::{MPS, MixedCanonical};

/// Configuration for excited-state targeting.
pub struct ExcitedStateConfig {
    /// Previously converged states to penalize.
    pub penalized_states: Vec<MPS<f64, U1, MixedCanonical>>,
    /// Penalty weight (default: 0.1).
    pub penalty_weight: f64,
}

impl Default for ExcitedStateConfig {
    fn default() -> Self {
        ExcitedStateConfig {
            penalized_states: Vec::new(),
            penalty_weight: 0.1,
        }
    }
}

/// Build a penalized effective Hamiltonian matvec closure.
///
/// The closure applies: H_eff · x + weight · Σ_j ⟨ψ_j|x⟩ · |ψ_j⟩
///
/// Full implementation requires overlaps between the trial state
/// and each penalized state, computed via transfer matrices.
pub fn build_heff_penalized<'a>(
    base_matvec: Box<dyn Fn(&[f64], &mut [f64]) + 'a>,
    _config: &'a ExcitedStateConfig,
) -> Box<dyn Fn(&[f64], &mut [f64]) + 'a> {
    // Skeleton: just return the base matvec without penalty
    base_matvec
}
