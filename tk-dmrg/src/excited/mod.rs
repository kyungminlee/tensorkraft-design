//! Excited-state DMRG via penalty method.
//!
//! H_pen = H + weight · Σ_j |ψ_j⟩⟨ψ_j|
//!
//! The penalty term shifts converged eigenstates upward in the spectrum,
//! so the next eigensolve finds the next-lowest state. The overlap vectors
//! are the projections of each penalized state into the effective Hilbert
//! space at the current two-site (or single-site) block.

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
/// `overlap_vectors` contains the projection of each penalized state
/// into the effective Hilbert space at the current DMRG step. These
/// are computed via transfer matrix contractions of the penalized MPS
/// with the current left/right environments.
///
/// For the full pipeline:
/// 1. For each penalized state j, contract its MPS tensors at the active
///    sites with the current left and right environments to get |ψ_j_eff⟩.
/// 2. Pass these vectors as `overlap_vectors`.
/// 3. The returned closure adds the penalty: H_pen·x = H·x + w·Σ_j ⟨ψ_j|x⟩·|ψ_j⟩.
pub fn build_heff_penalized<'a>(
    base_matvec: Box<dyn Fn(&[f64], &mut [f64]) + 'a>,
    weight: f64,
    overlap_vectors: Vec<Vec<f64>>,
) -> Box<dyn Fn(&[f64], &mut [f64]) + 'a> {
    if overlap_vectors.is_empty() || weight == 0.0 {
        return base_matvec;
    }

    Box::new(move |x: &[f64], y: &mut [f64]| {
        // Apply base H_eff
        base_matvec(x, y);

        // Add penalty: y += weight * Σ_j ⟨ψ_j|x⟩ * |ψ_j⟩
        for psi_j in &overlap_vectors {
            debug_assert_eq!(psi_j.len(), x.len());
            let overlap: f64 = psi_j.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            for (yi, psi_ji) in y.iter_mut().zip(psi_j.iter()) {
                *yi += weight * overlap * psi_ji;
            }
        }
    })
}

/// Legacy wrapper for the skeleton API — delegates to `build_heff_penalized`
/// with empty overlap vectors (penalty has no effect).
pub fn build_heff_penalized_from_config<'a>(
    base_matvec: Box<dyn Fn(&[f64], &mut [f64]) + 'a>,
    config: &'a ExcitedStateConfig,
) -> Box<dyn Fn(&[f64], &mut [f64]) + 'a> {
    // Full implementation would compute overlap vectors from config.penalized_states
    // via transfer matrix contractions. For now, pass empty overlaps.
    build_heff_penalized(base_matvec, config.penalty_weight, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn penalized_matvec_shifts_eigenvalue() {
        // H = diag(1, 2, 3). Ground state is |0⟩ with eigenvalue 1.
        // Penalize |0⟩ with weight 10 → H_pen = H + 10·|0⟩⟨0| = diag(11, 2, 3).
        // New ground state should be |1⟩ with eigenvalue 2.
        let base_matvec: Box<dyn Fn(&[f64], &mut [f64])> = Box::new(|x: &[f64], y: &mut [f64]| {
            y[0] = 1.0 * x[0];
            y[1] = 2.0 * x[1];
            y[2] = 3.0 * x[2];
        });

        let psi_0 = vec![1.0, 0.0, 0.0]; // ground state to penalize
        let penalized = build_heff_penalized(base_matvec, 10.0, vec![psi_0]);

        // Apply to |0⟩: should give (1 + 10)·|0⟩ = 11·|0⟩
        let mut y = vec![0.0; 3];
        penalized(&[1.0, 0.0, 0.0], &mut y);
        assert!((y[0] - 11.0).abs() < 1e-10);
        assert!(y[1].abs() < 1e-10);
        assert!(y[2].abs() < 1e-10);

        // Apply to |1⟩: should give 2·|1⟩ (no penalty)
        penalized(&[0.0, 1.0, 0.0], &mut y);
        assert!(y[0].abs() < 1e-10);
        assert!((y[1] - 2.0).abs() < 1e-10);
        assert!(y[2].abs() < 1e-10);
    }

    #[test]
    fn penalized_matvec_no_penalty_states() {
        let base_matvec: Box<dyn Fn(&[f64], &mut [f64])> = Box::new(|x: &[f64], y: &mut [f64]| {
            y.copy_from_slice(x);
        });

        let penalized = build_heff_penalized(base_matvec, 10.0, Vec::new());

        let mut y = vec![0.0; 3];
        penalized(&[1.0, 2.0, 3.0], &mut y);
        assert!((y[0] - 1.0).abs() < 1e-10);
        assert!((y[1] - 2.0).abs() < 1e-10);
        assert!((y[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn penalized_multiple_states() {
        // Penalize both |0⟩ and |1⟩ with weight 100
        let base_matvec: Box<dyn Fn(&[f64], &mut [f64])> = Box::new(|x: &[f64], y: &mut [f64]| {
            y[0] = 1.0 * x[0];
            y[1] = 2.0 * x[1];
            y[2] = 3.0 * x[2];
        });

        let psi_0 = vec![1.0, 0.0, 0.0];
        let psi_1 = vec![0.0, 1.0, 0.0];
        let penalized = build_heff_penalized(base_matvec, 100.0, vec![psi_0, psi_1]);

        // Apply to |2⟩: no penalty, should give 3·|2⟩
        let mut y = vec![0.0; 3];
        penalized(&[0.0, 0.0, 1.0], &mut y);
        assert!(y[0].abs() < 1e-10);
        assert!(y[1].abs() < 1e-10);
        assert!((y[2] - 3.0).abs() < 1e-10);

        // Apply to |0⟩: should give (1 + 100)·|0⟩ = 101
        penalized(&[1.0, 0.0, 0.0], &mut y);
        assert!((y[0] - 101.0).abs() < 1e-10);
    }
}
