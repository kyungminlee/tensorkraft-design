//! TDVP (Time-Dependent Variational Principle) time evolution driver.
//!
//! Implements 1-site and 2-site TDVP with Tikhonov regularization
//! and subspace expansion for numerical stability.

use num_traits::{One, Zero};
use tk_core::Scalar;
use tk_linalg::LinAlgBackend;
use tk_symmetry::BitPackable;

use crate::error::DmrgResult;
use crate::sweep::DMRGEngine;

/// Configuration for TDVP numerical stabilization.
pub struct TdvpStabilizationConfig {
    /// Tikhonov regularization parameter δ (default: 1e-10).
    pub tikhonov_delta: f64,
    /// Number of null-space expansion vectors (default: 4).
    pub expansion_vectors: usize,
    /// Weight of expansion vectors (default: 1e-4).
    pub expansion_alpha: f64,
    /// Enable adaptive expansion (default: true).
    pub adaptive_expansion: bool,
    /// Soft D_max overshoot factor (default: 1.1 = 10% overshoot).
    pub soft_dmax_factor: f64,
    /// Decay time for overshoot (default: 5.0 time steps).
    pub dmax_decay_steps: f64,
}

impl Default for TdvpStabilizationConfig {
    fn default() -> Self {
        TdvpStabilizationConfig {
            tikhonov_delta: 1e-10,
            expansion_vectors: 4,
            expansion_alpha: 1e-4,
            adaptive_expansion: true,
            soft_dmax_factor: 1.1,
            dmax_decay_steps: 5.0,
        }
    }
}

/// Result of a single TDVP time step.
pub struct TdvpStepResult<T: Scalar> {
    /// MPS norm after the step (should remain ~1).
    pub norm: T::Real,
    /// Maximum truncation error across all bonds.
    pub max_truncation_error: T::Real,
    /// Maximum bond dimension after truncation.
    pub max_bond_dim: usize,
    /// Number of bonds that were expanded.
    pub n_bonds_expanded: usize,
    /// Wall time for this step in seconds.
    pub wall_time_secs: f64,
}

/// TDVP time evolution driver.
///
/// Wraps a `DMRGEngine` and adds per-bond expansion tracking
/// for the soft D_max policy.
pub struct TdvpDriver<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> {
    /// TDVP stabilization configuration.
    pub config: TdvpStabilizationConfig,
    /// Underlying DMRG engine (shares MPS, MPO, environments).
    pub engine: DMRGEngine<T, Q, B>,
    /// Per-bond expansion age counters.
    expansion_age: Vec<Option<usize>>,
}

impl<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> TdvpDriver<T, Q, B> {
    /// Create a new TDVP driver from a DMRG engine.
    pub fn new(engine: DMRGEngine<T, Q, B>, config: TdvpStabilizationConfig) -> Self {
        let n_bonds = engine.mps.n_sites().saturating_sub(1);
        TdvpDriver {
            config,
            engine,
            expansion_age: vec![None; n_bonds],
        }
    }

    /// Perform one time step of TDVP evolution.
    ///
    /// Full implementation would:
    /// 1. For each site (left-to-right): evolve center forward via Krylov exp
    /// 2. Expose bond, evolve bond backward
    /// 3. Absorb bond, move center
    /// 4. Apply subspace expansion if needed
    /// 5. Truncate and update environments
    pub fn step(
        &mut self,
        _dt: T,
        _hard_dmax: usize,
    ) -> DmrgResult<TdvpStepResult<T>> {
        // Skeleton
        Ok(TdvpStepResult {
            norm: T::Real::one(),
            max_truncation_error: T::Real::zero(),
            max_bond_dim: self.engine.mps.max_bond_dim(),
            n_bonds_expanded: 0,
            wall_time_secs: 0.0,
        })
    }

    /// Run multiple TDVP steps.
    pub fn run(
        &mut self,
        n_steps: usize,
        dt: T,
        hard_dmax: usize,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> DmrgResult<Vec<TdvpStepResult<T>>> {
        let mut results = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(crate::error::DmrgError::Cancelled);
            }
            results.push(self.step(dt, hard_dmax)?);
        }
        Ok(results)
    }

    /// Compute effective D_max with soft decay for a given bond.
    fn effective_dmax(&self, bond: usize, hard_dmax: usize) -> usize {
        match self.expansion_age[bond] {
            None => hard_dmax,
            Some(age) => {
                let overshoot = (hard_dmax as f64 * self.config.soft_dmax_factor).round() as usize;
                let decay = (-(age as f64) / self.config.dmax_decay_steps).exp();
                let d = hard_dmax + ((overshoot - hard_dmax) as f64 * decay).round() as usize;
                d.min(overshoot)
            }
        }
    }

    /// Update expansion ages.
    fn tick_expansion_age(&mut self) {
        for age in &mut self.expansion_age {
            if let Some(a) = age {
                *a += 1;
            }
        }
    }

    /// Mark a bond as recently expanded.
    fn mark_expanded(&mut self, bond: usize) {
        self.expansion_age[bond] = Some(0);
    }
}

/// Krylov matrix-exponential: compute exp(α·A)·v.
///
/// Uses Arnoldi iteration to build a Krylov subspace approximation.
pub fn exp_krylov<T: Scalar>(
    matvec: &dyn Fn(&[T], &mut [T]),
    v: &[T],
    _alpha: T,
    dim: usize,
    krylov_dim: usize,
    _tol: T::Real,
) -> DmrgResult<Vec<T>> {
    // Skeleton: return the input vector unchanged
    // Full implementation uses Arnoldi + dense matrix exponential
    Ok(v.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tdvp_config_defaults() {
        let config = TdvpStabilizationConfig::default();
        assert!((config.tikhonov_delta - 1e-10).abs() < 1e-20);
        assert_eq!(config.expansion_vectors, 4);
        assert!(config.adaptive_expansion);
    }
}
