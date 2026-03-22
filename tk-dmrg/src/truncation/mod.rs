//! SVD truncation with bond dimension scheduling.
//!
//! Controls the trade-off between accuracy and computational cost
//! by capping bond dimensions and discarding small singular values.

use num_traits::{One, Zero};
use tk_core::Scalar;
use tk_linalg::LinAlgBackend;

use crate::error::DmrgResult;

/// Configuration for SVD truncation.
pub struct TruncationConfig {
    /// Hard upper bound on retained bond dimension.
    pub max_bond_dim: usize,
    /// Absolute singular value cutoff (default: 1e-12).
    pub svd_cutoff: f64,
    /// Minimum bond dimension to retain (default: 1).
    pub min_bond_dim: usize,
}

impl Default for TruncationConfig {
    fn default() -> Self {
        TruncationConfig {
            max_bond_dim: 200,
            svd_cutoff: 1e-12,
            min_bond_dim: 1,
        }
    }
}

/// Result of SVD truncation.
pub struct TruncationResult<T: Scalar> {
    /// Left isometric factor (rows × D_new).
    pub u: Vec<T>,
    pub u_rows: usize,
    pub u_cols: usize,
    /// Retained singular values in descending order.
    pub singular_values: Vec<T::Real>,
    /// Right isometric factor (D_new × cols).
    pub vt: Vec<T>,
    pub vt_rows: usize,
    pub vt_cols: usize,
    /// New bond dimension after truncation.
    pub bond_dim_new: usize,
    /// Number of discarded singular values.
    pub n_discarded: usize,
    /// Truncation error: Σ_discarded σ_i² / Σ_all σ_i².
    pub truncation_error: T::Real,
}

/// Perform SVD truncation on a matrix.
///
/// Full implementation calls `backend.svd_truncated()` and then applies
/// the cutoff and max_bond_dim constraints.
pub fn truncate_svd<T: Scalar, B: LinAlgBackend<T>>(
    _matrix_data: &[T],
    _rows: usize,
    _cols: usize,
    config: &TruncationConfig,
    _backend: &B,
) -> DmrgResult<TruncationResult<T>> {
    // Skeleton: return a trivial result
    Ok(TruncationResult {
        u: vec![T::one()],
        u_rows: 1,
        u_cols: 1,
        singular_values: vec![T::Real::one()],
        vt: vec![T::one()],
        vt_rows: 1,
        vt_cols: 1,
        bond_dim_new: 1,
        n_discarded: 0,
        truncation_error: T::Real::zero(),
    })
}

/// Schedule for ramping up bond dimension across sweeps.
pub struct BondDimensionSchedule {
    dims: Vec<usize>,
}

impl BondDimensionSchedule {
    /// Geometric warmup ramp from `d_init` to `d_max` over `n_warmup_sweeps`.
    pub fn warmup(d_init: usize, d_max: usize, n_warmup_sweeps: usize) -> Self {
        if n_warmup_sweeps == 0 {
            return BondDimensionSchedule { dims: vec![d_max] };
        }

        let ratio = (d_max as f64 / d_init as f64).powf(1.0 / n_warmup_sweeps as f64);
        let dims: Vec<usize> = (0..=n_warmup_sweeps)
            .map(|i| {
                let d = (d_init as f64 * ratio.powi(i as i32)).round() as usize;
                d.min(d_max).max(1)
            })
            .collect();

        BondDimensionSchedule { dims }
    }

    /// Fixed bond dimension for all sweeps.
    pub fn fixed(d: usize) -> Self {
        BondDimensionSchedule { dims: vec![d] }
    }

    /// Custom per-sweep schedule.
    pub fn custom(dims: Vec<usize>) -> Self {
        BondDimensionSchedule { dims }
    }

    /// Get the bond dimension for a given sweep index.
    /// Clamps to the last value for sweeps beyond the schedule.
    pub fn bond_dim_at_sweep(&self, sweep_index: usize) -> usize {
        if sweep_index < self.dims.len() {
            self.dims[sweep_index]
        } else {
            *self.dims.last().unwrap_or(&1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_config_defaults() {
        let config = TruncationConfig::default();
        assert_eq!(config.max_bond_dim, 200);
        assert_eq!(config.min_bond_dim, 1);
    }

    #[test]
    fn schedule_warmup_monotonic() {
        let sched = BondDimensionSchedule::warmup(10, 200, 5);
        for i in 1..sched.dims.len() {
            assert!(sched.dims[i] >= sched.dims[i - 1]);
        }
        assert_eq!(*sched.dims.last().unwrap(), 200);
    }

    #[test]
    fn schedule_fixed() {
        let sched = BondDimensionSchedule::fixed(100);
        assert_eq!(sched.bond_dim_at_sweep(0), 100);
        assert_eq!(sched.bond_dim_at_sweep(999), 100);
    }

    #[test]
    fn schedule_clamps_beyond() {
        let sched = BondDimensionSchedule::custom(vec![10, 20, 30]);
        assert_eq!(sched.bond_dim_at_sweep(0), 10);
        assert_eq!(sched.bond_dim_at_sweep(2), 30);
        assert_eq!(sched.bond_dim_at_sweep(100), 30);
    }
}
