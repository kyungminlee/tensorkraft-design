//! SVD truncation with bond dimension scheduling.
//!
//! Controls the trade-off between accuracy and computational cost
//! by capping bond dimensions and discarding small singular values.

use num_traits::{NumCast, Zero};
use tk_core::{MatRef, Scalar};
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
/// Delegates to `LinAlgBackend::svd_truncated` and applies the truncation
/// criteria from `config` (max_bond_dim, svd_cutoff, min_bond_dim).
pub fn truncate_svd<T: Scalar, B: LinAlgBackend<T>>(
    matrix_data: &[T],
    rows: usize,
    cols: usize,
    config: &TruncationConfig,
    backend: &B,
) -> DmrgResult<TruncationResult<T>> {
    let mat = MatRef::from_slice(matrix_data, rows, cols);
    let cutoff: T::Real = NumCast::from(config.svd_cutoff).unwrap_or(T::Real::zero());

    // First pass: get full SVD up to min(m,n) to respect min_bond_dim
    let full_rank = rows.min(cols);
    let svd = backend.svd_truncated(&mat, full_rank, T::Real::zero())?;

    // Determine how many singular values to keep
    // Apply max_bond_dim and cutoff constraints
    let sigma_max = svd.singular_values.first().copied().unwrap_or(T::Real::zero());
    let threshold = cutoff * sigma_max;
    let mut bond_dim_new = 0;
    for &s in &svd.singular_values {
        if bond_dim_new >= config.max_bond_dim || s < threshold {
            break;
        }
        bond_dim_new += 1;
    }
    // Apply min_bond_dim: keep at least min_bond_dim
    bond_dim_new = bond_dim_new.max(config.min_bond_dim).min(svd.singular_values.len());

    // Compute truncation error: Σ_discarded σ² / Σ_all σ²
    let total_sq: T::Real = svd
        .singular_values
        .iter()
        .map(|&s| s * s)
        .fold(T::Real::zero(), |a, b| a + b);
    let kept_sq: T::Real = svd.singular_values[..bond_dim_new]
        .iter()
        .map(|&s| s * s)
        .fold(T::Real::zero(), |a, b| a + b);
    let truncation_error = if total_sq > T::Real::zero() {
        (total_sq - kept_sq) / total_sq
    } else {
        T::Real::zero()
    };

    // Extract U data (rows × bond_dim_new) from SvdResult U (rows × svd.rank)
    let u_src = svd.u.as_slice();
    let u_data: Vec<T> = (0..rows)
        .flat_map(|r| (0..bond_dim_new).map(move |c| u_src[r * svd.rank + c]))
        .collect();

    // Extract Vt data (bond_dim_new × cols) from SvdResult Vt (svd.rank × cols)
    let vt_src = svd.vt.as_slice();
    let vt_data: Vec<T> = (0..bond_dim_new)
        .flat_map(|r| (0..cols).map(move |c| vt_src[r * cols + c]))
        .collect();

    let singular_values = svd.singular_values[..bond_dim_new].to_vec();
    let n_discarded = svd.singular_values.len().saturating_sub(bond_dim_new);

    Ok(TruncationResult {
        u: u_data,
        u_rows: rows,
        u_cols: bond_dim_new,
        singular_values,
        vt: vt_data,
        vt_rows: bond_dim_new,
        vt_cols: cols,
        bond_dim_new,
        n_discarded,
        truncation_error,
    })
}

/// Schedule for ramping up bond dimension across sweeps.
#[derive(Clone, Debug)]
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
    fn truncate_svd_basic() {
        use tk_linalg::DeviceFaer;
        // Matrix: [[3, 0], [0, 2], [0, 0]]  (3x2, rank 2)
        let data = vec![3.0_f64, 0.0, 0.0, 2.0, 0.0, 0.0];
        let config = TruncationConfig::default();
        let backend = DeviceFaer;
        let result = truncate_svd(&data, 3, 2, &config, &backend).unwrap();
        assert_eq!(result.bond_dim_new, 2);
        assert_eq!(result.n_discarded, 0);
        assert!(result.truncation_error < 1e-14);
        assert!((result.singular_values[0] - 3.0).abs() < 1e-10);
        assert!((result.singular_values[1] - 2.0).abs() < 1e-10);
        assert_eq!(result.u_rows, 3);
        assert_eq!(result.u_cols, 2);
        assert_eq!(result.vt_rows, 2);
        assert_eq!(result.vt_cols, 2);
    }

    #[test]
    fn truncate_svd_with_truncation() {
        use tk_linalg::DeviceFaer;
        // Rank-3 matrix, truncate to bond dim 2
        let data = vec![3.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0];
        let config = TruncationConfig {
            max_bond_dim: 2,
            svd_cutoff: 1e-12,
            min_bond_dim: 1,
        };
        let backend = DeviceFaer;
        let result = truncate_svd(&data, 3, 3, &config, &backend).unwrap();
        assert_eq!(result.bond_dim_new, 2);
        assert_eq!(result.n_discarded, 1);
        // Truncation error = 1²/(3²+2²+1²) = 1/14
        assert!((result.truncation_error - 1.0 / 14.0).abs() < 1e-10);
    }

    #[test]
    fn truncate_svd_min_bond_dim() {
        use tk_linalg::DeviceFaer;
        // Matrix with one large and one tiny singular value
        let data = vec![10.0_f64, 0.0, 0.0, 1e-15];
        let config = TruncationConfig {
            max_bond_dim: 200,
            svd_cutoff: 1e-10, // cutoff would discard second SV
            min_bond_dim: 2,   // but min_bond_dim forces keeping it
        };
        let backend = DeviceFaer;
        let result = truncate_svd(&data, 2, 2, &config, &backend).unwrap();
        // min_bond_dim ensures we keep at least 2
        assert!(result.bond_dim_new >= 2);
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
