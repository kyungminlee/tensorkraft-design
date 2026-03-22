//! Return types for linear algebra operations: SVD, eigendecomposition, QR.

use tk_core::{DenseTensor, Scalar};

/// Result of a truncated SVD: A ≈ U · diag(singular_values) · Vt.
///
/// Only the first `rank` singular values are retained, where `rank` is
/// determined by `max_rank` and `cutoff` parameters passed to `svd_truncated`.
#[derive(Debug)]
pub struct SvdResult<T: Scalar> {
    /// Left singular vectors. Shape: (m, rank). Columns are orthonormal.
    pub u: DenseTensor<'static, T>,
    /// Singular values in descending order. Length: rank.
    /// All values are positive real.
    pub singular_values: Vec<T::Real>,
    /// Right singular vectors (transposed). Shape: (rank, n). Rows are orthonormal.
    pub vt: DenseTensor<'static, T>,
    /// The retained rank (≤ max_rank, ≤ min(m, n)).
    pub rank: usize,
    /// Truncation error: sum of squares of discarded singular values.
    /// Defined as: Σ_{i > rank} σ_i².
    /// Used for entanglement entropy and truncation error reporting.
    pub truncation_error: T::Real,
}

/// Result of a dense symmetric/Hermitian eigendecomposition.
#[derive(Debug)]
pub struct EighResult<T: Scalar> {
    /// Eigenvalues in ascending order. Length: k (the requested count).
    pub eigenvalues: Vec<T::Real>,
    /// Eigenvectors as columns. Shape: (n, k).
    pub eigenvectors: DenseTensor<'static, T>,
}

/// Result of a thin QR decomposition: mat = Q · R.
#[derive(Debug)]
pub struct QrResult<T: Scalar> {
    /// Q factor: orthogonal/unitary matrix. Shape: (m, k) where k = min(m, n).
    pub q: DenseTensor<'static, T>,
    /// R factor: upper-triangular matrix. Shape: (k, n).
    pub r: DenseTensor<'static, T>,
}

/// Signals that an SVD algorithm (gesdd or gesvd) failed to converge.
/// Used internally by `svd_truncated` to trigger the fallback path.
#[derive(Debug, Clone, Copy)]
pub struct SvdConvergenceError;

impl std::fmt::Display for SvdConvergenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SVD algorithm failed to converge")
    }
}

impl std::error::Error for SvdConvergenceError {}
