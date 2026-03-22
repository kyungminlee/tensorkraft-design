//! Error types for `tk-linalg`.

/// Errors produced by `tk-linalg` operations.
#[derive(Debug, thiserror::Error)]
pub enum LinAlgError {
    /// Both `gesdd` and `gesvd` failed to converge. Extremely rare;
    /// indicates a pathologically ill-conditioned input matrix.
    #[error("SVD failed to converge (both gesdd and gesvd diverged)")]
    SvdConvergence,

    /// The requested number of eigenvalues/vectors exceeds the matrix size.
    #[error("requested k={k} eigenvalues but matrix has only {n} rows")]
    EighKTooLarge { k: usize, n: usize },

    /// A matrix dimension is incompatible with the requested operation.
    #[error("dimension mismatch in {op}: expected {expected:?}, got {got:?}")]
    DimensionMismatch {
        op: &'static str,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// A non-unit stride was passed to a BLAS backend that requires contiguous layout.
    /// The caller must materialize the tensor via `TensorCow::into_owned` first.
    #[error("non-unit stride passed to BLAS backend: use into_owned() first")]
    NonContiguousForBlas,

    /// CUDA operation returned an error (backend-cuda only).
    #[cfg(feature = "backend-cuda")]
    #[error("CUDA error in {op}: {detail}")]
    CudaError { op: &'static str, detail: String },

    /// Wraps errors from the tk-core layer.
    #[error(transparent)]
    Core(#[from] tk_core::TkError),
}

/// Convenience alias for `Result<T, LinAlgError>`.
pub type LinAlgResult<T> = Result<T, LinAlgError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_svd_convergence() {
        let err = LinAlgError::SvdConvergence;
        assert!(err.to_string().contains("SVD failed to converge"));
    }

    #[test]
    fn error_display_eigh_k_too_large() {
        let err = LinAlgError::EighKTooLarge { k: 10, n: 5 };
        let msg = err.to_string();
        assert!(msg.contains("10"), "should contain k value: {msg}");
        assert!(msg.contains("5"), "should contain n value: {msg}");
    }

    #[test]
    fn error_display_dimension_mismatch() {
        let err = LinAlgError::DimensionMismatch {
            op: "gemm",
            expected: vec![3, 4],
            got: vec![3, 5],
        };
        assert!(err.to_string().contains("gemm"));
    }
}
