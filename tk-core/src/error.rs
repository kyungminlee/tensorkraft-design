//! Shared error types for the tensorkraft workspace.

/// Top-level error type for tk-core and, by re-export, the entire workspace.
#[derive(Debug, thiserror::Error)]
pub enum TkError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("reshape failed: {numel_src} elements cannot reshape to {dims_dst:?}")]
    ReshapeError {
        numel_src: usize,
        dims_dst: Vec<usize>,
    },

    #[error("non-contiguous tensor: operation requires contiguous memory layout")]
    NonContiguous,

    #[error("rank error: expected rank {expected}, got rank {got}")]
    RankError { expected: usize, got: usize },
}

/// Convenience alias used throughout the workspace.
pub type TkResult<T> = Result<T, TkError>;
