//! Error types for the DMRG engine.

use tk_core::TkError;
use tk_linalg::LinAlgError;
use tk_symmetry::SymmetryError;

/// All errors that can arise from `tk-dmrg` operations.
#[derive(Debug, thiserror::Error)]
pub enum DmrgError {
    #[error("shape mismatch: {context}")]
    ShapeMismatch { context: String },

    #[error("dimension mismatch: MPS has {mps_sites} sites, MPO has {mpo_sites} sites")]
    DimensionMismatch { mps_sites: usize, mpo_sites: usize },

    #[error("site index {site} out of bounds for MPS of length {n_sites}")]
    SiteBoundsError { site: usize, n_sites: usize },

    #[error("charge sector unreachable: {charge}")]
    ChargeSectorEmpty { charge: String },

    #[error("eigensolver did not converge after {iters} matvec calls; residual = {residual:.2e}")]
    EigensolverNotConverged { iters: usize, residual: f64 },

    #[error("OpSum compilation failed: {reason}")]
    OpSumCompilationFailed { reason: String },

    #[error("TDVP Krylov matrix-exponential did not converge in {krylov_dim} steps")]
    TdvpKrylovNotConverged { krylov_dim: usize },

    #[error("infinite DMRG convergence failed after {extensions} unit-cell extensions")]
    IDmrgConvergenceFailed { extensions: usize },

    #[error("bond singular values unavailable: MPS is not in BondCentered form")]
    BondSingularValuesUnavailable,

    #[error("checkpoint I/O error: {0}")]
    CheckpointIo(#[from] std::io::Error),

    #[error("checkpoint deserialization error: {0}")]
    CheckpointDeser(String),

    #[error("linear algebra error")]
    Linalg(#[from] LinAlgError),

    #[error("symmetry error")]
    Symmetry(#[from] SymmetryError),

    #[error("tensor core error")]
    Core(#[from] TkError),

    #[error("computation cancelled")]
    Cancelled,

    #[error("not implemented: {0}")]
    NotImplemented(String),
}

pub type DmrgResult<T> = Result<T, DmrgError>;
