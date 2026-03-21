//! Error types for `tk-symmetry`.

use tk_core::TkError;

/// Errors produced by symmetry operations.
#[derive(Debug, thiserror::Error)]
pub enum SymmetryError {
    #[error(
        "flux rule violated: sector {sector:?} has fused charge {actual:?}, expected {expected:?}"
    )]
    FluxRuleViolation {
        sector: Vec<String>,
        actual: String,
        expected: String,
    },

    #[error(
        "sector key overflow: rank {rank} × {bit_width} bits = {total} bits exceeds 64; \
         use PackedSectorKey128"
    )]
    SectorKeyOverflow {
        rank: usize,
        bit_width: usize,
        total: usize,
    },

    #[error("sector not found: quantum numbers {qns:?} not present in this tensor")]
    SectorNotFound { qns: Vec<String> },

    #[error("leg dimension mismatch on leg {leg}: expected {expected}, got {got}")]
    LegDimensionMismatch {
        leg: usize,
        expected: usize,
        got: usize,
    },

    #[error("incompatible quantum number types in operation")]
    QuantumNumberTypeMismatch,

    #[error(transparent)]
    Core(#[from] TkError),
}

/// Convenience alias for `Result<T, SymmetryError>`.
pub type SymResult<T> = Result<T, SymmetryError>;
