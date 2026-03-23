//! Error types for tensor contraction operations.

use crate::index::{IndexId, TensorId};

/// Error type for all contraction failures.
///
/// Each variant includes the context needed to diagnose physics-level bugs
/// (wrong tensor shapes, incompatible symmetry sectors) vs implementation
/// bugs (missing tensor IDs, algorithm failures).
#[derive(Debug, thiserror::Error)]
pub enum ContractionError {
    /// An `IndexId` appears on more than two tensors in the contraction spec.
    #[error("index {index:?} appears on {count} tensors; maximum is 2")]
    IndexAppearsTooManyTimes { index: IndexId, count: usize },

    /// Two legs on the same tensor share the same `IndexId`.
    #[error("tensor {tensor:?} has duplicate index {index:?} on legs {leg_a} and {leg_b}")]
    DuplicateIndexOnTensor {
        tensor: TensorId,
        index: IndexId,
        leg_a: usize,
        leg_b: usize,
    },

    /// An `output_indices` entry is a contracted (not free) index.
    #[error("index {index:?} is contracted and cannot appear in the output")]
    OutputIndexNotFree { index: IndexId },

    /// A contracted index pair has mismatched dimensions.
    #[error(
        "dimension mismatch on contracted index {index:?}: \
        tensor {tensor_a:?} leg {leg_a} has dim {dim_a}, \
        tensor {tensor_b:?} leg {leg_b} has dim {dim_b}"
    )]
    DimensionMismatch {
        index: IndexId,
        tensor_a: TensorId,
        leg_a: usize,
        dim_a: usize,
        tensor_b: TensorId,
        leg_b: usize,
        dim_b: usize,
    },

    /// A tensor required by the graph was not provided to the executor.
    #[error("tensor {0:?} is referenced in the contraction graph but not in the inputs map")]
    MissingTensor(TensorId),

    /// A tensor provided to the executor has a different shape than what was
    /// used to build the contraction graph.
    #[error(
        "shape mismatch for tensor {tensor:?}: plan expected {expected:?}, got {actual:?}"
    )]
    ShapeMismatch {
        tensor: TensorId,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Flux conservation violated during a block-sparse contraction step.
    #[error(
        "flux mismatch at contraction step: \
        fused flux {fused:?} does not match expected output flux {expected:?}"
    )]
    FluxMismatch { fused: String, expected: String },

    /// Path optimizer failed.
    #[error("path optimizer '{optimizer}' failed: {reason}")]
    OptimizerFailed { optimizer: String, reason: String },

    /// Internal: intermediate tensor consumed twice.
    #[error("internal: intermediate tensor {0:?} consumed twice in execution")]
    IntermediateConsumedTwice(TensorId),

    /// No tensors provided for contraction.
    #[error("contraction spec contains no tensors")]
    EmptySpec,

    /// Propagated error from `tk-core`.
    #[error(transparent)]
    Core(#[from] tk_core::TkError),

    /// Propagated error from `tk-symmetry`.
    #[error(transparent)]
    Symmetry(#[from] tk_symmetry::SymmetryError),

    /// Propagated error from `tk-linalg`.
    #[error(transparent)]
    LinAlg(#[from] tk_linalg::LinAlgError),
}

/// `Result` alias for tk-contract operations.
pub type ContractResult<T> = Result<T, ContractionError>;
