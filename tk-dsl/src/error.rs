//! Error types for the DSL layer.

/// All errors that can arise from `tk-dsl` operations.
#[derive(Debug, thiserror::Error)]
pub enum DslError {
    #[error("duplicate index tag '{tag}' in registry")]
    DuplicateIndexTag { tag: String },

    #[error("dimension mismatch on contracting index '{tag}': {dim_a} != {dim_b}")]
    DimensionMismatch {
        tag: String,
        dim_a: usize,
        dim_b: usize,
    },

    #[error("no contracting indices found between the two tensors")]
    NoContractingIndices,

    #[error("ambiguous contraction: index '{tag}' appears on more than two legs")]
    AmbiguousContraction { tag: String },

    #[error("site index {site} out of bounds for lattice with {n_sites} sites")]
    SiteOutOfBounds { site: usize, n_sites: usize },

    #[error("operator local_dim mismatch: operator '{name}' expects dim {expected}, lattice provides dim {got}")]
    LocalDimMismatch {
        name: String,
        expected: usize,
        got: usize,
    },

    #[error("empty operator product: OpProduct must contain at least one OpTerm")]
    EmptyProduct,

    #[error("boson operator requires n_max > 0")]
    InvalidBosonNMax,

    #[error(transparent)]
    Core(#[from] tk_core::TkError),
}

pub type DslResult<T> = Result<T, DslError>;
