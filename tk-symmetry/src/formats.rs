//! Sparsity format registry.

/// Sparsity strategy for a tensor or matrix.
///
/// Tracks which sparsity strategy is appropriate for a given tensor,
/// enabling `tk-linalg` to select the correct kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparsityFormat {
    /// Contiguous 1-D buffer; standard BLAS path.
    Dense,
    /// Block-sparse with packed sector keys; Abelian symmetry path.
    BlockSparse,
    /// Compressed sparse row/column; for irregular geometry operators.
    ElementSparse,
    /// Diagonal only; for identity operators and local terms.
    Diagonal,
}
