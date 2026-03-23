//! Matrix Product Operator (MPO) types and compilation from OpSum.
//!
//! MPO tensor leg ordering: (σ_in, σ_out, w_left, w_right)

use tk_core::Scalar;
use tk_symmetry::{BitPackable, BlockSparseTensor, LegDirection, QIndex};

use crate::error::{DmrgError, DmrgResult};

/// MPO compression configuration for OpSum → MPO compilation.
pub struct MpoCompressionConfig {
    /// Post-SVD compression bond dimension limit.
    pub max_bond_dim: usize,
    /// Singular value cutoff (default: 1e-12).
    pub svd_cutoff: f64,
    /// Enable debug-only compression validation.
    pub validate_compression: bool,
    /// Compression tolerance for validation (default: 1e-8).
    pub compression_tol: f64,
}

impl Default for MpoCompressionConfig {
    fn default() -> Self {
        MpoCompressionConfig {
            max_bond_dim: 50,
            svd_cutoff: 1e-12,
            validate_compression: cfg!(debug_assertions),
            compression_tol: 1e-8,
        }
    }
}

/// Matrix Product Operator.
///
/// Tensor leg ordering per site: (σ_in, σ_out, w_left, w_right)
pub struct MPO<T: Scalar, Q: BitPackable> {
    tensors: Vec<BlockSparseTensor<T, Q>>,
    local_dims: Vec<usize>,
    max_bond_dim: usize,
    flux: Q,
}

impl<T: Scalar, Q: BitPackable> MPO<T, Q> {
    /// Create an MPO from pre-built tensors.
    pub fn new(
        tensors: Vec<BlockSparseTensor<T, Q>>,
        local_dims: Vec<usize>,
        flux: Q,
    ) -> Self {
        let max_bond_dim = tensors
            .iter()
            .map(|t| t.indices().get(3).map_or(1, |idx| idx.total_dim()))
            .max()
            .unwrap_or(1);
        MPO {
            tensors,
            local_dims,
            max_bond_dim,
            flux,
        }
    }

    /// Number of sites.
    pub fn n_sites(&self) -> usize {
        self.tensors.len()
    }

    /// Physical dimension at a given site.
    pub fn local_dim(&self, site: usize) -> usize {
        self.local_dims[site]
    }

    /// MPO bond dimension at bond `site` (between site and site+1).
    pub fn mpo_bond_dim(&self, site: usize) -> usize {
        let t = &self.tensors[site];
        t.indices().get(3).map_or(1, |idx| idx.total_dim())
    }

    /// Maximum MPO bond dimension across all bonds.
    pub fn max_mpo_bond_dim(&self) -> usize {
        self.max_bond_dim
    }

    /// Access the site tensor.
    pub fn site_tensor(&self, site: usize) -> &BlockSparseTensor<T, Q> {
        &self.tensors[site]
    }

    /// Quantum number flux of the operator.
    pub fn flux(&self) -> &Q {
        &self.flux
    }
}

/// MPO compiler: converts OpSum into compressed MPO via FSA + SVD.
pub struct MpoCompiler<'b, T: Scalar, B> {
    backend: &'b B,
    config: MpoCompressionConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<'b, T: Scalar, B: tk_linalg::LinAlgBackend<T>> MpoCompiler<'b, T, B> {
    pub fn new(backend: &'b B, config: MpoCompressionConfig) -> Self {
        MpoCompiler {
            backend,
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compile an OpSum into a compressed MPO.
    ///
    /// The full FSA + SVD compilation is complex (transfer matrices, site-by-site
    /// SVD compression). This is a skeleton that returns an error indicating
    /// compilation is not yet implemented.
    pub fn compile<Q: BitPackable>(
        &self,
        _opsum: &tk_dsl::OpSum<T>,
        _n_sites: usize,
        _local_dims: &[usize],
        _flux: Q,
    ) -> DmrgResult<MPO<T, Q>> {
        Err(DmrgError::OpSumCompilationFailed {
            reason: "FSA + SVD MPO compilation not yet implemented".to_string(),
        })
    }

    /// Re-compress an existing MPO to lower bond dimension.
    pub fn compress<Q: BitPackable>(
        &self,
        _mpo: MPO<T, Q>,
    ) -> DmrgResult<MPO<T, Q>> {
        Err(DmrgError::OpSumCompilationFailed {
            reason: "MPO re-compression not yet implemented".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpo_compression_config_defaults() {
        let config = MpoCompressionConfig::default();
        assert_eq!(config.max_bond_dim, 50);
        assert!((config.svd_cutoff - 1e-12).abs() < 1e-20);
    }
}
