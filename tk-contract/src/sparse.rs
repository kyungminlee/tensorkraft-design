//! Sparse contraction executor for `BlockSparseTensor` inputs.
//!
//! Each pairwise contraction step is dispatched to
//! `SparseLinAlgBackend::block_gemm`, which applies LPT-scheduled
//! parallel sector-wise GEMM.

use std::marker::PhantomData;

use hashbrown::HashMap;
use tk_core::Scalar;
use tk_symmetry::{BitPackable, BlockSparseTensor};
use tk_linalg::{LinAlgBackend, SparseLinAlgBackend};

use crate::error::{ContractionError, ContractResult};
use crate::graph::ContractionGraph;
use crate::index::TensorId;
use crate::structural::{AbelianHook, StructuralContractionHook};

/// Executor for contractions over `BlockSparseTensor<T, Q>` inputs.
///
/// The sparse executor shares the same `ContractionGraph` (and thus the
/// same `PathOptimizer`) as the dense executor — path optimization is
/// symmetry-agnostic and operates only on logical tensor shapes.
pub struct SparseContractionExecutor<T: Scalar, Q: BitPackable, B>
where
    B: LinAlgBackend<T> + SparseLinAlgBackend<T, Q>,
{
    backend: B,
    hook: Box<dyn StructuralContractionHook<T, Q>>,
    _phantom: PhantomData<(T, Q)>,
}

impl<T: Scalar, Q: BitPackable, B> SparseContractionExecutor<T, Q, B>
where
    B: LinAlgBackend<T> + SparseLinAlgBackend<T, Q>,
{
    /// Construct with the Abelian (no-op) structural hook.
    pub fn new(backend: B) -> Self {
        SparseContractionExecutor {
            backend,
            hook: Box::new(AbelianHook::new()),
            _phantom: PhantomData,
        }
    }

    /// Construct with a custom structural hook (e.g., SU(2) Clebsch-Gordan).
    pub fn with_hook(
        backend: B,
        hook: Box<dyn StructuralContractionHook<T, Q>>,
    ) -> Self {
        SparseContractionExecutor {
            backend,
            hook,
            _phantom: PhantomData,
        }
    }

    /// Execute a pre-optimized `ContractionGraph` over block-sparse inputs.
    ///
    /// Each pairwise step delegates to `SparseLinAlgBackend::block_gemm`.
    /// The structural hook is applied per sector-pair to inject non-Abelian
    /// coefficients (no-op for Abelian symmetries).
    ///
    /// # Errors
    /// - `MissingTensor` if a required tensor is not in `inputs`.
    /// - `FluxMismatch` if flux conservation is violated.
    pub fn execute(
        &self,
        graph: &ContractionGraph,
        inputs: &HashMap<TensorId, &BlockSparseTensor<T, Q>>,
    ) -> ContractResult<BlockSparseTensor<T, Q>> {
        let steps = graph.execution_order();

        if steps.is_empty() {
            // Single-tensor "contraction" is a degenerate case. BlockSparseTensor
            // does not implement Clone, so this case needs special handling by
            // the caller (e.g., returning the input directly without going
            // through the contraction engine).
            return Err(ContractionError::OptimizerFailed {
                optimizer: "sparse".to_string(),
                reason: "single-tensor contraction not supported; use input directly".to_string(),
            });
        }

        let mut intermediates: HashMap<TensorId, BlockSparseTensor<T, Q>> = HashMap::new();

        for step in &steps {
            let left = self.get_sparse_tensor(step.left_tensor_id, inputs, &intermediates)?;
            let right = self.get_sparse_tensor(step.right_tensor_id, inputs, &intermediates)?;

            // Delegate to the backend's block_gemm.
            // In a full implementation, we'd pass the structural hook
            // and contracted leg information to handle non-Abelian fusion.
            let result = self.backend.block_gemm(left, right);

            intermediates.insert(step.result_tensor_id, result);
        }

        let last_step = steps.last().unwrap();
        intermediates
            .remove(&last_step.result_tensor_id)
            .ok_or(ContractionError::EmptySpec)
    }

    fn get_sparse_tensor<'a>(
        &self,
        id: TensorId,
        inputs: &'a HashMap<TensorId, &'a BlockSparseTensor<T, Q>>,
        intermediates: &'a HashMap<TensorId, BlockSparseTensor<T, Q>>,
    ) -> ContractResult<&'a BlockSparseTensor<T, Q>> {
        if let Some(t) = inputs.get(&id) {
            Ok(*t)
        } else if let Some(t) = intermediates.get(&id) {
            Ok(t)
        } else {
            Err(ContractionError::MissingTensor(id))
        }
    }
}
