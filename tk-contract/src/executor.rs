//! Dense contraction executor and cached execution plan.

use std::marker::PhantomData;

use hashbrown::HashMap;
use tk_core::{DenseTensor, MatMut, MatRef, Scalar, TensorShape};
use tk_linalg::LinAlgBackend;

use crate::cost::CostMetric;
use crate::error::{ContractionError, ContractResult};
use crate::graph::{ContractionGraph, PairwiseStep};
use crate::index::{ContractionSpec, IndexMap, TensorId};
use crate::optimizer::PathOptimizer;

/// Safe wrapper for tensors that may be either borrowed inputs or owned intermediates.
///
/// Eliminates the need for `unsafe` lifetime transmute when the executor holds
/// both borrowed input tensors and owned intermediate results simultaneously.
/// See techspec §7.1 Option B.
enum TensorRef<'a, T: Scalar> {
    Borrowed(&'a DenseTensor<'a, T>),
    Owned(DenseTensor<'static, T>),
}

impl<'a, T: Scalar> TensorRef<'a, T> {
    fn as_dense(&self) -> &DenseTensor<'_, T> {
        match self {
            TensorRef::Borrowed(t) => t,
            TensorRef::Owned(t) => t,
        }
    }

    fn into_owned(self) -> Option<DenseTensor<'static, T>> {
        match self {
            TensorRef::Borrowed(_) => None,
            TensorRef::Owned(t) => Some(t),
        }
    }
}

/// Executor for dense tensor contractions via `LinAlgBackend::gemm`.
///
/// Converts each `PairwiseStep` from the `ContractionGraph` into a matrix
/// multiply by:
/// 1. Fusing contracted legs into a single "K" axis and free legs into "M"/"N".
/// 2. Calling `backend.gemm(alpha, &a_mat, &b_mat, beta, &mut c_mat)`.
/// 3. Unfolding the result back into a tensor with correct axis ordering.
pub struct ContractionExecutor<T: Scalar, B: LinAlgBackend<T>> {
    backend: B,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> ContractionExecutor<T, B> {
    /// Construct a new executor wrapping the given backend.
    pub fn new(backend: B) -> Self {
        ContractionExecutor {
            backend,
            _phantom: PhantomData,
        }
    }

    /// Execute a pre-optimized `ContractionGraph` over `inputs`.
    ///
    /// All intermediate tensors are heap-allocated (in a full implementation
    /// these would come from the `SweepArena`).
    ///
    /// # Errors
    /// - `MissingTensor` if an input tensor is not found in the map.
    /// - `ShapeMismatch` if a tensor has unexpected dimensions.
    pub fn execute(
        &self,
        graph: &ContractionGraph,
        inputs: &HashMap<TensorId, &DenseTensor<'_, T>>,
    ) -> ContractResult<DenseTensor<'static, T>> {
        let steps = graph.execution_order();

        if steps.is_empty() {
            // Single tensor, no contraction needed.
            let tid = graph.inputs.first().ok_or(ContractionError::EmptySpec)?;
            let tensor = inputs
                .get(tid)
                .ok_or(ContractionError::MissingTensor(*tid))?;
            // Single tensor: clone into owned storage.
            let owned = DenseTensor::from_vec(
                TensorShape::row_major(tensor.shape().dims()),
                tensor.as_slice()[..tensor.numel()].to_vec(),
            );
            return Ok(owned);
        }

        // Unified storage using TensorRef enum — no unsafe transmute needed.
        // Inputs are wrapped as Borrowed, intermediates as Owned.
        let mut tensors: HashMap<TensorId, TensorRef<'_, T>> = HashMap::new();
        for (&tid, &tensor) in inputs.iter() {
            tensors.insert(tid, TensorRef::Borrowed(tensor));
        }

        for step in &steps {
            let left = tensors
                .get(&step.left_tensor_id)
                .ok_or(ContractionError::MissingTensor(step.left_tensor_id))?
                .as_dense();
            let right = tensors
                .get(&step.right_tensor_id)
                .ok_or(ContractionError::MissingTensor(step.right_tensor_id))?
                .as_dense();

            let result = self.execute_pairwise(left, right, step)?;

            tensors.insert(step.result_tensor_id, TensorRef::Owned(result));
        }

        // Extract the final result.
        let last_step = steps.last().unwrap();
        tensors
            .remove(&last_step.result_tensor_id)
            .ok_or(ContractionError::EmptySpec)?
            .into_owned()
            .ok_or(ContractionError::EmptySpec)
    }

    /// Execute a single pairwise contraction step.
    fn execute_pairwise(
        &self,
        left: &DenseTensor<'_, T>,
        right: &DenseTensor<'_, T>,
        step: &PairwiseStep,
    ) -> ContractResult<DenseTensor<'static, T>> {
        let left_shape = left.shape();
        let right_shape = right.shape();
        let left_dims = left_shape.dims();
        let right_dims = right_shape.dims();

        // Compute M, K, N from the step's contracted legs.
        let mut m: usize = 1;
        let mut k: usize = 1;
        let mut n: usize = 1;

        for (pos, &dim) in left_dims.iter().enumerate() {
            if step.left_contracted_legs.contains(&pos) {
                k *= dim;
            } else {
                m *= dim;
            }
        }

        for (pos, &dim) in right_dims.iter().enumerate() {
            if step.right_contracted_legs.contains(&pos) {
                // Already counted in k
            } else {
                n *= dim;
            }
        }

        // Gather left and right data into contiguous M×K and K×N matrices.
        // In a full implementation this would use tensor_to_mat_ref with
        // arena-backed transpose. For now we use the simple gather approach.
        let left_data = gather_for_gemm(left, &step.left_contracted_legs, true);
        let right_data = gather_for_gemm(right, &step.right_contracted_legs, false);

        let a_mat = MatRef::from_slice(&left_data, m, k);
        let b_mat = MatRef::from_slice(&right_data, k, n);

        let mut c_data = vec![T::zero(); m * n];
        let mut c_mat = MatMut {
            data: &mut c_data,
            rows: m,
            cols: n,
            row_stride: n as isize,
            col_stride: 1,
        };

        self.backend
            .gemm(T::one(), &a_mat, &b_mat, T::zero(), &mut c_mat);

        // Build output dims from the step's output indices.
        // For now, output is simply M×N (flattened free legs).
        // A full implementation would unfold using the original free-leg dims.
        let mut output_dims = Vec::new();
        for (pos, &dim) in left_dims.iter().enumerate() {
            if !step.left_contracted_legs.contains(&pos) {
                output_dims.push(dim);
            }
        }
        for (pos, &dim) in right_dims.iter().enumerate() {
            if !step.right_contracted_legs.contains(&pos) {
                output_dims.push(dim);
            }
        }

        Ok(DenseTensor::from_vec(
            TensorShape::row_major(&output_dims),
            c_data,
        ))
    }

    /// Convenience method: build the index map from shapes, run the optimizer,
    /// and execute in one call. Intended for one-off contractions where the
    /// overhead of pre-optimization is not amortized over many calls.
    ///
    /// For recurring contractions (e.g., environment updates inside a DMRG
    /// sweep), callers should cache the `ContractionGraph` via `ExecutionPlan`
    /// and call `execute` directly.
    pub fn contract_once(
        &self,
        spec: &ContractionSpec,
        inputs: &HashMap<TensorId, &DenseTensor<'_, T>>,
        optimizer: &dyn PathOptimizer,
        cost: &CostMetric,
        max_memory_bytes: Option<usize>,
    ) -> ContractResult<DenseTensor<'static, T>> {
        // Build index map from actual tensor shapes.
        let mut index_map = IndexMap::new();
        for (tid, legs) in &spec.tensors {
            if let Some(tensor) = inputs.get(tid) {
                let shape = tensor.shape();
                let dims = shape.dims();
                let specs: Vec<crate::index::IndexSpec> = (0..legs.len())
                    .map(|i| crate::index::IndexSpec {
                        dim: dims.get(i).copied().unwrap_or(1),
                        is_contiguous: shape.is_contiguous(),
                    })
                    .collect();
                index_map.insert(*tid, specs);
            }
        }

        let graph = optimizer.optimize(spec, &index_map, cost, max_memory_bytes)?;
        self.execute(&graph, inputs)
    }
}

/// Gather tensor elements into a contiguous buffer suitable for GEMM.
///
/// `is_left == true`: free legs become rows (M), contracted become cols (K).
/// `is_left == false`: contracted become rows (K), free become cols (N).
///
/// Handles the general case by permuting axes so that free legs vary slowly
/// (leftmost) and contracted legs vary fast (rightmost) for left operands,
/// or contracted vary slowly and free vary fast for right operands.
///
/// For contiguous tensors with contracted legs already in the right position,
/// this is a simple memcpy (fast path).
fn gather_for_gemm<T: Scalar>(
    tensor: &DenseTensor<'_, T>,
    contracted_legs: &smallvec::SmallVec<[usize; 6]>,
    is_left: bool,
) -> Vec<T> {
    let shape = tensor.shape();
    let dims = shape.dims();
    let strides = shape.strides();
    let rank = shape.rank();
    let data = tensor.as_slice();

    if rank == 0 {
        return vec![data[0]];
    }

    let numel = shape.numel();

    // Fast path: contiguous row-major with contracted legs trailing.
    // For left operand (M×K), this means [free..., contracted...] which is
    // already the right layout. For right operand (K×N), contracted legs
    // must be leading — check that too.
    if shape.is_contiguous() {
        let trailing = crate::reshape::are_trailing_legs(contracted_legs, rank);
        if is_left && trailing {
            return data[..numel].to_vec();
        }
        let leading = crate::reshape::are_leading_legs(contracted_legs, rank);
        if !is_left && leading {
            return data[..numel].to_vec();
        }
    }

    // General path: build a permutation that puts legs in the desired order,
    // then gather elements in that order.
    //
    // Left operand:  [free_0, free_1, ..., contracted_0, contracted_1, ...]
    // Right operand: [contracted_0, contracted_1, ..., free_0, free_1, ...]
    let mut perm = Vec::with_capacity(rank);
    if is_left {
        for i in 0..rank {
            if !contracted_legs.contains(&i) {
                perm.push(i);
            }
        }
        for i in 0..rank {
            if contracted_legs.contains(&i) {
                perm.push(i);
            }
        }
    } else {
        for i in 0..rank {
            if contracted_legs.contains(&i) {
                perm.push(i);
            }
        }
        for i in 0..rank {
            if !contracted_legs.contains(&i) {
                perm.push(i);
            }
        }
    }

    // Gather elements in the permuted dimension order.
    let mut result = Vec::with_capacity(numel);
    let mut old_index = vec![0usize; rank];

    for _ in 0..numel {
        let linear: usize = old_index.iter().zip(strides).map(|(&i, &s)| i * s).sum();
        result.push(data[linear]);

        // Increment indices in permuted order (rightmost varies fastest).
        for &d in perm.iter().rev() {
            old_index[d] += 1;
            if old_index[d] < dims[d] {
                break;
            }
            old_index[d] = 0;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// ExecutionPlan
// ---------------------------------------------------------------------------

/// A cached, re-executable contraction plan for a fixed tensor network topology.
///
/// In DMRG sweeps, the same contraction patterns execute thousands of times
/// with different tensor *data* but the same *shapes*. Caching the
/// `ContractionGraph` eliminates repeated optimizer calls.
pub struct ExecutionPlan<T: Scalar> {
    /// The optimized contraction graph.
    pub graph: ContractionGraph,
    /// Pre-computed execution steps.
    pub steps: Vec<PairwiseStep>,
    /// The IndexMap used to build this plan, for invalidation checks.
    index_map_snapshot: IndexMap,
    _phantom: PhantomData<T>,
}

/// Convenience alias for the dense (non-symmetric) execution path.
pub type DenseExecutionPlan<T> = ExecutionPlan<T>;

impl<T: Scalar> ExecutionPlan<T> {
    /// Build a new plan by running the optimizer.
    pub fn build(
        spec: &ContractionSpec,
        index_map: &IndexMap,
        optimizer: &dyn PathOptimizer,
        cost: &CostMetric,
        max_memory_bytes: Option<usize>,
    ) -> ContractResult<Self> {
        let graph = optimizer.optimize(spec, index_map, cost, max_memory_bytes)?;
        let steps = graph.execution_order();
        Ok(ExecutionPlan {
            graph,
            steps,
            index_map_snapshot: index_map.clone(),
            _phantom: PhantomData,
        })
    }

    /// Returns true if the plan must be rebuilt because tensor dimensions changed.
    ///
    /// Compares only dimensions, not stride layouts — stride changes from
    /// permutations don't affect the plan's index connectivity.
    pub fn needs_rebuild(&self, current_index_map: &IndexMap) -> bool {
        !self.index_map_snapshot.dims_match(current_index_map)
    }

    /// Execute the plan over dense inputs.
    pub fn execute_dense<B: LinAlgBackend<T>>(
        &self,
        backend: &B,
        inputs: &HashMap<TensorId, &DenseTensor<'_, T>>,
    ) -> ContractResult<DenseTensor<'static, T>> {
        let executor = ContractionExecutor {
            backend: PhantomBackendRef(backend),
            _phantom: PhantomData,
        };
        executor.execute(&self.graph, inputs)
    }
}

/// Wrapper to allow `ContractionExecutor` to work with borrowed backends.
struct PhantomBackendRef<'a, B>(&'a B);

impl<'a, T: Scalar, B: LinAlgBackend<T>> LinAlgBackend<T> for PhantomBackendRef<'a, B> {
    fn svd_truncated_gesdd(
        &self,
        mat: &MatRef<T>,
        max_rank: usize,
        cutoff: T::Real,
    ) -> Result<tk_linalg::SvdResult<T>, tk_linalg::SvdConvergenceError> {
        self.0.svd_truncated_gesdd(mat, max_rank, cutoff)
    }

    fn svd_truncated_gesvd(
        &self,
        mat: &MatRef<T>,
        max_rank: usize,
        cutoff: T::Real,
    ) -> Result<tk_linalg::SvdResult<T>, tk_linalg::SvdConvergenceError> {
        self.0.svd_truncated_gesvd(mat, max_rank, cutoff)
    }

    fn gemm(&self, alpha: T, a: &MatRef<T>, b: &MatRef<T>, beta: T, c: &mut MatMut<T>) {
        self.0.gemm(alpha, a, b, beta, c)
    }

    fn eigh_lowest(
        &self,
        mat: &MatRef<T>,
        k: usize,
    ) -> tk_linalg::LinAlgResult<tk_linalg::EighResult<T>> {
        self.0.eigh_lowest(mat, k)
    }

    fn qr(&self, mat: &MatRef<T>) -> tk_linalg::LinAlgResult<tk_linalg::QrResult<T>> {
        self.0.qr(mat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{IndexId, IndexSpec};
    use crate::optimizer::GreedyOptimizer;

    /// A minimal backend that implements GEMM via naive triple loop.
    /// Used only for testing — not performant.
    struct NaiveBackend;

    impl LinAlgBackend<f64> for NaiveBackend {
        fn svd_truncated_gesdd(
            &self,
            _mat: &MatRef<f64>,
            _max_rank: usize,
            _cutoff: f64,
        ) -> Result<tk_linalg::SvdResult<f64>, tk_linalg::SvdConvergenceError> {
            unimplemented!("SVD not needed for contraction tests")
        }

        fn svd_truncated_gesvd(
            &self,
            _mat: &MatRef<f64>,
            _max_rank: usize,
            _cutoff: f64,
        ) -> Result<tk_linalg::SvdResult<f64>, tk_linalg::SvdConvergenceError> {
            unimplemented!("SVD not needed for contraction tests")
        }

        fn gemm(
            &self,
            alpha: f64,
            a: &MatRef<f64>,
            b: &MatRef<f64>,
            beta: f64,
            c: &mut MatMut<f64>,
        ) {
            // Naive GEMM: C = alpha * A * B + beta * C
            for i in 0..a.rows {
                for j in 0..b.cols {
                    let mut sum = 0.0;
                    for p in 0..a.cols {
                        let a_val = a.data[(i as isize * a.row_stride + p as isize * a.col_stride) as usize];
                        let b_val = b.data[(p as isize * b.row_stride + j as isize * b.col_stride) as usize];
                        sum += a_val * b_val;
                    }
                    let c_idx = (i as isize * c.row_stride + j as isize * c.col_stride) as usize;
                    c.data[c_idx] = alpha * sum + beta * c.data[c_idx];
                }
            }
        }

        fn eigh_lowest(
            &self,
            _mat: &MatRef<f64>,
            _k: usize,
        ) -> tk_linalg::LinAlgResult<tk_linalg::EighResult<f64>> {
            unimplemented!()
        }

        fn qr(
            &self,
            _mat: &MatRef<f64>,
        ) -> tk_linalg::LinAlgResult<tk_linalg::QrResult<f64>> {
            unimplemented!()
        }
    }

    #[test]
    fn executor_two_tensor_matmul() {
        let i = IndexId::from_raw(500);
        let j = IndexId::from_raw(501);
        let k = IndexId::from_raw(502);

        let spec = ContractionSpec::new(
            vec![
                (TensorId::new(0), vec![i, j]),
                (TensorId::new(1), vec![j, k]),
            ],
            vec![i, k],
        )
        .unwrap();

        let mut index_map = IndexMap::new();
        index_map.insert(
            TensorId::new(0),
            vec![
                IndexSpec { dim: 2, is_contiguous: true },
                IndexSpec { dim: 3, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(1),
            vec![
                IndexSpec { dim: 3, is_contiguous: true },
                IndexSpec { dim: 2, is_contiguous: true },
            ],
        );

        let graph = GreedyOptimizer
            .optimize(&spec, &index_map, &CostMetric::default(), None)
            .unwrap();

        // A = [[1,0,0],[0,1,0]] (2x3 identity-like)
        let a_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let a = DenseTensor::from_vec(TensorShape::row_major(&[2, 3]), a_data);

        // B = [[1,2],[3,4],[5,6]] (3x2)
        let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = DenseTensor::from_vec(TensorShape::row_major(&[3, 2]), b_data);

        let mut inputs = HashMap::new();
        inputs.insert(TensorId::new(0), &a);
        inputs.insert(TensorId::new(1), &b);

        let executor = ContractionExecutor::new(NaiveBackend);
        let result = executor.execute(&graph, &inputs).unwrap();

        // Expected: A*B = [[1,2],[3,4]] (first two rows of B)
        assert_eq!(result.shape().dims(), &[2, 2]);
        let data = result.as_slice();
        assert!((data[0] - 1.0).abs() < 1e-12);
        assert!((data[1] - 2.0).abs() < 1e-12);
        assert!((data[2] - 3.0).abs() < 1e-12);
        assert!((data[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn execution_plan_no_rebuild_same_shapes() {
        let i = IndexId::from_raw(600);
        let j = IndexId::from_raw(601);

        let spec = ContractionSpec::new(
            vec![
                (TensorId::new(0), vec![i, j]),
                (TensorId::new(1), vec![j]),
            ],
            vec![i],
        )
        .unwrap();

        let mut index_map = IndexMap::new();
        index_map.insert(
            TensorId::new(0),
            vec![
                IndexSpec { dim: 4, is_contiguous: true },
                IndexSpec { dim: 3, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(1),
            vec![IndexSpec { dim: 3, is_contiguous: true }],
        );

        let plan = ExecutionPlan::<f64>::build(
            &spec,
            &index_map,
            &GreedyOptimizer,
            &CostMetric::default(),
            None,
        )
        .unwrap();

        assert!(!plan.needs_rebuild(&index_map));
    }

    #[test]
    fn execution_plan_rebuild_after_dim_change() {
        let i = IndexId::from_raw(700);
        let j = IndexId::from_raw(701);

        let spec = ContractionSpec::new(
            vec![
                (TensorId::new(0), vec![i, j]),
                (TensorId::new(1), vec![j]),
            ],
            vec![i],
        )
        .unwrap();

        let mut index_map = IndexMap::new();
        index_map.insert(
            TensorId::new(0),
            vec![
                IndexSpec { dim: 4, is_contiguous: true },
                IndexSpec { dim: 3, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(1),
            vec![IndexSpec { dim: 3, is_contiguous: true }],
        );

        let plan = ExecutionPlan::<f64>::build(
            &spec,
            &index_map,
            &GreedyOptimizer,
            &CostMetric::default(),
            None,
        )
        .unwrap();

        // Change dimension.
        let mut new_map = IndexMap::new();
        new_map.insert(
            TensorId::new(0),
            vec![
                IndexSpec { dim: 8, is_contiguous: true }, // changed from 4 to 8
                IndexSpec { dim: 3, is_contiguous: true },
            ],
        );
        new_map.insert(
            TensorId::new(1),
            vec![IndexSpec { dim: 3, is_contiguous: true }],
        );

        assert!(plan.needs_rebuild(&new_map));
    }

    #[test]
    fn executor_three_tensor_chain() {
        // A(i,j) * B(j,k) * C(k,l) → D(i,l)
        // Tests that intermediate dimension tracking works correctly
        // when the greedy optimizer produces a multi-step plan.
        let i = IndexId::from_raw(800);
        let j = IndexId::from_raw(801);
        let k = IndexId::from_raw(802);
        let l = IndexId::from_raw(803);

        let spec = ContractionSpec::new(
            vec![
                (TensorId::new(0), vec![i, j]),
                (TensorId::new(1), vec![j, k]),
                (TensorId::new(2), vec![k, l]),
            ],
            vec![i, l],
        )
        .unwrap();

        let mut index_map = IndexMap::new();
        index_map.insert(
            TensorId::new(0),
            vec![
                IndexSpec { dim: 2, is_contiguous: true },
                IndexSpec { dim: 3, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(1),
            vec![
                IndexSpec { dim: 3, is_contiguous: true },
                IndexSpec { dim: 4, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(2),
            vec![
                IndexSpec { dim: 4, is_contiguous: true },
                IndexSpec { dim: 2, is_contiguous: true },
            ],
        );

        let graph = GreedyOptimizer
            .optimize(&spec, &index_map, &CostMetric::default(), None)
            .unwrap();

        assert_eq!(graph.n_pairwise_steps(), 2);

        // A = 2x3 identity-like: [[1,0,0],[0,1,0]]
        let a = DenseTensor::from_vec(
            TensorShape::row_major(&[2, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        // B = 3x4 all ones
        let b = DenseTensor::from_vec(
            TensorShape::row_major(&[3, 4]),
            vec![1.0; 12],
        );
        // C = 4x2 identity-like: [[1,0],[0,1],[0,0],[0,0]]
        let c = DenseTensor::from_vec(
            TensorShape::row_major(&[4, 2]),
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        );

        let mut inputs = HashMap::new();
        inputs.insert(TensorId::new(0), &a);
        inputs.insert(TensorId::new(1), &b);
        inputs.insert(TensorId::new(2), &c);

        let executor = ContractionExecutor::new(NaiveBackend);
        let result = executor.execute(&graph, &inputs).unwrap();

        // D = A * B * C, should be 2x2
        assert_eq!(result.shape().dims(), &[2, 2]);
        // A*B = [[1,1,1,1],[1,1,1,1]] (2x4, first 2 rows of B)
        // Wait: A = [[1,0,0],[0,1,0]], B = 3x4 all ones
        // A*B = [[1,1,1,1],[1,1,1,1]] (takes rows 0 and 1 of B)
        // (A*B)*C = [[1,1,1,1],[1,1,1,1]] * [[1,0],[0,1],[0,0],[0,0]]
        //         = [[1,1],[1,1]]
        let data = result.as_slice();
        assert!((data[0] - 1.0).abs() < 1e-12);
        assert!((data[1] - 1.0).abs() < 1e-12);
        assert!((data[2] - 1.0).abs() < 1e-12);
        assert!((data[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn contract_once_convenience() {
        let i = IndexId::from_raw(900);
        let j = IndexId::from_raw(901);
        let k = IndexId::from_raw(902);

        let spec = ContractionSpec::new(
            vec![
                (TensorId::new(0), vec![i, j]),
                (TensorId::new(1), vec![j, k]),
            ],
            vec![i, k],
        )
        .unwrap();

        // A = [[1,2],[3,4],[5,6]] (3x2)
        let a = DenseTensor::from_vec(
            TensorShape::row_major(&[3, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        // B = [[1,0],[0,1]] (2x2 identity)
        let b = DenseTensor::from_vec(
            TensorShape::row_major(&[2, 2]),
            vec![1.0, 0.0, 0.0, 1.0],
        );

        let mut inputs = HashMap::new();
        inputs.insert(TensorId::new(0), &a);
        inputs.insert(TensorId::new(1), &b);

        let executor = ContractionExecutor::new(NaiveBackend);
        let result = executor
            .contract_once(&spec, &inputs, &GreedyOptimizer, &CostMetric::default(), None)
            .unwrap();

        // A * I = A
        assert_eq!(result.shape().dims(), &[3, 2]);
        let data = result.as_slice();
        assert!((data[0] - 1.0).abs() < 1e-12);
        assert!((data[1] - 2.0).abs() < 1e-12);
        assert!((data[4] - 5.0).abs() < 1e-12);
        assert!((data[5] - 6.0).abs() < 1e-12);
    }
}
