# tk-contract Draft Implementation Notes

**Status:** Draft implementation — compiles, 31 tests pass, medium-severity gaps resolved.
**Date:** March 2026

---

## What is implemented

- **index.rs** — `IndexId`, `TensorId`, `IndexSpec`, `ContractionSpec`, `IndexMap` with full validation
- **graph.rs** — `ContractionNode` (binary DAG), `ContractionGraph`, `PairwiseStep` with post-order execution walk
- **cost.rs** — `CostMetric` (composite FLOP + bandwidth), `CostEstimator`
- **optimizer/** — `PathOptimizer` trait, `GreedyOptimizer` with `max_memory_bytes` constraint
- **reshape.rs** — `tensor_to_mat_ref`, `block_transpose`, `mat_to_tensor` helpers
- **executor.rs** — `ContractionExecutor` with naive GEMM dispatch, `ExecutionPlan` with invalidation
- **structural.rs** — `StructuralContractionHook` trait, `AbelianHook` no-op implementation
- **sparse.rs** — `SparseContractionExecutor` skeleton delegating to `block_gemm`
- **error.rs** — Full `ContractionError` enum with `thiserror`

### DenseTensor lifetime handling (complete)

Implemented techspec §7.1 Option B — a `TensorRef<'a, T>` enum with `Borrowed` and `Owned` variants. The executor maintains a single `HashMap<TensorId, TensorRef>` that safely holds both borrowed inputs and owned intermediates. The `unsafe transmute` has been removed entirely.

### IndexMap dimension tracking (complete)

Per techspec §3.4, `NodeEntry` carries a `dims: Vec<usize>` field alongside each node. For input nodes, dims are populated from the `IndexMap` at initialization. For intermediate nodes, dims are computed from the contracting pair's free-leg dimensions during `find_best_pair`. The `IndexMap` remains read-only throughout optimization.

### Non-contiguous tensor reshape (complete)

`tensor_to_mat_ref` returns a `ReshapeResult` struct that either borrows from the original tensor (zero-copy fast path when contiguous + trailing contracted legs) or owns a heap-allocated transposed copy (slow path via `block_transpose`). `gather_for_gemm` in the executor properly reorders elements when contracted legs are non-trailing.

### Execution plan (complete)

`execute` returns `DenseTensor<'static, T>` (heap-owned). Added `contract_once` convenience method per techspec. Arena integration deferred to when `SweepArena` is threaded through the executor.

---

## Design issues and decisions

1. **`BlockSparseTensor` does not implement `Clone`** — For the single-tensor "contraction" case, the executor returns an error. Acceptable for DMRG (all contractions involve ≥2 tensors). Recommend implementing `Clone` for `BlockSparseTensor` or changing sparse executor to take ownership.

2. **`ContractionSpec` added `shared_indices(a, b)`** — The greedy optimizer needs this query O(n²) times per step. Currently does a linear scan over cached pairs (fine for n ≤ 5; would need a `HashMap` for larger networks).

3. **`ContractionNode::contracted_indices` is `Vec<IndexId>`** — Proved correct since `ContractionSpec` uses shared `IndexId`s. The executor's `PairwiseStep` carries leg positions separately.

4. **`ExecutionPlan` simplified** — `NoQ` sentinel type removed; `ExecutionPlan<T>` is not generic over `Q`. `DenseExecutionPlan<T>` is a simple alias.

5. **`PhantomBackendRef` wrapper for borrowed backend** — Required because `ContractionExecutor` is generic over `B: LinAlgBackend<T>` (owned). Boilerplate-heavy. Consider `&B` or `Arc<B>` for the backend.

---

## Remaining limitations

| Issue | Severity |
|:------|:---------|
| `BlockSparseTensor` not cloneable; sparse executor can't return single-tensor pass-through | Low |
| `NoQ` sentinel adds complexity without benefit in Phase 2 (already simplified) | Low |
| Feature flags reference backends that don't exist yet in tk-linalg | Low |

---

## What works well

1. **Greedy optimizer** — O(n³) for n ≤ 5 is trivially fast.
2. **`ContractionSpec` validation** — catches index-appears-too-many-times, duplicate-on-tensor, and output-not-free at construction time.
3. **`PairwiseStep` abstraction** — separating DAG construction from execution makes GEMM dispatch clean and testable.
4. **`StructuralContractionHook` design** — Abelian path is a zero-cost no-op via `AbelianHook`.
5. **`CostMetric` composite scoring** — α·FLOPs + β·Bytes with default β=50 correctly biases toward avoiding transposes.
6. **Cached `contracted_pairs`/`free_indices`** — computed eagerly in `ContractionSpec::new()`.
