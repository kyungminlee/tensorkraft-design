# tk-contract Draft Implementation Notes

**Status:** Draft implementation — compiles, 49 tests pass, major gaps filled.
**Date:** March 2026

---

## What is implemented

- **index.rs** — `IndexId`, `TensorId`, `IndexSpec`, `ContractionSpec`, `IndexMap` with full validation including dimension mismatch detection via `validate_dimensions()`
- **graph.rs** — `ContractionNode` (binary DAG), `ContractionGraph`, `PairwiseStep` with post-order execution walk
- **cost.rs** — `CostMetric` (composite FLOP + bandwidth), `CostEstimator`
- **optimizer/** — `PathOptimizer` trait, `GreedyOptimizer`, `DPOptimizer` (Held-Karp DP, exact optimal for n ≤ 15), `TreeSAOptimizer` (simulated annealing for large networks)
- **reshape.rs** — `tensor_to_mat_ref`, `block_transpose`, `mat_to_tensor` helpers
- **executor.rs** — `ContractionExecutor` with naive GEMM dispatch, `ExecutionPlan` with invalidation, `execute_sparse` method on `ExecutionPlan`
- **structural.rs** — `StructuralContractionHook` trait, `AbelianHook` no-op implementation
- **sparse.rs** — `SparseContractionExecutor` with pairwise leg-fuse preprocessing and `block_gemm` dispatch
- **error.rs** — Full `ContractionError` enum with `thiserror`

### DenseTensor lifetime handling (complete)

Implemented techspec §7.1 Option B — a `TensorRef<'a, T>` enum with `Borrowed` and `Owned` variants. The executor maintains a single `HashMap<TensorId, TensorRef>` that safely holds both borrowed inputs and owned intermediates. The `unsafe transmute` has been removed entirely.

### IndexMap dimension tracking (complete)

Per techspec §3.4, `NodeEntry` carries a `dims: Vec<usize>` field alongside each node. For input nodes, dims are populated from the `IndexMap` at initialization. For intermediate nodes, dims are computed from the contracting pair's free-leg dimensions during `find_best_pair`. The `IndexMap` remains read-only throughout optimization.

### Non-contiguous tensor reshape (complete)

`tensor_to_mat_ref` returns a `ReshapeResult` struct that either borrows from the original tensor (zero-copy fast path when contiguous + trailing contracted legs) or owns a heap-allocated transposed copy (slow path via `block_transpose`). `gather_for_gemm` in the executor properly reorders elements when contracted legs are non-trailing.

### Execution plan (complete)

`execute` returns `DenseTensor<'static, T>` (heap-owned). Added `contract_once` convenience method per techspec. `ExecutionPlan::execute_sparse` method added for block-sparse execution. Arena integration deferred to when `SweepArena` is threaded through the executor.

### DPOptimizer (complete)

Implements the Held-Karp DP variant for tensor contraction ordering. Enumerates all 2^n subsets via Gosper's hack, finding the globally optimal pairwise contraction order. `max_width` constrains intermediate tensor size. Time O(3^n), practical for n ≤ 15. Verified against greedy results in tests.

### TreeSAOptimizer (complete)

Simulated annealing over random contraction trees. Uses a built-in xoshiro256++ RNG (no external `rand` dependency needed at runtime) with optional seed for reproducibility. Generates neighbor solutions by random tree reconstruction and accepts worse solutions with Boltzmann probability. Suitable for n > 15 where DP is intractable.

### Dimension mismatch validation (complete)

`ContractionSpec::validate_dimensions(&self, &IndexMap)` checks that all contracted index pairs have matching dimensions across the two tensors they connect. Returns `DimensionMismatch` error with full context (which index, which tensors, which legs, what dimensions).

### Sparse executor improvements (complete)

`SparseContractionExecutor` now performs pairwise leg-fuse preprocessing via `BlockSparseTensor::fuse_legs` to produce rank-2 matrices before dispatching to `block_gemm`. Debug-mode rank verification added. Structural hook field is stored for future non-Abelian use.

---

## Design issues and decisions

1. **`BlockSparseTensor` does not implement `Clone`** — For the single-tensor "contraction" case, the executor returns an error. Acceptable for DMRG (all contractions involve ≥2 tensors). Recommend implementing `Clone` for `BlockSparseTensor` or changing sparse executor to take ownership.

2. **`ContractionSpec` added `shared_indices(a, b)`** — The greedy optimizer needs this query O(n²) times per step. Currently does a linear scan over cached pairs (fine for n ≤ 5; would need a `HashMap` for larger networks).

3. **`ContractionNode::contracted_indices` is `Vec<IndexId>`** — Proved correct since `ContractionSpec` uses shared `IndexId`s. The executor's `PairwiseStep` carries leg positions separately.

4. **`ExecutionPlan` simplified** — `NoQ` sentinel type removed; `ExecutionPlan<T>` is not generic over `Q`. `DenseExecutionPlan<T>` is a simple alias.

5. **`PhantomBackendRef` wrapper for borrowed backend** — Required because `ContractionExecutor` is generic over `B: LinAlgBackend<T>` (owned). Boilerplate-heavy. Consider `&B` or `Arc<B>` for the backend.

6. **TreeSAOptimizer uses built-in RNG** — Avoids `rand` crate dependency by implementing xoshiro256++ directly. The `treesa` feature flag in the techspec (for `rand`/`rand_xoshiro`) is not needed in the current implementation.

7. **`IndexSpec::is_contracted` field omitted** — The techspec §3.2 includes this field, but no code path reads it (contraction status is determined from `ContractionSpec`). Omitted to reduce boilerplate. Can be added if downstream callers need it.

---

## Remaining limitations

| Issue | Severity |
|:------|:---------|
| `BlockSparseTensor` not cloneable; sparse executor can't return single-tensor pass-through | Low |
| Feature flags reference backends that don't exist yet in tk-linalg (`backend-mkl`, `backend-openblas`, `backend-cuda`) | Low |
| `SweepArena` integration deferred — executor heap-allocates intermediates | Medium |
| `IndexSpec::is_contracted` field not present (informational only; no code reads it) | Low |
| Sparse executor leg-fuse assumes contracted legs are contiguous ranges | Low |
| `StructuralContractionHook` stored but not yet called during sparse execution (Abelian path is no-op; SU(2) path not yet needed) | Low |
| `criterion` benchmarks not yet set up | Low |
| Property-based tests (`proptest`) for contraction result vs. einsum reference not yet implemented | Low |

---

## What works well

1. **All three optimizers** — Greedy (O(n³), default for DMRG n ≤ 5), DP (exact optimal for n ≤ 15), TreeSA (annealing for large n).
2. **`ContractionSpec` validation** — catches index-appears-too-many-times, duplicate-on-tensor, output-not-free at construction time, and dimension mismatches via `validate_dimensions()`.
3. **`PairwiseStep` abstraction** — separating DAG construction from execution makes GEMM dispatch clean and testable.
4. **`StructuralContractionHook` design** — Abelian path is a zero-cost no-op via `AbelianHook`.
5. **`CostMetric` composite scoring** — α·FLOPs + β·Bytes with default β=50 correctly biases toward avoiding transposes.
6. **Cached `contracted_pairs`/`free_indices`** — computed eagerly in `ContractionSpec::new()`.
7. **49 tests passing** — comprehensive coverage of index validation, optimizer correctness, executor pipeline, reshape helpers, and structural hooks.
