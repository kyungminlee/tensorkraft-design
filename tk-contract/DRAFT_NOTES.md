# tk-contract Draft Implementation Notes

**Status:** Draft implementation — compiles, 53 tests pass (49 unit + 4 proptest), all major gaps resolved.
**Date:** March 2026

---

## What is implemented

- **index.rs** — `IndexId`, `TensorId`, `IndexSpec` (with `is_contracted` field per techspec §3.2), `ContractionSpec`, `IndexMap` with full validation including dimension mismatch detection via `validate_dimensions()`
- **graph.rs** — `ContractionNode` (binary DAG), `ContractionGraph`, `PairwiseStep` with post-order execution walk
- **cost.rs** — `CostMetric` (composite FLOP + bandwidth), `CostEstimator`
- **optimizer/** — `PathOptimizer` trait, `GreedyOptimizer`, `DPOptimizer` (Held-Karp DP, exact optimal for n ≤ 15), `TreeSAOptimizer` (simulated annealing for large networks)
- **reshape.rs** — `tensor_to_mat_ref` (with `ReshapeResult` zero-copy/owned enum), `block_transpose`, `mat_to_tensor` helpers
- **executor.rs** — `ContractionExecutor` with GEMM dispatch, `ExecutionPlan` with invalidation, `execute_sparse` method on `ExecutionPlan`, `contract_once` convenience method
- **structural.rs** — `StructuralContractionHook` trait, `AbelianHook` no-op implementation
- **sparse.rs** — `SparseContractionExecutor` with permute + fuse_legs preprocessing for arbitrary contracted leg positions, single-tensor clone pass-through, structural hook integration
- **error.rs** — Full `ContractionError` enum with `thiserror`
- **benches/** — Criterion benchmarks for optimizer throughput, executor matmul, plan rebuild latency
- **tests/** — Proptest integration tests: GEMM reference comparison, optimizer cost non-negativity, DP ≤ greedy optimality, dimension mismatch detection

### DenseTensor lifetime handling (complete)

Implemented techspec §7.1 Option B — a `TensorRef<'a, T>` enum with `Borrowed` and `Owned` variants. The executor maintains a single `HashMap<TensorId, TensorRef>` that safely holds both borrowed inputs and owned intermediates. The `unsafe transmute` has been removed entirely.

### IndexMap dimension tracking (complete)

Per techspec §3.4, `NodeEntry` carries a `dims: Vec<usize>` field alongside each node. For input nodes, dims are populated from the `IndexMap` at initialization. For intermediate nodes, dims are computed from the contracting pair's free-leg dimensions during `find_best_pair`. The `IndexMap` remains read-only throughout optimization.

### Non-contiguous tensor reshape (complete)

`tensor_to_mat_ref` returns a `ReshapeResult` struct that either borrows from the original tensor (zero-copy fast path when contiguous + trailing contracted legs) or owns a heap-allocated transposed copy (slow path via `block_transpose`). `gather_for_gemm` in the executor properly reorders elements when contracted legs are non-trailing.

### Execution plan (complete)

`execute` returns `DenseTensor<'static, T>` (heap-owned). Added `contract_once` convenience method per techspec. `ExecutionPlan::execute_sparse` method added for block-sparse execution. Arena integration deferred to when `SweepArena` is threaded through the executor.

### DPOptimizer (complete)

Implements the Held-Karp DP variant for tensor contraction ordering. Enumerates all 2^n subsets via Gosper's hack, finding the globally optimal pairwise contraction order. `max_width` constrains intermediate tensor size. Time O(3^n), practical for n ≤ 15. Property tests verify DP cost ≤ greedy cost.

### TreeSAOptimizer (complete)

Simulated annealing over random contraction trees. Uses a built-in xoshiro256++ RNG (no external `rand` dependency needed at runtime) with optional seed for reproducibility. Generates neighbor solutions by random tree reconstruction and accepts worse solutions with Boltzmann probability. Suitable for n > 15 where DP is intractable.

### Dimension mismatch validation (complete)

`ContractionSpec::validate_dimensions(&self, &IndexMap)` checks that all contracted index pairs have matching dimensions across the two tensors they connect. Returns `DimensionMismatch` error with full context. Property tests verify all mismatches are caught.

### Sparse executor (complete)

`SparseContractionExecutor` performs:
1. **Permute** — reorders legs so contracted legs are grouped (trailing for left, leading for right)
2. **Fuse** — `fuse_legs` on contiguous ranges to produce rank-2 matrices
3. **Dispatch** — `block_gemm` for sector-pair matching and parallel GEMM
4. **Single-tensor clone** — uses `BlockSparseTensor::clone()` for degenerate cases
5. **Debug verification** — rank check in debug builds

Handles arbitrary contracted leg positions (not just leading/trailing).

### IndexSpec::is_contracted (complete)

Per techspec §3.2, `IndexSpec` now carries `is_contracted: bool`. The field is informational metadata — the optimizer determines contracted vs. free from `ContractionSpec`, but the field is useful for diagnostic introspection by downstream callers.

---

## Design issues and decisions

1. **`ContractionSpec` added `shared_indices(a, b)`** — The greedy optimizer needs this query O(n²) times per step. Currently does a linear scan over cached pairs (fine for n ≤ 5; would need a `HashMap` for larger networks).

2. **`ContractionNode::contracted_indices` is `Vec<IndexId>`** — Proved correct since `ContractionSpec` uses shared `IndexId`s. The executor's `PairwiseStep` carries leg positions separately.

3. **`ExecutionPlan` simplified** — `NoQ` sentinel type removed; `ExecutionPlan<T>` is not generic over `Q`. `DenseExecutionPlan<T>` is a simple alias.

4. **`PhantomBackendRef` wrapper for borrowed backend** — Required because `ContractionExecutor` is generic over `B: LinAlgBackend<T>` (owned). Boilerplate-heavy. Consider `&B` or `Arc<B>` for the backend.

5. **TreeSAOptimizer uses built-in RNG** — Avoids `rand` crate dependency by implementing xoshiro256++ directly. The `treesa` feature flag in the techspec (for `rand`/`rand_xoshiro`) is not needed in the current implementation.

---

## Remaining limitations

| Issue | Severity |
|:------|:---------|
| `SweepArena` integration deferred — executor heap-allocates intermediates | Medium |
| Feature flags reference backends that don't exist yet in tk-linalg (`backend-mkl`, `backend-openblas`, `backend-cuda`) | Low |
| `StructuralContractionHook` stored but not yet invoked per-sector during sparse GEMM (Abelian path is no-op; SU(2) hook will be wired in Phase 5+) | Low |
| Sparse executor clones tensors for permute+fuse; could be optimized with in-place permutation | Low |

---

## What works well

1. **All three optimizers** — Greedy (O(n³), default for DMRG n ≤ 5), DP (exact optimal for n ≤ 15), TreeSA (annealing for large n). Property tests verify DP ≤ greedy.
2. **`ContractionSpec` validation** — catches index-appears-too-many-times, duplicate-on-tensor, output-not-free at construction time, and dimension mismatches via `validate_dimensions()`.
3. **`PairwiseStep` abstraction** — separating DAG construction from execution makes GEMM dispatch clean and testable.
4. **`StructuralContractionHook` design** — Abelian path is a zero-cost no-op via `AbelianHook`.
5. **`CostMetric` composite scoring** — α·FLOPs + β·Bytes with default β=50 correctly biases toward avoiding transposes.
6. **Cached `contracted_pairs`/`free_indices`** — computed eagerly in `ContractionSpec::new()`.
7. **53 tests passing** — comprehensive coverage of index validation, optimizer correctness, executor pipeline, reshape helpers, structural hooks, and property-based GEMM reference comparison.
8. **Criterion benchmarks** — optimizer throughput, matmul execution, plan rebuild latency.
