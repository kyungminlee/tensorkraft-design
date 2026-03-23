# tk-contract Draft Implementation Notes

**Status:** Draft implementation — compiles, 31 tests pass, medium-severity gaps resolved.
**Date:** March 2026 (updated)

---

## What Was Implemented

- **index.rs** — `IndexId`, `TensorId`, `IndexSpec`, `ContractionSpec`, `IndexMap` with full validation
- **graph.rs** — `ContractionNode` (binary DAG), `ContractionGraph`, `PairwiseStep` with post-order execution walk
- **cost.rs** — `CostMetric` (composite FLOP + bandwidth), `CostEstimator`
- **optimizer/** — `PathOptimizer` trait, `GreedyOptimizer` with `max_memory_bytes` constraint
- **reshape.rs** — `tensor_to_mat_ref`, `block_transpose`, `mat_to_tensor` helpers
- **executor.rs** — `ContractionExecutor` with naive GEMM dispatch, `ExecutionPlan` with invalidation
- **structural.rs** — `StructuralContractionHook` trait, `AbelianHook` no-op implementation
- **sparse.rs** — `SparseContractionExecutor` skeleton delegating to `block_gemm`
- **error.rs** — Full `ContractionError` enum with `thiserror`

---

## Design Issues Discovered During Implementation

### 1. `DenseTensor` lifetime makes executor generics painful — **RESOLVED**

The spec assumes `ContractionExecutor::execute` can freely store and pass around `DenseTensor` references between steps. In practice, `DenseTensor<'a, T>` carries a lifetime `'a` tied to its storage. When intermediates (owned) and inputs (borrowed) coexist in the same `HashMap`, the lifetime annotations become difficult to reconcile.

**Resolution:** Implemented techspec §7.1 Option B — a `TensorRef<'a, T>` enum with `Borrowed` and `Owned` variants. The executor now maintains a single `HashMap<TensorId, TensorRef>` that safely holds both borrowed inputs and owned intermediates. The `unsafe transmute` has been removed entirely.

### 2. `BlockSparseTensor` does not implement `Clone`

The spec's `SparseContractionExecutor::execute` signature returns `ContractResult<BlockSparseTensor<T, Q>>`. For the degenerate case of a single-tensor "contraction" (no pairwise steps), the executor needs to return a copy of the input. But `BlockSparseTensor` doesn't implement `Clone` (its blocks are `DenseTensor<'static, T>` which don't derive `Clone` either due to the `TensorStorage` enum).

**Workaround used:** Return an error for the single-tensor case. This is acceptable for DMRG where all contractions involve ≥2 tensors, but it's a gap in the API.

**Recommendation:** Either:
- Implement `Clone` for `BlockSparseTensor` and `DenseTensor<'static, T>`
- Change the sparse executor to take ownership of inputs (accepting `HashMap<TensorId, BlockSparseTensor<T, Q>>` instead of `&BlockSparseTensor`)
- Document that single-tensor pass-through is the caller's responsibility

### 3. `ContractionSpec` caches should be computed eagerly — spec was right, but missed `shared_indices`

The spec's B9 recommendation to cache `contracted_pairs()` and `free_indices()` was correct. During implementation, I found that the greedy optimizer also needs a `shared_indices(a, b)` query — "which indices are shared between two specific tensors?" This isn't in the spec at all but is needed O(n²) times per optimization step.

**Action taken:** Added `shared_indices()` method to `ContractionSpec` and cached `contracted_pairs`/`free_indices` at construction time as recommended. The `shared_indices` method currently does a linear scan over the cached pairs, which is fine for n ≤ 5 but would need a `HashMap<(TensorId, TensorId), Vec<IndexId>>` for larger networks.

### 4. The spec's `contracted_indices: Vec<(IndexId, IndexId)>` vs. `Vec<IndexId>` — we were right to change it

The review changed `ContractionNode::Contraction::contracted_indices` from `Vec<(IndexId, IndexId)>` to `Vec<IndexId>`. During implementation, this proved correct: since `ContractionSpec` uses shared `IndexId`s, the contracted indices in the DAG are naturally single IDs. The executor's `PairwiseStep` carries the *leg positions* (`left_contracted_legs`, `right_contracted_legs`) separately, which is what the GEMM reshape actually needs.

### 5. The `IndexMap` dimension lookup pattern is awkward for intermediates — **RESOLVED**

The optimizer uses `IndexMap` to look up dimensions by `(TensorId, leg_position)`. But intermediate results from pairwise contractions get assigned new `TensorId`s, and their dimensions must be computed on the fly from the input dimensions and contracted legs. The current `IndexMap` doesn't support this well — you'd need to insert intermediate specs during optimization, which mutates a structure the spec treats as read-only.

**Resolution:** Per techspec §3.4 guidance, `NodeEntry` now carries a `dims: Vec<usize>` field alongside each node. For input nodes, dims are populated from the `IndexMap` at initialization. For intermediate nodes, dims are computed from the contracting pair's free-leg dimensions during `find_best_pair`. The `IndexMap` remains read-only throughout optimization.

### 6. `tensor_to_mat_ref` requires arena allocation for non-contiguous inputs — **RESOLVED**

The spec's `tensor_to_mat_ref` (§7.2) takes a `&SweepArena` parameter for allocating transposed copies of non-contiguous inputs. In the draft implementation, we don't have arena integration yet, so the fallback is to use the raw slice and hope strides work out. This means the draft only handles contiguous inputs correctly.

**Resolution:** `tensor_to_mat_ref` now returns a `ReshapeResult` struct that either borrows from the original tensor (zero-copy fast path when contiguous + trailing contracted legs) or owns a heap-allocated transposed copy (slow path via `block_transpose`). The `as_mat_ref()` method on `ReshapeResult` provides the `MatRef` from whichever source. `gather_for_gemm` in the executor was also fixed to properly reorder elements when contracted legs are non-trailing, using the same permutation logic.

### 7. Feature flags for backends not yet implemented in tk-linalg

The spec lists `backend-mkl`, `backend-openblas`, and `backend-cuda` feature flags. None of these exist in the actual `tk-linalg/Cargo.toml` yet. The tk-contract `Cargo.toml` had to comment them out to compile.

**Recommendation:** The spec should note which feature flags are "planned" vs. "implemented" to avoid confusion during incremental development.

### 8. `ExecutionPlan` re-execution via `PhantomBackendRef` is a hack

The spec says `ExecutionPlan::execute_dense` takes a backend reference. But `ContractionExecutor` is generic over `B: LinAlgBackend<T>` (owned backend). To make `execute_dense` work with a borrowed backend, I had to create a `PhantomBackendRef` wrapper that delegates all trait methods. This is boilerplate-heavy and fragile.

**Recommendation:** Either:
- Make `ContractionExecutor` generic over `&B` / `Arc<B>` from the start
- Change `execute_dense` to construct a fresh `ContractionExecutor` internally (which is what the draft does)
- Add a `where B: LinAlgBackend<T>` bound on the `ExecutionPlan` itself

### 9. `NoQ` sentinel type — unnecessary complexity for the draft

The spec introduces `NoQ` as a sentinel quantum-number type for the dense path, with `ExecutionPlan<T, Q = NoQ>`. In practice, the dense `ExecutionPlan` doesn't interact with quantum numbers at all. The `Q` parameter adds complexity without benefit in the current implementation.

**Action taken:** Simplified `ExecutionPlan<T>` to not be generic over `Q`. Added `DenseExecutionPlan<T>` as a simple alias. If the sparse execution plan is needed later, it can be a separate type or the `Q` parameter can be reintroduced.

### 10. `QuantumNumber` trait's `fuse` method returns owned values, not references

The `AbelianHook::compute_output_sectors` implementation calls `qa.fuse(qb)` in an iterator. Since `fuse` takes `&self, &Self` and returns `Self`, this works cleanly with value types like `U1(i32)`. But if a future quantum number type were heap-allocated, this would cause unnecessary cloning. The trait design is fine for now but should be noted.

---

## Gaps Between Spec and Reality

| Spec Section | Issue | Severity | Status |
|:-------------|:------|:---------|:-------|
| §7.1 (`ContractionExecutor`) | Spec shows arena parameter; actual `DenseTensor` lifetimes make arena integration non-trivial | Medium | **Resolved** — Replaced unsafe transmute with `TensorRef` enum (Option B from techspec). Safe split-storage approach. |
| §7.2 (`tensor_to_mat_ref`) | Spec assumes arena is always available for transpose; no fallback documented | Medium | **Resolved** — `tensor_to_mat_ref` now returns `ReshapeResult` that owns transposed data via heap fallback when zero-copy path unavailable. `gather_for_gemm` properly reorders for non-trailing contracted legs. |
| §6 (`PathOptimizer`) | `IndexMap` is read-only but optimizer needs to track intermediate dimensions | Medium | **Resolved** — `NodeEntry` now carries per-node `dims: Vec<usize>`, populated from `IndexMap` for inputs and computed from contracting pairs for intermediates. `find_best_pair` uses node-local dims. |
| §7.1 | Spec's `execute` return type `TempTensor<'arena>` requires arena-lifetime coupling not yet feasible | Medium | **Resolved** — `execute` returns `DenseTensor<'static, T>` (heap-owned). Added `contract_once` convenience method per techspec. Arena integration deferred to when `SweepArena` is threaded through the executor. |
| §8 (`SparseContractionExecutor`) | Spec assumes `BlockSparseTensor` is cloneable; it's not | Low | Open |
| §10 (`ExecutionPlan`) | `NoQ` sentinel adds complexity without benefit in Phase 2 | Low | Open (already simplified in draft) |
| §14 (`Cargo.toml`) | Feature flags reference backends that don't exist yet | Low | Open (documented as planned-only) |

---

## What Works Well

1. **The greedy optimizer is straightforward** — O(n³) for n ≤ 5 is trivially fast. The spec's decision to make this the default was correct.

2. **`ContractionSpec` validation is solid** — catching index-appears-too-many-times, duplicate-on-tensor, and output-not-free at construction time prevents entire categories of bugs.

3. **The `PairwiseStep` abstraction** — separating DAG construction from execution via the `PairwiseStep` intermediate representation makes the executor's GEMM dispatch clean and testable.

4. **The `StructuralContractionHook` design** — having the Abelian path be a zero-cost no-op via `AbelianHook` is elegant. The trait is simple to implement and the `SmallVec` return type avoids heap allocation.

5. **`CostMetric` composite scoring** — the α·FLOPs + β·Bytes formulation is clean and the default β=50 correctly biases toward avoiding transposes.

6. **Caching `contracted_pairs`/`free_indices` in `ContractionSpec::new()`** — avoids repeated computation as recommended in the review.
