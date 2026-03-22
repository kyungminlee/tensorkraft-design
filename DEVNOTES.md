# Development Notes — tensorkraft

## tk-core: Lessons Learned & Spec Feedback

### Architecture doc `TensorCow` wrapper is unnecessary — merge CoW into `TensorStorage`

The architecture doc (§3.1) defines three types: `TensorStorage<T>` (wrapping `Vec<T>`),
`TensorCow<'a, T>` (borrowing `&'a TensorStorage<T>` or owning one), and `DenseTensor<T>`
(holding a `TensorCow`). During implementation, we found that the indirection through
`TensorCow` adds complexity with no benefit. The tech spec's simpler design —
`TensorStorage<'a, T>` as a single enum with `Owned(Vec<T>)` / `Borrowed(&'a [T])` —
eliminates one type and avoids the double-indirection where `Borrowed` held a reference
to a `TensorStorage` struct wrapping a `Vec<T>`.

**Recommendation for future specs:** When the architecture doc defines wrapper types,
verify during tech spec writing that each layer of indirection adds real value. The
`TensorCow` → `TensorStorage` collapse removed a type without losing any capability.

### `DenseTensor` needs an `offset` field — architecture doc omission

The architecture doc (§3.1) defines `DenseTensor` with only `shape` and `storage` fields.
The tech spec adds `offset: usize`, which proved essential for `slice_axis` to work as a
zero-copy view. Without offset, slicing would require either copying data or creating a
new borrowed view pointing into the middle of a buffer (which `&[T]` can do, but then
chained slicing becomes complex because you'd need to re-slice the slice reference).

The `offset` field propagates cleanly: `slice_axis` accumulates offsets, `as_slice()`
applies it, and `into_owned()` gathers only the logical elements when offset is nonzero.
This is a case where the architecture doc's data structure was incomplete, and the tech
spec caught it. Future architecture docs should consider the full lifecycle of view
operations (slice → chain → materialize) when defining tensor structs.

### `DenseTensor` requires a lifetime parameter — architecture doc omission

The architecture doc (§3.1) defines `DenseTensor<T>` without a lifetime parameter, relying
on the separately-defined `TensorCow<'a, T>` to carry the borrow lifetime. When we merged
CoW into `TensorStorage<'a, T>`, the lifetime surfaced to `DenseTensor<'a, T>`. This is
the correct design — the lifetime must be visible at the tensor level so the borrow checker
can enforce arena safety. But it means every function signature involving `DenseTensor` must
be lifetime-annotated, which the architecture doc's pseudocode didn't anticipate.

**Impact:** Functions like `permute`, `reshape`, `slice_axis` all return
`DenseTensor<'_, T>` (borrowing from self). The `TempTensor<'a, T>` alias is just
`DenseTensor<'a, T>` — no separate type needed. The architecture doc's `TempTensor<'a, T>`
as a distinct concept is slightly misleading; it's the same type with a shorter lifetime.

### `Scalar` trait needs more bounds than the architecture doc shows

The architecture doc (§3.4) defines `Scalar` with `Add<Output=Self> + Mul<Output=Self>`.
The implementation requires additional bounds: `Sub<Output=Self>`, `Neg<Output=Self>`,
`Debug`, and `'static`. `Sub` and `Neg` are needed by any code that computes residuals or
differences. `Debug` is needed for error messages. `'static` is needed because `Scalar`
appears in struct definitions that need `'static` for owned storage. The `Real` associated
type also needs `PartialOrd` (for truncation thresholds) and `Float` (for `epsilon()`,
`sqrt()`, etc.), which the architecture doc omits.

**Recommendation:** When specifying trait hierarchies, enumerate the full bound set in the
architecture doc. Downstream crates discover missing bounds at compile time, and adding
bounds to a sealed trait is technically non-breaking but creates churn.

### `is_contiguous()` only checks row-major — potential column-major trap

`TensorShape::is_contiguous()` compares strides against the expected row-major strides. A
tensor created via `col_major()` will report `is_contiguous() == false` even though its
data is physically contiguous (just in Fortran order). This means `reshape()` fails on
column-major tensors, which could surprise users.

This is acceptable for DMRG where we always work in row-major, but the name
`is_contiguous()` is misleading — it really means `is_row_major_contiguous()`. If we ever
need column-major reshape (e.g., for Fortran FFI), we'll need to either rename this method
or add a more general contiguity check that accepts any ordering.

### `into_owned()` is more complex than specs suggest

Both specs show `into_owned()` as a simple clone-if-borrowed operation. The real
implementation has a three-way fast/slow path:

1. **Move path:** already owned, contiguous, offset 0, tight buffer → zero-cost move
2. **Memcpy path:** contiguous with nonzero offset → single `to_vec()` on a subslice
3. **Gather path:** non-contiguous strides → element-by-element gather via multi-index iteration

The gather path (`gather_elements()`) is O(numel × rank) due to multi-index arithmetic.
For rank-6 tensors this adds meaningful overhead. Future specs should mention the gather
path explicitly, since it affects performance of any operation that materializes a
transposed view (e.g., `tensor.permute(&perm).into_owned()`).

### `gather_elements()` multi-index iteration is a non-trivial algorithm

Neither spec mentions the algorithm for materializing a non-contiguous view. The
implementation uses row-major multi-index enumeration: maintain a `vec![0; rank]` counter,
compute `sum(index[i] * strides[i])` for each element, increment the counter with carry
from the last axis. This is correct but has two performance concerns:

1. The inner loop does `rank` multiplies per element (could be reduced to incremental
   offset updates for common stride patterns).
2. The temporary `vec![0; rank]` heap-allocates for rank > ~12 (could use `SmallVec`
   to stay consistent with `TensorShape`'s stack allocation strategy).

Neither concern matters for DMRG tensors (rank ≤ 6, gather is rare), but worth noting
for future extensions.

### Rank-0 tensor edge case: `numel()` returns 1 for empty dims

`TensorShape::row_major(&[])` creates a rank-0 shape with `numel() == 1` (empty product).
This is mathematically correct (a scalar tensor holds one element), but `strides` is empty,
so `offset(&[])` returns 0 and `is_contiguous()` returns true. The code works, but no tests
exercise this case because rank-0 is officially deferred (tech spec open question #1). If
we later decide to support rank-0 tensors, the current code accidentally works — but
`SweepArena::alloc_tensor` would allocate a 1-element buffer, which is correct.

### `borrow_storage()` pattern enables zero-copy chain

The key enabler for zero-copy `permute`/`reshape`/`slice_axis` is the `borrow_storage()`
helper, which creates a `TensorStorage::Borrowed` from either variant. This means even
an `Owned` tensor produces borrowed views — the original `Vec` stays alive as long as
the original tensor is alive, and the view borrows from it. This is not mentioned in
either spec but is critical for understanding why `tensor.permute(p1).slice_axis(0, 1, 3)`
compiles: each operation borrows from the previous, and Rust's lifetime chain ensures
the original data outlives all views.

### Debug-mode permutation validation has non-trivial cost

`TensorShape::permute()` validates the permutation in debug builds by cloning into a
`SmallVec`, sorting, and comparing against `0..rank`. This is O(rank log rank) per call.
In a DMRG sweep with thousands of contractions, each involving multiple permutations, this
adds up. The validation is valuable for catching bugs, but if debug-build performance
becomes an issue, consider a bitset-based O(rank) validation instead.

### Compile-fail tests require careful error message targeting

The five `trybuild` compile-fail tests verify borrow-checker enforcement, but they're
sensitive to Rust compiler version — error messages change across rustc releases. We pin
expected `.stderr` files to the current toolchain. When upgrading rustc, these tests may
need `.stderr` updates even though the safety properties haven't changed. Consider using
`trybuild`'s `compile_fail` mode (without `.stderr` matching) for robustness, at the cost
of less precise error checking.

### Architecture doc workspace layout doesn't match actual structure

The architecture doc (§2.1) shows crates under `crates/` (e.g., `crates/tk-core/`), but
the actual workspace uses top-level directories (`tk-core/`, `tk-symmetry/`, `tk-linalg/`).
This is a minor discrepancy but could confuse new contributors. Future tech specs should
note the actual layout or the architecture doc should be updated.

---

## tk-linalg: Lessons Learned & Design Observations

### faer 0.19 API surface

faer's public API has two tiers that are not equally documented:

1. **High-level API** (`Mat::thin_svd()`, `Mat::qr()`, `Mat::selfadjoint_eigendecomposition()`) —
   ergonomic, allocates internally, returns owned results. This is what we use.
2. **Low-level API** (`compute_svd`, `compute_hermitian_evd`, `qr_in_place`) —
   requires caller-managed `PodStack` scratch buffers and has different argument
   signatures across minor versions. The parameter lists changed between 0.18 and 0.19
   in ways that aren't obvious from the docs.

The high-level API is sufficient for our needs and avoids fragile coupling to
faer internals. If we ever need in-place decomposition for memory reuse in hot
loops, we'll need to pin a specific faer minor version and test the low-level
API carefully.

faer's `thin_svd()` returns singular values in **ascending** order. We
re-sort descending and permute U/V columns to match the LAPACK convention that
the rest of the codebase assumes. This is a subtle correctness trap — if a
future faer version changes the ordering convention, our tests will catch it.

### Object safety vs default methods

`LinAlgBackend<T>` is designed to be object-safe (`Box<dyn LinAlgBackend<f64>>`),
but `regularized_svd_inverse` has a default implementation that passes `self` as
`&dyn LinAlgBackend<T>` (through `construct_regularized_inverse`). This requires
a `where Self: Sized` bound on that method, which means it can't be called through
a trait object. This is an acceptable trade-off: callers that need the regularized
inverse through a trait object can call `construct_regularized_inverse` directly,
or we can add a non-default required method later.

An alternative design would make `regularized_svd_inverse` a free function
rather than a trait method. Worth reconsidering if more default methods hit
the same `Sized` constraint.

### Borrow checker and faer MatMut conversion

Converting `&mut MatMut<'_, T>` (tk-core's mutable matrix view) to faer's
`faer::MatMut<'_, f64>` can't be done with a regular function because the
double-reference `&mut MatMut<'a, T>` introduces two lifetimes that the
compiler can't unify with faer's expected single lifetime. The solution is a
macro (`faer_mat_mut!`) that expands inline and avoids the intermediate
borrow. This is ugly but correct — the alternatives (unsafe lifetime casts,
restructuring MatMut) are worse.

### SVD debug residual check and truncation

The debug-mode SVD residual assertion (`‖A − UΣV†‖_F / ‖A‖_F < 1e-10`)
initially fired on truncated SVDs, which is expected behavior — a rank-k
approximation doesn't reconstruct the full matrix. The fix is to only run the
residual check when `result.rank == min(m, n)` (i.e., no truncation occurred).
This was a non-obvious interaction between the debug validation and the
truncation feature.

### Scalar generics: type resolution pitfalls

Working with the `Scalar` trait's associated type `T::Real` requires explicit
fully-qualified syntax in many places:

- `<T::Real as num_traits::Zero>::zero()` instead of `T::Real::zero()`
- `<T::Real as num_traits::NumCast>::from(1e-10_f64)` instead of `T::Real::from(...)`
- `<T::Real as num_traits::Float>::epsilon()` instead of `T::Real::epsilon()`

This is because Rust's type inference can't resolve trait implementations on
associated types without explicit disambiguation. It's verbose but unavoidable
without adding more trait bounds or helper methods to `Scalar`.

### Block-sparse GEMM: fusion rule limitations

`compute_fusion_rule` currently only handles rank-2 × rank-2 contraction. This
is deliberate — in DMRG, higher-rank tensors are always reshaped to matrices
(via `fuse_legs`) before contraction. Supporting general rank contraction in
the fusion rule would add significant complexity for no practical benefit in
the DMRG use case.

The Abelian fusion rule is one-to-one (each input sector pair produces at most
one output sector). SU(2) symmetry makes this one-to-many via Clebsch-Gordan
coefficients. The current `Option<PackedSectorKey>` return type will need to
become `Vec<(PackedSectorKey, T::Real)>` for SU(2), which is a breaking change
to the task generation pipeline. This is deferred to Phase 5 and should be
designed together with the SU(2) `BlockSparseTensor` implementation.

### Threading regime: missing upstream API

The tech spec references `BlockSparseTensor::max_sector_dim_on_any_leg()`, but
this method doesn't exist in `tk-symmetry`. Only `max_sector_dim_on_leg(leg_index)`
exists. We implemented `max_sector_dim_any_leg` locally in `threading.rs` by
iterating over all legs. This should eventually be upstreamed to `tk-symmetry`
as a convenience method.

### Composite backend pattern

`DeviceAPI<D, S>` separates dense and sparse dispatch, which is forward-looking
for the case where we want e.g. MKL for dense GEMM but oxiblas for block-sparse
operations. Currently `DefaultDevice = DeviceAPI<DeviceFaer, DeviceFaer>` since
DeviceFaer provides a naive sequential `SparseLinAlgBackend`.

This pattern works well but means trait bounds propagate: any function that
accepts `DeviceAPI<D, S>` needs `D: LinAlgBackend<T>, S: SparseLinAlgBackend<T, Q>`.
In practice, most DMRG code should accept `impl LinAlgBackend<T>` or
`impl SparseLinAlgBackend<T, Q>` and not depend on the composite type directly.

### spmv implementation

The current `spmv` in DeviceFaer uses `BTreeMap`-based offset lookups per sector,
which is O(k log n) where k is the number of sectors. For production use, this
should be replaced with precomputed offset arrays. The block-sparse tensor's
`sector_keys()` is already sorted, so binary search or a flat offset table would
be straightforward.

---

## Ideas for improvement

### Macro-based scalar specialization

The tech spec suggests `macro_rules!` to generate `LinAlgBackend<f32>`,
`LinAlgBackend<f64>`, `LinAlgBackend<C32>`, `LinAlgBackend<C64>` from one template.
The main challenge is the faer type mapping:

| tk-core | faer        |
|---------|-------------|
| f32     | f32         |
| f64     | f64         |
| C32     | faer::c32   |
| C64     | faer::c64   |

A conversion trait `trait IntoFaer { type FaerScalar; }` with four impls would
let the macro body use `T::FaerScalar` everywhere. The complex case also needs
conjugation handling (faer uses `Conj` enum, tk-core uses `is_conjugated` bool).

### Rayon integration for block_gemm

The `FragmentedSectors` path should use `rayon::iter::ParallelIterator`. Key
concern: BLAS thread safety. When using multi-threaded BLAS (MKL/OpenBLAS),
each Rayon task must set BLAS threads to 1 to avoid thread oversubscription.
The `set_blas_num_threads` function exists for this but is currently a no-op.

Proposed approach:
1. `FatSectors`: set BLAS threads = available cores, run tasks sequentially
2. `FragmentedSectors`: set BLAS threads = 1, dispatch via `par_iter()`
3. Restore original BLAS thread count after block_gemm completes

### Truncation error reporting

`SvdResult::truncation_error` currently stores the sum of discarded squared
singular values. It might be more useful to store the relative truncation error
(ratio to total norm) so callers don't need to recompute the full norm. This
would require computing the full SVD norm during truncation, which is cheap
(sum of all σ²).

### Benchmark suite

The crate needs Criterion benchmarks for:
- Dense GEMM at various sizes (to calibrate `GPU_DISPATCH_THRESHOLD`)
- SVD gesdd vs gesvd performance crossover
- Block-sparse GEMM with realistic sector distributions from Heisenberg/Hubbard models
- Threading regime selection accuracy (does the heuristic pick the faster path?)
