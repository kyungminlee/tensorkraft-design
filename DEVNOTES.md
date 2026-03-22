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

## tk-linalg: Lessons Learned & Spec Feedback

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

**Recommendation:** Future tech specs should document the expected singular value
ordering convention explicitly (descending, matching LAPACK) and note that faer
uses ascending order internally. This ordering mismatch is a correctness hazard
that would not be caught by compilation alone.

### faer does not distinguish gesdd from gesvd — spec's fallback is a no-op

The tech spec (§3.2, §8.1.2) specifies a gesdd → gesvd fallback: try divide-and-
conquer first, fall back to QR-iteration on convergence failure. In practice, faer's
high-level `thin_svd()` does not expose separate algorithm selection — there is no
way to request "gesdd" vs "gesvd" specifically. The implementation maps both
`svd_truncated_gesdd` and `svd_truncated_gesvd` to the same `faer_mat.thin_svd()` call.

This means the fallback path in `svd_truncated()` is currently a no-op: if `gesdd`
fails (which it can't, since it's the same code), `gesvd` would produce the same
result. The fallback mechanism will only become meaningful when MKL or OpenBLAS
backends are implemented, which do expose separate `LAPACKE_dgesdd` / `LAPACKE_dgesvd`.

**Recommendation:** Future tech specs should note which backend-specific behaviors
a trait method relies on. When the default backend cannot distinguish the two code
paths, the spec should acknowledge that the fallback is a structural contract, not
a functional one, until FFI backends are available.

### `DenseTensor` lifetime `'static` required for SVD/eigh return types — spec omission

The tech spec defines `SvdResult<T>` as containing `DenseTensor<T>` fields (U, Vt)
without specifying a lifetime. Since `DenseTensor<'a, T>` carries a lifetime from
tk-core's CoW design, return types like `SvdResult`, `EighResult`, and `QrResult`
must use `DenseTensor<'static, T>` — the returned tensors always own their data.
Similarly, `regularized_svd_inverse` takes `&DenseTensor<'static, T>` parameters
and returns `DenseTensor<'static, T>`.

This `'static` requirement wasn't anticipated by the tech spec, which shows
`DenseTensor<T>` without lifetimes. The practical impact is minor (SVD results
are always owned), but it means `SvdResult` cannot hold arena-borrowed tensors.
If a future optimization needs arena-allocated SVD workspace, the result types
would need lifetime parameters, which would be a breaking change.

**Recommendation:** When a downstream crate depends on types from a leaf crate
that carries lifetimes, the tech spec should specify whether return types are
always owned (`'static`) or may borrow. This determines the API's flexibility
for future arena-based optimizations.

### `construct_regularized_inverse` takes `&dyn LinAlgBackend<T>` — hidden dynamic dispatch

The tech spec shows `construct_regularized_inverse` as a simple helper that
reconstructs `V · diag(inv_s) · U†`. The actual implementation takes
`backend: &dyn LinAlgBackend<T>` to perform the final GEMM, introducing dynamic
dispatch in a function that the spec implies is purely arithmetic. This design
emerges because `construct_regularized_inverse` is called from the default method
`regularized_svd_inverse`, which needs to delegate the GEMM back to the concrete
backend.

This also forces the `where Self: Sized` bound on `regularized_svd_inverse`,
breaking object-safety for that one method. Callers using `Box<dyn LinAlgBackend<T>>`
cannot call `regularized_svd_inverse` directly.

**Recommendation:** When a tech spec provides default trait method implementations
that call other trait methods internally, note which methods need `Self: Sized` and
the impact on `dyn` dispatch. An alternative design is to make `regularized_svd_inverse`
a free function rather than a trait method.

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

**Recommendation:** Future tech specs that bridge between tk-core matrix views
and FFI/external-crate matrix views should note the double-lifetime issue and
suggest macros as the conversion mechanism upfront. This saves implementors
from discovering the issue through compiler errors.

### SVD debug residual check and truncation

The debug-mode SVD residual assertion (`‖A − UΣV†‖_F / ‖A‖_F < 1e-10`)
initially fired on truncated SVDs, which is expected behavior — a rank-k
approximation doesn't reconstruct the full matrix. The fix is to only run the
residual check when `result.rank == min(m, n)` (i.e., no truncation occurred).
This was a non-obvious interaction between the debug validation and the
truncation feature.

The tech spec's pseudocode (§3.2) shows the residual check unconditionally inside
`svd_truncated`. The actual implementation adds a guard: `if result.rank == full_rank`.
It also adds a guard for near-zero norm (to avoid division by zero on zero matrices)
and uses `NumCast::from` for the threshold to work across scalar types.

**Recommendation:** When a spec defines debug assertions, enumerate the conditions
under which the assertion should be skipped. Truncation, zero-norm, and scalar-type-
dependent thresholds are foreseeable edge cases that should be called out.

### Scalar generics: type resolution pitfalls

Working with the `Scalar` trait's associated type `T::Real` requires explicit
fully-qualified syntax in many places:

- `<T::Real as num_traits::Zero>::zero()` instead of `T::Real::zero()`
- `<T::Real as num_traits::NumCast>::from(1e-10_f64)` instead of `T::Real::from(...)`
- `<T::Real as num_traits::Float>::epsilon()` instead of `T::Real::epsilon()`

This is because Rust's type inference can't resolve trait implementations on
associated types without explicit disambiguation. It's verbose but unavoidable
without adding more trait bounds or helper methods to `Scalar`.

### `compute_fusion_rule` signature differs from tech spec — extra parameters needed

The tech spec (§7.2) defines `compute_fusion_rule<Q>(key_a, key_b, rank_a, rank_b, flux)`
with five parameters. The actual implementation adds two more: `_indices_a` and
`_indices_b` (currently unused, prefixed with `_`). These were added anticipating
that non-rank-2 contractions or more complex flux validation might need access to
the full QIndex structures.

The parameters are unused today and could be removed, but keeping them avoids a
future signature change when the general-rank case is eventually supported.

**Recommendation:** When a tech spec defines internal function signatures, mark
parameters that are reserved for future extensions. The `_prefix` convention is
not visible in the spec's signature, leading to apparent mismatches during review.

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

### block_gemm accumulation: spec misses the multi-input-to-same-output case

The tech spec's block_gemm pseudocode (§7.3) generates one `SectorGemmTask` per
compatible input pair, dispatches them via Rayon, and then sorts results by key.
It does not explicitly address the case where multiple input pairs produce tasks
with the same output sector key. In the Abelian case, this happens when multiple
(key_a, key_b) pairs fuse to the same output key.

The actual implementation handles this with an accumulation step: after each GEMM,
it checks if a result for the output key already exists and adds to it element-wise.
This is a linear scan (`results.iter_mut().find(...)`) that is O(n_output_sectors)
per task — acceptable for small sector counts but would need a HashMap for large
symmetric tensors.

**Recommendation:** The spec should explicitly document the accumulation requirement
and suggest an efficient data structure for it. For the Rayon-parallel path, this
is non-trivial: parallel tasks writing to the same output key need either
pre-grouped task batches (group tasks by output key, sequential accumulation within
each group) or thread-safe accumulators (mutex per output key).

### `DefaultDevice` differs from tech spec — DeviceFaer serves double duty

The tech spec (§8.6) defines `DefaultDevice = DeviceAPI<DeviceFaer, DeviceOxiblas>`
requiring both `backend-faer` and `backend-oxiblas` features. The actual
implementation uses `DefaultDevice = DeviceAPI<DeviceFaer, DeviceFaer>` because
`DeviceOxiblas` is not yet implemented. DeviceFaer provides a naive sequential
`SparseLinAlgBackend` stub.

This is a pragmatic deviation, but it means the `DefaultDevice` type alias has a
different concrete meaning than the spec intends. When `DeviceOxiblas` is eventually
implemented, changing `DefaultDevice` will be a type-breaking change for any code
that explicitly depends on the sparse component being `DeviceFaer`.

**Recommendation:** When a spec defines type aliases that depend on unimplemented
features, the spec should note the interim alias and the migration path.

### Threading regime: missing upstream API

The tech spec references `BlockSparseTensor::max_sector_dim_on_any_leg()`, but
this method doesn't exist in `tk-symmetry`. Only `max_sector_dim_on_leg(leg_index)`
exists. We implemented `max_sector_dim_any_leg` locally in `threading.rs` by
iterating over all legs. This should eventually be upstreamed to `tk-symmetry`
as a convenience method.

**Recommendation:** When a tech spec references methods on types from other crates,
verify that those methods actually exist in the dependency's API. Missing convenience
methods are easy to implement locally but create API fragmentation when multiple
crates need the same helper.

### Threading regime: two-regime vs three-phase partitioned scheduler

The architecture doc (§5.3, v8.0) specifies a two-phase partitioned scheduler
that splits the LPT task queue at `BLAS_FLOP_THRESHOLD`: heavy-head tasks get
multithreaded BLAS, the light tail gets Rayon with single-threaded BLAS. The
tech spec (§6.1) simplifies this to a binary `ThreadingRegime::select()` heuristic
based on max sector dimension.

The implementation follows the tech spec's simpler binary heuristic, not the
architecture doc's partitioned scheduler. This is appropriate for Phase 1–3 where
all backends are pure-Rust (faer handles its own parallelism), but the binary
heuristic will need to be replaced with the partitioned approach when FFI BLAS
backends with global thread state are integrated.

### `to_faer_mat_ref` and `tk_mat_to_faer_owned` — unsafe conversion and costly copies

Converting tk-core matrix views to faer views requires two distinct helpers:

1. `to_faer_mat_ref()` — uses `unsafe { faer::mat::from_raw_parts() }` to create a
   zero-copy faer view from tk-core's raw pointer and strides. This is sound because
   tk-core guarantees the data is valid for the lifetime, but the `unsafe` block is
   not documented in the spec.

2. `tk_mat_to_faer_owned()` — performs an O(m×n) element-wise copy into a new
   `faer::Mat<f64>`. This is necessary because faer's `thin_svd()`, `qr()`, and
   `selfadjoint_eigendecomposition()` all require owned `faer::Mat`, not views. Every
   SVD, QR, and eigh call pays this copy cost.

Neither the tech spec nor architecture doc mentions this conversion overhead. For
large matrices (e.g., D=1000, producing a 1M-element copy), this could be
significant relative to the decomposition cost itself.

**Recommendation:** Future specs should note when the backend API requires owned
data (not views) and account for the copy overhead in performance estimates. If
faer's low-level API (which operates on views) is eventually adopted, this copy
can be eliminated.

### spmv implementation

The current `spmv` in DeviceFaer uses `BTreeMap`-based offset lookups per sector,
which is O(k log n) where k is the number of sectors. For production use, this
should be replaced with precomputed offset arrays. The block-sparse tensor's
`sector_keys()` is already sorted, so binary search or a flat offset table would
be straightforward.

### Composite backend pattern

`DeviceAPI<D, S>` separates dense and sparse dispatch, which is forward-looking
for the case where we want e.g. MKL for dense GEMM but oxiblas for block-sparse
operations. Currently `DefaultDevice = DeviceAPI<DeviceFaer, DeviceFaer>` since
DeviceFaer provides a naive sequential `SparseLinAlgBackend`.

This pattern works well but means trait bounds propagate: any function that
accepts `DeviceAPI<D, S>` needs `D: LinAlgBackend<T>, S: SparseLinAlgBackend<T, Q>`.
In practice, most DMRG code should accept `impl LinAlgBackend<T>` or
`impl SparseLinAlgBackend<T, Q>` and not depend on the composite type directly.

### Test coverage gaps relative to tech spec

The tech spec (§15.1) lists 26 unit tests and 5 property-based tests. The
implementation has 10 unit tests — roughly 40% of the spec'd suite. Key gaps:

- **Complex-valued tests** — no C32/C64 GEMM, SVD, or conjugation tests
  (because only f64 is implemented via DeviceFaer)
- **Conjugation-aware GEMM tests** — `gemm_conjugated_a_c64`, `gemm_both_conjugated_c64`,
  `gemm_real_ignores_conjugation` are all missing
- **SVD cutoff truncation** — `svd_truncation_cutoff` and `svd_truncation_error_sum`
  tests are missing
- **block_gemm integration tests** — `block_gemm_sector_presence`,
  `block_gemm_equivalence_dense`, `block_gemm_flux` are not yet implemented
- **spmv correctness test** — missing
- **Property-based tests** — `gemm_associativity`, `svd_truncation_monotonic` not yet written
- **BLAS layout tests** — not applicable until MKL/OpenBLAS backends exist

This is expected for a draft implementation, but the spec serves as a roadmap for
test coverage when the remaining scalar types and backends are implemented.

---

## tk-symmetry: Lessons Learned & Spec Feedback

### `leg_directions` field missing from architecture doc's `BlockSparseTensor`

The architecture doc (§4.2) defines `BlockSparseTensor` with four fields: `indices`,
`sector_keys`, `sector_blocks`, and `flux`. The actual implementation adds a fifth
field: `leg_directions: Vec<LegDirection>`, which records whether each leg is
`Incoming` or `Outgoing`. This field is essential for the flux rule
(`check_flux_rule`), which fuses quantum numbers respecting direction: incoming legs
contribute directly, outgoing legs contribute their `dual()`.

Without `leg_directions`, the flux rule cannot determine which quantum numbers to
dual-conjugate. The tech spec (§7.6) mentions the flux rule but its constructor
signatures (`zeros(indices, flux)`, `from_blocks(indices, flux, blocks)`) omit the
`leg_directions` parameter. The actual constructors are
`zeros(indices, flux, leg_directions)` and `from_blocks(indices, flux, leg_directions, blocks)`.

**Recommendation:** Architecture doc struct definitions should include all fields
that participate in invariant enforcement. The `leg_directions` field is not an
implementation detail — it's part of the tensor's mathematical identity.

### `QuantumNumber` trait has `'static` bound not shown in architecture doc

The architecture doc (§4.1) defines `QuantumNumber` without a `'static` bound:
`trait QuantumNumber: Clone + Eq + Hash + Debug`. The implementation adds `'static`,
which is required because `BlockSparseTensor` stores `QIndex<Q>` values that must
outlive any borrow scope. Without `'static`, the compiler cannot guarantee that
quantum number values stored inside tensors remain valid.

This is the same pattern as tk-core's `Scalar: 'static` — the architecture doc
consistently omits `'static` bounds, but every trait whose values are stored in
long-lived structs needs it.

### Constructor signatures differ from spec — `leg_directions` parameter required

The tech spec (§7.1) shows:
- `zeros(indices: &[QIndex<Q>], flux: Q) -> BlockSparseTensor<T, Q>`
- `from_blocks(indices, flux, blocks) -> Result<..., SymmetryError>`

The actual implementation requires `leg_directions` as an additional parameter in
both constructors. This cascades to every call site: any code creating a
`BlockSparseTensor` must know the direction of each leg upfront. This is physically
correct (direction is an intrinsic property of a tensor leg in symmetric tensor
networks), but the spec's omission makes the API look simpler than it actually is.

**Recommendation:** When a data structure has an invariant that requires per-field
metadata (like per-leg direction), the spec should include it in all constructor
signatures, not just in the field list.

### `permute()` is NOT zero-copy at block-sparse level — spec says "no data is copied"

The tech spec (§7.4) states that `permute()` "rearranges legs without copying data"
and "no data is copied." The implementation calls `block.permute(perm).into_owned()`
on every sector block. While `DenseTensor::permute()` is indeed a zero-copy view
(stride reorder), the `.into_owned()` forces a full data gather for each block
because permutation changes the strides to non-contiguous.

For a tensor with k sectors of average size n, this is O(k × n) element copies —
not zero-copy. The spec's claim holds only for dense tensors, not block-sparse ones.
This is because block-sparse tensors need contiguous blocks for BLAS (GEMM requires
contiguous row/column-major data), so each block must be materialized after permutation.

**Recommendation:** When a spec makes a performance claim like "zero-copy," it should
specify which representation it applies to. Block-sparse permutation is inherently
more expensive than dense permutation due to the contiguity requirement of each block.

### `fuse_legs` algorithm complexity not documented in spec

The tech spec (§7.7) describes `fuse_legs` conceptually: it combines adjacent legs
into a single fused leg via Cartesian product of quantum numbers. The actual
implementation is substantially more complex:

1. **Cartesian product enumeration** — iterates all combinations of sector indices
   on the fused legs to produce fused quantum numbers
2. **BTreeMap-based offset map** — maps each fused quantum number to a dimension offset,
   using `BTreeMap` for deterministic ordering (not `HashMap`)
3. **Block scatter** — for each original sector, determines the fused quantum number,
   looks up the offset, and copies data into the fused block at the correct position

The BTreeMap choice ensures that fused sector ordering is deterministic across runs,
which matters for reproducibility of DMRG sweeps. The Cartesian product step is
O(∏ sector_counts) which can be large for many-leg fusions.

### `split_leg` needs extra `original_directions` parameter — not in spec

The tech spec defines `split_leg(leg_index, original_indices)` with two parameters.
The implementation adds a third: `original_directions: &[LegDirection]`, which
provides the leg directions of the sub-legs being restored. This is needed because
the fused leg has a single direction, but the original sub-legs may have had mixed
directions (`Incoming` and `Outgoing`), and the unfusing must reconstruct the correct
flux rule for each sub-leg.

This parameter was noted in `DRAFT_NOTES.md` as a necessary addition discovered
during implementation. The asymmetry between `fuse_legs` (which reads directions
from the tensor) and `split_leg` (which needs them externally) is because fusing
loses the per-sub-leg direction information.

### `FlatBlockStorage` loses shape info for non-rank-2 blocks

The tech spec (§7.5) describes `FlatBlockStorage` as a GPU-ready contiguous layout
for block data. The implementation uses `flatten()` and `unflatten()` methods.
During `flatten()`, each block's shape is mapped to `(rows, cols)` for rank-2 blocks,
but non-rank-2 blocks fall back to `(numel, 1)`, losing the original shape.

On `unflatten()`, blocks are reconstructed with the `(numel, 1)` shape — the
original tensor shape is not restored. This means flatten→unflatten is not a
round-trip for tensors with rank != 2. This is acceptable because `FlatBlockStorage`
is only used for BLAS dispatch (which requires rank-2), but the loss of shape
information should be documented as a constraint.

### `flatten()` uses `alloc_slice_uninit` — unsafe not mentioned in spec

`FlatBlockStorage::flatten()` uses `unsafe { arena.alloc_slice_uninit::<T>(total_elems) }`
to allocate the contiguous buffer, then overwrites it with `copy_from_slice`. This
is sound (the uninit memory is fully written before any read), but the spec does
not mention the unsafe block. The arena-based allocation is a performance
optimization to avoid individual allocations per block.

### CG cache uses hand-rolled Racah formula instead of `lie-groups` dependency

The tech spec (§10.3) mentions `lie-groups` as an optional dependency for Clebsch-
Gordan coefficient computation. The implementation uses a hand-rolled Racah formula
in `cg_cache.rs` (318 lines) with direct factorial computation and the standard
CG series summation. The `lie-groups` crate is not used.

This is a pragmatic choice: the CG coefficient formula is well-known and
self-contained, and adding an external dependency for a single formula would
increase compile times and coupling. The `DRAFT_NOTES.md` notes that the `lie-groups`
dependency decision is still pending. The hand-rolled implementation includes:
- `ClebschGordanCache` with `DashMap`-based thread-safe lazy caching
- Racah formula with exact rational arithmetic via factorials
- Validation for triangle inequality and selection rules

### All `DenseTensor` blocks use `'static` lifetime — ownership invariant

`BlockSparseTensor<T, Q>` stores blocks as `Vec<DenseTensor<'static, T>>`. This
means every block must own its data — no arena-borrowed blocks allowed. This
simplifies memory management (blocks can be freely moved, cloned, and stored in
collections) but means that `fuse_legs`, `split_leg`, and `permute` must always
produce owned blocks, paying allocation and copy costs.

If a future optimization needs arena-allocated blocks (e.g., for workspace reuse
in DMRG sweeps), the lifetime parameter would need to propagate to
`BlockSparseTensor<'a, T, Q>`, which is a pervasive change affecting every function
that handles block-sparse tensors.

### Error handling uses panics over `Result` in many paths

The tech spec (§11) defines `SymmetryError` with six variants and `SymResult<T>`.
However, many implementation paths use `panic!` or `debug_assert!` rather than
returning `SymmetryError`:

- `zeros()` panics on flux rule violation instead of returning `SymmetryError::FluxViolation`
- `from_blocks()` panics on shape mismatches
- `fuse_legs()` / `split_leg()` panic on invalid leg indices
- `PackedSectorKey::pack()` panics on overflow

The `SymmetryError` type exists and is well-defined, but the constructors and
operations prefer panic-on-violation over graceful error propagation. This is
acceptable for internal use (callers in DMRG are expected to provide valid inputs),
but differs from the spec's error-returning signatures.

### Test count exceeds spec but property-based tests are absent

`DRAFT_NOTES.md` reports 40 passing unit tests, versus the tech spec's 19 unit test
specifications and 3 proptest specifications. The implementation has more tests than
specified — good coverage — but the proptest-based property tests (quantum number
group axioms, flux rule invariance, pack/unpack roundtrip) are not yet written.

Property-based tests are particularly valuable for symmetry code because group axioms
(associativity of fuse, identity element, dual as inverse) have subtle edge cases
that unit tests may miss. The `DRAFT_NOTES.md` lists property-based tests as a
known gap.

### `iter_keyed_blocks()` — useful method not in spec

The implementation provides `iter_keyed_blocks()` returning an iterator over
`(PackedSectorKey, &DenseTensor<'static, T>)` pairs. This is heavily used in
`fuse_legs`, `split_leg`, `flatten`, and `block_gemm` task generation. The spec
defines `sector_keys()` and `sector_block(key)` separately, but the paired iterator
pattern is more ergonomic and avoids repeated key lookups.

### `from_raw_parts()` is `pub(crate)` — escape hatch for `unflatten`

`BlockSparseTensor::from_raw_parts()` bypasses all validation (flux rule, sorted
key invariant) and constructs a tensor from pre-built components. It's used by
`FlatBlockStorage::unflatten()` to reconstruct a tensor after compute operations
modify the flat buffer. This is a sound design (unflatten preserves the structural
invariants from the original flatten), but the `pub(crate)` visibility means any
code within `tk-symmetry` can bypass validation. Future maintenance should be
careful not to call `from_raw_parts` without ensuring invariants hold.

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
