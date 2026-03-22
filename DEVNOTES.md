# Development Notes — tensorkraft

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
