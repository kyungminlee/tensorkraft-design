# Technical Specification: `tk-linalg`

**Crate:** `tensorkraft/crates/tk-linalg`
**Version:** 0.1.0 (Draft Implementation)
**Status:** Draft
**Last Updated:** March 2026

---

## 1. Overview

`tk-linalg` is the linear algebra backend abstraction layer for the tensorkraft workspace. It sits directly above `tk-core` and `tk-symmetry` in the dependency graph and is consumed by every higher-level crate that performs numerical computation: `tk-contract`, `tk-dmrg`, and `tk-dmft`.

**Core responsibilities:**

- **Backend trait abstraction** — Define `LinAlgBackend<T>` and `SparseLinAlgBackend<T, Q>` as object-safe trait interfaces for GEMM, SVD, EVD, QR, and related operations. Backend selection is entirely compile-time via feature flags; no dynamic dispatch overhead in inner loops. Note: `regularized_svd_inverse` requires a `where Self: Sized` bound on the default implementation (which calls `construct_regularized_inverse` with `&dyn LinAlgBackend<T>`), breaking object-safety for that method only.
- **Conjugation-aware GEMM** — Propagate `MatRef::is_conjugated` through to hardware micro-kernels (faer lazy conjugation, BLAS `CblasConjTrans`), eliminating the O(N) conjugation memory passes that would otherwise saturate memory bandwidth before computation begins.
- **SVD with residual validation** — Use faer's high-level `Mat::thin_svd()` API. The faer high-level API does not expose separate algorithm selection (`gesdd` vs `gesvd`); both `svd_truncated_gesdd` and `svd_truncated_gesvd` map to the same `thin_svd()` call. The `gesdd`-to-`gesvd` fallback is meaningful only with MKL/OpenBLAS FFI backends. Validate reconstruction residual in debug builds with appropriate guards (see S3.3). **Important:** faer returns singular values in ascending order; the implementation re-sorts to descending order to match LAPACK convention. This is a correctness trap that must be tested explicitly.
- **Tikhonov-regularized pseudo-inverse** — For TDVP gauge restoration: compute `s / (s² + δ²)` instead of `1/s`, preventing NaN explosion when singular values approach machine zero.
- **LPT-scheduled block-sparse GEMM** — For `SparseLinAlgBackend`: sort sector GEMM tasks by descending FLOP cost (Longest Processing Time scheduling) before dispatching. Multiple input sector pairs can map to the same output sector (accumulation case); the current implementation uses linear scan for same-output-sector accumulation, which should be replaced with `HashMap` for large sector counts.
- **Hybrid threading regime selection** — Automatically switch between fat-sector mode (multithreaded BLAS per sector) and fragmented-sector mode (Rayon-parallel single-threaded BLAS per sector) to prevent thread oversubscription. The current implementation uses a simple binary heuristic, not the partitioned scheduler described in the architecture doc. This is appropriate for Phase 1-3.
- **Concrete backend implementations** — `DeviceFaer` (pure-Rust, default). `DeviceOxiblas`, `DeviceMKL`, `DeviceOpenBLAS`, and `DeviceCuda` are specified but not yet implemented.

**Current implementation state:**

- Only `f64` is implemented. `f32`, `Complex<f32>` (`C32`), and `Complex<f64>` (`C64`) are not yet done. Macro-based specialization is recommended for adding these.
- `DefaultDevice` is currently `DeviceAPI<DeviceFaer, DeviceFaer>`, not `DeviceAPI<DeviceFaer, DeviceOxiblas>` as originally specified. `DeviceOxiblas` is not yet implemented.
- Rayon-parallel dispatch in `block_gemm` is deferred; LPT sorting is implemented but execution is sequential.
- `max_sector_dim_on_any_leg()` referenced in the threading heuristic does not exist in `tk-symmetry`. It is implemented locally in `threading.rs`.

Mathematical operations on tensors (contraction, trace, element-wise) are implemented in `tk-contract` and higher-level crates, not here.

---

## 2. Module Structure

```
tk-linalg/
├── Cargo.toml
├── build.rs                  # compile_error! for mutually-exclusive BLAS flags
└── src/
    ├── lib.rs                re-exports all public items
    ├── traits.rs             LinAlgBackend<T>, SparseLinAlgBackend<T, Q>
    ├── results.rs            SvdResult<T>, EighResult<T>, QrResult<T>, SvdConvergenceError
    ├── threading.rs          ThreadingRegime, select() heuristic, max_sector_dim_on_any_leg()
    ├── tasks.rs              SectorGemmTask<T>, LPT scheduling, fusion_rule
    ├── faer_convert.rs       faer_mat_mut! macro, to_faer_mat_ref(), tk_mat_to_faer_owned()
    ├── device/
    │   ├── mod.rs            DeviceAPI<D, S>, DefaultDevice type alias
    │   ├── faer.rs           DeviceFaer: LinAlgBackend<f64>
    │   ├── oxiblas.rs        DeviceOxiblas: SparseLinAlgBackend<T, Q> (stub)
    │   ├── mkl.rs            DeviceMKL (cfg: backend-mkl) (stub)
    │   ├── openblas.rs       DeviceOpenBLAS (cfg: backend-openblas) (stub)
    │   └── cuda.rs           DeviceCuda (cfg: backend-cuda) (stub)
    └── error.rs              LinAlgError, LinAlgResult<T>
```

---

## 3. The `LinAlgBackend<T>` Trait

### 3.1 Object Safety Design

The trait is parameterized at the **trait level** over `T: Scalar`. This is the critical design decision that restores object safety (Rust E0038). When each method is individually generic over `T`, the vtable cannot be constructed. By fixing `T` at the trait level, all method signatures are concrete for a given `T`, and `Box<dyn LinAlgBackend<f64>>` is a valid Rust expression.

**Object-safety exception:** `regularized_svd_inverse` has a default implementation that calls `construct_regularized_inverse`, which takes `&dyn LinAlgBackend<T>`. This introduces hidden dynamic dispatch. The method requires a `where Self: Sized` bound, which means it cannot be called through a trait object. This is acceptable because `regularized_svd_inverse` is always called on concrete backend types in practice.

- **Inner loops (GEMM dispatch, matvec):** static dispatch via `impl LinAlgBackend<T> for DeviceFaer`. Zero virtual-call overhead.
- **Sweep scheduler level:** may use `Box<dyn LinAlgBackend<T>>` if compile-time monomorphization budget becomes a concern (see S12 on monomorphization).

### 3.2 Definition

```rust
/// Object-safe linear algebra backend, parameterized over a single scalar type.
///
/// Implementations provide: GEMM (conjugation-aware), truncated SVD,
/// lowest eigenvalue/eigenvector (for reference only; DMRG uses in-house Krylov),
/// QR decomposition, and Tikhonov-regularized pseudo-inverse for TDVP gauge shifts.
///
/// # Object Safety
///
/// This trait is object-safe: `Box<dyn LinAlgBackend<f64>>` compiles. The scalar
/// type T is a trait-level parameter, not a per-method generic. This differs from
/// the pre-v5.0 design which had per-method generics and violated E0038.
///
/// Exception: `regularized_svd_inverse` has a `where Self: Sized` bound due to
/// its use of `construct_regularized_inverse` which takes `&dyn LinAlgBackend<T>`.
/// This method is not callable through a trait object.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync`. BLAS handles (MKL, OpenBLAS) must be
/// initialized once and wrapped in types that satisfy this contract.
///
/// # Scalar Generics
///
/// Implementation note: scalar generics require fully-qualified syntax for
/// associated type bounds, e.g., `<T::Real as num_traits::Zero>::zero()`.
/// This affects all methods that operate on `T::Real`.
pub trait LinAlgBackend<T: Scalar>: Send + Sync {
    // -----------------------------------------------------------------------
    // SVD
    // -----------------------------------------------------------------------

    /// Compute truncated SVD of `mat`, retaining at most `max_rank` singular values
    /// with singular value >= `cutoff` (relative to the largest singular value).
    ///
    /// **Algorithm selection:** defaults to divide-and-conquer (`gesdd`) for speed.
    /// Falls back to QR-iteration (`gesvd`) on convergence failure. Note: with the
    /// faer backend, both `gesdd` and `gesvd` map to the same `thin_svd()` call;
    /// the fallback is meaningful only with MKL/OpenBLAS FFI backends.
    ///
    /// **Singular value ordering:** faer returns singular values in ascending order.
    /// The implementation re-sorts to descending order to match LAPACK convention.
    /// Callers must not assume any particular ordering from the backend; they should
    /// rely on the `SvdResult` contract (descending order).
    ///
    /// **Residual validation (debug only):** after SVD returns, asserts
    /// `||A - U*Sigma*Vt||_F / ||A||_F < 1e-10` with the following guards:
    /// - Only runs when `result.rank == min(m, n)` (full rank, not truncated)
    /// - Near-zero norm guard: skips check if `||A||_F < epsilon` to avoid division by zero
    /// - Uses `NumCast` for threshold conversion across scalar types
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the reconstruction residual exceeds 1e-10
    /// (only for full-rank, non-trivial matrices).
    ///
    /// # Errors
    ///
    /// Returns `LinAlgError::SvdConvergence` if both `gesdd` and `gesvd` fail
    /// to converge (pathological input only).
    fn svd_truncated(
        &self,
        mat: &MatRef<T>,
        max_rank: usize,
        cutoff: T::Real,
    ) -> LinAlgResult<SvdResult<T>> {
        match self.svd_truncated_gesdd(mat, max_rank, cutoff) {
            Ok(result) => {
                #[cfg(debug_assertions)]
                {
                    let full_rank = std::cmp::min(mat.rows, mat.cols);
                    // Only validate residual for full-rank (non-truncated) results
                    if result.rank == full_rank {
                        let norm = frobenius_norm(mat);
                        // Guard against division by zero for near-zero matrices
                        let eps = <T::Real as num_traits::NumCast>::from(1e-30)
                            .unwrap_or(<T::Real as num_traits::Zero>::zero());
                        if norm > eps {
                            let residual = svd_reconstruction_error(mat, &result);
                            let threshold = <T::Real as num_traits::NumCast>::from(1e-10)
                                .unwrap_or(<T::Real as num_traits::Zero>::zero());
                            debug_assert!(
                                residual / norm < threshold,
                                "SVD reconstruction residual {:.2e} exceeds tolerance 1e-10",
                                residual / norm
                            );
                        }
                    }
                }
                Ok(result)
            }
            Err(SvdConvergenceError) => {
                log::warn!(
                    target: "tensorkraft::linalg",
                    "gesdd failed to converge on {}x{} matrix; falling back to gesvd",
                    mat.rows, mat.cols
                );
                self.svd_truncated_gesvd(mat, max_rank, cutoff)
                    .map_err(LinAlgError::from)
            }
        }
    }

    /// Truncated SVD using divide-and-conquer (`gesdd`).
    ///
    /// O(mn*min(m,n)) time, O(min(m,n)^2) extra workspace.
    /// Faster than `gesvd` but uses more workspace and can fail to converge on
    /// highly degenerate singular values.
    ///
    /// **faer backend note:** Maps to `Mat::thin_svd()` high-level API (faer 0.19).
    /// The low-level faer API is fragile across minor versions and is not used.
    /// faer returns singular values in ascending order; the implementation re-sorts
    /// to descending order.
    ///
    /// # Errors
    ///
    /// Returns `Err(SvdConvergenceError)` if the divide-and-conquer routine
    /// does not converge. This is rare but possible with nearly-rank-deficient input.
    fn svd_truncated_gesdd(
        &self,
        mat: &MatRef<T>,
        max_rank: usize,
        cutoff: T::Real,
    ) -> Result<SvdResult<T>, SvdConvergenceError>;

    /// Truncated SVD using QR iteration (`gesvd`).
    ///
    /// O(mn*min(m,n)) time, O(min(m,n)) extra workspace.
    /// Slower than `gesdd` but guaranteed to converge for all non-pathological inputs.
    /// Used as a fallback when `gesdd` fails.
    ///
    /// **faer backend note:** With `DeviceFaer`, this maps to the same `Mat::thin_svd()`
    /// call as `svd_truncated_gesdd`. The fallback is a no-op with faer. It is meaningful
    /// only with MKL/OpenBLAS FFI backends which expose separate `gesdd`/`gesvd` routines.
    ///
    /// # Errors
    ///
    /// Returns `Err` only for genuinely pathological inputs (all-NaN, infinite values).
    fn svd_truncated_gesvd(
        &self,
        mat: &MatRef<T>,
        max_rank: usize,
        cutoff: T::Real,
    ) -> Result<SvdResult<T>, SvdConvergenceError>;

    // -----------------------------------------------------------------------
    // GEMM
    // -----------------------------------------------------------------------

    /// Conjugation-aware GEMM: C = alpha*op(A)*op(B) + beta*C.
    ///
    /// The `is_conjugated` flag on each `MatRef` determines the operation applied
    /// to each input before multiplication:
    ///
    /// | `a.is_conjugated` | `b.is_conjugated` | Operation |
    /// |:------------------|:------------------|:----------|
    /// | false             | false             | C = alpha*A*B + beta*C |
    /// | true              | false             | C = alpha*conj(A)*B + beta*C |
    /// | false             | true              | C = alpha*A*conj(B) + beta*C |
    /// | true              | true              | C = alpha*conj(A)*conj(B) + beta*C |
    ///
    /// Note: `adjoint()` on `MatRef` also swaps strides (transpose), so
    /// a call with `a = original.adjoint()` produces C = alpha*A^dagger*B + beta*C.
    ///
    /// For real `T` (`T::is_real() == true`), all four cases reduce to
    /// C = alpha*A*B + beta*C. Backends may skip conjugation-flag checks in the
    /// real path using `T::is_real()`.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if dimension compatibility is violated:
    /// `a.cols != b.rows` (before transpose adjustment) or `c` dimensions
    /// do not match the expected output shape.
    fn gemm(
        &self,
        alpha: T,
        a: &MatRef<T>,
        b: &MatRef<T>,
        beta: T,
        c: &mut MatMut<T>,
    );

    // -----------------------------------------------------------------------
    // Eigenvalue decomposition
    // -----------------------------------------------------------------------

    /// Compute the lowest `k` eigenvalues and corresponding eigenvectors of a
    /// symmetric/Hermitian matrix.
    ///
    /// This is a **full dense EVD** suitable for small matrices (e.g., small
    /// auxiliary matrices in MPO compression). For the iterative ground-state
    /// eigenvalue problem in DMRG sweeps, the in-house `LanczosSolver` /
    /// `DavidsonSolver` in `tk-dmrg` must be used instead — they avoid
    /// materializing the full dense Hamiltonian.
    ///
    /// **faer backend note:** Maps to `Mat::selfadjoint_eigendecomposition()` high-level
    /// API (faer 0.19).
    ///
    /// # Panics
    ///
    /// Panics if `k > mat.rows`. Panics in debug mode if `mat` is not square.
    ///
    /// # Returns
    ///
    /// `(eigenvalues, eigenvectors)` where `eigenvalues` is sorted ascending
    /// and `eigenvectors` has columns as eigenvectors.
    fn eigh_lowest(
        &self,
        mat: &MatRef<T>,
        k: usize,
    ) -> LinAlgResult<EighResult<T>>;

    // -----------------------------------------------------------------------
    // QR decomposition
    // -----------------------------------------------------------------------

    /// Thin QR decomposition: mat = Q * R.
    ///
    /// Returns `(Q, R)` where Q has orthonormal columns (dimensions m x k where
    /// k = min(m,n)) and R is upper-triangular (dimensions k x n).
    ///
    /// Used in MPS gauge fixing and MPO compression preprocessing.
    ///
    /// **faer backend note:** Maps to `Mat::qr()` high-level API (faer 0.19).
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `mat` has zero rows or columns.
    fn qr(
        &self,
        mat: &MatRef<T>,
    ) -> LinAlgResult<QrResult<T>>;

    // -----------------------------------------------------------------------
    // Tikhonov-regularized pseudo-inverse (TDVP gauge restoration)
    // -----------------------------------------------------------------------

    /// Compute the Tikhonov-regularized pseudo-inverse of a matrix given its
    /// pre-computed SVD factors.
    ///
    /// For each singular value `s_i`, computes `s_i / (s_i^2 + delta^2)` instead
    /// of the naive `1 / s_i`. This prevents NaN when singular values approach
    /// machine zero during TDVP backward bond evolution (S8.1.1 of architecture doc).
    ///
    /// The reconstruction is: V * diag(s_i / (s_i^2 + delta^2)) * U^dagger
    ///
    /// **Tikhonov regularization behavior:**
    /// - When `s_i >> delta`: `s_i / (s_i^2 + delta^2) ~= 1/s_i` (accurate inverse)
    /// - When `s_i -> 0`: `s_i / (s_i^2 + delta^2) -> s_i/delta^2 -> 0` (safe, no NaN)
    ///
    /// **Object-safety note:** This method has a `where Self: Sized` bound because
    /// `construct_regularized_inverse` takes `&dyn LinAlgBackend<T>`, introducing
    /// hidden dynamic dispatch. This method cannot be called through a trait object.
    ///
    /// # Parameters
    ///
    /// - `s_values`: singular values (positive, as returned by `svd_truncated`)
    /// - `u`: left singular vectors, columns are U's columns (m x k)
    /// - `vt`: right singular vectors transposed (k x n)
    /// - `delta`: Tikhonov regularization parameter delta. Typical range: 1e-12 to 1e-8.
    ///   Must be positive. For pure inversion (no regularization), pass `T::Real::EPSILON`.
    ///
    /// # Returns
    ///
    /// The regularized pseudo-inverse as an owned `DenseTensor<'static, T>` with shape (n, m).
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `delta <= 0.0` or if the shapes of `u`, `vt`,
    /// and `s_values` are mutually inconsistent.
    fn regularized_svd_inverse(
        &self,
        s_values: &[T::Real],
        u: &DenseTensor<'static, T>,
        vt: &DenseTensor<'static, T>,
        delta: T::Real,
    ) -> DenseTensor<'static, T>
    where
        Self: Sized,
    {
        debug_assert!(
            delta > <T::Real as num_traits::Zero>::zero(),
            "regularized_svd_inverse: delta must be positive, got {:?}",
            delta
        );
        let delta_sq = delta * delta;
        let inv_s: Vec<T::Real> = s_values
            .iter()
            .map(|&s| s / (s * s + delta_sq))
            .collect();
        construct_regularized_inverse(u, &inv_s, vt)
    }
}
```

### 3.3 Default Method Implementations

`regularized_svd_inverse` and `svd_truncated` (the algorithm-selecting wrapper) have provided default implementations in the trait body. All other methods must be implemented by each backend struct. The default for `svd_truncated` delegates to `svd_truncated_gesdd` with `svd_truncated_gesvd` fallback, adding the debug residual check with the following guards:

1. **Full-rank guard:** The residual check only runs when `result.rank == min(m, n)`. If the result is truncated, reconstruction error is expected and the check is skipped.
2. **Near-zero norm guard:** If `||A||_F < 1e-30`, the check is skipped to avoid division by zero on near-zero matrices.
3. **`NumCast` threshold:** The `1e-10` threshold is converted via `num_traits::NumCast` to support `f32`, `f64`, and other scalar types uniformly.

`regularized_svd_inverse` has a `where Self: Sized` bound, making it unavailable through `dyn LinAlgBackend<T>` trait objects.

---

## 4. The `SparseLinAlgBackend<T, Q>` Trait

### 4.1 Definition

```rust
/// Object-safe sparse backend, parameterized over both scalar and quantum number.
///
/// Extends `LinAlgBackend<T>` with operations that exploit block-sparse structure:
/// sparse matrix-vector multiply (`spmv`) and block-sparse GEMM with LPT scheduling.
///
/// `Box<dyn SparseLinAlgBackend<f64, U1>>` is a valid Rust trait object.
pub trait SparseLinAlgBackend<T: Scalar, Q: BitPackable>: LinAlgBackend<T> {
    /// Block-sparse matrix-vector multiply: y = A * x.
    ///
    /// Each non-zero block of `a` multiplies the corresponding sub-vector of `x`
    /// and accumulates into `y`. The sector structure of `a` determines which
    /// sub-vectors are touched.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `x.len() != a.total_col_dim()` or
    /// `y.len() != a.total_row_dim()`.
    fn spmv(
        &self,
        a: &BlockSparseTensor<T, Q>,
        x: &[T],
        y: &mut [T],
    );

    /// Block-sparse GEMM with LPT scheduling and automatic threading regime selection.
    ///
    /// Computes the block-sparse matrix product `a x b` using a three-phase strategy:
    ///
    /// **Phase 1 — Task Generation:** For each pair of compatible sectors
    /// (key_a from `a`, key_b from `b`), compute the fusion rule to determine
    /// if their product contributes to an output sector. If yes, generate a
    /// `SectorGemmTask` record containing references to the two input blocks and
    /// the estimated FLOP count (M x N x K).
    ///
    /// **Phase 2 — LPT Scheduling:** Sort the task list by descending FLOP cost.
    /// This is the Longest Processing Time heuristic: dispatching the heaviest
    /// tasks first to Rayon's work-stealing scheduler minimizes load imbalance
    /// caused by the binomial sector-size distribution typical of Abelian DMRG.
    /// **Current state:** LPT sorting is implemented but Rayon dispatch is deferred;
    /// execution is sequential.
    ///
    /// **Same-output-sector accumulation:** Multiple input pairs may map to the
    /// same output sector key. The current implementation uses linear scan to detect
    /// and accumulate these contributions. For large sector counts, a `HashMap`-based
    /// approach is recommended.
    ///
    /// **Phase 3 — Structural Restoration:** After execution, re-sort
    /// results by `PackedSectorKey` to restore the binary-search invariant that
    /// `BlockSparseTensor` requires. This sort is on output indices only and
    /// is dominated by Phase 2 compute time.
    ///
    /// **Threading regime:** automatically selected by `ThreadingRegime::select`:
    /// - Fat-sector mode (few large sectors): BLAS uses full machine thread pool;
    ///   Rayon disabled.
    /// - Fragmented-sector mode (many small sectors): BLAS is single-threaded;
    ///   Rayon distributes sector GEMMs across all cores.
    ///
    /// # Returns
    ///
    /// A new `BlockSparseTensor<T, Q>` containing the product, with flux
    /// `a.flux().fuse(b.flux())`.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the contracted leg dimensions of `a` and `b`
    /// are incompatible.
    fn block_gemm(
        &self,
        a: &BlockSparseTensor<T, Q>,
        b: &BlockSparseTensor<T, Q>,
    ) -> BlockSparseTensor<T, Q>;
}
```

---

## 5. Return Types

### 5.1 `SvdResult<T>`

```rust
/// Result of a truncated SVD: A ~= U * diag(singular_values) * Vt.
///
/// Only the first `rank` singular values are retained, where `rank` is
/// determined by `max_rank` and `cutoff` parameters passed to `svd_truncated`.
///
/// All `DenseTensor` fields use `'static` lifetime because SVD results are
/// always fully owned data (no borrows from the input matrix).
pub struct SvdResult<T: Scalar> {
    /// Left singular vectors. Shape: (m, rank). Columns are orthonormal.
    pub u: DenseTensor<'static, T>,
    /// Singular values in descending order. Length: rank.
    /// All values are positive real.
    ///
    /// **Implementation note:** faer's `thin_svd()` returns singular values in
    /// ascending order. The implementation re-sorts to descending order to match
    /// LAPACK convention. This re-sorting also requires corresponding reordering
    /// of columns in U and rows in Vt.
    pub singular_values: Vec<T::Real>,
    /// Right singular vectors (transposed). Shape: (rank, n). Rows are orthonormal.
    pub vt: DenseTensor<'static, T>,
    /// The retained rank (<= max_rank, <= min(m, n)).
    pub rank: usize,
    /// Truncation error: sum of squares of discarded singular values.
    /// Defined as: sum_{i > rank} sigma_i^2.
    /// Used for entanglement entropy and truncation error reporting.
    pub truncation_error: T::Real,
}
```

### 5.2 `EighResult<T>`

```rust
/// Result of a dense symmetric/Hermitian eigendecomposition.
///
/// All `DenseTensor` fields use `'static` lifetime because eigendecomposition
/// results are always fully owned data.
pub struct EighResult<T: Scalar> {
    /// Eigenvalues in ascending order. Length: k (the requested count).
    pub eigenvalues: Vec<T::Real>,
    /// Eigenvectors as columns. Shape: (n, k).
    pub eigenvectors: DenseTensor<'static, T>,
}
```

### 5.3 `QrResult<T>`

```rust
/// Result of a thin QR decomposition: mat = Q * R.
///
/// All `DenseTensor` fields use `'static` lifetime because QR results are
/// always fully owned data.
pub struct QrResult<T: Scalar> {
    /// Q factor: orthogonal/unitary matrix. Shape: (m, k) where k = min(m, n).
    pub q: DenseTensor<'static, T>,
    /// R factor: upper-triangular matrix. Shape: (k, n).
    pub r: DenseTensor<'static, T>,
}
```

### 5.4 `SvdConvergenceError`

```rust
/// Signals that an SVD algorithm (gesdd or gesvd) failed to converge.
/// Used internally by `svd_truncated` to trigger the fallback path.
#[derive(Debug, Clone, Copy)]
pub struct SvdConvergenceError;

impl std::fmt::Display for SvdConvergenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SVD algorithm failed to converge")
    }
}

impl std::error::Error for SvdConvergenceError {}
```

---

## 6. Faer Conversion Layer

### 6.1 Overview

Converting between `tk-core` tensor types and faer matrix types requires careful handling of lifetimes and unsafe code. The conversion layer is in `src/faer_convert.rs`.

### 6.2 `to_faer_mat_ref()` — Unsafe Zero-Copy View

```rust
/// Convert a tk-core MatRef into a faer MatRef (zero-copy, shared view).
///
/// # Safety
///
/// Uses `faer::mat::from_raw_parts` which is unsafe. The caller must ensure
/// the source data outlives the returned faer view. The raw pointer is derived
/// from the MatRef's data slice.
pub(crate) fn to_faer_mat_ref<'a, T: Scalar>(
    mat: &'a MatRef<'_, T>,
) -> faer::MatRef<'a, T> {
    unsafe {
        faer::mat::from_raw_parts::<T>(
            mat.data.as_ptr(), mat.rows, mat.cols, mat.row_stride, mat.col_stride,
        )
    }
}
```

### 6.3 `faer_mat_mut!` Macro — Double-Lifetime Workaround

The borrow checker cannot express a mutable faer view with both the data lifetime and the view lifetime. A macro is used to work around this:

```rust
/// Create a mutable faer MatMut from a tk-core MatMut.
///
/// This macro exists because Rust's borrow checker cannot express the
/// double-lifetime relationship needed for `faer::MatMut` construction
/// from a mutable reference in a single function signature. The macro
/// inlines the unsafe construction at the call site.
macro_rules! faer_mat_mut {
    ($mat:expr) => {
        unsafe {
            faer::mat::from_raw_parts_mut::<_>(
                $mat.data.as_mut_ptr(),
                $mat.rows,
                $mat.cols,
                $mat.row_stride,
                $mat.col_stride,
            )
        }
    };
}
```

### 6.4 `tk_mat_to_faer_owned()` — O(m x n) Copy

```rust
/// Convert a tk-core MatRef into an owned faer Mat by copying all elements.
///
/// This performs an O(m*n) copy. Used when a faer function requires an owned
/// Mat (e.g., for decompositions that consume the input). The copy overhead
/// is acceptable because the decomposition itself is O(m*n*min(m,n)) or higher.
pub(crate) fn tk_mat_to_faer_owned<T: Scalar>(
    mat: &MatRef<'_, T>,
) -> faer::Mat<T>;
```

---

## 7. Threading Regime

### 7.1 `ThreadingRegime`

```rust
/// Adaptive threading strategy for block-sparse operations.
///
/// Mixing Rayon's work-stealing scheduler with multithreaded BLAS backends
/// creates thread oversubscription. Two disjoint regimes are used:
///
/// - **FatSectors**: Few large sectors (bond dimension D > ~500, few sectors).
///   Use the full machine thread pool per BLAS call. Rayon disabled.
///
/// - **FragmentedSectors**: Many small sectors (many symmetry sectors, small D).
///   Force BLAS to single-threaded mode. Rayon distributes independent sector
///   GEMMs in parallel. LPT pre-sorting ensures balanced load.
///
/// **Current state:** The implementation uses a simple binary heuristic (see
/// `select()`), not the partitioned scheduler from the architecture document.
/// This is appropriate for Phase 1-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadingRegime {
    /// Use multithreaded BLAS for each operation; do not use Rayon over sectors.
    FatSectors {
        /// Number of threads to grant to the BLAS backend.
        blas_threads: usize,
    },
    /// Use single-threaded BLAS per sector; parallelize over sectors with Rayon.
    FragmentedSectors {
        /// Number of Rayon worker threads.
        rayon_threads: usize,
    },
}

impl ThreadingRegime {
    /// Select the appropriate regime for a given block-sparse tensor and core count.
    ///
    /// **Heuristic:** If the maximum sector dimension exceeds 500 and the number of
    /// sectors is less than `n_cores`, use FatSectors (BLAS can fill all cores with
    /// one sector). Otherwise, use FragmentedSectors.
    ///
    /// This threshold (500) is a conservative default. Profiling at D=1000 on target
    /// hardware should calibrate it per deployment.
    ///
    /// **Implementation note:** `max_sector_dim_on_any_leg()` does not exist in
    /// `tk-symmetry`. It is implemented locally in `threading.rs` as a helper
    /// that iterates over all legs and sectors to find the maximum dimension.
    pub fn select<T: Scalar, Q: BitPackable>(
        tensor: &BlockSparseTensor<T, Q>,
        n_cores: usize,
    ) -> Self {
        let max_dim = max_sector_dim_on_any_leg(tensor);
        if max_dim > 500 && tensor.n_sectors() < n_cores {
            ThreadingRegime::FatSectors { blas_threads: n_cores }
        } else {
            ThreadingRegime::FragmentedSectors { rayon_threads: n_cores }
        }
    }
}

/// Compute the maximum sector dimension across all legs of a block-sparse tensor.
///
/// This function is implemented locally in threading.rs because it does not
/// exist in tk-symmetry. If tk-symmetry later provides this as a method on
/// BlockSparseTensor, this local implementation should be removed.
pub(crate) fn max_sector_dim_on_any_leg<T: Scalar, Q: BitPackable>(
    tensor: &BlockSparseTensor<T, Q>,
) -> usize;
```

---

## 8. LPT-Scheduled Block-Sparse Dispatch

### 8.1 `SectorGemmTask`

```rust
/// A single dense GEMM within a block-sparse contraction.
///
/// Represents one element of the task queue generated in Phase 1 of `block_gemm`.
/// Each task is fully independent: it reads from two input blocks (borrowed from
/// the input tensors, which are immutable for the duration of Phase 2 execution)
/// and writes to an output block identified by `out_key`.
struct SectorGemmTask<'a, T: Scalar> {
    /// Output sector key (determines where the result accumulates).
    out_key: PackedSectorKey,
    /// Immutable reference to the left input block.
    block_a: &'a DenseTensor<T>,
    /// Immutable reference to the right input block.
    block_b: &'a DenseTensor<T>,
    /// Estimated FLOP count: rows(A) x cols(B) x cols(A).
    /// Used for LPT scheduling (sort descending before dispatch).
    flops: usize,
}
```

### 8.2 Fusion Rule

```rust
/// Compute the output sector key for a pair of input sector keys.
///
/// For Abelian symmetries, each input sector pair produces at most one output sector.
/// This function encodes the Abelian fusion rule: the output quantum number is the
/// fused (summed, for U(1)) combination of the input quantum numbers.
///
/// Returns `None` if the input sectors are not compatible (do not satisfy the
/// output tensor's flux rule), signaling that no GEMM task should be generated
/// for this pair.
///
/// **Non-Abelian note:** For SU(2) symmetry (behind `su2-symmetry` feature flag),
/// the fusion rule is one-to-many: j1 (x) j2 = |j1-j2| (+) ... (+) (j1+j2).
/// This function returns `Option<PackedSectorKey>` (single output) and handles
/// only the Abelian case. The SU(2) path must fan out to a `Vec<SectorGemmTask>`
/// per input pair (see S19 Open Questions).
///
/// **Implementation note:** The `_indices_a` and `_indices_b` parameters are
/// currently unused but reserved for future extensions (e.g., non-Abelian
/// Clebsch-Gordan coefficient lookup).
fn compute_fusion_rule<Q: BitPackable>(
    key_a: PackedSectorKey,
    key_b: PackedSectorKey,
    _indices_a: &[QIndex<Q>],
    _indices_b: &[QIndex<Q>],
    rank_a: usize,
    rank_b: usize,
    flux: &Q,
) -> Option<PackedSectorKey>;
```

### 8.3 Three-Phase `block_gemm` Algorithm

The algorithm in `SparseLinAlgBackend::block_gemm` follows three strict phases. The code sketch below documents the contract — exact implementation is in `src/tasks.rs` and the concrete `impl` blocks in `src/device/`.

```rust
// PHASE 1: Task generation
let mut tasks: Vec<SectorGemmTask<T>> = Vec::new();
for (i, key_a) in a.sector_keys.iter().enumerate() {
    for (j, key_b) in b.sector_keys.iter().enumerate() {
        if let Some(out_key) = compute_fusion_rule(
            *key_a, *key_b,
            &a.indices(), &b.indices(),  // reserved parameters
            a.rank(), b.rank(), &a.flux().fuse(b.flux())
        ) {
            let ba = &a.sector_blocks()[i];
            let bb = &b.sector_blocks()[j];
            let flops = ba.shape().dims()[0] * bb.shape().dims()[1] * ba.shape().dims()[1];
            tasks.push(SectorGemmTask { out_key, block_a: ba, block_b: bb, flops });
        }
    }
}

// PHASE 2: LPT scheduling — heaviest GEMMs dispatched first
tasks.sort_unstable_by(|x, y| y.flops.cmp(&x.flops));

// Current state: sequential execution. Rayon dispatch deferred.
let results: Vec<(PackedSectorKey, DenseTensor<'static, T>)> = tasks
    .into_iter()
    .map(|task| {
        let mut out = DenseTensor::zeros(/* shape from block dims */);
        backend.gemm(T::one(), task.block_a.as_mat_ref()?, task.block_b.as_mat_ref()?,
                     T::zero(), &mut out.as_mat_mut()?);
        (task.out_key, out)
    })
    .collect();

// Same-output-sector accumulation: linear scan for multi-input-to-same-output case.
// TODO: Replace with HashMap for large sector counts.
let mut merged: Vec<(PackedSectorKey, DenseTensor<'static, T>)> = Vec::new();
for (key, block) in results {
    if let Some((_, existing)) = merged.iter_mut().find(|(k, _)| *k == key) {
        // Accumulate: existing += block
        existing.add_assign(&block);
    } else {
        merged.push((key, block));
    }
}

// PHASE 3: Structural restoration — re-sort by key, build output BlockSparseTensor
merged.sort_unstable_by_key(|(key, _)| *key);
let (out_keys, out_blocks): (Vec<_>, Vec<_>) = merged.into_iter().unzip();
BlockSparseTensor::from_raw_parts(
    compute_output_indices(&a.indices(), &b.indices()),
    out_keys,
    out_blocks,
    a.flux().fuse(b.flux()),
)
```

**Thread safety note:** When Rayon dispatch is enabled (future), Phase 2 in FragmentedSectors mode will use `into_par_iter()` over owned tasks. Each task holds immutable references to input blocks (shared read) and produces a new owned output block (no aliasing). There are no writes to shared state. This satisfies Rayon's data-race freedom requirements without `UnsafeCell` or explicit locking.

---

## 9. Concrete Backend Implementations

### 9.1 `DeviceFaer`

```rust
/// Pure-Rust dense backend using the `faer` crate (v0.19).
///
/// Default backend when `backend-faer` feature is active (enabled by default).
///
/// **API level:** Uses faer's high-level API exclusively (`Mat::thin_svd()`,
/// `Mat::qr()`, `Mat::selfadjoint_eigendecomposition()`). The low-level faer
/// API is fragile across minor versions and is not used. The high-level API is
/// sufficient for all current needs.
///
/// **Lazy conjugation:** Calling `.conjugate()` on a `faer::MatRef` flips one
/// bit in the view; the SIMD FMA micro-kernels handle negation of imaginary
/// parts during computation, never touching the data buffer.
///
/// **Scalar support:** Currently only `f64` is implemented. `f32`, `Complex<f32>`,
/// and `Complex<f64>` implementations are planned via `macro_rules!` specialization.
#[cfg(feature = "backend-faer")]
pub struct DeviceFaer;
```

#### 9.1.1 `gemm` Implementation (Conjugation Path)

```rust
#[cfg(feature = "backend-faer")]
impl LinAlgBackend<f64> for DeviceFaer {
    fn gemm(
        &self,
        alpha: f64,
        a: &MatRef<'_, f64>,
        b: &MatRef<'_, f64>,
        beta: f64,
        c: &mut MatMut<'_, f64>,
    ) {
        // Construct faer views from tk-core MatRef strides.
        // Uses unsafe from_raw_parts — see §6.2 for safety discussion.
        let faer_a = to_faer_mat_ref(a);
        let faer_b = to_faer_mat_ref(b);
        let mut faer_c = faer_mat_mut!(c);

        // Zero-copy conjugation: faer's lazy conjugation view flips a single bit.
        // For real f64, is_conjugated is always false (T::is_real() == true);
        // this branch compiles away via const propagation.
        let a_op: faer::MatRef<'_, f64> = if a.is_conjugated {
            faer_a.conjugate()
        } else {
            faer_a.as_ref()
        };
        let b_op: faer::MatRef<'_, f64> = if b.is_conjugated {
            faer_b.conjugate()
        } else {
            faer_b.as_ref()
        };

        faer::linalg::matmul::matmul(
            faer_c.as_mut(),
            a_op,
            b_op,
            Some(beta),
            alpha,
            faer::Parallelism::Rayon(0), // 0 = use Rayon's current thread count
        );
    }

    // ... svd_truncated_gesdd, svd_truncated_gesvd, eigh_lowest, qr ...
}

// f32, C32, C64 implementations are planned but not yet implemented.
// Macro-based specialization is recommended:
// impl LinAlgBackend<f32> for DeviceFaer { /* macro-generated, not yet done */ }
// impl LinAlgBackend<C32> for DeviceFaer { /* macro-generated, not yet done */ }
// impl LinAlgBackend<C64> for DeviceFaer { /* macro-generated, not yet done */ }
```

#### 9.1.2 SVD Implementation Notes

`DeviceFaer::svd_truncated_gesdd` maps to `faer::Mat::thin_svd()` — the high-level API in faer 0.19. The low-level `faer::linalg::svd::compute_svd` API is not used because it is fragile across minor faer versions.

**Critical: faer singular value ordering.** faer's `thin_svd()` returns singular values in **ascending** order. The implementation must re-sort to **descending** order to match LAPACK convention and the `SvdResult` contract. This re-sorting requires corresponding reordering of:
- Columns of U (swap column i with column n-1-i)
- Rows of Vt (swap row i with row n-1-i)

Failing to re-sort is a silent correctness bug: truncation by `max_rank` would discard the *largest* singular values instead of the smallest.

The `cutoff` parameter is applied relative to `sigma[0]` (the largest singular value after re-sorting): a singular value `sigma[i]` is retained iff `sigma[i] >= cutoff * sigma[0]`. If `sigma[0]` is zero (zero matrix), no values are retained.

`truncation_error` is computed as the sum of squares of all discarded singular values: `sum_{i > rank} sigma_i^2`. This is the standard DMRG truncation error metric.

**faer conversion overhead:** The input `MatRef` is converted to an owned `faer::Mat` via `tk_mat_to_faer_owned()`, which performs an O(m*n) copy. This is acceptable because the SVD itself is O(m*n*min(m,n)).

### 9.2 `DeviceOxiblas`

```rust
/// Pure-Rust backend using the `oxiblas` crate for sparse and SIMD operations.
///
/// **Current state:** Not yet implemented. Stub only.
///
/// Planned: 9 sparse matrix formats (BSR, CSR, CSC, COO, DIA, ELL, HYB, BCSR, BCSC),
/// explicit SIMD (AVX-512, AVX2, NEON) for element-wise operations, and
/// `f128` extended-precision arithmetic when `backend-oxiblas` feature is active.
///
/// Used as the sparse backend in `DeviceAPI<DeviceFaer, DeviceOxiblas>`.
/// Dense SVD/EVD/QR are delegated to `DeviceFaer`.
#[cfg(feature = "backend-oxiblas")]
pub struct DeviceOxiblas;
```

### 9.3 `DeviceMKL`

```rust
/// FFI-based backend using Intel Math Kernel Library.
///
/// Active only when `backend-mkl` feature is enabled.
/// Links against MKL via `intel-mkl-sys` or equivalent.
/// Provides vendor-optimized GEMM and LAPACK SVD (`LAPACKE_dgesdd`,
/// `LAPACKE_zgesdd`) with `CblasConjTrans` and `CblasNoTrans` dispatch.
///
/// **Note:** With MKL/OpenBLAS, the `gesdd`-to-`gesvd` fallback in
/// `svd_truncated` is meaningful because these libraries expose separate
/// `LAPACKE_dgesdd` and `LAPACKE_dgesvd` routines with different algorithms.
///
/// # Thread Safety
///
/// MKL uses a global thread pool (Intel TBB). Accessing MKL concurrently from
/// Rayon is safe only in FragmentedSectors mode with MKL single-threaded
/// (`MKL_NUM_THREADS=1`). The threading regime selection (S7) handles this.
#[cfg(feature = "backend-mkl")]
pub struct DeviceMKL {
    /// Thread count assigned to MKL's internal pool.
    /// Set to 1 in FragmentedSectors mode; set to n_cores in FatSectors mode.
    thread_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}
```

#### 9.3.1 BLAS Layout Resolution

MKL's `cblas_zgemm` requires at least one unit stride per matrix. The `resolve_blas_layout` helper maps `MatRef` strides and `is_conjugated` to the correct `CBLAS_TRANSPOSE` enum:

```rust
#[cfg(feature = "backend-mkl")]
impl DeviceMKL {
    /// Map a MatRef<f64> to (CBLAS_TRANSPOSE, leading_dimension).
    ///
    /// BLAS requires at least one unit stride (either row-major or col-major).
    /// Arbitrary-stride matrices (e.g., after a non-contiguous permute) cannot
    /// be passed to BLAS directly. Use DeviceFaer for those cases.
    ///
    /// # Panics
    ///
    /// Panics if neither `row_stride == 1` nor `col_stride == 1`.
    #[inline(always)]
    fn resolve_blas_layout(mat: &MatRef<'_, f64>) -> (CBLAS_TRANSPOSE, i32) {
        if mat.col_stride == 1 {
            // Row-major (C layout): leading dimension is row_stride (= cols).
            let trans = match mat.is_conjugated {
                false => CBLAS_TRANSPOSE::CblasNoTrans,
                true  => CBLAS_TRANSPOSE::CblasConjNoTrans,
            };
            (trans, mat.row_stride as i32)
        } else if mat.row_stride == 1 {
            // Column-major (Fortran layout): leading dimension is col_stride (= rows).
            let trans = match mat.is_conjugated {
                false => CBLAS_TRANSPOSE::CblasTrans,
                true  => CBLAS_TRANSPOSE::CblasConjTrans,
            };
            (trans, mat.col_stride as i32)
        } else {
            panic!(
                "DeviceMKL::resolve_blas_layout: matrix has non-unit strides \
                 (row_stride={}, col_stride={}). BLAS requires at least one unit stride. \
                 Use DeviceFaer for strided views.",
                mat.row_stride, mat.col_stride
            );
        }
    }
}
```

### 9.4 `DeviceOpenBLAS`

```rust
/// FFI-based backend using OpenBLAS.
///
/// Active only when `backend-openblas` feature is enabled.
/// Cannot coexist with `backend-mkl` (compile_error! in build.rs).
/// API is structurally identical to `DeviceMKL` but links against OpenBLAS
/// symbols (`cblas_dgemm`, `cblas_zgemm`, `LAPACKE_dgesdd`, etc.).
///
/// Conjugation dispatch follows the same `resolve_blas_layout` pattern
/// as `DeviceMKL` (S9.3.1), substituting `openblas_sys` types.
#[cfg(feature = "backend-openblas")]
pub struct DeviceOpenBLAS;
```

### 9.5 `DeviceCuda`

```rust
/// GPU-accelerated backend using cuBLAS and cuSOLVER.
///
/// Active only when `backend-cuda` feature is enabled.
/// Wraps CUDA handles managed via the `cudarc` crate.
///
/// # Memory requirements
///
/// All matrix operands must reside in device memory (GPU VRAM) before
/// calling GEMM or SVD. Transfers from host to device use DMA-capable
/// pinned memory arenas managed by `PinnedMemoryTracker` (tk-core).
///
/// # Asynchronous execution
///
/// cuBLAS and cuSOLVER calls are asynchronous. The `ContractionExecutor`
/// in `tk-contract` is responsible for stream-aware DAG synchronization
/// using per-node `cuda::Event` objects. This backend does NOT insert
/// global pipeline-stalling syncs (`cudaDeviceSynchronize`).
///
/// # Conjugation dispatch
///
/// cuBLAS `cublasDgemm` / `cublasZgemm` accept `CUBLAS_OP_C` for
/// conjugate-only (no-transpose) and `CUBLAS_OP_CONJ_TRANS` for
/// conjugate-transpose. These are mapped from `is_conjugated` and stride
/// layout in the same pattern as `DeviceMKL::resolve_blas_layout`.
#[cfg(feature = "backend-cuda")]
pub struct DeviceCuda {
    /// CUDA compute stream for all operations on this backend instance.
    /// Independent `DeviceCuda` instances on different streams can run
    /// concurrently on the GPU.
    pub(crate) stream: cudarc::driver::CudaStream,
    /// cuBLAS context bound to `stream`.
    pub(crate) cublas_handle: cudarc::cublas::CudaBlas,
    /// cuSOLVER dense context bound to `stream`.
    pub(crate) cusolver_handle: cudarc::cusolver::CudaSolverDn,
}
```

#### 9.5.1 GPU Performance Threshold

Below bond dimension D ~= 500, the cuBLAS kernel launch overhead (1-5 us per call) begins to exceed the compute time for the GEMM itself, negating the GPU advantage. The `DeviceAPI` composite backend supports a hybrid threshold:

```rust
/// Threshold in matrix dimension below which GEMM is routed to the CPU backend.
/// Configurable; default 500 based on empirical cuBLAS launch overhead measurements.
pub const GPU_DISPATCH_THRESHOLD: usize = 500;
```

### 9.6 `DeviceAPI<D, S>` — Composite Backend

```rust
/// Composite backend pairing a dense backend `D` with a sparse backend `S`.
///
/// `D` handles: GEMM, SVD, QR, eigh, regularized_svd_inverse.
/// `S` handles: spmv, block_gemm (when sectors are in play).
///
/// **Current state:** `DefaultDevice` is `DeviceAPI<DeviceFaer, DeviceFaer>` because
/// `DeviceOxiblas` is not yet implemented. The intended target is
/// `DeviceAPI<DeviceFaer, DeviceOxiblas>`.
pub struct DeviceAPI<D, S> {
    pub dense: D,
    pub sparse: S,
}

/// Interim default: DeviceFaer for both dense and sparse.
/// Target: DeviceAPI<DeviceFaer, DeviceOxiblas> once DeviceOxiblas is implemented.
#[cfg(feature = "backend-faer")]
pub type DefaultDevice = DeviceAPI<DeviceFaer, DeviceFaer>;

/// `DeviceAPI` delegates `LinAlgBackend<T>` to the dense component.
impl<T, D, S> LinAlgBackend<T> for DeviceAPI<D, S>
where
    T: Scalar,
    D: LinAlgBackend<T>,
    S: Send + Sync,
{
    fn gemm(&self, alpha: T, a: &MatRef<T>, b: &MatRef<T>, beta: T, c: &mut MatMut<T>) {
        self.dense.gemm(alpha, a, b, beta, c)
    }
    fn svd_truncated_gesdd(
        &self, mat: &MatRef<T>, max_rank: usize, cutoff: T::Real,
    ) -> Result<SvdResult<T>, SvdConvergenceError> {
        self.dense.svd_truncated_gesdd(mat, max_rank, cutoff)
    }
    fn svd_truncated_gesvd(
        &self, mat: &MatRef<T>, max_rank: usize, cutoff: T::Real,
    ) -> Result<SvdResult<T>, SvdConvergenceError> {
        self.dense.svd_truncated_gesvd(mat, max_rank, cutoff)
    }
    fn eigh_lowest(&self, mat: &MatRef<T>, k: usize) -> LinAlgResult<EighResult<T>> {
        self.dense.eigh_lowest(mat, k)
    }
    fn qr(&self, mat: &MatRef<T>) -> LinAlgResult<QrResult<T>> {
        self.dense.qr(mat)
    }
}

/// `DeviceAPI` delegates `SparseLinAlgBackend<T, Q>` to the sparse component.
impl<T, Q, D, S> SparseLinAlgBackend<T, Q> for DeviceAPI<D, S>
where
    T: Scalar,
    Q: BitPackable,
    D: LinAlgBackend<T>,
    S: SparseLinAlgBackend<T, Q>,
{
    fn spmv(&self, a: &BlockSparseTensor<T, Q>, x: &[T], y: &mut [T]) {
        self.sparse.spmv(a, x, y)
    }
    fn block_gemm(
        &self,
        a: &BlockSparseTensor<T, Q>,
        b: &BlockSparseTensor<T, Q>,
    ) -> BlockSparseTensor<T, Q> {
        self.sparse.block_gemm(a, b)
    }
}
```

---

## 10. Error Handling

### 10.1 `LinAlgError`

```rust
/// Errors produced by `tk-linalg` operations.
#[derive(Debug, thiserror::Error)]
pub enum LinAlgError {
    /// Both `gesdd` and `gesvd` failed to converge. Extremely rare;
    /// indicates a pathologically ill-conditioned input matrix.
    #[error("SVD failed to converge (both gesdd and gesvd diverged)")]
    SvdConvergence,

    /// The requested number of eigenvalues/vectors exceeds the matrix size.
    #[error("requested k={k} eigenvalues but matrix has only {n} rows")]
    EighKTooLarge { k: usize, n: usize },

    /// A matrix dimension is incompatible with the requested operation.
    #[error("dimension mismatch in {op}: expected {expected:?}, got {got:?}")]
    DimensionMismatch {
        op: &'static str,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// A non-unit stride was passed to a BLAS backend that requires contiguous layout.
    /// The caller must materialize the tensor via `TensorCow::into_owned` first.
    #[error("non-unit stride passed to BLAS backend: use into_owned() first")]
    NonContiguousForBlas,

    /// CUDA operation returned an error (backend-cuda only).
    #[cfg(feature = "backend-cuda")]
    #[error("CUDA error in {op}: {detail}")]
    CudaError { op: &'static str, detail: String },

    /// Wraps errors from the tk-core layer.
    #[error(transparent)]
    Core(#[from] tk_core::TkError),
}

pub type LinAlgResult<T> = Result<T, LinAlgError>;
```

### 10.2 Error Propagation Strategy

- `svd_truncated` returns `LinAlgResult<SvdResult<T>>`. The `SvdConvergence` error is only returned when both `gesdd` and `gesvd` fail. In practice this should not occur with valid floating-point inputs. With the faer backend, since both map to the same `thin_svd()` call, a failure on one means the fallback will also fail.
- `gemm` does not return `Result`; it panics in debug mode on dimension mismatch and is infallible in release mode (the caller is responsible for passing correct dimensions, enforced by `BlockSparseTensor` invariants).
- CUDA errors are surfaced as `LinAlgError::CudaError` and propagated to the DMRG sweep engine, which reports them in `SolverError`.
- `DimensionMismatch` is the primary error during integration testing; it should not appear in production code where `BlockSparseTensor` invariants are maintained.

---

## 11. Public API Surface (`lib.rs`)

```rust
// tk-linalg/src/lib.rs

pub mod traits;
pub mod results;
pub mod threading;
pub mod tasks;
pub mod device;
pub mod error;

// Flat re-exports:
pub use traits::{LinAlgBackend, SparseLinAlgBackend};
pub use results::{SvdResult, EighResult, QrResult, SvdConvergenceError};
pub use threading::ThreadingRegime;
pub use error::{LinAlgError, LinAlgResult};

// Device re-exports (conditional on feature flags):
#[cfg(feature = "backend-faer")]
pub use device::faer::DeviceFaer;

#[cfg(feature = "backend-oxiblas")]
pub use device::oxiblas::DeviceOxiblas;

#[cfg(feature = "backend-faer")]
pub use device::{DeviceAPI, DefaultDevice};

#[cfg(feature = "backend-mkl")]
pub use device::mkl::DeviceMKL;

#[cfg(feature = "backend-openblas")]
pub use device::openblas::DeviceOpenBLAS;

#[cfg(feature = "backend-cuda")]
pub use device::cuda::DeviceCuda;
```

---

## 12. Feature Flags

| Flag | Effect in `tk-linalg` |
|:-----|:----------------------|
| `backend-faer` | Enables `DeviceFaer`; pure-Rust SVD/GEMM/QR via faer 0.19 high-level API; default on |
| `backend-oxiblas` | Enables `DeviceOxiblas` (stub); sparse formats, SIMD, f128; not yet implemented |
| `backend-mkl` | Enables `DeviceMKL`; Intel MKL FFI; mutually exclusive with `backend-openblas`; not yet implemented |
| `backend-openblas` | Enables `DeviceOpenBLAS`; OpenBLAS FFI; mutually exclusive with `backend-mkl`; not yet implemented |
| `backend-cuda` | Enables `DeviceCuda`; cuBLAS + cuSOLVER via `cudarc`; requires CUDA toolkit; not yet implemented |
| `parallel` | Enables Rayon for block-sparse sector dispatch in FragmentedSectors mode; LPT sorting implemented but Rayon dispatch deferred |
| `su2-symmetry` | Propagates SU(2) feature flag from `tk-symmetry`; see S12.1 |

### 12.1 `su2-symmetry` Feature in `tk-linalg`

When `su2-symmetry` is active, `compute_fusion_rule` must be generalized. The current Abelian implementation returns `Option<PackedSectorKey>` (one-to-one). For SU(2), j1 (x) j2 produces multiple output irreps. The `SectorGemmTask` generation loop must fan out to `Vec<SectorGemmTask>` per input pair, each weighted by the corresponding Clebsch-Gordan coefficient. This is a structural change to `tasks.rs`, scoped behind the `su2-symmetry` feature flag and does not affect the Abelian code path (see S19 Open Questions).

---

## 13. Build-Level Concerns

```rust
// tk-linalg/build.rs

fn main() {
    // FFI BLAS backends both expose global symbols (dgemm_, dsyev_, etc.).
    // Enabling both causes linker collisions that produce cryptic symbol errors.
    // A compile-time check is vastly more ergonomic than a link-time error.
    #[cfg(all(feature = "backend-mkl", feature = "backend-openblas"))]
    compile_error!(
        "Features `backend-mkl` and `backend-openblas` are mutually exclusive. \
         Both expose global BLAS/LAPACK symbols and will cause linker collisions \
         (e.g., duplicate symbol `dgemm_`). Enable only one FFI-based backend. \
         If you need both on the same system, use separate workspaces."
    );
}
```

### Monomorphization Budget

#### 13.1 The Problem

The compute stack is generic over `<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>`. With 4 scalar types x 3 quantum-number types x 4 backend types, naive full monomorphization produces up to 48 copies of any generic function that reaches through all three parameters.

#### 13.2 Mitigation Strategy

`tk-linalg` applies two strategies:

**1. Macro-generated scalar implementations.** Each backend struct (DeviceFaer, DeviceOxiblas, etc.) implements `LinAlgBackend<T>` for each of the four scalar types. Rather than writing 4 separate `impl` blocks with identical bodies differing only in type, a `macro_rules!` template generates all four. This does not reduce monomorphization but keeps the source manageable. **Current state:** Only `f64` is implemented; the macro template is designed but the other three types are not yet generated.

**2. Feature-gated type combinations.** The `DefaultDevice` type alias compiles only when `backend-faer` is active. Users requiring only `DeviceFaer` (e.g., for debugging) use it directly. The `tk-python` crate's `DmftLoopVariant` enum explicitly enumerates only the user-facing combinations, preventing the compiler from generating all 48 variants.

**3. `dyn`-eligible at sweep level.** Since `LinAlgBackend<T>` is object-safe (except `regularized_svd_inverse`), the sweep scheduler in `tk-dmrg` can accept `Box<dyn LinAlgBackend<f64>>` if compile times become problematic. The inner GEMM loops remain statically dispatched.

**4. CI compile-time monitoring.** A CI job tracks per-crate compile times in release mode. If `tk-linalg` exceeds 60 seconds, `cargo-llvm-lines` is run to identify the largest generic expansions. The threshold for remediation action is 60 seconds single-crate compile in release.

#### 13.3 Common Combination

```rust
// The combination compiled by default (the vast majority of users):
#[cfg(feature = "backend-faer")]
pub type DefaultEngine = DMRGEngine<f64, U1, DefaultDevice>;
```

---

## 14. Internal Helpers

These are `pub(crate)` functions in `src/traits.rs` or `src/device/mod.rs` used by backend implementations.

```rust
/// Compute the Frobenius norm of a matrix: sqrt(sum |a_ij|^2).
/// Used in the SVD residual validation debug_assert.
pub(crate) fn frobenius_norm<T: Scalar>(mat: &MatRef<T>) -> T::Real;

/// Compute the SVD reconstruction residual: ||A - U*diag(sigma)*Vt||_F.
/// Called inside the debug_assert in svd_truncated. Compiled out in release.
///
/// Only called when result.rank == min(m, n) (full rank, not truncated).
/// Uses NumCast for threshold conversion across scalar types.
#[cfg(debug_assertions)]
pub(crate) fn svd_reconstruction_error<T: Scalar>(
    mat: &MatRef<T>,
    result: &SvdResult<T>,
) -> T::Real;

/// Reconstruct the regularized pseudo-inverse: V * diag(inv_s) * U^dagger.
/// Used by the default regularized_svd_inverse implementation.
///
/// Takes `&dyn LinAlgBackend<T>` for the GEMM call — this is hidden dynamic
/// dispatch. The calling method (`regularized_svd_inverse`) has a
/// `where Self: Sized` bound to compensate.
pub(crate) fn construct_regularized_inverse<T: Scalar>(
    u: &DenseTensor<'static, T>,
    inv_s: &[T::Real],
    vt: &DenseTensor<'static, T>,
) -> DenseTensor<'static, T>;

/// Compute the output QIndex structure for a block-sparse GEMM.
/// Given the QIndex of the uncontracted legs of `a` (outer legs) and
/// the uncontracted legs of `b` (outer legs), return the QIndex list
/// for the output tensor.
pub(crate) fn compute_output_indices<Q: BitPackable>(
    indices_a: &[QIndex<Q>],
    indices_b: &[QIndex<Q>],
) -> Vec<QIndex<Q>>;

/// Set the BLAS internal thread count for MKL/OpenBLAS FFI backends.
/// No-op for DeviceFaer (Rayon controls its own thread pool).
pub(crate) fn set_blas_num_threads(n: usize);

/// Compute the maximum sector dimension across all legs of a block-sparse tensor.
/// Implemented locally because tk-symmetry does not provide this method.
pub(crate) fn max_sector_dim_on_any_leg<T: Scalar, Q: BitPackable>(
    tensor: &BlockSparseTensor<T, Q>,
) -> usize;
```

---

## 15. Dependencies and Integration

### 15.1 Dependencies (Cargo.toml)

```toml
[dependencies]
tk-core    = { path = "../tk-core" }
tk-symmetry = { path = "../tk-symmetry" }

# Default (pure-Rust) backends
faer       = { version = "0.19", optional = true }   # feature: backend-faer
oxiblas    = { version = "0.3",  optional = true }   # feature: backend-oxiblas

# Scalar trait utilities
num-traits = { version = "0.2" }                     # NumCast for cross-type thresholds

# Data parallelism
rayon      = { version = "1.10", optional = true }   # feature: parallel

# FFI backends
intel-mkl-sys    = { version = "0.7", optional = true }  # feature: backend-mkl
openblas-src      = { version = "0.10", optional = true } # feature: backend-openblas
cblas-sys         = { version = "0.1", optional = true }  # shared by MKL + OpenBLAS
lapacke           = { version = "0.3", optional = true }  # shared by MKL + OpenBLAS

# CUDA backend
cudarc     = { version = "0.11", optional = true }   # feature: backend-cuda

# Utilities
num-cpus   = { version = "1.16" }                    # threading regime selection
log        = { version = "0.4" }                     # gesdd fallback warning
thiserror  = { version = "1.0" }

[features]
default         = ["backend-faer", "parallel"]
backend-faer    = ["faer"]
backend-oxiblas = ["oxiblas"]
backend-mkl     = ["intel-mkl-sys", "cblas-sys", "lapacke"]
backend-openblas = ["openblas-src", "cblas-sys", "lapacke"]
backend-cuda    = ["cudarc"]
parallel        = ["rayon"]
su2-symmetry    = ["tk-symmetry/su2-symmetry"]
```

**Note:** The default features no longer include `backend-oxiblas` since it is not yet implemented.

### 15.2 Downstream Consumers

| Crate | Usage |
|:------|:------|
| `tk-contract` | `LinAlgBackend::gemm` for each pairwise contraction step; `SparseLinAlgBackend::block_gemm` for block-sparse contraction |
| `tk-dmrg` | `LinAlgBackend::svd_truncated` for two-site SVD truncation; `regularized_svd_inverse` for TDVP gauge restoration; `SparseLinAlgBackend::spmv` for environment contraction |
| `tk-dmft` | `LinAlgBackend::svd_truncated` for bath discretization; inherits all `tk-dmrg` usage transitively |

---

## 16. Testing Strategy

### 16.1 Implementation Status

Test coverage is approximately 40%. 10 of 26 specified unit tests are implemented. The following categories are not yet covered:

- Complex-valued tests (`C32`, `C64`)
- Property-based tests (`proptest`)
- `block_gemm` integration tests
- Cross-backend equivalence tests

### 16.2 Implemented Unit Tests

| Test | Description |
|:-----|:------------|
| `gemm_identity_f64` | `C = 1*A*I + 0*C` equals `A` for random `A` |
| `svd_reconstruction_f64` | `||A - U*Sigma*Vt||_F / ||A||_F < 1e-12` for random 50x30 matrix |
| `svd_truncation_max_rank` | Returns exactly `max_rank` singular values when matrix rank exceeds limit |
| `svd_truncation_cutoff` | Returns fewer singular values when cutoff eliminates small values |
| `svd_truncation_error_sum` | `truncation_error == sum_{i>rank} sigma_i^2` |
| `svd_descending_order` | Singular values are returned in descending order (tests faer ascending-to-descending re-sort) |
| `regularized_inverse_large_s` | For `s >> delta`: result ~= true inverse `V*diag(1/s)*U^dagger` |
| `regularized_inverse_zero_s` | For `s = 0`: result is 0 (no NaN, no Inf) |
| `threading_regime_fat` | Large max_sector_dim > 500 + few sectors -> FatSectors |
| `threading_regime_fragmented` | Small max_sector_dim or many sectors -> FragmentedSectors |

### 16.3 Specified but Not Yet Implemented Unit Tests

| Test | Description |
|:-----|:------------|
| `gemm_conjugated_a_c64` | `C = A^dagger*B` via `is_conjugated=true` + transposed strides matches explicit conjugate + transpose + multiply |
| `gemm_conjugated_b_c64` | Same for conjugated `B` |
| `gemm_both_conjugated_c64` | Both `A` and `B` conjugated: `C = conj(A)*conj(B)` |
| `gemm_real_ignores_conjugation` | `f64` GEMM with `is_conjugated=true` produces same result as `false` |
| `svd_reconstruction_c64` | SVD reconstruction for complex matrix |
| `svd_gesdd_fallback` | Synthetically corrupt gesdd to fail (mock); verify gesvd is called |
| `svd_residual_debug_assert` | In debug build: synthetically bad SVD triggers debug_assert |
| `regularized_inverse_formula` | `s/(s^2 + delta^2)` vs analytically computed values for delta=1e-8 |
| `lpt_sort_descending_flops` | After LPT sort, tasks[0].flops >= tasks[n-1].flops |
| `block_gemm_sector_presence` | All valid output sectors present; absent sectors absent |
| `block_gemm_sector_sorted` | Output `sector_keys` are in ascending order |
| `block_gemm_equivalence_dense` | Block-sparse result matches manually assembled dense matrix product |
| `block_gemm_flux` | Output tensor flux equals `a.flux.fuse(b.flux)` |
| `spmv_correctness` | Sparse matvec matches dense reference |
| `blas_layout_col_major_no_conjugate` | `resolve_blas_layout` for col-major, no conjugation -> CblasNoTrans |
| `blas_layout_row_major_with_conjugate` | `resolve_blas_layout` for row-major, conjugated -> CblasConjTrans |
| `blas_layout_arbitrary_stride_panics` | Non-unit strides in both dimensions -> panic |
| `build_mutual_exclusivity` | (trybuild compile-fail) enabling both `backend-mkl` and `backend-openblas` -> compile_error! |

### 16.4 Property-Based Tests (Not Yet Implemented)

```rust
proptest! {
    // Bounded strategies: max dimension 32, complex values in [-5, 5] + i[-5, 5].

    #[test]
    fn gemm_associativity(
        m in 2usize..=16, n in 2usize..=16, k in 2usize..=16, l in 2usize..=16,
    ) {
        // (A*B)*C == A*(B*C) for random A(m x k), B(k x n), C(n x l)
    }

    #[test]
    fn svd_round_trip(
        m in 2usize..=20, n in 2usize..=20,
        // Values drawn from bounded range to ensure min condition number >= 1e-8
    ) {
        // ||A - U*diag(sigma)*Vt||_F / ||A||_F < 1e-10 for full-rank reconstruction
    }

    #[test]
    fn regularized_inverse_decreasing_delta(
        s in 1e-4f64..=10.0,
        delta_small in 1e-12f64..=1e-8,
        delta_large in 1e-4f64..=1.0,
    ) {
        // Smaller delta -> result closer to true inverse for s > delta
        let inv_small = s / (s * s + delta_small * delta_small);
        let inv_large = s / (s * s + delta_large * delta_large);
        prop_assert!(inv_small >= inv_large);
    }

    #[test]
    fn block_gemm_output_sectors_subset_of_valid(
        // Bounded: rank 2, sectors 2..=6, dim 1..=8
    ) {
        // Every sector in the output satisfies the flux rule
    }
}
```

### 16.5 Cross-Backend Equivalence Tests (Not Yet Implemented)

Cross-backend tests (DeviceFaer vs DeviceMKL, DeviceFaer vs DeviceCuda) must use gauge-invariant comparisons. Direct singular-vector comparison will fail spuriously due to SVD sign ambiguity:

```rust
/// Assert two SVD results are equivalent modulo gauge freedom.
/// Singular values are unique and positive, so they must match exactly
/// (up to floating-point tolerance). Singular vectors may differ by sign.
macro_rules! assert_svd_equivalent {
    ($svd_a:expr, $svd_b:expr, $tol:expr) => {
        // Singular values must match
        assert_eq!($svd_a.rank, $svd_b.rank);
        for (s_a, s_b) in $svd_a.singular_values.iter().zip($svd_b.singular_values.iter()) {
            assert!(
                (s_a - s_b).abs() < $tol,
                "Singular value mismatch: {} vs {}", s_a, s_b
            );
        }
        // Reconstruction error must be within tolerance for both
        let err_a = svd_reconstruction_error_full($svd_a);
        let err_b = svd_reconstruction_error_full($svd_b);
        assert!(err_a < $tol, "SVD_a reconstruction error: {}", err_a);
        assert!(err_b < $tol, "SVD_b reconstruction error: {}", err_b);
    };
}
```

---

## 17. Performance Invariants

| Operation | Invariant |
|:----------|:----------|
| `DeviceFaer::gemm` (f64, 1000x1000) | >= 90% of peak DGEMM FLOP/s on the test machine |
| `svd_truncated` (f64, 200x200, rank 50) | < 5 ms wall time on reference hardware (to be calibrated) |
| `block_gemm` (U1, 10 sectors, D=100/sector) | LPT scheduling overhead < 1% of total GEMM time |
| `ThreadingRegime::select` | Zero allocations (operates on `BlockSparseTensor` metadata only) |
| `compute_fusion_rule` | < 10 ns per call (single arithmetic operation on packed u64) |
| `tk_mat_to_faer_owned` (f64, m x n) | O(m*n) copy; acceptable as dominated by O(m*n*min(m,n)) decomposition cost |

CI uses `iai` (instruction counting) for regression gating (+/-2% threshold). `criterion` benchmarks run locally on bare-metal hardware only.

---

## 18. Implementation Notes and Design Decisions

### 18.1 Why faer as Default

`faer` is a pure-Rust crate with state-of-the-art cache-oblivious GEMM and multithreaded SVD performance comparable to MKL on large matrices. Its native lazy conjugation support (`.conjugate()` flips one bit in the view struct) maps directly to `MatRef::is_conjugated`, eliminating the need for any conjugation-flag translation logic in `DeviceFaer`. MKL/OpenBLAS are available as optional FFI backends for users on Intel Xeon or HPC clusters where vendor-tuned BLAS libraries are available and already installed.

### 18.2 Why faer High-Level API

The implementation uses faer's high-level API (`Mat::thin_svd()`, `Mat::qr()`, `Mat::selfadjoint_eigendecomposition()`) exclusively, not the low-level `faer::linalg::svd::compute_svd` API. The high-level API is sufficient for all current needs. The low-level API is fragile across minor faer versions (e.g., function signatures and module paths change between 0.19.x releases), making it unsuitable for a production dependency. This decision was validated during draft implementation.

### 18.3 Faer Singular Value Ordering Trap

faer's `thin_svd()` returns singular values in **ascending** order, which is the opposite of LAPACK convention (descending). The implementation re-sorts to descending order after every SVD call. This re-sorting also requires corresponding reordering of the columns of U and the rows of Vt. Failing to re-sort is a silent correctness bug: truncation by `max_rank` would discard the *largest* singular values instead of the smallest, producing catastrophically wrong MPS truncations. This behavior is tested explicitly by `svd_descending_order`.

### 18.4 Why Tikhonov Regularization Instead of Simple Cutoff

A simple singular-value cutoff (discard s_i < epsilon, invert the rest) produces a pseudo-inverse that can still diverge: if s_i = 2*epsilon, the inverse is 1/(2*epsilon) which may be 10^8 for epsilon = 5e-9. This generates large tensor entries that corrupt all subsequent contractions. The Tikhonov formula `s/(s^2 + delta^2)` has a bounded maximum value of `1/(2*delta)` (achieved at s = delta), providing a hard ceiling on inverse magnitudes regardless of the input. For well-conditioned singular values (s >> delta), the Tikhonov formula approximates the true inverse to relative error O(delta^2/s^2).

### 18.5 LPT vs Work-Stealing Alone

Rayon's work-stealing ensures that idle threads steal tasks from busy threads. However, if the task list is sorted in ascending FLOP order (quantum-number order, which is the natural storage order), the last few tasks in the queue are the heaviest. Threads that finish their light early tasks steal from the tail — but the stealing itself is not free, and a single massive sector at the end will still serialize the final phase of execution. Sorting in descending order (LPT) ensures the heaviest tasks are dispatched first, when all threads are free, and the tail consists only of cheap tasks. Analytical worst-case analysis shows LPT scheduling achieves at most (4/3 - 1/(3m)) x OPT makespan for m machines, a well-known bound from scheduling theory.

### 18.6 SVD Residual Check in Debug Builds

The `debug_assert!` reconstruction check after SVD catches a known failure mode: SVD can return success while producing subtly inaccurate small singular values for pathologically ill-conditioned inputs. The residual check detects this "silent inaccuracy" case. It is compiled out in `--release` (`debug_assert` is a no-op), adding zero production overhead. Three guards protect against false positives:

1. **Full-rank guard:** Only runs when `result.rank == min(m, n)`. If the result is truncated, reconstruction error is expected.
2. **Near-zero norm guard:** Skips check if `||A||_F < 1e-30` to avoid division by zero.
3. **NumCast threshold:** Uses `num_traits::NumCast` for the `1e-10` threshold to support multiple scalar types.

### 18.7 Arbitrary-Stride Matrices and BLAS

Standard BLAS (MKL, OpenBLAS) requires at least one unit stride per matrix (either row-major or column-major layout). Tensors that have been permuted via `DenseTensor::permute` may have non-unit strides in both dimensions. The BLAS backends panic in this case with a clear error message directing the caller to use `TensorCow::into_owned()` (which materializes a contiguous copy) before passing to BLAS. `DeviceFaer` handles arbitrary strides natively via faer's strided view constructors and is the correct backend for post-permute operations.

### 18.8 Faer Conversion Overhead and Unsafe Code

The faer conversion layer (`src/faer_convert.rs`) contains two categories of unsafe code:

1. **`to_faer_mat_ref()` / `faer_mat_mut!`:** Zero-copy view construction using `faer::mat::from_raw_parts` / `from_raw_parts_mut`. Unsafe because it constructs a reference from a raw pointer. Safety relies on the source `MatRef`/`MatMut` outliving the faer view.

2. **`faer_mat_mut!` macro:** Exists to work around a double-lifetime issue. The borrow checker cannot express a function that takes `&mut MatMut<T>` and returns a `faer::MatMut<T>` with the correct lifetime relationship. The macro inlines the unsafe construction at the call site.

3. **`tk_mat_to_faer_owned()`:** Safe O(m*n) copy. Used for SVD/QR/eigh where faer requires an owned `Mat`. The copy overhead is acceptable as it is dominated by the decomposition cost.

### 18.9 `DenseTensor<'static, T>` in Return Types

All decomposition result types (`SvdResult<T>`, `EighResult<T>`, `QrResult<T>`) use `DenseTensor<'static, T>` for their tensor fields. This is required because:

1. SVD/QR/eigh results are always freshly allocated — they never borrow from the input matrix.
2. The `'static` lifetime communicates that the data is fully owned and can be stored, moved, and returned without lifetime constraints.
3. Attempting to use a generic lifetime `'a` creates borrow-checker complications when the results are stored in structs or returned from functions.

---

## 19. Security Considerations

### 19.1 Unsafe Code Inventory

| Location | Purpose | Safety Argument |
|:---------|:--------|:----------------|
| `faer_convert.rs::to_faer_mat_ref()` | Zero-copy view from `MatRef` to `faer::MatRef` | Source data outlives the faer view; enforced by lifetime parameter `'a` |
| `faer_convert.rs::faer_mat_mut!` | Mutable zero-copy view from `MatMut` to `faer::MatMut` | Macro inlines at call site; borrow checker enforces exclusive access to the underlying `MatMut` |
| `DeviceMKL` / `DeviceOpenBLAS` (future) | FFI calls to C BLAS/LAPACK functions | Dimension validation before FFI call; `resolve_blas_layout` ensures valid stride configuration |
| `DeviceCuda` (future) | FFI calls to cuBLAS/cuSOLVER | CUDA handles managed by `cudarc` safe wrappers; stream synchronization managed by `tk-contract` |

---

## 20. Out of Scope

The following are explicitly **not** implemented in `tk-linalg`:

- Tensor contraction path optimization or DAG execution (-> `tk-contract`)
- In-house iterative eigensolvers (Lanczos, Davidson, Block-Davidson) for the DMRG ground-state problem (-> `tk-dmrg`; these require zero-allocation matvec closures and tight `SweepArena` integration)
- Krylov matrix-exponential for TDVP time evolution (-> `tk-dmft`)
- MPS/MPO data structures or gauge canonicalization (-> `tk-dmrg`)
- Any physical model logic, quantum number types, or block-sparse tensor construction (-> `tk-symmetry`)
- DMFT self-consistency loop, bath discretization, or spectral function computation (-> `tk-dmft`)
- Python bindings (-> `tk-python`)

---

## 21. Open Questions

| # | Question | Status |
|:--|:---------|:-------|
| 1 | **SU(2) fusion-rule fan-out:** `compute_fusion_rule` currently returns `Option<PackedSectorKey>` (one-to-one). For SU(2), j1 (x) j2 produces multiple output irreps. Task generation must produce `Vec<SectorGemmTask>` per input pair, with each task weighted by the corresponding Clebsch-Gordan coefficient. This is a structural change to `tasks.rs` scoped to the `su2-symmetry` feature flag. | Deferred to Phase 5; does not affect Abelian code path |
| 2 | **SU(2) output-sector collision (map-reduce):** Multiple input pairs (j_a, j_b) can map to the same output sector j_c. Naive parallel dispatch over all tasks creates a data race. The SU(2) task generation must group tasks by output sector key and accumulate partial contributions sequentially within each group before writing. The three-phase algorithm must be extended with a reduce phase between Phases 2 and 3. | Deferred to Phase 5; does not affect Abelian code path |
| 3 | **GPU dispatch threshold calibration:** The `GPU_DISPATCH_THRESHOLD = 500` constant routing small GEMM calls to CPU is based on general cuBLAS launch overhead estimates. This should be calibrated empirically on the target hardware (consumer A100 vs data-center H100 vs older V100) using Criterion benchmarks at D = 100, 200, 500, 1000. | Deferred to Phase 5 GPU integration |
| 4 | **Batched cuBLAS for fragmented-sector CUDA:** In FragmentedSectors mode on GPU, launching one `cublasXgemm` call per sector has high per-call overhead (~5 us). `cublasXgemmBatched` or `cublasXgemmStridedBatched` can amortize this across all sector tasks in a single kernel launch, potentially recovering 2-10x performance for large sector counts at small D. | Deferred to Phase 5 |
| 5 | **`eigh_lowest` for large matrices:** The current spec routes all dense EVD to `LinAlgBackend::eigh_lowest`. For the DMRG ground-state problem, the matrix is far too large to diagonalize densely (typically 2D^2 x 2D^2 with D=1000). The spec correctly documents that `eigh_lowest` is for small auxiliary matrices only. But the interface could be confusing. Should `eigh_lowest` carry a dimension limit (panic if n > threshold) to prevent misuse? | Open — decision needed before implementation |
| 6 | **`f128` GEMM path in DeviceOxiblas:** The architecture doc notes that `f128` SVD might need to route through `DeviceOxiblas` exclusively. Clarification needed: does `faer` provide an `f128` GEMM path via its generic arithmetic? Or is `f128` restricted to oxiblas for all operations? | Open — needs investigation against faer 0.19 changelog |
| 7 | **BLAS thread-count API for MKL/OpenBLAS:** `set_blas_num_threads` must call `MKL_Set_Num_Threads` or `openblas_set_num_threads` respectively. These are global state mutations that are unsafe in the presence of concurrent BLAS calls from other threads. The threading regime must guarantee that `set_blas_num_threads` is only called when no BLAS operations are in flight. Document and enforce this invariant. | Open — design clarification needed |
| 8 | **`max_sector_dim_on_any_leg` location:** Currently implemented locally in `threading.rs` because `tk-symmetry` does not provide this method on `BlockSparseTensor`. Should this be upstreamed to `tk-symmetry`? | Open — coordinate with `tk-symmetry` spec |
| 9 | **Same-output-sector accumulation scalability:** The current linear scan for multi-input-to-same-output sector accumulation in `block_gemm` is O(n^2) in the number of output sectors. For large sector counts (e.g., SU(2) with many irreps), this should use a `HashMap<PackedSectorKey, DenseTensor>`. When should this optimization be applied? | Open — profile with realistic SU(2) sector counts |
| 10 | **`f32`/`C32`/`C64` implementation timeline:** Only `f64` is currently implemented. The macro-based specialization pattern is designed but not yet applied. These types are needed for single-precision prototyping and complex-valued Hamiltonians. | Open — needed before Phase 3 complex-valued models |

---

## 22. Future Considerations

- **`DeviceOxiblas` implementation:** Required for the intended `DefaultDevice = DeviceAPI<DeviceFaer, DeviceOxiblas>` configuration. Blocked on oxiblas crate maturity.
- **Rayon parallel dispatch in `block_gemm`:** LPT sorting is ready; Rayon `into_par_iter()` dispatch is deferred until sequential correctness is fully validated.
- **`f32`/`C32`/`C64` scalar support:** Macro-based specialization for `DeviceFaer` to generate `impl LinAlgBackend<f32>`, etc.
- **Partitioned threading scheduler:** The current simple binary heuristic should be replaced with the partitioned scheduler from the architecture document when profiling data from real DMRG sweeps is available (Phase 4+).
- **HashMap-based sector accumulation:** Replace linear scan in `block_gemm` for same-output-sector case. Priority increases with SU(2) symmetry support.
- **MKL/OpenBLAS/CUDA backend implementations:** Deferred to Phase 5+.
