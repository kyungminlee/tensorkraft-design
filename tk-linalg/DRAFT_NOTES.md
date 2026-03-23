# tk-linalg — Draft Notes

**Status:** Working draft — compiles and passes basic tests but is not production-ready.
**Based on:** `techspec/3_tech-spec_tk-linalg.md` and `ARCHITECTURE.md`

---

## What is implemented

### Core abstractions (complete)
- **`LinAlgBackend<T>`** trait (object-safe, `Send + Sync`) with:
  - `gemm` — conjugation-aware GEMM (α·op(A)·op(B) + β·C)
  - `svd_truncated` — with gesdd/gesvd fallback and debug residual validation
  - `svd_truncated_gesdd` / `svd_truncated_gesvd` — split algorithm methods
  - `eigh_lowest` — dense symmetric/Hermitian eigendecomposition (lowest k)
  - `qr` — thin QR decomposition
  - `regularized_svd_inverse` — Tikhonov regularization (s/(s²+δ²))
- **`SparseLinAlgBackend<T, Q>`** trait with:
  - `spmv` — block-sparse matrix-vector multiply
  - `block_gemm` — block-sparse GEMM with LPT scheduling

### Return types (complete)
- `SvdResult<T>` — u, singular_values, vt, rank, truncation_error
- `EighResult<T>` — eigenvalues, eigenvectors
- `QrResult<T>` — q, r
- `SvdConvergenceError`

### Error handling (complete)
- `LinAlgError` enum with all spec'd variants
- `LinAlgResult<T>` type alias

### Threading regime (complete)
- `ThreadingRegime::FatSectors` / `FragmentedSectors`
- `ThreadingRegime::select()` heuristic

### LPT task scheduling (complete)
- `SectorGemmTask<T>` with FLOP estimates
- `lpt_sort()` — descending FLOP sort
- `compute_fusion_rule()` — Abelian rank-2 fusion rule
- `compute_output_indices()` — output QIndex construction

### DeviceFaer backend (functional for f32, f64, C32, C64)
- `LinAlgBackend<T>` for all four scalar types via `macro_rules!`
- `SparseLinAlgBackend<T, Q>` with Rayon parallel dispatch (`#[cfg(feature = "parallel")]`)
- Real types (f32, f64): zero-copy GEMM via faer pointer views
- Complex types (C32, C64): copy-based GEMM (faer split storage)
- Conjugation-aware GEMM: real uses faer lazy conjugation, complex applies via `MatRef::get()`
- SVD with descending singular value ordering and proper conjugate-transpose for V†
- Dense Hermitian eigendecomposition (real eigenvalues for complex matrices)
- QR via Householder factorization

### DeviceAPI composite backend (complete)
- `DeviceAPI<D, S>` — delegates dense ops to D, sparse ops to S
- `DefaultDevice` type alias → `DeviceAPI<DeviceFaer, DeviceFaer>`

### Build script (complete)
- Mutual exclusivity enforcement for backend-mkl + backend-openblas

---

## What is NOT yet implemented (known gaps)

### High priority — COMPLETED
1. ~~**f32, C32, C64 backend implementations**~~ — **DONE.** All four scalar types (f32, f64,
   C32, C64) are now generated via `macro_rules!`. Real types (f32, f64) use zero-copy faer
   `MatRef` conversion for GEMM; complex types (C32, C64) use copy-based conversion to handle
   faer's split real/imaginary internal storage. V→V† in SVD uses `Scalar::conj()` (no-op
   for real types). Debug SVD residual check uses precision-aware tolerance.

2. ~~**Rayon parallelism in block_gemm**~~ — **DONE.** `#[cfg(feature = "parallel")]` path
   uses `par_iter()` with `faer::Parallelism::None` per task (single-threaded BLAS, Rayon
   distributes independent sector GEMMs). Sequential accumulation by sector key follows the
   parallel map phase. `#[cfg(not(feature = "parallel"))]` retains sequential execution.

3. ~~**`max_sector_dim_on_any_leg`**~~ — **DONE.** Implemented locally in `threading.rs`
   as `max_sector_dim_any_leg()`, calling `tensor.max_sector_dim_on_leg(leg)`.

### Medium priority
4. **DeviceOxiblas backend** — Stub only. The `oxiblas` crate provides sparse formats
   (BSR, CSR, etc.) and SIMD-accelerated operations. Integration requires:
   - `SparseLinAlgBackend` impl with oxiblas BSR conversion
   - `f128` scalar support when both backend-oxiblas and f128 are active

5. **DeviceMKL backend** — Stub only. Requires:
   - FFI bindings via `intel-mkl-sys`
   - `resolve_blas_layout()` for stride → CBLAS_TRANSPOSE mapping
   - Thread count management via `mkl_set_num_threads`

6. **DeviceOpenBLAS backend** — Stub only. Structurally identical to MKL.

7. **DeviceCuda backend** — Stub only. Requires:
   - `cudarc` integration for cuBLAS/cuSOLVER
   - Stream-aware async execution
   - Three-way GPU/CPU/Rayon LPT partition

8. **`set_blas_num_threads`** — Currently a no-op. Needs MKL/OpenBLAS FFI calls.

### Low priority (deferred per spec)
9. **SU(2) fusion-rule fan-out** — `compute_fusion_rule` returns `Option` (one-to-one).
   SU(2) needs `Vec<SectorGemmTask>` per input pair with Clebsch-Gordan weights.
   Deferred to Phase 5.

10. **SU(2) output-sector collision (map-reduce)** — Multiple input pairs mapping to
    the same output sector need grouped accumulation. Deferred to Phase 5.

11. **GPU dispatch threshold calibration** — `GPU_DISPATCH_THRESHOLD = 500` is
    a placeholder. Needs Criterion benchmarks on target hardware.

12. **Partitioned LPT dispatch** — The spec describes a three-phase partitioned
    scheduler splitting tasks at `BLAS_FLOP_THRESHOLD`. Currently only two-regime
    (FatSectors / FragmentedSectors) is implemented.

---

## Design decisions made in this draft

1. **DeviceFaer as both dense and sparse backend** — Until oxiblas is integrated,
   `DefaultDevice = DeviceAPI<DeviceFaer, DeviceFaer>`. DeviceFaer provides a naive
   sequential `SparseLinAlgBackend` implementation for testing.

2. **Rayon-parallel block_gemm** — LPT sorting is implemented, and Rayon parallel
   dispatch is gated behind `#[cfg(feature = "parallel")]`. Each Rayon task uses
   `faer::Parallelism::None` (single-threaded BLAS) to avoid thread oversubscription.

3. **Fusion rule limited to rank-2** — `compute_fusion_rule` only handles rank-2
   tensor × tensor contraction. Higher-rank tensors must be reshaped to rank-2
   (via `fuse_legs`) before calling `block_gemm`, which is the standard DMRG approach.

4. **Cargo.toml uses commented-out deps** — FFI backend dependencies (intel-mkl-sys,
   openblas-src, cudarc, oxiblas) are commented out because they require system libraries.
   Uncomment when the corresponding build infrastructure is in place.

---

## Testing status

Unit tests included for:
- `LinAlgError` display formatting
- `frobenius_norm` for real and complex matrices
- `ThreadingRegime` equality and debug formatting
- `lpt_sort` descending FLOP ordering
- `DeviceFaer::gemm` — identity multiplication, alpha/beta scaling (f64)
- `DeviceFaer::gemm` — identity multiplication (f32, C32, C64)
- `DeviceFaer::gemm` — conjugated complex GEMM (C64)
- `DeviceFaer::svd_truncated` — reconstruction accuracy (f32, f64, C32, C64), rank truncation
- `DeviceFaer::eigh_lowest` — symmetric eigenvalue correctness (f32, f64)
- `DeviceFaer::eigh_lowest` — Hermitian eigenvalue correctness (C64)
- `DeviceFaer::qr` — Q·R reconstruction accuracy (f64, C64)
- `regularized_svd_inverse` — large-s accuracy, zero-s safety (no NaN/Inf)

Not yet tested:
- Cross-backend equivalence (needs MKL/OpenBLAS)
- Property-based tests (proptest strategies)
- Block-sparse GEMM with realistic quantum numbers
- Performance benchmarks (Criterion/iai)

---

## Files

```
tk-linalg/
├── Cargo.toml           Feature flags, dependencies
├── build.rs             Mutual exclusivity enforcement
├── DRAFT_NOTES.md       This file
└── src/
    ├── lib.rs           Module declarations and re-exports
    ├── error.rs         LinAlgError, LinAlgResult
    ├── results.rs       SvdResult, EighResult, QrResult, SvdConvergenceError
    ├── traits.rs        LinAlgBackend<T>, SparseLinAlgBackend<T, Q>, helpers
    ├── threading.rs     ThreadingRegime enum and select() heuristic
    ├── tasks.rs         SectorGemmTask, LPT scheduling, fusion_rule
    └── device/
        ├── mod.rs       DeviceAPI<D,S>, DefaultDevice type alias
        └── faer.rs      DeviceFaer: LinAlgBackend<T> for f32/f64/C32/C64
```
