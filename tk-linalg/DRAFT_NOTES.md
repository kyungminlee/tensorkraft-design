# tk-linalg — Draft Notes

**Status:** Working draft — compiles and passes all tests but is not production-ready.
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
- `ThreadingRegime::partition_tasks()` — partitioned LPT dispatch that splits tasks
  into heavy (≥ `BLAS_FLOP_THRESHOLD = 1M FLOPs`, multithreaded BLAS) and light
  (< threshold, Rayon parallel with single-threaded BLAS per task)
- `ThreadingRegime::blas_flop_threshold()` — exposes the constant for calibration

### LPT task scheduling (complete)
- `SectorGemmTask<T>` with FLOP estimates
- `lpt_sort()` — descending FLOP sort
- `compute_fusion_rule()` — Abelian rank-2 fusion rule
- `compute_fusion_rule_su2()` — SU(2) non-Abelian fusion-rule fan-out
  (behind `su2-symmetry` feature). Uses `SU2Irrep::fuse_all()` to enumerate
  all output irreps from j₁ ⊗ j₂ = |j₁−j₂| ⊕ ... ⊕ (j₁+j₂).
  Generates `Vec<SectorGemmTask>` per input pair. CG coefficient weighting
  is delegated to `tk-contract`'s structural contraction injection point.
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

### BLAS thread management (complete)
- `set_blas_num_threads()` with `#[cfg(feature)]`-gated FFI:
  - `backend-mkl`: calls `MKL_Set_Num_Threads`
  - `backend-openblas`: calls `openblas_set_num_threads`
  - Neither enabled: no-op (DeviceFaer uses Rayon's own thread pool)
- Safety invariant: must only be called when no BLAS operations are in flight.
  The `ThreadingRegime` enforces this by calling it once before dispatch.

### SU(2) output-sector collision handling (complete)
- The `block_gemm` accumulation logic handles output-sector collision for both
  Abelian and non-Abelian cases. Multiple input pairs mapping to the same output
  sector key are accumulated element-wise via sequential scan.
- For large sector counts, a HashMap-based accumulator would be more efficient
  but is not yet needed at current scale.

---

## Remaining limitations

### Backend stubs (require system libraries)

1. **DeviceOxiblas backend** — Stub only. The `oxiblas` crate provides sparse formats
   (BSR, CSR, etc.) and SIMD-accelerated operations. Integration requires:
   - `SparseLinAlgBackend` impl with oxiblas BSR conversion
   - `f128` scalar support when both backend-oxiblas and f128 are active

2. **DeviceMKL backend** — Stub only. Requires:
   - FFI bindings via `intel-mkl-sys`
   - `resolve_blas_layout()` for stride → CBLAS_TRANSPOSE mapping

3. **DeviceOpenBLAS backend** — Stub only. Structurally identical to MKL.

4. **DeviceCuda backend** — Stub only. Requires:
   - `cudarc` integration for cuBLAS/cuSOLVER
   - Stream-aware async execution
   - Three-way GPU/CPU/Rayon LPT partition

### Calibration (require target hardware)

5. **GPU dispatch threshold calibration** — `GPU_DISPATCH_THRESHOLD = 500` is
   a placeholder. Needs Criterion benchmarks on target hardware (A100, H100, V100).
   Criterion benchmark infrastructure is now in place.

### Testing gaps

6. **Cross-backend equivalence tests** — Cannot be tested until MKL/OpenBLAS
   backends are integrated.

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

**39 tests total** (28 unit + 3 integration + 4 proptest + 4 spmv integration).

### Changes in cross-crate gap-filling pass

- Added `spmv_test.rs` integration tests: `spmv_matches_dense_reference`, `spmv_zero_flux`, `spmv_nonzero_flux`, `spmv_empty_tensor`.

Unit tests (28):
- `LinAlgError` display formatting (3 tests)
- `frobenius_norm` for real and complex matrices (2 tests)
- `ThreadingRegime` equality, debug formatting, partition, threshold (4 tests)
- `lpt_sort` descending FLOP ordering (1 test)
- `DeviceFaer::gemm` — identity multiplication, alpha/beta scaling (f32, f64, C32, C64, conjugated C64) (5 tests)
- `DeviceFaer::svd_truncated` — reconstruction accuracy (f32, f64, C32, C64), rank truncation (5 tests)
- `DeviceFaer::eigh_lowest` — symmetric eigenvalue correctness (f32, f64), Hermitian (C64) (3 tests)
- `DeviceFaer::qr` — Q·R reconstruction accuracy (f64, C64) (2 tests)
- `regularized_svd_inverse` — large-s accuracy, zero-s safety (2 tests)
- Frobenius norm real/complex (1 test)

Integration tests (3, in `tests/block_gemm_realistic.rs`):
- `block_gemm_matches_dense_reference` — U1 with Sz=-1,0,+1 charges, non-trivial data,
  compared against dense GEMM reference (max error < 1e-10)
- `block_gemm_nonzero_flux` — creation/annihilation operators with flux ±1
- `block_gemm_sector_count_bounded` — output sector count bounded by input sectors

Property-based tests (4, in `tests/proptest_linalg.rs`):
- `gemm_associativity` — (A*B)*C == A*(B*C) for random dims 2..=8
- `svd_round_trip` — ||A - U·Σ·V†||_F / ||A||_F < 1e-10 for random dims 2..=16
- `regularized_inverse_decreasing_delta` — smaller δ → closer to true inverse
- `block_gemm_output_sectors_valid` — all output sectors satisfy flux rule

Criterion benchmarks (in `benches/linalg_benchmarks.rs`):
- `gemm_f64_100x100` — GEMM throughput measurement
- `svd_truncated_f64_50x50` — SVD latency measurement
- `block_gemm_u1_10sectors_d10` — block-sparse GEMM with LPT scheduling
- `threading_regime_select` — regime selection overhead (metadata-only, zero alloc)

---

## Files

```
tk-linalg/
├── Cargo.toml           Feature flags, dependencies (criterion, proptest dev-deps)
├── build.rs             Mutual exclusivity enforcement
├── DRAFT_NOTES.md       This file
├── benches/
│   └── linalg_benchmarks.rs   Criterion benchmarks (gemm, svd, block_gemm, threading)
├── tests/
│   ├── block_gemm_realistic.rs  Integration tests with realistic U1 quantum numbers
│   └── proptest_linalg.rs       Property-based tests (gemm, svd, regularized_inverse, block_gemm)
└── src/
    ├── lib.rs           Module declarations and re-exports
    ├── error.rs         LinAlgError, LinAlgResult
    ├── results.rs       SvdResult, EighResult, QrResult, SvdConvergenceError
    ├── traits.rs        LinAlgBackend<T>, SparseLinAlgBackend<T, Q>, helpers, set_blas_num_threads
    ├── threading.rs     ThreadingRegime enum, select(), partition_tasks()
    ├── tasks.rs         SectorGemmTask, LPT scheduling, fusion_rule, compute_fusion_rule_su2
    └── device/
        ├── mod.rs       DeviceAPI<D,S>, DefaultDevice type alias
        └── faer.rs      DeviceFaer: LinAlgBackend<T> for f32/f64/C32/C64
```
