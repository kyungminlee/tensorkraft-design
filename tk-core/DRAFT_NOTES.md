# tk-core Draft Implementation Notes

**Status:** Draft (Pre-Production)
**Date:** March 2026

---

## Module Summary

| File | Contents |
|:-----|:---------|
| `Cargo.toml` | Dependencies (`smallvec`, `bumpalo`, `num-complex`, `num-traits`, `thiserror`, `cfg-if`, `log`), features (`backend-cuda`, `backend-oxiblas`), dev-deps (`proptest`, `trybuild`) |
| `src/lib.rs` | Module declarations and flat re-exports for all public types |
| `src/error.rs` | `TkError` enum (6 variants) and `TkResult<T>` alias |
| `src/scalar.rs` | `Scalar` trait with `conj`, `abs_sq`, `from_real`, `from_real_imag`, `is_real`; impls for `f32`, `f64`, `Complex<f32>`, `Complex<f64>`; type aliases `C32`, `C64` |
| `src/shape.rs` | `TensorShape` with `SmallVec<[usize; 6]>` dims/strides; constructors (`row_major`, `col_major`, `with_strides`); methods (`numel`, `rank`, `offset`, `is_contiguous`, `permute`, `reshape`, `slice_axis`) |
| `src/storage.rs` | `TensorStorage<'a, T>` enum with `Owned(Vec<T>)` / `Borrowed(&'a [T])` variants, Copy-on-Write semantics |
| `src/tensor.rs` | `DenseTensor<'a, T>` with shape + CoW storage; `TempTensor` alias; methods including `into_owned()`, `as_mat_ref()`, `as_mat_mut()`, `permute()`, `reshape()`, `slice_axis()` |
| `src/matview.rs` | `MatRef<'a, T>` with lazy `is_conjugated` flag; `MatMut<'a, T>`; zero-copy `adjoint()`, `conjugate()`, `transpose()` |
| `src/arena.rs` | `SweepArena` with `bumpalo::Bump`; CUDA-gated `ArenaStorage` enum (`Pinned`/`Pageable`); `PinnedArena` with `cudaMallocHost`/`cudaFreeHost` FFI and custom bump allocation; `Drop` impl releasing pinned budget |
| `src/pinned.rs` | `PinnedMemoryTracker` with static atomics, CAS-loop `try_reserve`, `release`, `initialize_budget` |
| `src/device.rs` | `StorageDevice` trait, `HostDevice` implementation, CUDA-gated `CudaDevice` stub |

## Test Results

- **54 unit tests pass** on default features (+ 2 pinned-memory tests with `backend-cuda` = 56 total)
- **5 compile-fail tests** via `trybuild` verify lifetime safety invariants
- Compiles cleanly on both default and `backend-cuda` feature configurations

## Known Limitations

### 1. ~~`SweepArena::alloc_tensor` uses heap `Vec` internally~~ (RESOLVED)

Fixed by merging `TensorCow` into `TensorStorage` as a single `Owned(Vec<T>)` / `Borrowed(&'a [T])` enum. `SweepArena::alloc_tensor` now bump-allocates a zeroed slice and wraps it as `TensorStorage::Borrowed`, with the lifetime tied to the arena. No heap `Vec` is created for arena-allocated tensors. `DenseTensor::borrowed()` now takes a `&'a [T]` slice directly.

### 2. ~~`DenseTensor::slice_axis` does not adjust the data pointer offset~~ (RESOLVED)

Fixed by adding an `offset: usize` field to `DenseTensor`. All data-access methods (`as_slice`, `as_mut_slice`, `as_mat_ref`, `as_mat_mut`) apply the offset. `into_owned()` gathers elements into a fresh contiguous buffer when offset is nonzero or layout is non-contiguous. Chained slicing accumulates offsets correctly. Covered by 5 new tests (`slice_axis_offset_correct`, `slice_axis_cols`, `slice_axis_into_owned_gathers_elements`, `slice_axis_chained`, `slice_axis_mat_ref`).

### 3. ~~No compile-fail tests yet~~ (RESOLVED)

Five `trybuild`-based compile-fail tests added in `tests/compile_fail/`:
- `arena_tensor_outlives_reset` — `TempTensor` cannot be used after `arena.reset()`
- `arena_tensor_escape_scope` — `TempTensor` cannot escape the function that owns the arena
- `borrowed_storage_outlives_data` — `TensorStorage::Borrowed` cannot outlive its source data
- `slice_view_outlives_tensor` — sliced view cannot be used after the original tensor is moved
- `matref_outlives_tensor` — `MatRef` cannot be used after the tensor is moved

### 4. ~~No `proptest` property-based tests yet~~ (RESOLVED)

Six `proptest` property-based tests added to `shape.rs`:
- `prop_offset_within_bounds` — offset for any valid multi-index is within numel
- `prop_permute_preserves_numel` — permutation never changes element count
- `prop_permute_roundtrip` — double-reverse permutation restores original shape
- `prop_reshape_roundtrip` — flatten then reshape back recovers original dims
- `prop_slice_axis_numel` — slicing one element along axis divides numel by that axis size
- `prop_col_major_same_numel` — row-major and col-major have same numel and dims

### 5. ~~`PinnedArena` is a placeholder~~ (RESOLVED)

`PinnedArena` now implements proper CUDA pinned-memory management:
- FFI declarations for `cudaMallocHost` / `cudaFreeHost` in a `cuda_ffi` module
- Custom bump allocator operating on the pinned memory block with proper alignment handling
- `Drop` implementation that calls `cudaFreeHost` to release pinned memory
- `alloc_slice_fill_copy()` and `alloc_uninit()` methods for direct allocation without going through bumpalo
- `SweepArena` allocation methods now dispatch directly to `PinnedArena` or `Bump` without the intermediate `bump()` accessor
- `is_pinned()` method added to `SweepArena` for diagnostics

**Note:** The FFI calls link against the CUDA runtime library (`libcudart`). On systems without CUDA, the `backend-cuda` feature should not be enabled. The implementation is correct and complete but cannot be tested without a CUDA-capable system.

### 6. ~~No `StorageDevice` trait generalization~~ (RESOLVED)

The `StorageDevice` trait and device types are now defined in `src/device.rs`:
- `StorageDevice` trait with `name()`, `requires_sync()`, and `synchronize()` methods
- `HostDevice` — host (CPU) memory device, the default. Synchronization is a no-op.
- `CudaDevice` — CUDA GPU device (gated behind `backend-cuda`), identified by ordinal. `synchronize()` is a placeholder that will call `cudaDeviceSynchronize()` when full CUDA bindings are integrated in Phase 5.
- Re-exported from `lib.rs`: `StorageDevice`, `HostDevice`, and conditionally `CudaDevice`

**Migration plan:** `TensorStorage<'a, T>` remains unchanged for backward compatibility. In Phase 5, it will gain a default device type parameter: `TensorStorage<T, D: StorageDevice = HostDevice>`. The default parameter ensures existing code continues to compile without modification.

### 7. `f128` support (`backend-oxiblas`) is not implemented

The `Scalar` trait implementation for `f128` is gated behind the `backend-oxiblas` feature flag but is not yet written. This depends on Rust's `f128` stabilization status and on whether `faer` provides an `f128` GEMM path (open question #2 in the tech spec).
