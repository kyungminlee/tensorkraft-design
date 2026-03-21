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
| `src/scalar.rs` | `Scalar` trait with `conj`, `abs_sq`, `from_real`, `is_real`; impls for `f32`, `f64`, `Complex<f32>`, `Complex<f64>`; type aliases `C32`, `C64` |
| `src/shape.rs` | `TensorShape` with `SmallVec<[usize; 6]>` dims/strides; constructors (`row_major`, `col_major`, `with_strides`); methods (`numel`, `rank`, `offset`, `is_contiguous`, `permute`, `reshape`, `slice_axis`) |
| `src/storage.rs` | `TensorStorage<'a, T>` enum with `Owned(Vec<T>)` / `Borrowed(&'a [T])` variants, Copy-on-Write semantics |
| `src/tensor.rs` | `DenseTensor<'a, T>` with shape + CoW storage; `TempTensor` alias; methods including `into_owned()`, `as_mat_ref()`, `as_mat_mut()`, `permute()`, `reshape()`, `slice_axis()` |
| `src/matview.rs` | `MatRef<'a, T>` with lazy `is_conjugated` flag; `MatMut<'a, T>`; zero-copy `adjoint()`, `conjugate()`, `transpose()` |
| `src/arena.rs` | `SweepArena` with `bumpalo::Bump`; CUDA-gated `ArenaStorage` enum (`Pinned`/`Pageable`); `PinnedArena` placeholder; `Drop` impl releasing pinned budget |
| `src/pinned.rs` | `PinnedMemoryTracker` with static atomics, CAS-loop `try_reserve`, `release`, `initialize_budget` |

## Test Results

- **43 tests pass** (41 base + 2 pinned-memory tests with `backend-cuda`)
- Compiles cleanly on both default and `backend-cuda` feature configurations

## Known Limitations

### 1. ~~`SweepArena::alloc_tensor` uses heap `Vec` internally~~ (RESOLVED)

Fixed by merging `TensorCow` into `TensorStorage` as a single `Owned(Vec<T>)` / `Borrowed(&'a [T])` enum. `SweepArena::alloc_tensor` now bump-allocates a zeroed slice and wraps it as `TensorStorage::Borrowed`, with the lifetime tied to the arena. No heap `Vec` is created for arena-allocated tensors. `DenseTensor::borrowed()` now takes a `&'a [T]` slice directly.

### 2. ~~`DenseTensor::slice_axis` does not adjust the data pointer offset~~ (RESOLVED)

Fixed by adding an `offset: usize` field to `DenseTensor`. All data-access methods (`as_slice`, `as_mut_slice`, `as_mat_ref`, `as_mat_mut`) apply the offset. `into_owned()` gathers elements into a fresh contiguous buffer when offset is nonzero or layout is non-contiguous. Chained slicing accumulates offsets correctly. Covered by 5 new tests (`slice_axis_offset_correct`, `slice_axis_cols`, `slice_axis_into_owned_gathers_elements`, `slice_axis_chained`, `slice_axis_mat_ref`).

### 3. `PinnedArena` is a placeholder

`PinnedArena` in `src/arena.rs` wraps a standard `bumpalo::Bump` allocator. A real implementation must call `cudaMallocHost` / `cudaFreeHost` via FFI to allocate page-locked memory that is DMA-capable for high-bandwidth GPU transfers. This is blocked on CUDA toolkit bindings and is expected to be implemented in Phase 5.

### 4. `f128` support (`backend-oxiblas`) is not implemented

The `Scalar` trait implementation for `f128` is gated behind the `backend-oxiblas` feature flag but is not yet written. This depends on Rust's `f128` stabilization status and on whether `faer` provides an `f128` GEMM path (open question #2 in the tech spec).

### 5. No `StorageDevice` trait generalization

The tech spec (section 15) describes a future `StorageDevice` trait that parameterizes `TensorStorage<T, D>` over a device type (`HostDevice`, `CudaDevice`, `MpiDevice`). This is deferred to Phase 5. The current `TensorStorage<'a, T>` enum (`Owned(Vec<T>)` / `Borrowed(&'a [T])`) is hardcoded to host memory.

### 6. ~~No compile-fail tests yet~~ (RESOLVED)

Five `trybuild`-based compile-fail tests added in `tests/compile_fail/`:
- `arena_tensor_outlives_reset` — `TempTensor` cannot be used after `arena.reset()`
- `arena_tensor_escape_scope` — `TempTensor` cannot escape the function that owns the arena
- `borrowed_storage_outlives_data` — `TensorStorage::Borrowed` cannot outlive its source data
- `slice_view_outlives_tensor` — sliced view cannot be used after the original tensor is moved
- `matref_outlives_tensor` — `MatRef` cannot be used after the tensor is moved

### 7. ~~No `proptest` property-based tests yet~~ (RESOLVED)

Six `proptest` property-based tests added to `shape.rs`:
- `prop_offset_within_bounds` — offset for any valid multi-index is within numel
- `prop_permute_preserves_numel` — permutation never changes element count
- `prop_permute_roundtrip` — double-reverse permutation restores original shape
- `prop_reshape_roundtrip` — flatten then reshape back recovers original dims
- `prop_slice_axis_numel` — slicing one element along axis divides numel by that axis size
- `prop_col_major_same_numel` — row-major and col-major have same numel and dims
