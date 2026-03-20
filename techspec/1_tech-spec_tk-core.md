# Technical Specification: `tk-core`

**Crate:** `tensorkraft/crates/tk-core`
**Version:** 0.1.0 (Pre-Implementation)
**Status:** Specification
**Last Updated:** March 2026

---

## 1. Overview

`tk-core` is the leaf crate of the tensorkraft workspace. Every other crate depends on it, so it must remain stable, minimal, and free of mathematical logic. Its sole responsibility is:

- **Dimensional metadata** — shape/stride management for zero-copy tensor views
- **Memory management** — arena allocators (`SweepArena`), Copy-on-Write storage (`TensorCow`), and optional pinned-memory budget tracking (`PinnedMemoryTracker`)
- **Matrix view types** — `MatRef`/`MatMut` carrying a lazy conjugation flag
- **Element-type abstraction** — the `Scalar` trait hierarchy
- **Shared error types** — `TkError` and subtypes used across the workspace

Mathematical operations on tensors (contraction, addition, trace, decomposition) are **not** implemented here. They belong in `tk-linalg` or `tk-contract`.

**No dependencies** — `tk-core` is a pure-Rust leaf crate. It may depend only on:
- `smallvec` (stack-allocated small vectors)
- `bumpalo` (arena allocator)
- `num-complex`, `num-traits` (numeric type abstractions)
- `thiserror` (error derive macros)
- `cfg-if` (feature-flag conditional compilation)

---

## 2. Module Structure

```
tk-core/
├── Cargo.toml
├── build.rs              (optional: feature conflict detection)
└── src/
    ├── lib.rs            re-exports all public items
    ├── scalar.rs         Scalar trait + implementations
    ├── shape.rs          TensorShape, stride computation
    ├── storage.rs        TensorStorage<T>, TensorCow<T>
    ├── tensor.rs         DenseTensor<T>, TempTensor<T>
    ├── matview.rs        MatRef<T>, MatMut<T>, adjoint/conjugate
    ├── arena.rs          SweepArena, ArenaStorage, TempTensor ownership
    ├── pinned.rs         PinnedMemoryTracker (cfg: backend-cuda)
    └── error.rs          TkError, TkResult<T>
```

---

## 3. The `Scalar` Trait

### 3.1 Definition

```rust
/// Sealed marker for element types supported by tensorkraft.
/// Implemented for: f32, f64, Complex<f32>, Complex<f64>,
/// and optionally f128 when feature "backend-oxiblas" is active.
pub trait Scalar:
    Copy + Clone + Send + Sync
    + num_traits::Zero + num_traits::One
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::fmt::Debug
    + 'static
{
    /// The underlying real type. `f64` for `f64` and `Complex<f64>`.
    type Real: Scalar<Real = Self::Real>
        + num_traits::Float
        + PartialOrd;

    /// Complex conjugate. No-op for real types.
    fn conj(self) -> Self;

    /// Squared absolute value: |z|².
    fn abs_sq(self) -> Self::Real;

    /// Embed a real value into this scalar type.
    fn from_real(r: Self::Real) -> Self;

    /// Returns true iff complex conjugation is a no-op (i.e., T is real).
    /// Used by the contraction engine to skip conjugation-flag propagation
    /// in tight loops over real-valued models.
    fn is_real() -> bool;
}
```

### 3.2 Implementations

| Type | `Real` | `is_real()` | Notes |
|:-----|:-------|:------------|:------|
| `f32` | `f32` | `true` | |
| `f64` | `f64` | `true` | Default for most DMRG runs |
| `Complex<f32>` | `f32` | `false` | |
| `Complex<f64>` | `f64` | `false` | Quantum impurity solver |
| `f128` | `f128` | `true` | `#[cfg(feature = "backend-oxiblas")]` only |

### 3.3 Type Aliases

```rust
pub type C64 = num_complex::Complex<f64>;
pub type C32 = num_complex::Complex<f32>;
```

---

## 4. `TensorShape` — Dimensional Metadata

### 4.1 Definition

```rust
/// Shape and stride metadata for an N-dimensional tensor.
/// Stores no data — only the logical layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorShape {
    /// Extent of each dimension. Typical rank ≤ 6; SmallVec avoids heap
    /// allocation for the common case.
    dims: SmallVec<[usize; 6]>,
    /// Byte-offset multiplier per unit step along each dimension.
    /// Default: row-major (C-order) layout.
    strides: SmallVec<[usize; 6]>,
}
```

### 4.2 Constructors

```rust
impl TensorShape {
    /// Create a row-major shape from extents.
    pub fn row_major(dims: &[usize]) -> Self;

    /// Create a column-major shape from extents (Fortran order).
    pub fn col_major(dims: &[usize]) -> Self;

    /// Create a shape with explicit strides (e.g., for a non-contiguous view).
    /// Panics in debug mode if any stride is zero while the corresponding
    /// dimension is > 1.
    pub fn with_strides(dims: &[usize], strides: &[usize]) -> Self;
}
```

### 4.3 Key Methods

```rust
impl TensorShape {
    /// Total number of elements: product of all dims.
    pub fn numel(&self) -> usize;

    /// Number of dimensions.
    pub fn rank(&self) -> usize;

    /// Linear offset for a multi-index: sum_i index[i] * strides[i].
    pub fn offset(&self, index: &[usize]) -> usize;

    /// True if data is stored contiguously in row-major order.
    pub fn is_contiguous(&self) -> bool;

    /// Returns a new TensorShape with dimensions permuted by `perm`.
    /// Zero-copy: only strides are rearranged.
    pub fn permute(&self, perm: &[usize]) -> TensorShape;

    /// Returns the shape after a reshape to `new_dims`.
    /// Errors if the total element count differs or if the current
    /// layout is non-contiguous (reshape requires contiguous memory).
    pub fn reshape(&self, new_dims: &[usize]) -> TkResult<TensorShape>;

    /// Slice along one axis: returns the shape and data-pointer offset
    /// for the sub-tensor at `axis` in `start..end`.
    pub fn slice_axis(
        &self, axis: usize, start: usize, end: usize
    ) -> (TensorShape, usize);
}
```

### 4.4 Design Notes

- `TensorShape` is `Copy`-cheap enough for stack use. The internal `SmallVec<[usize; 6]>` avoids heap allocation for tensors up to rank 6, covering every tensor that appears in DMRG (rank-3 MPS tensors, rank-4 MPO tensors, rank-6 environment blocks).
- Strides enable **zero-copy transpose**: swapping `strides[i]` and `strides[j]` without touching data is how `MatRef::adjoint()` works without allocating.

---

## 5. `TensorStorage<T>` — Contiguous Buffer

```rust
/// Owned, contiguous, 1-D memory buffer.
/// Invariant: data.len() == shape.numel() for any DenseTensor using this storage.
pub struct TensorStorage<T: Scalar> {
    data: Vec<T>,
}

impl<T: Scalar> TensorStorage<T> {
    pub fn zeros(n: usize) -> Self;
    pub fn from_vec(data: Vec<T>) -> Self;
    pub fn as_slice(&self) -> &[T];
    pub fn as_mut_slice(&mut self) -> &mut [T];
    pub fn len(&self) -> usize;
}
```

`TensorStorage` has no shape knowledge; that lives exclusively in `TensorShape`. This strict separation means shape-manipulation operations never touch the data buffer.

---

## 6. `TensorCow<'a, T>` — Copy-on-Write Storage

```rust
/// Copy-on-Write wrapper for tensor storage.
/// Borrows a view when possible; materializes into owned storage on demand.
pub enum TensorCow<'a, T: Scalar> {
    /// Zero-copy view: points into an existing buffer (arena or heap).
    Borrowed(&'a TensorStorage<T>),
    /// Heap-allocated owned data.
    Owned(TensorStorage<T>),
}

impl<'a, T: Scalar> TensorCow<'a, T> {
    pub fn as_slice(&self) -> &[T];

    /// Clone data into owned storage if borrowed.
    /// Called when a GEMM kernel requires a contiguous input and the
    /// current view has non-unit strides from a permutation.
    pub fn into_owned(self) -> TensorStorage<T>;

    pub fn is_owned(&self) -> bool;
    pub fn is_borrowed(&self) -> bool;
}
```

**Rule:** Shape operations (permute, reshape, slice) always return `Borrowed`. Data is cloned into `Owned` only when strict contiguity is required (e.g., BLAS kernel input with non-unit strides). This minimizes copies in the critical DMRG contraction path.

---

## 7. `DenseTensor<T>` — Primary Dense Tensor

### 7.1 Definition

```rust
/// The primary N-dimensional dense tensor.
/// Shape metadata is always owned; storage is Copy-on-Write.
pub struct DenseTensor<T: Scalar> {
    pub shape: TensorShape,
    pub storage: TensorCow<'static, T>,  // 'static for owned; shorter for views
}
```

In practice, arena-allocated tensors use a shorter lifetime via `TempTensor<'a, T>`:

```rust
/// Convenience alias: a DenseTensor whose storage borrows from an arena.
/// The 'a lifetime is tied to the SweepArena's current allocation epoch.
pub type TempTensor<'a, T> = DenseTensor<T>;
// where storage is TensorCow::Borrowed(&'a TensorStorage<T>)
```

### 7.2 Core Methods

```rust
impl<T: Scalar> DenseTensor<T> {
    /// Allocate a zero-filled owned tensor.
    pub fn zeros(shape: TensorShape) -> Self;

    /// Create from a flat Vec with the given shape.
    /// Panics if vec.len() != shape.numel().
    pub fn from_vec(shape: TensorShape, data: Vec<T>) -> Self;

    /// Return a zero-copy transposed view by permuting strides.
    pub fn permute(&self, perm: &[usize]) -> DenseTensor<T>;  // Borrowed

    /// Reshape to new dims. Returns Err if non-contiguous or numel mismatch.
    pub fn reshape(&self, new_dims: &[usize]) -> TkResult<DenseTensor<T>>;

    /// Slice along one axis.
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> DenseTensor<T>;

    /// Materialize into heap-allocated owned storage.
    /// Must be called before SweepArena::reset() for any tensor
    /// whose data must survive past the current sweep step.
    pub fn into_owned(self) -> DenseTensor<T>;

    /// View as a 2-D matrix (row-major). Errors if rank != 2.
    pub fn as_mat_ref(&self) -> TkResult<MatRef<'_, T>>;
    pub fn as_mat_mut(&mut self) -> TkResult<MatMut<'_, T>>;

    pub fn as_slice(&self) -> &[T];
    pub fn as_mut_slice(&mut self) -> &mut [T];
    pub fn shape(&self) -> &TensorShape;
    pub fn numel(&self) -> usize;
    pub fn rank(&self) -> usize;
}
```

### 7.3 Ownership Discipline

Every DMRG step produces intermediate tensors (environment blocks, Krylov vectors, contraction results) that are temporary and one SVD output that must persist in the `MPS` struct. The invariant:

```
ALL intermediates within one sweep step  →  allocated from SweepArena
ONLY the final SVD result               →  .into_owned() before arena reset
```

The borrow checker enforces this statically. Any `TempTensor<'a>` that escapes the arena's lifetime `'a` is a compile error. Calling `.into_owned()` produces a `DenseTensor` with `'static` storage (heap-allocated), which may be stored anywhere.

See §9 for the exact data-flow sequence.

---

## 8. `MatRef<T>` and `MatMut<T>` — 2-D Matrix Views

### 8.1 Motivation

Hermitian conjugation (A†) is the dominant operation in quantum physics contractions. Eagerly conjugating a complex matrix before a GEMM call requires an O(N) memory pass that saturates memory bandwidth before the multiply-add pipeline begins. Instead, `MatRef` carries a `is_conjugated` boolean that is passed through to the BLAS/faer micro-kernel, where conjugation is fused into FMA instructions at zero additional cost.

### 8.2 Definitions

```rust
/// An immutable 2-D view into a contiguous-or-strided buffer.
/// Carries lazy conjugation semantics for zero-copy Hermitian transposes.
#[derive(Clone, Copy, Debug)]
pub struct MatRef<'a, T: Scalar> {
    pub data: &'a [T],
    pub rows: usize,
    pub cols: usize,
    /// Offset in elements to advance by one row.
    pub row_stride: isize,
    /// Offset in elements to advance by one column.
    pub col_stride: isize,
    /// If true, the backend treats each element as its complex conjugate
    /// during GEMM/SVD. For real T, this flag has no effect.
    pub is_conjugated: bool,
}

/// A mutable 2-D view. No conjugation flag — writes are always literal.
#[derive(Debug)]
pub struct MatMut<'a, T: Scalar> {
    pub data: &'a mut [T],
    pub rows: usize,
    pub cols: usize,
    pub row_stride: isize,
    pub col_stride: isize,
}
```

### 8.3 View Constructors

```rust
impl<'a, T: Scalar> MatRef<'a, T> {
    /// Row-major contiguous matrix (C layout).
    pub fn from_slice(data: &'a [T], rows: usize, cols: usize) -> Self;

    /// Column-major contiguous matrix (Fortran layout).
    pub fn from_slice_col_major(data: &'a [T], rows: usize, cols: usize) -> Self;

    /// Arbitrary strides.
    pub fn from_slice_with_strides(
        data: &'a [T],
        rows: usize, cols: usize,
        row_stride: isize, col_stride: isize,
    ) -> Self;
}
```

### 8.4 Adjoint and Conjugate

```rust
impl<'a, T: Scalar> MatRef<'a, T> {
    /// Returns a zero-copy view of A† (Hermitian conjugate).
    /// Swaps rows ↔ cols and row_stride ↔ col_stride.
    /// Flips is_conjugated.
    /// For real T: equivalent to transpose.
    #[inline(always)]
    pub fn adjoint(&self) -> MatRef<'a, T> {
        MatRef {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            is_conjugated: !self.is_conjugated,
        }
    }

    /// Returns a zero-copy conjugated view without transposing.
    #[inline(always)]
    pub fn conjugate(&self) -> MatRef<'a, T> {
        MatRef { is_conjugated: !self.is_conjugated, ..*self }
    }

    /// Returns a zero-copy transposed view (not conjugated).
    #[inline(always)]
    pub fn transpose(&self) -> MatRef<'a, T> {
        MatRef {
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            ..*self
        }
    }

    /// Element access by (row, col).
    pub fn get(&self, row: usize, col: usize) -> T;

    pub fn is_square(&self) -> bool { self.rows == self.cols }
}
```

### 8.5 Backend Contract

Every `LinAlgBackend::gemm` implementation in `tk-linalg` **must** correctly handle all four combinations of `is_conjugated` on the two input `MatRef`s:

| `a.is_conjugated` | `b.is_conjugated` | Operation |
|:-----------------:|:-----------------:|:----------|
| false | false | C = α·A·B + β·C |
| true | false | C = α·A*·B + β·C |
| false | true | C = α·A·B* + β·C |
| true | true | C = α·A*·B* + β·C |

For real `T` (`T::is_real() == true`), all four cases reduce to C = α·A·B + β·C. Backends may use `T::is_real()` to skip conjugation-flag checks in tight loops.

---

## 9. `SweepArena` — Arena Memory Management

### 9.1 Motivation

DMRG sweeps execute thousands of contraction-SVD-truncation cycles. Naive heap allocation per intermediate tensor causes severe fragmentation and `malloc`/`free` overhead. Arena allocation amortizes all allocations in one step down to a single pointer-bump; the entire arena is reclaimed in O(1) by resetting the pointer.

### 9.2 Definition

```rust
pub struct SweepArena {
    #[cfg(not(feature = "backend-cuda"))]
    inner: bumpalo::Bump,

    #[cfg(feature = "backend-cuda")]
    storage: ArenaStorage,
}

#[cfg(feature = "backend-cuda")]
pub enum ArenaStorage {
    /// Page-locked memory, DMA-capable for high-throughput host→GPU transfers.
    /// Used when PinnedMemoryTracker has remaining budget.
    Pinned(PinnedArena),
    /// Standard pageable heap. Fallback when pinned budget is exhausted.
    Pageable(bumpalo::Bump),
}
```

### 9.3 Interface

```rust
impl SweepArena {
    /// Construct with a pre-allocated capacity (bytes).
    pub fn with_capacity(bytes: usize) -> Self;

    /// Allocate a zero-filled temporary tensor in the arena.
    /// The returned tensor's storage lifetime is tied to 'a (this arena).
    pub fn alloc_tensor<'a, T: Scalar>(
        &'a self,
        shape: TensorShape,
    ) -> TempTensor<'a, T>;

    /// Allocate an uninitialized slice. Used for Krylov vectors
    /// and other byte buffers that will be fully overwritten.
    ///
    /// # Safety
    /// Caller must initialize all elements before reading.
    pub unsafe fn alloc_slice_uninit<'a, T: Scalar>(
        &'a self,
        len: usize,
    ) -> &'a mut [T];

    /// Reset the arena in O(1): reclaims all allocations made since
    /// the last reset (or construction). Any TempTensor<'a> referencing
    /// this arena must not be accessed after this call.
    ///
    /// The borrow checker statically enforces this: TempTensor<'a> cannot
    /// outlive the arena's current allocation epoch. This call ends the epoch.
    pub fn reset(&mut self);

    /// Current allocation usage in bytes.
    pub fn allocated_bytes(&self) -> usize;
}
```

### 9.4 Ownership Boundary

The following pseudocode shows exactly where `.into_owned()` must be called within a DMRG sweep step (detailed data flow is in the architecture document §9):

```rust
fn dmrg_step<T: Scalar>(
    arena: &mut SweepArena,
    mps: &mut MPS<T>,
    mpo: &MPO<T>,
    site: usize,
) {
    // 1. Allocate temporaries — all in the arena.
    let two_site = arena.alloc_tensor(two_site_shape(mps, site));
    let env_l    = arena.alloc_tensor(env_shape_left(mps, mpo, site));
    let env_r    = arena.alloc_tensor(env_shape_right(mps, mpo, site));

    // 2. Build effective Hamiltonian, run Lanczos — all temporary.
    let ground_state = lanczos_eigenstate(arena, &env_l, &env_r, mpo, &two_site);

    // 3. SVD to split two-site tensor → new A_L, S, A_R.
    let (al, sv, ar) = svd_truncate(arena, &ground_state, max_bond, cutoff);

    // 4. THE OWNERSHIP BOUNDARY: materialize before arena reset.
    let al_owned = al.into_owned();  // heap-allocated
    let ar_owned = ar.into_owned();  // heap-allocated

    // 5. Store persistent results.
    mps.set_site(site,     al_owned);
    mps.set_site(site + 1, ar_owned);
    // sv absorbed into the next step's initial state (also persisted).

    // 6. Reset: reclaims all arena memory in O(1).
    //    two_site, env_l, env_r, ground_state, al, sv, ar
    //    are all gone. al_owned and ar_owned live on the heap.
    arena.reset();
}
```

---

## 10. `PinnedMemoryTracker` (CUDA Feature)

Enabled only when `features = ["backend-cuda"]`.

### 10.1 Purpose

Host-to-GPU DMA transfers are only high-bandwidth when the source memory is page-locked (pinned). Pinning too much memory starves the OS page cache and can deadlock the system. `PinnedMemoryTracker` provides a global atomic budget to bound pinned allocation.

### 10.2 Definition

```rust
#[cfg(feature = "backend-cuda")]
pub struct PinnedMemoryTracker {
    /// Maximum allowed pinned allocation in bytes across all arenas.
    max_bytes: usize,
    /// Current pinned allocation, updated atomically (lock-free).
    current_bytes: AtomicUsize,
    /// Number of times a pinned allocation request fell back to pageable.
    /// Exposed in DMRGEngine stats for observability.
    fallback_count: AtomicUsize,
}
```

### 10.3 Interface

```rust
#[cfg(feature = "backend-cuda")]
impl PinnedMemoryTracker {
    /// Initialize the global tracker.
    /// Should be called once at program startup.
    /// On MPI nodes, `max_bytes` must be divided by the number of
    /// co-resident ranks before calling this function.
    pub fn init_global(max_bytes: usize);

    /// Access the process-global instance.
    pub fn global() -> &'static Self;

    /// Attempt to reserve `bytes` of pinned memory.
    /// Returns Ok(PinnedGuard) on success (budget decremented atomically).
    /// Returns Err(PinnedBudgetExhausted) when the budget is full,
    /// and records a structured telemetry event with the fallback counter.
    pub fn try_reserve(bytes: usize) -> Result<PinnedGuard, PinnedBudgetExhausted>;

    /// Number of pinned-allocation fallbacks since initialization.
    pub fn fallback_count() -> usize;
}

/// RAII guard: releases pinned budget when dropped.
#[cfg(feature = "backend-cuda")]
pub struct PinnedGuard {
    bytes: usize,
}

impl Drop for PinnedGuard {
    fn drop(&mut self) {
        PinnedMemoryTracker::global()
            .current_bytes
            .fetch_sub(self.bytes, Ordering::Release);
    }
}
```

### 10.4 MPI Interaction

On multi-GPU nodes with MPI, each rank shares the same physical NUMA domain and pinned-memory budget. The calling code (in `tk-dmft`) must query the number of co-resident MPI ranks and divide `max_bytes` accordingly before calling `PinnedMemoryTracker::init_global`. `tk-core` itself has no MPI dependency.

### 10.5 Telemetry

When `try_reserve` falls back to pageable allocation, it emits a structured telemetry event (not just a log line) so that the `DMRGEngine` stats struct can surface the fallback count to the user. Format:

```rust
// Emitted inside try_reserve on fallback:
tracing::event!(
    tracing::Level::WARN,
    kind = "pinned_memory_fallback",
    requested_bytes = bytes,
    current_bytes = self.current_bytes.load(Ordering::Relaxed),
    max_bytes = self.max_bytes,
    fallback_count = self.fallback_count.fetch_add(1, Ordering::Relaxed) + 1,
);
```

---

## 11. Error Types

```rust
/// Top-level error type for tk-core and, by re-export, the entire workspace.
#[derive(Debug, thiserror::Error)]
pub enum TkError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("reshape failed: {numel_src} elements cannot reshape to {dims_dst:?}")]
    ReshapeError { numel_src: usize, dims_dst: Vec<usize> },

    #[error("non-contiguous tensor: operation requires contiguous memory layout")]
    NonContiguous,

    #[error("rank error: expected rank {expected}, got rank {got}")]
    RankError { expected: usize, got: usize },

    #[error("index out of bounds: axis {axis}, index {index}, dim {dim}")]
    IndexOutOfBounds { axis: usize, index: usize, dim: usize },

    #[error("scalar type mismatch")]
    ScalarTypeMismatch,

    #[cfg(feature = "backend-cuda")]
    #[error("pinned memory budget exhausted")]
    PinnedBudgetExhausted,
}

pub type TkResult<T> = Result<T, TkError>;
```

---

## 12. Public API Surface (`lib.rs`)

```rust
// tk-core/src/lib.rs

pub mod scalar;
pub mod shape;
pub mod storage;
pub mod tensor;
pub mod matview;
pub mod arena;
pub mod error;

#[cfg(feature = "backend-cuda")]
pub mod pinned;

// Flat re-exports for ergonomic downstream use:
pub use scalar::{Scalar, C32, C64};
pub use shape::TensorShape;
pub use storage::TensorStorage;
pub use tensor::{DenseTensor, TempTensor};
pub use matview::{MatRef, MatMut};
pub use arena::SweepArena;
pub use error::{TkError, TkResult};

#[cfg(feature = "backend-cuda")]
pub use pinned::{PinnedMemoryTracker, PinnedGuard};
```

---

## 13. Feature Flags (tk-core–relevant)

| Flag | Effect in tk-core |
|:-----|:------------------|
| `backend-cuda` | Enables `ArenaStorage::Pinned` variant and the entire `pinned` module |
| `backend-oxiblas` | Adds `f128` to the `Scalar` implementations |

`tk-core` does not activate `backend-faer`, `backend-mkl`, `backend-openblas`, or `parallel` directly — those are `tk-linalg` concerns.

---

## 14. Testing Requirements

### 14.1 Unit Tests (within `tk-core`)

| Test | Description |
|:-----|:------------|
| `shape_row_major_strides` | Verify strides for a 3×4×5 row-major tensor |
| `shape_permute_strides` | Permute [2,0,1], verify strides rearranged, numel unchanged |
| `shape_reshape_ok` | 3×4×5 → 60 → 4×15, verify contiguity check |
| `shape_reshape_noncontiguous_err` | Permuted view → reshape → must return `NonContiguous` |
| `matref_adjoint_zero_copy` | Check rows, cols, strides swap; `is_conjugated` flipped; no allocation |
| `matref_adjoint_real_type` | Adjoint of a real `MatRef<f64>` — `is_conjugated` flips but `is_real()` noted |
| `tensorcow_borrowed_no_clone` | Shape ops on `Borrowed` variant don't clone data |
| `tensorcow_into_owned_clones` | `into_owned()` on `Borrowed` produces new heap allocation |
| `arena_reset_reclaims` | `allocated_bytes()` returns to ~0 after `reset()` |
| `arena_lifetime_compile_error` | (compile-fail test) TempTensor<'a> cannot escape past reset |
| `scalar_conj_complex` | `C64::conj()` produces correct imaginary sign flip |
| `scalar_is_real_f64` | `f64::is_real()` returns true |
| `scalar_is_real_c64` | `C64::is_real()` returns false |

### 14.2 Property-Based Tests

Use `proptest` with **bounded strategies** (never unbounded random ranks/sizes — they inflate CI runtime):

```rust
proptest! {
    #[test]
    fn offset_is_within_bounds(
        dims in prop::collection::vec(1usize..=8, 2..=6),
        // index within [0, dim) for each axis
    ) {
        let shape = TensorShape::row_major(&dims);
        // generate valid multi-index and verify offset < numel
    }

    #[test]
    fn permute_preserves_numel(
        dims in prop::collection::vec(1usize..=16, 2..=6),
        perm in /* valid permutation strategy */
    ) {
        let s1 = TensorShape::row_major(&dims);
        let s2 = s1.permute(&perm);
        assert_eq!(s1.numel(), s2.numel());
    }
}
```

### 14.3 Compile-Fail Tests

Use the `trybuild` crate to verify borrow-checker enforcement:

```rust
// tests/compile_fail/arena_escape.rs  (expected to fail compilation)
fn escape_temp_tensor() {
    let mut arena = SweepArena::with_capacity(1024);
    let t: TempTensor<f64> = arena.alloc_tensor(TensorShape::row_major(&[4, 4]));
    arena.reset();
    let _ = t.as_slice(); // use-after-reset: borrow checker must reject this
}
```

---

## 15. Performance Invariants

The following must be validated by CI benchmarks (Criterion, instruction-counting mode):

| Operation | Invariant |
|:----------|:----------|
| `TensorShape::permute` | Zero allocations (SmallVec stays on stack for rank ≤ 6) |
| `MatRef::adjoint` | Zero allocations, zero data movement |
| `TensorCow::as_slice` (Borrowed) | Zero allocations |
| `SweepArena::reset` | O(1) wall time independent of number of prior allocations |
| `SweepArena::alloc_tensor` | Single pointer-bump; no `malloc` system call |

---

## 16. Out of Scope

The following are explicitly **not** implemented in `tk-core`:

- Tensor addition, subtraction, or element-wise operations (→ `tk-linalg`)
- Tensor contraction or trace (→ `tk-contract`)
- Block-sparse formats or quantum numbers (→ `tk-symmetry`)
- BLAS/SVD/EVD dispatch (→ `tk-linalg`)
- Index types or operator enums (→ `tk-dsl`)
- Any `unsafe` block that is not strictly required for arena bump-allocation or pinned memory

---

## 17. Open Questions

| # | Question | Status |
|:--|:---------|:-------|
| 1 | Should `TensorShape` support rank-0 (scalar) tensors? | Deferred; current code panics if `dims` is empty |
| 2 | `f128` support via `backend-oxiblas`: does faer provide an `f128` GEMM path, or must it go through DeviceOxiblas exclusively? | Needs investigation before implementing `Scalar for f128` |
| 3 | Arena capacity growth policy: fixed `with_capacity`, or auto-doubling? `bumpalo::Bump` doubles internally; document expected initial size for a DMRG step at D=2000 | To be benchmarked in Phase 2 |
| 4 | NUMA-aware pinned allocation (`PinnedArena` binding to PCIe root NUMA node) | Deferred to Phase 5+ per architecture doc §10.2.6 |
