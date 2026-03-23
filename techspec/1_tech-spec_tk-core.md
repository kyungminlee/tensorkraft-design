# Technical Specification: `tk-core`

**Crate:** `tensorkraft/tk-core`
**Version:** 0.2.0 (Post-Draft-Implementation)
**Status:** Draft
**Last Updated:** March 2026

---

## 1. Overview

`tk-core` is the leaf crate of the tensorkraft workspace. Every other crate depends on it, so it must remain stable, minimal, and free of mathematical logic. Its sole responsibility is:

- **Dimensional metadata** — shape/stride management for zero-copy tensor views
- **Memory management** — arena allocators (`SweepArena`), Copy-on-Write storage (`TensorStorage`), and optional pinned-memory budget tracking (`PinnedMemoryTracker`)
- **Matrix view types** — `MatRef`/`MatMut` carrying a lazy conjugation flag
- **Element-type abstraction** — the `Scalar` trait hierarchy
- **Shared error types** — `TkError` and subtypes used across the workspace

Mathematical operations on tensors (contraction, addition, trace, decomposition) are **not** implemented here. They belong in `tk-linalg` or `tk-contract`.

**Minimal dependencies** — `tk-core` is a pure-Rust leaf crate. It may depend only on:
- `smallvec` (stack-allocated small vectors)
- `bumpalo` (arena allocator)
- `num-complex`, `num-traits` (numeric type abstractions)
- `thiserror` (error derive macros)
- `cfg-if` (feature-flag conditional compilation)
- `log` (structured logging facade, used for pinned-memory fallback telemetry when `backend-cuda` is active)

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
    ├── storage.rs        TensorStorage<'a, T> (Owned/Borrowed CoW enum)
    ├── tensor.rs         DenseTensor<'a, T>, TempTensor<'a, T>
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
/// Implemented for: f32, f64, Complex<f32>, Complex<f64>.
/// f128 support is deferred pending Rust f128 stabilization.
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

    /// Construct a scalar from real and imaginary parts.
    /// For real types, panics (debug) or returns `re` (release) if `im != 0`.
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self;

    /// Returns `Some(i)` where `i` is the imaginary unit for complex types,
    /// or `None` for real types.
    /// Essential for constructing operators such as SpinOp::Sy (which
    /// contains factors of i) and Green's functions in tk-dmft.
    fn imaginary_unit() -> Option<Self>;

    /// Returns true iff complex conjugation is a no-op (i.e., T is real).
    /// Used by the contraction engine to skip conjugation-flag propagation
    /// in tight loops over real-valued models.
    fn is_real() -> bool;
}
```

**Implementation note:** The `Sub`, `Neg`, `Debug`, and `'static` bounds were added during draft implementation because downstream crates (`tk-contract`, `tk-dsl`, `tk-dmft`) require subtraction for Hamiltonian assembly, negation for sign-flip operations, `Debug` for error messages, and `'static` for storage in long-lived data structures. The `from_real_imag` constructor and `imaginary_unit` method were discovered as essential by `tk-dsl` (for `SpinOp::Sy` which involves `i * sigma_y`) and `tk-dmft` (for Green's function construction).

### 3.2 Implementations

| Type | `Real` | `is_real()` | Notes |
|:-----|:-------|:------------|:------|
| `f32` | `f32` | `true` | |
| `f64` | `f64` | `true` | Default for most DMRG runs |
| `Complex<f32>` | `f32` | `false` | |
| `Complex<f64>` | `f64` | `false` | Quantum impurity solver |

**Implementation note:** `f128` support is deferred pending Rust `f128` stabilization. The `backend-oxiblas` feature flag is reserved but the `Scalar` implementation for `f128` is not yet provided.

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
    /// For rank-0 tensors (empty dims), returns 1 (the mathematical
    /// convention for an empty product).
    pub fn numel(&self) -> usize;

    /// Number of dimensions.
    pub fn rank(&self) -> usize;

    /// Linear offset for a multi-index: sum_i index[i] * strides[i].
    pub fn offset(&self, index: &[usize]) -> usize;

    /// True if data is stored contiguously in row-major order.
    /// Note: column-major tensors return `false` even though they are
    /// physically contiguous in memory. This method checks specifically
    /// for row-major (C-order) contiguity. A rename to
    /// `is_row_major_contiguous()` is under consideration (see Open
    /// Questions).
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
- **Rank-0 edge case:** `numel()` returns 1 for empty dims (mathematically correct as the empty product). This currently works correctly but lacks dedicated test coverage.

---

## 5. `TensorStorage<'a, T>` — Copy-on-Write Contiguous Buffer

```rust
/// Contiguous memory buffer with Copy-on-Write semantics.
///
/// TensorStorage has no shape knowledge — that lives exclusively in
/// TensorShape. This strict separation means shape-manipulation
/// operations never touch the data buffer.
///
/// The original architecture document proposed a three-type design
/// (TensorStorage, TensorCow, DenseTensor). During implementation,
/// TensorCow was merged into TensorStorage as a single enum with
/// Owned/Borrowed variants, simplifying the type hierarchy without
/// losing any functionality.
pub enum TensorStorage<'a, T: Scalar> {
    /// Heap-allocated owned data. Mutable and freely movable.
    Owned(Vec<T>),
    /// Zero-copy view into an existing buffer (arena, another tensor, etc.).
    /// Immutable — must call `into_owned()` before mutation.
    Borrowed(&'a [T]),
}

impl<'a, T: Scalar> TensorStorage<'a, T> {
    pub fn zeros(n: usize) -> TensorStorage<'static, T>;
    pub fn from_vec(data: Vec<T>) -> TensorStorage<'static, T>;
    pub fn from_slice(data: &'a [T]) -> Self;
    pub fn as_slice(&self) -> &[T];
    pub fn as_mut_slice(&mut self) -> &mut [T];  // panics if Borrowed
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn into_owned(self) -> TensorStorage<'static, T>;
    pub fn into_owned_vec(self) -> Vec<T>;
    pub fn is_owned(&self) -> bool;
    pub fn is_borrowed(&self) -> bool;

    /// Create a Borrowed view of this storage, regardless of whether
    /// the underlying data is Owned or Borrowed. Key enabler for
    /// zero-copy permute/reshape/slice_axis chains: each operation
    /// calls borrow_storage() to produce a Borrowed variant pointing
    /// at the same data, then attaches new shape metadata.
    pub fn borrow_storage(&self) -> TensorStorage<'_, T>;
}
```

**Design:** `TensorStorage` unifies owned and borrowed storage into a single enum, eliminating the previous `TensorCow` wrapper from the architecture document's three-type design. This enables `SweepArena::alloc_tensor` to produce `Borrowed` views pointing directly at bump-allocated memory with no intermediate heap `Vec`.

**`borrow_storage()` pattern:** This method is the key enabler for zero-copy view chains. When `DenseTensor::permute()`, `reshape()`, or `slice_axis()` creates a new view, it calls `borrow_storage()` on the parent's storage to produce a `Borrowed` variant, then pairs it with the new `TensorShape`. This avoids cloning data even when the parent is `Owned`.

**Rule:** Shape operations (permute, reshape, slice) always produce `Borrowed` views. Data is cloned into `Owned` only when strict ownership is required (e.g., persisting past an arena reset, or mutating in-place). This minimizes copies in the critical DMRG contraction path.

---

## 6. `DenseTensor<'a, T>` — Primary Dense Tensor

### 6.1 Definition

```rust
/// The primary N-dimensional dense tensor.
/// Shape metadata is always owned; storage is Copy-on-Write.
///
/// The lifetime parameter 'a tracks the borrow lifetime of the
/// underlying storage. This is required so the borrow checker can
/// enforce arena safety — a DenseTensor borrowing from a SweepArena
/// cannot outlive the arena's current allocation epoch.
///
/// The `offset` field tracks where this tensor's data begins within the
/// underlying storage buffer (nonzero for sliced views created by
/// `slice_axis`).
pub struct DenseTensor<'a, T: Scalar> {
    shape: TensorShape,
    storage: TensorStorage<'a, T>,
    offset: usize,
}
```

**Implementation note:** The lifetime parameter `'a` on `DenseTensor` is essential. An earlier design used `DenseTensor<T>` without a lifetime, but this failed to encode the borrow relationship between arena-allocated tensors and the arena itself, allowing use-after-reset bugs to compile. The `offset` field was added to support zero-copy `slice_axis` — without it, slicing would require creating a new storage allocation containing only the sliced region.

Arena-allocated tensors use a shorter lifetime via `TempTensor<'a, T>`:

```rust
/// Convenience alias: a DenseTensor whose storage borrows from an arena.
/// The 'a lifetime is tied to the SweepArena's current allocation epoch.
pub type TempTensor<'a, T> = DenseTensor<'a, T>;
```

### 6.2 Core Methods

```rust
impl<'a, T: Scalar> DenseTensor<'a, T> {
    /// Allocate a zero-filled owned tensor.
    pub fn zeros(shape: TensorShape) -> DenseTensor<'static, T>;

    /// Create from a flat Vec with the given shape.
    /// Panics if vec.len() != shape.numel().
    pub fn from_vec(shape: TensorShape, data: Vec<T>) -> DenseTensor<'static, T>;

    /// Create a tensor that borrows from a slice.
    pub fn borrowed(shape: TensorShape, data: &'a [T]) -> Self;

    /// Return a zero-copy transposed view by permuting strides.
    /// Calls borrow_storage() internally to produce a Borrowed view.
    pub fn permute(&self, perm: &[usize]) -> DenseTensor<'_, T>;

    /// Reshape to new dims. Returns Err if non-contiguous or numel mismatch.
    /// Calls borrow_storage() internally to produce a Borrowed view.
    pub fn reshape(&self, new_dims: &[usize]) -> TkResult<DenseTensor<'_, T>>;

    /// Slice along one axis. Returns a zero-copy view with the offset
    /// advanced to the start of the sliced region.
    /// Calls borrow_storage() internally to produce a Borrowed view.
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> DenseTensor<'_, T>;

    /// Materialize into heap-allocated owned storage.
    /// Must be called before SweepArena::reset() for any tensor
    /// whose data must survive past the current sweep step.
    ///
    /// Three code paths depending on internal state:
    ///
    /// 1. **Zero-cost move** — Storage is Owned, strides are contiguous
    ///    (row-major), and offset is 0. The Vec is moved directly with
    ///    no allocation or copy.
    ///
    /// 2. **Memcpy** — Storage is contiguous but offset is nonzero
    ///    (e.g., from slice_axis). A single memcpy gathers the
    ///    contiguous sub-region into a new Vec.
    ///
    /// 3. **Gather** — Strides are non-contiguous (e.g., after permute).
    ///    Uses gather_elements() (see §6.4) to enumerate all logical
    ///    elements via row-major multi-index iteration, copying each
    ///    into a fresh Vec. Complexity: O(numel * rank).
    ///
    /// The owned copy is always contiguous, row-major, with offset 0.
    pub fn into_owned(self) -> DenseTensor<'static, T>;

    /// View as a 2-D matrix (row-major). Errors if rank != 2.
    pub fn as_mat_ref(&self) -> TkResult<MatRef<'_, T>>;
    pub fn as_mat_mut(&mut self) -> TkResult<MatMut<'_, T>>;

    /// Returns a slice of the underlying data, adjusted by offset.
    /// For a tensor with offset `k`, returns &storage[k..k+numel()]
    /// when contiguous. For non-contiguous tensors, the returned slice
    /// may contain elements that are not logically part of this tensor.
    pub fn as_slice(&self) -> &[T];
    /// Offset-adjusted mutable access. Panics if storage is Borrowed.
    pub fn as_mut_slice(&mut self) -> &mut [T];
    pub fn shape(&self) -> &TensorShape;
    pub fn offset(&self) -> usize;
    pub fn numel(&self) -> usize;
    pub fn rank(&self) -> usize;
}
```

### 6.3 Ownership Discipline

Every DMRG step produces intermediate tensors (environment blocks, Krylov vectors, contraction results) that are temporary and one SVD output that must persist in the `MPS` struct. The invariant:

```
ALL intermediates within one sweep step  ->  allocated from SweepArena
ONLY the final SVD result               ->  .into_owned() before arena reset
```

The borrow checker enforces this statically. Any `TempTensor<'a>` that escapes the arena's lifetime `'a` is a compile error. Calling `.into_owned()` produces a `DenseTensor` with `'static` storage (heap-allocated), which may be stored anywhere.

See §8.5 for the exact data-flow sequence.

### 6.4 `gather_elements()` — Non-Contiguous Materialization

```rust
impl<'a, T: Scalar> DenseTensor<'a, T> {
    /// Materializes a non-contiguous view into a contiguous Vec by
    /// iterating over all logical elements in row-major order.
    ///
    /// Algorithm: For each logical element, compute its multi-index
    /// via row-major enumeration, then compute the physical offset
    /// as sum_i(index[i] * strides[i]) + self.offset. Copy the
    /// element at that offset into the output Vec.
    ///
    /// Complexity: O(numel * rank) — for each of the numel elements,
    /// the multi-index update and dot product with strides is O(rank).
    ///
    /// This method is called internally by into_owned() when strides
    /// are non-contiguous (path 3 in §6.2).
    fn gather_elements(&self) -> Vec<T>;
}
```

### 6.5 Known Limitations

**`DenseTensor` does not implement `Clone`.** The `Borrowed` variant's lifetime makes a blanket `Clone` impl unsound without careful handling. This is a known limitation affecting downstream crates:

- `tk-contract` needs to clone intermediate tensors during contraction tree evaluation.
- `tk-dsl` needs to clone operator tensors during Hamiltonian assembly.
- `tk-dmrg` needs to clone MPS tensors for convergence checks.

**Recommended resolution:** Implement `Clone` for `DenseTensor<'static, T>` only (owned tensors), which sidesteps the lifetime issue. Borrowed tensors that need cloning should use `into_owned()` first.

```rust
impl<T: Scalar> Clone for DenseTensor<'static, T> {
    fn clone(&self) -> Self {
        // Storage is guaranteed Owned for 'static lifetime.
        // Clone the Vec and copy the shape.
        DenseTensor {
            shape: self.shape.clone(),
            storage: TensorStorage::Owned(self.storage.as_slice().to_vec()),
            offset: self.offset,
        }
    }
}
```

---

## 7. `MatRef<T>` and `MatMut<T>` — 2-D Matrix Views

### 7.1 Motivation

Hermitian conjugation (A†) is the dominant operation in quantum physics contractions. Eagerly conjugating a complex matrix before a GEMM call requires an O(N) memory pass that saturates memory bandwidth before the multiply-add pipeline begins. Instead, `MatRef` carries a `is_conjugated` boolean that is passed through to the BLAS/faer micro-kernel, where conjugation is fused into FMA instructions at zero additional cost.

### 7.2 Definitions

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

### 7.3 View Constructors

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

### 7.4 Adjoint and Conjugate

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

### 7.5 Backend Contract

Every `LinAlgBackend::gemm` implementation in `tk-linalg` **must** correctly handle all four combinations of `is_conjugated` on the two input `MatRef`s:

| `a.is_conjugated` | `b.is_conjugated` | Operation |
|:------------------|:------------------|:----------|
| false | false | C = α·A·B + β·C |
| true | false | C = α·A*·B + β·C |
| false | true | C = α·A·B* + β·C |
| true | true | C = α·A*·B* + β·C |

For real `T` (`T::is_real() == true`), all four cases reduce to C = α·A·B + β·C. Backends may use `T::is_real()` to skip conjugation-flag checks in tight loops.

---

## 8. `SweepArena` — Arena Memory Management

### 8.1 Motivation

DMRG sweeps execute thousands of contraction-SVD-truncation cycles. Naive heap allocation per intermediate tensor causes severe fragmentation and `malloc`/`free` overhead. Arena allocation amortizes all allocations in one step down to a single pointer-bump; the entire arena is reclaimed in O(1) by resetting the pointer.

### 8.2 Definition

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

**Implementation note:** `PinnedArena` is currently a placeholder using standard `bumpalo::Bump` internally. The real CUDA-backed pinned memory implementation is deferred to Phase 5. The type exists now to establish the correct API surface and feature-flag plumbing.

### 8.3 Interface

```rust
impl SweepArena {
    /// Construct with a pre-allocated capacity (bytes).
    ///
    /// On CPU-only builds: wraps a `bumpalo::Bump`.
    /// On CUDA builds: attempts to allocate pinned (DMA-capable) memory
    /// via `PinnedMemoryTracker::try_reserve`. Falls back to pageable
    /// memory if the budget is exhausted or `cudaMallocHost` fails.
    pub fn new(capacity_bytes: usize) -> Self;

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
    pub fn reset(&mut self) {
        #[cfg(not(feature = "backend-cuda"))]
        { self.inner.reset(); }
        #[cfg(feature = "backend-cuda")]
        {
            match &mut self.storage {
                ArenaStorage::Pinned(arena) => arena.reset(),
                ArenaStorage::Pageable(bump) => bump.reset(),
            }
        }
    }

    /// Current allocation usage in bytes.
    pub fn allocated_bytes(&self) -> usize;
}
```

### 8.4 CUDA Constructor and Drop

When `backend-cuda` is active, the constructor integrates with `PinnedMemoryTracker` to attempt pinned allocation with automatic pageable fallback:

```rust
#[cfg(feature = "backend-cuda")]
impl SweepArena {
    pub fn new(capacity_bytes: usize) -> Self {
        if PinnedMemoryTracker::try_reserve(capacity_bytes) {
            match PinnedArena::new(capacity_bytes) {
                Ok(arena) => {
                    log::info!("SweepArena: {} bytes pinned memory", capacity_bytes);
                    return SweepArena { storage: ArenaStorage::Pinned(arena) };
                }
                Err(_) => {
                    // cudaMallocHost failed despite budget check — release reservation.
                    PinnedMemoryTracker::release(capacity_bytes);
                }
            }
        }
        // Fallback: pageable memory with telemetry (see §9.5).
        let count = PINNED_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        log::warn!(
            target: "tensorkraft::telemetry",
            "PINNED_MEMORY_FALLBACK: SweepArena fell back to pageable memory \
             ({} bytes requested, {} total fallbacks). GPU DMA transfers will \
             use hidden staging buffers, halving effective PCI-e bandwidth.",
            capacity_bytes, count
        );
        SweepArena {
            storage: ArenaStorage::Pageable(bumpalo::Bump::with_capacity(capacity_bytes)),
        }
    }

    /// Number of times any SweepArena construction fell back to pageable memory.
    /// Exposed in DMRGEngine stats for observability.
    pub fn pinned_fallback_count() -> usize {
        PINNED_FALLBACK_COUNT.load(Ordering::Relaxed)
    }
}
```

The `Drop` implementation releases the pinned budget when a pinned arena is dropped:

```rust
#[cfg(feature = "backend-cuda")]
impl Drop for SweepArena {
    fn drop(&mut self) {
        if let ArenaStorage::Pinned(arena) = &self.storage {
            PinnedMemoryTracker::release(arena.capacity());
        }
    }
}
```

### 8.5 Ownership Boundary

The following pseudocode shows exactly where `.into_owned()` must be called within a DMRG sweep step (detailed data flow is in the architecture document §9; arena step 7):

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

## 9. `PinnedMemoryTracker` (CUDA Feature)

Enabled only when `features = ["backend-cuda"]`.

### 9.1 Purpose

Host-to-GPU DMA transfers are only high-bandwidth when the source memory is page-locked (pinned). Pinning too much memory starves the OS page cache and can deadlock the system. `PinnedMemoryTracker` provides a process-local atomic budget to bound pinned allocation.

### 9.2 Definition

The tracker uses **module-level static atomics** rather than an instance-based struct. This avoids lifetime issues with a global singleton and aligns with the process-local isolation semantics required by MPI (§9.4).

```rust
#[cfg(feature = "backend-cuda")]
use std::sync::atomic::{AtomicUsize, Ordering};

static PINNED_BYTES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PINNED_BYTES_LIMIT: AtomicUsize = AtomicUsize::new(0);
static PINNED_FALLBACK_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Unit struct providing static methods to manage the process-global
/// pinned-memory budget. All state lives in module-level atomics.
#[cfg(feature = "backend-cuda")]
pub struct PinnedMemoryTracker;
```

### 9.3 Interface

```rust
#[cfg(feature = "backend-cuda")]
impl PinnedMemoryTracker {
    /// Initialize the global pinned-memory budget.
    /// Should be called once at program startup.
    /// On MPI nodes, `max_bytes` must already be divided by the number of
    /// co-resident ranks before calling this function (see §9.4).
    pub fn initialize_budget(max_bytes: usize) {
        PINNED_BYTES_LIMIT.store(max_bytes, Ordering::Release);
    }

    /// Attempt to reserve `bytes` of pinned memory from the budget.
    /// Returns `true` on success (budget decremented atomically via CAS loop).
    /// Returns `false` when the budget would be exceeded.
    /// Callers are responsible for falling back to pageable allocation
    /// and incrementing the fallback counter on failure.
    pub fn try_reserve(bytes: usize) -> bool {
        let mut current = PINNED_BYTES_ALLOCATED.load(Ordering::Relaxed);
        loop {
            let limit = PINNED_BYTES_LIMIT.load(Ordering::Acquire);
            if current + bytes > limit { return false; }
            match PINNED_BYTES_ALLOCATED.compare_exchange_weak(
                current, current + bytes,
                Ordering::AcqRel, Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    /// Release `bytes` from the pinned budget.
    /// Called when a pinned arena is dropped or reset.
    pub fn release(bytes: usize) {
        PINNED_BYTES_ALLOCATED.fetch_sub(bytes, Ordering::Release);
    }
}
```

**Design note:** The architecture document specifies `try_reserve` returning `bool` (not `Result<PinnedGuard, ...>`) with explicit `release()` calls managed by `SweepArena`'s `Drop` implementation (§8.5). This avoids the need for a separate `PinnedGuard` RAII type and keeps the budget logic concentrated in `SweepArena`'s lifecycle methods.

### 9.4 MPI Process-Isolation Semantics

**Critical clarification:** Rust's `AtomicUsize` is strictly **process-local** — each MPI rank runs as an independent OS process with an isolated virtual memory space. Rank 0 cannot read Rank 1's `PINNED_BYTES_ALLOCATED`. The `PinnedMemoryTracker` does *not* coordinate dynamically across ranks at runtime.

Instead, the node-level budget is **statically partitioned once at startup** via the `initialize_dmft_node_budget` topology query (in `tk-dmft`), which divides the safe node limit evenly across co-resident ranks. Each rank then independently enforces its pre-negotiated slice using its own process-local atomic counter. This design is correct because pinned-memory allocation is monotonic within a DMFT iteration (allocate at start, release at end) — no dynamic rebalancing between ranks is needed.

```rust
// In tk-dmft (not tk-core — tk-core has no MPI dependency):
#[cfg(all(feature = "backend-cuda", feature = "backend-mpi"))]
pub fn initialize_dmft_node_budget(comm: &MpiComm) {
    let total_ram = sys_info::mem_info().unwrap().total;
    let local_ranks = comm.split_by_shared_memory().size();
    let safe_node_limit = (total_ram as f64 * 0.60) as usize;
    let rank_budget = safe_node_limit / local_ranks;
    PinnedMemoryTracker::initialize_budget(rank_budget);
}
```

### 9.5 Telemetry

When `SweepArena::new` falls back to pageable allocation, it emits a warning via the `log` crate with the `tensorkraft::telemetry` target so that the `DMRGEngine` stats struct can surface the fallback count to the user:

```rust
// Emitted inside SweepArena::new on pinned fallback (see §8.5):
let count = PINNED_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
log::warn!(
    target: "tensorkraft::telemetry",
    "PINNED_MEMORY_FALLBACK: SweepArena fell back to pageable memory \
     ({} bytes requested, {} total fallbacks). GPU DMA transfers will \
     use hidden staging buffers, halving effective PCI-e bandwidth.",
    capacity_bytes, count
);
```

---

## 10. Error Handling

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
}

pub type TkResult<T> = Result<T, TkError>;
```

### 10.1 Error Propagation Strategy

`TkError` is the root error type for the entire tensorkraft workspace. It is defined in `tk-core` and re-exported by all downstream crates via `pub use tk_core::{TkError, TkResult}`. Downstream crates that define their own error enums wrap `TkError` using `#[from]` conversions:

```rust
// Example: in a downstream crate such as tk-linalg
#[derive(Debug, thiserror::Error)]
pub enum LinAlgError {
    #[error(transparent)]
    Core(#[from] TkError),

    #[error("...")]
    SomeOtherVariant,
}
```

This ensures that any `TkError` produced within `tk-core` can be propagated through downstream crate boundaries using the `?` operator without manual conversion.

---

## 11. Public API Surface (`lib.rs`)

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
pub use pinned::PinnedMemoryTracker;
```

---

## 12. Feature Flags

| Flag | Effect in tk-core |
|:-----|:------------------|
| `backend-cuda` | Enables `ArenaStorage::Pinned` variant and the entire `pinned` module. Note: `PinnedArena` is currently a placeholder using standard `bumpalo`; real CUDA pinned memory is deferred to Phase 5. |
| `backend-oxiblas` | Reserved for `f128` `Scalar` implementation; currently unimplemented pending Rust `f128` stabilization |

`tk-core` does not activate `backend-faer`, `backend-mkl`, `backend-openblas`, or `parallel` directly — those are `tk-linalg` concerns.

---

## 13. Dependencies and Integration

### 13.1 `Cargo.toml` Sketch

```toml
[package]
name = "tk-core"
version = "0.1.0"
edition = "2021"

[dependencies]
smallvec = "1"
bumpalo = "3"
num-complex = "0.4"
num-traits = "0.2"
thiserror = "1"
cfg-if = "1"
log = "0.4"

[dev-dependencies]
proptest = "1"
trybuild = "1"

[features]
default = []
backend-cuda = []
backend-oxiblas = []
```

### 13.2 Upstream Dependencies

None. `tk-core` is a leaf crate with no workspace dependencies.

### 13.3 Downstream Consumers

The following workspace crates depend on `tk-core`:

- `tk-symmetry`
- `tk-linalg`
- `tk-contract`
- `tk-dsl`
- `tk-dmrg`
- `tk-dmft`
- `tk-python`

---

## 14. Testing Strategy

### 14.1 Unit Tests (within `tk-core`)

| Test | Description |
|:-----|:------------|
| `shape_row_major_strides` | Verify strides for a 3x4x5 row-major tensor |
| `shape_permute_strides` | Permute [2,0,1], verify strides rearranged, numel unchanged |
| `shape_reshape_ok` | 3x4x5 → 60 → 4x15, verify contiguity check |
| `shape_reshape_noncontiguous_err` | Permuted view → reshape → must return `NonContiguous` |
| `matref_adjoint_zero_copy` | Check rows, cols, strides swap; `is_conjugated` flipped; no allocation |
| `matref_adjoint_real_type` | Adjoint of a real `MatRef<f64>` — `is_conjugated` flips but `is_real()` noted |
| `storage_borrowed_no_clone_on_access` | Accessing `Borrowed` variant doesn't clone data |
| `storage_into_owned_clones_borrowed` | `into_owned()` on `Borrowed` produces new heap allocation |
| `storage_borrowed_mut_panics` | `as_mut_slice()` on `Borrowed` panics |
| `arena_reset_reclaims` | `allocated_bytes()` returns to ~0 after `reset()` |
| `matref_adjoint_roundtrip` | `mat.adjoint().adjoint()` recovers original strides and conjugation flag |
| `pinned_budget_enforcement` | `PinnedMemoryTracker::try_reserve` returns false when budget exceeded (cfg: backend-cuda) |
| `pinned_drop_releases_budget` | Dropping a pinned `SweepArena` releases budget via `PinnedMemoryTracker::release` (cfg: backend-cuda) |
| `scalar_conj_complex` | `C64::conj()` produces correct imaginary sign flip |
| `scalar_is_real_f64` | `f64::is_real()` returns true |
| `scalar_is_real_c64` | `C64::is_real()` returns false |

### 14.2 Property-Based Tests

Use `proptest` with **bounded strategies** (never unbounded random ranks/sizes — they inflate CI runtime). Six property tests are implemented on `TensorShape`:

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

    // Additional property tests:
    // - reshape_roundtrip: reshape to flat then back preserves numel
    // - row_major_is_contiguous: freshly constructed row-major shapes
    //   always report is_contiguous() == true
    // - permute_twice_roundtrip: permute then inverse-permute recovers
    //   original shape
    // - slice_axis_reduces_dim: slicing along an axis reduces that
    //   dimension's extent to (end - start)
}
```

### 14.3 Compile-Fail Tests

Use the `trybuild` crate to verify borrow-checker enforcement. Five compile-fail tests are implemented:

| Test | Description |
|:-----|:------------|
| `arena_tensor_outlives_reset` | `TempTensor<'a>` cannot be used after `arena.reset()` — borrow checker rejects |
| `arena_tensor_escape_scope` | `TempTensor<'a>` cannot escape the scope where the arena is borrowed |
| `borrowed_storage_outlives_data` | `TensorStorage::Borrowed` cannot outlive the slice it borrows from |
| `slice_view_outlives_tensor` | A `DenseTensor` view from `slice_axis` cannot outlive the parent tensor |
| `matref_outlives_tensor` | A `MatRef` obtained via `as_mat_ref()` cannot outlive the source `DenseTensor` |

```rust
// Example: tests/compile_fail/arena_tensor_outlives_reset.rs
fn escape_temp_tensor() {
    let mut arena = SweepArena::new(1024);
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
| `TensorStorage::as_slice` (Borrowed) | Zero allocations |
| `SweepArena::reset` | O(1) wall time independent of number of prior allocations |
| `SweepArena::alloc_tensor` | Single pointer-bump; no `malloc` system call |
| `DenseTensor::into_owned` (path 1) | Zero allocations, zero copies (move only) |
| `DenseTensor::borrow_storage` | Zero allocations |

---

## 16. Implementation Notes and Design Decisions

### Note 1 — TensorCow Merged into TensorStorage

The architecture document's three-type design (TensorStorage, TensorCow, DenseTensor) was simplified during implementation. `TensorCow` as a separate type added indirection without clear benefit; its `Owned`/`Borrowed` semantics mapped directly onto `TensorStorage`'s enum variants. Merging them reduces the number of types callers must understand and eliminates a layer of wrapping in the `DenseTensor` struct.

### Note 2 — Lifetime Parameter on DenseTensor

The original spec used `DenseTensor<T>` without a lifetime. Implementation revealed that the lifetime must be visible on the type (`DenseTensor<'a, T>`) so that the borrow checker can enforce arena safety. Without the lifetime parameter, there is no static guarantee that a tensor borrowing from a `SweepArena` cannot outlive the arena's current epoch.

### Note 3 — The `offset` Field

The `offset` field on `DenseTensor` was not in the original architecture document but proved essential for zero-copy `slice_axis`. Without it, slicing along an axis would require either unsafe pointer arithmetic or creating a new storage allocation containing only the sliced region. The offset approach keeps `slice_axis` zero-copy and composable with `permute` and `reshape`.

### Note 4 — `is_contiguous()` Semantics

The current `is_contiguous()` method checks specifically for row-major (C-order) contiguity. A column-major tensor reports `false` even though its data is physically contiguous in memory. This is intentional for the current implementation (DMRG workloads use row-major layout exclusively), but the name is misleading. See Open Questions for the rename discussion.

### Note 5 — PinnedArena Placeholder

`PinnedArena` currently wraps a standard `bumpalo::Bump` rather than actual CUDA pinned memory. This placeholder allows the `backend-cuda` feature-flag plumbing and `ArenaStorage` enum to be tested and validated before CUDA SDK integration in Phase 5. The API surface is designed to be stable across the transition.

---

## 17. Out of Scope

The following are explicitly **not** implemented in `tk-core`:

- Tensor addition, subtraction, or element-wise operations (-> `tk-linalg`)
- Tensor contraction or trace (-> `tk-contract`)
- Block-sparse formats or quantum numbers (-> `tk-symmetry`)
- BLAS/SVD/EVD dispatch (-> `tk-linalg`)
- Index types or operator enums (-> `tk-dsl`)
- `unsafe` blocks beyond those strictly required for arena bump-allocation or pinned memory (-> self-imposed constraint; all `unsafe` usage is reviewed and minimized within `tk-core`)
- Real CUDA pinned memory allocation (-> Phase 5)
- `f128` scalar support (-> deferred pending Rust `f128` stabilization)

---

## 18. Open Questions

| # | Question | Status |
|:--|:---------|:-------|
| 1 | Should `TensorShape` support rank-0 (scalar) tensors? `numel()` currently returns 1 for empty dims (mathematically correct), but this behavior is untested. | Open — needs dedicated test coverage |
| 2 | `f128` support via `backend-oxiblas`: does faer provide an `f128` GEMM path, or must it go through DeviceOxiblas exclusively? | Deferred — blocked on Rust `f128` stabilization |
| 3 | Arena capacity growth policy: fixed `with_capacity`, or auto-doubling? `bumpalo::Bump` doubles internally; document expected initial size for a DMRG step at D=2000 | Deferred — to be benchmarked in Phase 2 |
| 4 | NUMA-aware pinned allocation (`PinnedArena` binding to PCIe root NUMA node) | Deferred — deferred to Phase 5+ per architecture doc §10.2.6 |
| 5 | Should `is_contiguous()` be renamed to `is_row_major_contiguous()`? Current name is misleading for column-major tensors. | Open — rename would be a breaking API change; consider adding `is_col_major_contiguous()` as an alternative |
| 6 | `DenseTensor` `Clone` implementation: should `Clone` be implemented for `DenseTensor<'static, T>` only, or should a more general approach (e.g., `to_owned_clone()` method) be used? | Open — downstream crates (`tk-contract`, `tk-dsl`, `tk-dmrg`) all need this capability |

---

## 19. Future Considerations

- **`StorageDevice` trait (Phase 5):** The architecture document §10.1 introduces a `StorageDevice` trait that generalizes `TensorStorage` to support GPU and MPI device memory. A `StorageDevice` trait with associated `Allocator` type would define `HostDevice`, `CudaDevice`, and `MpiDevice` implementations. `TensorStorage` would gain a default type parameter `D: StorageDevice = HostDevice` for backward compatibility. In Phases 1–3, `TensorStorage<'a, T>` remains the `Owned(Vec<T>)` / `Borrowed(&'a [T])` enum defined in §5. The `StorageDevice` trait and `HostDevice` would be added to `tk-core`; `CudaDevice` and `MpiDevice` implementations would live in their respective backend crates.

    ```rust
    pub trait StorageDevice: Send + Sync + 'static {
        type Alloc: Allocator;
        fn alloc<T: Scalar>(len: usize) -> DeviceBuffer<T, Self>;
        fn synchronize(&self);
    }

    pub struct HostDevice;

    #[cfg(feature = "backend-cuda")]
    pub struct CudaDevice { pub ordinal: usize }

    #[cfg(feature = "backend-mpi")]
    pub struct MpiDevice { pub comm: MpiComm, pub rank: usize }

    /// Default type parameter preserves backward compatibility.
    pub struct TensorStorage<T: Scalar, D: StorageDevice = HostDevice> {
        data: DeviceBuffer<T, D>,
        device: D,
    }
    ```

- **Cross-crate arena usage via `flatten()` (Phase 4):** `SweepArena` is used not only for `alloc_tensor` within DMRG sweep steps, but also by `tk-symmetry`'s `BlockSparseTensor::flatten()` method (architecture document §4.2). The `flatten()` method accepts a `&SweepArena` parameter and packs fragmented block data into a single contiguous buffer allocated from the arena. When `backend-cuda` is active, the arena's pinned memory ensures that the flat buffer is DMA-capable for GPU transfers without the NVIDIA driver's hidden pin-copy-unpin staging dance. This cross-crate usage pattern does not change `SweepArena`'s API but motivates the `alloc_slice_uninit` method and confirms that the arena must support arbitrary-size allocations beyond just `TensorShape`-sized tensors.

    ```rust
    // In tk-symmetry (not tk-core), but depends on SweepArena from tk-core:
    impl<T: Scalar, Q: BitPackable> BlockSparseTensor<T, Q> {
        pub fn flatten<'a>(&self, arena: &'a SweepArena) -> FlatBlockStorage<'a, T>;
    }
    ```
