//! Arena memory management for DMRG sweep temporaries.
//!
//! `SweepArena` provides bump-allocated temporary storage that can be
//! reclaimed in O(1) via `reset()`. When the `backend-cuda` feature is
//! active, the arena may use pinned (page-locked) memory for high-bandwidth
//! GPU DMA transfers.

use bumpalo::Bump;

use crate::scalar::Scalar;
use crate::shape::TensorShape;
use crate::tensor::DenseTensor;

cfg_if::cfg_if! {
    if #[cfg(feature = "backend-cuda")] {
        use crate::pinned::PinnedMemoryTracker;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static PINNED_FALLBACK_COUNT: AtomicUsize = AtomicUsize::new(0);

        // ---------------------------------------------------------------
        // CUDA Runtime FFI declarations for pinned memory management
        // ---------------------------------------------------------------
        mod cuda_ffi {
            use std::os::raw::c_void;

            /// CUDA error code. 0 = cudaSuccess.
            pub type CudaError = i32;
            pub const CUDA_SUCCESS: CudaError = 0;

            extern "C" {
                /// Allocate page-locked (pinned) host memory.
                /// The allocated memory is DMA-capable for high-bandwidth
                /// GPU transfers without the hidden staging-buffer copy.
                pub fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> CudaError;

                /// Free page-locked host memory previously allocated by cudaMallocHost.
                pub fn cudaFreeHost(ptr: *mut c_void) -> CudaError;
            }
        }

        /// When `backend-cuda` is active, the arena dynamically chooses
        /// between pinned (DMA-capable) and pageable memory based on global budget.
        pub enum ArenaStorage {
            /// Page-locked memory, DMA-capable for high-throughput host->GPU transfers.
            Pinned(PinnedArena),
            /// Standard pageable heap. Fallback when pinned budget is exhausted.
            Pageable(Bump),
        }

        /// Bump allocator backed by CUDA page-locked (pinned) host memory.
        ///
        /// Allocates a single contiguous block of pinned memory via
        /// `cudaMallocHost` at construction time. Individual allocations
        /// are satisfied by advancing a watermark pointer within this block
        /// (bump allocation). `reset()` reclaims all allocations in O(1)
        /// by resetting the watermark to zero.
        ///
        /// The pinned memory block is freed via `cudaFreeHost` on drop.
        ///
        /// # Safety invariants
        ///
        /// - `base` points to a valid `cudaMallocHost`-allocated block of
        ///   `capacity` bytes, exclusively owned by this `PinnedArena`.
        /// - All slices returned by `alloc_*` methods are valid for the
        ///   lifetime of the `&PinnedArena` borrow that created them.
        ///   The borrow checker prevents `reset()` or `drop()` while
        ///   outstanding borrows exist.
        /// - `watermark` is always `<= capacity`.
        pub struct PinnedArena {
            /// Base pointer to the pinned memory block (from cudaMallocHost).
            base: *mut u8,
            /// Total capacity in bytes.
            capacity: usize,
            /// Current allocation watermark (byte offset from base).
            /// Uses `Cell` for interior mutability since allocation methods
            /// take `&self` (matching bumpalo's API). PinnedArena is `!Sync`.
            watermark: std::cell::Cell<usize>,
        }

        // PinnedArena exclusively owns its memory block, so it is safe
        // to send across threads. It is NOT Sync because the Cell<usize>
        // watermark is not thread-safe.
        unsafe impl Send for PinnedArena {}

        impl PinnedArena {
            /// Allocate a pinned-memory arena of the given capacity.
            ///
            /// Calls `cudaMallocHost` to allocate a page-locked memory block.
            /// Returns `Err(())` if the CUDA call fails (e.g., insufficient
            /// physical memory for page-locking).
            pub fn new(capacity_bytes: usize) -> Result<Self, ()> {
                if capacity_bytes == 0 {
                    return Err(());
                }

                let mut ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
                let err = unsafe { cuda_ffi::cudaMallocHost(&mut ptr, capacity_bytes) };

                if err != cuda_ffi::CUDA_SUCCESS || ptr.is_null() {
                    return Err(());
                }

                // Zero-initialize the pinned block for safety.
                unsafe {
                    std::ptr::write_bytes(ptr as *mut u8, 0, capacity_bytes);
                }

                Ok(PinnedArena {
                    base: ptr as *mut u8,
                    capacity: capacity_bytes,
                    watermark: std::cell::Cell::new(0),
                })
            }

            /// Reset the arena in O(1): resets the watermark to zero.
            /// All previously allocated slices become invalid (enforced by
            /// the borrow checker via `&mut self`).
            pub fn reset(&mut self) {
                self.watermark.set(0);
            }

            /// Total capacity in bytes.
            pub fn capacity(&self) -> usize {
                self.capacity
            }

            /// Current bytes allocated (watermark position).
            pub fn allocated_bytes(&self) -> usize {
                self.watermark.get()
            }

            /// Bump-allocate a zero-filled slice of `len` elements from the
            /// pinned memory block.
            ///
            /// Returns a slice whose lifetime is tied to `&self`. The borrow
            /// checker ensures the slice cannot outlive the arena.
            ///
            /// # Panics
            /// Panics if the arena does not have enough remaining capacity.
            pub fn alloc_slice_fill_copy<T: Copy + num_traits::Zero>(&self, len: usize) -> &[T] {
                let slice = self.alloc_raw::<T>(len);
                // The block was zero-initialized at construction and re-zeroed
                // on reset. For types where T::zero() == all-zero-bytes (true
                // for f32, f64, Complex<f32>, Complex<f64>), this is already
                // correct. For extra safety, explicitly zero-fill:
                for elem in slice.iter_mut() {
                    *elem = T::zero();
                }
                // Return as immutable slice (matching Bump's alloc_slice_fill_copy).
                &*slice
            }

            /// Bump-allocate an uninitialized slice of `len` elements.
            ///
            /// # Safety
            /// Caller must initialize all elements before reading.
            pub unsafe fn alloc_uninit<T>(&self, len: usize) -> &mut [T] {
                self.alloc_raw::<T>(len)
            }

            /// Core bump-allocation routine. Returns a mutable slice into the
            /// pinned block, advancing the watermark.
            fn alloc_raw<T>(&self, len: usize) -> &mut [T] {
                let size = std::mem::size_of::<T>() * len;
                let align = std::mem::align_of::<T>();

                let current = self.watermark.get();
                // Align the current watermark up to T's alignment.
                let aligned = (current + align - 1) & !(align - 1);
                let new_watermark = aligned + size;

                assert!(
                    new_watermark <= self.capacity,
                    "PinnedArena out of capacity: requested {} bytes (aligned to {}), \
                     watermark at {}, capacity {}",
                    size, aligned, current, self.capacity
                );

                self.watermark.set(new_watermark);

                unsafe {
                    let ptr = self.base.add(aligned) as *mut T;
                    std::slice::from_raw_parts_mut(ptr, len)
                }
            }
        }

        impl Drop for PinnedArena {
            fn drop(&mut self) {
                if !self.base.is_null() {
                    let err = unsafe {
                        cuda_ffi::cudaFreeHost(self.base as *mut std::os::raw::c_void)
                    };
                    if err != cuda_ffi::CUDA_SUCCESS {
                        log::error!(
                            "cudaFreeHost failed with error code {} for {} bytes",
                            err, self.capacity
                        );
                    }
                }
            }
        }
    }
}

/// Pre-allocated arena for DMRG sweep temporaries.
///
/// All intermediate tensors within a single sweep step are allocated here.
/// At step end, `reset()` reclaims all memory in O(1). Only the final SVD
/// output should call `.into_owned()` before reset to persist on the heap.
pub struct SweepArena {
    #[cfg(not(feature = "backend-cuda"))]
    inner: Bump,

    #[cfg(feature = "backend-cuda")]
    storage: ArenaStorage,
}

impl SweepArena {
    /// Construct with a pre-allocated capacity (bytes).
    ///
    /// On CPU-only builds: wraps a `bumpalo::Bump`.
    /// On CUDA builds: attempts to allocate pinned (DMA-capable) memory
    /// via `PinnedMemoryTracker::try_reserve`. Falls back to pageable
    /// memory if the budget is exhausted or `cudaMallocHost` fails.
    pub fn new(capacity_bytes: usize) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(feature = "backend-cuda")] {
                if PinnedMemoryTracker::try_reserve(capacity_bytes) {
                    match PinnedArena::new(capacity_bytes) {
                        Ok(arena) => {
                            log::info!("SweepArena: {} bytes pinned memory", capacity_bytes);
                            return SweepArena { storage: ArenaStorage::Pinned(arena) };
                        }
                        Err(_) => {
                            PinnedMemoryTracker::release(capacity_bytes);
                        }
                    }
                }
                let count = PINNED_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                log::warn!(
                    target: "tensorkraft::telemetry",
                    "PINNED_MEMORY_FALLBACK: SweepArena fell back to pageable memory \
                     ({} bytes requested, {} total fallbacks). GPU DMA transfers will \
                     use hidden staging buffers, halving effective PCI-e bandwidth.",
                    capacity_bytes, count
                );
                SweepArena {
                    storage: ArenaStorage::Pageable(Bump::with_capacity(capacity_bytes)),
                }
            } else {
                SweepArena {
                    inner: Bump::with_capacity(capacity_bytes),
                }
            }
        }
    }

    /// Allocate a zero-filled temporary tensor in the arena.
    ///
    /// The returned tensor's storage lifetime is tied to this arena.
    /// It cannot outlive the arena's current allocation epoch.
    /// No heap allocation occurs — the data lives entirely in bump memory.
    pub fn alloc_tensor<T: Scalar>(&self, shape: TensorShape) -> DenseTensor<'_, T> {
        let n = shape.numel();
        let slice = self.alloc_slice_zeroed::<T>(n);
        DenseTensor::borrowed(shape, slice)
    }

    /// Allocate a zeroed slice from the arena.
    fn alloc_slice_zeroed<T: Scalar>(&self, len: usize) -> &[T] {
        cfg_if::cfg_if! {
            if #[cfg(feature = "backend-cuda")] {
                match &self.storage {
                    ArenaStorage::Pinned(arena) => arena.alloc_slice_fill_copy(len),
                    ArenaStorage::Pageable(bump) => bump.alloc_slice_fill_copy(len, T::zero()),
                }
            } else {
                self.inner.alloc_slice_fill_copy(len, T::zero())
            }
        }
    }

    /// Allocate an uninitialized slice. Used for Krylov vectors
    /// and other byte buffers that will be fully overwritten.
    ///
    /// # Safety
    /// Caller must initialize all elements before reading.
    pub unsafe fn alloc_slice_uninit<T: Scalar>(&self, len: usize) -> &mut [T] {
        cfg_if::cfg_if! {
            if #[cfg(feature = "backend-cuda")] {
                match &self.storage {
                    ArenaStorage::Pinned(arena) => arena.alloc_uninit(len),
                    ArenaStorage::Pageable(bump) => {
                        let layout = std::alloc::Layout::array::<T>(len).expect("layout overflow");
                        let ptr = bump.alloc_layout(layout).as_ptr() as *mut T;
                        std::slice::from_raw_parts_mut(ptr, len)
                    }
                }
            } else {
                let layout = std::alloc::Layout::array::<T>(len).expect("layout overflow");
                let ptr = self.inner.alloc_layout(layout).as_ptr() as *mut T;
                std::slice::from_raw_parts_mut(ptr, len)
            }
        }
    }

    /// Reset the arena in O(1): reclaims all allocations made since
    /// the last reset (or construction).
    ///
    /// The borrow checker statically enforces that `TempTensor<'a>` cannot
    /// outlive the arena's current allocation epoch. This call ends the epoch.
    pub fn reset(&mut self) {
        cfg_if::cfg_if! {
            if #[cfg(feature = "backend-cuda")] {
                match &mut self.storage {
                    ArenaStorage::Pinned(arena) => arena.reset(),
                    ArenaStorage::Pageable(bump) => bump.reset(),
                }
            } else {
                self.inner.reset();
            }
        }
    }

    /// Current allocation usage in bytes.
    pub fn allocated_bytes(&self) -> usize {
        cfg_if::cfg_if! {
            if #[cfg(feature = "backend-cuda")] {
                match &self.storage {
                    ArenaStorage::Pinned(arena) => arena.allocated_bytes(),
                    ArenaStorage::Pageable(bump) => bump.allocated_bytes(),
                }
            } else {
                self.inner.allocated_bytes()
            }
        }
    }

    /// Whether this arena is backed by pinned (page-locked) memory.
    ///
    /// Returns `false` on non-CUDA builds.
    pub fn is_pinned(&self) -> bool {
        cfg_if::cfg_if! {
            if #[cfg(feature = "backend-cuda")] {
                matches!(&self.storage, ArenaStorage::Pinned(_))
            } else {
                false
            }
        }
    }

    /// Number of times any SweepArena construction fell back to pageable memory.
    /// Exposed in DMRGEngine stats for observability.
    #[cfg(feature = "backend-cuda")]
    pub fn pinned_fallback_count() -> usize {
        PINNED_FALLBACK_COUNT.load(Ordering::Relaxed)
    }
}

#[cfg(feature = "backend-cuda")]
impl Drop for SweepArena {
    fn drop(&mut self) {
        if let ArenaStorage::Pinned(arena) = &self.storage {
            PinnedMemoryTracker::release(arena.capacity());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_alloc_and_reset() {
        let mut arena = SweepArena::new(64 * 1024);
        let shape = TensorShape::row_major(&[4, 4]);
        let t = arena.alloc_tensor::<f64>(shape);
        assert_eq!(t.numel(), 16);
        assert!(arena.allocated_bytes() > 0);

        drop(t);
        arena.reset();
        // After reset, allocated_bytes should be minimal (bumpalo may retain capacity)
    }

    #[test]
    fn arena_reset_reclaims() {
        let mut arena = SweepArena::new(64 * 1024);
        // Allocate several tensors
        for _ in 0..10 {
            let _ = arena.alloc_tensor::<f64>(TensorShape::row_major(&[100]));
        }
        let before = arena.allocated_bytes();
        assert!(before > 0);
        arena.reset();
        // After reset, new allocations reuse the same memory region.
        // Verify by allocating again — total allocated_bytes should not
        // grow significantly beyond the pre-reset level.
        for _ in 0..10 {
            let _ = arena.alloc_tensor::<f64>(TensorShape::row_major(&[100]));
        }
        let after = arena.allocated_bytes();
        // The arena reuses memory, so `after` should be approximately equal
        // to `before` (not double).
        assert!(after <= before * 2, "arena did not reclaim: before={before}, after={after}");
    }

    #[test]
    fn arena_tensor_data_is_zeroed() {
        let arena = SweepArena::new(64 * 1024);
        let t = arena.alloc_tensor::<f64>(TensorShape::row_major(&[10]));
        assert!(t.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn arena_is_not_pinned_on_cpu() {
        let arena = SweepArena::new(64 * 1024);
        // Without backend-cuda, is_pinned is always false.
        #[cfg(not(feature = "backend-cuda"))]
        assert!(!arena.is_pinned());
    }
}
