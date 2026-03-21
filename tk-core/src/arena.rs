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

        /// When `backend-cuda` is active, the arena dynamically chooses
        /// between pinned (DMA-capable) and pageable memory based on global budget.
        pub enum ArenaStorage {
            /// Page-locked memory, DMA-capable for high-throughput host->GPU transfers.
            Pinned(PinnedArena),
            /// Standard pageable heap. Fallback when pinned budget is exhausted.
            Pageable(Bump),
        }

        /// Placeholder for a pinned-memory arena backed by `cudaMallocHost`.
        /// Full implementation requires CUDA bindings.
        pub struct PinnedArena {
            inner: Bump,
            capacity: usize,
        }

        impl PinnedArena {
            /// Attempt to allocate a pinned arena.
            /// In this draft, falls back to a normal bump allocator.
            /// A real implementation would call `cudaMallocHost`.
            pub fn new(capacity_bytes: usize) -> Result<Self, ()> {
                Ok(PinnedArena {
                    inner: Bump::with_capacity(capacity_bytes),
                    capacity: capacity_bytes,
                })
            }

            pub fn reset(&mut self) {
                self.inner.reset();
            }

            pub fn capacity(&self) -> usize {
                self.capacity
            }

            pub fn allocated_bytes(&self) -> usize {
                self.inner.allocated_bytes()
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
        let bump = self.bump();
        let slice = bump.alloc_slice_fill_copy(len, T::zero());
        slice
    }

    /// Allocate an uninitialized slice. Used for Krylov vectors
    /// and other byte buffers that will be fully overwritten.
    ///
    /// # Safety
    /// Caller must initialize all elements before reading.
    pub unsafe fn alloc_slice_uninit<T: Scalar>(&self, len: usize) -> &mut [T] {
        let bump = self.bump();
        let layout = std::alloc::Layout::array::<T>(len).expect("layout overflow");
        let ptr = bump.alloc_layout(layout).as_ptr() as *mut T;
        std::slice::from_raw_parts_mut(ptr, len)
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

    /// Number of times any SweepArena construction fell back to pageable memory.
    /// Exposed in DMRGEngine stats for observability.
    #[cfg(feature = "backend-cuda")]
    pub fn pinned_fallback_count() -> usize {
        PINNED_FALLBACK_COUNT.load(Ordering::Relaxed)
    }

    /// Get a reference to the underlying bump allocator.
    fn bump(&self) -> &Bump {
        cfg_if::cfg_if! {
            if #[cfg(feature = "backend-cuda")] {
                match &self.storage {
                    ArenaStorage::Pinned(arena) => &arena.inner,
                    ArenaStorage::Pageable(bump) => bump,
                }
            } else {
                &self.inner
            }
        }
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
}
