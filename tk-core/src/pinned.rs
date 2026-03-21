//! Pinned-memory budget tracking for CUDA GPU DMA transfers.
//!
//! Enabled only when `features = ["backend-cuda"]`.
//!
//! The tracker uses module-level static atomics rather than an instance-based
//! struct, aligning with the process-local isolation semantics required by MPI.
//! Each MPI rank runs as an independent OS process; cross-rank coordination
//! occurs only once at startup via budget partitioning in `tk-dmft`.

use std::sync::atomic::{AtomicUsize, Ordering};

static PINNED_BYTES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PINNED_BYTES_LIMIT: AtomicUsize = AtomicUsize::new(0);

/// Unit struct providing static methods to manage the process-global
/// pinned-memory budget. All state lives in module-level atomics.
pub struct PinnedMemoryTracker;

impl PinnedMemoryTracker {
    /// Initialize the global pinned-memory budget.
    ///
    /// Should be called once at program startup. On MPI nodes, `max_bytes`
    /// must already be divided by the number of co-resident ranks before
    /// calling this function.
    pub fn initialize_budget(max_bytes: usize) {
        PINNED_BYTES_LIMIT.store(max_bytes, Ordering::Release);
    }

    /// Attempt to reserve `bytes` of pinned memory from the budget.
    ///
    /// Returns `true` on success (budget decremented atomically via CAS loop).
    /// Returns `false` when the budget would be exceeded.
    /// Callers are responsible for falling back to pageable allocation
    /// and incrementing the fallback counter on failure.
    pub fn try_reserve(bytes: usize) -> bool {
        let mut current = PINNED_BYTES_ALLOCATED.load(Ordering::Relaxed);
        loop {
            let limit = PINNED_BYTES_LIMIT.load(Ordering::Acquire);
            if current + bytes > limit {
                return false;
            }
            match PINNED_BYTES_ALLOCATED.compare_exchange_weak(
                current,
                current + bytes,
                Ordering::AcqRel,
                Ordering::Relaxed,
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

    /// Current pinned bytes allocated (for diagnostics).
    pub fn allocated_bytes() -> usize {
        PINNED_BYTES_ALLOCATED.load(Ordering::Relaxed)
    }

    /// Current budget limit.
    pub fn budget_limit() -> usize {
        PINNED_BYTES_LIMIT.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_budget_enforcement() {
        // Reset state for test isolation
        PINNED_BYTES_ALLOCATED.store(0, Ordering::Release);
        PinnedMemoryTracker::initialize_budget(1000);

        assert!(PinnedMemoryTracker::try_reserve(500));
        assert_eq!(PinnedMemoryTracker::allocated_bytes(), 500);

        assert!(PinnedMemoryTracker::try_reserve(400));
        assert_eq!(PinnedMemoryTracker::allocated_bytes(), 900);

        // This should fail — would exceed budget
        assert!(!PinnedMemoryTracker::try_reserve(200));
        assert_eq!(PinnedMemoryTracker::allocated_bytes(), 900);

        // Release and try again
        PinnedMemoryTracker::release(500);
        assert_eq!(PinnedMemoryTracker::allocated_bytes(), 400);
        assert!(PinnedMemoryTracker::try_reserve(200));

        // Cleanup
        PinnedMemoryTracker::release(600);
    }

    #[test]
    fn pinned_zero_budget_rejects_all() {
        PINNED_BYTES_ALLOCATED.store(0, Ordering::Release);
        PinnedMemoryTracker::initialize_budget(0);
        assert!(!PinnedMemoryTracker::try_reserve(1));
    }
}
