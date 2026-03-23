//! MPI Mode B integration for multi-rank DMFT.
//!
//! Provides `initialize_dmft_node_budget` which divides the node's
//! pinned-memory budget across co-resident MPI ranks. Feature-gated
//! behind `backend-mpi`.

// The actual implementation requires the `mpi` and `sys-info` crates,
// which are gated behind the `backend-mpi` feature flag. Since this
// feature is currently commented out in Cargo.toml (no MPI dependency
// available in the development environment), we provide a stub module.

/// Placeholder for MPI Mode B budget initialization.
///
/// When `backend-mpi` is enabled, this function:
/// 1. Queries system RAM via `sys_info::mem_info()`
/// 2. Splits the MPI communicator by shared memory
/// 3. Divides 60% of total RAM by the number of local ranks
/// 4. Sets each rank's `PinnedMemoryTracker` budget
///
/// Must be called before any `SweepArena` construction.
/// Must be called by all ranks simultaneously (collective operation).
///
/// See design doc Section 10.2.2 for the full implementation.
pub fn initialize_dmft_node_budget_stub() {
    // Actual implementation:
    //
    // #[cfg(all(feature = "backend-cuda", feature = "backend-mpi"))]
    // pub fn initialize_dmft_node_budget(comm: &MpiComm) {
    //     let total_ram = sys_info::mem_info().unwrap().total as usize * 1024;
    //     let local_comm = comm.split_by_shared_memory();
    //     let local_ranks = local_comm.size();
    //     let safe_node_limit = (total_ram as f64 * 0.60) as usize;
    //     let rank_budget = safe_node_limit / local_ranks;
    //     PinnedMemoryTracker::initialize_budget(rank_budget);
    // }
}
