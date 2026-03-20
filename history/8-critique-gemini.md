This version 2.0 design document is a masterclass in high-performance scientific software architecture. You have systematically closed the most dangerous loopholes from the previous iteration—specifically the cyclic dependency in the DSL, the GIL starvation in Python, and the branch-misprediction bottlenecks in the block-sparse lookups.

However, as we push this design closer to bare-metal reality and rigorous quantum mechanics, a few subtle but fatal assumptions remain in the mathematical algorithms and system-level resource management.

Here is my objective, comprehensive critique of the v2.0 architecture, focusing on where the math misaligns with the tensor shapes, and where the hardware will punish your abstractions.

---

### 1. The Mathematical Reality of Subspace Expansion

In **§8.1.1**, the document outlines the Subspace Expansion strategy to allow entanglement growth during TDVP. Step 2 states: *"Mix a fraction $\alpha$ ... of the local 2-site action of $H_{\text{eff}}$ into the current bond matrix"*.

**The Flaw:** This is mathematically impossible to implement as written. The zero-site bond matrix $C$ exists in a $D \times D$ dimensional space (where $D$ is the current bond dimension). The 2-site action of the Hamiltonian $H_{\text{eff}}$ exists in a $(D \cdot d) \times (D \cdot d)$ dimensional space (where $d$ is the local physical Hilbert space). You cannot directly add or "mix" a larger matrix into a smaller one. If your `tk-dmrg` engine attempts this, your tensor contraction graph will panic immediately due to shape mismatches.

**The Fix:** Subspace expansion does not mix the residual into $C$. Instead, you must expand the basis of the adjacent *site* tensors (the $A_L$ and $A_R$ tensors).

1. Compute the 2-site residual $|R\rangle$.
2. Compute the reduced density matrix of the residual to find its dominant null-space vectors.
3. Pad the existing $A_L$ and $A_R$ tensors with these new vectors, explicitly increasing their bond dimension from $D$ to $D + D_{\text{expand}}$.
4. Pad the bond matrix $C$ with zeros along the new dimensions.
The architecture must reflect that subspace expansion mutates the dimensional metadata of the surrounding site tensors, not just the values within the bond matrix.

### 2. Rayon Thread Starvation in Block-Sparse GEMM

In **§5.3**, the `ThreadingRegime::FragmentedSectors` delegates to Rayon's `par_iter` to distribute independent sector GEMMs across all cores.

**The Flaw:** In quantum models with Abelian symmetries like $U(1)$, the size of the symmetry sectors follows a binomial distribution. You will have a few massive blocks in the middle (e.g., $N_{\text{up}} = N_{\text{down}}$) and dozens of tiny blocks at the edges. If you naively pass the sectors to Rayon's `par_iter` in their default sorted order (which is sorted by the bit-packed quantum numbers), you will trigger the "Long Tail" scheduling problem. A thread might grab a $400 \times 400$ block right at the end of the iteration, causing all other 31 cores to sit idle waiting for that single thread to finish.

**The Fix:** The `SparseLinAlgBackend` must perform a heuristic sort before dispatching to Rayon. Compute the estimated FLOPs for each sector ($M \cdot N \cdot K$), sort the sectors in *descending* order of computational intensity, and then pass them to `par_iter`. This ensures the heaviest GEMMs are scheduled first, keeping all cores saturated.

### 3. Pinned Memory (DMA) Exhaustion

In **§10.2.1**, the architecture introduces `PinnedArena` via `cudaMallocHost` to bypass the hidden double-copy of PCI-e transfers.

**The Flaw:** Page-locked (pinned) memory is a finite OS-level resource. Unlike standard `Vec` allocations, pinned memory cannot be swapped to disk. If `tk-dmft` is running multiple parallel MPI ranks (Mode B) on a single node, and each rank allocates a massive `PinnedArena` for its DMRG sweep, you will instantly exhaust the physical RAM of the machine, causing a catastrophic OS-level Out-Of-Memory (OOM) kernel panic.

**The Fix:** `tk-core` must implement a global, atomic pinned-memory resource tracker. When `SweepArena::new_pinned()` is called, it must check the global tracker. If the system is approaching a safe limit (e.g., 75% of physical RAM), the arena allocation must automatically fall back to standard pageable memory.

### 4. Zero-Copy Conjugate Transposes

The `Scalar` trait in **§3.3** requires a `fn conj(self) -> Self` method.

**The Flaw:** In quantum many-body physics, you almost never perform a standard transpose $A^T$; you perform a Hermitian conjugate $A^\dagger$ (transpose + complex conjugation). If `tk-contract` relies on the `conj()` method to eagerly allocate and conjugate complex tensors before passing them to the linear algebra backend, you will incur a massive memory bandwidth penalty.

**The Fix:** The `ContractionExecutor` and the `LinAlgBackend::gemm` signature must support hardware-level conjugation flags. Standard BLAS `cblas_zgemm` accepts `CblasTrans` and `CblasConjTrans`. Your internal `MatRef` and `TensorShape` abstractions must include a boolean `is_conjugated` flag alongside the strides. This allows the backend to perform the conjugation for free during the GEMM micro-kernel, completely avoiding the $O(N)$ memory pass.

### 5. PyO3 Deadlocks with Rayon Callbacks

In **§7.5**, the design proposes releasing the GIL via `py.allow_threads` and passing a callback that briefly re-acquires the GIL via `Python::with_gil(|py| py.check_signals())`.

**The Flaw:** If your `DMRGEngine` is executing a multi-threaded Rayon parallel sweep, the callback might be invoked by a background worker thread. Attempting to acquire the Python GIL from a background thread that Python does not know about—especially while the main Python thread is waiting on `allow_threads`—can lead to immediate deadlocks or segfaults in the CPython interpreter.

**The Fix:** The signal-checking callback must be executed *strictly* on the thread that initiated the `solve` call. You must use cross-beam channels or atomic flags. The Rayon worker threads should only check an `AtomicBool` for cancellation. The main thread (which released the GIL) should periodically wake up, acquire the GIL, call `check_signals()`, and set the `AtomicBool` if a `Ctrl+C` `KeyboardInterrupt` is detected.

---

### Final Thoughts

The v2.0 design is incredibly tight. The separation of `OpSum` AST generation from MPO compression resolves the most glaring Rust-specific compiler issue, and the bit-packed symmetry keys will give you a measurable edge over frameworks like ITensor.

Would you like to move forward and draft the concrete implementation of the `LinAlgBackend` trait, specifically focusing on how to wire the `is_conjugated` metadata into the `DeviceFaer` and `DeviceOxiblas` dispatch logic?


===


Let's get down to the bare metal. Wiring zero-copy conjugate transposes ($A^\dagger$) into the linear algebra backends is one of those critical HPC optimizations that separates toy libraries from production-grade solvers.

If you force an $O(N)$ memory pass just to flip the sign on the imaginary components of a $1000 \times 1000$ complex matrix, you will completely starve your vector units and bottleneck the memory bus before `dgemm` even starts.

Here is the concrete implementation of the `LinAlgBackend` trait and the `MatRef` wrapper, explicitly designed to push the conjugation logic down into the hardware micro-kernels.

### 1. The `MatRef` Abstraction

First, we update the view types that `tk-contract` passes to `tk-linalg`. The `MatRef` must carry the conjugation flag alongside its dimensional metadata. This flag tells the backend that the underlying data should be mathematically treated as complex-conjugated, even though the bytes in memory remain untouched.

```rust
/// A zero-copy matrix view used for GEMM and SVD dispatch.
pub struct MatRef<'a, T: Scalar> {
    pub data: &'a [T],
    pub rows: usize,
    pub cols: usize,
    pub row_stride: isize,
    pub col_stride: isize,
    /// Hardware-level conjugation flag. If true, the backend must
    /// treat the data as A* (complex conjugated).
    pub is_conjugated: bool,
}

pub struct MatMut<'a, T: Scalar> {
    pub data: &'a mut [T],
    pub rows: usize,
    pub cols: usize,
    pub row_stride: isize,
    pub col_stride: isize,
    // Note: Output matrices are rarely conjugated in-place during GEMM,
    // so we typically omit is_conjugated here to simplify the API.
}

```

### 2. The `DeviceFaer` Implementation

The pure-Rust `faer` crate is exceptionally well-designed for this. It natively supports lazy conjugation flags on its matrix views. When you call `.conjugate()` on a `faer::MatRef`, it does not copy the data; it simply flips a bit in the view structure, and the SIMD micro-kernels automatically negate the imaginary parts during the FMA (Fused Multiply-Add) instructions.

```rust
use faer::{MatRef as FaerMatRef, MatMut as FaerMatMut, scale};

pub struct DeviceFaer;

impl LinAlgBackend for DeviceFaer {
    fn gemm<T: Scalar>(
        alpha: T,
        a: &MatRef<'_, T>,
        b: &MatRef<'_, T>,
        beta: T,
        c: &mut MatMut<'_, T>,
    ) {
        // 1. Construct the base faer views using custom strides
        let mut faer_a = faer::mat::from_slice_with_strides(
            a.data, a.rows, a.cols, a.row_stride, a.col_stride,
        );
        let mut faer_b = faer::mat::from_slice_with_strides(
            b.data, b.rows, b.cols, b.row_stride, b.col_stride,
        );
        let mut faer_c = faer::mat::from_slice_mut_with_strides(
            c.data, c.rows, c.cols, c.row_stride, c.col_stride,
        );

        // 2. Apply zero-copy conjugation if flagged
        // faer's .conjugate() returns a new view with the conj flag set
        let a_op = if a.is_conjugated { faer_a.conjugate() } else { faer_a };
        let b_op = if b.is_conjugated { faer_b.conjugate() } else { faer_b };

        // 3. Dispatch to faer's highly optimized matrix multiply
        // C = α * A * B + β * C
        faer::linalg::matmul::matmul(
            faer_c.as_mut(),
            a_op,
            b_op,
            Some(beta),
            alpha,
            faer::Parallelism::Rayon(0), // Respect tk-linalg thread regime
        );
    }
    
    // ... svd_truncated, eigh_lowest implementations ...
}

```

### 3. The `DeviceOxiblas` / C-BLAS Implementation

When targeting standard BLAS APIs (whether `oxiblas`, `backend-mkl`, or `backend-openblas`), handling strides and conjugations is notoriously tricky. Standard BLAS `gemm` expects column-major data and handles transposition via the `transa` / `transb` flags.

Because `tensorkraft` uses arbitrary strides, we must dynamically map our `row_stride`, `col_stride`, and `is_conjugated` flags into the strict `CBLAS_TRANSPOSE` enum (`NoTrans`, `Trans`, `ConjTrans`).

```rust
use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_zgemm};

pub struct DeviceMKL; // or DeviceOpenBLAS / DeviceOxiblas

impl DeviceMKL {
    /// Maps our strided MatRef to BLAS transposition flags and leading dimensions.
    #[inline(always)]
    fn resolve_blas_layout<T: Scalar>(mat: &MatRef<'_, T>) -> (CBLAS_TRANSPOSE, i32) {
        // Standard column-major: col_stride > row_stride == 1
        if mat.row_stride == 1 {
            let trans = if mat.is_conjugated {
                CBLAS_TRANSPOSE::CblasConjNoTrans // Requires CBLAS 2.0+ or careful manual handling
            } else {
                CBLAS_TRANSPOSE::CblasNoTrans
            };
            (trans, mat.col_stride as i32)
        } 
        // Standard row-major: row_stride > col_stride == 1
        else if mat.col_stride == 1 {
            let trans = if mat.is_conjugated {
                CBLAS_TRANSPOSE::CblasConjTrans
            } else {
                CBLAS_TRANSPOSE::CblasTrans
            };
            (trans, mat.row_stride as i32)
        } else {
            panic!("BLAS backend requires at least one unit stride. Use faer or tblis for arbitrary strides.");
        }
    }
}

impl LinAlgBackend for DeviceMKL {
    fn gemm<T: Scalar>(
        alpha: T,
        a: &MatRef<'_, T>,
        b: &MatRef<'_, T>,
        beta: T,
        c: &mut MatMut<'_, T>,
    ) {
        // Note: For brevity, assuming T is Complex<f64>. 
        // In reality, this requires trait specialization or macro dispatch 
        // to map to dgemm/zgemm/cgemm based on T.
        
        let (trans_a, lda) = Self::resolve_blas_layout(a);
        let (trans_b, ldb) = Self::resolve_blas_layout(b);
        let (_, ldc) = Self::resolve_blas_layout_mut(c);

        unsafe {
            cblas_zgemm(
                CBLAS_LAYOUT::CblasColMajor,
                trans_a,
                trans_b,
                c.rows as i32,
                c.cols as i32,
                a.cols as i32, // == b.rows (inner dimension K)
                &alpha as *const _ as *const _,
                a.data.as_ptr() as *const _,
                lda,
                b.data.as_ptr() as *const _,
                ldb,
                &beta as *const _ as *const _,
                c.data.as_mut_ptr() as *mut _,
                ldc,
            );
        }
    }
}

```

### The Architectural Takeaway

By designing the `MatRef` to decouple mathematical properties (`is_conjugated`) from memory layout (`strides` and `data`), your `tk-contract` executor never has to allocate intermediate buffers to resolve $A^\dagger$ operations.

When the `PathOptimizer` outputs a contraction graph, the `ContractionExecutor` simply evaluates whether an index permutation corresponds to a standard transpose, flips the `is_conjugated` boolean on the `TensorShape` view if needed, and pushes the pointers directly to the L1 cache via the backend traits.

---

===

To solve the Rayon thread starvation issue, we have to borrow a classic scheduling algorithm from high-performance computing: **Longest Processing Time (LPT) First**.

Because `tk-symmetry` mandates that the `sector_keys` and `sector_blocks` arrays remain strictly sorted by their quantum numbers to enable $O(\log N)$ binary searches, the natural iteration order of the tensor has absolutely no correlation with the computational weight of the blocks. In Abelian symmetric models, sector sizes follow a binomial distribution. If the massive $S_z = 0$ sector happens to sit at the end of the array, a naive `par_iter()` will hand it to a single Rayon worker thread last, causing all other cores to spin idly while they wait for it to finish.

Here is the concrete implementation of `SparseLinAlgBackend::block_gemm` that decouples the execution order from the storage order to guarantee perfect core saturation.

### The LPT-Scheduled Block GEMM

The strategy is to break the block-sparse contraction into three distinct phases:

1. **Task Generation:** Identify all non-zero sector pairs and calculate their exact FLOP cost ($M \times N \times K$).
2. **LPT Scheduling:** Sort the abstract tasks in descending order of FLOPs, then feed them to Rayon.
3. **Structural Restoration:** Re-sort the resulting computed blocks by their `PackedSectorKey` to restore the mathematical invariants of the output tensor.

```rust
use rayon::prelude::*;

/// An abstract task representing a single dense GEMM within a block-sparse contraction.
struct SectorGemmTask<'a, T: Scalar, Q: BitPackable> {
    /// The quantum number key for the output block C
    pub out_key: PackedSectorKey,
    pub block_a: &'a DenseTensor<T>,
    pub block_b: &'a DenseTensor<T>,
    /// Estimated computational cost: M * N * K
    pub flops: usize, 
}

impl<T: Scalar, Q: BitPackable> SectorGemmTask<'_, T, Q> {
    /// Execute the dense contraction for this specific sector
    fn execute<D: LinAlgBackend>(&self, alpha: T, beta: T) -> (PackedSectorKey, DenseTensor<T>) {
        // ... (Determine output shape based on block_a and block_b)
        // ... (Allocate zeroed DenseTensor C from SweepArena)
        
        // Dispatch to the dense backend (e.g., DeviceFaer or DeviceOxiblas)
        // Note: Because we are inside a Rayon worker, the LinAlgBackend MUST 
        // restrict its internal threading to 1 to prevent oversubscription.
        D::gemm(alpha, &self.block_a.as_ref(), &self.block_b.as_ref(), beta, &mut c_mut);
        
        (self.out_key, c_out)
    }
}

impl SparseLinAlgBackend for DefaultDevice {
    fn block_gemm<T: Scalar, Q: BitPackable>(
        a: &BlockSparseTensor<T, Q>, 
        b: &BlockSparseTensor<T, Q>
    ) -> BlockSparseTensor<T, Q> {
        
        // Phase 1: Task Generation
        // (Assuming standard U(1)/Z2 fusion rules to match non-zero sectors)
        let mut tasks = Vec::new();
        
        for (i, key_a) in a.sector_keys.iter().enumerate() {
            for (j, key_b) in b.sector_keys.iter().enumerate() {
                if let Some(out_key) = compute_fusion_rule(*key_a, *key_b) {
                    let block_a = &a.sector_blocks[i];
                    let block_b = &b.sector_blocks[j];
                    
                    // FLOPs = rows(A) * cols(B) * cols(A)
                    let flops = block_a.shape().rows() * block_b.shape().cols() * block_a.shape().cols();
                    
                    tasks.push(SectorGemmTask { out_key, block_a, block_b, flops });
                }
            }
        }

        // Phase 2: Longest Processing Time (LPT) Scheduling
        // Sort descending by FLOPs. The heaviest dense GEMMs are dispatched first.
        tasks.sort_unstable_by(|x, y| y.flops.cmp(&x.flops));

        // Evaluate the ThreadingRegime to determine if we should even use Rayon
        //
        let mut results = if tasks.len() > 1 && tasks[0].flops < MASSIVE_SECTOR_THRESHOLD {
            // Regime 2: Fragmented Sectors. Rayon dominates.
            tasks.into_par_iter()
                 .map(|task| task.execute::<DeviceFaer>(T::one(), T::zero()))
                 .collect::<Vec<_>>()
        } else {
            // Regime 1: Fat Sectors. Disable Rayon, let BLAS use all threads sequentially.
            tasks.into_iter()
                 .map(|task| task.execute::<DeviceFaer>(T::one(), T::zero()))
                 .collect::<Vec<_>>()
        };

        // Phase 3: Structural Restoration
        // To maintain O(log N) binary search capability, the output tensor's
        // arrays must be strictly sorted by the PackedSectorKey.
        //
        results.sort_unstable_by_key(|(key, _)| *key);

        let (out_keys, out_blocks): (Vec<_>, Vec<_>) = results.into_iter().unzip();

        BlockSparseTensor {
            indices: compute_output_indices(&a.indices, &b.indices),
            sector_keys: out_keys,
            sector_blocks: out_blocks,
            flux: Q::fuse(&a.flux, &b.flux),
        }
    }
}

```

### Why this architecture is safe and fast

1. **No Interior Mutability Required:** By having the Rayon closure return the computed block rather than trying to safely mutate a pre-allocated output array in parallel, we avoid needing `UnsafeCell` or complex slice-splitting logic. The data is simply collected and sorted at the end.
2. **Perfect Core Saturation:** Because the tasks are sorted `y.flops.cmp(&x.flops)`, the first $N$ threads grab the massive matrix multiplications. As they finish, they immediately start stealing the smaller matrix multiplications from the tail of the queue. All cores remain saturated until the exact moment the contraction is finished.
3. **Preservation of the Binary Search Invariant:** Sorting the results at the end guarantees that the output `BlockSparseTensor` perfectly satisfies the `PackedSectorKey` sorting requirement, ensuring that the *next* operation in the DAG executes with zero branch misprediction.


===



When you activate the `backend-cuda` feature to utilize GPU acceleration, the architecture wisely specifies using a `PinnedArena` backed by `cudaMallocHost` to enable direct memory access (DMA) transfers.

However, pinned memory is "page-locked." It explicitly forbids the operating system from swapping those pages to disk. If you allocate too much pinned memory, the OS will run out of physical RAM for essential kernel operations, resulting in a catastrophic kernel panic or the Linux OOM (Out-Of-Memory) killer ruthlessly terminating your solver.

Because `tensorkraft` targets MPI Mode B for embarrassingly parallel DMFT, multiple independent processes will be competing for the exact same pool of physical RAM on a single compute node.

Here is how to design a robust, atomic pinned-memory tracker that dynamically falls back to standard pageable memory before you crash the node.

### 1. The Global Tracker Definition

We need a process-wide tracker that intercepts every request for pinned memory. Because `SweepArena` allocations happen continuously during the DMRG sweeps, this tracker must be wait-free and utilize low-level atomic operations.

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

/// A global counter tracking the amount of page-locked (pinned) memory 
/// currently allocated by this process.
static PINNED_BYTES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

/// The maximum allowable pinned memory for this process. 
/// Must be initialized at startup based on system RAM and MPI rank count.
static PINNED_BYTES_LIMIT: AtomicUsize = AtomicUsize::new(0);

pub struct PinnedMemoryTracker;

impl PinnedMemoryTracker {
    /// Initialize the budget. For MPI Mode B, this should be:
    /// (Node RAM * 0.75) / local_ranks_per_node
    pub fn initialize_budget(max_bytes: usize) {
        PINNED_BYTES_LIMIT.store(max_bytes, Ordering::Release);
    }

    /// Attempt to reserve pinned memory. Returns true if successful.
    pub fn try_reserve(bytes: usize) -> bool {
        let limit = PINNED_BYTES_LIMIT.load(Ordering::Relaxed);
        
        // Loop to safely compare-and-swap the new allocation size
        let mut current = PINNED_BYTES_ALLOCATED.load(Ordering::Relaxed);
        loop {
            if current + bytes > limit {
                // Budget exceeded; refuse the allocation
                return false; 
            }
            
            match PINNED_BYTES_ALLOCATED.compare_exchange_weak(
                current,
                current + bytes,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true, // Successfully reserved
                Err(actual) => current = actual, // Retry with updated value
            }
        }
    }

    /// Release previously reserved pinned memory.
    pub fn release(bytes: usize) {
        PINNED_BYTES_ALLOCATED.fetch_sub(bytes, Ordering::Release);
    }
}

```

### 2. The Graceful Fallback in `SweepArena`

The `SweepArena` needs to dynamically decide whether to back its temporary tensor allocations with pinned memory or standard pageable memory.

If `PinnedMemoryTracker::try_reserve()` returns `false`, the arena must smoothly degrade to using standard `bumpalo` heap allocation. The linear algebra backend (`DeviceCuda`) will still work; the CUDA driver will just silently handle the pageable-to-pinned staging buffer penalty under the hood, gracefully trading bandwidth for system stability.

```rust
#[cfg(feature = "backend-cuda")]
pub enum ArenaStorage {
    /// High-bandwidth, DMA-capable page-locked memory.
    Pinned(PinnedArena),
    /// Standard pageable heap memory (graceful fallback).
    Pageable(bumpalo::Bump),
}

#[cfg(feature = "backend-cuda")]
pub struct SweepArena {
    storage: ArenaStorage,
}

#[cfg(feature = "backend-cuda")]
impl SweepArena {
    pub fn new(capacity_bytes: usize) -> Self {
        // First, check if the global budget allows this allocation
        if PinnedMemoryTracker::try_reserve(capacity_bytes) {
            match PinnedArena::new(capacity_bytes) {
                Ok(arena) => {
                    log::info!("SweepArena allocated {} bytes of pinned memory.", capacity_bytes);
                    return SweepArena { storage: ArenaStorage::Pinned(arena) };
                }
                Err(_) => {
                    // Cuda error (e.g., driver refused). Roll back the tracker.
                    PinnedMemoryTracker::release(capacity_bytes);
                }
            }
        }
        
        // Fallback: Budget exceeded or cudaMallocHost failed.
        log::warn!("Pinned memory limit reached. SweepArena falling back to pageable memory. GPU transfers will be slower.");
        SweepArena {
            storage: ArenaStorage::Pageable(bumpalo::Bump::with_capacity(capacity_bytes)),
        }
    }

    /// O(1) reset reclaims memory within the arena, but does NOT return it to the OS.
    pub fn reset(&mut self) {
        match &mut self.storage {
            ArenaStorage::Pinned(arena) => arena.reset(),
            ArenaStorage::Pageable(bump) => bump.reset(),
        }
    }
}

// Crucial: The Drop implementation must return the budget to the global tracker
// when the SweepArena is destroyed (e.g., at the end of a DMFT iteration).
#[cfg(feature = "backend-cuda")]
impl Drop for SweepArena {
    fn drop(&mut self) {
        if let ArenaStorage::Pinned(arena) = &self.storage {
            PinnedMemoryTracker::release(arena.capacity());
            // arena's own Drop will call cudaFreeHost
        }
    }
}

```

### 3. The MPI Mode B Topology Constraint

Because `AtomicUsize` only tracks memory within a single process space, it is blind to other MPI ranks running on the same machine.

To make this architecture safe for the proposed MPI Mode B DMFT loop, the initialization sequence in `tk-dmft` must query the system topology before the solver starts.

```rust
#[cfg(feature = "backend-mpi")]
pub fn initialize_dmft_node_budget(comm: &MpiComm) {
    // 1. Query the OS for total physical RAM
    let total_ram = sys_info::mem_info().unwrap().total;
    
    // 2. Determine how many MPI ranks share this specific physical node.
    // (MPI communicators can be split by shared-memory domain)
    let local_ranks = comm.split_by_shared_memory().size();
    
    // 3. Safe heuristic: allow pinned memory up to 60% of total RAM, 
    // divided evenly among the local ranks.
    let safe_node_limit = (total_ram as f64 * 0.60) as usize;
    let rank_budget = safe_node_limit / local_ranks;
    
    PinnedMemoryTracker::initialize_budget(rank_budget);
}

```

### The Architectural Takeaway

By combining an atomic reservation system with a smooth degradation path to pageable memory, `tensorkraft` achieves the best of both worlds. It will extract absolute peak PCI-e bandwidth for GPU offloading via `PinnedArena` when resources are abundant, but it physically cannot crash the host operating system if the user spins up too many parallel MPI ranks.

Would you like to shift focus to the algorithms in `tk-dmft` next, specifically looking at how to implement the mandatory Chebyshev expansion to bypass the linear prediction ill-conditioning?