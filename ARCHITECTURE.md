# Software Architecture Design Document
# Rust-Based Tensor Network Library
**for Quantum Many-Body Physics — DMRG Algorithms & DMFT Impurity Solver Integration**

*Version 8.5 — March 2026 | Status: Architectural Specification (Post-Draft-Implementation)*

---
## Revision Notes (v8.4 → v8.5)

This revision incorporates cross-cutting findings from the draft implementation of all crates (tk-core through tk-python). Changes reflect lessons learned from actual Rust compilation and runtime behavior, not speculative design. Key changes include:

- **§2.1** — Workspace layout corrected: crates live in top-level directories (`tk-core/`, `tk-symmetry/`, etc.), not under a `crates/` subdirectory.
- **§3.1** — `TensorCow` eliminated: `TensorStorage<'a, T>` is now a single enum with `Owned(Vec<T>)` / `Borrowed(&'a [T])`. `DenseTensor` gains a lifetime parameter (`DenseTensor<'a, T>`) and an `offset: usize` field essential for zero-copy `slice_axis`. `TempTensor` is just `DenseTensor` with a shorter lifetime, not a distinct type.
- **§3.4** — `Scalar` trait expanded: requires `Sub<Output=Self>`, `Neg<Output=Self>`, `Debug`, `'static`. `Real` associated type requires `PartialOrd` and `Float`. Added `from_real_imag()` for complex number construction (needed by `SpinOp::Sy`, Green's functions).
- **§4.1** — `QuantumNumber` requires `'static` bound. `BitPackable` requires `Copy`.
- **§4.2** — `BlockSparseTensor` gains `leg_directions: Vec<LegDirection>` field (essential for flux rule enforcement). Blocks are always `DenseTensor<'static, T>` (no arena-borrowed blocks). `permute()` is NOT zero-copy — requires `into_owned()` on each block for BLAS contiguity. `DenseTensor` and `BlockSparseTensor` require `Clone` (cross-cutting issue affecting tk-contract, tk-dsl, tk-dmrg, tk-dmft).
- **§4.4** — CG cache uses hand-rolled Racah formula, not `lie-groups` dependency.
- **§5.1** — `regularized_svd_inverse` requires `where Self: Sized` bound (breaks object-safety for that method only). Return types for SVD/eigh/QR are `DenseTensor<'static, T>` (always owned).
- **§5.2** — `DefaultDevice` interim definition: `DeviceAPI<DeviceFaer, DeviceFaer>` (not `DeviceOxiblas`). `gesdd`→`gesvd` fallback is no-op with faer (only meaningful with MKL/OpenBLAS). faer `thin_svd()` returns ascending order; must re-sort descending.
- **§5.3** — Threading uses simple binary heuristic for Phase 1–3, not partitioned scheduler.
- **§7.3** — Operator overloading (`*`, `+` on `OpSum`) restricted to `f64` due to Rust orphan rules. `hamiltonian!{}` proc-macro deferred (not yet implemented).
- **§8.3** — `DMRGConfig` should be split into immutable `DMRGConfig` and mutable `DMRGState`. `Box<dyn IterativeEigensolver>` prevents `Clone`. Davidson/Block-Davidson delegate to Lanczos (not yet independently implemented).
- **§8.3** — Checkpoint non-functional: blocked by `BlockSparseTensor` lacking serde.
- **§8.4** — Double-occupancy measurement requires `CustomOp` (no `NPairInteraction` in `FermionOp`). Positivity restoration edge case: only rescale when both sums are positive. `LinearPredictionConfig` field names: `prediction_order` (not `lp_order`), `toeplitz_solver` (not `solver`).
- **§7.5** — `ComplexU1` variant blocked: `DeviceFaer` doesn't implement `LinAlgBackend<Complex<f64>>` yet. PyO3 `extension-module` vs test linking conflict requires feature flag pattern. Config mirror pattern needed because `DMRGConfig` is not `Clone`.
- **§6** — `DenseTensor` lifetime makes executor generics painful; unsafe transmute workaround for lifetime unification.
- **§11** — `lie-groups` dependency replaced by hand-rolled Racah formula for CG coefficients.
- **General** — `DenseTensor Clone` is the #1 cross-cutting issue affecting tk-contract, tk-dsl, tk-dmrg, tk-dmft. Must be resolved.

---
## Revision Notes (v8.3 → v8.4)

This revision addresses the thirteenth external review targeting pinned-memory allocation in the flatten path, MPI process-isolation semantics, and physics-triggered SVD validation for TDVP. Changes include:

- **§4.2** — Arena-backed `flatten()`: The `flatten()` method now accepts a `&SweepArena` parameter and packs fragmented blocks directly into the arena's pinned memory (when `backend-cuda` is active), rather than allocating fresh pageable heap memory. This prevents the NVIDIA driver from silently allocating a hidden staging buffer for the DMA transfer, which would halve effective PCI-e bandwidth and negate the purpose of the flat-buffer refactor.
- **§5.1** — Physics-triggered SVD validation for TDVP: The periodic modulo-counter validation (every K-th call) is supplemented by a dynamic trigger. If the TDVP truncation error jumps by more than `svd_anomaly_factor` (default: 10×) between consecutive time steps, or if the energy variance exceeds `svd_anomaly_energy_tol`, an immediate out-of-band SVD residual check is forced. This catches acute `gesdd` corruption the moment it impacts physics, rather than waiting up to K steps. For DMRG (where corrupted SVDs waste a recoverable sweep), the modulo counter alone is sufficient; for TDVP (where corrupted gauge restorations compound exponentially through the matrix exponential), the physics trigger is essential.
- **§10.2.1** — MPI process-isolation clarification: Explicit statement that `PinnedMemoryTracker` is a *process-local* guard enforcing a statically pre-negotiated per-rank budget slice, not an inter-process shared-memory atomic. The `AtomicUsize` tracks only the calling process's own allocations; cross-rank coordination occurs only once at startup via the `initialize_dmft_node_budget` topology query (§10.2.2).
- **§13** — Risk Analysis: Updated rows for flatten allocation trap, SVD blind spot in TDVP, and MPI atomic isolation.

## Revision Notes (v8.2 → v8.3)

This revision addresses the twelfth external review targeting flat-buffer mutation hazards, SVD validation in production builds, SU(2) task memory bounds, and soft D_max time-scaling ambiguity. Changes include:

- **§4.2** — Dual-layout block storage: The `FlatBlockStorage` refactor is clarified as a *compute-side read-only representation* for GPU DMA and GEMM dispatch. During TDVP subspace expansion (which mutates tensor structure), blocks remain in the fragmented `Vec<DenseTensor<T>>` mutation layout where appending columns is O(D_sector²). A `flatten()` method packs into contiguous `FlatBlockStorage` only after structural mutations are complete, before shipping to GPU or entering the GEMM phase. This prevents the O(D_total²) reallocation penalty that would occur if subspace expansion operated directly on a flat buffer.
- **§5.1** — Periodic SVD residual validation in release builds: The `debug_assert!` residual check is supplemented by a release-mode periodic check every `svd_validation_interval` calls (default: 1000). At this interval, the amortized overhead is 0.1% of SVD cost. Controlled by a runtime `DMRGConfig::svd_validation_interval` field rather than a compile-time flag, ensuring multi-week production DMFT runs detect silent `gesdd` corruption within ~1000 sweep steps of occurrence.
- **§4.4** — SU(2) task generation memory bound: Added capacity analysis showing that for typical SU(2) parameters (D_reduced ≤ 50, j_max ≤ 10), the task fan-out is ~250,000 tasks at ~16 MB — well within L3 cache. `Vec::with_capacity` pre-allocation based on estimated fan-out documented to avoid incremental reallocation.
- **§8.1.1** — Soft D_max physical time tracking: `expansion_age` changed from discrete iteration counter (`usize`) to accumulated physical time (`f64`, in units of the simulation time). The decay formula `exp(−t_physical / dmax_decay_time)` is now invariant to the TDVP integrator's adaptive time-stepping. `dmax_decay_steps` renamed to `dmax_decay_time` with units clarified. `tick_expansion_age` replaced by `advance_expansion_age(dt: f64)`.
- **§13** — Risk Analysis: Updated rows for flat-buffer mutation hazard, SVD silent corruption in production, SU(2) task memory scaling, and soft D_max time-step coupling.

## Revision Notes (v8.1 → v8.2)

This revision addresses the eleventh external review targeting spectral restoration physics, Tikhonov parameter adaptivity, GPU kernel dispatch semantics, eigensolver workspace lifecycle, environment memory management, and fermionic optimizer scaffolding. Changes include:

- **§8.4.2** — Fermi-level distortion diagnostic: The positivity restoration pass now compares A(ω=0) before and after clamping+rescaling. If the Fermi-level value shifts by more than `fermi_level_shift_tolerance` (default: 1%), a structured warning is emitted. Global L₁ rescaling is retained but the diagnostic catches the specific failure mode where high-frequency tail clamping unphysically shifts spectral weight at the Fermi level.
- **§8.1.1** — Tikhonov parameter adaptive annealing: `TdvpStabilizationConfig` extended with `adaptive_tikhonov: bool`. When enabled, δ is dynamically scaled relative to the largest discarded singular value from the previous truncation step, preventing the regularization floor from masking physics in near-product-state bonds.
- **§10.2.5** — GPU grouped GEMM: Corrected `cublasDgemmBatched` reference to specify dimension-grouped dispatch or `cublasGemmGroupedBatchedEx` (CUDA 12.1+) / CUTLASS grouped GEMM for heterogeneous sector sizes. Standard batched cuBLAS requires uniform (M,N,K) and would silently serialize on heterogeneous block-sparse sectors.
- **§8.2** — Persistent Krylov workspace: Eigensolver workspace refined from "heap-allocated and dropped every step" to a persistent reusable buffer owned by `DMRGEngine`, allocated once and reused across all sweep steps. Avoids allocator churn and heap fragmentation from repeated 640 MB alloc/drop cycles.
- **§8.3** — Environment caching strategy: New documentation specifying that `Environments<T, Q>` caches all N−2 left/right environment pairs in host RAM, built incrementally during the sweep. Memory budget, disk-offload strategy for memory-constrained nodes, and recomputation trade-offs documented.
- **§6.2** — Fermionic cost scaffolding: Forward-compatibility note added to `PathOptimizer` documenting that fermionic swap gates (Phase 5+) are penalized by the existing `bandwidth_weight` in `CostMetric` (a fermionic swap is a transpose with an O(1) sign flip). No interface change required now, but the note prevents future developers from assuming the optimizer needs restructuring.
- **§13** — Risk Analysis: Updated rows for Fermi-level spectral distortion, Krylov workspace fragmentation, environment memory scaling, and GPU heterogeneous batching.

## Revision Notes (v8.0 → v8.1)

This revision addresses the ninth and tenth external reviews targeting block storage layout, BLAS thread safety, GPU dispatch granularity, memory-constrained path optimization, symmetry-preserving subspace expansion, eigensolver memory ownership, adaptive solver selection, and cross-validation infrastructure. Changes include:

- **§4.2** — Flat-buffer block storage refactor: Documented the planned transition from `Vec<DenseTensor<T>>` per-block storage to a single contiguous flat buffer with an offset table, scheduled as a Phase 4 prerequisite for efficient GPU DMA transfers. The `SparseLinAlgBackend` trait does not expose storage internals, making this a non-breaking refactor.
- **§5.3** — BLAS thread safety caveat: Documented that `set_blas_threads` is global mutable state, safe only because the partitioned scheduler serializes access between phases. `mkl_set_num_threads_local` noted as preferred on newer MKL versions. `faer` avoids the problem entirely via per-call `Parallelism` parameter.
- **§5.3** — Three-way GPU/CPU/Rayon LPT partition: When `backend-cuda` is active, `PartitionedLptConfig` gains a `gpu_flop_threshold`. Tasks above this threshold are diverted to the GPU queue; tasks between `gpu_flop_threshold` and `blas_flop_threshold` get CPU multithreaded BLAS; tasks below `blas_flop_threshold` go to Rayon. Prevents GPU kernel launch overhead from dominating tiny sector GEMMs.
- **§6.2** — Memory-constrained path optimization: `PathOptimizer::optimize` gains an optional `max_memory_bytes` parameter. For DMRG (small fixed DAGs), this defaults to `None`. For future PEPS/tree TNS extensions, the optimizer can reject paths exceeding the memory budget.
- **§8.1.1** — Sector-preserving subspace expansion: SVD of the null-space residual |R_null⟩ is performed per symmetry sector (block-sparse SVD), not as a single dense operation, guaranteeing that expansion vectors are flux-preserving by construction.
- **§8.2** — Eigensolver memory ownership: Clarified that Krylov vectors are heap-allocated (`Vec<Vec<T>>`) owned by the eigensolver, not arena-allocated. Dropped after eigensolver returns, before the SVD phase begins. Contraction temporaries inside the matvec closure remain arena-allocated.
- **§8.4.1** — Adaptive TDVP/Chebyshev solver selection: `DMFTLoop` checks the entanglement spectrum gap from the DMRG ground state. If the gap indicates a metallic/gapless phase, Chebyshev expansion is automatically promoted to primary solver. TDVP + linear prediction remains primary for gapped/insulating phases.
- **§10.2.5** — GPU performance: Updated to reference three-way LPT partition; fragmented tail never touches GPU.
- **§10.2.6** — NUMA verification: Added `numactl --show` verification recommendation for multi-socket single-GPU deployments.
- **§12.1** — ARPACK cross-validation: In-house eigensolvers are cross-validated against ARPACK via `arpack-ng` FFI in integration tests, gated behind a `test-arpack` feature flag.
- **§13** — Risk Analysis: Updated rows for flat-buffer storage, BLAS thread safety, GPU sector dispatch, memory-constrained optimization, sector-preserving expansion, eigensolver numerical maintenance, and adaptive solver selection.

## Revision Notes (v7.1 → v8.0)

This revision addresses the eighth external review targeting CPU thread-pool scheduling pathology, GPU submission isolation, linear prediction solver complexity, spectral positivity, and proc-macro ergonomics. Changes include:

- **§5.3** — Partitioned LPT dispatch: The binary `ThreadingRegime::select` switch has been replaced with a two-phase partitioned scheduler. The LPT-sorted task queue is split at a calibrated `BLAS_FLOP_THRESHOLD`: heavy-head tasks run sequentially under full BLAS multithreading, then the light tail is cleared by Rayon with single-threaded BLAS. This eliminates the "middle ground" pathology where tensors with a mix of massive and tiny sectors were routed entirely to the wrong regime. An optional startup microbenchmark calibrates the threshold to the host machine's actual BLAS crossover point.
- **§7.2.1** — Macro span hygiene: New subsection documenting `Span::call_site()` for user-provided physics parameters and `Span::mixed_site()` for macro-internal temporaries, preventing variable shadowing between the DSL and surrounding user code.
- **§7.2.2** — Macro diagnostic reporting: New subsection specifying `syn::parse::Parse` implementations for the physics DSL grammar, with `syn::Error::new_spanned` producing IDE-friendly error diagnostics that point to the exact token where the user made a syntax error.
- **§8.4.2** — Levinson-Durbin solver: Levinson-Durbin recursion documented as the recommended O(P²) solver for the Toeplitz prediction system, replacing SVD-based pseudo-inversion as the default. SVD retained as a fallback for non-Toeplitz extensions.
- **§8.4.2** — Spectral positivity restoration: Mandatory post-deconvolution positivity pass added. Negative spectral weights are clamped to a near-zero floor, followed by L₁ renormalization to preserve the spectral sum rule. A diagnostic warning fires if clamped weight exceeds 5% of total spectral weight, indicating misconfigured deconvolution parameters.
- **§10.2.4** — GPU submission clarification: Documented that the topological DAG walk is performed by a single thread issuing all CUDA API calls, avoiding driver-level lock contention. Dedicated GPU submission queue with OS-thread isolation documented as a Phase 5+ requirement for multi-GPU and pipelined execution.
- **§13** — Risk Analysis: Updated rows for thread-pool scheduling, spectral positivity, macro hygiene, and GPU submission contention.

## Revision Notes (v7.0 → v7.1)

This revision addresses critical concurrency and numerical correctness edge cases for non-Abelian SU(2) symmetry implementations. Changes include:
- **§4.4** — SU(2) output-sector collision hazard: Documented the necessity of a map-reduce pattern during LPT task generation to prevent data races when multiple input pairs target the same output sector due to fusion-rule multiplicity.
- **§4.4** — Multiplet-aware SVD truncation: Added the requirement for a two-phase SVD truncation approach in SU(2)-symmetric DMRG. Truncation must apply (2j+1) weighting to discarded singular values and snap to the nearest multiplet edge to prevent explicit symmetry breaking.
- **§13** — Risk Analysis: Updated the SU(2) Wigner-Eckart complexity row to reflect the map-reduce task accumulation and multiplet-aware SVD truncation mitigations.

## Revision Notes (v6.0 → v7.0)

This revision addresses a sixth external review targeting deconvolution stability, SVD silent failures, MPI load imbalance, pinned-memory telemetry, NUMA topology, and SU(2) fusion-rule multiplicity. Changes include:

- **§5.1** — SVD residual validation: `debug_assert!` on ‖A − UΣV†‖/‖A‖ after `gesdd` returns, catching silently inaccurate small singular values in debug/test builds without production overhead.
- **§4.4** — SU(2) fusion-rule multiplicity: documented that `compute_fusion_rule` returns one-to-many for non-Abelian irreps (j₁⊗j₂ = |j₁−j₂| ⊕ ... ⊕ (j₁+j₂)). `SectorGemmTask` generation must produce `Vec<SectorGemmTask>` per input pair. Noted as a known Phase 5 refactoring requirement.
- **§8.4.2** — Regularized Lorentzian deconvolution: Tikhonov-style damping prevents quadratic noise amplification in high-frequency tails. `LinearPredictionConfig` extended with `deconv_tikhonov_delta` and `deconv_omega_max`.
- **§10.2.1** — Pinned-memory fallback upgraded from `log::warn` to structured telemetry event with fallback counter exposed in `DMRGEngine` stats.
- **§10.2.6** — NUMA-aware pinned allocation: documented requirement for multi-GPU nodes to bind pinned memory to the NUMA node of the target GPU's PCIe root complex. Deferred to Phase 5+.
- **§10.3** — MPI Mode B load imbalance: `MPI_Allgather` barrier documented as a bottleneck for heterogeneous impurity solvers. Asynchronous convergence checks noted for multi-orbital/cluster DMFT in Phase 5+.

---

## Revision Notes (v5.0 → v6.0)

This revision addresses a fifth external review targeting a Rust object-safety violation, a GIL re-acquisition deadlock, a Fourier transform error, and missing per-bond state for soft D_max. Changes include:

- **§5.1** — `LinAlgBackend` trait parameterized over `T: Scalar` at the trait level (`LinAlgBackend<T>`) instead of per-method generics, restoring object safety. `Box<dyn LinAlgBackend<f64>>` is now valid Rust. `SparseLinAlgBackend` parameterized over `<T, Q>`.
- **§5.4** — Monomorphization strategy updated to reference object-safe `dyn LinAlgBackend<T>`.
- **§7.5** — GIL deadlock fix: `done_tx.send()` and `monitor_handle.join()` moved *inside* the `py.allow_threads` closure, ensuring the monitor thread is fully terminated before the main thread re-acquires the GIL.
- **§8.1.1** — Soft D_max per-bond state: `TdvpDriver` carries `Vec<Option<usize>>` tracking steps since last expansion at each bond. `TdvpStabilizationConfig` is stateless; the driver is stateful.
- **§8.4.2** — Fourier transform correction: Gaussian window exp(−ηt²) replaced with exponential window exp(−η|t|), whose Fourier transform is a Lorentzian η/(η²+ω²). Terminology, formula, and deconvolution procedure now mutually consistent.

---

## Revision Notes (v4.0 → v5.0)

This revision addresses a fourth external review targeting Rust-specific ergonomics, borrow-checker friction, compile-time explosion, and physics-level omissions. Changes include:

- **§3.3.1** — Arena ownership boundary: explicit `TempTensor<'a>` vs `OwnedTensor` distinction. DMRG step outputs call `.into_owned()` before arena reset. Documented in §9 data flow.
- **§5.4** — Monomorphization budget strategy: static dispatch reserved for inner loops (GEMM, matvec); `dyn LinAlgBackend` permitted at sweep-scheduler level. Compile-time explosion mitigated by feature-gated type combinations.
- **§6.4** — Fermionic sign convention: contraction engine operates with bosonic tensor legs only. Jordan-Wigner strings encoded in MPO. Documented limitation for tree/PEPS extensions.
- **§7.3** — Strongly-typed operator enums (`SpinOp`, `FermionOp`, `BosonOp`) replace string-based `op("S+", i)` API. Custom operators via `op(CustomMatrix(...), i)` for non-standard models.
- **§8.1.1** — Soft D_max with exponential decay: `TdvpStabilizationConfig` extended with `soft_dmax_factor` and `dmax_decay_rate` to prevent bond-dimension oscillation at the truncation threshold during subspace expansion.
- **§8.4.2** — Exponential windowing for linear prediction: `LinearPredictionConfig` extended with `broadening_eta` to enforce decay in metallic G(t), stabilizing the Toeplitz pseudo-inverse. Lorentzian broadening deconvolved post-FFT.

---

## Revision Notes (v3.0 → v4.0)

This revision addresses a third external review targeting two edge cases introduced by v3.0 fixes. Changes include:

- **§7.5** — Signal-monitor thread lifecycle: `mpsc::channel` with `recv_timeout` replaces bare `loop`/`sleep`. The solver sends a completion message on the channel when it finishes, guaranteeing monitor thread termination on the happy path. `JoinHandle::join()` ensures the monitor has fully exited before returning to Python, closing a narrow race window during interpreter teardown.
- **§8.1.1** — Subspace expansion null-space projection: explicit projector construction (P_null = I − A_L · A_L†) replaced with matrix-free sequential projection (compute overlap O = A_L† · |R⟩, subtract A_L · O). Reduces projection cost from O(d²D³) to O(dD²), preserving the O(D³) DMRG scaling bound.

---

## Revision Notes (v2.0 → v3.0)

This revision addresses a second comprehensive external review targeting mathematical correctness of the TDVP subspace expansion algorithm, hardware-level scheduling pathologies, OS-level resource exhaustion, BLAS-level conjugation semantics, and Python/Rayon thread-safety deadlocks. Major changes include:

- **§4.2** — `MatRef` extended with `is_conjugated: bool` flag, enabling zero-copy Hermitian conjugation via BLAS `ConjTrans` flags and `faer`'s lazy conjugation views. Eliminates O(N) conjugation memory passes.
- **§5.1** — `LinAlgBackend::gemm` signature updated to propagate conjugation metadata through hardware micro-kernels.
- **§5.3** — Longest Processing Time (LPT) scheduling for block-sparse Rayon dispatch. Sectors sorted by descending FLOPs before `par_iter` to prevent long-tail thread starvation from binomial sector-size distributions.
- **§7.5** — PyO3 signal-checking redesigned: `AtomicBool` cancellation flag replaces direct `Python::with_gil` from Rayon workers. Only the originating thread re-acquires the GIL.
- **§8.1.1** — Subspace expansion algorithm corrected: expansion operates on site tensors A_L/A_R, not on the bond matrix C. Dimensional analysis now consistent (D×d pad, not D×D mix).
- **§10.2** — `PinnedMemoryTracker`: global atomic budget for pinned memory with automatic fallback to pageable allocation. MPI-aware topology query divides budget across co-resident ranks.

---

## Revision Notes (v1.0 → v2.0)

This revision addresses a comprehensive external review. Major changes include:

- **§4.2** — Bit-packed sector keys (`PackedSectorKey`) replace `SmallVec`-based lookups for Abelian symmetries, eliminating branch misprediction during BLAS dispatch.
- **§5.1** — SVD algorithm selection policy: default to divide-and-conquer (`gesdd`) with `gesvd` fallback.
- **§5.1** — Regularized pseudo-inverse (`regularized_svd_inverse`) added to `LinAlgBackend` for TDVP gauge restoration.
- **§7.2** — `hamiltonian!{}` macro scope clarified: generates `OpSum` AST only; MPO compression is a runtime operation in `tk-dmrg`.
- **§8.1** — TDVP numerical stabilization: Tikhonov regularization and subspace expansion integrated into `BondCentered` workflow.
- **§8.4** — `TimeEvolutionConfig` extended with regularization and expansion parameters.
- **§7.5** — Python bindings overhauled: explicit GIL release, `Ctrl+C` signal forwarding, zero-copy NumPy via `rust-numpy`.
- **§10.2** — CUDA pinned-memory arena for DMA-capable host-device transfers.
- **§11** — `eigenvalues` crate removed; iterative eigensolvers are implemented in-house within `tk-dmrg`.
- **§12** — Testing strategy hardened: gauge-invariant assertions, snapshot-based ED validation, instruction-counting CI benchmarks, bounded `proptest` strategies.
- **§2.2** — `tk-dsl` dependency on `tk-linalg` removed; `OpSum → MPO` compilation moved to `tk-dmrg`.
- **§2.3** — Mutual exclusivity enforced for FFI-based BLAS feature flags.

---

## 1. Executive Summary

This document specifies the complete software architecture for a new, high-performance tensor network library written in Rust, provisionally named **tensorkraft**. The library targets quantum many-body physics with a primary focus on Density Matrix Renormalization Group (DMRG) algorithms, designed to serve as a real-frequency quantum impurity solver within Dynamical Mean-Field Theory (DMFT) self-consistency loops.

The architecture leverages Rust's zero-cost abstractions, ownership-based memory safety, and fearless concurrency to achieve performance competitive with established C++ frameworks (ITensor, Block2) and Julia libraries while providing compile-time safety guarantees that eliminate entire categories of runtime errors endemic to scientific computing.

The design is organized around five foundational pillars, each addressed by a dedicated sub-crate within a Cargo workspace:

- **tk-core:** Tensor data structures with strict shape/storage separation (`DenseTensor<'a, T>` with CoW `TensorStorage<'a, T>`), arena allocation, and offset-based zero-copy slicing. Matrix views (`MatRef`) carry lazy conjugation flags for zero-copy Hermitian transposes.
- **tk-symmetry:** Native Abelian symmetry support (U(1), Z₂) via block-sparse formats with bit-packed sector keys, with a clear extension path for non-Abelian SU(2) via Wigner-Eckart factorization.
- **tk-linalg:** Trait-based linear algebra backend abstraction defaulting to faer (dense) and oxiblas (sparse), swappable via feature flags. GEMM dispatch propagates conjugation metadata to hardware micro-kernels. Includes regularized pseudo-inverse for numerically stable gauge restoration. Three-way partitioned LPT-scheduled block-sparse parallelism dynamically routes GPU-heavy sectors, CPU multithreaded BLAS sectors, and fragmented Rayon tails.
- **tk-contract:** DAG-based contraction engine with separated path optimization and execution phases. Bosonic tensor legs only; fermionic sign rules encoded in MPO via Jordan-Wigner.
- **tk-dsl:** Ergonomic API with intelligent indices, strongly-typed operator enums (`SpinOp`, `FermionOp`, `BosonOp`) with `CustomOp` escape hatch, an `OpSum` builder pattern, and a `hamiltonian!{}` proc-macro with span hygiene guarantees and IDE-friendly diagnostics that generates uncompressed operator sums as compile-time boilerplate reduction; actual MPO compression is a runtime operation delegated to `tk-dmrg`.

Additionally, integration crates **tk-dmrg** and **tk-dmft** implement the full DMRG sweep algorithm, iterative eigensolvers (Lanczos/Davidson, written in-house with ARPACK cross-validation in tests), MPO compilation from `OpSum`, time-evolution methods (TDVP with Tikhonov-regularized gauge shifts and site-tensor subspace expansion for entanglement growth, TEBD as fallback), and the DMFT self-consistency loop with adaptive TDVP/Chebyshev solver selection, Levinson-Durbin-based linear prediction, and mandatory spectral positivity restoration. A **tk-python** crate provides PyO3 bindings with explicit GIL management, thread-safe `AtomicBool` cancellation with `mpsc`-guarded monitor thread lifecycle, and zero-copy NumPy interop.

---

## 2. Workspace & Crate Architecture

The library is structured as a Cargo workspace containing focused, independently testable sub-crates.

### 2.1 Workspace Layout

```
tensorkraft/
├── Cargo.toml              # workspace root
├── tk-core/                 # Tensor shape, storage, memory mgmt, MatRef/MatMut
├── tk-symmetry/             # Quantum numbers, block-sparse formats
├── tk-linalg/               # Backend abstraction (faer, oxiblas), LPT scheduling
├── tk-contract/             # DAG engine, path optimization
├── tk-dsl/                  # Macros, OpSum, lattice builders
├── tk-dmrg/                 # DMRG sweeps, eigensolvers, MPS/MPO, OpSum→MPO compilation
├── tk-dmft/                 # DMFT loop, bath discretization, TDVP/TEBD
├── tk-python/               # PyO3 bindings for DMFT integration
├── benches/                 # Criterion benchmarks (local bare-metal only)
├── fixtures/                # Reference snapshot data (ED energies, spectra)
├── examples/                # Heisenberg chain, Hubbard DMFT, etc.
└── tests/                   # Integration tests
```

### 2.2 Crate Dependency Graph

| Crate | Responsibility | Depends On |
|:------|:---------------|:-----------|
| **tk-core** | Tensor shape/stride metadata, `TensorStorage<'a, T>` with CoW semantics (`Owned`/`Borrowed`), `DenseTensor<'a, T>` with offset field, `MatRef`/`MatMut` with lazy conjugation flag, arena allocators with explicit ownership boundary (`.into_owned()`) and pinned-memory budget tracking, element-type generics, error types | *(none — leaf crate)* |
| **tk-symmetry** | QuantumNumber trait (`'static` bound), BitPackable trait (`Copy` supertrait), U(1)/Z₂ implementations, PackedSectorKey, block-sparse storage with `leg_directions: Vec<LegDirection>` per tensor, `fuse_legs`/`split_leg` with BTreeMap-based deterministic ordering, `iter_keyed_blocks()` paired iterator, Wigner-Eckart scaffolding for SU(2) with DashMap-based CG cache (hand-rolled Racah formula) | tk-core |
| **tk-linalg** | LinAlgBackend trait (conjugation-aware GEMM), SVD/EVD dispatch (gesdd default), regularized pseudo-inverse, DeviceFaer and DeviceOxiblas implementations, three-way partitioned LPT-scheduled block-sparse dispatch (GPU/CPU-BLAS/Rayon), Rayon-parallel element-wise ops | tk-core, tk-symmetry |
| **tk-contract** | ContractionGraph DAG, PathOptimizer trait, greedy/TreeSA heuristics, ContractionExecutor with reshape-free GEMM, conjugation flag propagation. Bosonic legs only (§6.4); fermionic swap gates deferred to Phase 5+ | tk-core, tk-symmetry, tk-linalg |
| **tk-dsl** | Index struct with unique IDs and prime levels, `hamiltonian!{}` proc-macro with span hygiene (§7.2.1) and diagnostic reporting (§7.2.2) — generates `OpSum` AST only, typed operator enums (`SpinOp`, `FermionOp`, `BosonOp`, `CustomOp`), OpSum builder, Lattice trait, snake-path mappers | tk-core, tk-symmetry |
| **tk-dmrg** | MPS/MPO types with typestate canonicality, `OpSum → MPO` SVD compression, two-site sweep engine, in-house Lanczos/Davidson/Block-Davidson eigensolvers, SVD truncation, site-tensor subspace expansion | tk-core, tk-symmetry, tk-linalg, tk-contract, tk-dsl |
| **tk-dmft** | Anderson Impurity Model mapping, bath discretization (Lanczos tridiagonalization), TDVP time evolution (with Tikhonov regularization and site-tensor subspace expansion), TEBD fallback, adaptive TDVP/Chebyshev solver selection, linear prediction (Levinson-Durbin default, SVD fallback), spectral positivity restoration, Chebyshev expansion, DMFT self-consistency loop, pinned-memory budget initialization for MPI | tk-dmrg (and transitively all) |
| **tk-python** | PyO3/maturin bindings with explicit GIL release, `AtomicBool`-based cancellation with `mpsc`-guarded monitor thread lifecycle (no GIL re-acquisition from Rayon workers), zero-copy NumPy via rust-numpy | tk-dmft |

**Key architectural constraint (cyclic dependency prevention):** `tk-dsl` generates only uncompressed `OpSum` structures and lattice geometry mappings. It has no dependency on `tk-linalg` or `tk-dmrg`. The `OpSum → MPO` compilation step (which requires SVD compression via `LinAlgBackend`) lives entirely within `tk-dmrg`, which has access to both `tk-dsl` and `tk-linalg`.

**Post-implementation note (dependency visibility):** `tk-dmrg` does not re-export its transitive dependencies. Downstream crates (tk-dmft, tk-python) that use types from tk-core, tk-symmetry, or tk-linalg must declare direct `[dependencies]` on those crates in their `Cargo.toml`, even though tk-dmrg already depends on them. This is standard Rust practice (no implicit transitive exports) but means each crate's `Cargo.toml` must list all directly-used upstream crates.

### 2.3 Feature Flags

| Feature Flag | Effect | Default | Exclusivity |
|:-------------|:-------|:--------|:------------|
| **backend-faer** | Enables DeviceFaer for dense SVD/EVD/QR using the pure-Rust faer crate | Yes | — |
| **backend-oxiblas** | Enables DeviceOxiblas for sparse BSR/CSR operations and extended-precision (f128) math. **Post-implementation note:** DeviceOxiblas is not yet implemented; `DefaultDevice` uses `DeviceAPI<DeviceFaer, DeviceFaer>` as interim | Yes | — |
| **backend-mkl** | Links Intel MKL via FFI for vendor-optimized BLAS on Intel hardware | No | Conflicts with `backend-openblas` |
| **backend-openblas** | Links OpenBLAS via FFI for broad HPC cluster compatibility | No | Conflicts with `backend-mkl` |
| **su2-symmetry** | Activates non-Abelian SU(2) support with Clebsch-Gordan caching (hand-rolled Racah formula, no external dependency) | No | — |
| **python-bindings** | Builds tk-python via PyO3/maturin for TRIQS integration | No | — |
| **parallel** | Enables Rayon-based data parallelism for element-wise tensor operations | Yes | — |
| **backend-cuda** | Enables DeviceCuda for GPU-accelerated GEMM (cuBLAS), SVD (cuSOLVER), and sparse ops (cuSPARSE); requires CUDA toolkit. Activates `PinnedMemoryTracker` for budget-managed pinned allocations | No | — |
| **backend-mpi** | Enables MPI-distributed block-sparse tensors and parallel DMFT loop via the mpi crate; requires system MPI library | No | — |

**FFI backend mutual exclusivity:** `backend-mkl` and `backend-openblas` both expose global BLAS symbols (`dgemm_`, `dsyev_`, etc.) through C/Fortran linkage. Enabling both simultaneously causes linker collisions. The workspace `Cargo.toml` enforces mutual exclusivity:

```toml
# In workspace Cargo.toml
[workspace.metadata.feature-constraints]
# Enforced by a build.rs check at the workspace root
mutually-exclusive = [["backend-mkl", "backend-openblas"]]
```

```rust
// In tk-linalg/build.rs
#[cfg(all(feature = "backend-mkl", feature = "backend-openblas"))]
compile_error!(
    "Features `backend-mkl` and `backend-openblas` are mutually exclusive. \
     Both expose global BLAS/LAPACK symbols and will cause linker collisions. \
     Enable only one FFI-based backend."
);
```

**PyPI distribution constraint:** Pre-built `tk-python` wheels for PyPI must be compiled strictly with pure-Rust backends (`backend-faer`, `backend-oxiblas`) to guarantee cross-platform compatibility (Linux/macOS/Windows) without requiring vendor BLAS installations.

---

## 3. Core Tensor Data Structure & Memory Management (tk-core)

The foundational design principle is a strict separation between tensor shape/stride metadata and contiguous memory storage. All tensor data resides as a single flat buffer, irrespective of dimensionality. Element offsets are computed via inner products of index coordinates and strides. This separation enables zero-copy view operations (transpose, permutation, slicing) that mutate only metadata.

**Scope discipline:** `tk-core` is the leaf crate upon which the entire workspace depends. To prevent compilation-cache invalidation cascading through the workspace, `tk-core` is strictly limited to: memory allocation (`SweepArena`, `PinnedMemoryTracker`), dimensional metadata (`TensorShape`, `TensorStorage<'a, T>`), the primary tensor type (`DenseTensor<'a, T>`), matrix view types (`MatRef`, `MatMut` with conjugation flags), the `Scalar` trait hierarchy, and shared error types. Mathematical operations on tensors (addition, trace, contraction) belong in `tk-linalg` or `tk-contract`.

### 3.1 Core Type Definitions

```rust
/// Dimensional metadata: shapes and strides for zero-copy views.
pub struct TensorShape {
    dims: SmallVec<[usize; 6]>,    // typical rank ≤ 6
    strides: SmallVec<[usize; 6]>,  // row-major by default
}

/// Copy-on-Write storage: a single enum merging the original TensorStorage
/// and TensorCow into one type. The Borrowed variant holds a direct &[T]
/// slice (not a reference to a wrapper struct), eliminating double indirection.
///
/// Post-implementation note: The v8.4 design used a separate TensorCow<'a, T>
/// wrapping &'a TensorStorage<T>. Implementation showed this indirection adds
/// complexity with no benefit. The tech spec's flatter design is adopted.
pub enum TensorStorage<'a, T: Scalar> {
    Owned(Vec<T>),          // materialized, heap-allocated data
    Borrowed(&'a [T]),      // zero-copy view into existing buffer
}

/// The primary dense tensor: shape metadata + owned/borrowed storage.
///
/// Post-implementation note: The lifetime parameter 'a is essential —
/// it surfaces the borrow lifetime to the type level so the borrow checker
/// can enforce arena safety. Every function returning a view (permute,
/// reshape, slice_axis) returns DenseTensor<'_, T>.
///
/// The `offset` field is essential for zero-copy `slice_axis`: slicing
/// accumulates offsets rather than creating new borrowed sub-slices,
/// enabling chained slicing without re-slicing complexity.
///
/// TempTensor<'a, T> (arena-allocated temporaries) is simply
/// DenseTensor<'a, T> with a shorter lifetime — not a distinct type.
pub struct DenseTensor<'a, T: Scalar> {
    shape: TensorShape,
    storage: TensorStorage<'a, T>,
    /// Byte offset into the storage buffer for the first logical element.
    /// Accumulates across chained slice_axis calls. as_slice() applies
    /// this offset; into_owned() gathers only logical elements when
    /// offset is nonzero.
    offset: usize,
}
```

**`Clone` requirement (cross-cutting issue):** `DenseTensor` must implement `Clone`. This is the #1 cross-cutting issue discovered during draft implementation — `Clone` is required by tk-contract (executor temporaries), tk-dsl (operator matrix copies), tk-dmrg (MPS site tensor duplication), and tk-dmft (spectral function copies). For `DenseTensor<'static, T>` (owned data), `Clone` clones the `Vec<T>`. For `DenseTensor<'a, T>` (borrowed), `Clone` produces another borrowed view with the same lifetime.

The `Scalar` trait constrains `T` to types supporting the required arithmetic: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, and optionally `f128` when the `backend-oxiblas` feature is active.

### 3.2 Matrix View Types with Lazy Conjugation

In quantum many-body physics, the Hermitian conjugate A† (transpose + complex conjugation) is far more common than the plain transpose Aᵀ. Eagerly conjugating a complex tensor before passing it to GEMM incurs an O(N) memory pass that starves the vector units before the multiply-add pipeline even begins. Instead, `MatRef` carries a boolean conjugation flag that is propagated to the BLAS micro-kernel, where conjugation is fused into the FMA instruction at zero cost.

```rust
/// A zero-copy matrix view used for GEMM and SVD dispatch.
/// The `is_conjugated` flag instructs the backend to treat the
/// underlying data as complex-conjugated without touching memory.
pub struct MatRef<'a, T: Scalar> {
    pub data: &'a [T],
    pub rows: usize,
    pub cols: usize,
    pub row_stride: isize,
    pub col_stride: isize,
    /// Hardware-level conjugation flag. If true, the backend treats
    /// data as A* (complex conjugated) during GEMM/SVD.
    /// For real T, this flag is ignored.
    pub is_conjugated: bool,
}

pub struct MatMut<'a, T: Scalar> {
    pub data: &'a mut [T],
    pub rows: usize,
    pub cols: usize,
    pub row_stride: isize,
    pub col_stride: isize,
}

impl<'a, T: Scalar> MatRef<'a, T> {
    /// Return a view that is mathematically the Hermitian conjugate (A†).
    /// Zero-copy: flips the conjugation flag and swaps strides.
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

    /// Return a conjugated view without transposing.
    #[inline(always)]
    pub fn conjugate(&self) -> MatRef<'a, T> {
        MatRef { is_conjugated: !self.is_conjugated, ..*self }
    }
}
```

When the `ContractionExecutor` encounters an index permutation corresponding to a Hermitian transpose, it simply calls `.adjoint()` on the `MatRef` — no allocation, no data movement. The conjugation flag flows through to `LinAlgBackend::gemm`, which maps it to the appropriate hardware instruction (§5.1).

### 3.3 Memory Management Strategy

DMRG sweeps perform thousands of contraction-SVD-truncation cycles per iteration. Naive heap allocation for each intermediate tensor causes severe fragmentation and allocator overhead. The architecture employs two complementary strategies:

#### 3.3.1 Arena Allocators

Temporary tensors within a single DMRG step are allocated from a pre-allocated memory arena (using the `bumpalo` crate). At the end of each sweep step, the arena's allocation pointer is reset to zero in O(1) time, entirely bypassing individual deallocation overhead. The arena is scoped to the sweep step via Rust's lifetime system, ensuring that no dangling references escape.

```rust
pub struct SweepArena {
    #[cfg(not(feature = "backend-cuda"))]
    inner: bumpalo::Bump,

    #[cfg(feature = "backend-cuda")]
    storage: ArenaStorage,
}

/// When backend-cuda is active, the arena dynamically chooses between
/// pinned (DMA-capable) and pageable memory based on global budget.
/// See §10.2 for PinnedMemoryTracker details.
#[cfg(feature = "backend-cuda")]
pub enum ArenaStorage {
    /// High-bandwidth, DMA-capable page-locked memory.
    Pinned(PinnedArena),
    /// Standard pageable heap memory (graceful fallback).
    Pageable(bumpalo::Bump),
}

impl SweepArena {
    pub fn alloc_tensor<'a, T: Scalar>(
        &'a self, shape: TensorShape
    ) -> DenseTensor<'a, T> { /* ... */ }

    /// O(1) reset: reclaims all arena memory.
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
}
```

#### 3.3.2 Arena Ownership Boundary: Temporary vs Persistent Tensors

Arena-allocated tensors carry a lifetime `'a` tied to the `SweepArena`. This creates a fundamental tension: intermediate contraction results (environments, Krylov workspace) are temporary and should live in the arena, but the final SVD output must be stored permanently in the `MPS` struct, which outlives the arena reset.

The architecture enforces an explicit ownership transfer at the sweep-step boundary:

```rust
/// Arena-allocated temporary: borrows from SweepArena.
/// Cannot outlive the arena's current allocation epoch.
/// Post-implementation note: TempTensor is just DenseTensor with a
/// shorter lifetime — not a distinct type.
pub type TempTensor<'a, T> = DenseTensor<'a, T>;

impl<'a, T: Scalar> DenseTensor<'a, T> {
    /// Create a borrowed view of this tensor's storage.
    /// The key enabler for zero-copy permute/reshape/slice_axis chains:
    /// even an Owned tensor produces a Borrowed view — the original Vec
    /// stays alive as long as the original tensor is alive, and the view
    /// borrows from it. This is why `tensor.permute(p1).slice_axis(0, 1, 3)`
    /// compiles: each operation borrows from the previous, and Rust's
    /// lifetime chain ensures the original data outlives all views.
    pub fn borrow_storage(&self) -> TensorStorage<'_, T> {
        match &self.storage {
            TensorStorage::Owned(v) => TensorStorage::Borrowed(v.as_slice()),
            TensorStorage::Borrowed(s) => TensorStorage::Borrowed(s),
        }
    }

    /// Materialize into heap-allocated owned storage.
    /// Called exactly once per sweep step on the final SVD output
    /// before SweepArena::reset() reclaims all temporaries.
    ///
    /// Three-way fast/slow path:
    /// 1. Move path: already owned, contiguous, offset 0 → zero-cost move
    /// 2. Memcpy path: contiguous with nonzero offset → single to_vec() on subslice
    /// 3. Gather path: non-contiguous strides → element-by-element gather
    ///    via gather_elements(), which uses row-major multi-index enumeration:
    ///    maintain a vec![0; rank] counter, compute sum(index[i] * strides[i])
    ///    for each element, increment the counter with carry from the last axis.
    ///    Cost: O(numel × rank) due to per-element multi-index arithmetic.
    ///    For rank-6 tensors this adds meaningful overhead vs the memcpy path.
    pub fn into_owned(self) -> DenseTensor<'static, T> {
        match self.storage {
            TensorStorage::Owned(data) if self.offset == 0 => DenseTensor {
                shape: self.shape,
                storage: TensorStorage::Owned(data),
                offset: 0,
            },
            _ => {
                let gathered = self.gather_elements();
                DenseTensor {
                    shape: TensorShape::contiguous(&self.shape.dims),
                    storage: TensorStorage::Owned(gathered),
                    offset: 0,
                }
            }
        }
    }
}
```

The rule is simple: everything computed within a DMRG step lives in the arena. The *only* outputs that escape are the updated site tensors, which must call `.into_owned()` before `SweepArena::reset()`. The borrow checker enforces this statically — any attempt to hold a borrowed arena reference past the reset point is a compile error, not a runtime bug. See §9, step 4 for the exact point in the data flow where this ownership transfer occurs.

#### 3.3.3 Copy-on-Write (Cow) Semantics

Shape-manipulation operations (transpose, permute, reshape) return `TensorStorage::Borrowed` views whenever the operation can be expressed as a pure stride permutation. Data is cloned into `TensorStorage::Owned` only when a contiguous memory layout is strictly required (e.g., as input to a GEMM kernel). This pattern, modeled after the rstsr framework, ensures copies are generated only when mathematically necessary.

### 3.4 The Scalar Trait Hierarchy

```rust
/// Post-implementation note: The original spec had only Add + Mul.
/// Draft implementation revealed that Sub, Neg, Debug, and 'static
/// are all required by downstream crates. The Real associated type
/// needs PartialOrd (truncation thresholds) and Float (epsilon, sqrt).
/// from_real_imag() is needed for complex number construction
/// (SpinOp::Sy matrix, Green's function assembly).
pub trait Scalar:
    Copy + Clone + Send + Sync + 'static + Debug
    + num::Zero + num::One
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
{
    type Real: Scalar + PartialOrd + num::Float;
    fn conj(self) -> Self;
    fn abs_sq(self) -> Self::Real;
    fn from_real(r: Self::Real) -> Self;
    /// Construct from real and imaginary parts.
    /// For real types, panics if imag != 0. Essential for building
    /// complex operator matrices (e.g., SpinOp::Sy = -i|↑⟩⟨↓| + i|↓⟩⟨↑|).
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self;
    /// Return the imaginary unit if this type supports complex numbers.
    /// Returns `Some(i)` for Complex<f32>/Complex<f64>, `None` for f32/f64.
    /// Useful for constructing operator matrices (e.g., SpinOp::Sy = -i|up><dn| + i|dn><up|)
    /// without branching on is_real().
    fn imaginary_unit() -> Option<Self>;
    /// Whether complex conjugation is a no-op for this type.
    /// Returns true for f32/f64, false for Complex<f32>/Complex<f64>.
    fn is_real() -> bool;
}
```

The `is_real()` method allows the contraction engine to skip setting `is_conjugated` entirely for real-valued models, avoiding unnecessary flag checks in tight loops.

**Operator overloading limitation (Rust orphan rules):** `std::ops::Mul<OpTerm<T>>` for arbitrary `T` cannot be implemented outside the crate defining `T`. In practice, operator overloading on `OpSum` (e.g., `J * op(SpinOp::Sz, i)`) is restricted to `f64` coefficients. Complex coefficients require explicit `OpSum::add_term(coeff, term)` calls.

---

## 4. Physical Symmetries & Block Sparsity (tk-symmetry)

In quantum systems with global symmetries, tensors become block-sparse: elements are non-zero only when the algebraic sum of incoming quantum numbers equals the outgoing quantum numbers (the "flux rule"). Exploiting this structure avoids storing and computing zeros, yielding order-of-magnitude speedups.

### 4.1 Quantum Number Trait & Bit-Packing

```rust
/// Post-implementation note: 'static bound required because QuantumNumber
/// appears in struct definitions needing owned storage.
pub trait QuantumNumber:
    Clone + Eq + Hash + Ord + Debug + Send + Sync + 'static
{
    fn identity() -> Self;
    fn fuse(&self, other: &Self) -> Self;
    fn dual(&self) -> Self;
}

/// Extension trait: compresses a quantum number into a fixed-width bitfield.
/// Enables O(log N) sector lookup via single-cycle u64 comparisons
/// instead of element-by-element SmallVec traversal.
/// Post-implementation note: Copy bound added — all quantum number types
/// (U1, Z2, U1Z2) are small value types that should be Copy.
pub trait BitPackable: QuantumNumber + Copy {
    /// Number of bits required to store this quantum number.
    const BIT_WIDTH: usize;
    /// Compress into the lower bits of a u64.
    fn pack(&self) -> u64;
    /// Reconstruct from the lower bits.
    fn unpack(bits: u64) -> Self;
}

/// U(1) charge conservation (e.g., particle number).
pub struct U1(pub i32);
impl QuantumNumber for U1 {
    fn identity() -> Self { U1(0) }
    fn fuse(&self, other: &Self) -> Self { U1(self.0 + other.0) }
    fn dual(&self) -> Self { U1(-self.0) }
}
impl BitPackable for U1 {
    const BIT_WIDTH: usize = 8; // supports charges -128..+127
    #[inline(always)]
    fn pack(&self) -> u64 { (self.0 as u8) as u64 }
    #[inline(always)]
    fn unpack(bits: u64) -> Self { U1(((bits & 0xFF) as u8) as i8 as i32) }
}

/// Z₂ parity conservation (e.g., fermion parity).
pub struct Z2(pub bool);
impl QuantumNumber for Z2 {
    fn identity() -> Self { Z2(false) }
    fn fuse(&self, other: &Self) -> Self { Z2(self.0 ^ other.0) }
    fn dual(&self) -> Self { self.clone() }
}
impl BitPackable for Z2 {
    const BIT_WIDTH: usize = 1;
    #[inline(always)]
    fn pack(&self) -> u64 { if self.0 { 1 } else { 0 } }
    #[inline(always)]
    fn unpack(bits: u64) -> Self { Z2(bits & 1 == 1) }
}

/// Composite symmetry: U(1) ⊗ Z₂.
pub struct U1Z2(pub U1, pub Z2);
impl QuantumNumber for U1Z2 {
    fn identity() -> Self { U1Z2(U1::identity(), Z2::identity()) }
    fn fuse(&self, other: &Self) -> Self {
        U1Z2(self.0.fuse(&other.0), self.1.fuse(&other.1))
    }
    fn dual(&self) -> Self { U1Z2(self.0.dual(), self.1.dual()) }
}
impl BitPackable for U1Z2 {
    const BIT_WIDTH: usize = 9; // 8 bits for U1 + 1 bit for Z2
    #[inline(always)]
    fn pack(&self) -> u64 {
        self.0.pack() | (self.1.pack() << U1::BIT_WIDTH)
    }
    #[inline(always)]
    fn unpack(bits: u64) -> Self {
        U1Z2(U1::unpack(bits), Z2::unpack(bits >> U1::BIT_WIDTH))
    }
}
```

### 4.2 Block-Sparse Tensor Architecture with Packed Sector Keys

The block-sparse tensor stores data as a collection of dense sub-blocks using a Structure-of-Arrays (SoA) layout. For Abelian symmetries implementing `BitPackable`, sector keys are compressed into single `u64` integers, enabling LLVM-vectorized binary search with zero pointer chasing and zero branch misprediction.

```rust
/// Bit-packed sector key: the entire quantum-number tuple for one block
/// compressed into a single u64. Binary search over Vec<PackedSectorKey>
/// resolves in nanoseconds with no pointer chasing.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PackedSectorKey(pub u64);

impl PackedSectorKey {
    /// Pack a slice of quantum numbers into a single u64.
    /// Panics in debug mode if the total bits exceed 64.
    pub fn pack<Q: BitPackable>(qns: &[Q]) -> Self {
        debug_assert!(qns.len() * Q::BIT_WIDTH <= 64,
            "Tensor rank too high to pack into u64; use u128 variant");
        let mut packed: u64 = 0;
        for (i, q) in qns.iter().enumerate() {
            let shift = i * Q::BIT_WIDTH;
            let mask = (1u64 << Q::BIT_WIDTH) - 1;
            packed |= (q.pack() & mask) << shift;
        }
        PackedSectorKey(packed)
    }

    /// Unpack back into a SmallVec (for debugging or structural operations).
    pub fn unpack<Q: BitPackable>(&self, rank: usize) -> SmallVec<[Q; 8]> {
        let mask = (1u64 << Q::BIT_WIDTH) - 1;
        (0..rank).map(|i| Q::unpack((self.0 >> (i * Q::BIT_WIDTH)) & mask))
            .collect()
    }
}

/// Post-implementation note: LegDirection is essential for the flux rule.
/// Each tensor leg is either incoming or outgoing; the flux rule sums
/// incoming quantum numbers and subtracts outgoing ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegDirection { Incoming, Outgoing }

/// INVARIANT: sector_keys is always sorted for O(log N) binary search.
/// Keys are contiguous u64s in a cache-friendly array; the CPU prefetcher
/// pulls the entire key array into L1 cache, and search resolves via
/// single-cycle register comparisons.
///
/// Post-implementation notes:
/// - leg_directions field added (omitted in original spec). Essential for
///   flux rule enforcement in contractions and permutations.
/// - Blocks are always DenseTensor<'static, T> (owned, no arena-borrowed blocks).
/// - permute() is NOT zero-copy: requires into_owned() on each block to
///   ensure BLAS-compatible contiguous layout.
/// - BlockSparseTensor must implement Clone (required by tk-contract,
///   tk-dmrg, tk-dmft).
pub struct BlockSparseTensor<T: Scalar, Q: BitPackable> {
    indices: Vec<QIndex<Q>>,
    leg_directions: Vec<LegDirection>,
    sector_keys: Vec<PackedSectorKey>,
    sector_blocks: Vec<DenseTensor<'static, T>>,
    flux: Q,
}

impl<T: Scalar, Q: BitPackable> BlockSparseTensor<T, Q> {
    /// O(log N) lookup running entirely in registers.
    #[inline(always)]
    pub fn get_block(&self, target_qns: &[Q]) -> Option<&DenseTensor<T>> {
        let search_key = PackedSectorKey::pack(target_qns);
        self.sector_keys
            .binary_search(&search_key)
            .ok()
            .map(|idx| &self.sector_blocks[idx])
    }

    /// Iterate over (key, block) pairs. Heavily used in fuse_legs, split_leg,
    /// flatten, and block_gemm task generation. More ergonomic than separate
    /// sector_keys()/sector_block(key) lookups and avoids repeated key searches.
    pub fn iter_keyed_blocks(&self) -> impl Iterator<Item = (PackedSectorKey, &DenseTensor<'static, T>)> {
        self.sector_keys.iter().copied().zip(self.sector_blocks.iter())
    }
}
```

**`fuse_legs` algorithm complexity:** The `fuse_legs` operation combines adjacent tensor legs into a single fused leg via Cartesian product of quantum numbers. The implementation has three phases: (1) Cartesian product enumeration — iterates all combinations of sector indices on the fused legs to produce fused quantum numbers, cost O(product of sector counts on fused legs); (2) BTreeMap-based offset mapping — maps each fused quantum number to a dimension offset, using `BTreeMap` (not `HashMap`) for deterministic ordering that ensures reproducible DMRG sweeps across runs; (3) block scatter — for each original sector, determines the fused quantum number, looks up the offset, and copies data into the fused block at the correct position.

**`split_leg` requires `original_directions` parameter:** The `split_leg(leg_index, original_indices, original_directions)` method takes an additional `original_directions: &[LegDirection]` parameter beyond what the original spec showed. This is needed because fusing loses per-sub-leg direction information — the fused leg has a single direction, but the original sub-legs may have had mixed directions (`Incoming`/`Outgoing`), and unfusing must reconstruct the correct flux rule for each sub-leg.

**Capacity note:** With 8 tensor legs and 8 bits per quantum number (the U(1) default), bit-packing fits in exactly 64 bits. For models requiring larger charge range or higher-rank tensors (e.g., multi-orbital Hubbard with U(1)\_charge ⊗ U(1)\_spin, rank > 8), the `PackedSectorKey` can be promoted to `u128`, giving 16 bits per leg at rank 8 or supporting rank 16 at 8 bits per leg.

**Non-Abelian fallback:** SU(2) irreps are not trivially bit-packable due to multiplicity structure. The `BlockSparseTensor` definition is parameterized over `Q: BitPackable` for the Abelian fast path. The SU(2) extension (§4.4) uses its own `WignerEckartTensor` with `SU2Irrep`-keyed storage, not constrained to `BitPackable`.

**Flat-buffer block storage refactor (Phase 4) — Dual-Layout Strategy:** The current layout stores each block as an independent `DenseTensor<T>`, which is adequate for CPU-only Phases 1–3 (blocks are large enough that per-allocation overhead is negligible). However, for GPU transfers in Phase 5, hundreds of individually-allocated blocks would require either hundreds of small `cudaMemcpyAsync` calls (terrible PCIe utilization) or manual gathering into a staging buffer.

The architecture distinguishes two storage layouts with distinct roles:

**Mutation Layout (fragmented, default):** The current `Vec<DenseTensor<T>>` storage, where each block is an independent heap allocation. This layout is optimal for *structural mutations* — appending columns to a sector block during TDVP subspace expansion (§8.1.1) is O(D_sector²) because only the affected block is reallocated; downstream blocks are untouched.

**Compute Layout (contiguous, read-only):** A single flat buffer with an offset table, optimized for GPU DMA and cache-friendly GEMM dispatch. This layout is *read-only* — structural mutations would require shifting all downstream byte offsets, transforming an O(D_sector²) operation into an O(D_total²) memory copy.

```rust
/// Compute-side read-only block storage: all sector data in one contiguous allocation.
/// Enables single-DMA GPU transfer of the entire tensor.
/// NOT used during structural mutations (subspace expansion); see mutation layout.
pub struct FlatBlockStorage<'a, T: Scalar> {
    /// Single contiguous buffer containing all sector blocks back-to-back.
    /// Allocated from SweepArena (pinned memory when backend-cuda is active),
    /// NOT from the pageable heap. This guarantees DMA-capable memory for
    /// GPU transfers without the NVIDIA driver's hidden pin-copy-unpin dance.
    data: &'a mut [T],
    /// Start index of each sector block within `data`.
    /// offsets[i] is the start index of sector_keys[i]'s block data.
    offsets: Vec<usize>,
    /// Dimensions (rows, cols) of each sector block.
    shapes: Vec<(usize, usize)>,
}

impl<T: Scalar, Q: BitPackable> BlockSparseTensor<T, Q> {
    /// Pack fragmented blocks into a contiguous flat buffer for GPU/GEMM.
    /// Called after structural mutations are complete, before dispatch.
    ///
    /// CRITICAL: The flat buffer is allocated from the SweepArena, NOT from
    /// fresh pageable heap memory. When `backend-cuda` is active, the arena
    /// uses pinned memory (§10.2.1), so the resulting buffer is directly
    /// DMA-capable — no hidden staging copy by the NVIDIA driver.
    /// If flatten() instead called Vec::with_capacity() on the pageable heap,
    /// the driver would silently allocate a pinned staging buffer, copy into it,
    /// then DMA from the staging buffer — halving effective PCI-e bandwidth
    /// and negating the entire purpose of the flat-buffer refactor.
    ///
    /// Cost: O(D_total²) — a single memcpy pass, negligible relative to
    /// the O(D³) GEMM it feeds.
    pub fn flatten<'a>(&self, arena: &'a SweepArena) -> FlatBlockStorage<'a, T> {
        let total_elems = self.sector_blocks.iter().map(|b| b.num_elements()).sum();
        let buf = arena.alloc_slice::<T>(total_elems);
        // Copy each block into the contiguous arena buffer.
        let mut offset = 0;
        let mut offsets = Vec::with_capacity(self.sector_blocks.len());
        let mut shapes = Vec::with_capacity(self.sector_blocks.len());
        for block in &self.sector_blocks {
            offsets.push(offset);
            shapes.push((block.rows(), block.cols()));
            buf[offset..offset + block.num_elements()].copy_from_slice(block.as_slice());
            offset += block.num_elements();
        }
        FlatBlockStorage { data: buf, offsets, shapes }
    }

    /// Restore fragmented layout from flat buffer (e.g., after GPU computation).
    pub fn unflatten(flat: &FlatBlockStorage<T>, keys: &[PackedSectorKey]) -> Self { /* ... */ }
}
```

The transition flow during a TDVP step is: (1) subspace expansion mutates A_L in fragmented layout → (2) `A_L.flatten(&arena)` packs into the arena's pinned memory → (3) GEMM/GPU operates on the flat buffer (DMA-direct, no staging copy) → (4) results unflattened back to mutable layout for the next step → (5) `SweepArena::reset()` reclaims the flat buffer in O(1). The `SparseLinAlgBackend` trait takes `&BlockSparseTensor` as an opaque input, so the internal layout switching is invisible to callers.

### 4.3 Sparsity Format Summary

| Strategy | Format | Application | Benefit |
|:---------|:-------|:------------|:--------|
| **Dense** | Contiguous 1D | Non-symmetric models | Max BLAS throughput, SIMD |
| **Block-Sparse** | BSR / Packed flat arrays | Abelian U(1), Z₂ | Skip zeros, preserve symmetry |
| **Element-Sparse** | CSR / CSC | Irregular geometry | Memory-efficient for extreme sparsity |
| **Diagonal** | DIA | Identity / local ops | Zero-overhead storage |

### 4.4 Non-Abelian Symmetry Roadmap (SU(2))

```rust
#[cfg(feature = "su2-symmetry")]
pub struct SU2Irrep { pub twice_j: u32 }

#[cfg(feature = "su2-symmetry")]
pub struct WignerEckartTensor<T: Scalar> {
    structural: ClebschGordanCache,
    reduced: BlockSparseTensor<T, SU2Irrep>,
}
```

**Design-forward consideration:** The core contraction engine includes an optional `structural_contraction` callback from day one. The Abelian code path passes a no-op (zero overhead); the SU(2) code path injects 6j/9j symbol evaluations. This prevents retroactive redesign when non-Abelian support is added.

**Note:** `SU2Irrep` implements `QuantumNumber` but not `BitPackable`. Its `BlockSparseTensor` uses a separate `SmallVec`-keyed variant behind the `su2-symmetry` feature flag. This is acceptable because non-Abelian contractions are dominated by Clebsch-Gordan coefficient evaluation, not sector lookup.

**Post-implementation note (CG coefficients):** The `ClebschGordanCache` uses a hand-rolled Racah formula for computing Clebsch-Gordan coefficients, rather than depending on the `lie-groups` crate. This eliminates an external dependency for a relatively small amount of well-known numerical code. The cache uses `DashMap`-based thread-safe lazy caching for concurrent access from Rayon workers, with exact rational arithmetic via factorials and standard CG series summation. Triangle inequality and selection rules are validated before computation.

**Known refactoring requirement (fusion-rule multiplicity):** The Abelian block-sparse GEMM (§5.3.1) assumes a one-to-one mapping in `compute_fusion_rule(*key_a, *key_b) -> Option<PackedSectorKey>`: each input sector pair produces at most one output sector. For SU(2), the tensor product of irreps yields *multiple* output sectors: j₁ ⊗ j₂ = |j₁−j₂| ⊕ (|j₁−j₂|+1) ⊕ ... ⊕ (j₁+j₂). The `SectorGemmTask` generation loop in §5.3.1 must be generalized to produce a `Vec<SectorGemmTask>` per input pair, with each task's output block weighted by the corresponding Clebsch-Gordan coefficient. The `structural_contraction` callback handles the coefficient evaluation, but the task-generation fan-out is a structural change to the LPT scheduling phase. This refactoring is scoped to the `su2-symmetry` feature flag and does not affect the Abelian code path.

**Output-sector collision hazard:** Because multiple input pairs (j_a, j_b) can map to the *same* output sector j_c, a naive `Vec<SectorGemmTask>` fed to `par_iter` creates a data race: two threads writing to the same output block. The SU(2) task generation must use a map-reduce pattern — group tasks by output sector key, then accumulate (reduce) partial contributions within each group before writing the final block. This is a structural change to the LPT scheduling phase that does not affect the Abelian code path, where fusion is always one-to-one.

**Task generation memory bound:** The combinatorial fan-out of the map-reduce pattern is bounded by the physics. In SU(2)-symmetric DMRG, the number of distinct irreps at a given bond (D_reduced) is typically 10–50 even at large total bond dimensions, because many states share the same j label. The fan-out per input pair is at most (2·j_max + 1) output sectors. The total task count before reduction is therefore O(D_reduced² × j_max). For j_max = 10 and D_reduced = 50, this yields ~250,000 tasks at ~64 bytes each ≈ 16 MB — well within L3 cache and posing no allocation pressure. The task vector should be pre-allocated with `Vec::with_capacity(d_reduced * d_reduced * (2 * j_max + 1))` to avoid incremental reallocation during the generation loop. For exotic high-spin models with j_max > 50 or D_reduced > 200, a streaming/chunked reduction should be considered, but this is outside the scope of Phase 5 targets.

**Multiplet-aware SVD truncation:** In SU(2)-symmetric DMRG, singular values come in degenerate multiplets of dimension 2j+1. Truncation must keep or discard entire multiplets — splitting a multiplet explicitly breaks the symmetry and crashes the simulation. The truncation error must be weighted by multiplet dimension: each singular value σ_j contributes (2j+1)·σ_j² to the discarded weight. The current `svd_truncated` logic (which slices at `max_bond_dim` without multiplet awareness) must be extended with a two-phase approach: (1) sort by σ, (2) snap the truncation boundary to the nearest multiplet edge. This is scoped to the `su2-symmetry` feature flag.

---

## 5. Abstract Linear Algebra Backend (tk-linalg)

DMRG performance is bottlenecked by three operations: GEMM for tensor contraction, SVD for basis truncation, and EVD for ground-state targeting. The tk-linalg crate abstracts these behind trait interfaces, allowing compile-time backend selection.

### 5.1 Backend Trait Definitions

```rust
/// Object-safe linear algebra backend trait.
/// The scalar type T is a trait-level parameter (not per-method generic),
/// enabling `Box<dyn LinAlgBackend<f64>>` for dynamic dispatch at the
/// sweep-scheduler level while preserving static dispatch in inner loops.
///
/// Rust's object-safety rules (E0038) prohibit traits with generic methods
/// from being used as trait objects. By parameterizing the trait itself
/// over T, all methods are concrete for a given T, and the vtable is
/// well-formed.
pub trait LinAlgBackend<T: Scalar>: Send + Sync {
    /// Default implementation uses divide-and-conquer SVD (gesdd).
    /// Falls back to QR-iteration SVD (gesvd) on convergence failure.
    fn svd_truncated(
        &self, mat: &MatRef<T>, max_rank: usize, cutoff: T::Real,
    ) -> SvdResult<T> {
        match self.svd_truncated_gesdd(mat, max_rank, cutoff) {
            Ok(result) => result,
            Err(SvdConvergenceError) => {
                log::warn!("gesdd failed to converge; falling back to gesvd");
                self.svd_truncated_gesvd(mat, max_rank, cutoff)
            }
        }
    }

    fn svd_truncated_gesdd(
        &self, mat: &MatRef<T>, max_rank: usize, cutoff: T::Real,
    ) -> Result<SvdResult<T>, SvdConvergenceError>;

    fn svd_truncated_gesvd(
        &self, mat: &MatRef<T>, max_rank: usize, cutoff: T::Real,
    ) -> SvdResult<T>;

    /// Conjugation-aware GEMM: C = α·op(A)·op(B) + β·C.
    /// The `is_conjugated` flags on MatRef<A> and MatRef<B> determine
    /// whether op(X) = X, X*, Xᵀ, or X† at the hardware level.
    /// For real T, conjugation flags are ignored.
    fn gemm(
        &self, alpha: T, a: &MatRef<T>, b: &MatRef<T>,
        beta: T, c: &mut MatMut<T>,
    );

    /// Post-implementation note: Return types are DenseTensor<'static, T>
    /// (always owned). SVD/eigh/QR results are freshly allocated.
    fn eigh_lowest(&self, mat: &MatRef<T>, k: usize) -> (Vec<T::Real>, DenseTensor<'static, T>);
    fn qr(&self, mat: &MatRef<T>) -> (DenseTensor<'static, T>, DenseTensor<'static, T>);

    /// Tikhonov-regularized pseudo-inverse for TDVP gauge restoration.
    /// Computes s_i / (s_i² + δ²) instead of 1/s_i, preventing NaN
    /// explosion when singular values approach machine zero.
    /// See §8.1 for usage context.
    ///
    /// Post-implementation note: `where Self: Sized` is required because
    /// this method has a default implementation, which breaks object safety
    /// for this method only. Callers using `dyn LinAlgBackend<T>` cannot
    /// call this method through the trait object.
    fn regularized_svd_inverse(
        &self,
        s_values: &[T::Real],
        u: &DenseTensor<'static, T>,
        vt: &DenseTensor<'static, T>,
        cutoff: T::Real,  // δ: the Tikhonov parameter
    ) -> DenseTensor<'static, T> where Self: Sized {
        let mut inv_s = Vec::with_capacity(s_values.len());
        let delta_sq = cutoff * cutoff;
        for &s in s_values {
            // Tikhonov regularization: safe even when s → 0
            inv_s.push(s / (s * s + delta_sq));
        }
        // Reconstruct: V · diag(inv_s) · U†
        construct_inverse_matrix(u, &inv_s, vt)
    }
}

/// Object-safe sparse backend trait, parameterized over both scalar
/// and quantum number types. Enables `Box<dyn SparseLinAlgBackend<f64, U1>>`.
pub trait SparseLinAlgBackend<T: Scalar, Q: BitPackable>: LinAlgBackend<T> {
    fn spmv(&self, a: &BlockSparseTensor<T, Q>, x: &[T], y: &mut [T]);

    /// Block-sparse GEMM with partitioned LPT scheduling.
    /// Tasks are sorted by descending FLOP cost, then partitioned:
    /// heavy head → multithreaded BLAS; light tail → Rayon. See §5.3.1.
    fn block_gemm(
        &self, a: &BlockSparseTensor<T, Q>, b: &BlockSparseTensor<T, Q>,
    ) -> BlockSparseTensor<T, Q>;
}
```

**SVD algorithm selection rationale:** The divide-and-conquer algorithm (`gesdd`) is substantially faster than QR-iteration (`gesvd`) for the moderately sized dense matrices arising from DMRG two-site updates (typically 2D² × 2D², where D is bond dimension 100–2000). The trade-off is higher workspace memory (O(n²) vs O(n)), which is acceptable given the arena allocation strategy. Convergence failure with `gesdd` is rare but can occur with highly degenerate singular values; the automatic fallback to `gesvd` handles this gracefully.

**Post-implementation notes (faer backend):**
- **SVD ordering correctness trap:** faer's `thin_svd()` returns singular values in *ascending* order (smallest first). DMRG truncation requires descending order (largest first) to correctly discard small singular values. The `svd_truncated` wrapper must re-sort the singular values and permute the corresponding U/V columns into descending order. Failing to do so silently keeps the *wrong* singular values, producing subtly incorrect physics.
- **`gesdd`→`gesvd` fallback is a no-op with faer:** The fallback mechanism is only meaningful when using MKL or OpenBLAS FFI backends, where `gesdd` and `gesvd` are distinct LAPACK routines. The pure-Rust faer crate uses a single SVD algorithm; the fallback path simply re-invokes the same code.

**Silent inaccuracy guard:** `gesdd` can occasionally return without signaling an error while producing inaccurate small singular values for pathologically ill-conditioned matrices. The architecture employs two layers of validation:

**Layer 1 — Debug builds (every call):** In debug and test builds, the `svd_truncated` wrapper validates the reconstruction residual after every call:

```rust
// Inside svd_truncated default implementation, after obtaining `result`:
debug_assert!({
    let residual = reconstruction_error(mat, &result);  // ‖A − UΣV†‖_F
    let norm = frobenius_norm(mat);                      // ‖A‖_F
    residual / norm < 1e-10
}, "SVD reconstruction residual {:.2e} exceeds tolerance", residual / norm);
```

This catches silently corrupt SVD results during development and CI, before the Tikhonov regularization masks the damage downstream. Compiled out in release builds (`--release`).

**Layer 2 — Release builds (periodic sampling):** In multi-week production DMFT runs, `gesdd` silent failure is rare but real. If the debug check is entirely stripped, corrupt small singular values are seamlessly fed into the Tikhonov-regularized pseudo-inverse — the regularization stabilizes the numerics, but the physics silently drifts. To catch this, the release-mode `svd_truncated` wrapper performs the full residual check every K-th call, where K is controlled by `DMRGConfig::svd_validation_interval` (default: 1000).

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global SVD call counter for periodic release-mode validation.
static SVD_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

// Inside svd_truncated, after obtaining `result` (release mode):
let count = SVD_CALL_COUNT.fetch_add(1, Ordering::Relaxed);
if count % config.svd_validation_interval == 0 {
    let residual = reconstruction_error(mat, &result);
    let norm = frobenius_norm(mat);
    if residual / norm > 1e-8 {
        log::error!(
            target: "tensorkraft::telemetry",
            "SVD_SILENT_CORRUPTION: gesdd reconstruction residual {:.2e} \
             at SVD call #{} (checked every {} calls). Singular values may be \
             inaccurate. Falling back to gesvd for this and subsequent calls.",
            residual / norm, count, config.svd_validation_interval
        );
        // Permanently switch this backend instance to gesvd.
        self.force_gesvd_fallback();
    }
}
```

The residual check has the same O(mn·min(m,n)) complexity as the SVD itself. At K=1000, the amortized overhead is 0.1% of total SVD cost — undetectable in profiling. Upon detection of corruption, the backend permanently falls back to `gesvd` for the remainder of the run, preventing further silent failures.

**Layer 3 — TDVP physics-triggered validation (dynamic):** The periodic modulo counter has a blind spot: if `gesdd` corruption occurs at call #1001, the system runs with corrupted data for 999 calls before detection at #2000. For DMRG sweeps, this wastes a recoverable sweep. For TDVP, it is catastrophic — corrupted singular values fed into the Tikhonov-regularized gauge restoration compound exponentially through the matrix exponential across 999 time steps, irreparably destroying the MPS tangent space. The physical state is unrecoverable by the time the modulo counter triggers.

To close this gap, the `TdvpDriver` monitors two physical observables that serve as canary signals for SVD corruption:

1. **Truncation error spike:** If the SVD truncation error at bond `i` jumps by more than `svd_anomaly_factor` (default: 10×) relative to the previous time step at the same bond, an immediate out-of-band SVD residual check is forced on the next SVD call.
2. **Energy variance spike:** If the energy expectation value changes by more than `svd_anomaly_energy_tol` (default: 1e-4 in natural units) between consecutive TDVP steps without a corresponding change in the Hamiltonian, an immediate check is forced.

These checks are O(1) — two scalar comparisons per time step. The expensive O(mn·min(m,n)) residual computation is triggered only when physics signals an anomaly, making the amortized cost negligible even for long TDVP runs.

```rust
pub struct SvdAnomalyDetector {
    /// If truncation error jumps by more than this factor between consecutive
    /// TDVP steps at the same bond, force immediate SVD validation.
    pub svd_anomaly_factor: f64,     // default: 10.0
    /// If |ΔE| between consecutive TDVP steps exceeds this tolerance
    /// (without a Hamiltonian change), force immediate SVD validation.
    pub svd_anomaly_energy_tol: f64, // default: 1e-4
    /// Previous step's truncation errors per bond (for comparison).
    prev_truncation_errors: Vec<f64>,
    /// Previous step's energy.
    prev_energy: f64,
    /// Flag: when set, the next SVD call forces a residual check.
    pub force_next_validation: bool,
}
```

The three layers are complementary: Layer 1 (debug, every call) catches bugs during development; Layer 2 (release, periodic K-th call) provides background coverage in production; Layer 3 (physics-triggered) catches acute TDVP corruption the moment it impacts observables. For pure DMRG workloads (no TDVP), Layer 3 is inactive and adds zero overhead.

### 5.2 Backend Implementations

| Backend Struct | Feature Flag | Characteristics |
|:---------------|:-------------|:----------------|
| **DeviceFaer** | backend-faer (default) | Pure Rust; state-of-the-art multithreaded SVD; native lazy conjugation via `faer::MatRef::conjugate()`; ideal for high bond-dimension DMRG |
| **DeviceOxiblas** | backend-oxiblas (default) | Pure Rust; explicit SIMD (AVX-512, NEON); 9 sparse formats; f128 extended precision |
| **DeviceMKL** | backend-mkl | FFI to Intel MKL; `CblasConjTrans` support via `cblas_zgemm`; vendor-optimized for Xeon |
| **DeviceOpenBLAS** | backend-openblas | FFI to OpenBLAS; `CblasConjTrans` support; broad HPC cluster compatibility |

#### 5.2.1 DeviceFaer Conjugation-Aware GEMM

The pure-Rust `faer` crate natively supports lazy conjugation. Calling `.conjugate()` on a `faer::MatRef` flips a bit in the view structure without touching data; the SIMD micro-kernels automatically negate imaginary parts during FMA instructions.

```rust
pub struct DeviceFaer;

/// DeviceFaer implements LinAlgBackend<T> for each supported scalar type.
/// The trait-level T parameter makes this object-safe:
/// `Box<dyn LinAlgBackend<f64>>` compiles; the old per-method generic did not.
impl LinAlgBackend<f64> for DeviceFaer {
    fn gemm(
        &self, alpha: f64, a: &MatRef<'_, f64>, b: &MatRef<'_, f64>,
        beta: f64, c: &mut MatMut<'_, f64>,
    ) {
        // Construct base faer views using custom strides
        let faer_a = faer::mat::from_slice_with_strides(
            a.data, a.rows, a.cols, a.row_stride, a.col_stride,
        );
        let faer_b = faer::mat::from_slice_with_strides(
            b.data, b.rows, b.cols, b.row_stride, b.col_stride,
        );
        let mut faer_c = faer::mat::from_slice_mut_with_strides(
            c.data, c.rows, c.cols, c.row_stride, c.col_stride,
        );

        // Zero-copy conjugation: faer's .conjugate() flips a bit,
        // the SIMD micro-kernels handle the rest during FMA.
        let a_op = if a.is_conjugated { faer_a.conjugate() } else { faer_a };
        let b_op = if b.is_conjugated { faer_b.conjugate() } else { faer_b };

        faer::linalg::matmul::matmul(
            faer_c.as_mut(), a_op, b_op,
            Some(beta), alpha,
            faer::Parallelism::Rayon(0),
        );
    }

    // ... svd_truncated_gesdd, svd_truncated_gesvd, eigh_lowest, qr ...
}

// Repeat for Complex<f64> — the impl body is identical modulo type.
// In practice, a macro_rules! generates both impls from one template.
impl LinAlgBackend<Complex64> for DeviceFaer { /* ... */ }
```

#### 5.2.2 C-BLAS Conjugation Dispatch (MKL / OpenBLAS)

Standard BLAS `cblas_zgemm` accepts `CblasTrans` and `CblasConjTrans` flags. The stride layout and conjugation flag on `MatRef` are mapped to the appropriate `CBLAS_TRANSPOSE` enum variant:

```rust
impl DeviceMKL {
    /// Maps strided MatRef to BLAS transposition flags and leading dimensions.
    /// Not generic over T — called from the concrete impl LinAlgBackend<f64>.
    #[inline(always)]
    fn resolve_blas_layout(mat: &MatRef<'_, f64>) -> (CBLAS_TRANSPOSE, i32) {
        if mat.row_stride == 1 {
            let trans = if mat.is_conjugated {
                CBLAS_TRANSPOSE::CblasConjNoTrans
            } else {
                CBLAS_TRANSPOSE::CblasNoTrans
            };
            (trans, mat.col_stride as i32)
        } else if mat.col_stride == 1 {
            let trans = if mat.is_conjugated {
                CBLAS_TRANSPOSE::CblasConjTrans
            } else {
                CBLAS_TRANSPOSE::CblasTrans
            };
            (trans, mat.row_stride as i32)
        } else {
            panic!("BLAS backend requires at least one unit stride. \
                    Use faer or tblis for arbitrary strides.");
        }
    }
}
```

```rust
pub struct DeviceAPI<D, S> { dense: D, sparse: S }

/// Post-implementation note: Interim Phase 1-3 default uses DeviceFaer
/// for both dense and sparse backends (DeviceOxiblas integration deferred).
#[cfg(all(feature = "backend-faer", feature = "backend-oxiblas"))]
pub type DefaultDevice = DeviceAPI<DeviceFaer, DeviceFaer>;

/// DefaultDevice delegates LinAlgBackend<T> to its dense component.
impl<T: Scalar> LinAlgBackend<T> for DefaultDevice
where DeviceFaer: LinAlgBackend<T> { /* delegate to self.dense */ }
```

### 5.3 Hybrid Parallelism & Heterogeneous Dispatch

**Post-implementation note (Phase 1–3 status):** The full partitioned LPT scheduler described below is the target architecture. The draft implementation uses a simpler binary heuristic (threshold-based switch between single-threaded Rayon dispatch and multithreaded BLAS) which is adequate for Phase 1–3 workloads. The partitioned scheduler will be implemented as part of the Phase 4 performance optimization pass.

Mixing Rayon's work-stealing scheduler with multithreaded BLAS backends creates severe resource contention if not carefully orchestrated. Rather than using a rigid binary switch based on overall tensor topology, the architecture partitions the Longest Processing Time (LPT) queue dynamically.

**The "middle ground" pathology:** A binary regime switch (e.g., "if `max_sector_dim > 500` then single-threaded BLAS" or "if `n_sectors < n_cores` then multithreaded BLAS") fails catastrophically for the most common DMRG workload: a U(1)-symmetric tensor at moderate bond dimension (D = 500–1000) where 3–5 large sectors near half-filling coexist with dozens of tiny sectors at the wings of the binomial distribution. Routing the entire batch to one regime abandons either BLAS multithreading on the heavy sectors or Rayon parallelism on the light tail. The partitioned approach eliminates this cliff.

**1. Partitioned LPT CPU Dispatch:**
Tasks are generated and sorted by descending FLOP cost. Instead of routing the *entire* batch to either Rayon or multithreaded BLAS, the queue is split at a calibrated `blas_flop_threshold`:

* **The Head (Fat Sectors):** The heaviest GEMMs are executed sequentially by the main thread, granting the underlying BLAS backend (faer/OpenBLAS) the full machine thread pool to parallelize the internal matrix multiplications.
* **The Tail (Fragmented Sectors):** Once the massive blocks are computed, the remaining blocks are handed to Rayon's `par_iter()`. During this phase, BLAS threading is explicitly forced to 1, and Rayon utilizes all cores to clear the long tail.

**BLAS thread-count safety caveat:** The `set_blas_threads()` calls (`openblas_set_num_threads`, `MKL_Set_Num_Threads`) mutate *global* process state. This is safe in the partitioned scheduler because phases 3a and 3b are strictly sequential — no concurrent callers exist. However, for the FFI backends (MKL, OpenBLAS), the global mutation is inherently incompatible with nested parallelism or concurrent BLAS consumers in the same process (e.g., a Python NumPy call from another thread). On newer MKL versions (≥2020), `mkl_set_num_threads_local` provides a thread-local override and is the preferred mechanism when available. The default pure-Rust `faer` backend avoids this problem entirely: its parallelism is controlled per-call via the `faer::Parallelism::Rayon(n)` argument, which is stack-local and requires no global state mutation.

**2. BLAS Crossover Calibration:**
The `blas_flop_threshold` determines which tasks are "heavy enough" to justify full BLAS multithreading. A hardcoded default (e.g., 250 MFLOPs) is provided, but the actual crossover point depends on the host machine's BLAS implementation, core count, and cache hierarchy. An optional startup microbenchmark in `DMRGEngine::new()` sweeps GEMM size from 64×64 to 1024×1024, measures single-thread vs multi-thread throughput, and records the crossover point. This calibration runs once per process launch (~3–5 seconds) and is cached for the session.

```rust
/// Configuration for the partitioned LPT scheduler.
pub struct PartitionedLptConfig {
    /// FLOPs below this threshold go to Rayon (single-threaded BLAS).
    /// FLOPs at or above this threshold get full BLAS multithreading.
    /// Default: 250_000_000 (250 MFLOPs). Overridden by calibration.
    pub blas_flop_threshold: usize,
    /// Number of physical cores available.
    pub n_cores: usize,
    /// When `backend-cuda` is active: FLOPs at or above this threshold
    /// are diverted to the GPU queue instead of CPU BLAS.
    /// Tasks below this but above `blas_flop_threshold` get CPU multithreaded BLAS.
    /// Tasks below `blas_flop_threshold` go to Rayon as usual.
    /// Dispatching tiny sectors (10×10, 50×50) to the GPU wastes kernel launch
    /// overhead and saturates the PCIe queue. Default: 50_000_000 (50 MFLOPs).
    /// Invariant: gpu_flop_threshold >= blas_flop_threshold.
    #[cfg(feature = "backend-cuda")]
    pub gpu_flop_threshold: usize,
}

impl PartitionedLptConfig {
    /// Run a quick microbenchmark to find the GEMM size where multithreaded
    /// BLAS outperforms single-threaded by at least 2×. Converts the crossover
    /// matrix dimension to a FLOP threshold (M*N*K).
    pub fn calibrate(backend: &impl LinAlgBackend<f64>) -> Self {
        let n_cores = num_cpus::get_physical();
        let threshold = benchmark_blas_crossover(backend, n_cores);
        PartitionedLptConfig {
            blas_flop_threshold: threshold,
            n_cores,
            #[cfg(feature = "backend-cuda")]
            gpu_flop_threshold: 50_000_000,
        }
    }

    /// Use a conservative default without benchmarking.
    pub fn default() -> Self {
        PartitionedLptConfig {
            blas_flop_threshold: 250_000_000,
            n_cores: num_cpus::get_physical(),
            #[cfg(feature = "backend-cuda")]
            gpu_flop_threshold: 50_000_000,
        }
    }
}
```

#### 5.3.1 Partitioned LPT-Scheduled Block-Sparse Dispatch

In Abelian symmetric models, symmetry sector sizes follow a binomial distribution: a few massive blocks near the center (e.g., Sz = 0) and many tiny blocks at the edges. The partitioned LPT scheduler handles both ends optimally:

```rust
/// An abstract task representing a single dense GEMM within a block-sparse contraction.
struct SectorGemmTask<'a, T: Scalar> {
    out_key: PackedSectorKey,
    block_a: &'a DenseTensor<T>,
    block_b: &'a DenseTensor<T>,
    /// Estimated computational cost: M * N * K
    flops: usize,
}

impl<T: Scalar, Q: BitPackable> SparseLinAlgBackend<T, Q> for DefaultDevice
where DeviceFaer: LinAlgBackend<T>
{
    fn block_gemm(
        &self,
        a: &BlockSparseTensor<T, Q>,
        b: &BlockSparseTensor<T, Q>,
    ) -> BlockSparseTensor<T, Q> {
        // Phase 1: Map-Reduce Task Generation (SU(2) Safe)
        // For Abelian symmetries, compute_fusion_rule is one-to-one.
        // For SU(2), the generation loop fans out to Vec<SectorGemmTask>
        // per input pair, grouped by output sector (see §4.4).
        let mut tasks: Vec<SectorGemmTask<T>> = Vec::new();
        for (i, key_a) in a.sector_keys.iter().enumerate() {
            for (j, key_b) in b.sector_keys.iter().enumerate() {
                if let Some(out_key) = compute_fusion_rule(*key_a, *key_b) {
                    let ba = &a.sector_blocks[i];
                    let bb = &b.sector_blocks[j];
                    let flops = ba.rows() * bb.cols() * ba.cols();
                    tasks.push(SectorGemmTask { out_key, block_a: ba, block_b: bb, flops });
                }
            }
        }

        // Phase 2: LPT Sort — heaviest GEMMs first.
        tasks.sort_unstable_by(|x, y| y.flops.cmp(&x.flops));

        // Phase 3: Partitioned Dispatch
        // When backend-cuda is active, this becomes a three-way partition:
        //   GPU queue (flops >= gpu_flop_threshold)
        //   → CPU multithreaded BLAS (blas_flop_threshold <= flops < gpu_flop_threshold)
        //   → Rayon single-threaded (flops < blas_flop_threshold)
        // Without CUDA, the GPU tier is absent and dispatch is two-way as before.
        let config = &self.lpt_config;

        #[cfg(feature = "backend-cuda")]
        let (gpu_tasks, cpu_tasks): (Vec<_>, Vec<_>) = tasks
            .into_iter()
            .partition(|t| t.flops >= config.gpu_flop_threshold);

        #[cfg(not(feature = "backend-cuda"))]
        let cpu_tasks = tasks;

        let (heavy_tasks, light_tasks): (Vec<_>, Vec<_>) = cpu_tasks
            .into_iter()
            .partition(|t| t.flops >= config.blas_flop_threshold);

        let mut results = Vec::with_capacity(
            heavy_tasks.len() + light_tasks.len()
            + { #[cfg(feature = "backend-cuda")] { gpu_tasks.len() } #[cfg(not(feature = "backend-cuda"))] { 0 } }
        );

        // 3a. (CUDA only) Submit GPU-tier tasks as a batched cublasDgemmBatched call.
        #[cfg(feature = "backend-cuda")]
        {
            let gpu_results = submit_batched_gpu_gemm(&gpu_tasks, &self.cuda_device);
            results.extend(gpu_results);
        }

        // 3b. Execute heavy CPU tasks with full BLAS threading.
        // Each GEMM gets the entire machine thread pool.
        set_blas_threads(config.n_cores);
        for task in heavy_tasks {
            results.push(task.execute_sequential::<DeviceFaer>());
        }

        // 3c. Execute long tail with Rayon (single-threaded BLAS).
        // Rayon distributes independent tiny GEMMs across all cores.
        set_blas_threads(1);
        let tail_results: Vec<_> = light_tasks.into_par_iter()
            .map(|task| task.execute_parallel::<DeviceFaer>())
            .collect();
        results.extend(tail_results);

        // Restore BLAS threading for non-block-sparse operations.
        set_blas_threads(config.n_cores);

        // Phase 4: Structural Restoration — re-sort by PackedSectorKey
        // to restore the binary-search invariant on the output tensor.
        results.sort_unstable_by_key(|(key, _)| *key);
        let (out_keys, out_blocks) = results.into_iter().unzip();

        BlockSparseTensor {
            indices: compute_output_indices(&a.indices, &b.indices),
            leg_directions: compute_output_leg_directions(&a.leg_directions, &b.leg_directions),
            sector_keys: out_keys,
            sector_blocks: out_blocks,
            flux: a.flux.fuse(&b.flux),
        }
    }
}
```

The four-phase design (generate → LPT sort → partitioned dispatch → restore invariant) achieves optimal core saturation for both the heavy head and the fragmented tail. When `backend-cuda` is active, the dispatch becomes three-way (GPU → CPU-BLAS → Rayon), ensuring that tiny sectors never saturate the PCIe queue with individual kernel launches. The serialization barrier between dispatch phases is acceptable because in the common case the heavy tasks dominate wall time — the light tail is a rounding error.

**TEBD parallelism:** For TEBD, all even-bond gates commute: Rayon distributes even-bond SVDs in parallel using single-threaded BLAS per SVD, then repeats for odd bonds.

### 5.4 Monomorphization Budget & Compile-Time Strategy

The entire compute stack is generic over `<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>`. Rust monomorphizes each concrete combination, generating independent machine code for every instantiation. With 3 scalar types × 3 symmetry types × 4 backends, the theoretical maximum is 36 copies of the DMRG engine — an unacceptable binary size and compile-time explosion.

**Mitigation strategy (layered dispatch):**

1. **Inner loops: static dispatch (mandatory).** The matvec closure, GEMM dispatch, and contraction executor require zero-overhead abstraction. These remain fully generic and monomorphized.

2. **Sweep scheduler: static dispatch (default), `dyn`-eligible.** `DMRGEngine` is generic over `<T, Q, B>` by default. Because `LinAlgBackend<T>` is now object-safe (§5.1), the sweep engine can accept `Box<dyn LinAlgBackend<T>>` if compile times become problematic — the scalar type is still known statically, only the backend is dynamically dispatched.

3. **Feature-gated combinations:** Most users need exactly one combination: `f64 + U1 + DeviceFaer`. The `tk-python` crate's `DmftLoopVariant` enum explicitly enumerates the supported combinations, and only those variants are compiled. Adding a new combination requires a one-line enum variant, not a recompilation of the entire stack.

4. **Compile-time monitoring:** Phase 1 CI tracks per-crate compile times. If any single crate exceeds 60 seconds in release mode, the monomorphization strategy is revisited. The `cargo-llvm-lines` tool identifies the largest generic expansions.

```rust
// The common case: only this combination is compiled unless
// the user explicitly opts into others via feature flags.
#[cfg(all(feature = "backend-faer", not(feature = "backend-mkl")))]
pub type DefaultEngine = DMRGEngine<f64, U1, DeviceFaer>;
```

---

## 6. Contraction Engine (tk-contract)

**Post-implementation note (lifetime friction):** The `DenseTensor<'a, T>` lifetime parameter makes executor generics painful. When the executor holds tensors with different lifetimes (input borrows vs intermediate owned results), Rust's type system cannot unify them without explicit lifetime bounds. The current workaround uses `unsafe` transmute to unify lifetimes in controlled contexts where the executor guarantees the borrow validity. This is a known ergonomic cost of the CoW design and may motivate a future refactor to separate borrowed-view types from owned-tensor types.

The contraction engine separates: (1) finding the optimal contraction sequence (NP-hard), and (2) executing each pairwise contraction.

### 6.1 Contraction Graph (DAG)

```rust
pub enum ContractionNode {
    Input {
        tensor_id: TensorId,
        indices: Vec<IndexId>,  // logical index ordering
    },
    Contraction {
        left: Box<ContractionNode>,
        right: Box<ContractionNode>,
        contracted_indices: Vec<IndexId>,  // shared IndexIds summed over
        result_indices: Vec<IndexId>,      // output index ordering
    },
}

pub struct ContractionGraph {
    inputs: Vec<TensorId>,
    root: ContractionNode,
    estimated_flops: f64,
    estimated_memory_bytes: usize,    // memory traffic including transposes
    max_intermediate_size: usize,     // peak intermediate tensor size (elements)
}
```

### 6.2 Path Optimizer Trait

Optimizing purely for FLOPs is a classical trap: on modern hardware, memory bandwidth is the actual bottleneck. A path with 15% fewer FLOPs that forces explicit out-of-place transposes will execute slower than a "suboptimal" path allowing strided contraction. The `PathOptimizer` therefore uses a composite cost function that tracks tensor strides through the DAG and penalizes paths inducing memory permutations:

```rust
/// Composite cost metric: C_total = α·FLOPs + β·Bytes_Moved
/// β heavily penalizes explicit reshapes relative to arithmetic cost.
pub struct CostMetric {
    pub flop_weight: f64,       // α
    pub bandwidth_weight: f64,  // β (typically 10–100× flop_weight)
}

pub trait PathOptimizer: Send + Sync {
    fn optimize(
        &self,
        spec: &ContractionSpec,
        index_map: &IndexMap,
        cost: &CostMetric,
        /// Optional hard constraint on peak intermediate memory.
        /// If `Some(limit)`, the optimizer rejects candidate paths whose
        /// estimated peak memory exceeds `limit` bytes.
        /// For DMRG (fixed 4-tensor contractions), this defaults to `None`
        /// because memory is dominated by environments and MPS tensors,
        /// not intermediate contraction results.
        /// Essential for future PEPS/tree TNS extensions where intermediate
        /// memory blow-up is often the limiting factor, not FLOPs.
        max_memory_bytes: Option<usize>,
    ) -> ContractResult<ContractionGraph>;
}

pub struct GreedyOptimizer;                                        // O(n³)
pub struct TreeSAOptimizer { pub max_iterations: usize, ... }      // Simulated annealing
pub struct DPOptimizer { pub max_width: usize }                    // Dynamic programming
```

The optimizer propagates stride information through candidate contraction trees. At each pairwise node, it checks whether the contracted indices are already contiguous; if not, the estimated bytes-moved penalty for the required transpose is added to the path cost. This ensures the optimizer favors paths that align naturally with the memory layout, even at the expense of slightly higher FLOP counts.

The optimizer also tracks conjugation metadata: when a Hermitian conjugate is required, the `is_conjugated` flag on the `MatRef` view is set rather than scheduling an explicit O(N) conjugation pass. The bandwidth cost for conjugation is therefore zero.

**Forward-compatibility note (fermionic swap gates, Phase 5+):** When the `fermionic-swap` feature flag is implemented for tree/PEPS geometries (§6.4), fermionic leg permutations will incur physical memory reshuffles identical to standard transposes, plus an O(1) sign computation. The existing `bandwidth_weight` in `CostMetric` already penalizes these reshuffles correctly — a fermionic swap has the same memory-bandwidth cost as a bosonic transpose, with negligible additional arithmetic. No structural change to the `PathOptimizer` trait or `CostMetric` is required. Future implementers should add the sign-tracking logic to the `ContractionExecutor`, not to the optimizer.

### 6.3 Contraction Executor

Two execution strategies, selected by backend capabilities:

**Strategy A — Strided Tensor Contraction:** tblis-style arbitrary-stride micro-kernels bypass reshape entirely. Zero memory-bandwidth cost.

**Strategy B — Pre-Allocated Transpose Arenas:** Standard GEMM (faer) requires transposition for non-contiguous contractions. Cache-aligned buffers from SweepArena; cache-oblivious block-transpose (8×8 or 16×16 tiles) maximizes cache-line utilization.

```rust
/// The arena is passed by parameter to `execute()`, not owned by the executor.
/// This avoids lifetime entanglement and lets the caller (e.g., `DMRGEngine`)
/// control arena lifecycle. LPT scheduling is delegated to
/// `SparseLinAlgBackend::block_gemm` in `tk-linalg`.
pub struct ContractionExecutor<T: Scalar, B: LinAlgBackend<T>> {
    backend: B,
    _phantom: PhantomData<T>,
}
```

**Lifetime design note:** The executor must hold both borrowed input tensors and owned intermediate results simultaneously. Since `DenseTensor<'a, T>` carries a storage lifetime, these have incompatible type parameters. Two viable strategies are documented in the tk-contract tech spec §7.1: (A) copy all inputs into the arena upfront to unify lifetimes, or (B) maintain separate typed maps for inputs vs. intermediates. Strategy A is simpler and recommended for the arena-centric execution model; the extra memcpy per input is negligible relative to the O(D³) GEMM cost.

### 6.4 Fermionic Sign Convention

The contraction engine operates with **bosonic tensor legs only**. It does not implement native fermionic swap gates or automatic Jordan-Wigner string insertion during leg permutations.

For fermionic models (Hubbard, Anderson impurity, t-J), the Jordan-Wigner transformation is applied at the MPO construction stage in `tk-dmrg`. The resulting MPO tensors carry the sign factors as explicit matrix elements, and the contraction engine processes them as ordinary dense/block-sparse data without needing to know that the underlying particles are fermionic.

This design is correct and complete for all 1D chain and star-to-chain geometries targeted through Phase 4. It covers the full Anderson Impurity Model workflow (star geometry mapped to a chain via Lanczos tridiagonalization) and all standard 1D models (Hubbard, Heisenberg, t-J).

**Known limitation:** For tree tensor networks or 2D PEPS, where tensor legs cross and cannot be linearly ordered, the Jordan-Wigner approach breaks down. These geometries require native fermionic swap gates in the contraction engine — a `FermionicLegSwap` callback analogous to the existing `structural_contraction` callback for SU(2). This is deferred to Phase 5+ alongside tree/PEPS support, behind a `fermionic-swap` feature flag.

---

## 7. Ergonomic API & Domain-Specific Language (tk-dsl)

### 7.1 Intelligent Index System

```rust
pub struct Index {
    id: IndexId,
    tag: SmallString<[u8; 32]>,
    dim: usize,
    prime_level: u32,
    direction: IndexDirection,
}

pub fn contract<T: Scalar>(a: &IndexedTensor<T>, b: &IndexedTensor<T>) -> IndexedTensor<T> {
    // Indices with matching id + complementary prime levels contract automatically.
}
```

### 7.2 The `hamiltonian!{}` Macro: Scope, Hygiene, and Diagnostics

**Post-implementation note:** The `hamiltonian!{}` proc-macro is deferred and not yet implemented. The `OpSum` builder pattern (§7.3) with operator overloading provides adequate ergonomics for Phase 1–3. The macro design below remains the target specification.

The `hamiltonian!{}` proc-macro performs *exclusively* compile-time syntax-tree transformation. It parses a lattice/Hamiltonian DSL and emits Rust code that constructs an `OpSum` at runtime. No numerical computation — no SVD, no FSA minimization, no matrix construction — occurs at compile time. Because physics parameters (J, U, V_k) are evaluated dynamically at runtime, the macro must safely interact with the user's surrounding lexical scope.

**Rationale:** MPO compression involves substantial numerical linear algebra (SVD truncation, finite-state automaton minimization) that would freeze `rustc` for non-trivial systems. Furthermore, physics parameters (N, J, U, coupling arrays) are typically determined at runtime from configuration files, not known at compile time.

The two-phase pipeline is:

1. **Compile time (`tk-dsl`):** `hamiltonian!{}` → generates `OpSum` builder code (boilerplate reduction only).
2. **Runtime (`tk-dmrg`):** `OpSum::compile_mpo(&self, backend: &B, config: &MpoCompressionConfig) → MPO<T, Q>` performs SVD-based compression using the active `LinAlgBackend`.

```rust
// Phase 1: The macro generates an OpSum at compile time.
// No numerical computation occurs here.
let opsum = hamiltonian! {
    lattice: Chain(N = 100, d = 2);
    sum i in 0..N-1 {
        J  * (Sp(i) * Sm(i+1) + Sm(i) * Sp(i+1))
      + Jz * Sz(i) * Sz(i+1)
    }
    sum i in 0..N { h * Sz(i) }
};

// Phase 2: Runtime MPO compression in tk-dmrg.
// This is where the heavy linear algebra happens.
let mpo: MPO<f64, U1> = opsum.compile_mpo(&backend, &MpoCompressionConfig {
    max_bond_dim: 50,
    svd_cutoff: 1e-12,
})?;
```

For the Anderson Impurity Model:

```rust
let opsum_aim = hamiltonian! {
    lattice: Star(n_bath = 6, d = 4);
    U * Nup(0) * Ndn(0)
    sum k in 1..=n_bath {
        V[k] * (Cdag_up(0) * C_up(k) + h.c.)
      + V[k] * (Cdag_dn(0) * C_dn(k) + h.c.)
      + eps[k] * (Nup(k) + Ndn(k))
    }
};
```

#### 7.2.1 Macro Span Hygiene

In Rust, procedural macros can easily cause variable-shadowing bugs if generated code accidentally overwrites or captures local variables. The `tk-dsl` compiler guarantees hygiene by manipulating `proc_macro2::Span`:

* **Call-site spans for user variables:** User-provided physics parameters (e.g., `J`, `U`, the array `V[k]` in an Anderson Impurity Model, the loop bound `N`) are emitted using their original `Span::call_site()`. This allows them to resolve naturally to the user's local variables — precisely the intended behavior, since these parameters are runtime values in the user's scope.
* **Mixed-site spans for macro internals:** Internal loop counters, temporary accumulators, and builder variables generated by the macro (e.g., the `__tk_opsum_builder` variable, the `__tk_site_idx` loop counter) are emitted with `Span::mixed_site()`. This makes them hygienic: they are invisible to the user's code and cannot collide with any user-defined variable. A user naming their coupling constant `__tk_opsum_builder` is astronomically unlikely, but mixed-site hygiene eliminates even that edge case by construction.

The practical consequence is that a user can freely write:

```rust
let J = 1.0;
let U = 4.0;
let builder = some_other_thing();  // not shadowed by macro internals
let opsum = hamiltonian! { ... J ... U ... };
// `builder` still refers to `some_other_thing()`, not the macro's accumulator.
```

#### 7.2.2 Diagnostic Error Reporting

Standard string-based tensor libraries fail at runtime if a user makes a typo in an operator string. `tensorkraft` prevents this with typed operator enums (§7.3), but the `hamiltonian!{}` macro's custom DSL grammar still requires its own parser. Naive proc-macros produce cryptic compiler errors pointing to the macro invocation site rather than the specific syntax error inside the DSL block.

The `tk-dsl` parser implements `syn::parse::Parse` for the custom physics DSL grammar. When the parser encounters a syntax error, it generates a `syn::Error::new_spanned(token, "message")` attached to the offending token. This forces `rustc` to draw its diagnostic annotation (red squiggly line in IDEs, caret in terminal) exactly under the user's typo, matching the ergonomics of native Rust code.

Examples of errors caught with precise spans:

* **Missing operator:** `J * Sz(i) Sz(i+1)` → error at the second `Sz`: "expected binary operator `*` or `+` between terms, found identifier `Sz`"
* **Invalid site index:** `Sp(i) * Sm(i+2)` with `lattice: Chain(N=100, d=2)` → error at `i+2`: "next-nearest-neighbor coupling; use explicit `sum` ranges for non-local terms"
* **Unknown operator name:** `Sx(i)` for a spin-1/2 model → error at `Sx`: "unknown operator `Sx` for `d=2`; did you mean `Sp`, `Sm`, or `Sz`?"
* **Unbalanced parentheses:** `J * (Sp(i) * Sm(i+1)` → error at end of block: "unclosed parenthesis opened here"

This transforms the debugging experience from "the macro invocation on line 47 failed" to "you wrote `Sx` on column 12 inside the macro block; here are the valid alternatives."

### 7.3 OpSum Builder Pattern with Typed Operators

Standard physical operators are represented as strongly-typed enums, eliminating the runtime errors that arise from string-based APIs (e.g., `op("S_plus", i)` silently failing or panicking during MPO compilation):

```rust
/// Spin-1/2 operators. Compile-time exhaustive — typos are impossible.
pub enum SpinOp { SPlus, SMinus, Sz, Identity }

/// Spinful fermion operators (for Hubbard / Anderson models).
pub enum FermionOp { CdagUp, CUp, CdagDn, CDn, Nup, Ndn, Ntotal, Identity }

/// Bosonic operators (for Bose-Hubbard, phonon models).
pub enum BosonOp { BDag, B, N, Identity }

/// Arbitrary user-defined operator for non-standard models.
pub struct CustomOp<T: Scalar> {
    pub matrix: DenseTensor<T>,
    pub name: SmallString<[u8; 32]>,
}

/// Unified operator type accepting either standard or custom operators.
pub enum SiteOperator<T: Scalar> {
    Spin(SpinOp),
    Fermion(FermionOp),
    Boson(BosonOp),
    Custom(CustomOp<T>),
}

/// Type-safe site-operator reference.
pub fn op<T: Scalar>(operator: impl Into<SiteOperator<T>>, site: usize) -> OpTerm<T> {
    OpTerm { operator: operator.into(), site }
}
```

Usage with typed operators:

```rust
let mut opsum = OpSum::new();
for i in 0..n_sites - 1 {
    opsum += J * op(SpinOp::SPlus, i) * op(SpinOp::SMinus, i+1);
    opsum += J * op(SpinOp::SMinus, i) * op(SpinOp::SPlus, i+1);
    opsum += Jz * op(SpinOp::Sz, i) * op(SpinOp::Sz, i+1);
}
// Runtime compilation to MPO (in tk-dmrg, not tk-dsl):
let mpo: MPO<f64, U1> = opsum.compile_mpo(&backend, &mpo_config)?;
```

For non-standard models, the `CustomOp` escape hatch allows arbitrary operator matrices:

```rust
let pauli_y = CustomOp {
    matrix: DenseTensor::from_slice(&[0.0, -1.0, 1.0, 0.0], &[2, 2]),
    name: "σ_y".into(),
};
opsum += coupling * op(pauli_y, i) * op(pauli_y, i+1);
```

### 7.4 Lattice Abstraction

```rust
pub trait Lattice {
    fn n_sites(&self) -> usize;
    fn bonds(&self) -> &[(usize, usize)];
    fn dmrg_ordering(&self) -> Vec<usize>;  // snake-path for 2D→1D mapping
}

pub struct Chain { pub n: usize }
pub struct Square { pub lx: usize, pub ly: usize }
pub struct Triangular { pub lx: usize, pub ly: usize }
pub struct BetheLattice { pub z: usize, pub depth: usize }
```

### 7.5 Python Bindings: Thread-Safe GIL Management & Zero-Copy NumPy

PyO3's `#[pyclass]` cannot be applied to generic structs. The tk-python crate bridges Rust's monomorphization to Python's dynamic dispatch via a type-erased enum:

```rust
enum DmftLoopVariant {
    RealU1(DMFTLoop<f64, U1, DefaultDevice>),
    // Post-implementation note: ComplexU1 variant is currently blocked
    // because DeviceFaer does not yet implement LinAlgBackend<Complex<f64>>.
    // Uncomment when complex backend support is added.
    // ComplexU1(DMFTLoop<Complex64, U1, DefaultDevice>),
    RealZ2(DMFTLoop<f64, Z2, DefaultDevice>),
}

#[pyclass(name = "DMFTLoop")]
pub struct PyDmftLoop { inner: DmftLoopVariant }
```

**Post-implementation notes (Python bindings):**
- **`extension-module` vs test linking conflict:** PyO3's `extension-module` feature (required for building `.so`/`.dylib` shared libraries) conflicts with test linking — `cargo test` cannot link against a cdylib. The workaround is a `pyo3-extension-module` feature flag that is only activated during `maturin build`, not during `cargo test`.
- **Config mirror pattern:** `DMRGConfig` contains `Box<dyn IterativeEigensolver>`, which prevents `Clone`. Python bindings cannot pass Rust config structs directly. Instead, `PyDmftConfig` mirrors the config fields as plain Python-compatible types, and constructs the Rust `DMRGConfig` internally when `solve()` is called.

**GIL release:** A `DMFTLoop::solve()` call can run for hours. Holding the Python GIL for the entire duration freezes the Jupyter kernel, prevents `Ctrl+C` handling, and starves OS-level context switching. All compute-heavy methods release the GIL before entering Rust numerics.

**Thread-safe cancellation via `AtomicBool` + `mpsc` lifecycle guard:** The original v2 design proposed re-acquiring the GIL from within Rayon worker threads via `Python::with_gil`. This is unsound: Rayon worker threads are not registered with the CPython interpreter, and attempting to acquire the GIL from an unregistered thread while the main Python thread is blocked inside `allow_threads` can cause deadlocks or segfaults. The corrected design uses an `AtomicBool` flag as a decoupled cancellation channel, with an `mpsc::channel` to guarantee monitor thread termination when the solver completes.

**Critical ordering constraint:** The monitor thread shutdown (`done_tx.send()` + `monitor_handle.join()`) must occur *inside* the `py.allow_threads` closure, while the GIL is still released. If shutdown occurs *after* `allow_threads` returns, the main thread holds the GIL, and the monitor thread may be blocked on `Python::with_gil` — a classic AB/BA deadlock.

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

#[pymethods]
impl PyDmftLoop {
    pub fn solve(&mut self, py: Python<'_>) -> PyResult<PySpectralFunction> {
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let (done_tx, done_rx) = mpsc::channel::<()>();

        let monitor_cancel = cancel_flag.clone();
        let monitor_handle = std::thread::spawn(move || {
            loop {
                match done_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(()) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        let interrupted = Python::with_gil(|py| py.check_signals().is_err());
                        if interrupted {
                            monitor_cancel.store(true, Ordering::Release);
                            break;
                        }
                    }
                }
            }
        });

        let result = py.allow_threads(|| {
            let r = match &mut self.inner {
                DmftLoopVariant::RealU1(solver) => {
                    solver.solve_with_cancel_flag(&cancel_flag)
                }
                // ...
            };

            // CRITICAL: shutdown while GIL is still released.
            let _ = done_tx.send(());
            let _ = monitor_handle.join();

            r
        });

        result.map(PySpectralFunction::from).map_err(Into::into)
    }
}
```

The monitor thread has exactly three exit conditions, all of which guarantee clean shutdown:
1. `done_rx.recv_timeout` returns `Ok(())` — the solver finished normally.
2. `done_rx` returns `Err(Disconnected)` — `done_tx` was dropped (solver panicked). The monitor exits without segfaulting.
3. `py.check_signals()` detects `SIGINT` — the monitor sets the cancellation flag and exits.

The `done_tx.send()` and `monitor_handle.join()` execute while the GIL is still released. By the time `allow_threads` returns and the main thread re-acquires the GIL, the monitor thread is dead — no deadlock is possible.

The DMRG engine checks the `AtomicBool` at the end of each sweep step — a single `Relaxed` atomic load, effectively free:

```rust
impl<T, Q, B> DMRGEngine<T, Q, B>
where T: Scalar, Q: BitPackable, B: LinAlgBackend<T>
{
    pub fn run_with_cancel_flag(
        &mut self,
        cancel: &AtomicBool,
    ) -> Result<T::Real, SolverError> {
        for sweep in 0..self.config.max_sweeps {
            for site in self.sweep_schedule() {
                self.dmrg_step(site)?;
                if cancel.load(Ordering::Relaxed) {
                    return Err(SolverError::Cancelled);
                }
            }
            if self.converged() { break; }
        }
        Ok(self.energy())
    }
}
```

This design guarantees that: (1) GIL re-acquisition only happens on the monitor thread (never from Rayon workers), (2) the monitor thread always terminates when the solver does, and (3) panic safety is automatic via channel disconnection.

**Zero-copy NumPy interop:** Bath parameter updates from TRIQS Green's functions use `rust-numpy` to share memory directly between Python and Rust without element-wise copying:

```rust
use numpy::{PyArray1, IntoPyArray};

#[pymethods]
impl PySpectralFunction {
    #[getter]
    fn omega<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.omega.clone().into_pyarray(py)
    }
}
```

---

## 8. DMRG Algorithm & DMFT Integration

### 8.1 MPS with Typestate Canonical Forms

Rust's type system enforces gauge conditions at compile time. An MPS can only be in one of four well-defined states, and operations requiring a specific gauge form will not compile on an improperly conditioned state:

```rust
pub struct LeftCanonical;
pub struct RightCanonical;
pub struct MixedCanonical { pub center: usize }
pub struct BondCentered { pub left: usize }

pub struct MPS<T: Scalar, Q: BitPackable, Gauge> {
    tensors: Vec<BlockSparseTensor<T, Q>>,
    bonds: Option<Vec<DenseTensor<T>>>,
    _gauge: PhantomData<Gauge>,
}

impl<T: Scalar, Q: BitPackable> MPS<T, Q, MixedCanonical> {
    pub fn dmrg_step(
        &mut self, mpo: &MPO<T, Q>, env: &mut Environments<T, Q>,
        solver: &dyn IterativeEigensolver<T>,
    ) -> T::Real { /* returns energy */ }

    pub fn expose_bond(self) -> MPS<T, Q, BondCentered> { /* ... */ }
}

impl<T: Scalar, Q: BitPackable> MPS<T, Q, BondCentered> {
    /// Evolve the bond matrix backward in time: exp(+i H_bond Δt/2).
    /// Uses Tikhonov-regularized pseudo-inverse for gauge restoration.
    pub fn evolve_bond_backward(
        &mut self,
        h_bond: &dyn Fn(&[T], &mut [T]),
        dt: T::Real,
        config: &TdvpStabilizationConfig,
    ) { /* Krylov matrix-exponential on the bond matrix */ }

    pub fn absorb_bond(self) -> MPS<T, Q, MixedCanonical> { /* ... */ }
}
```

The `BondCentered` state is essential for TDVP's projector-splitting scheme. A single 1-site TDVP step requires: (1) evolving the center site forward via exp(-i H_eff Δt/2), (2) exposing the bond matrix (transition to `BondCentered`), (3) evolving the bond matrix *backward* via exp(+i H_bond Δt/2), (4) absorbing the bond and shifting the gauge center (transition back to `MixedCanonical`). Without `BondCentered`, the backward bond evolution would require breaking the typestate abstraction.

#### 8.1.1 TDVP Numerical Stabilization

The backward evolution step requires inverting the bond matrix (containing the singular values of the bipartition). When the state has low entanglement across a bond, many singular values approach machine zero, causing 1/s_i → ∞ and filling tensors with NaN. Two complementary stabilization strategies are employed:

**Strategy 1 — Tikhonov Regularization (numerical floor):**

Every gauge-restoration inversion routes through `LinAlgBackend::regularized_svd_inverse` (§5.1). Instead of computing s_i⁻¹, the regularized form s_i / (s_i² + δ²) is used, where δ is a configurable noise floor (typically 1e-8 to 1e-12). When s_i ≫ δ, this perfectly approximates the true inverse; when s_i → 0, the regularized inverse safely drops to zero.

**Adaptive δ annealing:** A static δ may be too aggressive for bonds that are genuinely low-entanglement (e.g., near a product state), where many singular values are legitimately zero. In this regime, a fixed δ = 1e-10 can mask the physics by assigning artificial weight to null-space directions. When `adaptive_tikhonov` is enabled, δ is dynamically scaled relative to the largest discarded singular value from the previous SVD truncation at that bond: δ_adaptive = max(δ_min, α_δ × σ_discarded_max), where α_δ is a configurable scaling factor (default: 0.01) and δ_min is the absolute noise floor (default: 1e-14). This ensures the regularization floor tracks the actual entanglement scale of the bipartition rather than being a global constant.

**Strategy 2 — Site-Tensor Subspace Expansion (algorithmic growth):**

Tikhonov regularization prevents NaN but restricts the MPS to its existing entanglement subspace. Real-time evolution (especially post-quench in DMFT) requires entanglement *growth*, which pure 1-site TDVP cannot achieve since it operates within the fixed-D tangent space.

The subspace expansion operates on the *site tensors* A_L and A_R, not on the bond matrix C. The bond matrix C lives in a D × D space, while the Hamiltonian action generates vectors in the larger D·d × D·d two-site space. Direct mixing of the two-site residual into the bond matrix is dimensionally inconsistent. Instead, the expansion enlarges the site-tensor basis:

1. At the `BondCentered` step, compute the Hamiltonian residual: |R⟩ = H_eff · |ψ_center⟩.
2. Project out the existing tangent-space component using **matrix-free sequential projection** (never form the explicit projector P = I − A_L · A_L†, which would cost O(d²D³)):
   - Compute the overlap: O = A_L† · |R⟩. This is a matrix-vector product costing O(dD²).
   - Subtract the projection: |R_null⟩ = |R⟩ − A_L · O. Another O(dD²) matrix-vector product.
   - Total projection cost: O(dD²), safely within the O(D³) DMRG scaling bound.
3. **Per-sector SVD** of |R_null⟩ → retain the top D_expand left singular vectors with the largest singular values. These vectors span directions orthogonal to the current tangent space. **Critical:** the SVD must be performed *per symmetry sector* (i.e., as a block-sparse SVD decomposing each sector block independently), not as a single dense SVD of the flattened residual. Because H_eff is flux-conserving and A_L is block-sparse with correct quantum numbers, |R_null⟩ is automatically block-sparse with the same sector structure. Per-sector SVD guarantees that expansion vectors are flux-preserving by construction — no numerical noise can introduce elements in forbidden sectors. The top D_expand vectors are selected globally across all sectors by comparing singular values, preserving the greedy optimal truncation property.
4. Pad the site tensor A_L by appending these D_expand new basis vectors as additional columns, increasing its bond dimension from D to D + D_expand.
5. Pad the bond matrix C with zeros along the corresponding new rows/columns. The expanded C is now (D + D_expand) × (D + D_expand).
6. SVD the expanded bond matrix. The injected null-space vectors now provide physically relevant non-zero singular values that push the smallest eigenvalues above the noise floor naturally.
7. Truncate using **soft D_max** policy (see below).

**Scaling note:** Explicitly constructing the null-space projector P_null = (I − A_L · A_L†) would form a dense (dD × dD) matrix at O(d²D³) cost — an unnecessary factor of d·D more expensive than the matrix-free approach. For D = 1000 and d = 4, this is a ~4000× performance difference inside the innermost TDVP loop. The matrix-free two-step projection (overlap → subtract) is mandatory.

**Bond-dimension oscillation pathology:** If the injected null-space vectors have singular values very close to the truncation threshold, a hard truncation at D_max discards the exact entanglement that was just injected. On the next time step, the expansion re-injects the same vectors, which are again truncated — producing a discontinuous, oscillating bond dimension that corrupts the time evolution.

The soft D_max policy prevents this by allowing the bond dimension to temporarily exceed D_max by a configurable factor (typically 1.1×), then smoothly decaying the excess over physical simulation time rather than hard-cutting immediately:

- After expansion + SVD in step 7, truncate to `D_soft = floor(D_max × soft_dmax_factor)` instead of D_max.
- On each subsequent time step *without* expansion, the effective truncation target decays exponentially in *physical time*: `D_target(t) = D_max + (D_soft − D_max) × exp(−t_elapsed / dmax_decay_time)`, where `t_elapsed` is the accumulated physical simulation time since the last expansion at that bond, and `dmax_decay_time` is in the same physical time units as the TDVP integrator's dt.
- This gives newly injected entanglement a physically meaningful relaxation period to either grow into significant singular values or decay naturally below the noise floor.

**Physical time invariance:** The decay is expressed in physical time units (not discrete iteration counts) to ensure invariance under adaptive time-stepping. If the TDVP integrator shrinks dt by 10× near a sharp feature, the decay proceeds at the same *physical* rate — the bond dimension does not relax 10× faster simply because the integrator took more steps. This prevents a numerical tolerance (adaptive dt) from coupling to a physical relaxation timescale (entanglement decay).

**Per-bond state tracking:** The decay formula requires knowing the accumulated physical time since the last expansion *at each bond independently*. This is mutable algorithmic state that does not belong in the immutable `TdvpStabilizationConfig` or in the MPS typestate (which should not carry algorithm-specific metadata). Instead, the `TdvpDriver` struct (see below) carries a `Vec<Option<f64>>` of per-bond expansion ages in physical time units, updated after each time step via `advance_expansion_age(dt)`.

This procedure ensures dimensional consistency at every step: the site tensor expansion changes the shape of A_L from (d, D_left, D) to (d, D_left, D + D_expand), and the bond matrix is padded to match. The contraction graph in `tk-contract` handles the dimension change because `TensorShape` is dynamically determined, not statically fixed.

```rust
pub struct TdvpStabilizationConfig {
    pub tikhonov_delta: f64,       // default: 1e-10 (static floor, used when adaptive_tikhonov = false)
    /// When true, δ is dynamically scaled to α_δ × σ_discarded_max per bond,
    /// preventing the regularization floor from masking physics in near-product-state bonds.
    pub adaptive_tikhonov: bool,   // default: true
    /// Scaling factor for adaptive δ: δ_adaptive = max(tikhonov_delta_min, tikhonov_delta_scale × σ_discarded_max).
    pub tikhonov_delta_scale: f64, // default: 0.01
    /// Absolute minimum δ (prevents underflow when σ_discarded_max is extremely small).
    pub tikhonov_delta_min: f64,   // default: 1e-14
    pub expansion_vectors: usize,  // default: 4
    pub expansion_alpha: f64,      // default: 1e-4
    pub adaptive_expansion: bool,  // default: true
    pub soft_dmax_factor: f64,     // default: 1.1 (allow 10% overshoot)
    /// Physical time constant for soft D_max decay (in simulation time units).
    /// The overshoot decays as exp(−t_elapsed / dmax_decay_time).
    /// Invariant to adaptive time-stepping: decay rate is physical, not per-iteration.
    pub dmax_decay_time: f64,      // default: 5.0 (in units of the TDVP dt)
}

pub fn expand_bond_subspace<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>(
    a_l: &mut BlockSparseTensor<T, Q>,
    bond: &DenseTensor<T>,
    h_mpo: &MPO<T, Q>,
    env_l: &Environment<T, Q>,
    env_r: &Environment<T, Q>,
    config: &TdvpStabilizationConfig,
    backend: &B,
) -> DenseTensor<T> {
    // 1. Compute residual: |R⟩ = H_eff · |ψ_center⟩
    // 2. Matrix-free null-space projection (O(dD²), NOT O(d²D³)):
    //    a. Overlap: O = A_L† · |R⟩       [mat-vec, O(dD²)]
    //    b. Project: |R_null⟩ = |R⟩ - A_L · O  [mat-vec, O(dD²)]
    // 3. Per-sector SVD of |R_null⟩ (block-sparse, flux-preserving):
    //    → take top `expansion_vectors` left singular vectors globally across sectors
    // 4. Pad A_L with new vectors (scaled by expansion_alpha)
    // 5. Return zero-padded bond matrix with dimensions (D + D_expand) × (D + D_expand)
    todo!()
}

pub struct TdvpDriver<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> {
    pub config: TdvpStabilizationConfig,
    pub engine: DMRGEngine<T, Q, B>,
    /// Per-bond expansion age in physical time units.
    /// `expansion_age[bond]` is `Some(t_elapsed)` if the bond was last expanded
    /// `t_elapsed` simulation-time units ago, `None` if never expanded or fully decayed.
    /// Length = n_sites - 1 (one entry per bond).
    expansion_age: Vec<Option<f64>>,
}

impl<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> TdvpDriver<T, Q, B> {
    pub fn new(engine: DMRGEngine<T, Q, B>, config: TdvpStabilizationConfig) -> Self {
        let n_bonds = engine.mps.n_sites() - 1;
        TdvpDriver {
            config,
            engine,
            expansion_age: vec![None; n_bonds],
        }
    }

    /// Compute the effective truncation target for a given bond.
    fn effective_dmax(&self, bond: usize, hard_dmax: usize) -> usize {
        match self.expansion_age[bond] {
            None => hard_dmax,
            Some(t_elapsed) => {
                let d_soft = (hard_dmax as f64 * self.config.soft_dmax_factor).floor() as usize;
                let overshoot = (d_soft - hard_dmax) as f64;
                let decay = (-t_elapsed / self.config.dmax_decay_time).exp();
                hard_dmax + (overshoot * decay).round() as usize
            }
        }
    }

    /// Called after each TDVP time step with the physical time step dt.
    /// Advances all expansion age counters by dt (physical time, not iteration count).
    /// Bonds whose effective D_target has decayed to within 1 of D_max are reset to None.
    fn advance_expansion_age(&mut self, dt: f64, hard_dmax: usize) {
        for (bond, age) in self.expansion_age.iter_mut().enumerate() {
            if let Some(t) = age {
                *t += dt;
                if self.effective_dmax(bond, hard_dmax) <= hard_dmax {
                    *age = None;
                }
            }
        }
    }

    /// Mark a bond as freshly expanded (resets its physical time counter to 0.0).
    fn mark_expanded(&mut self, bond: usize) {
        self.expansion_age[bond] = Some(0.0);
    }
}
```

### 8.2 Iterative Eigensolver Trait

The eigensolver implementations are written **in-house within `tk-dmrg`**, not delegated to external crates. Off-the-shelf Rust eigensolvers are designed around standard dense matrix types and do not support the zero-allocation `&mut [T]` matvec workspace closures or tight integration with `SweepArena` that DMRG demands.

```rust
pub enum InitialSubspace<'a, T: Scalar> {
    None,
    SingleVector(&'a [T]),
    SubspaceBasis { vectors: &'a [&'a [T]], num_vectors: usize },
}

pub trait IterativeEigensolver<T: Scalar>: Send + Sync {
    fn lowest_eigenpair(
        &self,
        matvec: &dyn Fn(&[T], &mut [T]),
        dim: usize,
        initial: InitialSubspace<'_, T>,
    ) -> EigenResult<T>;
}

pub struct EigenResult<T: Scalar> {
    pub eigenvalue: T::Real,
    pub eigenvector: Vec<T>,
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: T::Real,
}

pub struct LanczosSolver {
    pub max_iter: usize, pub tol: f64,
    pub max_krylov_dim: usize,
}

pub struct DavidsonSolver {
    pub max_iter: usize, pub tol: f64,
    pub max_subspace: usize,
    pub restart_vectors: usize,
}

/// Block-Davidson: converts memory-bound dgemv into compute-bound dgemm.
pub struct BlockDavidsonSolver {
    pub max_iter: usize, pub tol: f64,
    pub block_size: usize,
    pub max_subspace: usize,
    pub restart_vectors: usize,
}
```

Inside the matvec closure, all intermediate contraction temporaries (T1, T2, T3) are pre-allocated from the SweepArena before the Krylov loop, reducing the closure to a pure sequence of GEMMs into pre-allocated workspace.

**Eigensolver memory ownership:** The Krylov basis vectors are **heap-allocated** and owned as a **persistent reusable workspace** by `DMRGEngine`, *not* arena-allocated. At D = 2000 with `max_krylov_dim = 20`, the Krylov workspace is ~640 MB (20 × 2000² × 8 bytes). This workspace must not live in the `SweepArena` (which resets every step), but it also must not be allocated and dropped every step — doing so would invoke the system allocator thousands of times per sweep for 640 MB blocks, severely fragmenting the heap even under jemalloc.

Instead, `DMRGEngine` owns a `KrylovWorkspace` struct that is allocated once during `DMRGEngine::new()` and reused across all sweep steps. Between steps, the workspace is logically cleared (the eigensolver overwrites vectors in place during the next Krylov iteration), but the underlying memory remains at the same addresses. The workspace is deallocated only when the `DMRGEngine` is dropped.

```rust
/// Persistent, reusable workspace for iterative eigensolvers.
/// Allocated once per DMRGEngine, reused across all sweep steps.
/// Avoids heap fragmentation from repeated alloc/drop of ~640 MB buffers.
pub struct KrylovWorkspace<T: Scalar> {
    /// Krylov basis vectors. Length = max_krylov_dim.
    /// Each vector has length = effective Hilbert space dimension.
    vectors: Vec<Vec<T>>,
    /// Scratch space for Hessenberg/tridiagonal matrices.
    projection_matrix: Vec<T>,
}

impl<T: Scalar> KrylovWorkspace<T> {
    pub fn new(max_krylov_dim: usize, hilbert_dim: usize) -> Self {
        KrylovWorkspace {
            vectors: (0..max_krylov_dim)
                .map(|_| vec![T::zero(); hilbert_dim])
                .collect(),
            projection_matrix: vec![T::zero(); max_krylov_dim * max_krylov_dim],
        }
    }
    // No deallocation between steps — vectors are overwritten in place.
}
```

The `SweepArena` remains reserved for contraction temporaries inside the matvec closure (which *are* recycled within the step) and for the SVD workspace that follows. If bond dimension changes between sweeps (e.g., during subspace expansion), the `KrylovWorkspace` is resized via `Vec::resize` — an amortized O(1) operation if dimension only grows.

### 8.3 DMRG Sweep Engine

```rust
pub struct DMRGEngine<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> {
    pub mps: MPS<T, Q, MixedCanonical>,
    pub mpo: MPO<T, Q>,
    pub environments: Environments<T, Q>,
    pub backend: B,
    pub config: DMRGConfig,
    /// Persistent Krylov workspace, allocated once, reused across all sweep steps.
    krylov_workspace: KrylovWorkspace<T>,
}

/// Post-implementation note: DMRGConfig should be split into immutable
/// DMRGConfig (parameters) and mutable DMRGState (convergence history,
/// sweep counters). The Box<dyn IterativeEigensolver> field prevents
/// Clone, which causes friction in tk-python (config mirroring needed).
///
/// Additionally, Davidson and Block-Davidson eigensolvers currently
/// delegate to Lanczos internally — they are not yet independently
/// implemented. The Lanczos solver is the only production-ready eigensolver.
///
/// Checkpoint functionality is non-functional: blocked by
/// BlockSparseTensor lacking serde support. Serialization of MPS
/// state for restart requires serde derives on all tensor types.
pub struct DMRGConfig {
    pub max_bond_dim: usize,
    pub svd_cutoff: f64,
    pub max_sweeps: usize,
    pub energy_tol: f64,
    pub eigensolver: Box<dyn IterativeEigensolver<f64>>,
    /// If `Some(path)`, environment blocks are memory-mapped to disk at the
    /// specified directory, reducing host RAM usage from O(N·D²·d·D_MPO) to
    /// the OS page cache budget. See §8.3 for memory scaling analysis.
    /// Default: None (full in-memory caching).
    pub environment_offload: Option<std::path::PathBuf>,
    /// In release builds, perform a full SVD reconstruction residual check
    /// every K-th SVD call to detect silent `gesdd` corruption.
    /// Default: 1000 (0.1% overhead). Set to 0 to disable.
    /// See §5.1 for details.
    pub svd_validation_interval: usize,
}
```

**Environment caching strategy:** The `Environments<T, Q>` struct caches all N−2 left and right contracted environment blocks in host RAM. During a left-to-right sweep, L[i] is incrementally built from L[i-1] by contracting the updated MPS tensor at site i with the old environment and the MPO row at site i. On the return sweep (right-to-left), R[i] is built analogously. Each environment block has dimensions O(D² × d × D_MPO), where D is bond dimension, d is local Hilbert space dimension, and D_MPO is the MPO bond dimension.

**Memory budget:** At D = 2000, d = 4, D_MPO = 5 (typical Heisenberg/Hubbard), each environment is ~2000² × 4 × 5 × 8 bytes ≈ 640 MB. For an N = 100 chain, the full cache is ~100 × 640 MB ≈ 62 GB. This is the dominant memory consumer in large-D DMRG — far exceeding the MPS tensors (~6.4 GB total) and the Krylov workspace (~640 MB).

```rust
pub struct Environments<T: Scalar, Q: BitPackable> {
    /// Left environment blocks L[0..N-2]. L[i] is the contraction of
    /// all MPS/MPO tensors to the left of bond (i, i+1).
    left: Vec<BlockSparseTensor<T, Q>>,
    /// Right environment blocks R[0..N-2]. R[i] is the contraction of
    /// all MPS/MPO tensors to the right of bond (i, i+1).
    right: Vec<BlockSparseTensor<T, Q>>,
}
```

**Memory-constrained nodes:** For systems where the full environment cache exceeds available RAM:

1. **Disk offload (recommended for memory-constrained nodes):** Write environment blocks to a memory-mapped file (via `memmap2` crate) as they are computed. On access, the OS pages them in transparently. This trades PCIe/NVMe bandwidth for RAM, with SSD-backed mmap providing ~3 GB/s read throughput — sufficient for the sequential access pattern of DMRG sweeps.
2. **Partial caching with recomputation:** Cache only a window of K environment blocks around the current sweep position, recomputing the rest on demand. This reduces memory to O(K × D² × d × D_MPO) but increases sweep cost by a factor of ~N/K. For K = 10 and N = 100, memory drops 10× at the cost of 10× slower sweeps — rarely worthwhile.
3. **Checkpointing:** For very long DMFT runs, periodic environment snapshots to disk enable restart after node failures without recomputing from scratch.

The default strategy is full in-memory caching (option 1 without mmap). Disk offload is activated via a `DMRGConfig::environment_offload: Option<PathBuf>` field.

impl<T, Q, B> DMRGEngine<T, Q, B>
where T: Scalar, Q: BitPackable, B: LinAlgBackend<T>
{
    pub fn run_with_cancel_flag(
        &mut self,
        cancel: &AtomicBool,
    ) -> Result<T::Real, SolverError> {
        for sweep in 0..self.config.max_sweeps {
            for site in self.sweep_schedule() {
                self.dmrg_step(site)?;
                if cancel.load(Ordering::Relaxed) {
                    return Err(SolverError::Cancelled);
                }
            }
            if self.converged() { break; }
        }
        Ok(self.energy())
    }
}
```

### 8.4 DMFT Self-Consistency Loop

**Post-implementation note (double-occupancy measurement):** The `FermionOp` enum does not include an `NPairInteraction` variant. Measuring double occupancy ⟨n↑n↓⟩ requires constructing a `CustomOp` with the explicit 4×4 matrix. This is adequate for DMFT but less ergonomic than a dedicated variant.

#### 8.4.1 Adaptive TDVP/Chebyshev Solver Selection

TDVP is designated as the primary time-evolution engine for gapped (insulating) phases. TEBD's Suzuki-Trotter decomposition violates unitarity over long time scales, causing norm drift that corrupts spectral functions. TDVP projects the time-dependent Schrödinger equation onto the MPS tangent-space manifold, rigorously preserving energy and unitarity. It reuses the same H_eff machinery and zero-allocation Krylov workspace from DMRG. TEBD is retained as a fallback.

TDVP integration is stabilized via the dual Tikhonov + site-tensor subspace expansion strategy described in §8.1.1.

**Metallic/gapless phase risk:** For metallic phases, the linear prediction pipeline (exponential windowing → Toeplitz solve → deconvolution → positivity restoration) is fragile because the exponential window artificially imposes structure that the signal does not have. The deconvolution step must then recover physics that was destroyed by the window — a process notorious for numerical artifacts. Chebyshev expansion, which works directly in frequency space, does not suffer from this instability.

**Adaptive solver selection:** The `DMFTLoop` automatically selects the primary spectral function engine based on the entanglement spectrum of the DMRG ground state:

1. After the DMRG ground state is computed, the entanglement spectrum (singular values of the bipartition at the center bond) is inspected.
2. If the spectral gap Δ (ratio of the largest to second-largest singular value, or equivalently the entanglement gap) exceeds a configurable threshold `gap_threshold`, the system is classified as gapped/insulating. TDVP + linear prediction is the primary engine; Chebyshev serves as cross-validation.
3. If the spectral gap is below `gap_threshold`, the system is classified as gapless/metallic. Chebyshev expansion is automatically promoted to primary; TDVP + linear prediction serves as cross-validation.

This is a runtime decision within `DMFTLoop::solve()`, not an architectural change. Both engines are always available; only the primary/validation roles swap.

```rust
pub enum SpectralSolverMode {
    /// TDVP + linear prediction primary, Chebyshev cross-validation.
    /// Appropriate for gapped/insulating phases.
    TdvpPrimary,
    /// Chebyshev primary, TDVP + linear prediction cross-validation.
    /// Appropriate for gapless/metallic phases.
    ChebyshevPrimary,
    /// Automatically select based on entanglement spectrum gap.
    Adaptive { gap_threshold: f64 },
}

impl Default for SpectralSolverMode {
    fn default() -> Self {
        SpectralSolverMode::Adaptive { gap_threshold: 0.1 }
    }
}
```

#### 8.4.2 Linear Prediction: Levinson-Durbin Recursion, Regularization & Spectral Positivity

Linear prediction extrapolates the time-domain Green's function G(t) to larger times to artificially increase frequency resolution. The prediction equations form a Toeplitz system from the autocorrelation of the signal. The architecture mandates a regularized solver for this system, with mandatory positivity restoration of the resulting spectral function.

**1. O(P²) Regularized Levinson-Durbin (default solver):**
The prediction equations form a P×P Toeplitz system (constant diagonals), where P is the prediction order (typically 50–200). The Levinson-Durbin recursion exploits Toeplitz structure to solve the system in O(P²) operations, compared to O(P³) for a general dense SVD. While the performance difference is negligible for typical prediction orders (P = 100 completes in microseconds either way), Levinson-Durbin is the numerically "correct" algorithm for this problem structure and is specified as the default.

To handle rank-deficient noise filtering, Tikhonov regularization (R → R + λI) is applied to the autocorrelation matrix prior to the recursion, maintaining noise-floor stability while preserving the O(P²) complexity.

SVD-based pseudo-inversion is retained as a fallback for potential future extensions where the prediction matrix may not be exactly Toeplitz (e.g., non-uniform time grids), but it is not the default code path.

```rust
pub enum ToeplitzSolver {
    /// O(P²) Levinson-Durbin recursion with Tikhonov regularization.
    /// Default and recommended for all standard DMFT workflows.
    LevinsonDurbin { tikhonov_lambda: f64 },
    /// O(P³) SVD-based pseudo-inverse. Retained as fallback for
    /// non-Toeplitz extensions (e.g., non-uniform time grids).
    SvdPseudoInverse { svd_noise_floor: f64 },
}

impl Default for ToeplitzSolver {
    fn default() -> Self {
        ToeplitzSolver::LevinsonDurbin { tikhonov_lambda: 1e-8 }
    }
}
```

**2. Exponential Windowing & Deconvolution:**
For metallic Green's functions G(t) that do not decay exponentially (e.g., Fermi-liquid behavior where G(t) ~ t⁻¹), the Toeplitz prediction matrix is poorly conditioned even with regularization — the signal simply does not have the exponential structure that linear prediction assumes. An exponential window W(t) = exp(−η|t|) is applied to G(t) before prediction, artificially enforcing decay and regularizing the pseudo-inverse. The Fourier transform of exp(−η|t|) is a Lorentzian: 2η/(η²+ω²). The broadening parameter η therefore introduces a Lorentzian convolution of half-width η in the frequency domain, which must be deconvolved from the spectral function A(ω) after FFT. The Chebyshev cross-validation (§8.4.3) serves as the check that the deconvolution has not distorted the physics.

**Regularized Lorentzian deconvolution:** The naive deconvolution formula A_true(ω) = A_windowed(ω) · (η²+ω²)/(2η) amplifies noise quadratically at high frequencies: the factor (η²+ω²) grows as ω², so numerical noise in the tails of A_windowed is magnified by a factor of ~ω²/(2η). At ω = 100η, this is a ~5000× amplification — catastrophic for any realistic spectral function.

The deconvolution must be regularized. Two complementary strategies are applied:

1. **Hard frequency cutoff:** Beyond ω_max, the deconvolution factor is clamped to 1.0 (no correction). A_true(ω) is assumed to be negligible in this region. `deconv_omega_max` defaults to 10× the bandwidth of the impurity model.

2. **Tikhonov-style damping:** The deconvolution factor is modified from (η²+ω²)/(2η) to (η²+ω²)/(2η + δ_deconv · ω²), where δ_deconv is a small regularization parameter. At low ω, this is nearly identical to the unregularized form; at high ω, the δ_deconv · ω² term in the denominator tames the growth, bounding the amplification to 1/δ_deconv regardless of frequency.

```
A_true(ω) ≈ A_windowed(ω) · (η² + ω²) / (2η + δ_deconv · ω²)
                                              ^^^^^^^^^^^^^^^^^^^
                                              regularized denominator
```

**3. Spectral Positivity Restoration (mandatory):**
Frequency-space deconvolution of noisy signals inevitably produces numerical ringing, causing the physical spectral function A(ω) to dip below zero. Negative spectral weights will crash the Lanczos bath-discretization step in the DMFT self-consistency loop, which fits a non-negative spectral function to bath parameters.

Post-deconvolution, the spectral function undergoes a mandatory positivity restoration pass:

1. **Diagnostic check:** Compute the total negative weight W_neg = ∫ |min(A(ω), 0)| dω and the total spectral weight W_total = ∫ |A(ω)| dω. If W_neg / W_total > `positivity_warning_threshold` (default: 0.05), emit a structured warning indicating that the deconvolution parameters (η, δ_deconv, ω_max) are likely misconfigured. Large negative regions indicate a physics problem that should not be silently papered over.
2. **Clamping:** All negative values are clamped to a near-zero noise floor: A(ω) = max(A(ω), ε_floor), where ε_floor is a configurable minimum (default: 1e-15).
3. **Sum rule rescaling:** Because clamping adds artificial weight, the entire spectrum is renormalized using the L₁ norm to strictly preserve the spectral sum rule (∫ A(ω) dω = 1). This guarantees physical consistency before the data is passed to the bath discretization step.

```rust
pub struct LinearPredictionConfig {
    /// Solver for the Toeplitz prediction system.
    /// Default: Levinson-Durbin with Tikhonov regularization.
    pub toeplitz_solver: ToeplitzSolver,
    pub prediction_order: usize,
    pub extrapolation_factor: f64,
    /// Exponential broadening parameter η for windowing G(t) before prediction.
    pub broadening_eta: f64,  // default: 0.0 (disabled)
    /// Tikhonov damping for Lorentzian deconvolution.
    pub deconv_tikhonov_delta: f64,  // default: 1e-3
    /// Hard frequency cutoff for deconvolution (in units of bandwidth).
    pub deconv_omega_max: f64,  // default: 10.0 × bandwidth
    /// Floor for spectral positivity clamping.
    pub positivity_floor: f64,  // default: 1e-15
    /// Warning threshold: if clamped negative weight exceeds this fraction
    /// of total spectral weight, emit a diagnostic warning.
    pub positivity_warning_threshold: f64,  // default: 0.05
    /// Fermi-level distortion tolerance: if the global L₁ rescaling shifts
    /// A(ω=0) by more than this relative fraction, emit a warning.
    /// In DMFT, A(ω=0) determines the quasiparticle residue and Luttinger
    /// pinning condition. Even a 2–3% unphysical shift can corrupt transport
    /// properties. Default: 0.01 (1%).
    pub fermi_level_shift_tolerance: f64,  // default: 0.01
}
```

#### 8.4.3 Chebyshev Cross-Validation (Mandatory)

Chebyshev expansion computes the spectral function directly in the frequency domain, bypassing both Trotter error and linear prediction instability. Built alongside TDVP in Phase 4 because cross-validating the two methods is the only rigorous way to verify correct physics.

```rust
```rust
pub struct DMFTLoop<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> {
    pub impurity: AndersonImpurityModel<T>,
    pub dmrg_config: DMRGConfig,
    pub time_evolution: TimeEvolutionConfig,
    pub linear_prediction: LinearPredictionConfig,
    pub solver_mode: SpectralSolverMode,
    pub self_consistency_tol: f64,
    pub max_dmft_iterations: usize,
    backend: B,
}

impl<T, Q, B> DMFTLoop<T, Q, B>
where T: Scalar<Real = f64>, Q: BitPackable, B: LinAlgBackend<T>
{
    pub fn solve(&mut self) -> SpectralFunction {
        loop {
            let chain = self.impurity.discretize_bath();
            let gs = DMRGEngine::new(chain, ...).run();

            // Adaptive solver selection: inspect entanglement spectrum.
            let use_chebyshev_primary = match &self.solver_mode {
                SpectralSolverMode::TdvpPrimary => false,
                SpectralSolverMode::ChebyshevPrimary => true,
                SpectralSolverMode::Adaptive { gap_threshold } => {
                    let gap = gs.mps.entanglement_gap_at_center();
                    gap < *gap_threshold  // gapless → Chebyshev primary
                }
            };

            // Compute both spectral functions; only the role (primary/cross) swaps.
            let g_t = self.tdvp_evolve(&gs);
            let g_t_ext = linear_predict_regularized(&g_t, &self.linear_prediction);
            let spectral_raw = fft(&g_t_ext);
            let spectral_deconv = deconvolve_lorentzian(
                &spectral_raw, &self.linear_prediction
            );
            let spectral_tdvp = restore_positivity(
                &spectral_deconv, &self.linear_prediction
            );
            let spectral_cheb = self.chebyshev_expand(&gs);

            let (spectral_primary, spectral_cross) = if use_chebyshev_primary {
                (&spectral_cheb, &spectral_tdvp)
            } else {
                (&spectral_tdvp, &spectral_cheb)
            };

            self.validate_consistency(spectral_primary, spectral_cross);

            if self.converged(spectral_primary) { return spectral_primary.clone(); }
            self.impurity.update_bath(spectral_primary);
        }
    }

    pub fn solve_with_cancel_flag(
        &mut self,
        cancel: &AtomicBool,
    ) -> Result<SpectralFunction, SolverError> {
        // Same loop but passes cancel flag through to DMRGEngine
    }
}

/// Clamp negative spectral weight and renormalize to preserve sum rule.
/// Includes Fermi-level distortion diagnostic.
fn restore_positivity(
    spectral: &SpectralFunction,
    config: &LinearPredictionConfig,
) -> SpectralFunction {
    let total_weight: f64 = spectral.values.iter().map(|v| v.abs()).sum();
    let negative_weight: f64 = spectral.values.iter()
        .filter(|&&v| v < 0.0)
        .map(|v| v.abs())
        .sum();

    if negative_weight / total_weight > config.positivity_warning_threshold {
        log::warn!(
            target: "tensorkraft::telemetry",
            "SPECTRAL_POSITIVITY_WARNING: {:.1}% of spectral weight is negative \
             (threshold: {:.1}%). Deconvolution parameters (η={}, δ={}, ω_max={}) \
             may need adjustment.",
            100.0 * negative_weight / total_weight,
            100.0 * config.positivity_warning_threshold,
            config.broadening_eta,
            config.deconv_tikhonov_delta,
            config.deconv_omega_max,
        );
    }

    // Record A(ω=0) before restoration for Fermi-level distortion check.
    let a_fermi_before = spectral.value_at_omega_zero();

    // Clamp to floor, then L₁ renormalize to preserve sum rule.
    let mut clamped: Vec<f64> = spectral.values.iter()
        .map(|&v| v.max(config.positivity_floor))
        .collect();
    let new_weight: f64 = clamped.iter().sum();
    // Post-implementation note: Only rescale when both total_weight and
    // new_weight are positive. Edge case: if the original spectrum is
    // entirely negative (pathological deconvolution failure), rescaling
    // with a negative scale factor would flip signs and create nonsense.
    let scale = if total_weight > 0.0 && new_weight > 0.0 {
        total_weight / new_weight
    } else {
        1.0
    };
    for v in &mut clamped { *v *= scale; }

    // Fermi-level distortion diagnostic: the global L₁ rescaling uniformly
    // scales the entire spectrum. If negative ringing concentrated in the
    // high-frequency tails added significant weight, the rescaling factor
    // shifts A(ω=0) downward — potentially corrupting the quasiparticle
    // residue and violating the Luttinger pinning condition.
    let result = SpectralFunction { omega: spectral.omega.clone(), values: clamped };
    let a_fermi_after = result.value_at_omega_zero();

    if a_fermi_before.abs() > 1e-20 {
        let relative_shift = (a_fermi_after - a_fermi_before).abs() / a_fermi_before.abs();
        if relative_shift > config.fermi_level_shift_tolerance {
            log::warn!(
                target: "tensorkraft::telemetry",
                "FERMI_LEVEL_DISTORTION: A(ω=0) shifted by {:.2}% after positivity \
                 restoration (tolerance: {:.1}%). Before: {:.6e}, After: {:.6e}. \
                 This may corrupt the quasiparticle residue. Consider reducing η \
                 or increasing ω_max to reduce tail ringing.",
                100.0 * relative_shift,
                100.0 * config.fermi_level_shift_tolerance,
                a_fermi_before, a_fermi_after,
            );
        }
    }

    result
}
```

---

## 9. Data Flow: DMRG Sweep Step

| Step | Operation | Crate(s) | Detail |
|:-----|:----------|:---------|:-------|
| 1 | **Build H_eff** | tk-contract | Contract L, R, and local MPO into effective Hamiltonian via PathOptimizer + ContractionExecutor. Hermitian conjugates resolved via `is_conjugated` flag on `MatRef` (zero-copy). |
| 2 | **Solve eigenvalue problem** | tk-dmrg | In-house Lanczos/Davidson/Block-Davidson via in-place matvec closure. Contraction buffers pre-allocated from SweepArena. |
| 3 | **SVD truncation** | tk-linalg (faer) | Two-site eigenvector decomposed via truncated SVD (gesdd default, gesvd fallback). Singular values below cutoff discarded. |
| 4 | **Update MPS tensors** | tk-dmrg | SVD factors `.into_owned()` from arena to heap; absorbed into MPS. MixedCanonical gauge preserved via typestate system. Ownership transfer must complete before step 7 (arena reset). |
| 5 | **Subspace expansion** (TDVP only) | tk-dmrg | Site tensors A_L/A_R padded with null-space vectors via matrix-free projection; bond matrix zero-padded and re-truncated using soft D_max policy. |
| 6 | **Update environments** | tk-contract, tk-symmetry | Environment block updated by contracting new MPS tensor with old environment and MPO. |
| 7 | **Arena reset** | tk-core | All temporaries reclaimed in O(1) via SweepArena::reset(). |

---

## 10. Hardware Extension Architecture: CUDA & MPI

### 10.1 Generalized Storage Trait

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

### 10.2 CUDA / GPU Backend

#### 10.2.1 Pinned Memory with Global Budget Tracking

Standard Rust `Vec` and `bumpalo` allocations reside in pageable memory. When copying to `CudaDevice`, the NVIDIA driver silently allocates a hidden page-locked (pinned) buffer, copies data there, then transfers over PCI-e — a hidden double-copy that halves effective bandwidth.

When the `backend-cuda` feature is active, `SweepArena` and `HostDevice` allocations *may* use `cudaMallocHost` for pinned memory, enabling direct DMA transfers. However, pinned memory is page-locked and cannot be swapped to disk. If multiple MPI ranks on the same node each allocate large pinned arenas, the combined allocation can exhaust physical RAM and trigger an OS-level OOM kernel panic.

The architecture employs a **process-local** atomic budget tracker that enforces a per-process pinned-memory ceiling with automatic fallback to pageable memory.

**MPI process-isolation semantics:** Rust's `AtomicUsize` is strictly process-local — each MPI rank runs as an independent OS process with an isolated virtual memory space. Rank 0 cannot read Rank 1's `PINNED_BYTES_ALLOCATED`. The `PinnedMemoryTracker` does *not* coordinate dynamically across ranks at runtime. Instead, the node-level budget is **statically partitioned once at startup** via the `initialize_dmft_node_budget` topology query (§10.2.2), which divides the safe node limit evenly across co-resident ranks. Each rank then independently enforces its pre-negotiated slice using its own process-local atomic counter. This design is correct because pinned-memory allocation is monotonic within a DMFT iteration (allocate at start, release at end) — no dynamic rebalancing between ranks is needed.

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

static PINNED_BYTES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PINNED_BYTES_LIMIT: AtomicUsize = AtomicUsize::new(0);

pub struct PinnedMemoryTracker;

impl PinnedMemoryTracker {
    pub fn initialize_budget(max_bytes: usize) {
        PINNED_BYTES_LIMIT.store(max_bytes, Ordering::Release);
    }

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

    pub fn release(bytes: usize) {
        PINNED_BYTES_ALLOCATED.fetch_sub(bytes, Ordering::Release);
    }
}
```

```rust
static PINNED_FALLBACK_COUNT: AtomicUsize = AtomicUsize::new(0);

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
            storage: ArenaStorage::Pageable(bumpalo::Bump::with_capacity(capacity_bytes)),
        }
    }

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
```

#### 10.2.2 MPI-Aware Budget Initialization

```rust
#[cfg(all(feature = "backend-cuda", feature = "backend-mpi"))]
pub fn initialize_dmft_node_budget(comm: &MpiComm) {
    let total_ram = sys_info::mem_info().unwrap().total;
    let local_ranks = comm.split_by_shared_memory().size();
    let safe_node_limit = (total_ram as f64 * 0.60) as usize;
    let rank_budget = safe_node_limit / local_ranks;
    PinnedMemoryTracker::initialize_budget(rank_budget);
}
```

#### 10.2.3 DeviceCuda Implementation

```rust
#[cfg(feature = "backend-cuda")]
pub struct DeviceCuda {
    stream: cuda::Stream,
    cublas_handle: cublas::Handle,
    cusolver_handle: cusolver::Handle,
}

#[cfg(feature = "backend-cuda")]
impl LinAlgBackend<f64> for DeviceCuda {
    /* dispatch to cuBLAS (with CUBLAS_OP_C for conjugation), cuSOLVER (gesdd default) */
}
// impl LinAlgBackend<Complex64> for DeviceCuda { /* ... */ }
```

#### 10.2.4 Stream-Aware DAG Execution

cuBLAS and cuSOLVER operations are *asynchronous* — they return immediately while the GPU is still computing. If the `ContractionExecutor` walks the DAG dispatching to `DeviceCuda` without explicit synchronization, race conditions arise where a temporary is consumed before the GPU has finished producing it.

**Single-thread submission model (Phase 5):** The topological DAG walk is performed by a **single CPU thread** that issues all CUDA API calls sequentially. This thread walks the contraction tree in topological order, issuing async GPU operations on one or more CUDA streams. It does not contend with Rayon workers for the CUDA driver's per-context mutex because Rayon workers never touch the CUDA runtime. Concurrency is achieved on the GPU side — independent branches of the contraction tree execute on separate CUDA streams, synchronized via `cudaStreamWaitEvent` — not on the CPU side.

This design is sufficient for the single-GPU DMRG pipeline because DMRG's sequential step structure (contract → eigensolve → SVD → update) offers no useful CPU work to overlap with GPU execution within a single step.

```rust
#[cfg(feature = "backend-cuda")]
pub struct DeviceLocation {
    pub device: StorageDeviceKind,  // Host or Cuda { ordinal }
    pub sync_event: Option<cuda::Event>,  // signaled when output is ready
}

#[cfg(feature = "backend-cuda")]
impl<T: Scalar, B: LinAlgBackend<T>> ContractionExecutor<T, B> {
    /// Single-thread DAG walk: one CPU thread issues all CUDA API calls.
    /// Uses cudaStreamWaitEvent between dependent nodes for fine-grained
    /// synchronization without global pipeline stalls.
    /// Rayon workers never touch the CUDA runtime.
    fn execute_cuda(
        &self, graph: &ContractionGraph, tensors: &TensorRegistry<T>,
    ) -> DenseTensor<T> {
        // For each node in topological order:
        // 1. Wait on sync_events of all input dependencies
        // 2. Dispatch async GEMM/SVD on the node's assigned stream
        // 3. Record a new sync_event on the output
        // Final: synchronize only the root node's event before returning
    }
}
```

**Dedicated GPU submission queue (Phase 5+, multi-GPU/pipelined):** For multi-GPU nodes or pipelined execution modes where CPU preprocessing of step N+1 overlaps with GPU execution of step N, the single-thread model is insufficient. In this regime, a dedicated OS thread must be introduced as the sole CUDA orchestrator:

* A bounded MPSC channel (`GPU_Queue`) receives `SectorGemmTask` items from CPU-side task generation.
* The dedicated GPU submission thread pops tasks from the queue, dispatches async `cublasDgemm` calls onto a pool of CUDA streams, records `cuda::Event` completion markers, and signals dependent DAG nodes via continuation passing.
* Rayon workers remain fully isolated from the CUDA runtime, free to continue CPU-bound preprocessing.

This extension is documented here for architectural foresight but is deferred to Phase 5+ because the single-GPU DMRG pipeline does not benefit from it.

#### 10.2.5 GPU Performance Considerations

DMRG is sequential at the sweep level. GPU helps within each step (large GEMM/SVD) but not across steps. Below D ≈ 500, kernel launch overhead may negate the GPU advantage. The architecture supports a hybrid strategy routing small operations to DeviceFaer on CPU and large operations to DeviceCuda, selected by a configurable dimension threshold.

**Three-way LPT partition for block-sparse GEMM:** When `backend-cuda` is active, the partitioned LPT scheduler (§5.3) performs a three-way split. The fragmented tail of tiny sectors runs on CPU via Rayon with single-threaded BLAS — these *never* touch the GPU, as dispatching hundreds of 10×10 or 50×50 GEMMs would saturate the PCIe queue. Tasks between `gpu_flop_threshold` and `blas_flop_threshold` run on CPU with full multithreaded BLAS.

**GPU-tier dispatch (heterogeneous sector sizes):** Tasks above `gpu_flop_threshold` are submitted to the GPU. However, standard `cublasDgemmBatched` requires all matrices in the batch to share identical (M, N, K) dimensions — a constraint that block-sparse sectors with their binomial size distribution cannot satisfy. Naively calling `cublasDgemmBatched` on heterogeneous sectors would either fail or silently serialize. The GPU dispatch therefore uses one of two strategies:

1. **Dimension-grouped batching (portable):** Sector tasks are grouped by their (M, N, K) tuple. Each group is submitted as a single `cublasDgemmBatched` call. For typical U(1) symmetry with ~5–10 distinct sector dimensions, this produces ~5–10 batched kernel launches — acceptable overhead.
2. **Grouped GEMM API (CUDA 12.1+, preferred):** `cublasGemmGroupedBatchedEx` or CUTLASS grouped GEMM accepts heterogeneous (M, N, K) tuples in a single kernel launch, eliminating the per-group launch overhead entirely. This is the preferred path when the CUDA toolkit version supports it, detected at build time via `cudarc` version queries.

#### 10.2.6 NUMA-Aware Pinned Allocation (Multi-GPU)

On multi-socket HPC nodes, `cudaMallocHost` allocates on whichever NUMA node the calling thread happens to be running on. If this NUMA node is remote from the PCIe root complex of the target GPU, pinned-memory DMA transfers traverse the inter-socket interconnect (QPI/UPI), throttling effective bandwidth by 30–50%.

For the single-GPU target (Phase 5), the OS typically schedules the solver thread on the same socket as the GPU, so this is not a concern. For multi-GPU extensions (Phase 5+), the `PinnedArena` must be extended to bind allocations to the correct NUMA node:

- Query the GPU's PCIe bus ID via `cudaDeviceGetPCIBusId`.
- Map the bus ID to a NUMA node via `/sys/bus/pci/devices/<bus_id>/numa_node`.
- Allocate pinned memory on that node via `libnuma`'s `numa_alloc_onnode` or `cudaHostAlloc` with `cudaHostAllocPortable` + explicit thread affinity.

This is deferred to the multi-GPU (NCCL) extension behind the `backend-cuda` feature flag.

**Verification for multi-socket single-GPU deployments:** Even for single-GPU Phase 5 deployments on multi-socket nodes (e.g., dual-socket AMD EPYC), the OS may schedule the solver thread on the wrong socket. Before benchmarking GPU-accelerated DMRG on such systems, users should verify NUMA placement via `numactl --show` and, if necessary, pin the process to the GPU's local socket with `numactl --cpunodebind=<N> --membind=<N>` where `<N>` is the NUMA node of the GPU's PCIe root complex.

### 10.3 MPI / Distributed-Memory Backend

**Mode A — Distributed Block-Sparse Tensors:** Sectors partitioned across ranks; cross-rank communication for boundary sectors. ContractionGraph DAG gains communication nodes and MPI-aware cost model. High risk, Phase 5+.

**Mode B — Embarrassingly Parallel DMFT:** Each rank runs an independent DMRGEngine. Synchronization only at DMFT convergence check via MPI_Allgather. No core changes needed. Recommended first target. Pinned-memory budget is automatically divided across co-resident ranks via `initialize_dmft_node_budget` (§10.2.2).

**Mode B load-imbalance risk:** For single-orbital Bethe lattice DMFT (Phase 4 target), impurity solver iteration counts are relatively uniform across momentum sectors, so the `MPI_Allgather` barrier causes minimal idle time. For multi-orbital or cluster DMFT with strongly heterogeneous bath parameters, solver convergence times can vary by 2–5× across ranks. In this regime, fast ranks idle at the barrier waiting for the slowest rank. Mitigation strategies for Phase 5+: (a) asynchronous convergence checks via non-blocking `MPI_Iallgather` with periodic polling; (b) dynamic work-stealing where idle ranks pull unconverged sectors from slow ranks; (c) adaptive mixing where faster-converging sectors use more aggressive Broyden steps to equalize iteration counts.

### 10.4 Extension Comparison

| Extension | Scope | Risk | Phase | Value |
|:----------|:------|:-----|:------|:------|
| **CUDA (single-node GPU)** | New DeviceCuda + StorageDevice generalization + PinnedArena + PinnedMemoryTracker; single-thread DAG submission | Medium | Phase 5 | High |
| **MPI Mode B (parallel DMFT)** | Application-layer only + pinned-memory topology query | Low | Phase 4–5 | High |
| **MPI Mode A (distributed tensors)** | ContractionExecutor + PathOptimizer | High | Phase 5+ | Medium |
| **Multi-GPU (NCCL)** | DeviceCuda + NCCL wrappers + dedicated GPU submission queue | High | Phase 5+ | Medium |

---

## 11. External Crate Dependencies

| Crate | Used By | Purpose |
|:------|:--------|:--------|
| **faer** | tk-linalg | Dense SVD (gesdd/gesvd), EVD, QR, LU; multithreaded cache-optimized; native lazy conjugation views |
| **oxiblas** | tk-linalg | Sparse ops (9 formats), SIMD BLAS, f128 |
| **bumpalo** | tk-core | Arena allocator for sweep temporaries |
| **smallvec** | tk-core | Stack-allocated small vectors for shapes/strides |
| **rayon** | tk-linalg, tk-contract | Data-parallel iterators (with partitioned LPT scheduling) |
| **num / num-complex** | tk-core | Complex<f64>, numeric traits |
| **dashmap** | tk-symmetry (optional, su2-symmetry) | Thread-safe concurrent hash map for CG coefficient cache |
| **omeco** | tk-contract | Greedy + TreeSA contraction path optimization |
| **cotengrust** | tk-contract | DP-based path optimization |
| ~~**lie-groups**~~ | ~~tk-symmetry (optional)~~ | ~~SU(N) CG coefficients, Casimirs~~ — **Replaced:** CG coefficients computed via hand-rolled Racah formula; no external dependency needed |
| **pyo3** | tk-python | Python bindings for TRIQS integration |
| **rust-numpy** | tk-python | Zero-copy NumPy array interop |
| **spenso** | tk-contract (reference) | Structural tensor graph inspiration |
| **cudarc** | tk-linalg (optional) | Safe Rust wrappers for CUDA driver, cuBLAS, cuSOLVER |
| **mpi** | tk-linalg, tk-dmft (optional) | Rust MPI bindings wrapping system MPI library |
| **sys-info** | tk-dmft (optional) | System RAM query for pinned-memory budget |
| **memmap2** | tk-dmrg (optional) | Memory-mapped file I/O for environment disk offload on memory-constrained nodes |

**Removed:** `eigenvalues` — iterative eigensolvers (Lanczos, Davidson, Block-Davidson) are implemented in-house within `tk-dmrg` to support zero-allocation matvec closures and tight `SweepArena` integration.

---

## 12. Testing & Benchmarking Strategy

### 12.1 Correctness Testing

Each sub-crate carries its own unit test suite with property-based testing (proptest):

- **tk-core:** Round-trip shape permutations, stride arithmetic, arena allocation/reset safety. **Arena ownership boundary:** verify `.into_owned()` produces independent heap copy; verify borrow checker rejects arena references held past `reset()`. **MatRef adjoint round-trip:** `mat.adjoint().adjoint()` recovers original strides and conjugation flag. **PinnedMemoryTracker:** budget enforcement, try_reserve failure, Drop release.
- **tk-symmetry:** Quantum number fusion associativity, flux conservation, bit-pack/unpack round-trip, block-dense equivalence.
- **tk-linalg:** SVD reconstruction error < machine epsilon, eigenvector orthogonality, GEMM reference comparison. **Conjugation-aware GEMM:** verify C = A†·B matches explicit conjugation + transpose + multiply for random complex matrices. **Regularized pseudo-inverse:** verify Tikhonov formula against analytically known cases. **Partitioned LPT scheduling:** verify FLOP-descending order after sort; verify partition split at threshold; verify both heavy and light tasks produce correct results; assert all sectors present in output. **SVD residual guard (debug):** verify `debug_assert!` fires on synthetically corrupted singular values; verify it passes on well-conditioned matrices. **Periodic SVD validation (release):** verify release-mode residual check fires at correct `svd_validation_interval`; verify permanent `gesvd` fallback activates upon synthetic corruption detection; verify amortized overhead ≤ 0.1% by timing 10,000 SVD calls with and without periodic checks. **Physics-triggered SVD validation:** verify `SvdAnomalyDetector` fires immediate check when truncation error spikes by `svd_anomaly_factor`; verify it fires on energy variance spike; verify it does not fire under normal convergence. **BLAS crossover calibration:** verify microbenchmark produces a reasonable threshold (not 0, not infinity).
- **tk-contract:** Path FLOP estimates vs brute-force, result equivalence across optimizers. **Adjoint contraction:** verify that contraction with `.adjoint()` views produces identical results to explicit Hermitian conjugation.
- **tk-dmrg:** Heisenberg chain ground-state energy against reference snapshots (see §12.2), canonical form invariants. **Subspace expansion dimensional consistency:** verify A_L column count increases by D_expand; bond matrix shape matches; SVD re-truncation produces valid MPS. **Matrix-free projection equivalence:** verify that sequential projection (O = A_L†·|R⟩, |R_null⟩ = |R⟩ − A_L·O) produces identical results to explicit (I − A_L·A_L†)·|R⟩ on small test cases; verify orthogonality ⟨A_L|R_null⟩ ≈ 0. **Sector-preserving expansion:** verify that per-sector SVD of |R_null⟩ produces expansion vectors with correct quantum number labels; verify that padded A_L passes flux-conservation assertions. **Dual-layout block storage:** verify `flatten()` produces contiguous buffer matching element-by-element iteration over fragmented blocks; verify `unflatten()` round-trips correctly; verify subspace expansion operates on mutation layout without triggering reallocation of unrelated sectors. **Persistent KrylovWorkspace:** verify workspace vectors are reused across sweep steps (same pointer addresses); verify `Vec::resize` handles bond-dimension changes correctly; verify no allocator calls between steps via instrumented allocator in test. **Adaptive Tikhonov:** verify δ_adaptive tracks σ_discarded_max for bonds with varying entanglement; verify near-product-state bonds get larger δ (less aggressive regularization); verify fully entangled bonds converge to δ_min. **Environment caching:** verify left/right environments match brute-force full contraction for N ≤ 12; verify disk-offloaded environments round-trip correctly via memmap; verify environment memory budget matches O(N·D²·d·D_MPO) estimate. **ARPACK cross-validation (gated behind `test-arpack` feature flag):** For N ≤ 20 systems, cross-validate in-house Lanczos/Davidson eigenvalues against ARPACK-NG (`arpack-ng` FFI). Assert eigenvalue agreement to within 1e-12. This provides an external numerical reference to catch subtle convergence bugs in the in-house solvers without adding a production dependency.
- **tk-dsl:** **Span hygiene:** verify that user variables named `__tk_opsum_builder` are not shadowed by macro internals. **Diagnostic spans:** verify that syntax errors in the DSL produce `syn::Error` with correct span pointing to the offending token, not the macro invocation site.
- **tk-dmft:** Spectral function sum rules, bath discretization accuracy, self-consistency loop convergence against Bethe lattice benchmarks. **Exponential windowing:** verify deconvolved A(ω) matches unwindowed A(ω) for gapped phases (η should be a no-op); verify metallic-phase G(t) prediction does not diverge with η > 0. **Regularized deconvolution:** verify Tikhonov-damped deconvolution recovers known Lorentzian spectral functions; verify high-frequency noise amplification is bounded by 1/δ_deconv; verify hard ω_max cutoff produces smooth A(ω) tails. **Spectral positivity restoration:** verify A(ω) ≥ 0 after restoration; verify sum rule ∫A(ω)dω = 1 is preserved; verify diagnostic warning fires when negative weight exceeds threshold; verify warning does not fire for well-conditioned spectra. **Fermi-level distortion diagnostic:** verify `FERMI_LEVEL_DISTORTION` warning fires when synthetic high-frequency ringing causes A(ω=0) to shift by more than `fermi_level_shift_tolerance` after rescaling; verify warning does not fire when negative weight is small and uniformly distributed. **Levinson-Durbin:** verify solution matches SVD-based pseudo-inverse for random Toeplitz systems; verify Tikhonov regularization stabilizes ill-conditioned systems. **Adaptive solver selection:** verify that `SpectralSolverMode::Adaptive` promotes Chebyshev to primary for gapless test systems (metallic Bethe lattice) and keeps TDVP primary for gapped test systems (Mott insulator). **Soft D_max (physical time):** verify bond dimension does not oscillate between expansion and truncation; verify smooth decay back to D_max over `dmax_decay_time`; verify `advance_expansion_age(dt)` accumulates physical time correctly; verify decay rate is invariant to dt halving (same physical decay curve regardless of step count — run identical physics with dt and dt/2, assert D_target(t) curves match).

#### 12.1.1 Gauge-Invariant Testing Macros

Cross-backend validation must never compare intermediate MPS tensors directly. The SVD has an inherent gauge freedom: singular vectors u and −u are equally valid, and different BLAS implementations will make different sign choices.

```rust
macro_rules! assert_mps_equivalent {
    ($mps_a:expr, $mps_b:expr, $tol:expr) => {
        let overlap = mps_overlap($mps_a, $mps_b).abs();
        assert!((overlap - 1.0).abs() < $tol,
            "MPS overlap = {}, expected ≈ 1.0 (tol = {})", overlap, $tol);
    };
}

macro_rules! assert_svd_equivalent {
    ($svd_a:expr, $svd_b:expr, $tol:expr) => {
        assert_allclose!(&$svd_a.singular_values, &$svd_b.singular_values, $tol);
        let recon_a = reconstruct($svd_a);
        let recon_b = reconstruct($svd_b);
        assert_allclose!(&recon_a, &recon_b, $tol);
    };
}
```

### 12.2 Reference Snapshot Testing

Exact Diagonalization for N=20 (Hilbert space dimension 2²⁰ ≈ 10⁶) is too expensive for routine `cargo test` runs. Dynamic ED is capped at N ≤ 12 (dimension ≤ 4096) in the standard test suite.

For larger validation targets (N=20, 50, 100), the project uses **reference snapshots**: ground-state energies, entanglement spectra, and spectral functions computed once by trusted external frameworks (ITensor, Block2) to full precision, stored as JSON fixture files in `fixtures/`:

```
fixtures/
├── heisenberg_chain_n20_d500.json    # E₀, S_vN, ε_α (from ITensor)
├── heisenberg_chain_n100_d200.json   # E₀ (from Block2)
├── hubbard_bethe_z4_u4.json          # A(ω) sum rule, self-energy (from ED/NRG)
└── README.md                         # provenance, framework versions, dates
```

```rust
#[test]
fn heisenberg_n20_energy_matches_itensor() {
    let reference: ReferenceData = load_fixture("heisenberg_chain_n20_d500.json");
    let energy = run_dmrg_heisenberg(n=20, d=500);
    assert!((energy - reference.energy).abs() < 1e-10,
        "E₀ = {}, reference = {}", energy, reference.energy);
}
```

### 12.3 Property-Based Testing Bounds

`proptest` strategies for `BlockSparseTensor` and `ContractionGraph` generation are explicitly bounded to prevent CI timeouts and flaky builds from ill-conditioned random matrices:

- Maximum tensor rank: 4 in random tests (vs. 6–8 in production)
- Maximum bond dimension: 10 in random tests
- Maximum number of sectors: 20
- Deterministic RNG seed: all proptest configurations use `PROPTEST_SEED` for reproducibility
- Maximum cases per test: 256 (not the default 10,000)
- Ill-conditioned matrix guard: random matrix generators enforce minimum condition number κ > 10⁻⁸ to prevent eigensolver non-convergence in tests

### 12.4 Performance Benchmarks

**Local bare-metal (Criterion.rs):** Tracks wall-clock time for dense SVD at D = 100, 500, 1000, 2000; block-sparse GEMM with U(1) sectors (including partitioned LPT scheduling overhead); full DMRG sweep (N=100, D=200); contraction path optimization for 10/20/50-tensor networks. Run only on dedicated developer machines or self-hosted CI runners.

**CI/CD gating (iai / divan):** Instruction counting replaces wall-clock measurement in cloud CI pipelines. Instruction counts are deterministic and unaffected by CPU throttling, noisy-neighbor effects, or cloud VM variance. Performance regressions are gated at ±2% instruction count change.

```toml
# In CI workflow:
[benchmark.ci]
tool = "iai"           # instruction counting, deterministic
regression_threshold = "2%"

[benchmark.local]
tool = "criterion"     # wall-clock, bare-metal only
```

---

## 13. Risk Analysis & Mitigation

| Risk | Severity | Mitigation |
|:-----|:---------|:-----------|
| faer API instability (pre-1.0) | Medium | Abstract behind LinAlgBackend trait; version-pin; fallback to OpenBLAS FFI |
| oxiblas sparse format coverage gaps | Medium | Validate BSR early; nalgebra-sparse as backup for CSR/CSC |
| SU(2) Wigner-Eckart complexity | High | Defer behind feature flag; structural_contraction callback from day one; fusion-rule multiplicity with map-reduce accumulation for colliding output sectors (§4.4); multiplet-aware SVD truncation (2j+1 weighting, no mid-multiplet splits) |
| FLOP-only path optimization | Medium | Composite CostMetric (α·FLOPs + β·Bytes_Moved) propagates stride info through candidate trees; penalizes paths requiring explicit transposes; zero-cost conjugation via `is_conjugated` flag |
| Thread pool oversubscription (middle-ground pathology) | High | Partitioned LPT dispatch (§5.3): tasks split at calibrated `blas_flop_threshold`; heavy head runs with full BLAS multithreading; light tail cleared by Rayon with single-threaded BLAS. Three-way GPU/CPU/Rayon partition when `backend-cuda` active (`gpu_flop_threshold`). Optional startup microbenchmark calibrates threshold to host machine |
| BLAS thread-count global state mutation | Medium | `set_blas_threads` is global; safe because partitioned scheduler serializes access between phases. `mkl_set_num_threads_local` preferred on MKL ≥2020. `faer` avoids entirely via per-call `Parallelism` parameter (§5.3) |
| Rayon long-tail starvation (binomial sectors) | Medium | LPT scheduling: sort sectors by descending FLOPs before dispatch; structural restoration re-sort after collection |
| TDVP projector-splitting instability | High | Tikhonov-regularized pseudo-inverse (§5.1) with adaptive δ annealing (§8.1.1) + site-tensor subspace expansion; TdvpStabilizationConfig with tunable δ, adaptive scaling to σ_discarded_max, and D_expand |
| TDVP entanglement growth bottleneck | High | Site-tensor subspace expansion: null-space vectors pad A_L/A_R to grow bond dimension; adaptive 1-site/2-site switching |
| Subspace expansion dimensional mismatch | High | Expansion operates on site tensors (D×d space), not bond matrix (D×D space); dimensional consistency enforced by TensorShape metadata |
| Subspace expansion scaling blowup | High | Matrix-free sequential projection (O = A_L†·|R⟩, |R_null⟩ = |R⟩ − A_L·O) preserves O(dD²) cost; explicit projector construction (I − A_L·A_L†) is forbidden |
| Signal-monitor thread leak on solver completion | Medium | `mpsc::channel` with `recv_timeout` guarantees monitor exit on solver completion or panic; `JoinHandle::join()` prevents dangling threads |
| O(N) conjugation memory passes | Medium | MatRef carries `is_conjugated` flag; backends map to BLAS `ConjTrans` or faer `.conjugate()` at zero memory cost |
| Krylov subspace exhaustion | Medium | InitialSubspace enum supports thick restarts; solvers retain best n vectors across restart boundaries; EigenResult reports convergence status |
| Compile-time overhead from deep generics | Medium | Layered dispatch strategy (§5.4): static dispatch in inner loops, `dyn`-eligible at sweep level; feature-gated type combinations; `cargo-llvm-lines` monitoring in CI; 60-second per-crate threshold |
| tk-core bloat invalidating compile cache | Medium | Strict scope discipline: only memory allocation, metadata, Scalar trait, MatRef/MatMut, arena ownership types, error types |
| PyO3 GIL deadlock from Rayon workers | High | AtomicBool cancellation flag; `mpsc`-guarded monitor thread; shutdown sequence inside `allow_threads` closure (§7.5); Rayon workers never touch the GIL |
| PyO3 generic monomorphization | Medium | Type-erased dispatch enum in tk-python; macro-generate match arms |
| DMFT loop convergence sensitivity | High | Mixing schemes (linear, Broyden); validate against Bethe lattice half-filling benchmarks |
| Linear prediction ill-conditioning | High | Levinson-Durbin with Tikhonov regularization (§8.4.2); SVD fallback retained; exponential windowing (η) for metallic G(t); mandatory Chebyshev cross-validation |
| Linear prediction metallic-phase failure | High | Exponential window W(t) = exp(−η\|t\|) enforces decay; FT = Lorentzian 2η/(η²+ω²); η = 0 for gapped phases; adaptive solver selection (§8.4.1) promotes Chebyshev to primary for gapless phases |
| Negative spectral weight crashing bath discretization | High | Mandatory positivity restoration (§8.4.2): clamp to ε_floor, L₁ renormalize to preserve sum rule. Diagnostic warning if clamped weight exceeds 5% of total. **Fermi-level distortion diagnostic:** if A(ω=0) shifts by more than `fermi_level_shift_tolerance` (default 1%) due to global rescaling, a structured warning is emitted identifying the specific risk to quasiparticle residue and Luttinger pinning |
| Block storage heap fragmentation (GPU transfers) | Medium | Phase 1–3: `Vec<DenseTensor<T>>` per-block (mutation layout). Phase 4: dual-layout strategy — fragmented layout for structural mutations (subspace expansion), `FlatBlockStorage` compute layout for GPU DMA/GEMM. `flatten(&arena)` packs blocks into pinned arena memory (not pageable heap), guaranteeing DMA-direct transfers without hidden staging copies (§4.2) |
| Flat-buffer reallocation during subspace expansion | High | TDVP subspace expansion mutates tensor structure (appends columns). Mutation layout (`Vec<DenseTensor<T>>`) keeps this O(D_sector²). Compute layout (`FlatBlockStorage`) is read-only, constructed only after expansion is complete. Dual-layout strategy prevents O(D_total²) reallocation penalty in the innermost time-evolution loop (§4.2) |
| GPU kernel launch overhead on fragmented sectors | Medium | Three-way LPT partition (§5.3): tasks below `gpu_flop_threshold` never touch GPU. GPU-tier tasks submitted via dimension-grouped `cublasDgemmBatched` or `cublasGemmGroupedBatchedEx` (CUDA 12.1+) for heterogeneous sector sizes. Standard batched cuBLAS requires uniform (M,N,K) and would silently serialize on heterogeneous sectors (§10.2.5) |
| Environment memory scaling (O(N·D²·d·D_MPO)) | High | Full in-memory caching is default (§8.3). At D=2000, N=100: ~62 GB. Disk offload via `memmap2`-backed memory-mapped files for memory-constrained nodes (sequential access pattern matches DMRG sweep). Partial caching with recomputation window available but rarely worthwhile due to N/K slowdown factor. `DMRGConfig::environment_offload` field controls strategy |
| Subspace expansion breaking symmetry sector boundaries | Medium | Per-sector SVD of |R_null⟩ guarantees expansion vectors are flux-preserving by construction. H_eff is flux-conserving, so residual is automatically block-sparse. Dense SVD of flattened residual is forbidden (§8.1.1) |
| In-house eigensolver numerical edge cases | Medium | Lanczos/Davidson scoped to lowest-eigenpair Hermitian problems only (~100 lines of critical code). ARPACK-NG cross-validation in integration tests (`test-arpack` feature flag, §12.1). Production dependency avoided; test dependency provides external reference |
| TDVP/linear prediction unreliable for metallic phases | High | Adaptive `SpectralSolverMode` (§8.4.1): entanglement gap checked after DMRG ground state. Gapless → Chebyshev primary, TDVP cross-validation. Gapped → TDVP primary, Chebyshev cross-validation. Both engines always computed; only primary/validation roles swap |
| Memory blow-up in contraction path optimization | Medium | `PathOptimizer::optimize` accepts optional `max_memory_bytes` constraint (§6.2). Default `None` for DMRG (small fixed DAGs). Essential for future PEPS/tree TNS where intermediate memory dominates |
| Eigensolver Krylov vectors inflating arena or fragmenting heap | Medium | Persistent `KrylovWorkspace` owned by `DMRGEngine`, allocated once, reused across all sweep steps by overwriting in place. Not arena-allocated (would inflate high-water mark) and not alloc/drop per step (would fragment heap). Resized via `Vec::resize` if bond dimension changes (§8.2) |
| Contraction reshape memory bandwidth | Medium | Cache-oblivious block-transpose (8×8/16×16) from SweepArena; tblis-style strided contraction as preferred path |
| GPU DAG race conditions | Medium | Single-thread DAG walker issues all CUDA API calls (§10.2.4); per-node cuda::Event synchronization; Rayon workers never touch CUDA runtime |
| GPU driver lock contention (multi-GPU) | Medium | Dedicated GPU submission thread with MPSC queue documented for Phase 5+ multi-GPU extension (§10.2.4); single-GPU Phase 5 avoids contention by design |
| Pinned memory exhaustion (multi-rank OOM) | High | PinnedMemoryTracker: **process-local** atomic budget with CAS loop; enforces statically pre-negotiated per-rank slice (not inter-process shared-memory atomic). Automatic fallback to pageable memory; MPI-aware topology query divides budget across co-resident ranks at startup (§10.2.1, §10.2.2) |
| `flatten()` allocating pageable heap instead of pinned arena | High | `flatten(&arena)` takes `&SweepArena` parameter, packing blocks directly into pinned arena memory when `backend-cuda` is active. Prevents NVIDIA driver's hidden pin-copy-unpin staging dance that halves PCI-e bandwidth. Fresh `Vec::with_capacity` on pageable heap is forbidden in the GPU transfer path (§4.2) |
| FFI BLAS linker collision | Medium | compile_error! enforcing mutual exclusivity of backend-mkl and backend-openblas |
| SVD gauge freedom breaking cross-backend tests | Medium | Gauge-invariant testing macros; compare singular values and state overlaps, not tensor elements |
| CI benchmark flakiness from cloud noise | Medium | iai/divan instruction counting for CI gating; Criterion reserved for bare-metal |
| tk-dsl cyclic dependency on tk-linalg/tk-dmrg | High | Strict two-phase pipeline: tk-dsl produces OpSum only; MPO compilation in tk-dmrg |
| Arena lifetime friction with persistent MPS | High | Explicit ownership boundary (§3.3.2): arena intermediates are `TempTensor<'a>`; final SVD output calls `.into_owned()` before `SweepArena::reset()`; borrow checker enforces statically |
| String-typed operators causing runtime failures | Medium | Strongly-typed operator enums (`SpinOp`, `FermionOp`, `BosonOp`) for standard models; `CustomOp` escape hatch for non-standard operators; typos caught at compile time |
| Fermionic sign errors in non-1D geometries | High | Contraction engine is bosonic-only (§6.4); Jordan-Wigner strings encoded in MPO; native fermionic swap gates deferred to tree/PEPS extension behind `fermionic-swap` feature flag |
| Bond-dimension oscillation during subspace expansion | Medium | Soft D_max policy (§8.1.1): `soft_dmax_factor` allows 10% overshoot; exponential decay in physical time via `dmax_decay_time`; per-bond `expansion_age` (accumulated Σdt) in `TdvpDriver` |
| `LinAlgBackend` object-safety violation (E0038) | High | Trait parameterized at trait level (`LinAlgBackend<T>`) not per-method; `Box<dyn LinAlgBackend<f64>>` valid; `SparseLinAlgBackend<T, Q>` likewise |
| GIL deadlock between monitor thread and main thread | High | `done_tx.send()` + `monitor_handle.join()` execute *inside* `py.allow_threads` closure; GIL is released during shutdown; monitor thread can complete `with_gil` freely |
| SVD `gesdd` silent inaccuracy | High | Three-layer validation (§5.1): Layer 1 (debug) checks every call via `debug_assert!`; Layer 2 (release) checks every K-th call (`svd_validation_interval`, default 1000, 0.1% overhead) with permanent `gesvd` fallback; Layer 3 (TDVP physics-triggered) forces immediate check if truncation error spikes by >10× or energy variance exceeds tolerance — catches acute corruption the moment it impacts observables, closing the K-step blind spot that would destroy the MPS tangent space |
| Lorentzian deconvolution noise amplification | High | Tikhonov-damped deconvolution (η²+ω²)/(2η+δ·ω²) bounds high-frequency amplification to 1/δ; hard cutoff ω_max clamps correction beyond bandwidth; `deconv_tikhonov_delta` and `deconv_omega_max` in `LinearPredictionConfig` |
| SU(2) fusion-rule one-to-many in task generation | Medium | Abelian `compute_fusion_rule` returns single sector; SU(2) j₁⊗j₂ yields multiple irreps. `SectorGemmTask` generation must fan out to `Vec<SectorGemmTask>` per input pair. Memory bound: O(D_reduced² × j_max) ≈ 16 MB for typical parameters (§4.4); pre-allocated with `Vec::with_capacity`. Documented in §4.4; scoped to `su2-symmetry` feature flag |
| Soft D_max decay coupling to adaptive time-stepping | Medium | Expansion age tracked in accumulated physical time (Σdt), not discrete iteration count (§8.1.1). Decay formula `exp(−t_physical / dmax_decay_time)` is invariant to TDVP integrator's adaptive dt. Prevents numerical tolerance (step size) from coupling to physical relaxation timescale (entanglement decay) |
| NUMA-blind pinned memory on multi-socket nodes | Medium | Single-GPU: OS schedules correctly. Multi-GPU: must bind pinned allocations to GPU's PCIe NUMA node via `numa_alloc_onnode`. Documented in §10.2.6; deferred to Phase 5+ |
| MPI Mode B load imbalance (heterogeneous solvers) | Medium | Single-orbital Bethe: minimal variance. Multi-orbital/cluster: 2–5× iteration spread. Async `MPI_Iallgather` or dynamic work-stealing for Phase 5+. Documented in §10.3 |
| Pinned-memory fallback silent performance cliff | Medium | Fallback emits structured telemetry event with `PINNED_FALLBACK_COUNT` counter; exposed in `DMRGEngine` stats; dashboards surface the 2× bandwidth regression |
| `hamiltonian!{}` macro variable shadowing | Medium | Call-site spans for user variables; mixed-site spans for macro internals (§7.2.1). Hygienic identifiers prevent collision even for pathological user variable names |
| `hamiltonian!{}` macro cryptic error messages | Medium | `syn::parse::Parse` implementations for DSL grammar; `syn::Error::new_spanned` attaches diagnostics to the offending token, not the macro invocation site (§7.2.2) |

---

## 14. Implementation Roadmap

| Phase | Target | Deliverables |
|:------|:-------|:-------------|
| **Phase 1** | Months 1–3 | tk-core (strict scope, MatRef with conjugation flag, arena ownership boundary with `.into_owned()`, PinnedMemoryTracker), tk-symmetry (U(1), Z₂, BitPackable, PackedSectorKey), tk-linalg (DeviceFaer with conjugation-aware GEMM, gesdd default, regularized pseudo-inverse, partitioned LPT-scheduled block_gemm with optional BLAS crossover calibration, `set_blas_threads` safety documented). >90% test coverage. SVD + conjugation benchmarks. iai CI integration. `cargo-llvm-lines` compile-time monitoring. |
| **Phase 2** | Months 4–6 | tk-contract (DAG, greedy optimizer, conjugation flag propagation, bosonic-only contraction documented), tk-dsl (Index, typed operator enums, OpSum, hamiltonian!{} macro with span hygiene and diagnostic reporting — AST generation only). OpSum→MPO compilation in tk-dmrg. Heisenberg ground state matches reference snapshots. |
| **Phase 3** | Months 7–9 | tk-dmrg (MPS typestates, two-site sweep, in-house Lanczos/Davidson/Block-Davidson with heap-owned Krylov workspace, site-tensor subspace expansion with per-sector SVD for flux preservation). N=100 Heisenberg at D=500. Gauge-invariant test macros. ITensor/TeNPy comparison via snapshot fixtures. ARPACK cross-validation behind `test-arpack` feature flag. |
| **Phase 4** | Months 10–12 | tk-dmft (TDVP with Tikhonov + subspace expansion + soft D_max, adaptive TDVP/Chebyshev solver selection, linear prediction with Levinson-Durbin default + exponential windowing + spectral positivity restoration, Chebyshev, DMFT loop). tk-python (GIL release, AtomicBool cancellation, rust-numpy). StorageDevice generalization. **Flat-buffer block storage refactor** (`FlatBlockStorage` in `BlockSparseTensor`, prerequisite for Phase 5 GPU). Memory-constrained `PathOptimizer` (`max_memory_bytes`). MPI Mode B with pinned-memory topology query. Bethe lattice validation (insulating + metallic phases; verify adaptive solver promotes Chebyshev for metallic; verify positivity restoration on metallic spectra). |
| **Phase 5** | Months 13+ | DeviceCuda + PinnedArena + CudaArena (budget-managed); single-thread DAG submission; three-way LPT partition (GPU/CPU-BLAS/Rayon) with `gpu_flop_threshold` and batched `cublasDgemmBatched` for GPU-tier sectors; flat-buffer `BlockSparseTensor` enables single-DMA transfers. SU(2) non-Abelian. TreeSA/DP optimizers. Fermionic swap gates for tree/PEPS (`fermionic-swap` feature flag). Multi-orbital DMFT. MPI Mode A if needed. Community release. PyPI wheel builds (pure-Rust backends only). |
| **Phase 5+** | TBD | Multi-GPU (NCCL) with dedicated GPU submission queue. NUMA-aware pinned allocation. Async MPI convergence checks for heterogeneous cluster DMFT. |

---

## 15. Conclusion

This architecture provides a rock-solid foundation for a Rust tensor network library that is modular, safe, and performant. By decoupling tensor shape from storage, abstracting linear algebra backends behind traits, separating contraction path optimization from execution, and encoding physical gauge conditions in the type system, the design eliminates entire categories of bugs at compile time while preserving the computational intensity required for state-of-the-art quantum many-body simulations.

The v8.5 revision incorporates cross-cutting findings from the draft implementation of all crates (tk-core through tk-python). Key changes include: `TensorCow` eliminated in favor of a single `TensorStorage<'a, T>` enum; `DenseTensor` gains lifetime parameter and offset field for zero-copy slicing; `BlockSparseTensor` gains `leg_directions` for flux rule enforcement; `Scalar` trait expanded with `Sub`, `Neg`, `Debug`, `'static`, `from_real_imag()`, and `imaginary_unit()`; CG cache uses hand-rolled Racah formula with DashMap-based thread-safe caching; `DenseTensor` and `BlockSparseTensor` `Clone` identified as the #1 cross-cutting issue affecting four downstream crates; `DMRGConfig` needs split into immutable config and mutable state; and `ComplexU1` variant blocked pending `LinAlgBackend<Complex<f64>>` implementation. The previous v8.4 revision addressed: (1) the `flatten()` method now accepts a `&SweepArena` parameter and packs fragmented blocks directly into pinned arena memory (when `backend-cuda` is active), preventing the NVIDIA driver's hidden pin-copy-unpin staging dance that would halve PCI-e bandwidth if flat buffers were naively allocated on the pageable heap; (2) SVD validation gains a third physics-triggered layer for TDVP — if truncation error spikes by >10× or energy variance jumps between consecutive time steps, an immediate out-of-band residual check is forced, closing the K-step blind spot that would allow corrupted gauge restorations to compound exponentially and destroy the MPS tangent space before the periodic modulo counter catches it; and (3) the `PinnedMemoryTracker` is explicitly documented as a process-local guard enforcing a statically pre-negotiated per-rank budget slice, clarifying that the `AtomicUsize` does not and cannot coordinate across MPI ranks at runtime.

Across fourteen revision cycles, the design has been hardened against: dimensional inconsistencies in TDVP subspace expansion, scaling violations in null-space projection, symmetry-sector corruption from dense expansion SVD, bond-dimension oscillation at truncation thresholds, soft D_max decay coupling to adaptive time-stepping, flat-buffer reallocation penalty during structural mutations, flat-buffer allocation on pageable heap bypassing pinned DMA path, Rust object-safety violations in backend traits, cyclic crate dependencies, GIL deadlocks (Rayon workers, monitor thread shutdown ordering) and thread leaks in Python bindings, pinned-memory exhaustion under multi-rank MPI, MPI atomic process-isolation confusion, silent pinned-memory performance cliffs, NUMA-blind GPU allocations, Rayon long-tail starvation from binomial sector distributions, thread-pool oversubscription from binary regime switching, BLAS global thread-count race conditions, GPU kernel launch overhead on fragmented sectors, GPU heterogeneous-batch silent serialization, O(N) conjugation memory bandwidth waste, Fourier transform errors in spectral windowing, deconvolution noise amplification, negative spectral weight from deconvolution ringing, Fermi-level spectral distortion from global L₁ rescaling, static Tikhonov regularization masking physics in product-state bonds, unreliable TDVP linear prediction for metallic phases, SVD silent inaccuracy escaping into production builds, SVD corruption blind spot compounding through TDVP gauge restoration, SU(2) task generation memory blow-up, SU(2) fusion-rule multiplicity in task generation, linear prediction instability in metallic phases, eigensolver Krylov workspace heap fragmentation, environment memory scaling exceeding node RAM, memory blow-up in contraction path optimization, MPI barrier load imbalance, arena lifetime conflicts with persistent state, monomorphization-driven compile-time explosion, SVD gauge freedom in cross-backend tests, FFI linker collisions between vendor BLAS libraries, block storage heap fragmentation for GPU transfers, proc-macro variable shadowing, cryptic DSL error diagnostics, TensorCow double-indirection overhead, missing DenseTensor/BlockSparseTensor Clone implementations, DenseTensor lifetime friction in contraction executor generics, and DMRGConfig non-clonability blocking Python bindings.

The clear crate boundaries and feature-flag system ensure that the library can evolve incrementally—adding non-Abelian symmetries, GPU backends, and additional lattice geometries—without destabilizing the core infrastructure. The phased implementation roadmap prioritizes delivering a working DMRG solver as early as Phase 3, enabling real physics research to begin while the DMFT integration matures in parallel.
