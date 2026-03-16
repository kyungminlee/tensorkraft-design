# Software Architecture Design Document
# Rust-Based Tensor Network Library
**for Quantum Many-Body Physics — DMRG Algorithms & DMFT Impurity Solver Integration**

*Version 7.0 — March 2026 | Status: Architectural Specification (Pre-Implementation)*

---

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

- **tk-core:** Tensor data structures with strict shape/storage separation, arena allocation, and Copy-on-Write semantics. Matrix views (`MatRef`) carry lazy conjugation flags for zero-copy Hermitian transposes.
- **tk-symmetry:** Native Abelian symmetry support (U(1), Z₂) via block-sparse formats with bit-packed sector keys, with a clear extension path for non-Abelian SU(2) via Wigner-Eckart factorization.
- **tk-linalg:** Trait-based linear algebra backend abstraction defaulting to faer (dense) and oxiblas (sparse), swappable via feature flags. GEMM dispatch propagates conjugation metadata to hardware micro-kernels. Includes regularized pseudo-inverse for numerically stable gauge restoration. LPT-scheduled block-sparse parallelism prevents Rayon long-tail starvation.
- **tk-contract:** DAG-based contraction engine with separated path optimization and execution phases. Bosonic tensor legs only; fermionic sign rules encoded in MPO via Jordan-Wigner.
- **tk-dsl:** Ergonomic API with intelligent indices, strongly-typed operator enums (`SpinOp`, `FermionOp`, `BosonOp`) with `CustomOp` escape hatch, an `OpSum` builder pattern, and a `hamiltonian!{}` proc-macro that generates uncompressed operator sums as compile-time boilerplate reduction; actual MPO compression is a runtime operation delegated to `tk-dmrg`.

Additionally, integration crates **tk-dmrg** and **tk-dmft** implement the full DMRG sweep algorithm, iterative eigensolvers (Lanczos/Davidson, written in-house), MPO compilation from `OpSum`, time-evolution methods (TDVP with Tikhonov-regularized gauge shifts and site-tensor subspace expansion for entanglement growth, TEBD as fallback), and the DMFT self-consistency loop. A **tk-python** crate provides PyO3 bindings with explicit GIL management, thread-safe `AtomicBool` cancellation with `mpsc`-guarded monitor thread lifecycle, and zero-copy NumPy interop.

---

## 2. Workspace & Crate Architecture

The library is structured as a Cargo workspace containing focused, independently testable sub-crates.

### 2.1 Workspace Layout

```
tensorkraft/
├── Cargo.toml              # workspace root
├── crates/
│   ├── tk-core/             # Tensor shape, storage, memory mgmt, MatRef/MatMut
│   ├── tk-symmetry/         # Quantum numbers, block-sparse formats
│   ├── tk-linalg/           # Backend abstraction (faer, oxiblas), LPT scheduling
│   ├── tk-contract/         # DAG engine, path optimization
│   ├── tk-dsl/              # Macros, OpSum, lattice builders
│   ├── tk-dmrg/             # DMRG sweeps, eigensolvers, MPS/MPO, OpSum→MPO compilation
│   ├── tk-dmft/             # DMFT loop, bath discretization, TDVP/TEBD
│   └── tk-python/           # PyO3 bindings for DMFT integration
├── benches/                 # Criterion benchmarks (local bare-metal only)
├── fixtures/                # Reference snapshot data (ED energies, spectra)
├── examples/                # Heisenberg chain, Hubbard DMFT, etc.
└── tests/                   # Integration tests
```

### 2.2 Crate Dependency Graph

| Crate | Responsibility | Depends On |
|:------|:---------------|:-----------|
| **tk-core** | Tensor shape/stride metadata, contiguous storage buffers, `MatRef`/`MatMut` with lazy conjugation flag, arena allocators with explicit ownership boundary (`TempTensor`/`.into_owned()`) and pinned-memory budget tracking, TensorCow (Copy-on-Write), element-type generics, error types | *(none — leaf crate)* |
| **tk-symmetry** | QuantumNumber trait, BitPackable trait, U(1)/Z₂ implementations, PackedSectorKey, block-sparse storage, Wigner-Eckart scaffolding for SU(2) | tk-core |
| **tk-linalg** | LinAlgBackend trait (conjugation-aware GEMM), SVD/EVD dispatch (gesdd default), regularized pseudo-inverse, DeviceFaer and DeviceOxiblas implementations, LPT-scheduled block-sparse dispatch, Rayon-parallel element-wise ops | tk-core, tk-symmetry |
| **tk-contract** | ContractionGraph DAG, PathOptimizer trait, greedy/TreeSA heuristics, ContractionExecutor with reshape-free GEMM, conjugation flag propagation. Bosonic legs only (§6.4); fermionic swap gates deferred to Phase 5+ | tk-core, tk-symmetry, tk-linalg |
| **tk-dsl** | Index struct with unique IDs and prime levels, `hamiltonian!{}` proc-macro (generates `OpSum` AST only), typed operator enums (`SpinOp`, `FermionOp`, `BosonOp`, `CustomOp`), OpSum builder, Lattice trait, snake-path mappers | tk-core, tk-symmetry |
| **tk-dmrg** | MPS/MPO types with typestate canonicality, `OpSum → MPO` SVD compression, two-site sweep engine, in-house Lanczos/Davidson/Block-Davidson eigensolvers, SVD truncation, site-tensor subspace expansion | tk-core, tk-symmetry, tk-linalg, tk-contract, tk-dsl |
| **tk-dmft** | Anderson Impurity Model mapping, bath discretization (Lanczos tridiagonalization), TDVP time evolution (with Tikhonov regularization and site-tensor subspace expansion), TEBD fallback, linear prediction, Chebyshev expansion, DMFT self-consistency loop, pinned-memory budget initialization for MPI | tk-dmrg (and transitively all) |
| **tk-python** | PyO3/maturin bindings with explicit GIL release, `AtomicBool`-based cancellation with `mpsc`-guarded monitor thread lifecycle (no GIL re-acquisition from Rayon workers), zero-copy NumPy via rust-numpy | tk-dmft |

**Key architectural constraint (cyclic dependency prevention):** `tk-dsl` generates only uncompressed `OpSum` structures and lattice geometry mappings. It has no dependency on `tk-linalg` or `tk-dmrg`. The `OpSum → MPO` compilation step (which requires SVD compression via `LinAlgBackend`) lives entirely within `tk-dmrg`, which has access to both `tk-dsl` and `tk-linalg`.

### 2.3 Feature Flags

| Feature Flag | Effect | Default | Exclusivity |
|:-------------|:-------|:--------|:------------|
| **backend-faer** | Enables DeviceFaer for dense SVD/EVD/QR using the pure-Rust faer crate | Yes | — |
| **backend-oxiblas** | Enables DeviceOxiblas for sparse BSR/CSR operations and extended-precision (f128) math | Yes | — |
| **backend-mkl** | Links Intel MKL via FFI for vendor-optimized BLAS on Intel hardware | No | Conflicts with `backend-openblas` |
| **backend-openblas** | Links OpenBLAS via FFI for broad HPC cluster compatibility | No | Conflicts with `backend-mkl` |
| **su2-symmetry** | Activates non-Abelian SU(2) support with Clebsch-Gordan caching (depends on lie-groups crate) | No | — |
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

**Scope discipline:** `tk-core` is the leaf crate upon which the entire workspace depends. To prevent compilation-cache invalidation cascading through the workspace, `tk-core` is strictly limited to: memory allocation (`SweepArena`, `PinnedMemoryTracker`), dimensional metadata (`TensorShape`, `TensorStorage`, `TensorCow`), matrix view types (`MatRef`, `MatMut` with conjugation flags), the `Scalar` trait hierarchy, and shared error types. Mathematical operations on tensors (addition, trace, contraction) belong in `tk-linalg` or `tk-contract`.

### 3.1 Core Type Definitions

```rust
/// Dimensional metadata: shapes and strides for zero-copy views.
pub struct TensorShape {
    dims: SmallVec<[usize; 6]>,    // typical rank ≤ 6
    strides: SmallVec<[usize; 6]>,  // row-major by default
}

/// Contiguous memory buffer, generic over element type.
pub struct TensorStorage<T: Scalar> {
    data: Vec<T>,  // or ArenaVec<T> when arena-allocated
}

/// The primary dense tensor: shape metadata + owned/borrowed storage.
pub struct DenseTensor<T: Scalar> {
    shape: TensorShape,
    storage: TensorCow<T>,  // Cow semantics: Borrowed view or Owned data
}

/// Copy-on-Write storage wrapper.
pub enum TensorCow<'a, T: Scalar> {
    Borrowed(&'a TensorStorage<T>),   // zero-copy view
    Owned(TensorStorage<T>),           // materialized copy
}
```

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
    ) -> DenseTensor<T> { /* ... */ }

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
pub type TempTensor<'a, T> = DenseTensor<T>;  // with TensorCow::Borrowed(&'a ...)

impl<'a, T: Scalar> DenseTensor<T> {
    /// Materialize into heap-allocated owned storage.
    /// Called exactly once per sweep step on the final SVD output
    /// before SweepArena::reset() reclaims all temporaries.
    pub fn into_owned(self) -> DenseTensor<T> {
        match self.storage {
            TensorCow::Owned(_) => self,
            TensorCow::Borrowed(storage) => DenseTensor {
                shape: self.shape,
                storage: TensorCow::Owned(storage.clone()),
            },
        }
    }
}
```

The rule is simple: everything computed within a DMRG step lives in the arena. The *only* outputs that escape are the updated site tensors, which must call `.into_owned()` before `SweepArena::reset()`. The borrow checker enforces this statically — any attempt to hold a borrowed arena reference past the reset point is a compile error, not a runtime bug. See §9, step 4 for the exact point in the data flow where this ownership transfer occurs.

#### 3.3.3 Copy-on-Write (Cow) Semantics

Shape-manipulation operations (transpose, permute, reshape) return `TensorCow::Borrowed` views whenever the operation can be expressed as a pure stride permutation. Data is cloned into `TensorCow::Owned` only when a contiguous memory layout is strictly required (e.g., as input to a GEMM kernel). This pattern, modeled after the rstsr framework, ensures copies are generated only when mathematically necessary.

### 3.4 The Scalar Trait Hierarchy

```rust
pub trait Scalar:
    Copy + Clone + Send + Sync + num::Zero + num::One
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
{
    type Real: Scalar;  // f64 for Complex<f64>, f64 for f64
    fn conj(self) -> Self;
    fn abs_sq(self) -> Self::Real;
    fn from_real(r: Self::Real) -> Self;
    /// Whether complex conjugation is a no-op for this type.
    /// Returns true for f32/f64, false for Complex<f32>/Complex<f64>.
    fn is_real() -> bool;
}
```

The `is_real()` method allows the contraction engine to skip setting `is_conjugated` entirely for real-valued models, avoiding unnecessary flag checks in tight loops.

---

## 4. Physical Symmetries & Block Sparsity (tk-symmetry)

In quantum systems with global symmetries, tensors become block-sparse: elements are non-zero only when the algebraic sum of incoming quantum numbers equals the outgoing quantum numbers (the "flux rule"). Exploiting this structure avoids storing and computing zeros, yielding order-of-magnitude speedups.

### 4.1 Quantum Number Trait & Bit-Packing

```rust
pub trait QuantumNumber:
    Clone + Eq + Hash + Ord + Debug + Send + Sync
{
    fn identity() -> Self;
    fn fuse(&self, other: &Self) -> Self;
    fn dual(&self) -> Self;
}

/// Extension trait: compresses a quantum number into a fixed-width bitfield.
/// Enables O(log N) sector lookup via single-cycle u64 comparisons
/// instead of element-by-element SmallVec traversal.
pub trait BitPackable: QuantumNumber {
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

/// INVARIANT: sector_keys is always sorted for O(log N) binary search.
/// Keys are contiguous u64s in a cache-friendly array; the CPU prefetcher
/// pulls the entire key array into L1 cache, and search resolves via
/// single-cycle register comparisons.
pub struct BlockSparseTensor<T: Scalar, Q: BitPackable> {
    indices: Vec<QIndex<Q>>,
    sector_keys: Vec<PackedSectorKey>,
    sector_blocks: Vec<DenseTensor<T>>,
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
}
```

**Capacity note:** With 8 tensor legs and 8 bits per quantum number (the U(1) default), bit-packing fits in exactly 64 bits. For models requiring larger charge range or higher-rank tensors (e.g., multi-orbital Hubbard with U(1)\_charge ⊗ U(1)\_spin, rank > 8), the `PackedSectorKey` can be promoted to `u128`, giving 16 bits per leg at rank 8 or supporting rank 16 at 8 bits per leg.

**Non-Abelian fallback:** SU(2) irreps are not trivially bit-packable due to multiplicity structure. The `BlockSparseTensor` definition is parameterized over `Q: BitPackable` for the Abelian fast path. The SU(2) extension (§4.4) uses its own `WignerEckartTensor` with `SU2Irrep`-keyed storage, not constrained to `BitPackable`.

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

**Known refactoring requirement (fusion-rule multiplicity):** The Abelian block-sparse GEMM (§5.3.1) assumes a one-to-one mapping in `compute_fusion_rule(*key_a, *key_b) -> Option<PackedSectorKey>`: each input sector pair produces at most one output sector. For SU(2), the tensor product of irreps yields *multiple* output sectors: j₁ ⊗ j₂ = |j₁−j₂| ⊕ (|j₁−j₂|+1) ⊕ ... ⊕ (j₁+j₂). The `SectorGemmTask` generation loop in §5.3.1 must be generalized to produce a `Vec<SectorGemmTask>` per input pair, with each task's output block weighted by the corresponding Clebsch-Gordan coefficient. The `structural_contraction` callback handles the coefficient evaluation, but the task-generation fan-out is a structural change to the LPT scheduling phase. This refactoring is scoped to the `su2-symmetry` feature flag and does not affect the Abelian code path.

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

    fn eigh_lowest(&self, mat: &MatRef<T>, k: usize) -> (Vec<T::Real>, DenseTensor<T>);
    fn qr(&self, mat: &MatRef<T>) -> (DenseTensor<T>, DenseTensor<T>);

    /// Tikhonov-regularized pseudo-inverse for TDVP gauge restoration.
    /// Computes s_i / (s_i² + δ²) instead of 1/s_i, preventing NaN
    /// explosion when singular values approach machine zero.
    /// See §8.1 for usage context.
    fn regularized_svd_inverse(
        &self,
        s_values: &[T::Real],
        u: &DenseTensor<T>,
        vt: &DenseTensor<T>,
        cutoff: T::Real,  // δ: the Tikhonov parameter
    ) -> DenseTensor<T> {
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

    /// Block-sparse GEMM with LPT scheduling.
    /// Sectors are sorted by descending FLOP cost before Rayon dispatch
    /// to prevent long-tail thread starvation. See §5.3.1.
    fn block_gemm(
        &self, a: &BlockSparseTensor<T, Q>, b: &BlockSparseTensor<T, Q>,
    ) -> BlockSparseTensor<T, Q>;
}
```

**SVD algorithm selection rationale:** The divide-and-conquer algorithm (`gesdd`) is substantially faster than QR-iteration (`gesvd`) for the moderately sized dense matrices arising from DMRG two-site updates (typically 2D² × 2D², where D is bond dimension 100–2000). The trade-off is higher workspace memory (O(n²) vs O(n)), which is acceptable given the arena allocation strategy. Convergence failure with `gesdd` is rare but can occur with highly degenerate singular values; the automatic fallback to `gesvd` handles this gracefully.

**Silent inaccuracy guard:** `gesdd` can occasionally return without signaling an error while producing inaccurate small singular values for pathologically ill-conditioned matrices. In debug and test builds, the `svd_truncated` wrapper validates the reconstruction residual after every call:

```rust
// Inside svd_truncated default implementation, after obtaining `result`:
debug_assert!({
    let residual = reconstruction_error(mat, &result);  // ‖A − UΣV†‖_F
    let norm = frobenius_norm(mat);                      // ‖A‖_F
    residual / norm < 1e-10
}, "SVD reconstruction residual {:.2e} exceeds tolerance", residual / norm);
```

This check is compiled out in release builds (`--release`), adding zero production overhead. It catches silently corrupt SVD results during development and CI, before the Tikhonov regularization masks the damage downstream.

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

#[cfg(all(feature = "backend-faer", feature = "backend-oxiblas"))]
pub type DefaultDevice = DeviceAPI<DeviceFaer, DeviceOxiblas>;

/// DefaultDevice delegates LinAlgBackend<T> to its dense component.
impl<T: Scalar> LinAlgBackend<T> for DefaultDevice
where DeviceFaer: LinAlgBackend<T>
{
    fn gemm(&self, alpha: T, a: &MatRef<T>, b: &MatRef<T>, beta: T, c: &mut MatMut<T>) {
        self.dense.gemm(alpha, a, b, beta, c)
    }
    // ... delegate remaining methods ...
}
```

### 5.3 Hybrid Parallelism & Thread Pool Management

Mixing Rayon's work-stealing scheduler with multithreaded BLAS backends creates thread oversubscription. The architecture defines two dynamically-selected parallelism regimes:

**Regime 1 — Fat Sectors (Dense-Dominated):** Few massive symmetry sectors (D > 1000). Rayon disabled; BLAS backend uses full machine thread pool per GEMM/SVD.

**Regime 2 — Fragmented Sectors (Sparsity-Dominated):** Many small sectors. BLAS threading forced to 1; Rayon `par_iter` distributes independent sector GEMMs across all cores. Sectors are pre-sorted by descending FLOP cost (LPT scheduling) to prevent long-tail starvation (§5.3.1).

```rust
pub enum ThreadingRegime {
    FatSectors { blas_threads: usize },
    FragmentedSectors { rayon_threads: usize },
}

impl ThreadingRegime {
    pub fn select<T: Scalar, Q: BitPackable>(
        tensor: &BlockSparseTensor<T, Q>, n_cores: usize,
    ) -> Self {
        let max_sector_dim = tensor.max_sector_dimension();
        if max_sector_dim > 500 && tensor.n_sectors() < n_cores {
            Self::FatSectors { blas_threads: n_cores }
        } else {
            Self::FragmentedSectors { rayon_threads: n_cores }
        }
    }
}
```

For TEBD, all even-bond gates commute: Rayon distributes even-bond SVDs in parallel using single-threaded BLAS per SVD, then repeats for odd bonds.

#### 5.3.1 LPT-Scheduled Block-Sparse Dispatch

In Abelian symmetric models, symmetry sector sizes follow a binomial distribution: a few massive blocks near the center (e.g., Sz = 0) and many tiny blocks at the edges. If sectors are dispatched to Rayon in storage order (sorted by quantum number), a heavy sector at the end of the array causes all other cores to idle while a single thread finishes. This "long-tail" problem negates the benefit of parallelism.

The block-sparse GEMM decouples execution order from storage order using Longest Processing Time (LPT) scheduling:

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
        // Phase 1: Task Generation
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

        // Phase 2: LPT Scheduling — heaviest GEMMs dispatched first.
        tasks.sort_unstable_by(|x, y| y.flops.cmp(&x.flops));

        let regime = ThreadingRegime::select(a, num_cpus::get());
        let mut results: Vec<(PackedSectorKey, DenseTensor<T>)> = match regime {
            ThreadingRegime::FragmentedSectors { .. } => {
                tasks.into_par_iter()
                    .map(|task| task.execute::<D>(T::one(), T::zero()))
                    .collect()
            }
            ThreadingRegime::FatSectors { .. } => {
                tasks.into_iter()
                    .map(|task| task.execute::<D>(T::one(), T::zero()))
                    .collect()
            }
        };

        // Phase 3: Structural Restoration — re-sort by PackedSectorKey
        // to restore the binary-search invariant on the output tensor.
        results.sort_unstable_by_key(|(key, _)| *key);
        let (out_keys, out_blocks) = results.into_iter().unzip();

        BlockSparseTensor {
            indices: compute_output_indices(&a.indices, &b.indices),
            sector_keys: out_keys,
            sector_blocks: out_blocks,
            flux: a.flux.fuse(&b.flux),
        }
    }
}
```

The three-phase design (generate → LPT sort → restore invariant) achieves perfect core saturation without interior mutability, `UnsafeCell`, or complex slice-splitting. Rayon's work-stealing ensures that as threads finish heavy tasks, they immediately steal lighter tasks from the tail.

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

The contraction engine separates: (1) finding the optimal contraction sequence (NP-hard), and (2) executing each pairwise contraction.

### 6.1 Contraction Graph (DAG)

```rust
pub enum ContractionNode {
    Input { tensor_id: TensorId },
    Contraction {
        left: Box<ContractionNode>,
        right: Box<ContractionNode>,
        contracted_indices: Vec<(IndexId, IndexId)>,
    },
}

pub struct ContractionGraph {
    inputs: Vec<TensorId>,
    root: ContractionNode,
    estimated_flops: f64,
    estimated_bytes_moved: f64,  // memory traffic including transposes
    estimated_memory: usize,
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
        inputs: &[&TensorShape],
        index_map: &IndexMap,
        cost: &CostMetric,
    ) -> ContractionGraph;
}

pub struct GreedyOptimizer;                                        // O(n³)
pub struct TreeSAOptimizer { pub max_iterations: usize, ... }      // Simulated annealing
pub struct DPOptimizer { pub max_width: usize }                    // Dynamic programming
```

The optimizer propagates stride information through candidate contraction trees. At each pairwise node, it checks whether the contracted indices are already contiguous; if not, the estimated bytes-moved penalty for the required transpose is added to the path cost. This ensures the optimizer favors paths that align naturally with the memory layout, even at the expense of slightly higher FLOP counts.

The optimizer also tracks conjugation metadata: when a Hermitian conjugate is required, the `is_conjugated` flag on the `MatRef` view is set rather than scheduling an explicit O(N) conjugation pass. The bandwidth cost for conjugation is therefore zero.

### 6.3 Contraction Executor

Two execution strategies, selected by backend capabilities:

**Strategy A — Strided Tensor Contraction:** tblis-style arbitrary-stride micro-kernels bypass reshape entirely. Zero memory-bandwidth cost.

**Strategy B — Pre-Allocated Transpose Arenas:** Standard GEMM (faer) requires transposition for non-contiguous contractions. Cache-aligned buffers from SweepArena; cache-oblivious block-transpose (8×8 or 16×16 tiles) maximizes cache-line utilization.

```rust
pub struct ContractionExecutor<T: Scalar, B: LinAlgBackend<T>> {
    backend: B,
    arena: SweepArena,
    threading: ThreadingRegime,
    _phantom: PhantomData<T>,
}
```

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

### 7.2 The `hamiltonian!{}` Macro: Scope & Limitations

The `hamiltonian!{}` proc-macro performs *exclusively* compile-time syntax-tree transformation. It parses a lattice/Hamiltonian DSL and emits Rust code that constructs an `OpSum` at runtime. No numerical computation — no SVD, no FSA minimization, no matrix construction — occurs at compile time.

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
    ComplexU1(DMFTLoop<Complex64, U1, DefaultDevice>),
    RealZ2(DMFTLoop<f64, Z2, DefaultDevice>),
}

#[pyclass(name = "DMFTLoop")]
pub struct PyDmftLoop { inner: DmftLoopVariant }
```

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

        // Spawn the signal-monitor thread.
        // recv_timeout serves as a combined sleep + shutdown check:
        //   - Timeout(100ms): poll for Ctrl+C
        //   - Ok(()): solver finished, exit cleanly
        //   - Disconnected: done_tx dropped (e.g., panic), exit cleanly
        let monitor_cancel = cancel_flag.clone();
        let monitor_handle = std::thread::spawn(move || {
            loop {
                match done_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(()) => break,  // Solver finished normally.
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,  // done_tx dropped (panic).
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        // 100ms elapsed — check for Ctrl+C.
                        let interrupted = Python::with_gil(|py| py.check_signals().is_err());
                        if interrupted {
                            monitor_cancel.store(true, Ordering::Release);
                            break;
                        }
                    }
                }
            }
        });

        // Release the GIL, run the solver, AND shut down the monitor thread
        // — all while the GIL is released. The monitor thread can freely
        // call Python::with_gil until we signal it to stop.
        let result = py.allow_threads(|| {
            let r = match &mut self.inner {
                DmftLoopVariant::RealU1(solver) => {
                    solver.solve_with_cancel_flag(&cancel_flag)
                }
                // ...
            };

            // Signal the monitor thread to exit and wait for it to finish.
            // CRITICAL: This happens while GIL is still released.
            // If we did this after allow_threads, the main thread would
            // hold the GIL, and the monitor thread might be blocked on
            // Python::with_gil — deadlock.
            let _ = done_tx.send(());
            let _ = monitor_handle.join();

            r
        });

        // GIL re-acquired here. Monitor thread is guaranteed dead.
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

                // AtomicBool check: zero-cost on x86 (MOV from cache line).
                // Rayon workers never touch the GIL.
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
        // Moves Vec<f64> into Python's memory manager; zero element-wise copy.
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

/// Bond-centered form: the singular value matrix between sites `left` and
/// `left+1` is exposed as a standalone object. Required for TDVP
/// projector-splitting, which evolves this bond matrix backward in time.
pub struct BondCentered { pub left: usize }

pub struct MPS<T: Scalar, Q: BitPackable, Gauge> {
    tensors: Vec<BlockSparseTensor<T, Q>>,
    /// Singular value / gauge bond matrices (populated in BondCentered form).
    bonds: Option<Vec<DenseTensor<T>>>,
    _gauge: PhantomData<Gauge>,
}

impl<T: Scalar, Q: BitPackable> MPS<T, Q, MixedCanonical> {
    /// Two-site DMRG update: only valid on MixedCanonical MPS.
    pub fn dmrg_step(
        &mut self, mpo: &MPO<T, Q>, env: &mut Environments<T, Q>,
        solver: &dyn IterativeEigensolver<T>,
    ) -> T::Real { /* returns energy */ }

    /// Expose the bond matrix between center and center+1,
    /// transitioning to BondCentered form for TDVP backward evolution.
    pub fn expose_bond(self) -> MPS<T, Q, BondCentered> { /* ... */ }
}

impl<T: Scalar, Q: BitPackable> MPS<T, Q, BondCentered> {
    /// Evolve the bond matrix backward in time: exp(+i H_bond Δt/2).
    /// This is the zero-site projector step in TDVP splitting.
    ///
    /// NUMERICAL STABILIZATION: Uses Tikhonov-regularized pseudo-inverse
    /// (LinAlgBackend::regularized_svd_inverse) for gauge restoration.
    /// Near-zero singular values produce safe 0 instead of ∞.
    pub fn evolve_bond_backward(
        &mut self,
        h_bond: &dyn Fn(&[T], &mut [T]),
        dt: T::Real,
        config: &TdvpStabilizationConfig,
    ) { /* Krylov matrix-exponential on the bond matrix */ }

    /// Absorb bond matrix back into site tensor, returning to MixedCanonical.
    pub fn absorb_bond(self) -> MPS<T, Q, MixedCanonical> { /* ... */ }
}
```

The `BondCentered` state is essential for TDVP's projector-splitting scheme. A single 1-site TDVP step requires: (1) evolving the center site forward via exp(-i H_eff Δt/2), (2) exposing the bond matrix (transition to `BondCentered`), (3) evolving the bond matrix *backward* via exp(+i H_bond Δt/2), (4) absorbing the bond and shifting the gauge center (transition back to `MixedCanonical`). Without `BondCentered`, the backward bond evolution would require breaking the typestate abstraction.

#### 8.1.1 TDVP Numerical Stabilization

The backward evolution step requires inverting the bond matrix (containing the singular values of the bipartition). When the state has low entanglement across a bond, many singular values approach machine zero, causing 1/s_i → ∞ and filling tensors with NaN. Two complementary stabilization strategies are employed:

**Strategy 1 — Tikhonov Regularization (numerical floor):**

Every gauge-restoration inversion routes through `LinAlgBackend::regularized_svd_inverse` (§5.1). Instead of computing s_i⁻¹, the regularized form s_i / (s_i² + δ²) is used, where δ is a configurable noise floor (typically 1e-8 to 1e-12). When s_i ≫ δ, this perfectly approximates the true inverse; when s_i → 0, the regularized inverse safely drops to zero.

**Strategy 2 — Site-Tensor Subspace Expansion (algorithmic growth):**

Tikhonov regularization prevents NaN but restricts the MPS to its existing entanglement subspace. Real-time evolution (especially post-quench in DMFT) requires entanglement *growth*, which pure 1-site TDVP cannot achieve since it operates within the fixed-D tangent space.

The subspace expansion operates on the *site tensors* A_L and A_R, not on the bond matrix C. The bond matrix C lives in a D × D space, while the Hamiltonian action generates vectors in the larger D·d × D·d two-site space. Direct mixing of the two-site residual into the bond matrix is dimensionally inconsistent. Instead, the expansion enlarges the site-tensor basis:

1. At the `BondCentered` step, compute the Hamiltonian residual: |R⟩ = H_eff · |ψ_center⟩.
2. Project out the existing tangent-space component using **matrix-free sequential projection** (never form the explicit projector P = I − A_L · A_L†, which would cost O(d²D³)):
   - Compute the overlap: O = A_L† · |R⟩. This is a matrix-vector product costing O(dD²).
   - Subtract the projection: |R_null⟩ = |R⟩ − A_L · O. Another O(dD²) matrix-vector product.
   - Total projection cost: O(dD²), safely within the O(D³) DMRG scaling bound.
3. SVD of |R_null⟩ → retain the top D_expand left singular vectors with the largest singular values. These vectors span directions orthogonal to the current tangent space.
4. Pad the site tensor A_L by appending these D_expand new basis vectors as additional columns, increasing its bond dimension from D to D + D_expand.
5. Pad the bond matrix C with zeros along the corresponding new rows/columns. The expanded C is now (D + D_expand) × (D + D_expand).
6. SVD the expanded bond matrix. The injected null-space vectors now provide physically relevant non-zero singular values that push the smallest eigenvalues above the noise floor naturally.
7. Truncate using **soft D_max** policy (see below).

**Scaling note:** Explicitly constructing the null-space projector P_null = (I − A_L · A_L†) would form a dense (dD × dD) matrix at O(d²D³) cost — an unnecessary factor of d·D more expensive than the matrix-free approach. For D = 1000 and d = 4, this is a ~4000× performance difference inside the innermost TDVP loop. The matrix-free two-step projection (overlap → subtract) is mandatory.

**Bond-dimension oscillation pathology:** If the injected null-space vectors have singular values very close to the truncation threshold, a hard truncation at D_max discards the exact entanglement that was just injected. On the next time step, the expansion re-injects the same vectors, which are again truncated — producing a discontinuous, oscillating bond dimension that corrupts the time evolution.

The soft D_max policy prevents this by allowing the bond dimension to temporarily exceed D_max by a configurable factor (typically 1.1×), then smoothly decaying the excess over subsequent time steps rather than hard-cutting immediately:

- After expansion + SVD in step 7, truncate to `D_soft = floor(D_max × soft_dmax_factor)` instead of D_max.
- On each subsequent time step *without* expansion, the effective truncation target decays exponentially: `D_target(t) = D_max + (D_soft − D_max) × exp(−t / dmax_decay_steps)`.
- This gives newly injected entanglement several time steps to either grow into physically significant singular values (and survive) or decay naturally below the noise floor (and be discarded organically).

**Per-bond state tracking:** The decay formula requires knowing how many time steps have elapsed since the last expansion *at each bond independently*. This is mutable algorithmic state that does not belong in the immutable `TdvpStabilizationConfig` or in the MPS typestate (which should not carry algorithm-specific metadata). Instead, the `TdvpDriver` struct (see below) carries a `Vec<Option<usize>>` of per-bond expansion ages, updated after each time step via `tick_expansion_age()`.

This procedure ensures dimensional consistency at every step: the site tensor expansion changes the shape of A_L from (d, D_left, D) to (d, D_left, D + D_expand), and the bond matrix is padded to match. The contraction graph in `tk-contract` handles the dimension change because `TensorShape` is dynamically determined, not statically fixed.

```rust
pub struct TdvpStabilizationConfig {
    /// Tikhonov regularization parameter δ for gauge-shift inversions.
    /// Prevents NaN from near-zero singular values.
    pub tikhonov_delta: f64,       // default: 1e-10
    /// Number of null-space vectors to inject per expansion step.
    /// Controls how aggressively the bond dimension is allowed to grow.
    pub expansion_vectors: usize,  // default: 4
    /// Mixing parameter α: weight of the null-space residual relative
    /// to the existing site-tensor basis during SVD re-truncation.
    pub expansion_alpha: f64,      // default: 1e-4
    /// If true, dynamically switch between 1-site and 2-site TDVP
    /// based on entanglement growth rate.
    pub adaptive_expansion: bool,  // default: true
    /// Soft D_max factor: after subspace expansion, truncate to
    /// floor(D_max × soft_dmax_factor) instead of hard D_max.
    /// Prevents bond-dimension oscillation at the truncation threshold.
    pub soft_dmax_factor: f64,     // default: 1.1 (allow 10% overshoot)
    /// Exponential decay rate (in time steps) for the soft D_max overshoot.
    /// After expansion, the effective D_target decays back to D_max
    /// over this many time steps. Larger values = gentler truncation.
    pub dmax_decay_steps: f64,     // default: 5.0
}

/// Expand the bond between sites `left` and `left+1` by injecting
/// null-space vectors from the Hamiltonian residual.
/// Mutates A_L in place, returns the zero-padded bond matrix.
///
/// CRITICAL: Uses matrix-free sequential projection to remain O(dD²).
/// Never constructs the explicit projector (I - A_L · A_L†).
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
    // 3. SVD of |R_null⟩ → take top `expansion_vectors` left singular vectors
    // 4. Pad A_L with new vectors (scaled by expansion_alpha)
    // 5. Return zero-padded bond matrix with dimensions (D + D_expand) × (D + D_expand)
    todo!()
}

/// Stateful TDVP time-evolution driver.
/// TdvpStabilizationConfig is immutable configuration;
/// TdvpDriver carries the mutable per-bond state needed for
/// soft D_max decay tracking across time steps.
pub struct TdvpDriver<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> {
    pub config: TdvpStabilizationConfig,
    pub engine: DMRGEngine<T, Q, B>,
    /// Per-bond expansion age: `expansion_age[bond]` is `Some(n)` if
    /// the bond was last expanded `n` time steps ago, `None` if never
    /// expanded or fully decayed back to D_max.
    /// Length = n_sites - 1 (one entry per bond).
    expansion_age: Vec<Option<usize>>,
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
            Some(age) => {
                let d_soft = (hard_dmax as f64 * self.config.soft_dmax_factor).floor() as usize;
                let overshoot = (d_soft - hard_dmax) as f64;
                let decay = (-age as f64 / self.config.dmax_decay_steps).exp();
                hard_dmax + (overshoot * decay).round() as usize
            }
        }
    }

    /// Called after each time step to age all expansion counters.
    /// Bonds whose effective D_target has decayed to within 1 of D_max
    /// are reset to None (fully decayed).
    fn tick_expansion_age(&mut self, hard_dmax: usize) {
        for (bond, age) in self.expansion_age.iter_mut().enumerate() {
            if let Some(a) = age {
                *a += 1;
                // Check if decay is negligible
                if self.effective_dmax(bond, hard_dmax) <= hard_dmax {
                    *age = None;
                }
            }
        }
    }

    /// Mark a bond as freshly expanded (resets its decay counter to 0).
    fn mark_expanded(&mut self, bond: usize) {
        self.expansion_age[bond] = Some(0);
    }
}
```

### 8.2 Iterative Eigensolver Trait

The eigensolver implementations are written **in-house within `tk-dmrg`**, not delegated to external crates. Off-the-shelf Rust eigensolvers (e.g., the `eigenvalues` crate) are designed around standard dense matrix types and do not support the zero-allocation `&mut [T]` matvec workspace closures or tight integration with `SweepArena` that DMRG demands. Forcing block-sparse contraction through a dense-matrix API would destroy performance.

The closure uses an **in-place signature** that writes into a pre-allocated output buffer, avoiding heap allocation on every Krylov iteration. The API additionally supports **thick restarts**: when the maximum subspace dimension is exhausted before convergence (common near quantum critical points), the solver collapses to the best *n* vectors and rebuilds, retaining momentum without returning control to the DMRG engine:

```rust
/// Restart hint passed to the eigensolver.
pub enum InitialSubspace<'a, T: Scalar> {
    /// No prior information; start from a random vector.
    None,
    /// Single initial guess vector (standard DMRG warm-start).
    SingleVector(&'a [T]),
    /// Retained subspace from a previous restart (Block-Davidson).
    SubspaceBasis { vectors: &'a [&'a [T]], num_vectors: usize },
}

pub trait IterativeEigensolver<T: Scalar>: Send + Sync {
    /// Find the lowest eigenvalue and eigenvector.
    /// The solver owns all Krylov workspace internally and performs
    /// thick restarts as needed until convergence or max_iter.
    fn lowest_eigenpair(
        &self,
        matvec: &dyn Fn(&[T], &mut [T]),  // in-place: reads x, writes y = H_eff * x
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
    pub max_krylov_dim: usize,     // trigger restart when exceeded
}

pub struct DavidsonSolver {
    pub max_iter: usize, pub tol: f64,
    pub max_subspace: usize,
    pub restart_vectors: usize,    // retain this many vectors on restart
}

/// Block-Davidson: converts memory-bound dgemv into compute-bound dgemm.
pub struct BlockDavidsonSolver {
    pub max_iter: usize, pub tol: f64,
    pub block_size: usize,
    pub max_subspace: usize,
    pub restart_vectors: usize,    // thick restart: retain best n vectors
}
```

Inside the matvec closure, all intermediate contraction temporaries (T1, T2, T3) are pre-allocated from the SweepArena before the Krylov loop, reducing the closure to a pure sequence of GEMMs into pre-allocated workspace.

### 8.3 DMRG Sweep Engine

```rust
pub struct DMRGEngine<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> {
    pub mps: MPS<T, Q, MixedCanonical>,
    pub mpo: MPO<T, Q>,
    pub environments: Environments<T, Q>,
    pub backend: B,
    pub config: DMRGConfig,
}

pub struct DMRGConfig {
    pub max_bond_dim: usize,
    pub svd_cutoff: f64,
    pub max_sweeps: usize,
    pub energy_tol: f64,
    pub eigensolver: Box<dyn IterativeEigensolver<f64>>,
}

impl<T, Q, B> DMRGEngine<T, Q, B>
where T: Scalar, Q: BitPackable, B: LinAlgBackend<T>
{
    /// Run DMRG sweeps with an AtomicBool cancellation flag.
    /// Checked at the end of each sweep step (effectively free).
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

#### 8.4.1 TDVP as Primary Time-Evolution Engine

TDVP is designated as the primary engine. TEBD's Suzuki-Trotter decomposition violates unitarity over long time scales, causing norm drift that corrupts spectral functions. TDVP projects the time-dependent Schrödinger equation onto the MPS tangent-space manifold, rigorously preserving energy and unitarity. It reuses the same H_eff machinery and zero-allocation Krylov workspace from DMRG. TEBD is retained as a fallback.

TDVP integration is stabilized via the dual Tikhonov + site-tensor subspace expansion strategy described in §8.1.1.

#### 8.4.2 Linear Prediction with SVD Regularization & Exponential Windowing

Linear prediction is inherently ill-conditioned: noise is amplified by the pseudo-inverse of the Toeplitz-like prediction matrix. The architecture mandates SVD-regularized pseudo-inversion with aggressive singular-value cutoff below a configurable noise floor.

**Exponential windowing for metallic phases:** For metallic Green's functions G(t) that do not decay exponentially (e.g., Fermi-liquid behavior where G(t) ~ t⁻¹), the Toeplitz prediction matrix is poorly conditioned even with SVD regularization — the signal simply does not have the exponential structure that linear prediction assumes. An exponential window W(t) = exp(−η|t|) is applied to G(t) before prediction, artificially enforcing decay and regularizing the pseudo-inverse. The Fourier transform of exp(−η|t|) is a Lorentzian: 2η/(η²+ω²). The broadening parameter η therefore introduces a Lorentzian convolution of half-width η in the frequency domain, which must be deconvolved from the spectral function A(ω) after FFT. The Chebyshev cross-validation (§8.4.3) serves as the check that the deconvolution has not distorted the physics.

**Regularized Lorentzian deconvolution:** The naive deconvolution formula A_true(ω) = A_windowed(ω) · (η²+ω²)/(2η) amplifies noise quadratically at high frequencies: the factor (η²+ω²) grows as ω², so numerical noise in the tails of A_windowed is magnified by a factor of ~ω²/(2η). At ω = 100η, this is a ~5000× amplification — catastrophic for any realistic spectral function.

The deconvolution must be regularized. Two complementary strategies are applied:

1. **Hard frequency cutoff:** Beyond ω_max, the deconvolution factor is clamped to 1.0 (no correction). A_true(ω) is assumed to be negligible in this region. `deconv_omega_max` defaults to 10× the bandwidth of the impurity model.

2. **Tikhonov-style damping:** The deconvolution factor is modified from (η²+ω²)/(2η) to (η²+ω²)/(2η + δ_deconv · ω²), where δ_deconv is a small regularization parameter. At low ω, this is nearly identical to the unregularized form; at high ω, the δ_deconv · ω² term in the denominator tames the growth, bounding the amplification to 1/δ_deconv regardless of frequency.

```
A_true(ω) ≈ A_windowed(ω) · (η² + ω²) / (2η + δ_deconv · ω²)
                                              ^^^^^^^^^^^^^^^^^^^
                                              regularized denominator
```

Physically, the exponential window corresponds to adding an artificial scattering rate η to the retarded Green's function — a well-understood "artificial lifetime broadening" in DMFT. This is more physically interpretable than a Gaussian window, whose Fourier transform (another Gaussian) does not correspond to a standard spectral broadening mechanism.

#### 8.4.3 Chebyshev Cross-Validation (Mandatory)

Chebyshev expansion computes the spectral function directly in the frequency domain, bypassing both Trotter error and linear prediction instability. Built alongside TDVP in Phase 4 because cross-validating the two methods is the only rigorous way to verify correct physics.

```rust
pub struct DMFTLoop<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> {
    pub impurity: AndersonImpurityModel<T>,
    pub dmrg_config: DMRGConfig,
    pub time_evolution: TimeEvolutionConfig,
    pub chebyshev: ChebyshevConfig,
    pub linear_prediction: LinearPredictionConfig,
    pub self_consistency_tol: f64,
    pub max_dmft_iterations: usize,
    backend: B,
}

pub struct TimeEvolutionConfig {
    pub dt: f64,
    pub max_time: f64,
    /// Primary method: TDVP with stabilization.
    pub tdvp_stabilization: TdvpStabilizationConfig,
    /// Fallback: TEBD with configurable Suzuki-Trotter order.
    pub tebd_fallback: bool,
}

pub struct LinearPredictionConfig {
    pub prediction_order: usize,
    pub svd_noise_floor: f64,
    pub extrapolation_factor: f64,
    /// Exponential broadening parameter η for windowing G(t) before prediction.
    /// W(t) = exp(−η|t|) enforces decay for metallic phases where G(t)
    /// does not decay naturally. FT[exp(−η|t|)] = 2η/(η²+ω²) (Lorentzian).
    /// The induced Lorentzian broadening is deconvolved post-FFT.
    /// Set to 0.0 to disable (appropriate for gapped/insulating phases).
    pub broadening_eta: f64,  // default: 0.0 (disabled)
    /// Tikhonov damping for Lorentzian deconvolution.
    /// Regularized formula: (η²+ω²) / (2η + δ·ω²).
    /// Prevents quadratic noise amplification at high frequencies.
    /// Typical range: 1e-4 to 1e-2.
    pub deconv_tikhonov_delta: f64,  // default: 1e-3
    /// Hard frequency cutoff for deconvolution (in units of bandwidth).
    /// Beyond ω_max, the deconvolution factor is clamped to 1.0.
    /// Set to f64::INFINITY to disable hard cutoff (rely on Tikhonov only).
    pub deconv_omega_max: f64,  // default: 10.0 × bandwidth
}

impl<T, Q, B> DMFTLoop<T, Q, B>
where T: Scalar<Real = f64>, Q: BitPackable, B: LinAlgBackend<T>
{
    pub fn solve(&mut self) -> SpectralFunction {
        loop {
            let chain = self.impurity.discretize_bath();
            let gs = DMRGEngine::new(chain, ...).run();

            let g_t = self.tdvp_evolve(&gs);
            let g_t_ext = linear_predict_regularized(&g_t, &self.linear_prediction);
            let spectral_tdvp = fft(&g_t_ext);

            let spectral_cheb = self.chebyshev_expand(&gs);
            self.validate_consistency(&spectral_tdvp, &spectral_cheb);

            if self.converged(&spectral_tdvp) { return spectral_tdvp; }
            self.impurity.update_bath(&spectral_tdvp);
        }
    }

    /// Variant with AtomicBool cancellation for Python signal checking.
    pub fn solve_with_cancel_flag(
        &mut self,
        cancel: &AtomicBool,
    ) -> Result<SpectralFunction, SolverError> {
        // Same loop but passes cancel flag through to DMRGEngine
    }
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

The architecture employs a global atomic budget tracker that enforces a per-process pinned-memory ceiling with automatic fallback to pageable memory:

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global counter: bytes of page-locked memory currently allocated.
static PINNED_BYTES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
/// Per-process ceiling. Initialized at startup based on system RAM and MPI topology.
static PINNED_BYTES_LIMIT: AtomicUsize = AtomicUsize::new(0);

pub struct PinnedMemoryTracker;

impl PinnedMemoryTracker {
    /// Initialize the budget. For MPI Mode B, this should be:
    /// (Node RAM × safe_fraction) / local_ranks_per_node
    pub fn initialize_budget(max_bytes: usize) {
        PINNED_BYTES_LIMIT.store(max_bytes, Ordering::Release);
    }

    /// Attempt to reserve pinned memory. Returns true if successful.
    /// Wait-free: uses compare-and-swap loop, no locks.
    pub fn try_reserve(bytes: usize) -> bool {
        let limit = PINNED_BYTES_LIMIT.load(Ordering::Relaxed);
        let mut current = PINNED_BYTES_ALLOCATED.load(Ordering::Relaxed);
        loop {
            if current + bytes > limit {
                return false;
            }
            match PINNED_BYTES_ALLOCATED.compare_exchange_weak(
                current, current + bytes,
                Ordering::AcqRel, Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    /// Release previously reserved pinned memory.
    pub fn release(bytes: usize) {
        PINNED_BYTES_ALLOCATED.fetch_sub(bytes, Ordering::Release);
    }
}
```

The `SweepArena` constructor checks the budget and gracefully degrades:

```rust
/// Global counter for telemetry: number of times pageable fallback was triggered.
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
        // TELEMETRY: pageable fallback silently halves GPU bandwidth.
        // Emit a structured warning (not just a log line) so monitoring
        // dashboards and Jupyter notebooks surface the performance cliff.
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

    /// Returns the number of times pageable fallback has been triggered
    /// across the lifetime of the process. Exposed in DMRGEngine stats.
    pub fn pinned_fallback_count() -> usize {
        PINNED_FALLBACK_COUNT.load(Ordering::Relaxed)
    }
}

/// Drop returns the budget to the global tracker.
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

Because `AtomicUsize` only tracks memory within a single process, it is blind to co-resident MPI ranks. For MPI Mode B, the DMFT initialization sequence queries the system topology to divide the pinned-memory budget evenly:

```rust
#[cfg(all(feature = "backend-cuda", feature = "backend-mpi"))]
pub fn initialize_dmft_node_budget(comm: &MpiComm) {
    let total_ram = sys_info::mem_info().unwrap().total;
    // Split communicator by shared-memory domain to count co-resident ranks.
    let local_ranks = comm.split_by_shared_memory().size();
    // Conservative: 60% of total RAM, divided evenly.
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

cuBLAS and cuSOLVER operations are *asynchronous* — they return immediately while the GPU is still computing. If the `ContractionExecutor` walks the DAG dispatching to `DeviceCuda` without explicit synchronization, race conditions arise where a temporary is consumed before the GPU has finished producing it. The `ContractionNode` is therefore extended with device location and synchronization metadata when the CUDA backend is active:

```rust
#[cfg(feature = "backend-cuda")]
pub struct DeviceLocation {
    pub device: StorageDeviceKind,  // Host or Cuda { ordinal }
    pub sync_event: Option<cuda::Event>,  // signaled when output is ready
}

#[cfg(feature = "backend-cuda")]
impl<T: Scalar, B: LinAlgBackend<T>> ContractionExecutor<T, B> {
    /// Stream-aware DAG walk: issues cudaStreamWaitEvent between
    /// dependent nodes rather than global pipeline-stalling syncs.
    fn execute_cuda<T: Scalar>(
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

This fine-grained synchronization allows independent branches of the contraction tree to execute concurrently on separate CUDA streams, maximizing GPU occupancy without pipeline stalls.

#### 10.2.5 GPU Performance Considerations

DMRG is sequential at the sweep level. GPU helps within each step (large GEMM/SVD) but not across steps. Below D ≈ 500, kernel launch overhead may negate the GPU advantage. The architecture supports a hybrid strategy routing small operations to DeviceFaer on CPU and large operations to DeviceCuda, selected by a configurable dimension threshold. Batched cuBLAS amortizes launch overhead for block-sparse sector GEMMs.

#### 10.2.6 NUMA-Aware Pinned Allocation (Multi-GPU)

On multi-socket HPC nodes, `cudaMallocHost` allocates on whichever NUMA node the calling thread happens to be running on. If this NUMA node is remote from the PCIe root complex of the target GPU, pinned-memory DMA transfers traverse the inter-socket interconnect (QPI/UPI), throttling effective bandwidth by 30–50%.

For the single-GPU target (Phase 5), the OS typically schedules the solver thread on the same socket as the GPU, so this is not a concern. For multi-GPU extensions (Phase 5+), the `PinnedArena` must be extended to bind allocations to the correct NUMA node:

- Query the GPU's PCIe bus ID via `cudaDeviceGetPCIBusId`.
- Map the bus ID to a NUMA node via `/sys/bus/pci/devices/<bus_id>/numa_node`.
- Allocate pinned memory on that node via `libnuma`'s `numa_alloc_onnode` or `cudaHostAlloc` with `cudaHostAllocPortable` + explicit thread affinity.

This is deferred to the multi-GPU (NCCL) extension behind the `backend-cuda` feature flag.

### 10.3 MPI / Distributed-Memory Backend

**Mode A — Distributed Block-Sparse Tensors:** Sectors partitioned across ranks; cross-rank communication for boundary sectors. ContractionGraph DAG gains communication nodes and MPI-aware cost model. High risk, Phase 5+.

**Mode B — Embarrassingly Parallel DMFT:** Each rank runs an independent DMRGEngine. Synchronization only at DMFT convergence check via MPI_Allgather. No core changes needed. Recommended first target. Pinned-memory budget is automatically divided across co-resident ranks via `initialize_dmft_node_budget` (§10.2.2).

**Mode B load-imbalance risk:** For single-orbital Bethe lattice DMFT (Phase 4 target), impurity solver iteration counts are relatively uniform across momentum sectors, so the `MPI_Allgather` barrier causes minimal idle time. For multi-orbital or cluster DMFT with strongly heterogeneous bath parameters, solver convergence times can vary by 2–5× across ranks. In this regime, fast ranks idle at the barrier waiting for the slowest rank. Mitigation strategies for Phase 5+: (a) asynchronous convergence checks via non-blocking `MPI_Iallgather` with periodic polling; (b) dynamic work-stealing where idle ranks pull unconverged sectors from slow ranks; (c) adaptive mixing where faster-converging sectors use more aggressive Broyden steps to equalize iteration counts.

### 10.4 Extension Comparison

| Extension | Scope | Risk | Phase | Value |
|:----------|:------|:-----|:------|:------|
| **CUDA (single-node GPU)** | New DeviceCuda + StorageDevice generalization + PinnedArena + PinnedMemoryTracker | Medium | Phase 5 | High |
| **MPI Mode B (parallel DMFT)** | Application-layer only + pinned-memory topology query | Low | Phase 4–5 | High |
| **MPI Mode A (distributed tensors)** | ContractionExecutor + PathOptimizer | High | Phase 5+ | Medium |
| **Multi-GPU (NCCL)** | DeviceCuda + NCCL wrappers | High | Phase 5+ | Medium |

---

## 11. External Crate Dependencies

| Crate | Used By | Purpose |
|:------|:--------|:--------|
| **faer** | tk-linalg | Dense SVD (gesdd/gesvd), EVD, QR, LU; multithreaded cache-optimized; native lazy conjugation views |
| **oxiblas** | tk-linalg | Sparse ops (9 formats), SIMD BLAS, f128 |
| **bumpalo** | tk-core | Arena allocator for sweep temporaries |
| **smallvec** | tk-core | Stack-allocated small vectors for shapes/strides |
| **rayon** | tk-linalg, tk-contract | Data-parallel iterators (with LPT pre-sorting) |
| **num / num-complex** | tk-core | Complex<f64>, numeric traits |
| **omeco** | tk-contract | Greedy + TreeSA contraction path optimization |
| **cotengrust** | tk-contract | DP-based path optimization |
| **lie-groups** | tk-symmetry (optional) | SU(N) CG coefficients, Casimirs |
| **pyo3** | tk-python | Python bindings for TRIQS integration |
| **rust-numpy** | tk-python | Zero-copy NumPy array interop |
| **spenso** | tk-contract (reference) | Structural tensor graph inspiration |
| **cudarc** | tk-linalg (optional) | Safe Rust wrappers for CUDA driver, cuBLAS, cuSOLVER |
| **mpi** | tk-linalg, tk-dmft (optional) | Rust MPI bindings wrapping system MPI library |
| **sys-info** | tk-dmft (optional) | System RAM query for pinned-memory budget |

**Removed:** `eigenvalues` — iterative eigensolvers (Lanczos, Davidson, Block-Davidson) are implemented in-house within `tk-dmrg` to support zero-allocation matvec closures and tight `SweepArena` integration.

---

## 12. Testing & Benchmarking Strategy

### 12.1 Correctness Testing

Each sub-crate carries its own unit test suite with property-based testing (proptest):

- **tk-core:** Round-trip shape permutations, stride arithmetic, arena allocation/reset safety. **Arena ownership boundary:** verify `.into_owned()` produces independent heap copy; verify borrow checker rejects arena references held past `reset()`. **MatRef adjoint round-trip:** `mat.adjoint().adjoint()` recovers original strides and conjugation flag. **PinnedMemoryTracker:** budget enforcement, try_reserve failure, Drop release.
- **tk-symmetry:** Quantum number fusion associativity, flux conservation, bit-pack/unpack round-trip, block-dense equivalence.
- **tk-linalg:** SVD reconstruction error < machine epsilon, eigenvector orthogonality, GEMM reference comparison. **Conjugation-aware GEMM:** verify C = A†·B matches explicit conjugation + transpose + multiply for random complex matrices. **Regularized pseudo-inverse:** verify Tikhonov formula against analytically known cases. **LPT scheduling:** verify FLOP-descending order after sort; assert all sectors present in output. **SVD residual guard:** verify `debug_assert!` fires on synthetically corrupted singular values; verify it passes on well-conditioned matrices.
- **tk-contract:** Path FLOP estimates vs brute-force, result equivalence across optimizers. **Adjoint contraction:** verify that contraction with `.adjoint()` views produces identical results to explicit Hermitian conjugation.
- **tk-dmrg:** Heisenberg chain ground-state energy against reference snapshots (see §12.3), canonical form invariants. **Subspace expansion dimensional consistency:** verify A_L column count increases by D_expand; bond matrix shape matches; SVD re-truncation produces valid MPS. **Matrix-free projection equivalence:** verify that sequential projection (O = A_L†·|R⟩, |R_null⟩ = |R⟩ − A_L·O) produces identical results to explicit (I − A_L·A_L†)·|R⟩ on small test cases; verify orthogonality ⟨A_L|R_null⟩ ≈ 0.
- **tk-dmft:** Spectral function sum rules, bath discretization accuracy, self-consistency loop convergence against Bethe lattice benchmarks. **Exponential windowing:** verify deconvolved A(ω) matches unwindowed A(ω) for gapped phases (η should be a no-op); verify metallic-phase G(t) prediction does not diverge with η > 0. **Regularized deconvolution:** verify Tikhonov-damped deconvolution recovers known Lorentzian spectral functions; verify high-frequency noise amplification is bounded by 1/δ_deconv; verify hard ω_max cutoff produces smooth A(ω) tails. **Soft D_max:** verify bond dimension does not oscillate between expansion and truncation; verify smooth decay back to D_max over `dmax_decay_steps`.

#### 12.1.1 Gauge-Invariant Testing Macros

Cross-backend validation (e.g., DeviceFaer vs DeviceCuda) must never compare intermediate MPS tensors directly. The SVD has an inherent gauge freedom: singular vectors u and −u are equally valid, and different BLAS implementations will make different sign choices. Direct tensor equality checks will fail spuriously.

Instead, all cross-backend comparison tests use gauge-invariant assertions:

```rust
/// Assert two MPS represent the same physical state.
/// Compares |⟨ψ_a | ψ_b⟩| ≈ 1.0 rather than tensor elements.
macro_rules! assert_mps_equivalent {
    ($mps_a:expr, $mps_b:expr, $tol:expr) => {
        let overlap = mps_overlap($mps_a, $mps_b).abs();
        assert!((overlap - 1.0).abs() < $tol,
            "MPS overlap = {}, expected ≈ 1.0 (tol = {})", overlap, $tol);
    };
}

/// Assert SVD results are equivalent modulo gauge.
/// Compares singular values (unique, positive) and reconstruction error.
macro_rules! assert_svd_equivalent {
    ($svd_a:expr, $svd_b:expr, $tol:expr) => {
        // Singular values must match exactly (up to floating-point tolerance)
        assert_allclose!(&$svd_a.singular_values, &$svd_b.singular_values, $tol);
        // Reconstruction error must be below tolerance
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

**Local bare-metal (Criterion.rs):** Tracks wall-clock time for dense SVD at D = 100, 500, 1000, 2000; block-sparse GEMM with U(1) sectors (including LPT scheduling overhead); full DMRG sweep (N=100, D=200); contraction path optimization for 10/20/50-tensor networks. Run only on dedicated developer machines or self-hosted CI runners.

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
| SU(2) Wigner-Eckart complexity | High | Defer implementation behind feature flag; structural_contraction callback from day one; fusion-rule multiplicity documented (§4.4): `SectorGemmTask` generation must fan out to `Vec` for non-Abelian irreps |
| FLOP-only path optimization | Medium | Composite CostMetric (α·FLOPs + β·Bytes_Moved) propagates stride info through candidate trees; penalizes paths requiring explicit transposes; zero-cost conjugation via `is_conjugated` flag |
| Thread pool oversubscription | Medium | Hybrid ThreadingRegime: Fat-Sectors vs Fragmented-Sectors auto-selected per contraction |
| Rayon long-tail starvation (binomial sectors) | Medium | LPT scheduling: sort sectors by descending FLOPs before `par_iter`; structural restoration re-sort after collection |
| TDVP projector-splitting instability | High | Tikhonov-regularized pseudo-inverse (§5.1) + site-tensor subspace expansion (§8.1.1); TdvpStabilizationConfig with tunable δ and D_expand |
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
| Linear prediction ill-conditioning | High | SVD-regularized pseudo-inverse with noise floor cutoff; exponential windowing (η) for metallic G(t); Tikhonov-regularized Lorentzian deconvolution (§8.4.2); mandatory Chebyshev cross-validation |
| Linear prediction metallic-phase failure | High | Exponential window W(t) = exp(−η\|t\|) enforces decay; FT = Lorentzian 2η/(η²+ω²); η = 0 for gapped phases; deconvolution checked against Chebyshev |
| Contraction reshape memory bandwidth | Medium | Cache-oblivious block-transpose (8×8/16×16) from SweepArena; tblis-style strided contraction as preferred path |
| GPU DAG race conditions | Medium | Stream-aware ContractionExecutor with per-node cuda::Event synchronization; fine-grained cudaStreamWaitEvent between dependent nodes |
| Pinned memory exhaustion (multi-rank OOM) | High | PinnedMemoryTracker: global atomic budget with CAS loop; automatic fallback to pageable memory; MPI-aware topology query divides budget across co-resident ranks |
| FFI BLAS linker collision | Medium | compile_error! enforcing mutual exclusivity of backend-mkl and backend-openblas |
| SVD gauge freedom breaking cross-backend tests | Medium | Gauge-invariant testing macros; compare singular values and state overlaps, not tensor elements |
| CI benchmark flakiness from cloud noise | Medium | iai/divan instruction counting for CI gating; Criterion reserved for bare-metal |
| tk-dsl cyclic dependency on tk-linalg/tk-dmrg | High | Strict two-phase pipeline: tk-dsl produces OpSum only; MPO compilation in tk-dmrg |
| Arena lifetime friction with persistent MPS | High | Explicit ownership boundary (§3.3.2): arena intermediates are `TempTensor<'a>`; final SVD output calls `.into_owned()` before `SweepArena::reset()`; borrow checker enforces statically |
| String-typed operators causing runtime failures | Medium | Strongly-typed operator enums (`SpinOp`, `FermionOp`, `BosonOp`) for standard models; `CustomOp` escape hatch for non-standard operators; typos caught at compile time |
| Fermionic sign errors in non-1D geometries | High | Contraction engine is bosonic-only (§6.4); Jordan-Wigner strings encoded in MPO; native fermionic swap gates deferred to tree/PEPS extension behind `fermionic-swap` feature flag |
| Bond-dimension oscillation during subspace expansion | Medium | Soft D_max policy (§8.1.1): `soft_dmax_factor` allows 10% overshoot; exponential decay via `dmax_decay_steps`; per-bond `expansion_age` state in `TdvpDriver` |
| `LinAlgBackend` object-safety violation (E0038) | High | Trait parameterized at trait level (`LinAlgBackend<T>`) not per-method; `Box<dyn LinAlgBackend<f64>>` valid; `SparseLinAlgBackend<T, Q>` likewise |
| GIL deadlock between monitor thread and main thread | High | `done_tx.send()` + `monitor_handle.join()` execute *inside* `py.allow_threads` closure; GIL is released during shutdown; monitor thread can complete `with_gil` freely |
| SVD `gesdd` silent inaccuracy | Medium | `debug_assert!` residual check ‖A−UΣV†‖/‖A‖ < 1e-10 after every `gesdd` return; compiled out in release builds; catches corrupt small singular values before Tikhonov masks them |
| Lorentzian deconvolution noise amplification | High | Tikhonov-damped deconvolution (η²+ω²)/(2η+δ·ω²) bounds high-frequency amplification to 1/δ; hard cutoff ω_max clamps correction beyond bandwidth; `deconv_tikhonov_delta` and `deconv_omega_max` in `LinearPredictionConfig` |
| SU(2) fusion-rule one-to-many in task generation | Medium | Abelian `compute_fusion_rule` returns single sector; SU(2) j₁⊗j₂ yields multiple irreps. `SectorGemmTask` generation must fan out to `Vec<SectorGemmTask>` per input pair. Documented in §4.4; scoped to `su2-symmetry` feature flag |
| NUMA-blind pinned memory on multi-socket nodes | Medium | Single-GPU: OS schedules correctly. Multi-GPU: must bind pinned allocations to GPU's PCIe NUMA node via `numa_alloc_onnode`. Documented in §10.2.6; deferred to Phase 5+ |
| MPI Mode B load imbalance (heterogeneous solvers) | Medium | Single-orbital Bethe: minimal variance. Multi-orbital/cluster: 2–5× iteration spread. Async `MPI_Iallgather` or dynamic work-stealing for Phase 5+. Documented in §10.3 |
| Pinned-memory fallback silent performance cliff | Medium | Fallback emits structured telemetry event with `PINNED_FALLBACK_COUNT` counter; exposed in `DMRGEngine` stats; dashboards surface the 2× bandwidth regression |

---

## 14. Implementation Roadmap

| Phase | Target | Deliverables |
|:------|:-------|:-------------|
| **Phase 1** | Months 1–3 | tk-core (strict scope, MatRef with conjugation flag, arena ownership boundary with `.into_owned()`, PinnedMemoryTracker), tk-symmetry (U(1), Z₂, BitPackable, PackedSectorKey), tk-linalg (DeviceFaer with conjugation-aware GEMM, gesdd default, regularized pseudo-inverse, LPT-scheduled block_gemm). >90% test coverage. SVD + conjugation benchmarks. iai CI integration. `cargo-llvm-lines` compile-time monitoring. |
| **Phase 2** | Months 4–6 | tk-contract (DAG, greedy optimizer, conjugation flag propagation, bosonic-only contraction documented), tk-dsl (Index, typed operator enums, OpSum, hamiltonian!{} macro — AST generation only). OpSum→MPO compilation in tk-dmrg. Heisenberg ground state matches reference snapshots. |
| **Phase 3** | Months 7–9 | tk-dmrg (MPS typestates, two-site sweep, in-house Lanczos/Davidson/Block-Davidson, site-tensor subspace expansion). N=100 Heisenberg at D=500. Gauge-invariant test macros. ITensor/TeNPy comparison via snapshot fixtures. |
| **Phase 4** | Months 10–12 | tk-dmft (TDVP with Tikhonov + subspace expansion + soft D_max, linear prediction with exponential windowing, Chebyshev, DMFT loop). tk-python (GIL release, AtomicBool cancellation, rust-numpy). StorageDevice generalization. MPI Mode B with pinned-memory topology query. Bethe lattice validation (insulating + metallic phases). |
| **Phase 5** | Months 13+ | DeviceCuda + PinnedArena + CudaArena (budget-managed). SU(2) non-Abelian. TreeSA/DP optimizers. Fermionic swap gates for tree/PEPS (`fermionic-swap` feature flag). Multi-orbital DMFT. MPI Mode A if needed. Community release. PyPI wheel builds (pure-Rust backends only). |

---

## 15. Conclusion

This architecture provides a rock-solid foundation for a Rust tensor network library that is modular, safe, and performant. By decoupling tensor shape from storage, abstracting linear algebra backends behind traits, separating contraction path optimization from execution, and encoding physical gauge conditions in the type system, the design eliminates entire categories of bugs at compile time while preserving the computational intensity required for state-of-the-art quantum many-body simulations.

The v7.0 revision addresses six points from the sixth external review: (1) Lorentzian deconvolution is Tikhonov-regularized to prevent quadratic noise amplification at high frequencies, with a hard ω_max cutoff and configurable δ_deconv damping parameter; (2) SVD `gesdd` results are validated via a `debug_assert!` reconstruction residual check, catching silently inaccurate singular values in debug builds without production overhead; (3) SU(2) fusion-rule multiplicity (one-to-many sector mapping) is documented as a known Phase 5 refactoring requirement for the `SectorGemmTask` generation loop; (4) pinned-memory fallback emits structured telemetry with a global fallback counter exposed in `DMRGEngine` stats; (5) NUMA-aware pinned allocation is documented for multi-GPU nodes, deferred to Phase 5+; and (6) MPI Mode B load imbalance from heterogeneous impurity solver convergence is documented with mitigation strategies for multi-orbital DMFT.

Across seven revision cycles, the design has been hardened against: dimensional inconsistencies in TDVP subspace expansion, scaling violations in null-space projection, bond-dimension oscillation at truncation thresholds, Rust object-safety violations in backend traits, cyclic crate dependencies, GIL deadlocks (Rayon workers, monitor thread shutdown ordering) and thread leaks in Python bindings, pinned-memory exhaustion under multi-rank MPI, silent pinned-memory performance cliffs, NUMA-blind GPU allocations, Rayon long-tail starvation from binomial sector distributions, O(N) conjugation memory bandwidth waste, Fourier transform errors in spectral windowing, deconvolution noise amplification, SVD silent inaccuracy in divide-and-conquer algorithms, SU(2) fusion-rule multiplicity in task generation, linear prediction instability in metallic phases, MPI barrier load imbalance, arena lifetime conflicts with persistent state, monomorphization-driven compile-time explosion, SVD gauge freedom in cross-backend tests, and FFI linker collisions between vendor BLAS libraries.

The clear crate boundaries and feature-flag system ensure that the library can evolve incrementally—adding non-Abelian symmetries, GPU backends, and additional lattice geometries—without destabilizing the core infrastructure. The phased implementation roadmap prioritizes delivering a working DMRG solver as early as Phase 3, enabling real physics research to begin while the DMFT integration matures in parallel.
