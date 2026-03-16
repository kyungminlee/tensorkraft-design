# Software Architecture Design Document
# Rust-Based Tensor Network Library
**for Quantum Many-Body Physics — DMRG Algorithms & DMFT Impurity Solver Integration**

*Version 1.0 — March 2026 | Status: Architectural Specification (Pre-Implementation)*

---

## 1. Executive Summary

This document specifies the complete software architecture for a new, high-performance tensor network library written in Rust, provisionally named **tensorkraft**. The library targets quantum many-body physics with a primary focus on Density Matrix Renormalization Group (DMRG) algorithms, designed to serve as a real-frequency quantum impurity solver within Dynamical Mean-Field Theory (DMFT) self-consistency loops.

The architecture leverages Rust's zero-cost abstractions, ownership-based memory safety, and fearless concurrency to achieve performance competitive with established C++ frameworks (ITensor, Block2) and Julia libraries while providing compile-time safety guarantees that eliminate entire categories of runtime errors endemic to scientific computing.

The design is organized around five foundational pillars, each addressed by a dedicated sub-crate within a Cargo workspace:

- **tk-core:** Tensor data structures with strict shape/storage separation, arena allocation, and Copy-on-Write semantics.
- **tk-symmetry:** Native Abelian symmetry support (U(1), Z₂) via block-sparse formats, with a clear extension path for non-Abelian SU(2) via Wigner-Eckart factorization.
- **tk-linalg:** Trait-based linear algebra backend abstraction defaulting to faer (dense) and oxiblas (sparse), swappable via feature flags.
- **tk-contract:** DAG-based contraction engine with separated path optimization and execution phases.
- **tk-dsl:** Ergonomic API with intelligent indices and a macro-based DSL for automated MPO generation from lattice Hamiltonians.

Additionally, an integration crate **tk-dmrg** implements the full DMRG sweep algorithm, iterative eigensolvers, and time-evolution methods (TEBD/TDVP) required for the DMFT impurity solver workflow.

---

## 2. Workspace & Crate Architecture

The library is structured as a Cargo workspace containing focused, independently testable sub-crates.

### 2.1 Workspace Layout

```
tensorkraft/
├── Cargo.toml              # workspace root
├── crates/
│   ├── tk-core/             # Tensor shape, storage, memory mgmt
│   ├── tk-symmetry/         # Quantum numbers, block-sparse formats
│   ├── tk-linalg/           # Backend abstraction (faer, oxiblas)
│   ├── tk-contract/         # DAG engine, path optimization
│   ├── tk-dsl/              # Macros, OpSum, lattice builders
│   ├── tk-dmrg/             # DMRG sweeps, eigensolvers, MPS/MPO
│   ├── tk-dmft/             # DMFT loop, bath discretization, TEBD
│   └── tk-python/           # PyO3 bindings for DMFT integration
├── benches/                 # Criterion benchmarks
├── examples/                # Heisenberg chain, Hubbard DMFT, etc.
└── tests/                   # Integration tests
```

### 2.2 Crate Dependency Graph

| Crate | Responsibility | Depends On |
|:------|:---------------|:-----------|
| **tk-core** | Tensor shape/stride metadata, contiguous storage buffers, arena allocators, TensorCow (Copy-on-Write), element-type generics | *(none — leaf crate)* |
| **tk-symmetry** | QuantumNumber trait, U(1)/Z₂ implementations, SectorIndex, block-sparse storage variants (BSR), Wigner-Eckart scaffolding for SU(2) | tk-core |
| **tk-linalg** | LinAlgBackend trait, SVD/EVD/GEMM dispatch, DeviceFaer and DeviceOxiblas implementations, Rayon-parallel element-wise ops | tk-core, tk-symmetry |
| **tk-contract** | ContractionGraph DAG, PathOptimizer trait, greedy/TreeSA heuristics, ContractionExecutor with reshape-free GEMM | tk-core, tk-symmetry, tk-linalg |
| **tk-dsl** | Index struct with unique IDs and prime levels, `hamiltonian!{}` proc_macro, OpSum builder, Lattice trait, snake-path mappers | tk-core, tk-symmetry |
| **tk-dmrg** | MPS/MPO types with typestate canonicality, two-site sweep engine, Lanczos/Davidson eigensolvers, SVD truncation | tk-core through tk-dsl (all above) |
| **tk-dmft** | Anderson Impurity Model mapping, bath discretization (Lanczos tridiagonalization), TEBD/TDVP time evolution, linear prediction, Chebyshev expansion, DMFT self-consistency loop | tk-dmrg (and transitively all) |
| **tk-python** | PyO3/maturin bindings exposing solver API to Python DMFT codes (TRIQS, soliDMFT) | tk-dmft |

### 2.3 Feature Flags

| Feature Flag | Effect | Default |
|:-------------|:-------|:--------|
| **backend-faer** | Enables DeviceFaer for dense SVD/EVD/QR using the pure-Rust faer crate | Yes |
| **backend-oxiblas** | Enables DeviceOxiblas for sparse BSR/CSR operations and extended-precision (f128) math | Yes |
| **backend-mkl** | Links Intel MKL via FFI for vendor-optimized BLAS on Intel hardware | No |
| **backend-openblas** | Links OpenBLAS via FFI for broad HPC cluster compatibility | No |
| **su2-symmetry** | Activates non-Abelian SU(2) support with Clebsch-Gordan caching (depends on lie-groups crate) | No |
| **python-bindings** | Builds tk-python via PyO3/maturin for TRIQS integration | No |
| **parallel** | Enables Rayon-based data parallelism for element-wise tensor operations | Yes |
| **backend-cuda** | Enables DeviceCuda for GPU-accelerated GEMM (cuBLAS), SVD (cuSOLVER), and sparse ops (cuSPARSE); requires CUDA toolkit | No |
| **backend-mpi** | Enables MPI-distributed block-sparse tensors and parallel DMFT loop via the mpi crate; requires system MPI library | No |

---

## 3. Core Tensor Data Structure & Memory Management (tk-core)

The foundational design principle is a strict separation between tensor shape/stride metadata and contiguous memory storage. All tensor data resides as a single flat buffer, irrespective of dimensionality. Element offsets are computed via inner products of index coordinates and strides. This separation enables zero-copy view operations (transpose, permutation, slicing) that mutate only metadata.

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

### 3.2 Memory Management Strategy

DMRG sweeps perform thousands of contraction-SVD-truncation cycles per iteration. Naive heap allocation for each intermediate tensor causes severe fragmentation and allocator overhead. The architecture employs two complementary strategies:

#### 3.2.1 Arena Allocators

Temporary tensors within a single DMRG step are allocated from a pre-allocated memory arena (using the `bumpalo` crate). At the end of each sweep step, the arena's allocation pointer is reset to zero in O(1) time, entirely bypassing individual deallocation overhead. The arena is scoped to the sweep step via Rust's lifetime system, ensuring that no dangling references escape.

```rust
pub struct SweepArena {
    inner: bumpalo::Bump,
}

impl SweepArena {
    pub fn alloc_tensor<'a, T: Scalar>(
        &'a self, shape: TensorShape
    ) -> DenseTensor<T> { /* ... */ }

    /// O(1) reset: reclaims all arena memory.
    pub fn reset(&mut self) { self.inner.reset(); }
}
```

#### 3.2.2 Copy-on-Write (Cow) Semantics

Shape-manipulation operations (transpose, permute, reshape) return `TensorCow::Borrowed` views whenever the operation can be expressed as a pure stride permutation. Data is cloned into `TensorCow::Owned` only when a contiguous memory layout is strictly required (e.g., as input to a GEMM kernel). This pattern, modeled after the rstsr framework, ensures copies are generated only when mathematically necessary.

### 3.3 The Scalar Trait Hierarchy

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
}
```

---

## 4. Physical Symmetries & Block Sparsity (tk-symmetry)

In quantum systems with global symmetries, tensors become block-sparse: elements are non-zero only when the algebraic sum of incoming quantum numbers equals the outgoing quantum numbers (the "flux rule"). Exploiting this structure avoids storing and computing zeros, yielding order-of-magnitude speedups.

### 4.1 Quantum Number Trait

```rust
pub trait QuantumNumber:
    Clone + Eq + Hash + Ord + Debug + Send + Sync
{
    fn identity() -> Self;
    fn fuse(&self, other: &Self) -> Self;
    fn dual(&self) -> Self;
}

/// U(1) charge conservation (e.g., particle number).
pub struct U1(pub i32);
impl QuantumNumber for U1 {
    fn identity() -> Self { U1(0) }
    fn fuse(&self, other: &Self) -> Self { U1(self.0 + other.0) }
    fn dual(&self) -> Self { U1(-self.0) }
}

/// Z₂ parity conservation (e.g., fermion parity).
pub struct Z2(pub bool);
impl QuantumNumber for Z2 {
    fn identity() -> Self { Z2(false) }
    fn fuse(&self, other: &Self) -> Self { Z2(self.0 ^ other.0) }
    fn dual(&self) -> Self { self.clone() }
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
```

### 4.2 Block-Sparse Tensor Architecture

The block-sparse tensor stores data as a collection of dense sub-blocks using a Structure-of-Arrays (SoA) layout with sorted flat arrays and stack-allocated keys for maximum cache performance:

```rust
pub type SectorKey<Q> = SmallVec<[Q; 8]>;

/// INVARIANT: sector_keys is always sorted for O(log N) binary search.
/// Keys and blocks in parallel contiguous arrays (SoA) keep key scanning
/// in L1/L2 cache without pulling heavy DenseTensor metadata.
pub struct BlockSparseTensor<T: Scalar, Q: QuantumNumber> {
    indices: Vec<QIndex<Q>>,
    sector_keys: Vec<SectorKey<Q>>,
    sector_blocks: Vec<DenseTensor<T>>,
    flux: Q,
}

impl<T: Scalar, Q: QuantumNumber> BlockSparseTensor<T, Q> {
    pub fn get_block(&self, key: &SectorKey<Q>) -> Option<&DenseTensor<T>> {
        self.sector_keys
            .binary_search(key)
            .ok()
            .map(|idx| &self.sector_blocks[idx])
    }
}
```

This avoids two critical performance traps: (1) pointer-chasing through heap-scattered BTreeMap nodes, and (2) dynamic allocation of Vec<Q> keys on every access. The binary search over a contiguous array dominates a tree traversal in both cache-line accesses and constant-factor overhead.

### 4.3 Sparsity Format Summary

| Strategy | Format | Application | Benefit |
|:---------|:-------|:------------|:--------|
| **Dense** | Contiguous 1D | Non-symmetric models | Max BLAS throughput, SIMD |
| **Block-Sparse** | BSR / Sorted flat arrays | Abelian U(1), Z₂ | Skip zeros, preserve symmetry |
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
---

## 5. Abstract Linear Algebra Backend (tk-linalg)

DMRG performance is bottlenecked by three operations: GEMM for tensor contraction, SVD for basis truncation, and EVD for ground-state targeting. The tk-linalg crate abstracts these behind trait interfaces, allowing compile-time backend selection.

### 5.1 Backend Trait Definitions

```rust
pub trait LinAlgBackend: Send + Sync {
    fn gemm<T: Scalar>(alpha: T, a: &MatRef<T>, b: &MatRef<T>, beta: T, c: &mut MatMut<T>);
    fn svd_truncated<T: Scalar>(mat: &MatRef<T>, max_rank: usize, cutoff: T::Real)
        -> (DenseTensor<T>, Vec<T::Real>, DenseTensor<T>);
    fn eigh_lowest<T: Scalar>(mat: &MatRef<T>, k: usize) -> (Vec<T::Real>, DenseTensor<T>);
    fn qr<T: Scalar>(mat: &MatRef<T>) -> (DenseTensor<T>, DenseTensor<T>);
}

pub trait SparseLinAlgBackend: LinAlgBackend {
    fn spmv<T: Scalar, Q: QuantumNumber>(a: &BlockSparseTensor<T, Q>, x: &[T], y: &mut [T]);
    fn block_gemm<T: Scalar, Q: QuantumNumber>(
        a: &BlockSparseTensor<T, Q>, b: &BlockSparseTensor<T, Q>,
    ) -> BlockSparseTensor<T, Q>;
}
```

### 5.2 Backend Implementations

| Backend Struct | Feature Flag | Characteristics |
|:---------------|:-------------|:----------------|
| **DeviceFaer** | backend-faer (default) | Pure Rust; state-of-the-art multithreaded SVD; ideal for high bond-dimension DMRG |
| **DeviceOxiblas** | backend-oxiblas (default) | Pure Rust; explicit SIMD (AVX-512, NEON); 9 sparse formats; f128 extended precision |
| **DeviceMKL** | backend-mkl | FFI to Intel MKL; vendor-optimized for Xeon; requires proprietary license |
| **DeviceOpenBLAS** | backend-openblas | FFI to OpenBLAS; broad HPC cluster support |

```rust
pub struct DeviceAPI<D: LinAlgBackend, S: SparseLinAlgBackend> { dense: D, sparse: S }

#[cfg(all(feature = "backend-faer", feature = "backend-oxiblas"))]
pub type DefaultDevice = DeviceAPI<DeviceFaer, DeviceOxiblas>;
```

### 5.3 Hybrid Parallelism & Thread Pool Management

Mixing Rayon's work-stealing scheduler with multithreaded BLAS backends creates thread oversubscription. The architecture defines two dynamically-selected parallelism regimes:

**Regime 1 — Fat Sectors (Dense-Dominated):** Few massive symmetry sectors (D > 1000). Rayon disabled; BLAS backend uses full machine thread pool per GEMM/SVD.

**Regime 2 — Fragmented Sectors (Sparsity-Dominated):** Many small sectors. BLAS threading forced to 1; Rayon `par_iter` distributes independent sector GEMMs across all cores.

```rust
pub enum ThreadingRegime {
    FatSectors { blas_threads: usize },
    FragmentedSectors { rayon_threads: usize },
}

impl ThreadingRegime {
    pub fn select<T: Scalar, Q: QuantumNumber>(
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
    estimated_memory: usize,
}
```

### 6.2 Path Optimizer Trait

```rust
pub trait PathOptimizer: Send + Sync {
    fn optimize(&self, inputs: &[&TensorShape], index_map: &IndexMap) -> ContractionGraph;
}

pub struct GreedyOptimizer { pub cost_fn: CostMetric }          // O(n³)
pub struct TreeSAOptimizer { pub max_iterations: usize, ... }    // Simulated annealing
pub struct DPOptimizer { pub max_width: usize }                  // Dynamic programming
```

### 6.3 Contraction Executor

Two execution strategies, selected by backend capabilities:

**Strategy A — Strided Tensor Contraction:** tblis-style arbitrary-stride micro-kernels bypass reshape entirely. Zero memory-bandwidth cost.

**Strategy B — Pre-Allocated Transpose Arenas:** Standard GEMM (faer) requires transposition for non-contiguous contractions. Cache-aligned buffers from SweepArena; cache-oblivious block-transpose (8×8 or 16×16 tiles) maximizes cache-line utilization.

```rust
pub struct ContractionExecutor<B: LinAlgBackend> {
    backend: B,
    arena: SweepArena,
    threading: ThreadingRegime,
}
```

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

### 7.2 Automated MPO Generation: The `hamiltonian!{}` Macro

```rust
let H = hamiltonian! {
    lattice: Chain(N = 100, d = 2);
    sum i in 0..N-1 {
        J  * (Sp(i) * Sm(i+1) + Sm(i) * Sp(i+1))
      + Jz * Sz(i) * Sz(i+1)
    }
    sum i in 0..N { h * Sz(i) }
};

let H_aim = hamiltonian! {
    lattice: Star(n_bath = 6, d = 4);
    U * Nup(0) * Ndn(0)
    sum k in 1..=n_bath {
        V[k] * (Cdag_up(0) * C_up(k) + h.c.)
      + V[k] * (Cdag_dn(0) * C_dn(k) + h.c.)
      + eps[k] * (Nup(k) + Ndn(k))
    }
};
```

### 7.3 OpSum Builder Pattern

```rust
let mut opsum = OpSum::new();
for i in 0..n_sites - 1 {
    opsum += J * op("S+", i) * op("S-", i+1);
    opsum += J * op("S-", i) * op("S+", i+1);
    opsum += Jz * op("Sz", i) * op("Sz", i+1);
}
let mpo: MPO<f64, U1> = opsum.build(&site_indices)?;
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

### 7.5 Python Bindings: Type-Erased Dispatch Pattern

PyO3's `#[pyclass]` cannot be applied to generic structs. The tk-python crate bridges Rust's monomorphization to Python's dynamic dispatch via a type-erased enum:

```rust
enum DmftLoopVariant {
    RealU1(DMFTLoop<f64, U1, DefaultDevice>),
    ComplexU1(DMFTLoop<Complex64, U1, DefaultDevice>),
    RealZ2(DMFTLoop<f64, Z2, DefaultDevice>),
}

#[pyclass(name = "DMFTLoop")]
pub struct PyDmftLoop { inner: DmftLoopVariant }

#[pymethods]
impl PyDmftLoop {
    #[new]
    fn new(dtype: &str, symmetry: &str) -> PyResult<Self> {
        let inner = match (dtype, symmetry) {
            ("float64", "U1") => DmftLoopVariant::RealU1(DMFTLoop::new(/* ... */)),
            ("complex128", "U1") => DmftLoopVariant::ComplexU1(DMFTLoop::new(/* ... */)),
            _ => return Err(PyValueError::new_err("Unsupported combination")),
        };
        Ok(PyDmftLoop { inner })
    }

    fn solve(&mut self) -> PyResult<PySpectralFunction> {
        match &mut self.inner {
            DmftLoopVariant::RealU1(s) => Ok(PySpectralFunction::from(s.solve())),
            DmftLoopVariant::ComplexU1(s) => Ok(PySpectralFunction::from(s.solve())),
            // ...
        }
    }
}
```

---

## 8. DMRG Algorithm & DMFT Integration

### 8.1 MPS with Typestate Canonical Forms

```rust
pub struct LeftCanonical;
pub struct RightCanonical;
pub struct MixedCanonical { pub center: usize }

pub struct MPS<T: Scalar, Q: QuantumNumber, Gauge> {
    tensors: Vec<BlockSparseTensor<T, Q>>,
    _gauge: PhantomData<Gauge>,
}

impl<T: Scalar, Q: QuantumNumber> MPS<T, Q, MixedCanonical> {
    /// Two-site DMRG update: only valid on MixedCanonical MPS.
    pub fn dmrg_step(
        &mut self, mpo: &MPO<T, Q>, env: &mut Environments<T, Q>,
        solver: &dyn IterativeEigensolver<T>,
    ) -> T::Real { /* returns energy */ }
}
```

### 8.2 Iterative Eigensolver Trait

The closure uses an **in-place signature** that writes into a pre-allocated output buffer, avoiding heap allocation on every Krylov iteration:

```rust
pub trait IterativeEigensolver<T: Scalar>: Send + Sync {
    fn lowest_eigenpair(
        &self,
        matvec: &dyn Fn(&[T], &mut [T]),  // in-place: reads x, writes y = H_eff * x
        dim: usize,
        initial_guess: Option<&[T]>,
    ) -> (T::Real, Vec<T>);
}

pub struct LanczosSolver { pub max_iter: usize, pub tol: f64 }
pub struct DavidsonSolver { pub max_iter: usize, pub tol: f64, pub max_subspace: usize }

/// Block-Davidson: converts memory-bound dgemv into compute-bound dgemm.
pub struct BlockDavidsonSolver {
    pub max_iter: usize, pub tol: f64,
    pub block_size: usize, pub max_subspace: usize,
}
```

Inside the matvec closure, all intermediate contraction temporaries (T1, T2, T3) are pre-allocated from the SweepArena before the Krylov loop, reducing the closure to a pure sequence of GEMMs into pre-allocated workspace.

### 8.3 DMRG Sweep Engine

```rust
pub struct DMRGEngine<T: Scalar, Q: QuantumNumber, B: LinAlgBackend> {
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
```

### 8.4 DMFT Self-Consistency Loop

#### 8.4.1 TDVP as Primary Time-Evolution Engine

TDVP is designated as the primary engine. TEBD's Suzuki-Trotter decomposition violates unitarity over long time scales, causing norm drift that corrupts spectral functions. TDVP projects the time-dependent Schrödinger equation onto the MPS tangent-space manifold, rigorously preserving energy and unitarity. It reuses the same H_eff machinery and zero-allocation Krylov workspace from DMRG. TEBD is retained as a fallback.

#### 8.4.2 Linear Prediction with SVD Regularization

Linear prediction is inherently ill-conditioned: noise is amplified by the pseudo-inverse of the Toeplitz-like prediction matrix. The architecture mandates SVD-regularized pseudo-inversion with aggressive singular-value cutoff below a configurable noise floor.

#### 8.4.3 Chebyshev Cross-Validation (Mandatory)

Chebyshev expansion computes the spectral function directly in the frequency domain, bypassing both Trotter error and linear prediction instability. Built alongside TDVP in Phase 4 because cross-validating the two methods is the only rigorous way to verify correct physics.

```rust
pub struct DMFTLoop<T: Scalar, Q: QuantumNumber, B: LinAlgBackend> {
    pub impurity: AndersonImpurityModel<T>,
    pub dmrg_config: DMRGConfig,
    pub time_evolution: TimeEvolutionConfig,
    pub chebyshev: ChebyshevConfig,
    pub linear_prediction: LinearPredictionConfig,
    pub self_consistency_tol: f64,
    pub max_dmft_iterations: usize,
    backend: B,
}

pub struct LinearPredictionConfig {
    pub prediction_order: usize,
    pub svd_noise_floor: f64,
    pub extrapolation_factor: f64,
}

impl<T, Q, B> DMFTLoop<T, Q, B>
where T: Scalar<Real = f64>, Q: QuantumNumber, B: LinAlgBackend
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
}
```

---

## 9. Data Flow: DMRG Sweep Step

| Step | Operation | Crate(s) | Detail |
|:-----|:----------|:---------|:-------|
| 1 | **Build H_eff** | tk-contract | Contract L, R, and local MPO into effective Hamiltonian via PathOptimizer + ContractionExecutor. |
| 2 | **Solve eigenvalue problem** | tk-dmrg | Lanczos/Davidson/Block-Davidson via in-place matvec closure. Contraction buffers pre-allocated from SweepArena. |
| 3 | **SVD truncation** | tk-linalg (faer) | Two-site eigenvector decomposed via truncated SVD. Singular values below cutoff discarded. |
| 4 | **Update MPS tensors** | tk-dmrg | SVD factors absorbed; MixedCanonical gauge preserved via typestate system. |
| 5 | **Update environments** | tk-contract, tk-symmetry | Environment block updated by contracting new MPS tensor with old environment and MPO. |
| 6 | **Arena reset** | tk-core | All temporaries reclaimed in O(1) via SweepArena::reset(). |

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

```rust
#[cfg(feature = "backend-cuda")]
pub struct DeviceCuda {
    stream: cuda::Stream,
    cublas_handle: cublas::Handle,
    cusolver_handle: cusolver::Handle,
}

#[cfg(feature = "backend-cuda")]
impl LinAlgBackend for DeviceCuda { /* dispatch to cuBLAS, cuSOLVER */ }
```

**GPU performance note:** DMRG is sequential at the sweep level. GPU helps within each step (large GEMM/SVD) but not across steps. Below D ≈ 500, kernel launch overhead may negate the GPU advantage. Batched cuBLAS amortizes overhead for block-sparse sector GEMMs.

### 10.3 MPI / Distributed-Memory Backend

**Mode A — Distributed Block-Sparse Tensors:** Sectors partitioned across ranks; cross-rank communication for boundary sectors. ContractionGraph DAG gains communication nodes and MPI-aware cost model. High risk, Phase 5+.

**Mode B — Embarrassingly Parallel DMFT:** Each rank runs an independent DMRGEngine. Synchronization only at DMFT convergence check via MPI_Allgather. No core changes needed. Recommended first target.

### 10.4 Extension Comparison

| Extension | Scope | Risk | Phase | Value |
|:----------|:------|:-----|:------|:------|
| **CUDA (single-node GPU)** | New DeviceCuda + StorageDevice generalization | Medium | Phase 5 | High |
| **MPI Mode B (parallel DMFT)** | Application-layer only | Low | Phase 4–5 | High |
| **MPI Mode A (distributed tensors)** | ContractionExecutor + PathOptimizer | High | Phase 5+ | Medium |
| **Multi-GPU (NCCL)** | DeviceCuda + NCCL wrappers | High | Phase 5+ | Medium |

---

## 11. External Crate Dependencies

| Crate | Used By | Purpose |
|:------|:--------|:--------|
| **faer** | tk-linalg | Dense SVD, EVD, QR, LU; multithreaded cache-optimized |
| **oxiblas** | tk-linalg | Sparse ops (9 formats), SIMD BLAS, f128 |
| **bumpalo** | tk-core | Arena allocator for sweep temporaries |
| **smallvec** | tk-core | Stack-allocated small vectors for shapes/strides |
| **rayon** | tk-linalg, tk-contract | Data-parallel iterators |
| **num / num-complex** | tk-core | Complex<f64>, numeric traits |
| **omeco** | tk-contract | Greedy + TreeSA contraction path optimization |
| **cotengrust** | tk-contract | DP-based path optimization |
| **lie-groups** | tk-symmetry (optional) | SU(N) CG coefficients, Casimirs |
| **eigenvalues** | tk-dmrg | Davidson/Lanczos iterative eigensolvers |
| **pyo3** | tk-python | Python bindings for TRIQS integration |
| **spenso** | tk-contract (reference) | Structural tensor graph inspiration |
| **cudarc** | tk-linalg (optional) | Safe Rust wrappers for CUDA driver, cuBLAS, cuSOLVER |
| **mpi** | tk-linalg, tk-dmft (optional) | Rust MPI bindings wrapping system MPI library |

---

## 12. Testing & Benchmarking Strategy

### 12.1 Correctness Testing

Each sub-crate carries its own unit test suite with property-based testing (proptest):

- **tk-core:** Round-trip shape permutations, stride arithmetic, arena allocation/reset safety.
- **tk-symmetry:** Quantum number fusion associativity, flux conservation, block-dense equivalence.
- **tk-linalg:** SVD reconstruction error < machine epsilon, eigenvector orthogonality, GEMM reference comparison.
- **tk-contract:** Path FLOP estimates vs brute-force, result equivalence across optimizers.
- **tk-dmrg:** Heisenberg chain (N=10,20) ground-state energy against exact diagonalization, canonical form invariants.
- **tk-dmft:** Spectral function sum rules, bath discretization accuracy, self-consistency loop convergence.

### 12.2 Performance Benchmarks

Criterion.rs tracks: dense SVD at D = 100, 500, 1000, 2000; block-sparse GEMM with U(1) sectors; full DMRG sweep (N=100, D=200); contraction path optimization for 10/20/50-tensor networks.

---

## 13. Risk Analysis & Mitigation

| Risk | Severity | Mitigation |
|:-----|:---------|:-----------|
| faer API instability (pre-1.0) | Medium | Abstract behind LinAlgBackend trait; version-pin; fallback to OpenBLAS FFI |
| oxiblas sparse format coverage gaps | Medium | Validate BSR early; nalgebra-sparse as backup for CSR/CSC |
| SU(2) Wigner-Eckart complexity | High | Defer implementation behind feature flag, but design contraction engine with structural_contraction callback from day one |
| Thread pool oversubscription | Medium | Hybrid ThreadingRegime: Fat-Sectors vs Fragmented-Sectors auto-selected per contraction |
| Compile-time overhead from deep generics | Low | Strategic dyn Trait at boundaries; profile compile times per crate |
| PyO3 generic monomorphization | Medium | Type-erased dispatch enum in tk-python; macro-generate match arms |
| DMFT loop convergence sensitivity | High | Mixing schemes (linear, Broyden); validate against Bethe lattice half-filling benchmarks |
| Linear prediction ill-conditioning | High | SVD-regularized pseudo-inverse with noise floor cutoff; mandatory Chebyshev cross-validation |
| Contraction reshape memory bandwidth | Medium | Cache-oblivious block-transpose (8×8/16×16) from SweepArena; tblis-style strided contraction as preferred path |

---

## 14. Implementation Roadmap

| Phase | Target | Deliverables |
|:------|:-------|:-------------|
| **Phase 1** | Months 1–3 | tk-core, tk-symmetry (U(1), Z₂), tk-linalg (DeviceFaer). >90% test coverage. SVD benchmarks. |
| **Phase 2** | Months 4–6 | tk-contract (DAG, greedy optimizer), tk-dsl (Index, OpSum, hamiltonian!{} macro). Heisenberg ground state matches exact diag. |
| **Phase 3** | Months 7–9 | tk-dmrg (MPS typestates, two-site sweep, Lanczos/Davidson). N=100 Heisenberg at D=500. ITensor/TeNPy comparison. |
| **Phase 4** | Months 10–12 | tk-dmft (TDVP, linear prediction, Chebyshev, DMFT loop). tk-python. StorageDevice generalization. MPI Mode B. Bethe lattice validation. |
| **Phase 5** | Months 13+ | DeviceCuda + CudaArena. SU(2) non-Abelian. TreeSA/DP optimizers. Multi-orbital DMFT. MPI Mode A if needed. Community release. |

---

## 15. Conclusion

This architecture provides a rock-solid foundation for a Rust tensor network library that is modular, safe, and performant. By decoupling tensor shape from storage, abstracting linear algebra backends behind traits, separating contraction path optimization from execution, and encoding physical gauge conditions in the type system, the design eliminates entire categories of bugs at compile time while preserving the computational intensity required for state-of-the-art quantum many-body simulations.

The clear crate boundaries and feature-flag system ensure that the library can evolve incrementally—adding non-Abelian symmetries, GPU backends, and additional lattice geometries—without destabilizing the core infrastructure. The phased implementation roadmap prioritizes delivering a working DMRG solver as early as Phase 3, enabling real physics research to begin while the DMFT integration matures in parallel.
