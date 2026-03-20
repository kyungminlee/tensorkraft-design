

**SOFTWARE ARCHITECTURE**

**DESIGN DOCUMENT**

**Rust-Based Tensor Network Library**

for Quantum Many-Body Physics

*DMRG Algorithms & DMFT Impurity Solver Integration*

Version 1.0  —  March 2026

Status: Architectural Specification (Pre-Implementation)

# **Table of Contents**

# **1\. Executive Summary**

This document specifies the complete software architecture for a new, high-performance tensor network library written in Rust, provisionally named tensorkraft. The library targets quantum many-body physics with a primary focus on Density Matrix Renormalization Group (DMRG) algorithms, designed to serve as a real-frequency quantum impurity solver within Dynamical Mean-Field Theory (DMFT) self-consistency loops.

The architecture leverages Rust’s zero-cost abstractions, ownership-based memory safety, and fearless concurrency to achieve performance competitive with established C++ frameworks (ITensor, Block2) and Julia libraries while providing compile-time safety guarantees that eliminate entire categories of runtime errors endemic to scientific computing.

The design is organized around five foundational pillars, each addressed by a dedicated sub-crate within a Cargo workspace:

* **tk-core:** Tensor data structures with strict shape/storage separation, arena allocation, and Copy-on-Write semantics.

* **tk-symmetry:** Native Abelian symmetry support (U(1), Z₂) via block-sparse formats, with a clear extension path for non-Abelian SU(2) via Wigner-Eckart factorization.

* **tk-linalg:** Trait-based linear algebra backend abstraction defaulting to faer (dense) and oxiblas (sparse), swappable via feature flags.

* **tk-contract:** DAG-based contraction engine with separated path optimization and execution phases.

* **tk-dsl:** Ergonomic API with intelligent indices and a macro-based DSL for automated MPO generation from lattice Hamiltonians.

Additionally, an integration crate tk-dmrg implements the full DMRG sweep algorithm, iterative eigensolvers, and time-evolution methods (TEBD/TDVP) required for the DMFT impurity solver workflow.

# **2\. Workspace & Crate Architecture**

The library is structured as a Cargo workspace containing focused, independently testable sub-crates. This modular architecture enables fine-grained dependency management and allows downstream users to depend on only the components they need.

## **2.1 Workspace Layout**

tensorkraft/  
├── Cargo.toml              \# workspace root  
├── crates/  
│   ├── tk-core/             \# Tensor shape, storage, memory mgmt  
│   ├── tk-symmetry/         \# Quantum numbers, block-sparse formats  
│   ├── tk-linalg/           \# Backend abstraction (faer, oxiblas)  
│   ├── tk-contract/         \# DAG engine, path optimization  
│   ├── tk-dsl/              \# Macros, OpSum, lattice builders  
│   ├── tk-dmrg/             \# DMRG sweeps, eigensolvers, MPS/MPO  
│   ├── tk-dmft/             \# DMFT loop, bath discretization, TEBD  
│   └── tk-python/           \# PyO3 bindings for DMFT integration  
├── benches/                 \# Criterion benchmarks  
├── examples/                \# Heisenberg chain, Hubbard DMFT, etc.  
└── tests/                   \# Integration tests

## **2.2 Crate Dependency Graph**

The following table specifies each crate’s role and its direct upstream dependencies. All dependency arrows flow upward; no circular dependencies exist.

| Crate | Responsibility | Depends On |
| :---- | :---- | :---- |
| **tk-core** | Tensor shape/stride metadata, contiguous storage buffers, arena allocators, TensorCow (Copy-on-Write), element-type generics | *(none — leaf crate)* |
| **tk-symmetry** | QuantumNumber trait, U(1)/Z₂ implementations, SectorIndex, block-sparse storage variants (BSR), Wigner-Eckart scaffolding for SU(2) | tk-core |
| **tk-linalg** | LinAlgBackend trait, SVD/EVD/GEMM dispatch, DeviceFaer and DeviceOxiblas implementations, Rayon-parallel element-wise ops | tk-core, tk-symmetry |
| **tk-contract** | ContractionGraph DAG, PathOptimizer trait, greedy/TreeSA heuristics, ContractionExecutor with reshape-free GEMM | tk-core, tk-symmetry, tk-linalg |
| **tk-dsl** | Index struct with unique IDs and prime levels, hamiltonian\!{} proc\_macro, OpSum builder, Lattice trait, snake-path mappers | tk-core, tk-symmetry |
| **tk-dmrg** | MPS/MPO types with typestate canonicality, two-site sweep engine, Lanczos/Davidson eigensolvers, SVD truncation | tk-core through tk-dsl (all above) |
| **tk-dmft** | Anderson Impurity Model mapping, bath discretization (Lanczos tridiagonalization), TEBD/TDVP time evolution, linear prediction, Chebyshev expansion, DMFT self-consistency loop | tk-dmrg (and transitively all) |
| **tk-python** | PyO3/maturin bindings exposing solver API to Python DMFT codes (TRIQS, soliDMFT) | tk-dmft |

## **2.3 Feature Flags**

Compile-time backend selection and optional capabilities are managed through Cargo feature flags in the workspace root:

| Feature Flag | Effect | Default |
| :---- | :---- | :---- |
| **backend-faer** | Enables DeviceFaer for dense SVD/EVD/QR using the pure-Rust faer crate | Yes (default) |
| **backend-oxiblas** | Enables DeviceOxiblas for sparse BSR/CSR operations and extended-precision (f128) math | Yes (default) |
| **backend-mkl** | Links Intel MKL via FFI for vendor-optimized BLAS on Intel hardware | No |
| **backend-openblas** | Links OpenBLAS via FFI for broad HPC cluster compatibility | No |
| **su2-symmetry** | Activates non-Abelian SU(2) support with Clebsch-Gordan caching (depends on lie-groups crate) | No |
| **python-bindings** | Builds tk-python via PyO3/maturin for TRIQS integration | No |
| **parallel** | Enables Rayon-based data parallelism for element-wise tensor operations | Yes (default) |
| **backend-cuda** | Enables DeviceCuda for GPU-accelerated GEMM (cuBLAS), SVD (cuSOLVER), and sparse ops (cuSPARSE); requires CUDA toolkit | No |
| **backend-mpi** | Enables MPI-distributed block-sparse tensors and parallel DMFT loop via the mpi crate; requires system MPI library | No |

# **3\. Core Tensor Data Structure & Memory Management (tk-core)**

The foundational design principle is a strict separation between tensor shape/stride metadata and contiguous memory storage. All tensor data resides as a single flat buffer, irrespective of dimensionality. Element offsets are computed via inner products of index coordinates and strides. This separation enables zero-copy view operations (transpose, permutation, slicing) that mutate only metadata.

## **3.1 Core Type Definitions**

The following Rust types define the core tensor architecture:

/// Dimensional metadata: shapes and strides for zero-copy views.  
pub struct TensorShape {  
    dims: SmallVec\<\[usize; 6\]\>,    // typical rank ≤ 6  
    strides: SmallVec\<\[usize; 6\]\>,  // row-major by default  
}

/// Contiguous memory buffer, generic over element type.  
pub struct TensorStorage\<T: Scalar\> {  
    data: Vec\<T\>,  // or ArenaVec\<T\> when arena-allocated  
}

/// The primary dense tensor: shape metadata \+ owned/borrowed storage.  
pub struct DenseTensor\<T: Scalar\> {  
    shape: TensorShape,  
    storage: TensorCow\<T\>,  // Cow semantics: Borrowed view or Owned data  
}

/// Copy-on-Write storage wrapper.  
pub enum TensorCow\<'a, T: Scalar\> {  
    Borrowed(&'a TensorStorage\<T\>),   // zero-copy view  
    Owned(TensorStorage\<T\>),           // materialized copy  
}

The Scalar trait constrains T to types supporting the required arithmetic: f32, f64, Complex\<f32\>, Complex\<f64\>, and optionally f128 when the backend-oxiblas feature is active.

## **3.2 Memory Management Strategy**

DMRG sweeps perform thousands of contraction-SVD-truncation cycles per iteration. Naive heap allocation for each intermediate tensor causes severe fragmentation and allocator overhead. The architecture employs two complementary strategies:

### **3.2.1 Arena Allocators**

Temporary tensors within a single DMRG step are allocated from a pre-allocated memory arena (using the bumpalo crate). At the end of each sweep step, the arena’s allocation pointer is reset to zero in O(1) time, entirely bypassing individual deallocation overhead. The arena is scoped to the sweep step via Rust’s lifetime system, ensuring that no dangling references escape.

pub struct SweepArena {  
    inner: bumpalo::Bump,  
}

impl SweepArena {  
    /// Allocate a tensor buffer within this arena's lifetime.  
    pub fn alloc\_tensor\<'a, T: Scalar\>(  
        &'a self, shape: TensorShape  
    ) \-\> DenseTensor\<T\> { /\* ... \*/ }

    /// O(1) reset: reclaims all arena memory.  
    pub fn reset(\&mut self) { self.inner.reset(); }  
}

### **3.2.2 Copy-on-Write (Cow) Semantics**

Shape-manipulation operations (transpose, permute, reshape) return TensorCow::Borrowed views whenever the operation can be expressed as a pure stride permutation. Data is cloned into TensorCow::Owned only when a contiguous memory layout is strictly required (e.g., as input to a GEMM kernel). This pattern, modeled after the rstsr framework, ensures copies are generated only when mathematically necessary.

## **3.3 The Scalar Trait Hierarchy**

pub trait Scalar:  
    Copy \+ Clone \+ Send \+ Sync \+ num::Zero \+ num::One  
    \+ std::ops::Add\<Output \= Self\>  
    \+ std::ops::Mul\<Output \= Self\>  
{  
    type Real: Scalar;  // f64 for Complex\<f64\>, f64 for f64  
    fn conj(self) \-\> Self;  
    fn abs\_sq(self) \-\> Self::Real;  
    fn from\_real(r: Self::Real) \-\> Self;  
}

# **4\. Physical Symmetries & Block Sparsity (tk-symmetry)**

In quantum systems with global symmetries, tensors become block-sparse: elements are non-zero only when the algebraic sum of incoming quantum numbers equals the outgoing quantum numbers (the “flux rule”). Exploiting this structure avoids storing and computing zeros, yielding order-of-magnitude speedups.

## **4.1 Quantum Number Trait**

The abstract QuantumNumber trait unifies all symmetry types behind a single interface:

/// Abstract quantum number supporting Abelian fusion rules.  
pub trait QuantumNumber:  
    Clone \+ Eq \+ Hash \+ Ord \+ Debug \+ Send \+ Sync  
{  
    /// Identity element (vacuum sector).  
    fn identity() \-\> Self;

    /// Abelian fusion: combine two quantum numbers.  
    fn fuse(\&self, other: \&Self) \-\> Self;

    /// Dual (conjugate) representation.  
    fn dual(\&self) \-\> Self;  
}

// \---- Concrete implementations \----

/// U(1) charge conservation (e.g., particle number).  
\#\[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)\]  
pub struct U1(pub i32);

impl QuantumNumber for U1 {  
    fn identity() \-\> Self { U1(0) }  
    fn fuse(\&self, other: \&Self) \-\> Self { U1(self.0 \+ other.0) }  
    fn dual(\&self) \-\> Self { U1(-self.0) }  
}

/// Z₂ parity conservation (e.g., fermion parity).  
\#\[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)\]  
pub struct Z2(pub bool);

impl QuantumNumber for Z2 {  
    fn identity() \-\> Self { Z2(false) }  
    fn fuse(\&self, other: \&Self) \-\> Self { Z2(self.0 ^ other.0) }  
    fn dual(\&self) \-\> Self { self.clone() }  
}

/// Composite symmetry: U(1) ⊗ Z₂ (particle number \+ parity).  
\#\[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)\]  
pub struct U1Z2(pub U1, pub Z2);

impl QuantumNumber for U1Z2 {  
    fn identity() \-\> Self { U1Z2(U1::identity(), Z2::identity()) }  
    fn fuse(\&self, other: \&Self) \-\> Self {  
        U1Z2(self.0.fuse(\&other.0), self.1.fuse(\&other.1))  
    }  
    fn dual(\&self) \-\> Self { U1Z2(self.0.dual(), self.1.dual()) }  
}

## **4.2 Block-Sparse Tensor Architecture**

The block-sparse tensor stores data as a collection of dense sub-blocks, each tagged by the quantum numbers of its symmetry sector. The SectorIndex maps quantum-number tuples to contiguous storage offsets:

/// A symmetry-aware index partitioned into charge sectors.  
pub struct QIndex\<Q: QuantumNumber\> {  
    /// Ordered list of (quantum\_number, sector\_dimension) pairs.  
    sectors: Vec\<(Q, usize)\>,  
    /// Total dimension \= sum of sector dimensions.  
    total\_dim: usize,  
}

/// Block-sparse tensor: only stores non-zero symmetry sectors.  
pub struct BlockSparseTensor\<T: Scalar, Q: QuantumNumber\> {  
    /// Quantum-number-aware indices for each leg.  
    indices: Vec\<QIndex\<Q\>\>,  
    /// Map from sector key \-\> dense sub-block.  
    blocks: BTreeMap\<Vec\<Q\>, DenseTensor\<T\>\>,  
    /// Total flux (net quantum number) of this tensor.  
    flux: Q,  
}

## **4.3 Sparsity Format Summary**

| Strategy | Format | Application | Benefit |
| :---- | :---- | :---- | :---- |
| **Dense** | Contiguous 1D | Non-symmetric models | Max BLAS throughput, SIMD |
| **Block-Sparse** | BSR / BTreeMap | Abelian U(1), Z₂ | Skip zeros, preserve symmetry |
| **Element-Sparse** | CSR / CSC | Irregular geometry | Memory-efficient for extreme sparsity |
| **Diagonal** | DIA | Identity / local ops | Zero-overhead storage |

## **4.4 Non-Abelian Symmetry Roadmap (SU(2))**

For non-Abelian symmetries such as SU(2) spin rotation, the Wigner-Eckart theorem factorizes a tensor into two components: (1) a structural tensor of Clebsch-Gordan coefficients dictated purely by representation theory, and (2) a dynamic reduced matrix element tensor containing the variational physics. The architecture reserves this extension behind the su2-symmetry feature flag:

/// Non-Abelian quantum number (SU(2) irrep label).  
\#\[cfg(feature \= "su2-symmetry")\]  
pub struct SU2Irrep {  
    pub twice\_j: u32,  // 2j to avoid half-integer arithmetic  
}

/// Factorized tensor: CG coefficients ⊗ reduced matrix elements.  
\#\[cfg(feature \= "su2-symmetry")\]  
pub struct WignerEckartTensor\<T: Scalar\> {  
    /// Pre-computed or cached Clebsch-Gordan coefficients.  
    structural: ClebschGordanCache,  
    /// Dynamic reduced matrix elements (the variational DOFs).  
    reduced: BlockSparseTensor\<T, SU2Irrep\>,  
}  
The lie-groups crate provides the group-theoretic machinery (Casimir operators, characters, CG decompositions) required for SU(N) algebras. When the feature flag is active, the contraction engine automatically applies CG coefficient lookup during block-sparse contractions.

# **5\. Abstract Linear Algebra Backend (tk-linalg)**

DMRG performance is bottlenecked by three fundamental operations: generalized matrix multiplication (GEMM) for tensor contraction, singular value decomposition (SVD) for basis truncation, and eigenvalue decomposition (EVD) for ground-state targeting. The tk-linalg crate abstracts these behind trait interfaces, allowing compile-time backend selection.

## **5.1 Backend Trait Definitions**

/// Core linear algebra operations required by the tensor library.  
pub trait LinAlgBackend: Send \+ Sync {  
    /// General matrix multiply: C \= α \* A \* B \+ β \* C  
    fn gemm\<T: Scalar\>(  
        alpha: T, a: \&MatRef\<T\>, b: \&MatRef\<T\>,  
        beta: T, c: \&mut MatMut\<T\>,  
    );

    /// Truncated SVD returning (U, S, V†) with at most max\_rank values.  
    fn svd\_truncated\<T: Scalar\>(  
        mat: \&MatRef\<T\>, max\_rank: usize, cutoff: T::Real,  
    ) \-\> (DenseTensor\<T\>, Vec\<T::Real\>, DenseTensor\<T\>);

    /// Lowest k eigenvalues/eigenvectors of a symmetric matrix.  
    fn eigh\_lowest\<T: Scalar\>(  
        mat: \&MatRef\<T\>, k: usize,  
    ) \-\> (Vec\<T::Real\>, DenseTensor\<T\>);

    /// QR decomposition.  
    fn qr\<T: Scalar\>(  
        mat: \&MatRef\<T\>,  
    ) \-\> (DenseTensor\<T\>, DenseTensor\<T\>);  
}

/// Sparse-aware backend extension for block-sparse operations.  
pub trait SparseLinAlgBackend: LinAlgBackend {  
    /// Sparse matrix-vector product: y \= A \* x  
    fn spmv\<T: Scalar, Q: QuantumNumber\>(  
        a: \&BlockSparseTensor\<T, Q\>, x: &\[T\], y: \&mut \[T\],  
    );

    /// Block-sparse GEMM dispatching dense GEMM per sector.  
    fn block\_gemm\<T: Scalar, Q: QuantumNumber\>(  
        a: \&BlockSparseTensor\<T, Q\>,  
        b: \&BlockSparseTensor\<T, Q\>,  
    ) \-\> BlockSparseTensor\<T, Q\>;  
}

## **5.2 Backend Implementations**

| Backend Struct | Feature Flag | Characteristics |
| :---- | :---- | :---- |
| **DeviceFaer** | backend-faer (default) | Pure Rust; state-of-the-art multithreaded SVD; ideal for high bond-dimension DMRG; cache-optimized dense decompositions |
| **DeviceOxiblas** | backend-oxiblas (default) | Pure Rust; explicit SIMD (AVX-512, NEON); 9 sparse formats (CSR, CSC, BSR, DIA, etc.); f128 extended precision |
| **DeviceMKL** | backend-mkl | FFI to Intel MKL; vendor-optimized for Xeon/Xeon Phi; requires proprietary license |
| **DeviceOpenBLAS** | backend-openblas | FFI to OpenBLAS; broad HPC cluster support; Fortran-compiled routines |

Following the rstsr framework pattern, the default configuration activates both DeviceFaer and DeviceOxiblas simultaneously. Dense operations (SVD, EVD, QR) are dispatched to faer for maximum single-node throughput, while sparse/block-sparse operations route to oxiblas. The DeviceAPI trait, modeled after rstsr’s architecture, encapsulates the dispatch logic:

/// Unified device abstraction routing to the active backend.  
pub struct DeviceAPI\<D: LinAlgBackend, S: SparseLinAlgBackend\> {  
    dense: D,  
    sparse: S,  
}

/// Default: faer for dense, oxiblas for sparse.  
\#\[cfg(all(feature \= "backend-faer", feature \= "backend-oxiblas"))\]  
pub type DefaultDevice \= DeviceAPI\<DeviceFaer, DeviceOxiblas\>;

## **5.3 Parallelism Strategy**

Element-wise operations (tensor addition, scalar multiplication, trace, norm computation) leverage Rayon for data-parallel execution across CPU cores. The parallel feature flag gates this behavior, allowing single-threaded builds for environments where thread spawning is restricted. Block-sparse contractions are naturally parallelizable because each symmetry-sector GEMM is independent; Rayon’s par\_iter dispatches these simultaneously.

# **6\. Contraction Engine (tk-contract)**

The contraction engine separates two distinct concerns: (1) finding the optimal pairwise contraction sequence (an NP-hard combinatorial optimization problem), and (2) executing each pairwise contraction by dispatching to the linear algebra backend. This separation allows path optimization to be improved independently of execution performance.

## **6.1 Contraction Graph (DAG)**

/// A node in the contraction DAG.  
pub enum ContractionNode {  
    /// Leaf: references an input tensor by its unique ID.  
    Input { tensor\_id: TensorId },  
    /// Internal: result of contracting two child nodes.  
    Contraction {  
        left: Box\<ContractionNode\>,  
        right: Box\<ContractionNode\>,  
        /// Pairs of (left\_axis, right\_axis) to contract over.  
        contracted\_indices: Vec\<(IndexId, IndexId)\>,  
    },  
}

/// Complete contraction specification: inputs \+ contraction tree.  
pub struct ContractionGraph {  
    inputs: Vec\<TensorId\>,  
    root: ContractionNode,  
    estimated\_flops: f64,  
    estimated\_memory: usize,  
}

## **6.2 Path Optimizer Trait**

/// Trait for contraction path optimization strategies.  
pub trait PathOptimizer: Send \+ Sync {  
    /// Given input tensor shapes and shared indices, return an optimal  
    /// contraction tree minimizing estimated FLOP count.  
    fn optimize(  
        \&self,  
        inputs: &\[\&TensorShape\],  
        index\_map: \&IndexMap,  
    ) \-\> ContractionGraph;  
}

/// Greedy O(n³) pathfinder for rapid optimization.  
pub struct GreedyOptimizer { pub cost\_fn: CostMetric }

/// Simulated annealing (TreeSA) for near-optimal contraction trees.  
pub struct TreeSAOptimizer {  
    pub max\_iterations: usize,  
    pub initial\_temperature: f64,  
    pub cooling\_rate: f64,  
}

/// Dynamic programming optimizer (cotengrust-compatible).  
pub struct DPOptimizer { pub max\_width: usize }

## **6.3 Contraction Executor**

The executor walks the contraction tree bottom-up, performing each pairwise contraction. For each node, it: (1) permutes the input tensors so that contracted indices are trailing/leading, (2) reshapes into matrices, (3) calls the backend GEMM, and (4) reshapes the result. When the tblis interface is available, the reshape steps are skipped entirely, performing direct strided tensor contraction.

/// Execute a pre-optimized contraction graph.  
pub struct ContractionExecutor\<B: LinAlgBackend\> {  
    backend: B,  
    arena: SweepArena,  // temporary allocations  
}

impl\<B: LinAlgBackend\> ContractionExecutor\<B\> {  
    pub fn execute\<T: Scalar\>(  
        \&self,  
        graph: \&ContractionGraph,  
        tensors: \&TensorRegistry\<T\>,  
    ) \-\> DenseTensor\<T\> { /\* recursive DAG walk \*/ }  
}

# **7\. Ergonomic API & Domain-Specific Language (tk-dsl)**

## **7.1 Intelligent Index System**

Following the ITensor paradigm, all tensor indices carry unique identity and metadata rather than being tracked by positional integers. This eliminates axis-ordering errors and enables automatic contraction:

/// A uniquely-identified tensor index with physical metadata.  
\#\[derive(Clone, Eq, PartialEq, Hash)\]  
pub struct Index {  
    /// Globally unique identifier.  
    id: IndexId,  
    /// Human-readable tag (e.g., "Site,3" or "Link,l=2").  
    tag: SmallString\<\[u8; 32\]\>,  
    /// Dimension of this index.  
    dim: usize,  
    /// Prime level for distinguishing bra vs ket indices.  
    prime\_level: u32,  
    /// Direction: Incoming (ket) or Outgoing (bra).  
    direction: IndexDirection,  
}

/// Tensor contraction: matching indices contract automatically.  
pub fn contract\<T: Scalar\>(  
    a: \&IndexedTensor\<T\>,  
    b: \&IndexedTensor\<T\>,  
) \-\> IndexedTensor\<T\> {  
    // Indices with matching id \+ complementary prime levels contract.  
    // Remaining indices become the output tensor's legs.  
}

The prime() method increments the prime level, allowing users to distinguish physical indices on bra and ket sides of an MPO without manual bookkeeping. Contracting an MPS with an MPO becomes a single function call; matching indices are resolved automatically.

## **7.2 Automated MPO Generation: The hamiltonian\!{} Macro**

Constructing Matrix Product Operators by hand is error-prone and mathematically involved. The tk-dsl crate provides a procedural macro that parses human-readable lattice Hamiltonian definitions and automatically generates compressed MPOs using finite-state-automaton or SVD-based compression:

use tk\_dsl::hamiltonian;

// Heisenberg XXZ chain with nearest-neighbor interactions  
let H \= hamiltonian\! {  
    lattice: Chain(N \= 100, d \= 2);  
    sum i in 0..N-1 {  
        J  \* (Sp(i) \* Sm(i+1) \+ Sm(i) \* Sp(i+1))  
      \+ Jz \* Sz(i) \* Sz(i+1)  
    }  
    sum i in 0..N {  
        h \* Sz(i)   // external magnetic field  
    }  
};

// Single-band Hubbard model for DMFT impurity  
let H\_aim \= hamiltonian\! {  
    lattice: Star(n\_bath \= 6, d \= 4);  // 4-dim local: |0\>, |up\>, |dn\>, |ud\>  
    // Impurity on-site interaction  
    U \* Nup(0) \* Ndn(0)  
    // Hybridization with bath sites  
    sum k in 1..=n\_bath {  
        V\[k\] \* (Cdag\_up(0) \* C\_up(k) \+ h.c.)  
      \+ V\[k\] \* (Cdag\_dn(0) \* C\_dn(k) \+ h.c.)  
      \+ eps\[k\] \* (Nup(k) \+ Ndn(k))  
    }  
};

The macro expands at compile time to an OpSum builder that assembles operator terms, validates site indices, and calls the MPO compression backend. Syntax errors in the Hamiltonian definition produce clear compile-time diagnostics, catching physics mistakes before any runtime execution.

## **7.3 OpSum Builder Pattern**

For users preferring a programmatic API over macro syntax, the OpSum builder provides equivalent functionality:

let mut opsum \= OpSum::new();  
for i in 0..n\_sites \- 1 {  
    opsum \+= J  \* op("S+", i) \* op("S-", i+1);  
    opsum \+= J  \* op("S-", i) \* op("S+", i+1);  
    opsum \+= Jz \* op("Sz", i) \* op("Sz", i+1);  
}  
let mpo: MPO\<f64, U1\> \= opsum.build(\&site\_indices)?;

## **7.4 Lattice Abstraction**

/// Abstract lattice geometry.  
pub trait Lattice {  
    /// Number of physical sites.  
    fn n\_sites(\&self) \-\> usize;  
    /// Nearest-neighbor bonds.  
    fn bonds(\&self) \-\> &\[(usize, usize)\];  
    /// Optimal 1D mapping for DMRG (snake-path).  
    fn dmrg\_ordering(\&self) \-\> Vec\<usize\>;  
}

pub struct Chain  { pub n: usize }  
pub struct Square { pub lx: usize, pub ly: usize }  
pub struct Triangular { pub lx: usize, pub ly: usize }  
pub struct BetheLattice { pub z: usize, pub depth: usize }

For 2D lattices, the dmrg\_ordering() method returns a space-filling curve (snake path) that minimizes the maximum entanglement width when the lattice is mapped to a 1D MPS chain. The library provides built-in heuristics for square and triangular lattices as a variant of the minimum linear arrangement problem.

# **8\. DMRG Algorithm & DMFT Integration**

## **8.1 MPS with Typestate Canonical Forms**

Rust’s type system enforces gauge conditions at compile time via the typestate pattern. An MPS can only be in one of three well-defined canonical states, and operations that require a specific gauge form will not compile if applied to an improperly conditioned state:

/// Marker types for MPS canonical form (zero-size, compile-time only).  
pub struct LeftCanonical;  
pub struct RightCanonical;  
pub struct MixedCanonical { pub center: usize }

/// Matrix Product State parameterized by its gauge condition.  
pub struct MPS\<T: Scalar, Q: QuantumNumber, Gauge\> {  
    tensors: Vec\<BlockSparseTensor\<T, Q\>\>,  
    \_gauge: PhantomData\<Gauge\>,  
}

impl\<T: Scalar, Q: QuantumNumber\> MPS\<T, Q, RightCanonical\> {  
    /// Shift center of orthogonality leftward via QR decomposition.  
    /// Consumes RightCanonical, produces MixedCanonical.  
    pub fn move\_center\_left(self, to: usize)  
        \-\> MPS\<T, Q, MixedCanonical\> { /\* ... \*/ }  
}

impl\<T: Scalar, Q: QuantumNumber\> MPS\<T, Q, MixedCanonical\> {  
    /// Two-site DMRG update: only valid on MixedCanonical MPS.  
    pub fn dmrg\_step(  
        \&mut self,  
        mpo: \&MPO\<T, Q\>,  
        env: \&mut Environments\<T, Q\>,  
        solver: \&dyn IterativeEigensolver\<T\>,  
    ) \-\> T::Real { /\* returns energy \*/ }  
}

## **8.2 Iterative Eigensolver Trait**

Because the effective Hamiltonian is far too large to store explicitly, it is treated as an implicit linear operator. The eigensolver receives only a closure performing the matrix-vector product:

/// Iterative eigensolver interface for DMRG local optimization.  
pub trait IterativeEigensolver\<T: Scalar\>: Send \+ Sync {  
    /// Find the lowest eigenvalue and eigenvector.  
    fn lowest\_eigenpair(  
        \&self,  
        /// Linear operator as a closure: |x| \-\> H\_eff \* x  
        matvec: \&dyn Fn(&\[T\]) \-\> Vec\<T\>,  
        dim: usize,  
        initial\_guess: Option\<&\[T\]\>,  
    ) \-\> (T::Real, Vec\<T\>);  
}

pub struct LanczosSolver  { pub max\_iter: usize, pub tol: f64 }  
pub struct DavidsonSolver { pub max\_iter: usize, pub tol: f64,  
                            pub max\_subspace: usize }

## **8.3 DMRG Sweep Engine**

The two-site DMRG sweep iterates across the chain, updating pairs of adjacent tensors. The full workflow per step is: (1) contract environment tensors L and R with the local MPO to form H\_eff, (2) solve the local eigenvalue problem via the iterative eigensolver, (3) decompose the updated two-site tensor via SVD, (4) truncate singular values to control bond dimension, and (5) update the environment blocks.

pub struct DMRGEngine\<T: Scalar, Q: QuantumNumber, B: LinAlgBackend\> {  
    pub mps: MPS\<T, Q, MixedCanonical\>,  
    pub mpo: MPO\<T, Q\>,  
    pub environments: Environments\<T, Q\>,  
    pub backend: B,  
    pub config: DMRGConfig,  
}

pub struct DMRGConfig {  
    pub max\_bond\_dim: usize,  
    pub svd\_cutoff: f64,  
    pub max\_sweeps: usize,  
    pub energy\_tol: f64,  
    pub eigensolver: Box\<dyn IterativeEigensolver\<f64\>\>,  
}

impl\<T, Q, B\> DMRGEngine\<T, Q, B\>  
where T: Scalar, Q: QuantumNumber, B: LinAlgBackend  
{  
    /// Run full DMRG optimization to convergence.  
    pub fn run(\&mut self) \-\> DMRGResult\<T\> {  
        for sweep in 0..self.config.max\_sweeps {  
            let energy \= self.sweep\_left\_to\_right()?;  
            let energy \= self.sweep\_right\_to\_left()?;  
            if self.converged(energy) { break; }  
        }  
        DMRGResult { energy, mps: \&self.mps, ... }  
    }  
}

## **8.4 DMFT Self-Consistency Loop**

The tk-dmft crate implements the full DMFT workflow: (1) bath discretization via Lanczos tridiagonalization of the hybridization function, (2) mapping the Anderson Impurity Model to a 1D chain geometry for MPS representation, (3) ground-state computation via DMRG, (4) real-time evolution (TEBD/TDVP) of the excited state to compute the retarded Green’s function, (5) linear prediction to extrapolate time-series data beyond the entanglement-limited simulation window, and (6) Fourier transform to obtain the spectral function.

pub struct DMFTLoop\<T: Scalar, Q: QuantumNumber, B: LinAlgBackend\> {  
    pub impurity: AndersonImpurityModel\<T\>,  
    pub dmrg\_config: DMRGConfig,  
    pub time\_evolution: TimeEvolutionConfig,  
    pub self\_consistency\_tol: f64,  
    pub max\_dmft\_iterations: usize,  
    backend: B,  
}

impl\<T, Q, B\> DMFTLoop\<T, Q, B\>  
where T: Scalar\<Real \= f64\>, Q: QuantumNumber, B: LinAlgBackend  
{  
    /// Execute the self-consistency loop.  
    pub fn solve(\&mut self) \-\> SpectralFunction {  
        loop {  
            let chain \= self.impurity.discretize\_bath();  
            let gs \= DMRGEngine::new(chain, ...).run();  
            let g\_t \= self.time\_evolve(\&gs);  
            let g\_t\_ext \= linear\_predict(\&g\_t);  
            let spectral \= fft(\&g\_t\_ext);  
            if self.converged(\&spectral) { return spectral; }  
            self.impurity.update\_bath(\&spectral);  
        }  
    }  
}

# **9\. Data Flow: DMRG Sweep Step**

The following describes the complete data flow through the system for a single two-site DMRG update step, demonstrating how all architectural components interact:

| Step | Operation | Crate(s) | Detail |
| :---- | :---- | :---- | :---- |
| 1 | **Build H\_eff** | tk-contract | Contract left environment L, right environment R, and local MPO tensors into effective Hamiltonian. PathOptimizer selects the contraction order; ContractionExecutor dispatches to backend. |
| 2 | **Solve eigenvalue problem** | tk-dmrg | IterativeEigensolver (Lanczos or Davidson) finds the lowest eigenpair of H\_eff using only matvec closure. The closure internally re-contracts H\_eff with the trial vector, exploiting block-sparsity. |
| 3 | **SVD truncation** | tk-linalg (faer) | The two-site eigenvector is reshaped into a matrix and decomposed via truncated SVD. Singular values below cutoff or beyond max bond dimension are discarded. |
| 4 | **Update MPS tensors** | tk-dmrg | SVD factors are absorbed: U \* S goes into the left tensor, V† into the right tensor. MPS gauge condition (MixedCanonical) is preserved via the typestate system. |
| 5 | **Update environments** | tk-contract, tk-symmetry | The environment block on the side the sweep is moving away from is updated by contracting the newly updated MPS tensor with the old environment and MPO. Block-sparse sector matching preserves symmetry. |
| 6 | **Arena reset** | tk-core | All temporary intermediate tensors allocated during this step are reclaimed in O(1) via SweepArena::reset(). The MPS tensors and environment blocks persist in standard heap storage. |

# **10\. Hardware Extension Architecture: CUDA & MPI**

A primary design goal is that the trait-based backend abstraction makes adding new hardware targets a matter of implementing existing interfaces rather than refactoring core logic. This section details the concrete extension points and the generalized storage abstraction required for GPU and distributed-memory backends.

## **10.1 Generalized Storage Trait**

The current TensorStorage\<T\> assumes host-resident Vec\<T\> memory. To support device-resident buffers (GPU VRAM) and distributed shards (MPI ranks), the storage layer must be parameterized over an allocator/location trait:

/// Abstract memory location for tensor data.  
pub trait StorageDevice: Send \+ Sync \+ 'static {  
    type Alloc: Allocator;

    /// Allocate a contiguous buffer of \`len\` elements.  
    fn alloc\<T: Scalar\>(len: usize) \-\> DeviceBuffer\<T, Self\>;

    /// Synchronize (e.g., stream sync on GPU, barrier on MPI).  
    fn synchronize(\&self);  
}

/// Host CPU memory (default).  
pub struct HostDevice;

/// CUDA device memory, parameterized by device ordinal.  
\#\[cfg(feature \= "backend-cuda")\]  
pub struct CudaDevice { pub ordinal: usize }

/// MPI-distributed storage across ranks.  
\#\[cfg(feature \= "backend-mpi")\]  
pub struct MpiDevice { pub comm: MpiComm, pub rank: usize }

/// Generalized tensor storage, generic over memory location.  
pub struct TensorStorage\<T: Scalar, D: StorageDevice \= HostDevice\> {  
    data: DeviceBuffer\<T, D\>,  
    device: D,  
}

Existing code remains unchanged because the default type parameter D \= HostDevice preserves backward compatibility. A DenseTensor\<T\> is implicitly DenseTensor\<T, HostDevice\>, and all current call sites compile without modification.

## **10.2 CUDA / GPU Backend**

Adding GPU acceleration requires three components:

**DeviceCuda backend struct:** Implements LinAlgBackend by dispatching GEMM to cuBLAS, SVD/EVD to cuSOLVER, and sparse operations to cuSPARSE. Gated behind the backend-cuda feature flag.

**Device memory management:** CudaDevice wraps CUDA stream and memory pool APIs. The SweepArena gains a GPU-side counterpart (CudaArena) that manages a pre-allocated device memory pool with O(1) reset semantics, mirroring the host-side bumpalo pattern.

**Host–device transfer:** Explicit copy operations move data between HostDevice and CudaDevice. These are surfaced as methods on TensorStorage, not hidden behind implicit magic, because transfer latency is a critical performance consideration the user must control.

\#\[cfg(feature \= "backend-cuda")\]  
pub struct DeviceCuda {  
    stream: cuda::Stream,  
    cublas\_handle: cublas::Handle,  
    cusolver\_handle: cusolver::Handle,  
}

\#\[cfg(feature \= "backend-cuda")\]  
impl LinAlgBackend for DeviceCuda {  
    fn gemm\<T: Scalar\>(  
        alpha: T, a: \&MatRef\<T\>, b: \&MatRef\<T\>,  
        beta: T, c: \&mut MatMut\<T\>,  
    ) {  
        // Dispatch to cuBLAS on self.stream  
        cublas::gemm(self.cublas\_handle, ...);  
    }

    fn svd\_truncated\<T: Scalar\>(  
        mat: \&MatRef\<T\>, max\_rank: usize, cutoff: T::Real,  
    ) \-\> (DenseTensor\<T\>, Vec\<T::Real\>, DenseTensor\<T\>) {  
        // Dispatch to cuSOLVER gesvd  
        cusolver::gesvd(self.cusolver\_handle, ...);  
    }

    // ... eigh\_lowest, qr follow the same pattern  
}

### **10.2.1 GPU Performance Considerations**

DMRG is inherently sequential at the sweep level: each two-site update depends on the environment blocks produced by the previous step. GPU acceleration therefore helps within each step (large GEMM and SVD calls), not across steps. For bond dimensions below roughly D ≈ 500, kernel launch overhead and host–device transfer latency may negate the GPU advantage. The architecture supports a hybrid strategy where the DeviceAPI routes small operations to DeviceFaer on the CPU and large operations to DeviceCuda, selected by a configurable dimension threshold.

Block-sparse contractions are naturally suited to GPU parallelism because each symmetry-sector GEMM is independent. A batched cuBLAS call can execute all sector GEMMs in a single kernel launch, amortizing overhead across hundreds of small matrix multiplications.

## **10.3 MPI / Distributed-Memory Backend**

MPI extension targets two distinct parallelism modes, each with different architectural implications:

### **10.3.1 Mode A: Distributed Block-Sparse Tensors**

For systems requiring bond dimensions too large for a single node, the BlockSparseTensor symmetry sectors are partitioned across MPI ranks. Each rank owns a disjoint subset of the sector keys in the BTreeMap. Contractions that involve cross-rank sectors require inter-rank communication:

\#\[cfg(feature \= "backend-mpi")\]  
pub struct DeviceMPI\<Inner: LinAlgBackend\> {  
    inner: Inner,           // local compute backend (e.g., DeviceFaer)  
    comm: MpiComm,  
    rank: usize,  
    world\_size: usize,  
}

\#\[cfg(feature \= "backend-mpi")\]  
impl\<Inner: LinAlgBackend\> SparseLinAlgBackend for DeviceMPI\<Inner\> {  
    fn block\_gemm\<T: Scalar, Q: QuantumNumber\>(  
        a: \&BlockSparseTensor\<T, Q\>,  
        b: \&BlockSparseTensor\<T, Q\>,  
    ) \-\> BlockSparseTensor\<T, Q\> {  
        // 1\. Compute local sector GEMMs via self.inner  
        // 2\. Identify cross-rank sector dependencies  
        // 3\. MPI\_Isend / MPI\_Irecv for boundary sectors  
        // 4\. Merge partial results via MPI\_Allreduce  
    }  
}

The ContractionGraph DAG is analyzed at optimization time to insert explicit communication nodes. The PathOptimizer gains an MPI-aware cost model that penalizes contraction orderings requiring excessive inter-rank data movement, favoring paths that keep communication volume minimal.

### **10.3.2 Mode B: Embarrassingly Parallel DMFT**

The simpler and higher-value MPI use case is parallelism at the DMFT self-consistency loop level. Each MPI rank runs an independent DMRGEngine with different bath parameters, initial conditions, or k-point samplings. Synchronization occurs only at the DMFT convergence check, where spectral functions are gathered via MPI\_Allgather:

\#\[cfg(feature \= "backend-mpi")\]  
impl\<T, Q, B\> DMFTLoop\<T, Q, B\>  
where T: Scalar\<Real \= f64\>, Q: QuantumNumber, B: LinAlgBackend  
{  
    /// Parallel DMFT: each rank solves one impurity problem.  
    pub fn solve\_parallel(\&mut self, comm: \&MpiComm) \-\> SpectralFunction {  
        let my\_rank \= comm.rank();  
        loop {  
            // Each rank solves its local impurity  
            let local\_spectral \= self.solve\_local\_impurity();

            // Gather all spectral functions  
            let all\_spectral \= comm.allgather(\&local\_spectral);

            // Rank 0 checks convergence, broadcasts decision  
            let converged \= comm.broadcast(  
                if my\_rank \== 0 { self.check\_convergence(\&all\_spectral) }  
                else { false }  
            );  
            if converged { return self.merge\_spectral(\&all\_spectral); }

            // Update bath from averaged self-energy  
            self.impurity.update\_bath(\&all\_spectral);  
        }  
    }  
}

This mode requires no changes to the core tensor types or the contraction engine. It operates entirely at the application layer (tk-dmft), making it the recommended first target for MPI support.

## **10.4 Extension Comparison & Recommendations**

| Extension | Scope of Change | Risk | Phase | Value |
| :---- | :---- | :---- | :---- | :---- |
| **CUDA (single-node GPU)** | New DeviceCuda \+ generalize StorageDevice trait | Medium: cuBLAS/cuSOLVER are mature; transfer overhead needs profiling | Phase 5 | High |
| **MPI Mode B (parallel DMFT)** | Application-layer only; no core changes | Low: embarrassingly parallel; well-understood pattern | Phase 4–5 | High |
| **MPI Mode A (distributed tensors)** | ContractionExecutor \+ BlockSparseTensor \+ PathOptimizer | High: communication-aware path optimization is research-grade | Phase 5+ | Medium |
| **Multi-GPU (NCCL)** | DeviceCuda \+ NCCL collective wrappers | High: requires careful stream synchronization across devices | Phase 5+ | Medium |

The recommended implementation order is: (1) generalize StorageDevice in Phase 4 as a non-breaking refactor, (2) implement MPI Mode B for parallel DMFT in Phase 4–5, (3) implement DeviceCuda in Phase 5, and (4) pursue distributed tensor MPI and multi-GPU only when single-node capacity is demonstrably insufficient for the target physics problems.

# **11\. External Crate Dependencies**

The following table maps each external Rust crate to its role within the library architecture:

| Crate | Version / Source | Used By | Purpose |
| :---- | :---- | :---- | :---- |
| **faer** | crates.io (latest) | tk-linalg | Dense SVD, EVD, QR, LU; multithreaded cache-optimized |
| **oxiblas** | github.com/cool-japan | tk-linalg | Sparse ops (9 formats), SIMD BLAS, f128 |
| **bumpalo** | crates.io | tk-core | Arena allocator for sweep temporaries |
| **smallvec** | crates.io | tk-core | Stack-allocated small vectors for shapes/strides |
| **rayon** | crates.io | tk-linalg, tk-contract | Data-parallel iterators |
| **num / num-complex** | crates.io | tk-core | Complex\<f64\>, numeric traits |
| **omeco** | crates.io | tk-contract | Greedy \+ TreeSA contraction path optimization |
| **cotengrust** | crates.io | tk-contract | DP-based path optimization |
| **lie-groups** | crates.io | tk-symmetry (optional) | SU(N) CG coefficients, Casimirs |
| **eigenvalues** | crates.io | tk-dmrg | Davidson/Lanczos iterative eigensolvers |
| **pyo3** | crates.io | tk-python | Python bindings for TRIQS integration |
| **spenso** | crates.io | tk-contract (reference) | Structural tensor graph inspiration |
| **cudarc** | crates.io (optional) | tk-linalg (optional) | Safe Rust wrappers for CUDA driver, cuBLAS, cuSOLVER |
| **mpi** | crates.io (optional) | tk-linalg, tk-dmft (optional) | Rust MPI bindings wrapping system MPI library |

# **12\. Testing & Benchmarking Strategy**

## **12.1 Correctness Testing**

Each sub-crate carries its own unit test suite. Critical numerical invariants are tested via property-based testing using the proptest crate:

* **tk-core:** Round-trip shape permutations, stride arithmetic consistency, arena allocation/reset safety.

* **tk-symmetry:** Quantum number fusion associativity, flux conservation after contraction, block-dense equivalence for small systems.

* **tk-linalg:** SVD reconstruction error \< machine epsilon, orthogonality of eigenvectors, GEMM against reference implementations.

* **tk-contract:** Contraction path FLOP estimates vs brute-force, result equivalence across optimizers.

* **tk-dmrg:** Ground-state energy of Heisenberg chain (N=10,20) against exact diagonalization, canonical form invariants.

* **tk-dmft:** Spectral function sum rules, bath discretization accuracy, convergence of the self-consistency loop on the single-impurity Anderson model.

## **12.2 Performance Benchmarks**

Criterion.rs benchmarks track regression across the following critical paths: dense SVD at bond dimensions D \= 100, 500, 1000, 2000; block-sparse GEMM with U(1) symmetry sectors; full DMRG sweep on the Heisenberg chain (N=100, D=200); and contraction path optimization wall time for 10-, 20-, and 50-tensor networks.

# **13\. Risk Analysis & Mitigation**

| Risk | Severity | Mitigation |
| :---- | :---- | :---- |
| faer API instability (pre-1.0) | Medium | Abstract behind LinAlgBackend trait; version-pin in Cargo.toml; fallback to OpenBLAS FFI |
| oxiblas sparse format coverage gaps | Medium | Validate BSR support early; maintain nalgebra-sparse as backup for CSR/CSC operations |
| SU(2) Wigner-Eckart complexity | High | Defer behind feature flag; prototype with QSpace reference implementation; fund dedicated development phase |
| Compile-time overhead from deep generics | Low | Strategic use of dynamic dispatch (dyn Trait) at boundaries; profile compile times per crate |
| DMFT loop convergence sensitivity | High | Implement mixing schemes (linear, Broyden); validate against published DMFT benchmarks (Bethe lattice half-filling) |
| Linear prediction extrapolation artifacts | Medium | Provide Chebyshev expansion as alternative real-frequency route; cross-validate spectral functions from both methods |

# **14\. Implementation Roadmap**

| Phase | Target | Deliverables |
| :---- | :---- | :---- |
| **Phase 1** | Months 1–3 | tk-core (shape/storage/arena/Cow), tk-symmetry (U(1), Z₂, block-sparse), tk-linalg (DeviceFaer dense SVD/EVD). Unit tests achieving \>90% coverage. Benchmark suite for SVD at multiple bond dimensions. |
| **Phase 2** | Months 4–6 | tk-contract (DAG, greedy optimizer, executor), tk-dsl (Index, OpSum, Chain/Square lattices, hamiltonian\!{} macro). Integration test: Heisenberg ground state energy matches exact diag. |
| **Phase 3** | Months 7–9 | tk-dmrg (MPS typestates, two-site sweep, Lanczos/Davidson). Full DMRG benchmark: N=100 Heisenberg chain at D=500. Performance comparison vs ITensor/TeNPy. |
| **Phase 4** | Months 10–12 | tk-dmft (bath discretization, TEBD/TDVP, linear prediction, Chebyshev expansion, DMFT loop). tk-python (PyO3 bindings). Generalize StorageDevice trait in tk-core (non-breaking refactor). MPI Mode B: parallel DMFT loop via mpi crate. Validation: single-band Hubbard DMFT on Bethe lattice. |
| **Phase 5** | Months 13+ | DeviceCuda (cuBLAS/cuSOLVER) with CudaArena memory pooling. SU(2) non-Abelian support (su2-symmetry feature). TreeSA/DP optimizers. Multi-orbital DMFT. MPI Mode A (distributed tensors) if single-node insufficient. Community release and documentation. |

# **15\. Conclusion**

This architecture provides a rock-solid foundation for a Rust tensor network library that is modular, safe, and performant. By decoupling tensor shape from storage, abstracting linear algebra backends behind traits, separating contraction path optimization from execution, and encoding physical gauge conditions in the type system, the design eliminates entire categories of bugs at compile time while preserving the computational intensity required for state-of-the-art quantum many-body simulations.

The clear crate boundaries and feature-flag system ensure that the library can evolve incrementally—adding non-Abelian symmetries, GPU backends, and additional lattice geometries—without destabilizing the core infrastructure. The phased implementation roadmap prioritizes delivering a working DMRG solver as early as Phase 3, enabling real physics research to begin while the DMFT integration matures in parallel.