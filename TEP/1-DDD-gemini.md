Here is a Detailed Design Document (DDD), structured around Domain-Driven Design principles, outlining the bounded contexts, domain models, and technical architecture of the **Tensorkraft** project based on the v7.1 specification.

---

# Detailed Design Document (DDD): Tensorkraft

**Domain:** Quantum Many-Body Physics (DMRG & DMFT)
**Architecture Pattern:** Domain-Driven Design (DDD) with Rust Typestates & Zero-Copy Abstractions

## 1. Domain Overview & Ubiquitous Language

Tensorkraft operates within the highly specialized domain of quantum impurity solvers and tensor network contractions. Establishing a ubiquitous language ensures that the software models perfectly reflect the physical and mathematical reality.

* **MPS (Matrix Product State):** A 1D tensor network representing a quantum many-body wavefunction.
* **MPO (Matrix Product Operator):** A 1D tensor network representing a quantum operator (e.g., a Hamiltonian).
* **Sweep:** A sequential, back-and-forth traversal of the MPS, optimizing local site tensors.
* **Gauge / Canonical Form:** Mathematical constraints on tensor basis vectors (Left, Right, Mixed, Bond-Centered) critical for numerical stability.
* **Quantum Number (Flux):** Conserved physical quantities (e.g., particle number, spin) that dictate block-sparsity in tensors.
* **TDVP (Time-Dependent Variational Principle):** The primary algorithm for real-time evolution of the MPS.

---

## 2. Bounded Contexts (Workspace Architecture)

The system is decomposed into strictly layered bounded contexts (implemented as Rust crates) to prevent cyclic dependencies and enforce domain boundaries.

| Bounded Context | Rust Crate | Responsibility |
| --- | --- | --- |
| **Foundation & Memory** | `tk-core` | Arena allocation, tensor shape/stride metadata, and zero-copy matrix views (`MatRef`). |
| **Physics Symmetry** | `tk-symmetry` | Quantum number bit-packing, flux conservation rules, and block-sparse memory layouts. |
| **Hardware & Math** | `tk-linalg` | Hardware abstraction layer (CPU/GPU), LPT thread scheduling, and trait-based SVD/GEMM dispatch. |
| **Contraction Engine** | `tk-contract` | DAG construction, composite cost path optimization, and strided execution. |
| **Domain Specific Language** | `tk-dsl` | Ergonomic operator definitions (`SpinOp`, `FermionOp`) and AST generation for Hamiltonians. |
| **DMRG Subsystem** | `tk-dmrg` | MPO compilation, MPS typestate transitions, eigensolvers, and subspace expansion. |
| **DMFT Application** | `tk-dmft` | Top-level DMFT loop, TDVP time evolution, and linear prediction deconvolutions. |
| **Python Interop** | `tk-python` | GIL-safe bindings, asynchronous cancellation, and zero-copy NumPy memory sharing. |

---

## 3. Core Domain Models (Entities & Value Objects)

### 3.1 Tensor Memory Model (`tk-core`)

The architecture enforces a strict separation between tensor metadata (Shape/Strides) and flat memory storage, utilizing Copy-on-Write (Cow) semantics.

* **Value Object:** `TensorShape` tracks dimensions and row-major strides.
* **Value Object:** `MatRef<'a, T>` represents a zero-copy matrix view. It includes an `is_conjugated` boolean flag. Instead of executing an $O(N)$ memory pass to compute $A^\dagger$, the system flips this flag and swaps strides, fusing conjugation directly into the BLAS/SIMD micro-kernels.
* **Entity:** `SweepArena` manages bump-allocated memory scoped to a single DMRG step.
* **Lifecycle Rule:** Temporary outputs are typed as `TempTensor<'a>`. They must be transformed via `.into_owned()` before the arena resets, shifting the allocation to the heap. The Rust borrow checker enforces this ownership boundary statically.

### 3.2 Block-Sparse Symmetry Model (`tk-symmetry`)

To handle physical conservation laws efficiently, the system utilizes block-sparse representations.

* **Value Object:** `PackedSectorKey` compresses Abelian quantum numbers (U(1), Z₂) into a single `u64` bitfield.
* **Domain Logic:** This bit-packing allows $O(\log N)$ binary searches over sector blocks using single-cycle register comparisons, bypassing expensive pointer chasing and branch mispredictions.
* **Future Extension:** Non-Abelian SU(2) symmetry utilizes `SU2Irrep` keys and a `structural_contraction` callback to inject Clebsch-Gordan coefficients, along with map-reduce task generation for fusion-rule multiplicity.

### 3.3 The Quantum State Model (`tk-dmrg`)

The `MPS` entity uses Rust's typestate pattern to encode mathematical gauge conditions directly into the type system, making invalid physical operations a compile-time error.

* **`MPS<T, Q, MixedCanonical>`:** Valid state for standard two-site DMRG updates.
* **`MPS<T, Q, BondCentered>`:** Exposes the singular value bond matrix. Required for the zero-site projector step in TDVP backward evolution.

---

## 4. Domain Services

### 4.1 Contraction Path Optimizer (`tk-contract`)

Finding the optimal contraction sequence is NP-hard. The `PathOptimizer` service evaluates candidate DAGs using a composite cost metric that penalizes memory bandwidth over pure FLOPs:

$$C_{total} = \alpha \cdot \text{FLOPs} + \beta \cdot \text{Bytes\_Moved}$$

This ensures the engine favors strided, in-place contractions over paths that require explicit out-of-place memory transposes.

### 4.2 Linear Algebra Dispatch & LPT Scheduling (`tk-linalg`)

The `LinAlgBackend<T>` trait serves as an object-safe anti-corruption layer separating domain logic from hardware specifics (e.g., `DeviceFaer`, `DeviceCuda`).

To prevent Rayon thread starvation caused by the binomial distribution of symmetry sectors (many small blocks, few massive blocks), the backend utilizes **Longest Processing Time (LPT) scheduling**:

1. Generate `SectorGemmTask` structs for all valid fusion rules.
2. Sort tasks descending by $M \cdot N \cdot K$ complexity.
3. Dispatch via `par_iter` to saturate all CPU cores evenly.
4. Re-sort the results by `PackedSectorKey` to restore the structural invariants.

### 4.3 TDVP Stabilization & Subspace Expansion (`tk-dmft`)

Real-time evolution requires dynamic entanglement growth. The service injects null-space vectors into the site tensors ($A_L$) without explicitly forming the dense projector, capping computational cost at $O(dD^2)$.

To prevent bond-dimension oscillation, a stateful `TdvpDriver` enforces a **Soft D_max policy**, allowing the bond dimension to temporarily exceed the threshold by 10% before smoothly decaying back down over subsequent time steps.

---

## 5. Cross-Cutting Concerns

### 5.1 Python GIL Management & Concurrency

Long-running DMFT loops freeze Jupyter kernels if the Python GIL is held. `tk-python` releases the GIL and uses a thread-safe architecture:

* An `AtomicBool` serves as a zero-cost cancellation flag checked by Rayon workers.
* An `mpsc`-guarded monitor thread listens for `SIGINT`.
* The shutdown sequence strictly occurs *inside* the `py.allow_threads` closure to prevent AB/BA deadlocks during GIL re-acquisition.

### 5.2 Pinned Memory Budgeting (CUDA / MPI)

GPU DMA transfers require page-locked (pinned) memory. To prevent OS-level Out-of-Memory kernel panics when multiple MPI ranks allocate pinned memory simultaneously:

* A global `PinnedMemoryTracker` uses lock-free atomic Compare-And-Swap (CAS) loops to enforce a strict byte ceiling.
* If the budget is exhausted, `SweepArena` gracefully falls back to pageable memory and emits a structured telemetry warning noting the 50% bandwidth regression.

---

Would you like me to elaborate on the exact memory layout of the `ContractionGraph` DAG, or should we draft the API signatures for the `tk-dsl` operator builder next?