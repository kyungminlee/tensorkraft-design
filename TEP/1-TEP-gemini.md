Here is a formal TensorKraft Enhancement Proposal (TEP) synthesized from the version 7.1 architecture document. I’ve structured this to serve as the foundational RFC/TEP for the project, capturing the core motivations, architectural pillars, and critical technical specifications.

---

# TEP-001: Core Architecture and Design of Tensorkraft

* **Status:** Architectural Specification (v7.1)
* **Type:** Standards Track
* **Domain:** Quantum Many-Body Physics, DMRG & DMFT
* **Target:** Real-frequency quantum impurity solver

---

## 1. Abstract

This TEP proposes **tensorkraft**, a high-performance, Rust-based tensor network library focused on Density Matrix Renormalization Group (DMRG) algorithms and Dynamical Mean-Field Theory (DMFT). By leveraging Rust's zero-cost abstractions, typestate patterns, and fearless concurrency, tensorkraft aims to match or exceed the performance of established C++ and Julia frameworks while eliminating runtime errors endemic to scientific computing.

## 2. Motivation

Existing tensor network libraries frequently suffer from runtime errors related to gauge conditions, memory fragmentation from deep contraction trees, thread oversubscription, and Python GIL deadlocks. Tensorkraft addresses these structurally:

* **Compile-Time Safety:** Enforcing MPS gauge conditions via Rust's typestate system.
* **Zero-Copy Abstractions:** Decoupling tensor metadata from flat memory buffers to allow zero-cost views (e.g., lazy Hermitian conjugation).
* **Hardware Efficiency:** Combating long-tail thread starvation with Longest Processing Time (LPT) scheduling and managing pinned-memory budgets across MPI nodes.

---

## 3. Workspace Architecture

The framework is divided into a strictly layered Cargo workspace to prevent dependency cycles and compilation bloat:

* **`tk-core`**: Foundational memory and metadata. Implements `SweepArena` for temporary allocations, `TensorShape`/`TensorStorage`, and the zero-copy `MatRef`/`MatMut` views.
* **`tk-symmetry`**: Quantum number abstractions (U(1), Z₂). Implements `BitPackable` for mapping sectors to `u64` integers, enabling $O(\log N)$ binary search without pointer chasing.
* **`tk-linalg`**: The linear algebra backend abstraction (`LinAlgBackend<T>`). Supports `faer` (dense pure-Rust), `oxiblas`, and FFI BLAS via object-safe trait dynamic dispatch.
* **`tk-contract`**: DAG-based contraction engine. Separates path optimization (cost metric: $\alpha \cdot \text{FLOPs} + \beta \cdot \text{Bytes\_Moved}$) from execution.
* **`tk-dsl`**: The user-facing ergonomic API. Features the `hamiltonian!{}` macro (AST generation only) and strongly-typed operator enums (e.g., `SpinOp`, `FermionOp`).
* **`tk-dmrg`**: Core DMRG logic, MPS typestates, `OpSum` $\rightarrow$ MPO SVD compression, and in-house iterative eigensolvers (Lanczos/Davidson).
* **`tk-dmft`**: DMFT self-consistency loops, TDVP/TEBD time evolution, linear prediction, and Chebyshev cross-validation.
* **`tk-python`**: PyO3 bindings with strict, decoupled GIL management and zero-copy NumPy integration.

---

## 4. Key Technical Specifications

### 4.1 Memory Management & Lazy Conjugation

DMRG contraction steps produce high volumes of intermediate tensors. `tk-core` mitigates fragmentation via `SweepArena`, an arena allocator cleared in $O(1)$ time at the end of each DMRG step. To cross the ownership boundary, SVD outputs must explicitly call `.into_owned()` before the arena resets.

Furthermore, `MatRef` includes a boolean `is_conjugated` flag. Instead of executing an $O(N)$ memory pass to compute $A^\dagger$, the matrix view flips this flag and swaps strides, propagating the metadata down to BLAS micro-kernels (or `faer` SIMD loops) which fuse conjugation into the FMA instruction.

### 4.2 LPT-Scheduled Block-Sparsity

Abelian symmetry sectors (U(1), Z₂) follow a binomial distribution, resulting in a few massive blocks and many tiny blocks. Dispatching these naively to Rayon causes thread long-tail starvation. `tk-linalg` solves this via Longest Processing Time (LPT) scheduling:

1. Generate all `SectorGemmTask` structures.
2. Sort descending by estimated FLOPs ($M \cdot N \cdot K$).
3. Dispatch via Rayon `par_iter`.
4. Re-sort results by `PackedSectorKey` to restore the binary search invariant.

### 4.3 Typestate-Enforced MPS and TDVP Stabilization

The `MPS` struct relies on compile-time typestates (`LeftCanonical`, `RightCanonical`, `MixedCanonical`, `BondCentered`). TDVP requires transitioning to `BondCentered` to evolve the bond matrix backward.

To handle ill-conditioned matrices during TDVP (where singular values approach zero), tensorkraft uses two stabilization techniques:

1. **Tikhonov-Regularized Pseudo-Inverse**: Replaces $1/s_i$ with $s_i / (s_i^2 + \delta^2)$.
2. **Matrix-Free Subspace Expansion**: Enlarges the site tensor basis dynamically without computing the dense projector $P_{null} = I - A_L \cdot A_L^\dagger$. This bounds the projection cost to $O(dD^2)$. A "soft D_max" policy prevents bond dimension oscillation by decaying the expanded basis over time.

### 4.4 DMFT Linear Prediction & Deconvolution

Linear prediction is stabilized via SVD regularization and exponential windowing $W(t) = \exp(-\eta|t|)$ for metallic phases. The resulting Lorentzian broadening is reversed post-FFT using a Tikhonov-damped deconvolution formula to prevent quadratic noise amplification:

$$A_{true}(\omega) \approx A_{windowed}(\omega) \cdot \frac{\eta^2 + \omega^2}{2\eta + \delta_{deconv} \cdot \omega^2}$$

### 4.5 Python GIL Safety

Standard GIL-release patterns in multithreaded Rayon environments cause deadlocks if a worker attempts to re-acquire the GIL while the main thread is blocked. `tk-python` isolates execution by:

* Using an `AtomicBool` as a cancellation flag.
* Spawning an `mpsc`-guarded monitor thread to check for Python signals (`SIGINT`).
* Executing the monitor thread shutdown sequence *inside* the `py.allow_threads` closure, guaranteeing the monitor thread dies before the main thread re-acquires the GIL.

---

## 5. Non-Abelian SU(2) Roadmap (Phase 5)

Non-Abelian symmetries introduce fusion rule multiplicities ($j_1 \otimes j_2 = |j_1 - j_2| \oplus \dots \oplus (j_1 + j_2)$). The architecture prepares for this via:

* A `structural_contraction` callback to inject 6j/9j symbol evaluations.
* Modifying LPT task generation to output a `Vec<SectorGemmTask>` per input pair, coupled with a map-reduce aggregation to prevent data races when multiple inputs target the same output sector.
* Multiplet-aware SVD truncation that applies $(2j+1)$ weighting and snaps to multiplet edges to prevent explicit symmetry breaking.

---

## 6. Drawbacks & Known Limitations

* **Fermionic Geometry Restrictions**: The contraction engine currently only supports bosonic legs. Fermionic models are handled via Jordan-Wigner transformations compiled directly into the MPO. This works for 1D chains and star-to-chain mappings (AIM), but Native Fermionic Swap Gates will be required before expanding to 2D PEPS or tree tensor networks.
* **Compile Times**: Heavy monomorphization across `T: Scalar, Q: BitPackable, B: LinAlgBackend<T>` risks compile-time bloat. Layered dynamic dispatch (`Box<dyn LinAlgBackend<T>>`) is supported at the sweep-scheduler level if compilation times exceed thresholds.

---

Would you like me to draft a deeper technical breakdown of a specific module, such as the matrix-free subspace expansion algorithm or the Python GIL deadlock-prevention implementation?