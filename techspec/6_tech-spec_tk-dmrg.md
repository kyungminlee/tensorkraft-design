# Technical Specification: `tk-dmrg`

**Crate:** `tensorkraft/crates/tk-dmrg`
**Version:** 0.1.0 (Pre-Implementation)
**Status:** Specification
**Last Updated:** March 2026

---

## 1. Overview

`tk-dmrg` is the DMRG algorithm engine for the tensorkraft workspace. It sits above `tk-core`, `tk-symmetry`, `tk-linalg`, `tk-contract`, and `tk-dsl` in the dependency graph and is consumed by `tk-dmft`.

**Core responsibilities:**

- **MPS representation** — Matrix Product State types with typestate-enforced canonical forms (`LeftCanonical`, `RightCanonical`, `MixedCanonical`, `BondCentered`). Site tensors are `BlockSparseTensor<T, Q>` for symmetry-exploiting storage; the typestate system ensures gauge operations compile only on validly conditioned states.
- **MPO representation** — Matrix Product Operator construction from `OpSum` (via SVD-based finite-state automaton compression) and manual construction for special-purpose operators (identity, projectors). MPO site tensors have rank 4: (physical-in, physical-out, bond-left, bond-right).
- **OpSum → MPO compilation** — The only crate that performs the SVD compression step that translates an `OpSum` from `tk-dsl` into an `MPO`. This is runtime linear algebra; it must not happen at compile time in `tk-dsl`.
- **DMRG sweep engine** — Two-site and single-site sweep variants. The sweep engine updates MPS tensors via effective Hamiltonian diagonalization, SVD truncation, and environment renewal. The `MixedCanonical` typestate enforces that the gauge center is always well-defined.
- **Environment management** — Left and right environment block tensors accumulate the partial contraction of the MPS–MPO–MPS network. Environments are updated incrementally as the sweep moves, not recomputed from scratch each step.
- **In-house iterative eigensolvers** — Lanczos, Davidson, and Block-Davidson implementations written from scratch within this crate. External eigensolvers are not used because they cannot accommodate the zero-allocation `&mut [T]` matvec closure and `SweepArena` integration required for competitive performance (design doc §8.2 / §11).
- **SVD truncation with bond dimension scheduling** — Truncated SVD via `tk-linalg` with configurable `max_bond_dim`, absolute singular value cutoff, and per-bond truncation error tracking. Supports SU(2) multiplet-aware truncation under the `su2-symmetry` feature flag.
- **TDVP time evolution** — Time-Dependent Variational Principle (1-site TDVP with projector splitting), including Tikhonov-regularized bond-matrix inversion and site-tensor subspace expansion for entanglement growth. Used by `tk-dmft` for real-time Green's function computation.
- **Excited states** — Energy-penalized DMRG targeting excited states by projecting out previously converged eigenstates; Block-Davidson for targeting the lowest k states simultaneously.
- **Infinite DMRG bootstrap** — iDMRG warm-start of the finite DMRG sweep, producing an initial MPS at the target bond dimension before expensive finite sweeps begin.
- **Checkpointing** — Serialization and deserialization of `MPS`, `MPO`, and `Environments` to disk using `serde + bincode`, enabling sweep restarts after interruption.

**Key design principle:** This crate is the integration point. All performance-critical inner loops (GEMM, SVD, sector dispatch) are delegated to `tk-linalg` and `tk-contract`. What `tk-dmrg` owns is algorithm structure: sweep ordering, gauge management, eigensolver iteration, and the ownership discipline that separates persistent MPS tensors from sweep-temporary intermediates.

---

## 2. Module Structure

```
tk-dmrg/
├── Cargo.toml
├── build.rs                    (feature conflict detection)
└── src/
    ├── lib.rs                  re-exports all public items
    ├── mps/
    │   ├── mod.rs              MPS<T, Q, Gauge> definition and gauge transitions
    │   ├── canonical.rs        left/right-canonicalization, mixed-canonical form
    │   ├── bond_centered.rs    BondCentered gauge: expose/absorb bond matrix
    │   ├── overlap.rs          <ψ|φ> computation for gauge-invariant tests
    │   └── io.rs               serde serialize/deserialize for MPS
    ├── mpo/
    │   ├── mod.rs              MPO<T, Q> definition
    │   ├── compile.rs          OpSum → MPO SVD compression (MpoCompiler)
    │   ├── identity.rs         identity MPO, single-site MPO
    │   └── io.rs               serde for MPO
    ├── environments/
    │   ├── mod.rs              Environments<T, Q>: left + right environment stacks
    │   ├── build.rs            initial left/right environment construction
    │   └── update.rs           incremental left/right environment renewal
    ├── sweep/
    │   ├── mod.rs              DMRGEngine<T, Q, B>, DMRGConfig
    │   ├── two_site.rs         two-site DMRG step
    │   ├── single_site.rs      single-site DMRG step
    │   ├── schedule.rs         SweepSchedule: site ordering, direction, warmup
    │   └── stats.rs            DMRGStats: per-sweep energy, truncation error, wall time
    ├── eigensolver/
    │   ├── mod.rs              IterativeEigensolver<T> trait, EigenResult<T>
    │   ├── lanczos.rs          LanczosSolver
    │   ├── davidson.rs         DavidsonSolver
    │   ├── block_davidson.rs   BlockDavidsonSolver
    │   └── krylov.rs           Krylov basis management, thick restart
    ├── truncation/
    │   ├── mod.rs              TruncationResult, truncate_svd
    │   └── schedule.rs         BondDimensionSchedule, TruncationConfig
    ├── tdvp/
    │   ├── mod.rs              TdvpDriver<T, Q, B>
    │   ├── single_site.rs      1-site TDVP step with projector splitting
    │   ├── expansion.rs        site-tensor subspace expansion (matrix-free)
    │   └── expm.rs             matrix-exponential via Krylov (exp_krylov)
    ├── excited/
    │   └── mod.rs              energy-penalization for excited-state DMRG
    ├── idmrg/
    │   └── mod.rs              infinite DMRG bootstrap
    ├── checkpoint.rs           DMRGCheckpoint: full state serialization
    └── error.rs                DmrgError, DmrgResult<T>
```

---

## 3. MPS Representation

### 3.1 Typestate Gauge Markers

```rust
/// Left-canonical form: all site tensors A[i] satisfy A[i]† A[i] = I.
/// The orthogonality center is to the right of the last site.
pub struct LeftCanonical;

/// Right-canonical form: all site tensors B[i] satisfy B[i] B[i]† = I.
/// The orthogonality center is to the left of the first site.
pub struct RightCanonical;

/// Mixed-canonical form: sites 0..center are left-canonical;
/// sites center+1..N are right-canonical. The orthogonality center
/// is at `center`. This is the required gauge for DMRG updates.
pub struct MixedCanonical {
    pub center: usize,
}

/// Bond-centered form: the singular value matrix between sites `left`
/// and `left+1` is exposed as a standalone `DenseTensor`. Required for
/// TDVP projector splitting (backward bond-matrix evolution).
pub struct BondCentered {
    pub left: usize,
}
```

### 3.2 `MPS<T, Q, Gauge>` Definition

```rust
/// Matrix Product State with typestate-enforced canonical form.
///
/// Site tensors are rank-3 `BlockSparseTensor<T, Q>` with legs ordered
/// (physical, bond-left, bond-right). Bond dimensions may vary per site.
///
/// # Type Parameters
/// - `T`: scalar type (`f64`, `Complex<f64>`, etc.)
/// - `Q`: quantum number type (`U1`, `Z2`, `U1Z2`, etc.)
/// - `Gauge`: typestate marker encoding the current canonical form
pub struct MPS<T: Scalar, Q: BitPackable, Gauge> {
    /// Site tensors. Leg ordering: (σ, α_left, α_right).
    /// σ ∈ 0..local_dim[site]; α_left/α_right are bond indices.
    tensors: Vec<BlockSparseTensor<T, Q>>,
    /// Bond matrices (singular value diagonal tensors) for BondCentered form.
    /// None in all other gauge forms.
    bonds: Option<Vec<DenseTensor<T>>>,
    /// Physical (local Hilbert space) dimension per site.
    local_dims: Vec<usize>,
    /// Total charge of the MPS (target quantum number sector).
    total_charge: Q,
    _gauge: PhantomData<Gauge>,
}
```

### 3.3 Constructors

```rust
impl<T: Scalar, Q: BitPackable> MPS<T, Q, MixedCanonical> {
    /// Construct a random initial MPS in mixed-canonical form.
    ///
    /// The MPS is initialized with a product state consistent with
    /// `total_charge`, then brought to mixed-canonical form at `center`
    /// by a single left-to-right followed by a right-to-left sweep.
    ///
    /// # Parameters
    /// - `n_sites`: number of lattice sites
    /// - `local_dims`: physical dimension of each site's Hilbert space
    /// - `max_bond_dim`: maximum initial bond dimension
    /// - `total_charge`: target quantum number sector (e.g., `U1(n_electrons)`)
    /// - `center`: initial orthogonality center site
    /// - `rng`: random number generator for initial state
    ///
    /// # Errors
    /// Returns `DmrgError::ChargeSectorEmpty` if `total_charge` is unreachable
    /// given `local_dims` (e.g., requesting N electrons on N/2 sites with d=2).
    pub fn random(
        n_sites: usize,
        local_dims: &[usize],
        max_bond_dim: usize,
        total_charge: Q,
        center: usize,
        rng: &mut impl rand::Rng,
    ) -> DmrgResult<Self>;

    /// Construct the vacuum (all-zero) product state.
    /// Bond dimensions are all 1. Useful as a starting point for DMRG.
    ///
    /// # Parameters
    /// - `local_dims`: physical dimension per site
    /// - `site_charges`: quantum number at each site; must fuse to `total_charge`
    ///
    /// # Errors
    /// Returns `DmrgError::ChargeSectorEmpty` if `site_charges` does not
    /// fuse to a valid total charge.
    pub fn product_state(
        local_dims: &[usize],
        site_charges: &[Q],
    ) -> DmrgResult<Self>;
}
```

### 3.4 Core MPS Methods

```rust
impl<T: Scalar, Q: BitPackable, Gauge> MPS<T, Q, Gauge> {
    pub fn n_sites(&self) -> usize;
    pub fn local_dim(&self, site: usize) -> usize;
    pub fn total_charge(&self) -> &Q;

    /// Bond dimension at the bond between sites `site` and `site+1`.
    pub fn bond_dim(&self, site: usize) -> usize;

    /// Maximum bond dimension across all bonds.
    pub fn max_bond_dim(&self) -> usize;

    /// Von Neumann entanglement entropy at the bond between `site` and `site+1`.
    /// S = -Σ_i σ_i² log(σ_i²) where σ_i are the Schmidt values.
    ///
    /// # Errors
    /// Returns `DmrgError::BondSingularValuesUnavailable` if the MPS is not in
    /// BondCentered form and the singular values have not been cached.
    pub fn entanglement_entropy(&self, site: usize) -> DmrgResult<T::Real>;

    /// Returns a reference to the site tensor at `site`.
    pub fn site_tensor(&self, site: usize) -> &BlockSparseTensor<T, Q>;

    /// Norm ||ψ||. For a properly canonicalized MPS this is 1.0.
    pub fn norm(&self) -> T::Real;
}

impl<T: Scalar, Q: BitPackable> MPS<T, Q, MixedCanonical> {
    /// Move the orthogonality center to `new_center` via QR/LQ factorizations.
    /// O(D³) per intermediate site.
    ///
    /// # Errors
    /// Returns `DmrgError::SiteBoundsError` if `new_center >= n_sites`.
    pub fn shift_center(self, new_center: usize) -> DmrgResult<Self>;

    /// Update the site tensor at `site`.
    ///
    /// # Errors
    /// Returns `DmrgError::ShapeMismatch` if the new tensor's bond dimensions
    /// do not match adjacent site tensors.
    pub fn set_site_tensor(
        &mut self,
        site: usize,
        tensor: BlockSparseTensor<T, Q>,
    ) -> DmrgResult<()>;

    /// Expose the bond matrix between sites `center` and `center+1`.
    /// Computes SVD of the center site tensor, storing U as the left site
    /// tensor and V† as the right site tensor.
    ///
    /// # Errors
    /// Returns `DmrgError::Linalg` if the internal SVD fails.
    pub fn expose_bond(self) -> DmrgResult<MPS<T, Q, BondCentered>>;
}

impl<T: Scalar, Q: BitPackable> MPS<T, Q, BondCentered> {
    /// Return a reference to the exposed bond matrix.
    pub fn bond_matrix(&self) -> &DenseTensor<T>;

    /// Absorb the bond matrix back into the left site tensor,
    /// returning to MixedCanonical form with center = `self.left`.
    pub fn absorb_bond(self) -> MPS<T, Q, MixedCanonical>;
}

impl<T: Scalar, Q: BitPackable> MPS<T, Q, LeftCanonical> {
    /// Convert to MixedCanonical by right-to-left QR sweep down to `center`.
    pub fn into_mixed_canonical(
        self,
        center: usize,
    ) -> DmrgResult<MPS<T, Q, MixedCanonical>>;
}

impl<T: Scalar, Q: BitPackable> MPS<T, Q, RightCanonical> {
    /// Convert to MixedCanonical by left-to-right QR sweep up to `center`.
    pub fn into_mixed_canonical(
        self,
        center: usize,
    ) -> DmrgResult<MPS<T, Q, MixedCanonical>>;
}
```

### 3.5 Canonical Form Free Functions

```rust
/// Left-canonicalize the MPS using QR factorizations from left to right.
/// After this call, all site tensors satisfy A†A = I.
///
/// # Complexity
/// O(N · d · D³)
pub fn left_canonicalize<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>(
    mps: MPS<T, Q, MixedCanonical>,
    backend: &B,
) -> DmrgResult<MPS<T, Q, LeftCanonical>>;

/// Right-canonicalize the MPS using LQ factorizations from right to left.
pub fn right_canonicalize<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>(
    mps: MPS<T, Q, MixedCanonical>,
    backend: &B,
) -> DmrgResult<MPS<T, Q, RightCanonical>>;

/// Bring MPS to mixed-canonical form at `center` by QR from both ends.
pub fn mixed_canonicalize<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>(
    mps: MPS<T, Q, MixedCanonical>,
    center: usize,
    backend: &B,
) -> DmrgResult<MPS<T, Q, MixedCanonical>>;
```

### 3.6 Overlap and Expectation Values

```rust
/// Compute the inner product <ψ|φ> between two MPS.
///
/// Uses boundary-vector contraction from the left boundary. O(N · d · D³).
/// Not a hot-path routine; intended for gauge-invariant tests.
pub fn mps_overlap<T: Scalar, Q: BitPackable, GaugeA, GaugeB>(
    bra: &MPS<T, Q, GaugeA>,
    ket: &MPS<T, Q, GaugeB>,
) -> T;

/// Compute the energy expectation value <ψ|H|ψ> / <ψ|ψ>.
pub fn mps_energy<T: Scalar, Q: BitPackable>(
    mps: &MPS<T, Q, MixedCanonical>,
    mpo: &MPO<T, Q>,
) -> DmrgResult<T::Real>;

/// Compute the energy variance <H²> - <H>² for convergence assessment.
/// A well-converged ground state has variance < energy_tol².
///
/// # Complexity
/// O(N · d · D³) — one additional sweep applying H twice.
pub fn energy_variance<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>(
    mps: &MPS<T, Q, MixedCanonical>,
    mpo: &MPO<T, Q>,
    backend: &B,
) -> DmrgResult<T::Real>;
```

---

## 4. MPO Representation

### 4.1 `MPO<T, Q>` Definition

```rust
/// Matrix Product Operator representation of a quantum Hamiltonian.
///
/// Each site tensor has rank 4 with leg ordering (σ_in, σ_out, w_left, w_right),
/// where σ_in/σ_out are physical (bra/ket) legs and w_left/w_right are MPO
/// bond (auxiliary) legs.
///
/// The left boundary tensor is (1 × w_right) and the right boundary is
/// (w_left × 1), consistent with the finite-state automaton construction.
pub struct MPO<T: Scalar, Q: BitPackable> {
    /// Site tensors. Leg ordering: (σ_in, σ_out, w_left, w_right).
    tensors: Vec<BlockSparseTensor<T, Q>>,
    /// Physical dimensions per site.
    local_dims: Vec<usize>,
    /// Maximum MPO bond dimension.
    max_bond_dim: usize,
    /// Total charge of the operator. Q::identity() for Hermitian Hamiltonians.
    flux: Q,
}

impl<T: Scalar, Q: BitPackable> MPO<T, Q> {
    pub fn n_sites(&self) -> usize;
    pub fn local_dim(&self, site: usize) -> usize;
    pub fn mpo_bond_dim(&self, site: usize) -> usize;
    pub fn max_mpo_bond_dim(&self) -> usize;
    pub fn site_tensor(&self, site: usize) -> &BlockSparseTensor<T, Q>;

    /// Construct the identity MPO (all MPO bond dims = 1).
    pub fn identity(local_dims: Vec<usize>, flux: Q) -> Self;

    /// Add a scalar multiple of another MPO: H_new = self + alpha * other.
    /// The result has bond dim = bond_dim(self) + bond_dim(other).
    /// Use `MpoCompiler::compress` to reduce afterward.
    pub fn add_scaled(&self, alpha: T, other: &MPO<T, Q>) -> Self;
}
```

### 4.2 `MpoCompiler` and `OpSum` Compilation

```rust
/// Configuration for OpSum → MPO compilation.
///
/// Compilation proceeds via the finite-state automaton (FSA) representation:
/// the `OpSum` is translated into an FSA whose transfer matrix at each site
/// defines the MPO site tensor. SVD compression then reduces the MPO bond
/// dimension to `max_bond_dim`.
#[derive(Clone, Debug)]
pub struct MpoCompressionConfig {
    /// Maximum MPO bond dimension after SVD compression.
    /// Heisenberg chain: exact bond dim = 5; Hubbard: ~8; long-range: ~50+.
    pub max_bond_dim: usize,
    /// SVD cutoff for discarding small singular values. Default: 1e-12.
    pub svd_cutoff: f64,
    /// If true (debug builds), verify ||H_compressed - H_full|| / ||H_full|| < compression_tol.
    pub validate_compression: bool,
    /// Compression error tolerance. Default: 1e-8.
    pub compression_tol: f64,
}

/// Compiles an `OpSum<T>` from `tk-dsl` into an `MPO<T, Q>`.
///
/// This is the only crate that performs this compilation. `tk-dsl` is
/// strictly forbidden from depending on `tk-linalg` (design doc §2.2).
///
/// # Example
/// ```rust
/// let opsum = hamiltonian! { /* ... */ };
/// let mpo: MPO<f64, U1> = MpoCompiler::new(&backend, MpoCompressionConfig {
///     max_bond_dim: 50,
///     svd_cutoff: 1e-12,
///     validate_compression: cfg!(debug_assertions),
///     compression_tol: 1e-8,
/// }).compile(&opsum, 100, &vec![2; 100], U1(0))?;
/// ```
pub struct MpoCompiler<'b, T: Scalar, B: LinAlgBackend<T>> {
    backend: &'b B,
    config: MpoCompressionConfig,
    _phantom: PhantomData<T>,
}

impl<'b, T: Scalar, B: LinAlgBackend<T>> MpoCompiler<'b, T, B> {
    pub fn new(backend: &'b B, config: MpoCompressionConfig) -> Self;

    /// Compile an `OpSum` into an `MPO`.
    ///
    /// # Parameters
    /// - `opsum`: operator sum from `tk-dsl`
    /// - `n_sites`: number of lattice sites
    /// - `local_dims`: physical dimension per site
    /// - `flux`: total charge of the operator (Q::identity() for Hamiltonians)
    ///
    /// # Errors
    /// - `DmrgError::OpSumCompilationFailed` if FSA construction fails
    /// - `DmrgError::Linalg` wrapping SVD error on ill-conditioned inputs
    pub fn compile(
        &self,
        opsum: &OpSum<T>,
        n_sites: usize,
        local_dims: &[usize],
        flux: Q,
    ) -> DmrgResult<MPO<T, Q>>
    where
        Q: BitPackable;

    /// Re-compress an existing MPO with tighter bond dimension.
    /// Used when a sum of MPOs is built via `MPO::add_scaled`.
    pub fn compress(&self, mpo: MPO<T, Q>) -> DmrgResult<MPO<T, Q>>
    where
        Q: BitPackable;
}
```

---

## 5. Environment Blocks

### 5.1 `Environment<T, Q>` — Single Block

```rust
/// One environment tensor: the partial contraction of the MPS–MPO–MPS
/// sandwich network from one boundary up to (but not including) the
/// current DMRG site.
///
/// Leg ordering: (MPS-bra bond, MPO bond, MPS-ket bond). Rank 3.
/// Stored as `BlockSparseTensor<T, Q>`.
///
/// For the leftmost environment (before site 0) and rightmost environment
/// (after site N-1), all bond dimensions equal 1 and the single element
/// is T::one().
pub struct Environment<T: Scalar, Q: BitPackable> {
    tensor: BlockSparseTensor<T, Q>,
    /// Site index this environment reaches up to (exclusive).
    up_to: usize,
}

impl<T: Scalar, Q: BitPackable> Environment<T, Q> {
    pub fn left_boundary() -> Self;
    pub fn right_boundary(n_sites: usize) -> Self;
    pub fn up_to(&self) -> usize;
    pub fn tensor(&self) -> &BlockSparseTensor<T, Q>;
}
```

### 5.2 `Environments<T, Q>` — Stack-Based Caching

```rust
/// Cached environment stack for an MPS–MPO–MPS sandwich.
///
/// During a left-to-right sweep, left environments are grown incrementally
/// (pushed) and right environments are consumed (popped). The inverse
/// holds during a right-to-left sweep.
///
/// Memory: O(N · D² · w) where D = MPS bond dim, w = MPO bond dim.
pub struct Environments<T: Scalar, Q: BitPackable> {
    left_envs: Vec<Environment<T, Q>>,
    right_envs: Vec<Environment<T, Q>>,
    n_sites: usize,
}

impl<T: Scalar, Q: BitPackable> Environments<T, Q> {
    /// Construct full environment stack by a left-to-right pass.
    ///
    /// # Complexity
    /// O(N · d · D² · w)
    pub fn build_from_scratch<B: LinAlgBackend<T>>(
        mps: &MPS<T, Q, MixedCanonical>,
        mpo: &MPO<T, Q>,
        backend: &B,
    ) -> DmrgResult<Self>;

    /// Return the left environment covering sites 0..site.
    pub fn left(&self, site: usize) -> &Environment<T, Q>;

    /// Return the right environment covering sites site..N.
    pub fn right(&self, site: usize) -> &Environment<T, Q>;

    /// Update the left environment to cover 0..site+1 by contracting
    /// site tensor `site` with the current left environment and MPO site tensor.
    ///
    /// # Complexity
    /// O(d · D² · w) per site.
    pub fn grow_left<B: LinAlgBackend<T>>(
        &mut self,
        site: usize,
        mps: &MPS<T, Q, MixedCanonical>,
        mpo: &MPO<T, Q>,
        backend: &B,
    ) -> DmrgResult<()>;

    /// Update the right environment to cover site..N.
    pub fn grow_right<B: LinAlgBackend<T>>(
        &mut self,
        site: usize,
        mps: &MPS<T, Q, MixedCanonical>,
        mpo: &MPO<T, Q>,
        backend: &B,
    ) -> DmrgResult<()>;
}
```

### 5.3 Effective Hamiltonian Construction

```rust
/// Construct the two-site effective Hamiltonian as a zero-allocation closure.
///
/// Returns a `Fn(&[T], &mut [T])` closure computing H_eff · |v⟩ = |w⟩.
/// The closure captures references to `env_l`, `env_r`, and MPO site tensors.
///
/// All intermediate contraction buffers are pre-allocated from `arena`
/// before the closure is returned. Inside the Krylov loop the closure
/// performs no heap allocation.
///
/// # Returns
/// `(matvec, dim)` where `dim = d² · D_left · D_right` is the effective
/// Hilbert space dimension.
///
/// # Errors
/// Returns `DmrgError::Core` if arena allocation fails.
pub fn build_heff_two_site<'arena, T, Q, B>(
    env_l: &Environment<T, Q>,
    env_r: &Environment<T, Q>,
    mpo: &MPO<T, Q>,
    sites: (usize, usize),
    arena: &'arena mut SweepArena,
    backend: &B,
) -> DmrgResult<(impl Fn(&[T], &mut [T]) + 'arena, usize)>
where
    T: Scalar,
    Q: BitPackable,
    B: LinAlgBackend<T>;

/// Single-site variant of `build_heff_two_site`.
/// `dim = d · D_left · D_right`.
pub fn build_heff_single_site<'arena, T, Q, B>(
    env_l: &Environment<T, Q>,
    env_r: &Environment<T, Q>,
    mpo: &MPO<T, Q>,
    site: usize,
    arena: &'arena mut SweepArena,
    backend: &B,
) -> DmrgResult<(impl Fn(&[T], &mut [T]) + 'arena, usize)>
where
    T: Scalar,
    Q: BitPackable,
    B: LinAlgBackend<T>;
```

---

## 6. Iterative Eigensolvers

### 6.1 `IterativeEigensolver<T>` Trait

```rust
/// Trait for in-place iterative eigensolvers.
///
/// All implementations must:
/// - Accept a zero-allocation matvec closure (no allocation inside Krylov loop)
/// - Manage all Krylov workspace internally
/// - Support thick restarts when subspace is exhausted
/// - Support warm-start initial vectors or subspace bases
///
/// The trait is object-safe: `Box<dyn IterativeEigensolver<f64>>` is valid.
pub trait IterativeEigensolver<T: Scalar>: Send + Sync {
    /// Find the lowest eigenvalue and eigenvector.
    ///
    /// # Parameters
    /// - `matvec`: reads `x`, writes `y = A·x` in place; called repeatedly
    /// - `dim`: dimension of the vector space
    /// - `initial`: initial guess for warm start
    ///
    /// # Returns
    /// `EigenResult` with `converged = false` and best approximation if
    /// `max_iter` is exhausted without convergence.
    fn lowest_eigenpair(
        &self,
        matvec: &dyn Fn(&[T], &mut [T]),
        dim: usize,
        initial: InitialSubspace<'_, T>,
    ) -> EigenResult<T>;

    /// Find the lowest `k` eigenpairs.
    ///
    /// Default implementation calls `lowest_eigenpair` k times with energy
    /// penalization. Override in `BlockDavidsonSolver` for simultaneous computation.
    fn lowest_k_eigenpairs(
        &self,
        matvec: &dyn Fn(&[T], &mut [T]),
        dim: usize,
        k: usize,
        initial: InitialSubspace<'_, T>,
    ) -> Vec<EigenResult<T>>;
}

/// Initial subspace specification for warm starts and thick restarts.
pub enum InitialSubspace<'a, T: Scalar> {
    /// No prior information; start from a random unit vector.
    None,
    /// Single initial guess (standard warm start from previous sweep step).
    SingleVector(&'a [T]),
    /// Dense basis for Block-Davidson thick restart.
    SubspaceBasis { vectors: &'a [&'a [T]], num_vectors: usize },
}

/// Result of one eigensolver call.
pub struct EigenResult<T: Scalar> {
    /// The lowest eigenvalue found.
    pub eigenvalue: T::Real,
    /// Corresponding eigenvector (normalized).
    pub eigenvector: Vec<T>,
    /// Whether the solver converged within `max_iter`.
    pub converged: bool,
    /// Total number of matvec products performed.
    pub matvec_count: usize,
    /// Residual norm ||A·v − λ·v|| at termination.
    pub residual_norm: T::Real,
}
```

### 6.2 `LanczosSolver`

```rust
/// Lanczos eigensolver with full reorthogonalization and thick restarts.
///
/// Builds a Krylov subspace K_m(A, v₀) and diagonalizes the tridiagonal
/// projection. Full reorthogonalization (modified Gram-Schmidt against all
/// previous vectors) prevents ghost eigenvalue artifacts from floating-point
/// loss of orthogonality at cost O(m²) per iteration.
///
/// A thick restart collapses the subspace to `restart_vectors` best Ritz
/// vectors when `max_krylov_dim` is reached.
pub struct LanczosSolver {
    /// Maximum Krylov dimension before thick restart. Default: 100.
    pub max_krylov_dim: usize,
    /// Ritz vectors retained across a thick restart. Default: 5.
    pub restart_vectors: usize,
    /// Maximum total matvec products. Default: 1000.
    pub max_iter: usize,
    /// Convergence tolerance on ||A·v − λ·v||. Default: 1e-10.
    pub tol: f64,
}

impl<T: Scalar> IterativeEigensolver<T> for LanczosSolver
where
    T::Real: Into<f64> + From<f64>,
{
    fn lowest_eigenpair(
        &self,
        matvec: &dyn Fn(&[T], &mut [T]),
        dim: usize,
        initial: InitialSubspace<'_, T>,
    ) -> EigenResult<T>;
}
```

### 6.3 `DavidsonSolver`

```rust
/// Davidson eigensolver with diagonal preconditioner and thick restarts.
///
/// Applies the diagonal preconditioner update:
///   δv_i = -r_i / (D_ii − λ)
/// where r is the residual and D is the diagonal of A. Converges faster
/// than Lanczos when H_eff is diagonal-dominant (typical for local
/// Hamiltonians in the local Fock basis).
pub struct DavidsonSolver {
    /// Maximum Krylov subspace dimension. Default: 60.
    pub max_subspace: usize,
    /// Ritz vectors retained on thick restart. Default: 5.
    pub restart_vectors: usize,
    /// Maximum total matvec products. Default: 1000.
    pub max_iter: usize,
    /// Convergence tolerance. Default: 1e-10.
    pub tol: f64,
    /// Diagonal of H_eff for preconditioning.
    /// Set by `DMRGEngine` before each call.
    /// If `None`, falls back to unpreconditioned update (gradient descent).
    pub diagonal: Option<Vec<f64>>,
}

impl<T: Scalar> IterativeEigensolver<T> for DavidsonSolver
where
    T::Real: Into<f64> + From<f64>,
{
    fn lowest_eigenpair(
        &self,
        matvec: &dyn Fn(&[T], &mut [T]),
        dim: usize,
        initial: InitialSubspace<'_, T>,
    ) -> EigenResult<T>;
}
```

### 6.4 `BlockDavidsonSolver`

```rust
/// Block-Davidson eigensolver for simultaneous multi-state targeting.
///
/// Converts memory-bound single-vector dgemv (BLAS Level 2) into
/// compute-bound batched dgemm (BLAS Level 3) by operating on a block
/// of `block_size` vectors simultaneously. Used for excited-state DMRG
/// targeting the lowest k eigenstates (k = block_size).
pub struct BlockDavidsonSolver {
    /// Number of target eigenstates (block size). Default: 1.
    pub block_size: usize,
    /// Maximum Krylov subspace dimension. Default: 80.
    pub max_subspace: usize,
    /// Ritz vectors retained on thick restart. Default: block_size * 2.
    pub restart_vectors: usize,
    /// Maximum total matvec products. Default: 2000.
    pub max_iter: usize,
    /// Convergence tolerance. Default: 1e-10.
    pub tol: f64,
}

impl<T: Scalar> IterativeEigensolver<T> for BlockDavidsonSolver
where
    T::Real: Into<f64> + From<f64>,
{
    fn lowest_eigenpair(
        &self,
        matvec: &dyn Fn(&[T], &mut [T]),
        dim: usize,
        initial: InitialSubspace<'_, T>,
    ) -> EigenResult<T>;

    /// Simultaneous computation of `k = block_size` eigenstates.
    fn lowest_k_eigenpairs(
        &self,
        matvec: &dyn Fn(&[T], &mut [T]),
        dim: usize,
        k: usize,
        initial: InitialSubspace<'_, T>,
    ) -> Vec<EigenResult<T>>;
}
```

---

## 7. SVD Truncation

### 7.1 Truncation Types

```rust
/// Result of a single SVD truncation step at a bond.
pub struct TruncationResult<T: Scalar> {
    /// Left isometric factor U (shape: D_left × D_new).
    pub u: DenseTensor<T>,
    /// Retained singular values, descending.
    pub singular_values: Vec<T::Real>,
    /// Right isometric factor Vt (shape: D_new × D_right).
    pub vt: DenseTensor<T>,
    /// Number of singular values retained.
    pub bond_dim_new: usize,
    /// Number of singular values discarded.
    pub n_discarded: usize,
    /// Truncation error: Σ_discarded σ_i² / Σ_all σ_i².
    /// 0.0 means no truncation occurred.
    pub truncation_error: T::Real,
}

/// Configuration for one truncation step.
#[derive(Clone, Debug)]
pub struct TruncationConfig {
    /// Hard upper bound on retained bond dimension.
    pub max_bond_dim: usize,
    /// Discard singular values below this absolute threshold. Default: 1e-12.
    pub svd_cutoff: f64,
    /// Minimum singular values to retain regardless of `svd_cutoff`.
    /// Prevents bond dimension collapsing to zero. Default: 1.
    pub min_bond_dim: usize,
}
```

### 7.2 `truncate_svd`

```rust
/// Perform a truncated SVD on a rank-2 matrix view.
///
/// Delegates to `LinAlgBackend::svd_truncated` (gesdd default, gesvd fallback)
/// and applies the truncation criteria from `config`.
///
/// Under the `su2-symmetry` feature flag, truncation is multiplet-aware:
/// - Phase 1: sort all singular values by magnitude descending
/// - Phase 2: snap the truncation boundary to the nearest multiplet edge
///   (never discard fewer than 2j+1 values from a multiplet)
/// - Truncation error weighted by (2j+1)·σ_i² per discarded value
/// See design doc §4.4.
///
/// # Errors
/// Returns `DmrgError::Linalg` wrapping `LinAlgError::SvdConvergenceError`
/// if both gesdd and gesvd fail to converge.
pub fn truncate_svd<T: Scalar, B: LinAlgBackend<T>>(
    matrix: &MatRef<T>,
    config: &TruncationConfig,
    backend: &B,
) -> DmrgResult<TruncationResult<T>>;
```

### 7.3 `BondDimensionSchedule`

```rust
/// Schedule for ramping bond dimension across DMRG sweeps.
///
/// Starting from a small `D_init` avoids local minima early in the sweep
/// sequence. The final target `D_max` is reached after `n_warmup_sweeps`.
///
/// # Example
/// ```rust
/// let sched = BondDimensionSchedule::warmup(64, 1000, 5);
/// // Returns: [64, 128, 256, 512, 1000, 1000, ...]
/// assert_eq!(sched.bond_dim_at_sweep(0), 64);
/// assert_eq!(sched.bond_dim_at_sweep(4), 1000);
/// assert_eq!(sched.bond_dim_at_sweep(100), 1000);
/// ```
pub struct BondDimensionSchedule {
    dims: Vec<usize>,  // dims[sweep] = bond dim for that sweep
}

impl BondDimensionSchedule {
    /// Geometrically ramped warmup schedule.
    pub fn warmup(d_init: usize, d_max: usize, n_warmup_sweeps: usize) -> Self;
    /// Fixed bond dimension for all sweeps.
    pub fn fixed(d: usize) -> Self;
    /// Custom sequence; last entry is used for all subsequent sweeps.
    pub fn custom(dims: Vec<usize>) -> Self;
    /// Bond dimension to use at sweep `sweep_index`.
    pub fn bond_dim_at_sweep(&self, sweep_index: usize) -> usize;
}
```

---

## 8. DMRG Sweep Engine

### 8.1 `DMRGConfig`

```rust
/// Configuration for a DMRG simulation run.
#[derive(Clone, Debug)]
pub struct DMRGConfig {
    /// Bond dimension schedule across sweeps.
    pub bond_dim_schedule: BondDimensionSchedule,
    /// SVD cutoff for MPS truncation. Default: 1e-12.
    pub svd_cutoff: f64,
    /// Maximum number of full sweeps (one L→R + one R→L = one full sweep).
    /// Default: 20.
    pub max_sweeps: usize,
    /// Convergence criterion: relative energy change between sweeps.
    /// Default: 1e-10.
    pub energy_tol: f64,
    /// Optional energy variance convergence criterion.
    /// DMRG also requires <H²> - <H>² < variance_tol² when set.
    /// Default: None (disabled).
    pub variance_tol: Option<f64>,
    /// Eigensolver for H_eff diagonalization.
    /// Default: `Box::new(DavidsonSolver::default())`.
    pub eigensolver: Box<dyn IterativeEigensolver<f64>>,
    /// If true, run iDMRG warm-start before finite sweeps. Default: false.
    pub idmrg_warmup: bool,
    /// iDMRG configuration, used only when `idmrg_warmup = true`.
    pub idmrg_config: IDmrgConfig,
    /// Two-site (default) or single-site updates.
    pub update_variant: UpdateVariant,
    /// Checkpoint file path. Written after each full sweep if set.
    pub checkpoint_path: Option<std::path::PathBuf>,
    /// Number of target eigenstates for excited-state DMRG.
    /// Default: None (ground state only).
    pub n_target_states: Option<usize>,
    /// Energy penalty weight for excited-state targeting.
    /// Default: 0.1 (in units of the energy gap).
    pub excited_state_weight: f64,
}

/// Whether to use two-site or single-site DMRG updates.
#[derive(Clone, Debug, Default)]
pub enum UpdateVariant {
    /// Two-site update: diagonalizes a rank-4 two-site tensor.
    /// Allows bond dimension growth. Required for the first sweeps.
    #[default]
    TwoSite,
    /// Single-site update: diagonalizes a rank-3 single-site tensor.
    /// Does not allow bond dimension growth; faster per step.
    SingleSite,
}
```

### 8.2 `DMRGEngine<T, Q, B>`

```rust
/// The main DMRG sweep engine.
///
/// # Type Parameters
/// - `T`: scalar type
/// - `Q`: quantum number type (use `NoQ` from `tk-contract` for no-symmetry models)
/// - `B`: linear algebra backend
///
/// The engine always holds the MPS in `MixedCanonical` form. The gauge
/// center moves by one site after each sweep step. `BondCentered` form is
/// used internally by `TdvpDriver` and is transparent to `DMRGEngine` users.
pub struct DMRGEngine<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> {
    pub mps: MPS<T, Q, MixedCanonical>,
    pub mpo: MPO<T, Q>,
    pub environments: Environments<T, Q>,
    pub backend: B,
    pub config: DMRGConfig,
    pub stats: DMRGStats,
    arena: SweepArena,
}

impl<T, Q, B> DMRGEngine<T, Q, B>
where
    T: Scalar,
    Q: BitPackable,
    B: LinAlgBackend<T>,
{
    /// Construct the engine from a pre-built MPS and MPO.
    /// Builds all environments in O(N · d · D² · w).
    ///
    /// # Errors
    /// Returns `DmrgError::DimensionMismatch` if MPS and MPO site counts differ.
    pub fn new(
        mps: MPS<T, Q, MixedCanonical>,
        mpo: MPO<T, Q>,
        backend: B,
        config: DMRGConfig,
    ) -> DmrgResult<Self>;

    /// Run DMRG sweeps until convergence or `max_sweeps`.
    ///
    /// Returns the converged ground state energy.
    ///
    /// # Errors
    /// Returns `DmrgError::EigensolverNotConverged` if any eigensolver call
    /// fails within its `max_iter`.
    pub fn run(&mut self) -> DmrgResult<T::Real>;

    /// Run with an `AtomicBool` cancellation flag.
    /// Flag is checked after each sweep step (single `Relaxed` load).
    /// Returns `DmrgError::Cancelled` if the flag is set mid-run.
    pub fn run_with_cancel_flag(
        &mut self,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> DmrgResult<T::Real>;

    /// Execute one two-site DMRG step at `site`.
    ///
    /// Ownership boundary: SVD outputs (`al`, `ar`) must call `.into_owned()`
    /// before `self.arena.reset()` at the end of this method.
    /// The borrow checker enforces this statically.
    ///
    /// Steps:
    /// 1. Merge site tensors `site` and `site+1` into a two-site tensor Θ
    /// 2. Pre-allocate H_eff contraction buffers from `self.arena`
    /// 3. Build H_eff closure (zero-allocation)
    /// 4. Diagonalize H_eff → ground state vector via eigensolver
    /// 5. SVD-truncate ground state into A_L, S, A_R
    /// 6. `.into_owned()` on A_L and A_R (arena → heap ownership transfer)
    /// 7. Store A_L/A_R back into `self.mps`
    /// 8. Grow environment in sweep direction
    /// 9. `self.arena.reset()`
    pub fn dmrg_step_two_site(
        &mut self,
        site: usize,
        direction: SweepDirection,
    ) -> DmrgResult<StepResult<T>>;

    /// Execute one single-site DMRG step at `site`.
    pub fn dmrg_step_single_site(
        &mut self,
        site: usize,
        direction: SweepDirection,
    ) -> DmrgResult<StepResult<T>>;

    /// Current energy estimate (eigenvalue from most recent eigensolver call).
    pub fn energy(&self) -> T::Real;

    /// Whether the engine has converged per its configured criteria.
    pub fn converged(&self) -> bool;

    /// Reset accumulated statistics and clear converged flag.
    pub fn reset_stats(&mut self);

    /// Restore engine state from a checkpoint file.
    pub fn load_checkpoint(
        path: &std::path::Path,
        backend: B,
        config: DMRGConfig,
    ) -> DmrgResult<Self>;
}
```

### 8.3 `SweepSchedule` and `SweepDirection`

```rust
/// Direction of sweep travel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepDirection {
    LeftToRight,
    RightToLeft,
}

/// Ordered list of (site, direction) pairs defining one full DMRG sweep.
pub struct SweepSchedule {
    n_sites: usize,
    lr_sites: Vec<usize>,  // sites visited during L→R pass
    rl_sites: Vec<usize>,  // sites visited during R→L pass
}

impl SweepSchedule {
    /// Standard: sites 0..N-2 L→R, then N-2..=0 R→L.
    pub fn standard(n_sites: usize) -> Self;
    /// Iterator of (site, direction) for one full sweep.
    pub fn iter(&self) -> impl Iterator<Item = (usize, SweepDirection)>;
}

/// Per-step statistics returned by each DMRG step.
pub struct StepResult<T: Scalar> {
    pub site: usize,
    pub direction: SweepDirection,
    pub energy: T::Real,
    pub truncation_error: T::Real,
    pub bond_dim_new: usize,
    pub eigensolver_converged: bool,
    pub eigensolver_iters: usize,
}
```

### 8.4 `DMRGStats`

```rust
/// Statistics accumulated across DMRG sweeps.
#[derive(Clone, Debug, Default)]
pub struct DMRGStats {
    /// Energy at the end of each full sweep.
    pub sweep_energies: Vec<f64>,
    /// Maximum truncation error per sweep.
    pub max_truncation_errors: Vec<f64>,
    /// Maximum bond dimension per sweep.
    pub max_bond_dims: Vec<usize>,
    /// Wall-clock seconds per sweep.
    pub sweep_times_secs: Vec<f64>,
    /// Total eigensolver calls.
    pub total_eigensolver_calls: usize,
    /// Calls that did not converge.
    pub unconverged_eigensolver_calls: usize,
    /// Number of times pinned memory allocation fell back to pageable memory.
    /// Non-zero values indicate degraded GPU DMA bandwidth.
    #[cfg(feature = "backend-cuda")]
    pub pinned_memory_fallbacks: usize,
}
```

---

## 9. TDVP Time Evolution

### 9.1 Overview

The TDVP driver performs real-time (or imaginary-time) evolution of an MPS. It is used by `tk-dmft` to compute real-frequency Green's functions G(t), whose Fourier transform yields A(ω). The 1-site TDVP projector-splitting scheme is specified in design doc §8.1 and §8.1.1.

The forward site evolution computes `exp(-i H_eff dt/2) · |ψ_site⟩`. The backward bond evolution computes `exp(+i H_bond dt/2) · |S⟩` where S is the exposed bond matrix. Both use the Krylov matrix-exponential approximation (`exp_krylov`).

### 9.2 `TdvpStabilizationConfig`

```rust
/// Configuration for TDVP numerical stabilization.
///
/// Two complementary strategies:
/// 1. Tikhonov regularization: s / (s² + δ²) prevents NaN from s → 0
/// 2. Site-tensor subspace expansion: injects null-space vectors to grow D
///
/// See design doc §8.1.1 for full derivation, scaling analysis, and the
/// soft D_max bond-dimension oscillation prevention policy.
#[derive(Clone, Debug)]
pub struct TdvpStabilizationConfig {
    /// Tikhonov regularization parameter δ.
    /// Range: 1e-8 to 1e-12. Default: 1e-10.
    pub tikhonov_delta: f64,
    /// Null-space vectors injected per expansion step. Default: 4.
    pub expansion_vectors: usize,
    /// Weight of null-space residual relative to existing basis. Default: 1e-4.
    pub expansion_alpha: f64,
    /// If true, dynamically switch between 1-site and 2-site TDVP. Default: true.
    pub adaptive_expansion: bool,
    /// After expansion, truncate to floor(D_max × soft_dmax_factor).
    /// Prevents bond-dimension oscillation at the truncation threshold.
    /// Default: 1.1 (allow 10% overshoot).
    pub soft_dmax_factor: f64,
    /// Decay steps for the soft D_max overshoot back to hard D_max. Default: 5.0.
    pub dmax_decay_steps: f64,
}
```

### 9.3 `TdvpDriver<T, Q, B>`

```rust
/// Stateful TDVP time-evolution driver.
///
/// `TdvpStabilizationConfig` is immutable per-run configuration.
/// `TdvpDriver` carries the mutable per-bond expansion age state for
/// the soft D_max decay policy (design doc §8.1.1).
pub struct TdvpDriver<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> {
    pub config: TdvpStabilizationConfig,
    pub engine: DMRGEngine<T, Q, B>,
    /// Per-bond expansion age. `Some(n)` = bond last expanded n steps ago.
    /// `None` = never expanded or fully decayed back to D_max.
    /// Length = n_sites - 1.
    expansion_age: Vec<Option<usize>>,
}

impl<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> TdvpDriver<T, Q, B> {
    pub fn new(engine: DMRGEngine<T, Q, B>, config: TdvpStabilizationConfig) -> Self;

    /// Perform a single 1-site TDVP time step of size `dt`.
    ///
    /// One full left-to-right and right-to-left sweep of the projector-splitting scheme:
    /// 1. Forward site evolution: exp(-i H_eff dt/2) on each site tensor
    /// 2. expose_bond → BondCentered
    /// 3. Backward bond evolution: exp(+i H_bond dt/2) via Krylov
    ///    with Tikhonov-regularized gauge restoration
    /// 4. absorb_bond → MixedCanonical
    /// 5. Subspace expansion (if `adaptive_expansion` triggers)
    /// 6. tick_expansion_age
    ///
    /// # Parameters
    /// - `dt`: time step. Real for imaginary-time evolution (T = f64);
    ///         complex for real-time evolution (T = Complex<f64>).
    /// - `hard_dmax`: absolute maximum bond dimension
    ///
    /// # Errors
    /// Returns `DmrgError::TdvpKrylovNotConverged` if the matrix-exponential
    /// Krylov solver does not converge.
    pub fn step(&mut self, dt: T, hard_dmax: usize) -> DmrgResult<TdvpStepResult<T>>;

    /// Run `n_steps` TDVP steps with optional cancellation.
    pub fn run(
        &mut self,
        n_steps: usize,
        dt: T,
        hard_dmax: usize,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> DmrgResult<Vec<TdvpStepResult<T>>>;

    /// Compute effective D_max for bond `bond` given its current expansion age.
    fn effective_dmax(&self, bond: usize, hard_dmax: usize) -> usize;

    /// Increment expansion age for all bonds; reset to None when decay is negligible.
    fn tick_expansion_age(&mut self, hard_dmax: usize);

    /// Mark bond `bond` as freshly expanded (decay counter reset to 0).
    fn mark_expanded(&mut self, bond: usize);
}

/// Per-step result of a TDVP time step.
pub struct TdvpStepResult<T: Scalar> {
    pub norm: T::Real,
    pub max_truncation_error: T::Real,
    pub max_bond_dim: usize,
    pub n_bonds_expanded: usize,
    pub wall_time_secs: f64,
}
```

### 9.4 Subspace Expansion (Matrix-Free)

```rust
/// Expand bond between sites `left` and `left+1` by injecting null-space
/// vectors from the Hamiltonian residual into A_L.
///
/// Uses matrix-free sequential projection (O(dD²)) to avoid forming the
/// explicit projector P_null = I - A_L·A_L† (O(d²D³)):
///   1. |R⟩ = H_eff · |ψ_center⟩
///   2. O = A_L† · |R⟩   (O(dD²))
///   3. |R_null⟩ = |R⟩ - A_L · O   (O(dD²))
///   4. SVD(|R_null⟩) → top D_expand left singular vectors
///   5. Pad A_L with new vectors (scaled by `config.expansion_alpha`)
///   6. Return zero-padded bond matrix of shape (D + D_expand) × (D + D_expand)
///
/// See design doc §8.1.1 for the full derivation and the O(dD²) vs O(d²D³)
/// scaling comparison.
///
/// # Panics (debug only)
/// Panics if ||<A_L | R_null>|| > 1e-10 (projection orthogonality check).
///
/// # Returns
/// Zero-padded bond matrix. Mutates A_L in `mps` in place.
pub fn expand_bond_subspace<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>(
    mps: &mut MPS<T, Q, BondCentered>,
    mpo: &MPO<T, Q>,
    env_l: &Environment<T, Q>,
    env_r: &Environment<T, Q>,
    config: &TdvpStabilizationConfig,
    backend: &B,
    arena: &mut SweepArena,
) -> DmrgResult<DenseTensor<T>>;
```

### 9.5 Matrix Exponential via Krylov

```rust
/// Compute exp(alpha * A) · v using the Arnoldi-Krylov method.
///
/// Builds a Krylov basis K_m(A, v) and evaluates exp(alpha * H_m) · e_1
/// where H_m is the m×m upper Hessenberg projection. Convergence is
/// monitored via the residual of the Arnoldi decomposition.
///
/// Used in TDVP for:
/// - Forward site evolution: alpha = -i·dt/2 (complex scalar)
/// - Backward bond evolution: alpha = +i·dt/2 (complex scalar)
/// - Imaginary-time evolution: alpha = -dt (real scalar)
///
/// # Parameters
/// - `matvec`: computes y = A·x
/// - `v`: initial vector of length `dim`
/// - `alpha`: scalar prefactor (may be complex even when T = f64 via f64 coercion)
/// - `krylov_dim`: Krylov subspace size. Default: 30.
/// - `tol`: residual convergence tolerance. Default: 1e-10.
///
/// # Errors
/// Returns `DmrgError::TdvpKrylovNotConverged` if not converged in `krylov_dim` steps.
pub fn exp_krylov<T: Scalar>(
    matvec: &dyn Fn(&[T], &mut [T]),
    v: &[T],
    alpha: T,
    dim: usize,
    krylov_dim: usize,
    tol: T::Real,
) -> DmrgResult<Vec<T>>;
```

---

## 10. Infinite DMRG Bootstrap

```rust
/// Configuration for infinite DMRG warm-start.
#[derive(Clone, Debug, Default)]
pub struct IDmrgConfig {
    /// Bond dimension target for the infinite-system phase.
    pub target_bond_dim: usize,
    /// Per-site energy convergence criterion. Default: 1e-10.
    pub energy_tol_per_site: f64,
    /// Maximum unit-cell extensions. Default: 500.
    pub max_extensions: usize,
}

/// Run infinite DMRG to generate a warm-start MPS for finite DMRG.
///
/// Grows the system two sites at a time (one unit cell). After convergence
/// to the thermodynamic-limit fixed point, maps into a finite-system initial
/// state of length `n_target_sites`.
///
/// # Returns
/// `MPS` in `MixedCanonical` form with bond dimension ≈ `config.target_bond_dim`.
///
/// # Errors
/// Returns `DmrgError::IDmrgConvergenceFailed` if the per-site energy does not
/// converge within `config.max_extensions`.
pub fn run_idmrg<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>(
    unit_cell_mpo: &MPO<T, Q>,
    n_target_sites: usize,
    config: &IDmrgConfig,
    backend: &B,
) -> DmrgResult<MPS<T, Q, MixedCanonical>>;
```

---

## 11. Excited States

```rust
/// Configuration for excited-state DMRG via energy penalization.
///
/// Penalized Hamiltonian: H_pen = H + weight × Σ_j |ψ_j⟩⟨ψ_j|
///
/// The penalty projects the ground state of H_pen above the penalized states,
/// causing the variational optimization to converge to the next eigenstate.
/// For targeting k states simultaneously, use `BlockDavidsonSolver` with
/// `DMRGConfig::n_target_states = Some(k)`.
#[derive(Clone, Debug)]
pub struct ExcitedStateConfig {
    /// Previously converged eigenstates to penalize.
    pub penalized_states: Vec<MPS<f64, U1, MixedCanonical>>,
    /// Energy penalty weight. Must exceed the energy gap to the target state.
    /// Default: 0.1.
    pub penalty_weight: f64,
}

/// Build a penalized H_eff closure for excited-state DMRG.
///
/// Returns a modified matvec that adds the penalty term
/// `weight * Σ_j |proj_j⟩⟨proj_j|` to the standard H_eff matvec,
/// where `proj_j` is the projection of the current vector onto
/// penalized state j.
pub fn build_heff_penalized<'arena, T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>(
    base_matvec: impl Fn(&[T], &mut [T]) + 'arena,
    excited_config: &'arena ExcitedStateConfig,
    penalized_envs: &'arena [Environments<T, Q>],
    sites: (usize, usize),
) -> impl Fn(&[T], &mut [T]) + 'arena;
```

---

## 12. Checkpointing

```rust
/// Full serializable checkpoint of a DMRG engine state.
///
/// Written atomically (temp file + rename) at the end of each full sweep
/// when `DMRGConfig::checkpoint_path` is set. Enables restart after
/// process interruption without losing converged state.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct DMRGCheckpoint<T, Q>
where
    T: Scalar + serde::Serialize + for<'de> serde::Deserialize<'de>,
    Q: BitPackable + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    pub mps_tensors: Vec<SerializedBlockSparseTensor<T, Q>>,
    pub mpo_tensors: Vec<SerializedBlockSparseTensor<T, Q>>,
    pub sweep_index: usize,
    pub energy: f64,
    pub stats: DMRGStats,
    /// `DMRGConfig` as JSON for human inspection.
    pub config_json: String,
}

impl<T, Q> DMRGCheckpoint<T, Q>
where
    T: Scalar + serde::Serialize + for<'de> serde::Deserialize<'de>,
    Q: BitPackable + serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    /// Write checkpoint atomically: write to `{path}.tmp`, then rename.
    ///
    /// # Errors
    /// Returns `DmrgError::CheckpointIo` on any filesystem error.
    pub fn write_to_file(&self, path: &std::path::Path) -> DmrgResult<()>;

    /// Read checkpoint from file.
    ///
    /// # Errors
    /// Returns `DmrgError::CheckpointIo` on I/O failure.
    /// Returns `DmrgError::CheckpointDeser` on deserialization failure.
    pub fn read_from_file(path: &std::path::Path) -> DmrgResult<Self>;
}
```

---

## 13. Error Handling

```rust
/// Error type for `tk-dmrg`.
#[derive(Debug, thiserror::Error)]
pub enum DmrgError {
    #[error("shape mismatch: {context}")]
    ShapeMismatch { context: String },

    #[error("dimension mismatch: MPS has {mps_sites} sites, MPO has {mpo_sites} sites")]
    DimensionMismatch { mps_sites: usize, mpo_sites: usize },

    #[error("site index {site} out of bounds for MPS of length {n_sites}")]
    SiteBoundsError { site: usize, n_sites: usize },

    #[error("charge sector unreachable: cannot reach charge {charge:?} with given local dims")]
    ChargeSectorEmpty { charge: String },

    #[error("eigensolver did not converge after {iters} matvec calls; residual = {residual:.2e}")]
    EigensolverNotConverged { iters: usize, residual: f64 },

    #[error("OpSum compilation failed: {reason}")]
    OpSumCompilationFailed { reason: String },

    #[error("TDVP Krylov matrix-exponential did not converge in {krylov_dim} steps")]
    TdvpKrylovNotConverged { krylov_dim: usize },

    #[error("infinite DMRG convergence failed after {extensions} unit-cell extensions")]
    IDmrgConvergenceFailed { extensions: usize },

    #[error("bond singular values unavailable: MPS is not in BondCentered form")]
    BondSingularValuesUnavailable,

    #[error("checkpoint I/O error: {0}")]
    CheckpointIo(#[from] std::io::Error),

    #[error("checkpoint deserialization error: {0}")]
    CheckpointDeser(String),

    #[error("linear algebra error")]
    Linalg(#[from] LinAlgError),

    #[error("symmetry error")]
    Symmetry(#[from] SymmetryError),

    #[error("tensor core error")]
    Core(#[from] TkError),

    #[error("computation cancelled")]
    Cancelled,
}

/// Result alias for `tk-dmrg`.
pub type DmrgResult<T> = Result<T, DmrgError>;
```

---

## 14. Data Structures and Internal Representations

### 14.1 Tensor Leg Ordering Convention

| Tensor | Rank | Leg order | Notes |
|:-------|:-----|:----------|:------|
| MPS site tensor A[i] | 3 | (σ, α_L, α_R) | σ = physical; α = MPS bond |
| MPO site tensor W[i] | 4 | (σ_in, σ_out, w_L, w_R) | w = MPO bond |
| Left environment L | 3 | (α_bra, w, α_ket) | bra above, ket below |
| Right environment R | 3 | (α_bra, w, α_ket) | same convention as L |
| Two-site tensor Θ | 4 | (σ_i, σ_{i+1}, α_L, α_R) | merged for 2-site update |
| Bond matrix S | 2 | (α_L, α_R) | diagonal; D × D |

### 14.2 Quantum Number Flow in U(1)-Symmetric MPS

The flux rule for each MPS site tensor A[i] with legs (σ, α_L, α_R):
```
q(σ) + q(α_L) - q(α_R) = 0
```
The outgoing bond α_R carries the cumulative charge of sites 0..i. The total charge of the MPS is the charge carried by the rightmost bond (which equals `total_charge`).

The left environment at site i carries charge q_L. The right environment at site i carries charge q_R such that q_L + q_W + q_R = total_charge, where q_W is the MPO flux (zero for Hermitian Hamiltonians).

### 14.3 Memory Layout for Environments

At D = 2000, w = 50, T = f64, the full environment tensor (dense) would be D × w × D × 8 bytes ≈ 1.6 GB per environment. With U(1) symmetry, typically 1/√D sectors are occupied, reducing to ~80 MB. Total for N = 100 sites: ~8 GB. A `max_env_memory` parameter (default 8 GB) in `DMRGConfig` triggers disk offloading of the most distant environments (see Implementation Note 3).

---

## 15. Dependencies and Integration

### 15.1 Upstream Dependencies (Cargo.toml)

```toml
[dependencies]
tk-core      = { path = "../tk-core",     version = "0.1.0" }
tk-symmetry  = { path = "../tk-symmetry", version = "0.1.0" }
tk-linalg    = { path = "../tk-linalg",   version = "0.1.0" }
tk-contract  = { path = "../tk-contract", version = "0.1.0" }
tk-dsl       = { path = "../tk-dsl",      version = "0.1.0" }

num-complex  = "0.4"
num-traits   = "0.2"
smallvec     = "1"
rand         = "0.8"
thiserror    = "1"

serde        = { version = "1", features = ["derive"] }
bincode      = "2"
serde_json   = "1"

tracing      = "0.1"

[dev-dependencies]
proptest     = "1"
approx       = "0.5"
rand_chacha  = "0.3"
criterion    = { version = "0.5", optional = true }

[features]
default          = ["backend-faer", "backend-oxiblas", "parallel"]
backend-faer     = ["tk-linalg/backend-faer"]
backend-oxiblas  = ["tk-linalg/backend-oxiblas"]
backend-mkl      = ["tk-linalg/backend-mkl"]
backend-openblas = ["tk-linalg/backend-openblas"]
backend-cuda     = ["tk-core/backend-cuda", "tk-linalg/backend-cuda"]
su2-symmetry     = ["tk-symmetry/su2-symmetry", "tk-linalg/su2-symmetry"]
parallel         = ["tk-linalg/parallel"]
bench            = ["criterion"]
```

### 15.2 Downstream Consumers

| Crate | Usage |
|:------|:------|
| `tk-dmft` | `DMRGEngine`, `TdvpDriver`, `MPS`, `MPO`, `MpoCompiler`, `DmrgResult` |
| `tk-python` | Type-erased through `DmftLoopVariant` enum in `tk-dmft` |

### 15.3 Integration Point: `tk-dsl` → `tk-dmrg`

`tk-dsl` produces `OpSum<T>`. `MpoCompiler::compile` in `tk-dmrg` is the sole consumer of `OpSum`. This is the hard architectural boundary preventing a cyclic dependency: `tk-dsl` must not import `tk-linalg` (design doc §2.2).

```rust
// Full end-to-end usage:
use tk_dsl::{hamiltonian, SpinOp};
use tk_dmrg::{DMRGEngine, DMRGConfig, MpoCompiler, MpoCompressionConfig,
              MPS, BondDimensionSchedule};
use tk_linalg::DeviceFaer;
use tk_symmetry::U1;

let backend = DeviceFaer;

// Phase 1: compile OpSum → MPO (runtime SVD compression)
let opsum = hamiltonian! {
    lattice: Chain(N = 100, d = 2);
    sum i in 0..N-1 {
        0.5 * (Sp(i) * Sm(i+1) + Sm(i) * Sp(i+1)) + Sz(i) * Sz(i+1)
    }
};
let mpo: MPO<f64, U1> = MpoCompiler::new(&backend, MpoCompressionConfig {
    max_bond_dim: 50,
    svd_cutoff: 1e-12,
    validate_compression: cfg!(debug_assertions),
    compression_tol: 1e-8,
}).compile(&opsum, 100, &vec![2; 100], U1(0))?;

// Phase 2: run DMRG
let mps = MPS::random(100, &vec![2; 100], 64, U1(0), 50, &mut rng)?;
let mut engine = DMRGEngine::new(mps, mpo, backend, DMRGConfig {
    bond_dim_schedule: BondDimensionSchedule::warmup(64, 500, 5),
    ..DMRGConfig::default()
})?;
let ground_energy: f64 = engine.run()?;
```

---

## 16. Feature Flags

| Flag | Effect in `tk-dmrg` |
|:-----|:--------------------|
| `backend-faer` | Propagates to `tk-linalg`; enables `DeviceFaer` for SVD/GEMM |
| `backend-oxiblas` | Propagates to `tk-linalg`; enables `DeviceOxiblas` for sparse ops |
| `backend-mkl` | Propagates to `tk-linalg`; enables FFI Intel MKL |
| `backend-openblas` | Propagates to `tk-linalg`; enables FFI OpenBLAS |
| `backend-cuda` | Propagates to `tk-core` and `tk-linalg`; `DMRGStats` gains `pinned_memory_fallbacks` counter |
| `su2-symmetry` | Propagates to `tk-symmetry` and `tk-linalg`; enables multiplet-aware SVD truncation (two-phase: sort + snap to multiplet boundary; (2j+1)-weighted truncation error) |
| `parallel` | Propagates to `tk-linalg`; enables Rayon-parallel LPT-scheduled block GEMM |

`backend-mkl` and `backend-openblas` remain mutually exclusive, enforced by `tk-linalg/build.rs`. No additional `build.rs` is required in `tk-dmrg`.

---

## 17. Testing Strategy

### 17.1 Unit Tests

| Test | Description |
|:-----|:------------|
| `mps_random_is_mixed_canonical` | A†A = I for all sites left of center; BB† = I for all sites right |
| `mps_shift_center_preserves_state` | `shift_center(k)` changes `center` without changing `mps_overlap(mps, mps)` |
| `mps_expose_absorb_bond_roundtrip` | `expose_bond().absorb_bond()` recovers the original state; overlap ≈ 1 |
| `mps_entanglement_entropy_product_state` | Product state has S = 0 at all bonds |
| `mpo_identity_energy` | `<ψ|I|ψ>` = `<ψ|ψ>` for any normalized ψ |
| `mpo_compile_heisenberg_bond_dim` | Heisenberg OpSum compiles to MPO with bond dim ≤ 5 |
| `mpo_compile_hubbard_bond_dim` | Hubbard model (with Jordan-Wigner strings) compiles to MPO bond dim ≤ 10 |
| `mpo_add_scaled_bond_dim_additive` | H + α*H has bond dim = 2 * bond_dim(H) before compression |
| `env_build_boundary_shape` | Left boundary has shape (1, 1, 1); right boundary same |
| `env_grow_left_coverage` | After `grow_left(i)`, left env `up_to()` = i+1 |
| `env_energy_matches_mps_energy` | Full environment contraction matches `mps_energy` |
| `lanczos_hermitian_convergence` | Converges to known eigenvalue of a 10×10 diagonal test matrix |
| `lanczos_thick_restart_correctness` | Thick restart does not degrade eigenvalue accuracy |
| `davidson_fewer_iters_than_lanczos` | DavidsonSolver converges in fewer iterations for diagonal-dominant test |
| `block_davidson_k_states` | Returns correct 3 lowest eigenvalues of a 20×20 test matrix |
| `truncate_svd_bond_dim_capped` | Returned bond dim ≤ max_bond_dim |
| `truncate_svd_cutoff_applied` | All retained singular values > svd_cutoff |
| `truncate_svd_truncation_error_formula` | Truncation error = Σ_discarded σ² / Σ_all σ² |
| `bond_dim_schedule_warmup_monotone` | Warmup schedule is non-decreasing and ends at D_max |
| `dmrg_heisenberg_n4_energy` | N=4 Heisenberg energy matches exact diagonalization |
| `dmrg_heisenberg_n8_energy` | N=8 energy within 1e-10 of ED |
| `dmrg_converged_flag_triggers` | `converged()` returns true when energy change < `energy_tol` |
| `dmrg_cancellation_flag` | `run_with_cancel_flag` returns `Cancelled` within 1 sweep step |
| `tdvp_norm_conservation` | `||ψ(t)||` ≈ 1 for all t (Hermitian H, imaginary time → real convergence) |
| `tdvp_tikhonov_no_nan` | Near-zero bond singular values do not produce NaN after Tikhonov regularization |
| `expand_bond_subspace_orthogonality` | `||<A_L | R_null>||` < 1e-10 after expansion |
| `expand_bond_subspace_dim_consistency` | A_L column count increases by D_expand; bond matrix shape (D+D_expand)×(D+D_expand) |
| `expand_bond_subspace_vs_explicit` | Matrix-free projection matches explicit `(I - A_L·A_L†)·|R>` for small test case |
| `soft_dmax_no_oscillation` | Bond dim does not oscillate between expansion and truncation over 20 steps |
| `exp_krylov_matches_exact_expm` | `exp_krylov` on a 4×4 tridiagonal matches `scipy.linalg.expm` reference |
| `checkpoint_write_read_roundtrip` | Write + read checkpoint recovers identical MPS energy and bond dims |
| `assert_mps_equivalent_macro` | `assert_mps_equivalent!` passes for two gauge-rotated representations of the same state |

### 17.2 Property-Based Tests

```rust
proptest! {
    // Bounded: max 6 sites, max bond dim 8, max 4 sectors, 256 cases
    #[test]
    fn mps_left_canonicalize_orthogonality(
        n_sites in 2usize..=6,
        local_dim in 2usize..=4,
        bond_dim in 1usize..=8,
    ) {
        // Construct random MPS, left-canonicalize, verify A†A = I at each site
    }

    #[test]
    fn truncation_error_formula_correct(
        singular_values in prop::collection::vec(0.01f64..=1.0, 2..=20),
        max_bond_dim in 1usize..=10,
    ) {
        // Verify: truncation_error = sum_discarded(σ²) / sum_all(σ²)
    }

    #[test]
    fn mpo_compile_preserves_hermiticity(
        j_values in prop::collection::vec(-2.0f64..=2.0, 1..=3),
    ) {
        // Verify <ψ|H|φ> = conj(<φ|H|ψ>) for random MPS ψ, φ
    }
}
```

### 17.3 Reference Snapshot Tests

```rust
// fixtures/heisenberg_chain_n20_d500.json  (from ITensor, N=20)
// fixtures/heisenberg_chain_n100_d200.json (from Block2, N=100)
// fixtures/hubbard_n10_u4_d200.json        (from ITensor, N=10)

#[test]
fn heisenberg_n20_energy_matches_itensor() {
    let reference: ReferenceData = load_fixture("heisenberg_chain_n20_d500.json");
    let energy = run_dmrg_heisenberg(n=20, d=500);
    assert!((energy - reference.energy).abs() < 1e-10);
}

#[test]
fn heisenberg_n100_truncation_error_acceptable() {
    let result = run_dmrg_heisenberg(n=100, d=200);
    assert!(result.max_truncation_error < 1e-6);
}
```

Cross-backend tests use `assert_mps_equivalent!` and `assert_svd_equivalent!` macros to avoid spurious failures from SVD sign gauge freedom (design doc §12.1.1).

### 17.4 Compile-Fail Tests

```rust
// tests/compile_fail/wrong_gauge_dmrg_step.rs
fn wrong_gauge_rejected() {
    let mps: MPS<f64, U1, LeftCanonical> = /* ... */;
    // must not compile: dmrg_step_two_site requires MixedCanonical
    let engine = DMRGEngine { mps, /* ... */ };
    engine.dmrg_step_two_site(0, SweepDirection::LeftToRight);
}
```

### 17.5 Performance Benchmarks

| Benchmark | Condition | Target |
|:----------|:----------|:-------|
| `bench_two_site_step` | N=100, D=200, U(1) | < 50 ms per step (DeviceFaer) |
| `bench_env_grow_left` | D=500, w=5 | < 5 ms per site |
| `bench_lanczos` | dim=8000, 100 iterations | < 200 ms |
| `bench_svd_truncation` | 4000×4000 dense matrix | < 500 ms (gesdd) |
| `bench_full_sweep` | N=100, D=200 | < 10 s per full sweep |
| `bench_mpo_compile_heisenberg` | N=100 | < 2 s |

CI uses `iai`/`divan` instruction counting with ±2% regression threshold. Criterion wall-clock benchmarks reserved for local bare-metal.

---

## 18. Implementation Notes and Design Decisions

### Note 1: Why In-House Eigensolvers

Off-the-shelf Rust eigensolvers such as `eigenvalues` accept dense matrix types and cannot be adapted to the `Fn(&[T], &mut [T])` closure required for zero-allocation H_eff matvec. Wrapping the effective Hamiltonian in a dense matrix would require materializing a D²d × D²d matrix (up to 64 million elements at D=2000, d=4) — defeating the entire purpose of DMRG. Additionally, in-house implementations allow thick-restart state management and pre-allocation from `SweepArena` that external crates cannot provide. This decision is specified in design doc §8.2 and §11.

### Note 2: Two-Site vs. Single-Site Updates

The two-site update (§8.2) diagonalizes a rank-4 object of size d²·D² and allows bond dimension growth (SVD can produce up to d·D new singular values per site). It is required for the first warmup sweeps. Single-site updates (§8.2) are O(d D³) per step and cannot grow the bond dimension; they are used for the final convergence sweeps. The `UpdateVariant` enum selects between them. An adaptive policy (start two-site, switch to single-site when truncation error drops below a threshold) can be layered on top of `DMRGConfig`.

### Note 3: Environment Memory Scaling

For large systems (N=200, D=1000, w=50), full in-memory environment caching requires ~10 GB. Two strategies are available:
1. **On-the-fly recomputation**: Recompute environments from scratch at each sweep-direction reversal. Cost: one extra O(N·d·D²·w) pass per sweep.
2. **Partial caching with disk offload**: Keep only the k nearest environments in memory; serialize distant ones via `checkpoint.rs`. A `max_env_memory` field in `DMRGConfig` (default: 8 GB) triggers this automatically.

The Phase 3 implementation uses full in-memory caching. Disk offloading is deferred pending benchmarks.

### Note 4: OpSum → MPO via Finite-State Automaton

The `MpoCompiler` translates `OpSum` terms into an MPO via the finite-state automaton (FSA) method. Each term (e.g., `J * Sz(i) * Sz(i+1)`) defines a path through the FSA, contributing one row/column to the MPO transfer matrix at each site. For nearest-neighbor models, the FSA gives the exact minimal-bond-dim MPO (Heisenberg = 5, Hubbard ≈ 8) without SVD compression. For long-range models the uncompressed bond dim is O(N); SVD compression reduces it to `max_bond_dim`. The standard approach used by ITensor and TeNPy.

### Note 5: Fermionic Sign Convention

The `tk-contract` engine is bosonic-only (design doc §6.4). Jordan-Wigner strings for fermionic models are encoded in the MPO tensors by `MpoCompiler::compile`. The Jordan-Wigner string operator F = (−1)^N is inserted between fermionic creation/annihilation operators by the compiler. This is correct for all 1D chain and star-to-chain geometries through Phase 4. Native fermionic swap gates for tree/PEPS are deferred to Phase 5+.

### Note 6: Checkpoint Atomicity

Checkpoint writes use `write_to_temp + rename` to guarantee atomic updates. On POSIX systems, `rename(2)` is atomic within the same filesystem. On Windows, `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING` provides similar semantics. The implementation uses `std::fs::rename` across all platforms.

### Note 7: SU(2) Multiplet-Aware Truncation

Under the `su2-symmetry` feature flag, `truncate_svd` implements two-phase truncation (design doc §4.4):
1. Sort all singular values by magnitude.
2. Snap the truncation boundary to the nearest multiplet edge (never split a 2j+1-degenerate multiplet).
3. Weight discarded values by (2j+1)·σ_i² in the truncation error sum.

`TruncationConfig` is extended (under `su2-symmetry`) with `multiplet_info: Option<Vec<(SU2Irrep, usize)>>` supplying the sector structure. When `None`, standard scalar truncation applies (Abelian path unchanged).

### Note 8: `DefaultEngine` Type Alias

Per design doc §5.4, only the most common concrete combination is compiled by default:

```rust
#[cfg(all(
    feature = "backend-faer",
    not(any(feature = "backend-mkl", feature = "backend-openblas"))
))]
pub type DefaultEngine = DMRGEngine<f64, tk_symmetry::U1, tk_linalg::DeviceFaer>;
```

Additional combinations (e.g., `Complex<f64>` for real-time TDVP, `Z2` for parity-symmetric models) are compiled only when explicitly requested via feature flags.

---

## 19. Public API Surface (`lib.rs`)

```rust
// tk-dmrg/src/lib.rs

pub mod mps;
pub mod mpo;
pub mod environments;
pub mod sweep;
pub mod eigensolver;
pub mod truncation;
pub mod tdvp;
pub mod excited;
pub mod idmrg;
pub mod checkpoint;
pub mod error;

// MPS types and gauge markers
pub use mps::{
    MPS,
    LeftCanonical, RightCanonical, MixedCanonical, BondCentered,
    left_canonicalize, right_canonicalize, mixed_canonicalize,
    mps_overlap, mps_energy, energy_variance,
};

// MPO types and compilation
pub use mpo::{MPO, MpoCompiler, MpoCompressionConfig};

// Environment management
pub use environments::{
    Environment, Environments,
    build_heff_two_site, build_heff_single_site,
};

// DMRG sweep engine and configuration
pub use sweep::{
    DMRGEngine, DMRGConfig, DMRGStats,
    SweepSchedule, SweepDirection, StepResult,
    UpdateVariant,
};

// Iterative eigensolvers
pub use eigensolver::{
    IterativeEigensolver, EigenResult, InitialSubspace,
    LanczosSolver, DavidsonSolver, BlockDavidsonSolver,
};

// SVD truncation
pub use truncation::{
    TruncationResult, TruncationConfig, BondDimensionSchedule,
    truncate_svd,
};

// TDVP time evolution
pub use tdvp::{
    TdvpDriver, TdvpStabilizationConfig, TdvpStepResult,
    expand_bond_subspace, exp_krylov,
};

// Excited states
pub use excited::{ExcitedStateConfig, build_heff_penalized};

// Infinite DMRG
pub use idmrg::{run_idmrg, IDmrgConfig};

// Checkpointing
pub use checkpoint::DMRGCheckpoint;

// Error handling
pub use error::{DmrgError, DmrgResult};

// Default engine alias for the common case: f64, U1, DeviceFaer
#[cfg(all(
    feature = "backend-faer",
    not(any(feature = "backend-mkl", feature = "backend-openblas"))
))]
pub type DefaultEngine = DMRGEngine<f64, tk_symmetry::U1, tk_linalg::DeviceFaer>;
```

---

## 20. Out of Scope

The following are explicitly **not** implemented in `tk-dmrg`:

- DMFT self-consistency loop, bath discretization, Anderson impurity model (→ `tk-dmft`)
- Python bindings and GIL management (→ `tk-python`)
- TEBD (Trotterized time evolution) as a primary method (→ `tk-dmft`; TEBD is a DMFT fallback, not a DMRG concern)
- Linear prediction and Chebyshev expansion for spectral functions (→ `tk-dmft`)
- Multi-dimensional (2D PEPS) or tree tensor network algorithms (→ Phase 5+)
- Native fermionic swap gates in the contraction engine (→ `tk-contract`, Phase 5+)
- MPI-distributed tensor contractions (→ `tk-linalg`/`tk-dmft`, Phase 5+)
- `#[pyclass]` Python-accessible wrappers (→ `tk-python`)
- Block-sparse GEMM dispatch and LPT scheduling (→ `tk-linalg`)
- DAG contraction path optimization (→ `tk-contract`)
- Index types and operator enum definitions (→ `tk-dsl`)

---

## 21. Open Questions

| # | Question | Status |
|:--|:---------|:-------|
| 1 | Should `Environments` support disk offloading for large systems (D > 1000, N > 100)? Current spec allocates all N environments in RAM. An LRU cache with disk eviction via `checkpoint.rs` may be required before Phase 3 benchmarks. | Open — benchmark environment memory at D=1000, N=100 first |
| 2 | Should `DMRGConfig::eigensolver` be `Box<dyn IterativeEigensolver<f64>>` (current) or `Box<dyn IterativeEigensolver<T>>`? The former requires a separate eigensolver field for complex-valued TDVP; the latter propagates T into DMRGConfig. | Open — decide when implementing TDVP detail |
| 3 | `DavidsonSolver::diagonal` is mutably set by the engine before each call. Is there a cleaner injection mechanism that does not require mutable state in the solver struct? A `DiagonalPreconditioner` strategy trait may be cleaner. | Deferred — refactor after Phase 3 prototype |
| 4 | Should `MpoCompiler` expose an `MpoCompressionStrategy` enum (FSA-only vs. FSA + SVD), or always apply SVD? For short-range models, FSA-only gives the exact minimal-bond-dim MPO; SVD adds overhead. | Open — determine whether the overhead is material during Phase 2 |
| 5 | `run_idmrg` takes a unit-cell MPO (2 sites). For non-trivial unit cells (dimerized chains, multi-orbital Hubbard), the unit cell size should be configurable. Add `unit_cell_size: usize` to `IDmrgConfig`? | Deferred — required only when iDMRG is tested with non-trivial unit cells |
| 6 | Should `energy_variance` be computed during the sweep (at the cost of one extra H² matvec per site, ~50% overhead per sweep) or only on-demand after convergence? The `variance_tol` criterion makes per-sweep computation useful. | Open — profile cost before choosing default |
| 7 | `TruncationConfig::multiplet_info` (SU(2) only) introduces a `Vec<(SU2Irrep, usize)>` into the common truncation path. A `TruncationPolicy` trait (with `select_cutoff` method) may decouple the SU(2)-specific type from the generic code path. | Deferred to when `su2-symmetry` is actively implemented |
| 8 | Should `BlockSparseTensor` derive `serde::Serialize` (propagating a `serde` dependency into `tk-symmetry`) or should `tk-dmrg` define its own `SerializedBlockSparseTensor` serialization proxy? | Open — check `serde` feature-gating conventions used across the workspace |
