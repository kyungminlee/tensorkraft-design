# Technical Specification: `tk-dsl`

**Crate:** `tensorkraft/crates/tk-dsl`
**Version:** 0.1.0 (Post-Draft-Implementation)
**Status:** Draft
**Last Updated:** March 2026

---

## 1. Overview

`tk-dsl` is the ergonomic API and domain-specific language layer for the tensorkraft workspace. It sits directly above `tk-core` and `tk-symmetry` in the dependency graph and is consumed exclusively by `tk-dmrg`. Its central responsibility is turning quantum physics intent — "the Heisenberg Hamiltonian on a chain of N sites" — into a well-typed, compiler-verified `OpSum` data structure that can be passed to `tk-dmrg` for MPO compilation.

**Core responsibilities:**

- **Intelligent index system** — An `Index` struct that carries a unique `IndexId`, a human-readable tag, a dimension, a prime level for the common prime-index pattern, and a direction. Two indices with matching `id` and complementary prime levels automatically identify a contraction pair when passed to `tk-contract` via `IndexedTensor`.
- **Strongly-typed operator enums** — `SpinOp`, `FermionOp`, `BosonOp`, and `CustomOp<T>` eliminate runtime string-matching errors that plague APIs like `op("S_plus", i)`. The unified `SiteOperator<T>` and the `op(operator, site)` constructor form the core of the DSL vocabulary.
- **`OpSum` builder** — A runtime accumulator of weighted products of site operators. `OpSum` is the only output `tk-dsl` produces; it does not perform SVD compression, FSA minimization, or MPO allocation. Those operations belong to `tk-dmrg` via `OpSum::compile_mpo`.
- **`hamiltonian!{}` proc-macro** — A procedural macro that parses a lattice/Hamiltonian DSL at compile time and emits Rust code that constructs an `OpSum` at runtime. It eliminates tedious boilerplate (`for i in 0..N-1 { opsum += ... }`) while keeping all numerical computation at runtime.
- **Lattice abstraction** — A `Lattice` trait with built-in implementations for `Chain`, `Square`, `Triangular`, `BetheLattice`, and `StarGeometry` (for the Anderson Impurity Model). The trait provides `n_sites`, `bonds`, and `dmrg_ordering` (snake-path for 2D-to-1D mapping).
- **`IndexedTensor<T>`** — A thin wrapper around `DenseTensor<T>` that attaches `Index` objects to each leg, providing an ITensor-style named-index tensor expression interface for use in `tk-dmrg` and `tk-contract`.

**Critical architectural constraint (no-linalg rule, design doc §2.2):** `tk-dsl` has **no dependency on `tk-linalg`** and **no dependency on `tk-dmrg`**. The `OpSum → MPO` compilation step involves SVD-based compression using a `LinAlgBackend` and lives entirely within `tk-dmrg`. `tk-dsl` generates uncompressed operator sums only. This rule exists to prevent cyclic dependencies and to keep `tk-dsl` as a lightweight compile-time / ergonomic layer that does not drag the entire linear algebra stack into compilation.

---

## 2. Module Structure

```
tk-dsl/
├── Cargo.toml
├── build.rs                    (version parity check: tk-dsl vs tk-dsl-macros)
└── src/
    ├── lib.rs                  re-exports all public items
    ├── index.rs                Index, IndexDirection, IndexRegistry
    ├── indexed_tensor.rs       IndexedTensor<T>, contract() function
    ├── operators.rs            SpinOp, FermionOp, BosonOp, CustomOp<T>, SiteOperator<T>
    ├── opterm.rs               OpTerm<T>, op() constructor, OpProduct<T>, ScaledOpProduct<T>
    ├── opsum.rs                OpSum<T>, OpSumTerm<T>, HermitianConjugate, hc()
    ├── lattice/
    │   ├── mod.rs              Lattice trait, re-exports
    │   ├── chain.rs            Chain
    │   ├── square.rs           Square, snake_path()
    │   ├── triangular.rs       Triangular
    │   ├── bethe.rs            BetheLattice
    │   └── star.rs             StarGeometry (Anderson Impurity Model)
    └── error.rs                DslError, DslResult<T>

tk-dsl-macros/                  (separate proc-macro crate, published together)
├── Cargo.toml
└── src/
    ├── lib.rs                  #[proc_macro] hamiltonian
    ├── parse.rs                HamiltonianInput, LatticeSpec, SumBlock, TermExpr
    ├── expand.rs               code generation: OpSum construction statements
    ├── validate.rs             semantic validation (operator name resolution)
    └── error.rs                MacroError, span-annotated diagnostics
```

**Proc-macro crate discipline:** Rust requires proc-macro crates to be separate crates with `proc-macro = true` in `[lib]`. `tk-dsl-macros` is the proc-macro implementation; `tk-dsl` re-exports the `hamiltonian!` macro from it. Users depend only on `tk-dsl`.

---

## 3. The `Index` Type and Index System

### 3.1 Definition

```rust
/// A named, directed tensor index.
///
/// Indices are the primary mechanism for expressing which legs of two tensors
/// should be contracted. Two `Index` values match for contraction if their
/// `id` fields are equal and their `prime_level` fields differ by exactly one
/// (the standard ITensor prime convention).
///
/// # Example
///
/// ```rust
/// let i = Index::new("phys", 2, IndexDirection::None);
/// let i_prime = i.prime();  // same id, prime_level + 1
/// // i and i_prime contract automatically when passed to contract(&a, &b).
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Index {
    /// Globally unique identifier. Assigned by `IndexRegistry` or `Index::new`.
    id: IndexId,
    /// Human-readable label for debug output.
    /// Not used for equality comparison; two indices with the same tag but
    /// different `id`s are distinct and do NOT contract.
    tag: SmallString<[u8; 32]>,
    /// Number of basis states along this dimension.
    dim: usize,
    /// Prime level. Level 0 is the "bare" index; level 1 is a primed copy.
    /// `contracts_with` pairs level k with level k+1 (in either direction).
    /// Higher prime levels (2, 3, ...) represent multiple priming steps, e.g.
    /// for operator-squared or multi-time-step expressions.
    prime_level: u32,
    /// Optional flow direction for quantum-number-carrying indices.
    /// `IndexDirection::None` for non-symmetric (dense) tensors.
    /// `Incoming` / `Outgoing` required when constructing `IndexedTensor`
    /// over a `BlockSparseTensor`.
    direction: IndexDirection,
}

/// Flow direction of a tensor leg under the active symmetry group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexDirection {
    /// Quantum numbers flow into the tensor along this leg.
    Incoming,
    /// Quantum numbers flow out of the tensor along this leg.
    Outgoing,
    /// No symmetry direction assigned (dense / no-symmetry path).
    None,
}
```

### 3.2 Constructors and Methods

```rust
impl Index {
    /// Create a new index with a globally unique `IndexId`.
    ///
    /// Uses an atomic counter to generate the `IndexId`; thread-safe.
    ///
    /// # Parameters
    /// - `tag`: Human-readable label used in error messages and debug output.
    /// - `dim`: Number of basis states. Must be > 0.
    /// - `direction`: Flow direction for symmetry-carrying legs; `None` for dense.
    ///
    /// # Panics
    /// Panics in debug mode if `dim == 0`.
    pub fn new(
        tag: impl Into<SmallString<[u8; 32]>>,
        dim: usize,
        direction: IndexDirection,
    ) -> Self;

    /// Return a copy of this index with `prime_level` incremented by one.
    ///
    /// Calling `i.prime()` in a tensor expression means "the same physical
    /// index after one application of the operator" — the standard ITensor
    /// convention for operator-state contractions.
    ///
    /// Does not heap-allocate; `SmallString` is copied inline.
    #[must_use]
    pub fn prime(&self) -> Self;

    /// Return a copy with `prime_level` incremented by `n`.
    #[must_use]
    pub fn prime_n(&self, n: u32) -> Self;

    /// Return a copy with `prime_level` reset to 0.
    #[must_use]
    pub fn unprime(&self) -> Self;

    /// True iff this index and `other` share the same `id`,
    /// regardless of prime level.
    pub fn same_id(&self, other: &Index) -> bool {
        self.id == other.id
    }

    /// True iff this index should contract with `other`:
    /// same `id`, and `|self.prime_level - other.prime_level| == 1`.
    pub fn contracts_with(&self, other: &Index) -> bool {
        self.id == other.id
            && self.prime_level.abs_diff(other.prime_level) == 1
    }

    pub fn id(&self) -> IndexId;
    pub fn tag(&self) -> &str;
    pub fn dim(&self) -> usize;
    pub fn prime_level(&self) -> u32;
    pub fn direction(&self) -> IndexDirection;
}
```

### 3.3 `IndexRegistry`

```rust
/// A scoped registry that creates named indices with guaranteed uniqueness.
///
/// Recommended for constructing all indices in a model definition in one
/// place, preventing accidental `IndexId` collisions across separately
/// defined models.
///
/// # Example
///
/// ```rust
/// let mut reg = IndexRegistry::new();
/// let phys = reg.register("phys", 2, IndexDirection::None).unwrap();
/// let bond = reg.register("bond", 64, IndexDirection::None).unwrap();
/// assert_ne!(phys.id(), bond.id());
/// ```
pub struct IndexRegistry {
    entries: Vec<(SmallString<[u8; 32]>, Index)>,
}

impl IndexRegistry {
    pub fn new() -> Self;

    /// Create a new `Index` with a unique `IndexId` and store it under `tag`.
    ///
    /// # Errors
    /// Returns `DslError::DuplicateIndexTag` if `tag` was already registered
    /// in this registry instance. Prevents silent aliasing from name reuse.
    pub fn register(
        &mut self,
        tag: impl Into<SmallString<[u8; 32]>>,
        dim: usize,
        direction: IndexDirection,
    ) -> DslResult<Index>;

    /// Look up a previously registered index by tag.
    /// Returns `None` if the tag was never registered.
    pub fn get(&self, tag: &str) -> Option<&Index>;
}
```

---

## 4. `IndexedTensor<T>` — Named-Index Tensor

### 4.1 Definition

```rust
/// A `DenseTensor<T>` with a named `Index` on each leg.
///
/// Provides an ITensor-style interface: pass two `IndexedTensor`s to
/// `contract()` and the contraction pairs are identified automatically by
/// matching `id` and the prime-level convention.
///
/// **Implementation note:** `DenseTensor` does not implement `Clone`.
/// `IndexedTensor` stores `DenseTensor<'static, T>` and provides a
/// `clone_owned()` method that performs a manual clone via
/// `DenseTensor::from_vec(shape, slice.to_vec())`.
#[derive(Debug)]
pub struct IndexedTensor<T: Scalar> {
    /// Underlying dense tensor data (owned, `'static` lifetime).
    /// Legs correspond 1-to-1 with `indices`.
    pub data: DenseTensor<'static, T>,
    /// Ordered list of indices, one per tensor leg.
    /// Invariant: `indices.len() == data.rank()`.
    pub indices: SmallVec<[Index; 6]>,
}

impl<T: Scalar> IndexedTensor<T> {
    /// Construct from a tensor and its leg indices.
    ///
    /// # Panics
    /// Panics in debug mode if `indices.len() != data.rank()`.
    pub fn new(data: DenseTensor<T>, indices: impl Into<SmallVec<[Index; 6]>>) -> Self;

    /// Return the index on leg `axis`.
    ///
    /// # Panics
    /// Panics if `axis >= self.data.rank()`.
    pub fn index(&self, axis: usize) -> &Index;

    /// Find the leg number whose index matches `idx` (by `id` and `prime_level`).
    /// Returns `None` if `idx` is not present on any leg.
    pub fn find_leg(&self, idx: &Index) -> Option<usize>;

    /// Increment the prime level of the index on `axis` by one in-place.
    pub fn prime_leg(&mut self, axis: usize);

    /// Increment the prime level of every index sharing `id` by one in-place.
    pub fn prime_index(&mut self, id: IndexId);

    /// Manual clone, required because `DenseTensor` does not implement `Clone`.
    ///
    /// Clones the underlying data via `DenseTensor::from_vec(shape, slice.to_vec())`
    /// and copies all indices.
    pub fn clone_owned(&self) -> Self;
}
```

### 4.2 Named-Index Contraction

```rust
/// Contract two `IndexedTensor`s by automatically pairing all index pairs
/// where `a_index.contracts_with(b_index)` is true.
///
/// **Implementation note:** `tk-dsl` has no dependency on `tk-linalg`.
/// The current implementation uses a naive O(n^3) GEMM loop rather than
/// delegating to `tk-contract`'s `ContractionExecutor`. This is acceptable
/// for the small operator matrices (2x2 to 4x4) typical in DSL usage.
/// For performance-critical tensor contractions, use `tk-contract` directly.
///
/// This is a pairwise operation; it does not perform multi-tensor path
/// optimization.
///
/// # Errors
///
/// - `DslError::NoContractingIndices` — no pair `(a_leg, b_leg)` satisfies
///   `contracts_with`; the tensors share no contracted indices.
/// - `DslError::AmbiguousContraction` — the same `IndexId` appears on more
///   than two legs total across both tensors; pairwise contraction cannot
///   resolve this.
/// - `DslError::DimensionMismatch` — a contracted index pair has different
///   `dim` values on the two tensors.
///
/// # Example
///
/// ```rust
/// // A[i, j] contracted with B[j', k] where j.contracts_with(j') is true:
/// let result = contract(&a, &b)?;
/// // result.indices == [i, k]  (free indices, in order A-free then B-free)
/// ```
pub fn contract<T: Scalar>(
    a: &IndexedTensor<T>,
    b: &IndexedTensor<T>,
) -> DslResult<IndexedTensor<T>>;
```

---

## 5. Operator Types

### 5.1 `SpinOp` — Spin-1/2 Operators

```rust
/// Standard spin-1/2 single-site operators.
///
/// These act on a 2-dimensional local Hilbert space with basis {|↑⟩, |↓⟩}.
/// Compile-time enumeration prevents typos and makes the `match` in
/// `tk-dmrg`'s MPO compiler exhaustive and warning-free.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpinOp {
    /// S⁺ = |↑⟩⟨↓|
    SPlus,
    /// S⁻ = |↓⟩⟨↑|
    SMinus,
    /// Sz = ½(|↑⟩⟨↑| − |↓⟩⟨↓|)
    Sz,
    /// Sx = ½(S⁺ + S⁻)
    Sx,
    /// Sy = −i/2 (S⁺ − S⁻) = [0, −i/2; i/2, 0].
    /// **Implementation note (HIGH SEVERITY):** `Sy.matrix::<T>()` requires
    /// `Scalar::from_real_imag` to construct purely imaginary values. For
    /// `T = f64`, `from_real_imag` maps the imaginary part to zero, so
    /// `matrix::<f64>()` returns the zero matrix (correct: Sy has no real
    /// representation). For `T = Complex<f64>`, the full matrix is returned.
    /// The `Scalar` trait must provide `fn from_real_imag(re: Self::Real,
    /// im: Self::Real) -> Self` (see §17.9).
    Sy,
    /// Identity on the 2-dimensional spin-1/2 space.
    Identity,
}

impl SpinOp {
    /// Physical Hilbert space dimension: always 2.
    pub const fn local_dim(self) -> usize { 2 }

    /// Dense matrix representation in row-major order, as a flat `Vec<T>`.
    /// Shape: [2, 2].
    ///
    /// # Note on `Sy` for `T = f64`
    ///
    /// `Sy` has purely imaginary matrix elements. The implementation uses
    /// `T::from_real_imag(re, im)` to construct matrix entries. For
    /// `T = f64`, `from_real_imag` drops the imaginary part and returns
    /// a zero matrix. Use `T = Complex<f64>` for models requiring `Sy`.
    pub fn matrix<T: Scalar>(self) -> Vec<T>;

    /// True iff this operator conserves the total U(1) charge (Sz quantum number).
    /// `SPlus` and `SMinus` change Sz by ±1; all others preserve it.
    pub fn preserves_u1(self) -> bool;

    /// Change in the Sz quantum number due to this operator: +1, -1, or 0.
    pub fn delta_sz(self) -> i32;
}
```

### 5.2 `FermionOp` — Spinful Fermion Operators

```rust
/// Standard spinful fermion single-site operators.
///
/// Act on a 4-dimensional local Hilbert space:
/// {|0⟩, |↑⟩, |↓⟩, |↑↓⟩} (empty, spin-up, spin-down, doubly occupied).
/// Used for Hubbard and Anderson Impurity Model Hamiltonians.
///
/// **Jordan-Wigner strings are NOT embedded in these matrices.** They are
/// encoded at the MPO construction stage in `tk-dmrg`. See design doc §6.4.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FermionOp {
    /// c†_↑: creation operator for spin-up fermion.
    CdagUp,
    /// c_↑: annihilation operator for spin-up fermion.
    CUp,
    /// c†_↓: creation operator for spin-down fermion.
    CdagDn,
    /// c_↓: annihilation operator for spin-down fermion.
    CDn,
    /// n_↑ = c†_↑ c_↑: spin-up occupation number.
    Nup,
    /// n_↓ = c†_↓ c_↓: spin-down occupation number.
    Ndn,
    /// n_total = n_↑ + n_↓: total occupation.
    Ntotal,
    /// Identity on the 4-dimensional local space.
    Identity,
}

impl FermionOp {
    /// Physical Hilbert space dimension: always 4.
    pub const fn local_dim(self) -> usize { 4 }

    /// Dense matrix representation in row-major order, as a flat `Vec<T>`.
    /// Shape: [4, 4].
    pub fn matrix<T: Scalar>(self) -> Vec<T>;

    /// Total particle-number change introduced by this operator:
    /// +1 for creation, -1 for annihilation, 0 for number / identity.
    pub fn delta_n(self) -> i32;

    /// Spin-up particle-number change (+1, -1, or 0).
    pub fn delta_n_up(self) -> i32;

    /// Spin-down particle-number change (+1, -1, or 0).
    pub fn delta_n_dn(self) -> i32;
}
```

### 5.3 `BosonOp` — Bosonic Operators

```rust
/// Bosonic single-site operators for Bose-Hubbard and phonon models.
///
/// The local Hilbert space dimension is `n_max + 1`, where `n_max` is the
/// maximum occupation number determined at model construction time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BosonOp {
    /// b†: bosonic creation operator.
    BDag,
    /// b: bosonic annihilation operator.
    B,
    /// n = b†b: occupation number operator.
    N,
    /// n(n-1): used for Bose-Hubbard interaction U/2 · n(n-1).
    NPairInteraction,
    /// Identity.
    Identity,
}

impl BosonOp {
    /// Dense matrix in the truncated Fock space {|0⟩, ..., |n_max⟩}.
    ///
    /// Returns a flat `Vec<T>` in row-major order; shape: [n_max+1, n_max+1].
    ///
    /// # Panics
    /// Panics in debug mode if `n_max == 0`.
    pub fn matrix<T: Scalar>(self, n_max: usize) -> Vec<T>;
}
```

### 5.4 `CustomOp<T>` — User-Defined Operator

```rust
/// Arbitrary user-defined single-site operator for non-standard models.
///
/// The escape hatch when `SpinOp`, `FermionOp`, and `BosonOp` do not cover
/// the physics (e.g., spin-1 models, Holstein polarons, Kondo models with
/// non-standard local spaces).
///
/// **Implementation note:** `DenseTensor` does not implement `Clone`.
/// `CustomOp` stores `DenseTensor<'static, T>` and provides `clone_owned()`
/// for manual cloning. The `new()` constructor validates that the matrix is
/// square.
///
/// # Example
///
/// ```rust
/// // Spin-1 Sz operator acting on a 3-dimensional local space:
/// let sz1 = CustomOp::new(
///     "Sz_spin1",
///     DenseTensor::from_vec(
///         TensorShape::row_major(&[3, 3]),
///         vec![1.0, 0.0, 0.0,
///              0.0, 0.0, 0.0,
///              0.0, 0.0, -1.0],
///     ),
/// );
/// opsum += J * op(sz1.clone_owned(), i) * op(sz1, i + 1);
/// ```
#[derive(Debug)]
pub struct CustomOp<T: Scalar> {
    /// Square matrix in row-major order. Shape must be `[d, d]` for some `d > 0`.
    matrix: DenseTensor<'static, T>,
    /// Display name used in error messages and debug formatting.
    name: SmallString<[u8; 32]>,
}

impl<T: Scalar> CustomOp<T> {
    /// Construct a new custom operator.
    ///
    /// # Panics
    /// Panics if `matrix` is not square (i.e., shape is not `[d, d]`).
    pub fn new(
        name: impl Into<SmallString<[u8; 32]>>,
        matrix: DenseTensor<'static, T>,
    ) -> Self {
        let dims = matrix.shape().dims();
        assert!(dims.len() == 2 && dims[0] == dims[1], "CustomOp matrix must be square");
        Self { matrix, name: name.into() }
    }

    /// Local Hilbert space dimension: the row count of `matrix`.
    pub fn local_dim(&self) -> usize {
        self.matrix.shape().dims()[0]
    }

    /// Read-only access to the underlying matrix data as a flat slice.
    pub fn matrix_data(&self) -> &[T] {
        self.matrix.as_slice()
    }

    /// Reference to the operator name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Manual clone, required because `DenseTensor` does not implement `Clone`.
    pub fn clone_owned(&self) -> Self {
        Self {
            matrix: DenseTensor::from_vec(
                self.matrix.shape().clone(),
                self.matrix.as_slice().to_vec(),
            ),
            name: self.name.clone(),
        }
    }
}
```

### 5.5 `SiteOperator<T>` — Unified Operator Enum

```rust
/// Unified single-site operator type that subsumes all operator variants.
///
/// The argument to `op(operator, site)`. Implement `Into<SiteOperator<T>>`
/// for any type to make it directly usable in `op(...)` calls.
pub enum SiteOperator<T: Scalar> {
    Spin(SpinOp),
    Fermion(FermionOp),
    /// Bosonic operator with an explicit `n_max` truncation.
    Boson { op: BosonOp, n_max: usize },
    Custom(CustomOp<T>),
}

impl<T: Scalar> SiteOperator<T> {
    /// Local Hilbert space dimension for this operator variant.
    pub fn local_dim(&self) -> usize;

    /// Dense matrix representation in row-major order.
    pub fn matrix(&self) -> Vec<T>;
}

// Blanket From impls for ergonomic usage in op(...):
impl<T: Scalar> From<SpinOp>       for SiteOperator<T> { ... }
impl<T: Scalar> From<FermionOp>    for SiteOperator<T> { ... }
impl<T: Scalar> From<CustomOp<T>>  for SiteOperator<T> { ... }
// Note: BosonOp requires explicit SiteOperator::Boson { op, n_max } construction
// because n_max is not stored in BosonOp itself.
```

---

## 6. `OpTerm<T>`, `OpProduct<T>`, and `ScaledOpProduct<T>`

### 6.1 `op()` Constructor

```rust
/// Construct a single-site operator term for use in `OpSum` expressions.
///
/// This is the primary DSL entry point. The returned `OpTerm<T>` overloads
/// `Mul<OpTerm<T>>` to form multi-site products and `Mul<T>` to apply
/// a scalar coefficient.
///
/// # Parameters
/// - `operator`: Any type implementing `Into<SiteOperator<T>>`, including
///   `SpinOp`, `FermionOp`, `BosonOp` (via explicit `SiteOperator::Boson`),
///   and `CustomOp<T>`.
/// - `site`: Zero-based site index. Validated against lattice bounds at
///   `OpSum::push_term` time (debug mode only).
///
/// # Example
///
/// ```rust
/// // Heisenberg exchange term:
/// opsum += J * op(SpinOp::SPlus, i) * op(SpinOp::SMinus, i + 1);
/// opsum += J * op(SpinOp::SMinus, i) * op(SpinOp::SPlus, i + 1);
/// opsum += Jz * op(SpinOp::Sz, i) * op(SpinOp::Sz, i + 1);
/// ```
pub fn op<T: Scalar>(operator: impl Into<SiteOperator<T>>, site: usize) -> OpTerm<T> {
    OpTerm { operator: operator.into(), site }
}
```

### 6.2 `OpTerm<T>`

```rust
/// A single-site operator reference at a specific site.
///
/// The coefficient is implicitly `T::one()` until scaled by `Mul<T>`.
#[derive(Clone, Debug)]
pub struct OpTerm<T: Scalar> {
    pub operator: SiteOperator<T>,
    pub site: usize,
}

impl<T: Scalar> Mul<OpTerm<T>> for OpTerm<T> {
    type Output = OpProduct<T>;
    fn mul(self, rhs: OpTerm<T>) -> OpProduct<T> {
        OpProduct { factors: smallvec![self, rhs] }
    }
}

impl<T: Scalar> Mul<T> for OpTerm<T> {
    type Output = ScaledOpProduct<T>;
    fn mul(self, scalar: T) -> ScaledOpProduct<T> {
        ScaledOpProduct { coeff: scalar, product: OpProduct { factors: smallvec![self] } }
    }
}
```

### 6.3 `OpProduct<T>` and `ScaledOpProduct<T>`

```rust
/// An ordered product of single-site operators: O₁(i₁) · O₂(i₂) · ... · Oₙ(iₙ).
///
/// Factors are stored in creation order (left-to-right as written).
/// `tk-dmrg` is responsible for inserting Jordan-Wigner strings when compiling
/// this product into an MPO tensor train. See design doc §6.4.
///
/// Sites need not be adjacent; long-range operators are fully supported.
#[derive(Clone, Debug)]
pub struct OpProduct<T: Scalar> {
    /// Ordered list of single-site operators. Invariant: len() >= 1.
    pub factors: SmallVec<[OpTerm<T>; 4]>,
}

impl<T: Scalar> Mul<OpTerm<T>> for OpProduct<T> {
    type Output = OpProduct<T>;
    fn mul(mut self, rhs: OpTerm<T>) -> OpProduct<T> {
        self.factors.push(rhs);
        self
    }
}

impl<T: Scalar> Mul<T> for OpProduct<T> {
    type Output = ScaledOpProduct<T>;
    fn mul(self, coeff: T) -> ScaledOpProduct<T> {
        ScaledOpProduct { coeff, product: self }
    }
}

/// A scalar-weighted operator product: `coeff * O₁(i₁) * O₂(i₂) * ...`
///
/// **Convenience constructors:** The spec originally assumed `single()` and
/// `two_site()` constructors. These do not exist in the draft implementation.
/// Use struct literal construction or the `scaled()` free function instead:
///
/// ```rust
/// // Single-site term:
/// let term = ScaledOpProduct {
///     coeff: 1.0,
///     product: OpProduct { factors: smallvec![op(SpinOp::Sz, 0)] },
/// };
///
/// // Or use the scaled() helper:
/// let term = scaled(1.0, op(SpinOp::Sz, 0) * op(SpinOp::Sz, 1));
/// ```
#[derive(Clone, Debug)]
pub struct ScaledOpProduct<T: Scalar> {
    pub coeff: T,
    pub product: OpProduct<T>,
}

/// Allow `J * op(Sz, i) * op(Sz, j)` by implementing `Mul<OpProduct<T>> for T`.
///
/// **Implementation note (Rust orphan rule limitation):** `impl<T: Scalar>
/// Mul<OpProduct<T>> for T` cannot be written as a blanket impl because `T`
/// is a foreign type. Only concrete implementations are possible. The draft
/// implementation provides `impl Mul<OpProduct<f64>> for f64` (and
/// similarly for `OpTerm`). For `Complex<f64>`, users must use the
/// `scaled()` helper or `OpProduct::scale()` method instead (see below).
impl Mul<OpProduct<f64>> for f64 {
    type Output = ScaledOpProduct<f64>;
    fn mul(self, product: OpProduct<f64>) -> ScaledOpProduct<f64> {
        ScaledOpProduct { coeff: self, product }
    }
}

/// Allow `J * op(Sz, i)` — scalar times a single `OpTerm`.
impl Mul<OpTerm<f64>> for f64 {
    type Output = ScaledOpProduct<f64>;
    fn mul(self, term: OpTerm<f64>) -> ScaledOpProduct<f64> {
        ScaledOpProduct {
            coeff: self,
            product: OpProduct { factors: smallvec![term] },
        }
    }
}

/// Convenience function for types where operator overloading is not
/// available (e.g., `Complex<f64>`). Wraps a coefficient and an operator
/// product into a `ScaledOpProduct`.
///
/// # Example
///
/// ```rust
/// use num_complex::Complex;
/// let t = Complex::new(0.5, 0.1);
/// let term = scaled(t, op(FermionOp::CdagUp, 0) * op(FermionOp::CUp, 1));
/// opsum += term;
/// ```
pub fn scaled<T: Scalar>(coeff: T, product: OpProduct<T>) -> ScaledOpProduct<T> {
    ScaledOpProduct { coeff, product }
}

impl<T: Scalar> OpProduct<T> {
    /// Attach a scalar coefficient to this product.
    ///
    /// Equivalent to `scaled(coeff, self)`. Useful when `T` does not
    /// support `Mul<OpProduct<T>> for T` due to the orphan rule.
    pub fn scale(self, coeff: T) -> ScaledOpProduct<T> {
        ScaledOpProduct { coeff, product: self }
    }
}
```

---

## 7. `OpSum<T>` — Operator Sum Builder

### 7.1 Core Type

```rust
/// A sum of weighted operator products representing a quantum Hamiltonian
/// or observable.
///
/// `OpSum` is an uncompressed, operator-level data structure. It is the
/// sole output that `tk-dsl` produces. Numerical MPO compilation — SVD
/// compression, finite-state automaton minimization — happens in `tk-dmrg`
/// via `OpSum::compile_mpo` (not defined here; defined in `tk-dmrg`).
///
/// # Building an `OpSum` directly
///
/// ```rust
/// let mut h = OpSum::<f64>::new();
/// for i in 0..n - 1 {
///     h += 0.5_f64 * op(SpinOp::SPlus, i) * op(SpinOp::SMinus, i + 1);
///     h += 0.5_f64 * op(SpinOp::SMinus, i) * op(SpinOp::SPlus, i + 1);
///     h += jz * op(SpinOp::Sz, i) * op(SpinOp::Sz, i + 1);
/// }
/// ```
///
/// # Using the macro
///
/// ```rust
/// let h = hamiltonian! {
///     lattice: Chain(N = n, d = 2);
///     sum i in 0..n-1 {
///         0.5 * (SPlus(i) * SMinus(i+1) + SMinus(i) * SPlus(i+1))
///       + jz * Sz(i) * Sz(i+1)
///     }
/// };
/// ```
#[derive(Clone, Debug)]
pub struct OpSum<T: Scalar> {
    terms: Vec<OpSumTerm<T>>,
    /// Optional lattice context for site-bounds validation.
    /// Set by `OpSum::with_lattice` or by `hamiltonian!{}`.
    lattice: Option<Box<dyn Lattice>>,
}

/// One term in an `OpSum`: a coefficient times an ordered product of site operators.
#[derive(Clone, Debug)]
pub struct OpSumTerm<T: Scalar> {
    /// Overall scalar coefficient.
    pub coeff: T,
    /// Ordered product of single-site operators.
    pub product: OpProduct<T>,
}
```

### 7.2 Constructors

```rust
impl<T: Scalar> OpSum<T> {
    /// Create an empty sum with no lattice context.
    pub fn new() -> Self;

    /// Create an empty sum associated with a lattice.
    ///
    /// When a lattice is attached, `push_term` validates that all site
    /// indices in each term are in bounds (`0 <= site < lattice.n_sites()`).
    pub fn with_lattice(lattice: impl Lattice + 'static) -> Self;
}
```

### 7.3 Accumulation and Query

```rust
impl<T: Scalar> OpSum<T> {
    /// Append a `ScaledOpProduct` as a new term.
    ///
    /// # Errors
    ///
    /// Returns `DslError::SiteOutOfBounds` if a lattice is attached and any
    /// site index in `term` is >= `lattice.n_sites()`.
    ///
    /// In production builds (no debug assertions), out-of-bounds sites are
    /// silently accepted; the error is surfaced later at MPO compilation.
    pub fn push_term(&mut self, term: ScaledOpProduct<T>) -> DslResult<()>;

    /// Number of terms in the sum (before any MPO-level simplification).
    ///
    /// **Naming note:** The method is named `n_terms()` rather than `len()`
    /// to avoid confusion with the Rust convention where `len()` implies a
    /// "size" semantic. `n_terms()` clarifies that this counts operator
    /// product terms, not individual site operators.
    pub fn n_terms(&self) -> usize { self.terms.len() }

    /// Iterator over all terms.
    pub fn iter_terms(&self) -> impl Iterator<Item = &OpSumTerm<T>>;

    /// Hermitian conjugate: conjugate all coefficients and reverse all
    /// operator products. Returns a new `OpSum<T>`.
    ///
    /// # Example
    ///
    /// ```rust
    /// // Build a hopping term and its h.c. in two steps:
    /// let hop = {
    ///     let mut s = OpSum::new();
    ///     s += J * op(FermionOp::CdagUp, 0) * op(FermionOp::CUp, 1);
    ///     s
    /// };
    /// let full = hop.clone() + hop.hc();
    /// ```
    pub fn hc(&self) -> OpSum<T>;

    /// Scale all term coefficients by `factor` in place.
    pub fn scale(&mut self, factor: T);

    /// Merge all terms from `other` into self (moves `other`).
    pub fn extend(&mut self, other: OpSum<T>);

    /// Reference to the attached lattice, if any.
    pub fn lattice(&self) -> Option<&dyn Lattice>;
}
```

### 7.4 Operator Overloads

```rust
impl<T: Scalar> AddAssign<ScaledOpProduct<T>> for OpSum<T> {
    fn add_assign(&mut self, rhs: ScaledOpProduct<T>) {
        // Panics in debug mode on site-out-of-bounds; silent in release.
        self.push_term(rhs).unwrap_or_default();
    }
}

impl<T: Scalar> AddAssign<OpSumPair<T>> for OpSum<T> {
    fn add_assign(&mut self, pair: OpSumPair<T>) {
        self.push_term(pair.forward).unwrap_or_default();
        self.push_term(pair.backward).unwrap_or_default();
    }
}

impl<T: Scalar> Add<OpSum<T>> for OpSum<T> {
    type Output = OpSum<T>;
    fn add(mut self, rhs: OpSum<T>) -> OpSum<T> {
        self.extend(rhs);
        self
    }
}

impl<T: Scalar> Mul<T> for OpSum<T> {
    type Output = OpSum<T>;
    fn mul(mut self, factor: T) -> OpSum<T> {
        self.scale(factor);
        self
    }
}
```

### 7.5 Hermitian Conjugate Marker

```rust
/// Zero-size marker type returned by `hc()`.
///
/// When added to a `ScaledOpProduct`, produces an `OpSumPair` containing
/// both the forward term and its Hermitian conjugate. Adding an `OpSumPair`
/// to an `OpSum` atomically inserts both terms, preventing the common error
/// of forgetting to add the h.c. separately.
///
/// # Example
///
/// ```rust
/// // One line adds both the hopping term and its Hermitian conjugate:
/// opsum += t * op(FermionOp::CdagUp, i) * op(FermionOp::CUp, i + 1) + hc();
/// ```
pub struct HermitianConjugate;

/// Convenience constructor for the `+ h.c.` pattern.
pub fn hc() -> HermitianConjugate;

impl<T: Scalar> Add<HermitianConjugate> for ScaledOpProduct<T> {
    type Output = OpSumPair<T>;
    fn add(self, _: HermitianConjugate) -> OpSumPair<T> {
        let backward = self.hermitian_conjugate();
        OpSumPair { forward: self, backward }
    }
}

/// A pair of terms (forward + Hermitian conjugate) for atomic insertion into `OpSum`.
pub struct OpSumPair<T: Scalar> {
    pub forward: ScaledOpProduct<T>,
    pub backward: ScaledOpProduct<T>,
}
```

---

## 8. Lattice Abstraction

### 8.1 `Lattice` Trait

```rust
/// Abstract lattice geometry: the set of sites, bonds, and suggested DMRG
/// site ordering for a physical model.
///
/// Object-safe: `Box<dyn Lattice>` is valid. No generic methods are defined.
///
/// **Clone support via `LatticeClone`:** `OpSum` stores `Box<dyn Lattice>`
/// and requires `Clone`. Because `Clone` is not object-safe, a helper
/// supertrait `LatticeClone` provides `fn clone_box(&self) -> Box<dyn Lattice>`.
/// All `Lattice` implementations that also implement `Clone` receive a
/// blanket `LatticeClone` implementation. An alternative design using
/// `Arc<dyn Lattice>` was considered but deferred (see Open Question §19.8).
pub trait Lattice: LatticeClone + Debug + Send + Sync {
    /// Total number of sites.
    fn n_sites(&self) -> usize;

    /// All nearest-neighbour bonds as `(i, j)` pairs with `i < j`.
    /// The list is pre-computed and stored in the implementing struct;
    /// this method returns a borrowed slice.
    fn bonds(&self) -> &[(usize, usize)];

    /// Suggested DMRG site ordering.
    ///
    /// For 1D lattices, returns `[0, 1, ..., N-1]`.
    /// For 2D lattices (Square, Triangular), returns the snake-path ordering
    /// that minimizes the number of MPO bonds skipping more than one site.
    ///
    /// The return value is a permutation: `ordering[dmrg_pos] = physical_site`.
    fn dmrg_ordering(&self) -> Vec<usize>;

    /// Local Hilbert space dimension if uniform across all sites.
    /// Returns `None` for site-dependent dimensions (e.g., mixed spin/fermion).
    fn local_dim(&self) -> Option<usize> { None }
}

/// Helper trait enabling `Clone` for `Box<dyn Lattice>`.
///
/// Rust's `Clone` trait is not object-safe because `clone()` returns `Self`.
/// `LatticeClone` provides `clone_box()` which returns `Box<dyn Lattice>`.
/// A blanket impl covers all `T: Lattice + Clone`.
pub trait LatticeClone {
    fn clone_box(&self) -> Box<dyn Lattice>;
}

impl<T: Lattice + Clone + 'static> LatticeClone for T {
    fn clone_box(&self) -> Box<dyn Lattice> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Lattice> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
```

### 8.2 `Chain`

```rust
/// 1D open chain with `n` sites.
///
/// - `bonds()` returns `[(0,1), (1,2), ..., (n-2, n-1)]`.
/// - `dmrg_ordering()` returns `[0, 1, ..., n-1]`.
#[derive(Clone, Debug)]
pub struct Chain {
    /// Number of sites. Must be >= 2.
    pub n: usize,
    /// Local Hilbert space dimension per site (e.g., 2 for spin-1/2, 4 for spinful fermions).
    pub d: usize,
    // pre-computed bond list stored privately
    bonds_cache: Vec<(usize, usize)>,
}

impl Chain {
    pub fn new(n: usize, d: usize) -> Self;
}

impl Lattice for Chain {
    fn n_sites(&self) -> usize { self.n }
    fn bonds(&self) -> &[(usize, usize)] { &self.bonds_cache }
    fn dmrg_ordering(&self) -> Vec<usize> { (0..self.n).collect() }
    fn local_dim(&self) -> Option<usize> { Some(self.d) }
}
```

### 8.3 `Square`

```rust
/// 2D square lattice with open boundary conditions: `lx × ly` sites.
///
/// - `bonds()` returns all horizontal and vertical nearest-neighbour bonds.
/// - `dmrg_ordering()` returns the boustrophedon (snake-path) ordering:
///   row 0 left-to-right, row 1 right-to-left, etc.
///   This ordering minimizes long-range coupling penalties in the MPO.
#[derive(Clone, Debug)]
pub struct Square {
    pub lx: usize,
    pub ly: usize,
    pub d: usize,
    bonds_cache: Vec<(usize, usize)>,
}

impl Square {
    pub fn new(lx: usize, ly: usize, d: usize) -> Self;
}

impl Lattice for Square {
    fn n_sites(&self) -> usize { self.lx * self.ly }
    fn bonds(&self) -> &[(usize, usize)] { &self.bonds_cache }
    fn dmrg_ordering(&self) -> Vec<usize> { snake_path(self.lx, self.ly) }
    fn local_dim(&self) -> Option<usize> { Some(self.d) }
}

/// Compute the boustrophedon (snake-path) site ordering for an `lx × ly` grid.
///
/// Returns a permutation `ordering` such that `ordering[dmrg_pos]` is the
/// physical site index at DMRG sweep position `dmrg_pos`.
///
/// Row-major site index: `site(x, y) = y * lx + x`.
///
/// # Example
///
/// For a 3×2 grid (lx=3, ly=2):
/// - Row 0 (y=0): left-to-right → sites 0, 1, 2
/// - Row 1 (y=1): right-to-left → sites 5, 4, 3
///
/// `snake_path(3, 2) == [0, 1, 2, 5, 4, 3]`
pub fn snake_path(lx: usize, ly: usize) -> Vec<usize>;
```

### 8.4 `Triangular`

```rust
/// 2D triangular lattice with open boundary conditions: `lx × ly` sites.
///
/// `bonds()` includes horizontal, vertical, and diagonal (lower-left to
/// upper-right) nearest-neighbour bonds.
/// `dmrg_ordering()` uses the same boustrophedon ordering as `Square`.
#[derive(Clone, Debug)]
pub struct Triangular {
    pub lx: usize,
    pub ly: usize,
    pub d: usize,
    bonds_cache: Vec<(usize, usize)>,
}

impl Triangular {
    pub fn new(lx: usize, ly: usize, d: usize) -> Self;
}

impl Lattice for Triangular { /* standard Lattice impl */ }
```

### 8.5 `BetheLattice`

```rust
/// Cayley tree (Bethe lattice) with coordination number `z` and `depth` shells.
///
/// Total site count: 1 + z * sum_{k=0}^{depth-2} (z-1)^k (for depth >= 1).
/// `dmrg_ordering()` uses breadth-first ordering with the root at position 0.
#[derive(Clone, Debug)]
pub struct BetheLattice {
    /// Coordination number: number of neighbours of each interior node.
    pub z: usize,
    /// Number of shells from the root.
    pub depth: usize,
    pub d: usize,
    sites_cache: usize,
    bonds_cache: Vec<(usize, usize)>,
}

impl BetheLattice {
    pub fn new(z: usize, depth: usize, d: usize) -> Self;
}

impl Lattice for BetheLattice { /* standard Lattice impl */ }
```

### 8.6 `StarGeometry` — Anderson Impurity Model

```rust
/// Star geometry: one impurity site (site 0) connected to `n_bath` bath sites
/// (sites 1..=n_bath). No bath-bath bonds.
///
/// Represents the Anderson Impurity Model (AIM) before the Lanczos
/// tridiagonalization in `tk-dmft` maps it to a `Chain`. The `StarGeometry`
/// is the natural description at the `tk-dsl` / `tk-dmrg` level.
///
/// `dmrg_ordering()` places the impurity at the center:
///   [..., bath_right_k, ..., bath_right_1, impurity, bath_left_1, ..., bath_left_k, ...]
/// for `n_bath = 2k` (even), minimizing long-range hopping penalties in the MPO.
/// For odd `n_bath`, the impurity is placed at `(n_bath+1)/2`.
#[derive(Clone, Debug)]
pub struct StarGeometry {
    /// Number of bath sites.
    pub n_bath: usize,
    /// Local Hilbert space dimension per site.
    /// For spinful AIM: 4 (empty, spin-up, spin-down, doubly occupied).
    pub d: usize,
    bonds_cache: Vec<(usize, usize)>,
}

impl StarGeometry {
    pub fn new(n_bath: usize, d: usize) -> Self;
}

impl Lattice for StarGeometry {
    fn n_sites(&self) -> usize { self.n_bath + 1 }
    fn bonds(&self) -> &[(usize, usize)] { &self.bonds_cache }
    fn dmrg_ordering(&self) -> Vec<usize>;
    fn local_dim(&self) -> Option<usize> { Some(self.d) }
}
```

---

## 9. The `hamiltonian!{}` Proc-Macro

### 9.1 Purpose and Scope

**Implementation status:** The proc-macro crate (`tk-dsl-macros`) is **deferred** and not yet implemented. The specification below describes the intended design for a future phase.

The `hamiltonian!` proc-macro is a **pure syntax-level transformation**. It parses a lattice/Hamiltonian DSL at compile time and emits Rust statements that construct an `OpSum<T>` at runtime. No numerical computation occurs at compile time.

**What `hamiltonian!` does:**
- Parses the `lattice:`, `sum ... in ... { }`, and operator shorthand syntax.
- Resolves operator shortnames to `SpinOp::`, `FermionOp::`, `BosonOp::` enum variants.
- Validates: all identifiers resolve to known operator names; `h.c.` appears only in an additive context; the `lattice:` clause is present and appears first.
- Emits a Rust block expression that evaluates to an `OpSum<T>`.

**What `hamiltonian!` does NOT do:**
- No SVD, no matrix construction, no MPO compression.
- No compile-time evaluation of runtime parameters (`J`, `U`, `t`, `eps[k]`, `V[k]`).
- No resolution of lattice size `N` as a compile-time constant; `N` is a runtime variable captured from the enclosing scope.

### 9.2 Grammar

The macro input follows this grammar (EBNF notation):

```ebnf
hamiltonian_input ::=
    lattice_decl ";"
    term_block+

lattice_decl ::=
    "lattice" ":" lattice_spec

lattice_spec ::=
    "Chain"       "(" "N" "=" expr "," "d" "=" expr ")"
  | "Square"      "(" "Lx" "=" expr "," "Ly" "=" expr "," "d" "=" expr ")"
  | "Triangular"  "(" "Lx" "=" expr "," "Ly" "=" expr "," "d" "=" expr ")"
  | "Star"        "(" "n_bath" "=" expr "," "d" "=" expr ")"
  | "Bethe"       "(" "z" "=" expr "," "depth" "=" expr "," "d" "=" expr ")"
  | expr           (* arbitrary Rust expression implementing Lattice *)

term_block ::=
    sum_block
  | single_term ";"

sum_block ::=
    "sum" ident "in" range_expr "{" term_line+ "}"

range_expr ::= expr ".." expr | expr "..=" expr

term_line ::=
    term_expr ("+" term_expr)* ";"?

term_expr ::=
    scalar_expr "*" operator_product
  | scalar_expr "*" "(" sum_of_products ")"
  | operator_product
  | term_expr "+ h.c."

sum_of_products ::=
    operator_product ("+" operator_product)*

operator_product ::=
    operator_call ("*" operator_call)*

operator_call ::=
    spin_op_name     "(" site_expr ")"
  | fermion_op_name  "(" site_expr ")"
  | boson_op_name    "(" site_expr "," "n_max" "=" expr ")"
  | ident            "(" site_expr ")"   (* CustomOp lookup from enclosing scope *)

spin_op_name    ::= "SPlus" | "SMinus" | "Sz" | "Sx" | "Sy" | "Id"
fermion_op_name ::= "CdagUp" | "CUp" | "CdagDn" | "CDn" | "Nup" | "Ndn" | "Ntotal"
boson_op_name   ::= "BDag" | "B" | "N" | "NPair"

site_expr   ::= expr   (* any Rust expression evaluating to usize *)
scalar_expr ::= expr   (* any Rust expression evaluating to T *)
```

### 9.3 Expanded Code Shape

Given input:

```rust
let h = hamiltonian! {
    lattice: Chain(N = n, d = 2);
    sum i in 0..n-1 {
        J * (SPlus(i) * SMinus(i+1) + SMinus(i) * SPlus(i+1))
      + Jz * Sz(i) * Sz(i+1)
    }
    sum i in 0..n { h_field * Sz(i) }
};
```

The macro emits the equivalent of:

```rust
let h = {
    let __lattice = Chain::new(n, 2);
    let mut __opsum = OpSum::<_>::with_lattice(__lattice);
    for i in (0usize)..(n - 1) {
        __opsum += J * (op(SpinOp::SPlus, i) * op(SpinOp::SMinus, i + 1));
        __opsum += J * (op(SpinOp::SMinus, i) * op(SpinOp::SPlus, i + 1));
        __opsum += Jz * op(SpinOp::Sz, i) * op(SpinOp::Sz, i + 1);
    }
    for i in (0usize)..(n) {
        __opsum += h_field * op(SpinOp::Sz, i);
    }
    __opsum
};
```

### 9.4 Anderson Impurity Model Example

```rust
let h_aim = hamiltonian! {
    lattice: Star(n_bath = n_bath, d = 4);
    U * Nup(0) * Ndn(0)
    eps_imp * (Nup(0) + Ndn(0))
    sum k in 1..=n_bath {
        V[k-1] * (CdagUp(0) * CUp(k) + h.c.)
      + V[k-1] * (CdagDn(0) * CDn(k) + h.c.)
      + eps[k-1] * (Nup(k) + Ndn(k))
    }
};
```

Here `V` and `eps` are `&[f64]` slices from the enclosing scope; `V[k-1]` and `eps[k-1]` are ordinary Rust index expressions evaluated inside the generated loop body.

### 9.5 Compile-Time Diagnostics

The proc-macro emits `compile_error!` with source-span accuracy for the following conditions:

| Error Condition | Diagnostic Message |
|:----------------|:-------------------|
| Unknown operator name | `error: unknown operator 'Foop'; expected one of: SPlus, SMinus, Sz, Sx, Sy, Id, CdagUp, CUp, ...` |
| `lattice:` clause missing | `error: hamiltonian! requires a 'lattice:' clause as the first item` |
| Duplicate `lattice:` clause | `error: duplicate 'lattice:' declaration; only one lattice may be specified` |
| `h.c.` in non-additive context | `error: 'h.c.' must appear as '+ h.c.' in an additive expression` |
| Range step other than 1 | `error: only unit-step ranges ('a..b', 'a..=b') are supported; use an explicit for loop for non-unit strides` |
| Malformed boson call | `error: boson operator requires explicit n_max: use BDag(site, n_max = N)` |

All diagnostics point to the token that caused the error, not to the macro invocation site.

### 9.6 Proc-Macro Implementation Notes

- **Parser:** Uses `syn 2.x` for tokenization and `proc_macro2::TokenStream` for output. Internal types (`HamiltonianInput`, `LatticeSpec`, `SumBlock`, `TermExpr`) implement `syn::parse::Parse`.
- **Operator resolution:** Operator shortnames (e.g., `SPlus`) are resolved to `SpinOp::SPlus` at macro-parse time via a static lookup table, catching misspellings before code generation.
- **Hygiene:** All generated identifiers (`__lattice`, `__opsum`, loop variable using the user-provided `ident`) are created with `proc_macro2::Span::call_site()` to prevent inadvertent shadowing of user-defined variables.
- **`h.c.` expansion:** `A + h.c.` is expanded to two separate `__opsum +=` statements: one for the forward term and one for its Hermitian conjugate (with conjugated coefficient and reversed operator order).
- **Site expression casting:** Loop variables and site expressions are cast to `usize` via `as usize` in the emitted code to prevent signed-integer underflow in expressions like `i - 1` at `i = 0`.

---

## 10. Error Handling

### 10.1 Error Type

```rust
/// All errors produced by `tk-dsl` at runtime.
///
/// Compile-time macro errors are emitted as `compile_error!` invocations,
/// not as instances of `DslError`.
#[derive(Debug, thiserror::Error)]
pub enum DslError {
    #[error("duplicate index tag '{tag}': each tag must be unique within a registry")]
    DuplicateIndexTag { tag: String },

    #[error("dimension mismatch on contracting index '{tag}': {dim_a} != {dim_b}")]
    DimensionMismatch { tag: String, dim_a: usize, dim_b: usize },

    #[error("no contracting indices found between the two tensors")]
    NoContractingIndices,

    #[error(
        "ambiguous contraction: index '{tag}' appears on more than two legs; \
         pairwise contraction requires exactly two legs per contracted index"
    )]
    AmbiguousContraction { tag: String },

    #[error("site index {site} out of bounds for lattice with {n_sites} sites")]
    SiteOutOfBounds { site: usize, n_sites: usize },

    #[error(
        "operator local_dim mismatch: operator '{name}' expects dim {expected}, \
         lattice provides dim {got}"
    )]
    LocalDimMismatch { name: String, expected: usize, got: usize },

    #[error("empty operator product: OpProduct must contain at least one OpTerm")]
    EmptyProduct,

    #[error("boson operator requires n_max > 0")]
    InvalidBosonNMax,

    #[error(transparent)]
    Core(#[from] tk_core::TkError),
}

pub type DslResult<T> = Result<T, DslError>;
```

### 10.2 Error Propagation Strategy

- Functions that can fail at runtime (`IndexRegistry::register`, `contract()`, `OpSum::push_term`) return `DslResult<T>`.
- `AddAssign` implementations on `OpSum` call `unwrap_or_default()` (which is a no-op for `()`) so that the `+=` syntax does not require the user to handle `Result`. In debug builds, `push_term` is called with an `expect` so that out-of-bounds site indices panic immediately with a clear message.
- The `hamiltonian!` macro never produces `DslError` at runtime; all detectable errors become `compile_error!` at macro expansion time.

---

## 11. Public API Surface

```rust
// tk-dsl/src/lib.rs

pub mod index;
pub mod indexed_tensor;
pub mod operators;
pub mod opterm;
pub mod opsum;
pub mod lattice;
pub mod error;

// Flat re-exports for ergonomic downstream use:
pub use index::{Index, IndexDirection, IndexRegistry};
pub use indexed_tensor::{IndexedTensor, contract};
pub use operators::{SpinOp, FermionOp, BosonOp, CustomOp, SiteOperator};
pub use opterm::{op, scaled, OpTerm, OpProduct, ScaledOpProduct};
pub use opsum::{OpSum, OpSumTerm, OpSumPair, HermitianConjugate, hc};
pub use lattice::{
    Lattice, LatticeClone,
    Chain, Square, Triangular, BetheLattice, StarGeometry,
    snake_path,
};
pub use error::{DslError, DslResult};

// Re-export IndexId from tk-contract (thin coupling; see §15 and §19.1):
pub use tk_contract::IndexId;

// Re-export the hamiltonian! proc-macro from the companion crate (deferred):
// pub use tk_dsl_macros::hamiltonian;
```

---

## 12. Feature Flags

| Flag | Effect in `tk-dsl` |
|:-----|:-------------------|
| `su2-symmetry` | Enables validation of `IndexDirection::Incoming` / `Outgoing` against `SU2Irrep` quantum number assignments; transitively enables `tk-symmetry/su2-symmetry` |
| `parallel` | No-op feature flag. Originally intended to propagate to `tk-core/parallel`, but `tk-core` does not define a `parallel` feature. Retained for forward compatibility; declaring it in `tk-dsl` prevents downstream breakage if `tk-core` adds the feature later |

`tk-dsl` does not use any backend feature flags (`backend-faer`, `backend-mkl`, `backend-openblas`, `backend-cuda`). It has no linear algebra dependency.

---

## 13. Build-Level Concerns

`tk-dsl/build.rs` performs one check: the `tk-dsl-macros` crate version must exactly match the `tk-dsl` crate version. A mismatch can produce cryptic type errors (e.g., `OpSum` from one version is not the `OpSum` expected by a macro generated against another); this build check converts the symptom into an explicit diagnostic.

```rust
// tk-dsl/build.rs
fn main() {
    // Cargo sets this variable when tk-dsl-macros is a path dependency:
    // CARGO_PKG_VERSION from the companion crate is not directly available,
    // but mismatched versions are caught by Cargo's semver resolver for
    // published crates. This check targets in-workspace path dependencies.
    println!("cargo:rerun-if-changed=../tk-dsl-macros/Cargo.toml");
}
```

For released crates, `tk-dsl` declares `tk-dsl-macros = { version = "=0.1.0" }` with an **exact** semver constraint (`=`), preventing Cargo from silently resolving to a different patch of the macros crate.

---

## 14. Internal Helpers

### 14.1 `ScaledOpProduct::hermitian_conjugate`

```rust
impl<T: Scalar> ScaledOpProduct<T> {
    /// Return the Hermitian conjugate of this term.
    ///
    /// Conjugation rules applied in order:
    ///  1. Coefficient: `coeff.conj()` (no-op for real `T`).
    ///  2. Factor order: reversed (O₁ · O₂ → O₂† · O₁†).
    ///  3. Each operator conjugated:
    ///     - `SpinOp::SPlus`  ↔  `SpinOp::SMinus`
    ///     - `SpinOp::Sz`, `Sx`, `Sy`, `Identity` are self-adjoint
    ///     - `FermionOp::CdagUp` ↔ `FermionOp::CUp`
    ///     - `FermionOp::CdagDn` ↔ `FermionOp::CDn`
    ///     - `FermionOp::Nup`, `Ndn`, `Ntotal`, `Identity` are self-adjoint
    ///     - `BosonOp::BDag` ↔ `BosonOp::B`
    ///     - `BosonOp::N`, `NPairInteraction`, `Identity` are self-adjoint
    ///     - `CustomOp`: `matrix` is explicitly conjugate-transposed via `matview`
    pub(crate) fn hermitian_conjugate(&self) -> Self;
}
```

### 14.2 `SiteOperator::adjoint`

```rust
impl<T: Scalar> SiteOperator<T> {
    /// Return the Hermitian conjugate of this single-site operator.
    /// Used by `ScaledOpProduct::hermitian_conjugate`.
    pub(crate) fn adjoint(&self) -> Self;
}
```

### 14.3 `snake_path` (re-exported from `lattice::square`)

See §8.3. Made `pub` so that custom `Lattice` implementations can reuse the standard snake-path algorithm without duplicating it.

---

## 15. Dependencies and Integration

```toml
[package]
name    = "tk-dsl"
version = "0.1.0"
edition = "2021"

[dependencies]
tk-core         = { path = "../tk-core" }
tk-symmetry     = { path = "../tk-symmetry" }
tk-contract     = { path = "../tk-contract" }   # for IndexId re-export only
tk-dsl-macros   = { path = "./tk-dsl-macros", version = "=0.1.0" }

smallvec  = "1"
smallstr  = "0.3"     # SmallString<[u8; 32]>: zero-alloc tags up to 32 bytes
thiserror = "1"

[features]
default      = ["parallel"]
parallel     = []   # no-op: tk-core/parallel does not exist yet; retained for forward compatibility
su2-symmetry = ["tk-symmetry/su2-symmetry"]

[dev-dependencies]
trybuild = "1"
proptest = "1"

# ---- tk-dsl-macros/Cargo.toml ----
[package]
name    = "tk-dsl-macros"
version = "0.1.0"
edition = "2021"

[lib]
proc-macro = true

[dependencies]
syn         = { version = "2", features = ["full", "extra-traits"] }
quote       = "1"
proc-macro2 = "1"
```

**`tk-contract` coupling note:** `tk-dsl` depends on `tk-contract` exclusively to re-export `IndexId`. This is a narrow coupling; `IndexId` could be moved to `tk-core` to sever it entirely. See Open Question §19.1.

---

## 16. Testing Strategy

### 16.1 Unit Tests

| Test | Description |
|:-----|:------------|
| `index_prime_increments` | `i.prime().prime_level() == i.prime_level() + 1`; `i.unprime().prime_level() == 0` |
| `index_contracts_with_prime` | `i.contracts_with(i.prime()) == true`; `i.contracts_with(i) == false` |
| `index_same_id_different_prime` | `i.same_id(i.prime_n(5)) == true` |
| `index_registry_unique_ids` | Two `register` calls with different tags produce different `IndexId`s |
| `index_registry_duplicate_tag_error` | Re-registering the same tag returns `DslError::DuplicateIndexTag` |
| `opsum_push_in_bounds` | Adding a term with `site < n_sites` returns `Ok(())` |
| `opsum_push_out_of_bounds` | Adding a term with `site >= n_sites` returns `DslError::SiteOutOfBounds` |
| `opsum_hc_spin_plus_minus` | `hc()` of `J * op(SPlus, i) * op(SMinus, j)` produces `J.conj() * op(SPlus, j) * op(SMinus, i)` |
| `opsum_hc_fermion_creation` | `hc()` of `CdagUp(0) * CUp(1)` yields `CdagUp(1) * CUp(0)` |
| `opsum_hc_involution` | `opsum.hc().hc()` has same term count and coefficients as original |
| `opsum_hc_self_adjoint_sz` | `hc()` of `Sz(i)` produces `Sz(i)` with conjugated coefficient |
| `opsum_hc_custom` | `hc()` of a `CustomOp` produces the correct conjugate-transpose matrix |
| `opsum_scale` | After `opsum.scale(2.0)`, all coefficients are doubled |
| `opsum_extend` | `a.extend(b)` results in `a.n_terms() == old_a_terms + b_terms` |
| `hc_marker_adds_two_terms` | `opsum += J * op(CdagUp, 0) * op(CUp, 1) + hc()` adds exactly 2 terms |
| `spin_op_matrix_splus` | `SpinOp::SPlus.matrix::<f64>()` returns `[0.0, 1.0, 0.0, 0.0]` (row-major 2×2) |
| `spin_op_matrix_sz` | `SpinOp::Sz.matrix::<f64>()` returns `[0.5, 0.0, 0.0, -0.5]` |
| `spin_op_delta_sz` | `SPlus.delta_sz() == 1`; `SMinus.delta_sz() == -1`; `Sz.delta_sz() == 0` |
| `fermion_op_matrix_nup` | `FermionOp::Nup.matrix::<f64>()` has 1.0 in the (|↑⟩, |↑⟩) and (|↑↓⟩, |↑↓⟩) entries |
| `fermion_op_delta_n` | `CdagUp.delta_n_up() == 1`; `CUp.delta_n_up() == -1`; `Nup.delta_n() == 0` |
| `boson_op_matrix_bdag_n3` | `BosonOp::BDag.matrix::<f64>(3)` matches analytic `sqrt(n)` values |
| `boson_op_n_max_zero_panics` | `BosonOp::B.matrix::<f64>(0)` panics in debug mode |
| `custom_op_local_dim` | `CustomOp` with 3×3 matrix reports `local_dim() == 3` |
| `chain_bonds_count` | `Chain::new(5, 2).bonds().len() == 4` |
| `chain_dmrg_ordering_identity` | `Chain::new(5, 2).dmrg_ordering() == vec![0, 1, 2, 3, 4]` |
| `square_bonds_count` | `Square::new(3, 3, 2).bonds().len() == 12` |
| `snake_path_correctness` | `snake_path(3, 3)` matches known expected ordering row-by-row |
| `snake_path_bijection_3x3` | `snake_path(3, 3)` contains each of 0..9 exactly once |
| `star_geometry_bonds_star_only` | `StarGeometry::new(4, 4).bonds()` contains only `(0, k)` pairs |
| `star_geometry_dmrg_ordering_center` | Impurity at position `n_bath / 2` in the ordering |
| `contract_matching_indices_f64` | `contract(&a, &b)` correctly identifies and contracts matching index |
| `contract_no_indices_error` | `contract(&a, &b)` with no matching indices -> `DslError::NoContractingIndices` |
| `contract_dim_mismatch_error` | Matching indices with different dims -> `DslError::DimensionMismatch` |
| `op_scaled_by_scalar` | `2.0_f64 * op(SpinOp::Sz, 0)` produces `ScaledOpProduct { coeff: 2.0, ... }` |
| `op_product_length` | `op(Sz, i) * op(Sz, j)` produces `OpProduct` with `factors.len() == 2` |

### 16.2 Draft Implementation Test Status

The draft implementation has **37 passing tests** covering the core abstractions: `Index`, `IndexedTensor`, `SpinOp`, `FermionOp`, `BosonOp`, `CustomOp`, `OpTerm`, `OpProduct`, `ScaledOpProduct`, `OpSum`, and all `Lattice` implementations. The proc-macro crate (`tk-dsl-macros`) is **deferred** and has no tests yet.

### 16.3 Macro Expansion Tests (`trybuild`) — Deferred

Located in `tests/ui/`. Each file is a self-contained Rust snippet compiled by `trybuild`. **Not yet implemented** — the proc-macro crate is deferred to a future phase.

**Pass cases (`tests/ui/pass/`):**

| File | Validates |
|:-----|:----------|
| `heisenberg_chain.rs` | Chain with `SPlus`/`SMinus`/`Sz`, runtime `N` and `J` |
| `aim_star.rs` | `StarGeometry` with `FermionOp` + `h.c.` + array coupling |
| `hubbard_chain.rs` | `Nup*Ndn` on-site + `CdagUp*CUp` hopping + `h.c.` |
| `bose_hubbard.rs` | `BosonOp::BDag * B` nearest-neighbour + `N*N` on-site with explicit `n_max` |
| `custom_op.rs` | `CustomOp` from enclosing scope used in the macro body |
| `runtime_n.rs` | `N` is a runtime `usize` variable, not a literal; range expressions work |
| `nested_sum.rs` | Two separate `sum` blocks in one macro invocation |

**Fail cases (`tests/ui/fail/`):**

| File | Expected error |
|:-----|:---------------|
| `unknown_op_name.rs` | `error: unknown operator 'Foop'` |
| `missing_lattice.rs` | `error: hamiltonian! requires a 'lattice:' clause` |
| `duplicate_lattice.rs` | `error: duplicate 'lattice:' declaration` |
| `hc_non_additive.rs` | `error: 'h.c.' must appear as '+ h.c.'` |
| `boson_missing_nmax.rs` | `error: boson operator requires explicit n_max` |

### 16.4 Property-Based Tests

```rust
proptest! {
    #[test]
    fn opsum_hc_involution(
        n in 2usize..=20,
        j in 0.5f64..=2.0,
    ) {
        let mut opsum = OpSum::<f64>::new();
        for i in 0..n-1 {
            opsum += j * op(SpinOp::SPlus, i) * op(SpinOp::SMinus, i + 1);
        }
        let double_hc = opsum.hc().hc();
        prop_assert_eq!(opsum.n_terms(), double_hc.n_terms());
        for (orig, recovered) in opsum.iter_terms().zip(double_hc.iter_terms()) {
            prop_assert!((orig.coeff - recovered.coeff).abs() < 1e-14);
        }
    }

    #[test]
    fn opsum_scale_distributes(
        n in 2usize..=20,
        j in 0.5f64..=2.0,
        scale in 0.1f64..=10.0,
    ) {
        let mut opsum = OpSum::<f64>::new();
        for i in 0..n-1 {
            opsum += j * op(SpinOp::Sz, i) * op(SpinOp::Sz, i + 1);
        }
        let scaled = opsum.clone() * scale;
        for (orig, sc) in opsum.iter_terms().zip(scaled.iter_terms()) {
            prop_assert!((sc.coeff - orig.coeff * scale).abs() < 1e-14);
        }
    }

    #[test]
    fn chain_bonds_count_property(n in 2usize..=100) {
        let chain = Chain::new(n, 2);
        prop_assert_eq!(chain.bonds().len(), n - 1);
    }

    #[test]
    fn snake_path_bijection(lx in 2usize..=8, ly in 2usize..=8) {
        let ordering = snake_path(lx, ly);
        let total = lx * ly;
        prop_assert_eq!(ordering.len(), total);
        let mut seen = vec![false; total];
        for &s in &ordering { seen[s] = true; }
        prop_assert!(seen.iter().all(|&v| v), "snake_path is not a bijection");
    }

    #[test]
    fn index_prime_round_trip(level in 0u32..=10) {
        let i = Index::new("test", 4, IndexDirection::None);
        let primed = i.prime_n(level);
        prop_assert_eq!(primed.prime_level(), level);
        prop_assert_eq!(primed.unprime().prime_level(), 0);
        prop_assert_eq!(i.id(), primed.id());
    }
}
```

### 16.5 Integration Test Contracts

Integration tests combining `tk-dsl` with `tk-dmrg` live in `tests/` at the workspace root and are the responsibility of `tk-dmrg`'s test suite. From `tk-dsl`'s perspective, the following contracts must hold:

1. Any `OpSum<f64>` produced by `hamiltonian!` for a `Chain` with N sites and Heisenberg interactions must contain exactly `3*(N-1)` terms (J/2 · SPlus·SMinus + J/2 · SMinus·SPlus + Jz · Sz·Sz per bond).
2. Any `OpSum<f64>` produced for an Anderson Impurity Model `StarGeometry` with `n_bath` bath sites must contain: 1 Hubbard term + 2 on-site impurity terms + 4·n_bath hopping terms (2 spin × 2 for h.c.) + 2·n_bath bath energy terms.
3. Round-trip: `hamiltonian!` output passed to `OpSum::compile_mpo` in `tk-dmrg` must not error for any lattice type and any combination of `SpinOp`, `FermionOp`, or `BosonOp` operators supported by the DSL.

---

## 17. Implementation Notes and Design Decisions

### 17.1 No `tk-linalg` Dependency (Cyclic Prevention)

The absence of `tk-linalg` in `tk-dsl`'s dependency list is a load-bearing architectural constraint documented in design doc §2.2. `tk-dmrg` depends on both `tk-dsl` (for `OpSum`) and `tk-linalg` (for MPO compilation via SVD). Allowing `tk-dsl` to depend on `tk-linalg` would create a diamond dependency that needlessly invalidates `tk-dsl`'s compilation cache whenever a BLAS backend changes. The constraint is enforced by omitting `tk-linalg` from `tk-dsl/Cargo.toml` entirely.

### 17.2 `SmallString<[u8; 32]>` for Tags

Index tags, operator names, and `CustomOp` names use `SmallString<[u8; 32]>` from the `smallstr` crate rather than `String`. Physical model identifiers are typically 2-12 characters ("phys", "bond_L", "Sz_spin1"). The 32-byte inline buffer avoids heap allocation for all realistic tags. The `smallstr 0.3` dependency has been validated in the draft implementation and works well for tags under 32 bytes, consistent with `tk-core`'s use of `SmallVec<[usize; 6]>` for tensor shapes.

### 17.3 Prime-Level Convention

`prime_level` follows the ITensor convention: level 0 is the "ket" (incoming) copy; level 1 is the "bra" (outgoing) copy. `contracts_with` checks that levels differ by exactly one, not that they are 0 and 1 specifically, so that higher prime levels (2, 3, ...) can represent multi-time-step expressions or power applications without special-casing.

### 17.4 `SiteOperator<T>` Generics and Monomorphization

`SiteOperator<T>` is generic over `T: Scalar`. The entire `OpSum<T>` machinery is monomorphized once per `T`. For the common case `T = f64` there is one instantiation; for `T = Complex<f64>` (complex hopping in DMFT) there is a separate one. The design avoids type-erased `Box<dyn Any>` storage, which would require unsafe downcasting in `matrix()` and heap-allocate every operator term.

### 17.5 `HermitianConjugate` Marker and Atomic `+ h.c.` Insertion

The `hc()` function returns a zero-size marker struct. `Add<HermitianConjugate>` on `ScaledOpProduct` returns `OpSumPair`, and `AddAssign<OpSumPair>` on `OpSum` inserts both forward and conjugate terms atomically. This prevents the bug where a user writes `opsum += J * term; opsum += hc();` — the latter is a type error because `HermitianConjugate` does not implement `AddAssign<OpSum>`.

### 17.6 Bosonic `n_max` as Runtime Parameter

`BosonOp` does not store `n_max`; instead it appears in `SiteOperator::Boson { op, n_max }`. This is necessary because `n_max` is determined by convergence studies at runtime. The matrix is constructed once per unique `(BosonOp, n_max)` pair during MPO compilation in `tk-dmrg`, not in hot loops.

### 17.7 Proc-Macro Crate Separation

Rust requires proc-macro crates to be separate compilation units. `tk-dsl-macros` is the proc-macro implementation; `tk-dsl` re-exports `hamiltonian!` from it. The exact-version constraint `version = "=0.1.0"` in `Cargo.toml` prevents Cargo from silently resolving to a different patch of the macros crate, which could produce type mismatch errors between the generated code and the runtime types.

### 17.8 `contract()` Uses Naive GEMM

The draft implementation of `IndexedTensor::contract()` uses a naive O(n^3) triple-loop GEMM rather than delegating to `tk-contract`'s `ContractionExecutor`. This is a consequence of the no-linalg rule (§17.1): `tk-dsl` has no dependency on `tk-linalg` or `tk-contract`'s execution layer. The naive implementation is acceptable for the small operator matrices (2x2 spin, 4x4 fermion) typical in DSL usage. For large tensor contractions, users should use `tk-contract` directly.

### 17.9 `Scalar::from_real_imag` Requirement

The draft implementation revealed that the `Scalar` trait in `tk-core` lacks a constructor for purely imaginary values. `SpinOp::Sy` requires matrix elements `[0, -i/2; i/2, 0]`, but without `from_real_imag`, the `matrix::<T>()` method cannot construct these entries generically. The required addition to the `Scalar` trait:

```rust
pub trait Scalar: ... {
    /// Construct a scalar from real and imaginary parts.
    ///
    /// For real types (`f32`, `f64`): returns `re`, discarding `im`.
    /// For complex types (`Complex<f32>`, `Complex<f64>`): returns `Complex { re, im }`.
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self;
}
```

Without this method, `Sy.matrix::<Complex<f64>>()` returns zeros, which is **incorrect**. This is a high-severity issue that requires updating the `tk-core` tech spec. See `tk-core` tech spec for the corresponding change.

### 17.10 `DenseTensor` Does Not Implement `Clone`

`DenseTensor` in `tk-core` does not derive or implement `Clone`. Both `IndexedTensor<T>` and `CustomOp<T>` store `DenseTensor<'static, T>` and need to be cloneable. The workaround is a `clone_owned()` method that performs:

```rust
DenseTensor::from_vec(self.data.shape().clone(), self.data.as_slice().to_vec())
```

This pattern is used consistently across `IndexedTensor` and `CustomOp`. A future `tk-core` update adding `Clone` to `DenseTensor` would eliminate the need for this workaround.

### 17.11 Operator Overloading Restricted to `f64`

Due to Rust's orphan rule, `impl<T: Scalar> Mul<OpTerm<T>> for T` is not expressible when `T` is a foreign type. The draft implementation provides `Mul` impls only for `f64`. For `Complex<f64>` users, the `scaled(coeff, product)` free function and `OpProduct::scale(coeff)` method serve as workarounds. This is documented in §6.3.

---

## 18. Out of Scope

The following are explicitly **not** implemented in `tk-dsl`:

- `OpSum -> MPO` compilation, SVD compression, finite-state automaton minimization (-> `tk-dmrg`)
- BLAS, SVD, or eigenvalue operations of any kind (-> `tk-linalg`)
- MPS or MPO data structures (-> `tk-dmrg`)
- Block-sparse tensor arithmetic (-> `tk-linalg`, `tk-contract`)
- Iterative eigensolvers: Lanczos, Davidson, Block-Davidson (-> `tk-dmrg`)
- TDVP time-evolution, TEBD operators, or bath discretization (-> `tk-dmft`)
- Python bindings (-> `tk-python`)
- Lattice geometry for tree tensor networks or 2D PEPS (-> Phase 5+)
- Fermionic swap gate or Jordan-Wigner string insertion (-> `tk-dmrg`, design doc §6.4)
- Hamiltonian symmetry analysis or quantum number assignment (site quantum numbers are supplied at `BlockSparseTensor` construction in `tk-dmrg`, not here) (-> `tk-dmrg`)

---

## 19. Open Questions

| # | Question | Status |
|:--|:---------|:-------|
| 1 | Should `IndexId` be moved to `tk-core` instead of `tk-contract`, so that `tk-dsl`'s only linalg-adjacent dependency is severed? This would let `tk-dsl` compile independently of `tk-contract` changes. Cost: a one-line change to `tk-core` and a re-export update in `tk-contract`. | Deferred; assess after initial implementations of both crates |
| 2 | Should `OpSum<T>` include an in-place simplification pass (combining terms with identical operator products by summing their coefficients)? This would reduce term count before MPO compilation. Cost: O(N² log N) de-duplication in `tk-dsl`. Alternative: a pre-pass inside `OpSum::compile_mpo` in `tk-dmrg`. | Deferred — defer to `tk-dmrg` as an MPO compiler pre-pass |
| 3 | Should `hamiltonian!` support `next_nearest_neighbour` and `all_pairs` keywords as sugar for common double-sum patterns in frustrated magnetism models? | Deferred — survey user demand before Phase 2 |
| 4 | `SpinOp::Sy` returns a zero matrix for `T = f64`. Should `op(SpinOp::Sy, i)` be a `compile_error!` when `T = f64` (enforced via a blanket impl with a `static_assertions` check)? The alternative — a runtime `DslError::IncompatibleScalarType` — is less ergonomic. | Resolved — `Scalar::from_real_imag` makes `Sy.matrix::<f64>()` return zeros by construction (correct behavior for real types); no compile-time error needed. Users must use `Complex<f64>` for models with `Sy`. See §17.9 |
| 5 | `CustomOp<T>` stores a `DenseTensor<'static, T>` and requires manual `clone_owned()` because `DenseTensor` does not implement `Clone`. For uniform models with the same custom operator on 1000 sites, this means 1000 heap allocations. An `Arc<DenseTensor<T>>` would share the storage. | Open — profile against `n_sites = 1000` with a 4x4 custom operator before deciding. The `clone_owned()` pattern works but is ergonomically awkward |
| 6 | The `Lattice` trait is currently object-safe because it has no generic methods. Adding a `visit_bonds<F: Fn(usize, usize)>` callback method for zero-allocation bond iteration would break object safety. Is the performance benefit worth the ergonomic cost of requiring `dyn Lattice` callers to use `bonds()` + iteration? | Resolved — keep `Lattice` object-safe; users can iterate `bonds()` slice directly |
| 7 | Should `StarGeometry` expose a `map_to_chain` method that returns a `Chain` with reordered coupling arrays, reflecting the Lanczos tridiagonalization used in `tk-dmft`? This would make the AIM→chain mapping visible at the `tk-dsl` level, improving discoverability. Alternatively, this mapping belongs entirely in `tk-dmft`. | Resolved — belongs in `tk-dmft`; keep `tk-dsl` geometry-agnostic w.r.t. chain mapping |
| 8 | `OpSum` stores `Box<dyn Lattice>` and needs `Clone`. The current implementation uses a `LatticeClone` helper trait with `clone_box()`. Should `Arc<dyn Lattice>` be used instead to avoid cloning lattice data? `Arc` would eliminate the `LatticeClone` boilerplate but adds reference-counting overhead and prevents mutable access. | Open — `clone_box()` pattern works; `Arc` may be preferable for large lattices |
| 9 | Should `DenseTensor` in `tk-core` implement `Clone`? Both `IndexedTensor` and `CustomOp` need cloneable tensor data and currently use `clone_owned()` workarounds. Adding `Clone` to `DenseTensor` would simplify the API. | Open — requires `tk-core` tech spec update |
| 10 | Should `ScaledOpProduct` provide convenience constructors `single(coeff, op, site)` and `two_site(coeff, op1, site1, op2, site2)` to reduce boilerplate? The draft implementation uses struct literals exclusively. The `scaled()` free function partially addresses this. | Open — assess user demand |
| 11 | Should `Mul<OpTerm<T>> for T` and `Mul<OpProduct<T>> for T` be implemented for `Complex<f64>` via a newtype wrapper to work around the orphan rule? This would restore `z * op(Sz, i)` syntax for complex scalars. | Open — the `scaled()` workaround is functional but less ergonomic |
