# Technical Specification: `tk-symmetry`

**Crate:** `tensorkraft/crates/tk-symmetry`
**Version:** 0.1.0 (Pre-Implementation)
**Status:** Specification
**Last Updated:** March 2026

---

## 1. Overview

`tk-symmetry` implements the quantum number types, block-sparse tensor format, and sector-lookup infrastructure that enable symmetry-exploiting algorithms throughout the tensorkraft workspace. It sits directly above `tk-core` in the dependency graph and is consumed by `tk-linalg`, `tk-contract`, and all higher-level crates.

**Core responsibility:** Represent the algebraic structure of physical symmetries (conservation laws) and expose tensors whose storage is partitioned into dense sub-blocks indexed by quantum number tuples (sectors). Only non-zero blocks are stored or computed.

**Performance motivation:** In a system with U(1) charge conservation, roughly 1/√N of tensor entries are non-zero at each charge sector. Exploiting block-sparsity yields O(1/√N) memory reduction and O(N^{1/2}) speedup in GEMM — order-of-magnitude gains for large bond dimensions.

**Dependencies:**
- `tk-core` — `Scalar`, `DenseTensor`, `TensorShape`, `TkError`
- `smallvec` — stack-allocated vectors for sector keys
- `hashbrown` — fast HashMap for sector metadata
- `thiserror` — error derive macros
- `lie-groups` (optional, `#[cfg(feature = "su2-symmetry")]`) — Clebsch-Gordan coefficients

---

## 2. Module Structure

```
tk-symmetry/
├── Cargo.toml
└── src/
    ├── lib.rs              re-exports all public items
    ├── quantum_number.rs   QuantumNumber + BitPackable traits
    ├── builtins.rs         U1, Z2, U1Z2 implementations
    ├── sector_key.rs       PackedSectorKey, QIndex
    ├── block_sparse.rs     BlockSparseTensor<T, Q>
    ├── formats.rs          Sparsity format enum, conversion utilities
    ├── flux.rs             Flux rule validation, sector enumeration
    └── su2/
        ├── mod.rs          (cfg: su2-symmetry) SU2Irrep, WignerEckartTensor
        └── cg_cache.rs     ClebschGordanCache
```

---

## 3. The `QuantumNumber` Trait

### 3.1 Definition

```rust
/// Abstract quantum number: the label attached to one leg of a symmetric tensor.
///
/// A quantum number encodes a conservation law. `fuse` combines two quantum
/// numbers on adjacent legs; `dual` gives the outgoing quantum number when
/// an incoming one is fixed by the flux rule.
///
/// All implementations must satisfy the group axioms:
///   - `q.fuse(Q::identity()) == q` (identity element)
///   - `q.fuse(q.dual()) == Q::identity()` (inverse)
///   - `a.fuse(b.fuse(c)) == a.fuse(b).fuse(c)` (associativity)
pub trait QuantumNumber:
    Clone + Eq + Hash + Ord + Debug + Send + Sync + 'static
{
    /// The group identity (additive zero, parity-even, trivial irrep, etc.).
    fn identity() -> Self;

    /// Group product / fusion: combine two quantum numbers into one.
    /// For U(1): addition. For Z₂: XOR. For SU(2): tensor-product irrep
    /// (one-to-many — see §7 for the SU(2) case).
    fn fuse(&self, other: &Self) -> Self;

    /// Group inverse / dual: the quantum number that cancels this one.
    /// For U(1): negation. For Z₂: identity (self-dual). For SU(2): same irrep.
    fn dual(&self) -> Self;
}
```

### 3.2 Flux Rule

A tensor T is symmetric under a quantum number Q if and only if its elements satisfy the **flux rule**: each non-zero element T_{i₁, i₂, ..., iₙ} satisfies

```
q(i₁) ⊕ q(i₂) ⊕ ... ⊕ q(iₙ) = flux
```

where `⊕` is `fuse`, and `flux` is the tensor's total charge. For most MPS site tensors, `flux = Q::identity()`. The MPO carries the Hamiltonian's symmetry sector.

---

## 4. The `BitPackable` Trait

### 4.1 Motivation

Sector lookup during block-sparse GEMM is on the hot path — it executes once per sector pair per contraction step, which in large DMRG sweeps means millions of times per second. A naive `HashMap<SmallVec<[Q; 8]>, usize>` incurs heap allocations and pointer chasing. Instead, Abelian quantum numbers implement `BitPackable`, which compresses a multi-leg sector key into a single `u64` register value. Binary search over a sorted `Vec<u64>` resolves in nanoseconds via LLVM-vectorized comparisons with no pointer chasing and no branch misprediction.

### 4.2 Definition

```rust
/// Extension of QuantumNumber: compresses the quantum number into a
/// fixed-width bitfield for register-speed sector lookup.
///
/// Only Abelian symmetries implement this trait. Non-Abelian symmetries
/// (SU(2)) use a separate SmallVec-keyed storage path.
pub trait BitPackable: QuantumNumber {
    /// Number of bits required to encode one quantum number.
    /// Must be a compile-time constant.
    const BIT_WIDTH: usize;

    /// Compress into the lower BIT_WIDTH bits of a u64.
    /// Implementors are responsible for lossless round-trip:
    ///   Self::unpack(self.pack()) == *self
    fn pack(&self) -> u64;

    /// Reconstruct from the lower BIT_WIDTH bits of a u64.
    fn unpack(bits: u64) -> Self;
}
```

### 4.3 Bit-Width Capacity

The maximum number of legs packable into a `u64` is `64 / BIT_WIDTH`:

| Quantum Number | `BIT_WIDTH` | Max legs (u64) | Max legs (u128) |
|:---------------|:-----------:|:--------------:|:---------------:|
| `Z2` | 1 | 64 | 128 |
| `U1` | 8 | 8 | 16 |
| `U1Z2` | 9 | 7 | 14 |

For multi-orbital Hubbard models (U(1)\_charge ⊗ U(1)\_spin, rank > 8), `PackedSectorKey` must be promoted to the `u128` variant (see §5.3).

---

## 5. Built-in Quantum Number Types

### 5.1 `U1` — Charge/Particle-Number Conservation

```rust
/// U(1) additive quantum number (e.g., particle number, total Sz).
/// Range: -128..=127 (8-bit signed, packed as u8 wrapping arithmetic).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct U1(pub i32);

impl QuantumNumber for U1 {
    fn identity() -> Self { U1(0) }
    fn fuse(&self, other: &Self) -> Self { U1(self.0 + other.0) }
    fn dual(&self) -> Self { U1(-self.0) }
}

impl BitPackable for U1 {
    const BIT_WIDTH: usize = 8;  // supports charges -128..=+127

    #[inline(always)]
    fn pack(&self) -> u64 { (self.0 as u8) as u64 }

    #[inline(always)]
    fn unpack(bits: u64) -> Self {
        U1(((bits & 0xFF) as u8) as i8 as i32)
    }
}
```

**Physical usage:** Total electron count in Hubbard / Anderson models; total spin projection Sz in spin models.

**Range note:** The 8-bit packing supports charges in -128..=+127. For systems with more than 127 sites at half-filling, the range must be extended. Use `U1Wide` (see §5.4) or promote `PackedSectorKey` to `u128`.

### 5.2 `Z2` — Parity Conservation

```rust
/// Z₂ parity quantum number (e.g., fermion parity: even/odd electron count).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Z2(pub bool);

impl QuantumNumber for Z2 {
    fn identity() -> Self { Z2(false) }
    fn fuse(&self, other: &Self) -> Self { Z2(self.0 ^ other.0) }
    fn dual(&self) -> Self { *self }  // Z₂ is self-dual
}

impl BitPackable for Z2 {
    const BIT_WIDTH: usize = 1;

    #[inline(always)]
    fn pack(&self) -> u64 { self.0 as u64 }

    #[inline(always)]
    fn unpack(bits: u64) -> Self { Z2(bits & 1 == 1) }
}
```

**Physical usage:** Fermion parity in superconducting systems; topological Z₂ invariants.

### 5.3 `U1Z2` — Composite Symmetry

```rust
/// Product symmetry: U(1) charge ⊗ Z₂ parity.
/// Common in fermionic Hubbard models where both particle number
/// and fermion parity are conserved.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct U1Z2(pub U1, pub Z2);

impl QuantumNumber for U1Z2 {
    fn identity() -> Self { U1Z2(U1::identity(), Z2::identity()) }
    fn fuse(&self, other: &Self) -> Self {
        U1Z2(self.0.fuse(&other.0), self.1.fuse(&other.1))
    }
    fn dual(&self) -> Self { U1Z2(self.0.dual(), self.1.dual()) }
}

impl BitPackable for U1Z2 {
    const BIT_WIDTH: usize = 9;  // 8 bits U1 + 1 bit Z2

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

### 5.4 `U1Wide` — Extended Range U(1)

```rust
/// U(1) with 16-bit packing for systems with more than 127 sites.
/// Supports charges -32768..=+32767.
/// Uses u128 sector keys; see PackedSectorKey128.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct U1Wide(pub i32);

impl QuantumNumber for U1Wide {
    fn identity() -> Self { U1Wide(0) }
    fn fuse(&self, other: &Self) -> Self { U1Wide(self.0 + other.0) }
    fn dual(&self) -> Self { U1Wide(-self.0) }
}

impl BitPackable for U1Wide {
    const BIT_WIDTH: usize = 16;
    #[inline(always)]
    fn pack(&self) -> u64 { (self.0 as u16) as u64 }
    #[inline(always)]
    fn unpack(bits: u64) -> Self {
        U1Wide(((bits & 0xFFFF) as u16) as i16 as i32)
    }
}
```

---

## 6. `PackedSectorKey` — Register-Speed Sector Lookup

### 6.1 Definition

```rust
/// A multi-leg sector key compressed into a single u64 register value.
/// Invariant: keys within a BlockSparseTensor are always sorted,
/// enabling O(log N) binary search that the CPU resolves via
/// LLVM-vectorized integer comparisons entirely in registers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PackedSectorKey(pub u64);

/// u128 variant for high-rank tensors or wide quantum numbers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PackedSectorKey128(pub u128);
```

### 6.2 Pack / Unpack

```rust
impl PackedSectorKey {
    /// Pack a slice of quantum numbers into a single u64.
    ///
    /// Layout: q[0] occupies bits 0..BIT_WIDTH,
    ///         q[1] occupies bits BIT_WIDTH..2*BIT_WIDTH, etc.
    ///
    /// Panics in debug mode if `qns.len() * Q::BIT_WIDTH > 64`.
    pub fn pack<Q: BitPackable>(qns: &[Q]) -> Self {
        debug_assert!(
            qns.len() * Q::BIT_WIDTH <= 64,
            "Sector rank {} × {} bits = {} bits exceeds u64; use PackedSectorKey128",
            qns.len(), Q::BIT_WIDTH, qns.len() * Q::BIT_WIDTH,
        );
        let mut packed: u64 = 0;
        for (i, q) in qns.iter().enumerate() {
            let shift = i * Q::BIT_WIDTH;
            let mask = (1u64 << Q::BIT_WIDTH) - 1;
            packed |= (q.pack() & mask) << shift;
        }
        PackedSectorKey(packed)
    }

    /// Unpack back to a SmallVec of quantum numbers.
    /// Used for debugging, display, and structural sector operations.
    pub fn unpack<Q: BitPackable>(&self, rank: usize) -> SmallVec<[Q; 8]> {
        let mask = (1u64 << Q::BIT_WIDTH) - 1;
        (0..rank)
            .map(|i| Q::unpack((self.0 >> (i * Q::BIT_WIDTH)) & mask))
            .collect()
    }
}
```

### 6.3 `QIndex<Q>` — Leg Basis Description

```rust
/// Describes the quantum-number basis on one tensor leg.
/// Stores the ordered list of (quantum_number, sector_dim) pairs
/// that span that leg's Hilbert space.
pub struct QIndex<Q: QuantumNumber> {
    /// Ordered list of (quantum_number, dimension) pairs for each sector.
    /// Invariant: sectors are sorted by quantum_number for binary search.
    sectors: Vec<(Q, usize)>,
    /// Total dimension: sum of all sector dimensions.
    total_dim: usize,
}

impl<Q: QuantumNumber> QIndex<Q> {
    pub fn new(sectors: Vec<(Q, usize)>) -> Self;
    pub fn total_dim(&self) -> usize { self.total_dim }
    pub fn n_sectors(&self) -> usize { self.sectors.len() }

    /// Row offset of the sector with quantum number q.
    /// Returns None if q is not present.
    pub fn offset_of(&self, q: &Q) -> Option<usize>;

    /// Dimension of the sector with quantum number q.
    pub fn dim_of(&self, q: &Q) -> Option<usize>;

    /// Iterator over (quantum_number, offset, dim) triples.
    pub fn iter_sectors(&self) -> impl Iterator<Item = (&Q, usize, usize)>;
}
```

---

## 7. `BlockSparseTensor<T, Q>` — Abelian Block-Sparse Tensor

### 7.1 Storage Layout (Structure-of-Arrays)

```rust
/// Block-sparse tensor for systems with Abelian symmetry Q.
///
/// Data is partitioned into dense sub-blocks, one per symmetry sector.
/// Only blocks satisfying the flux rule are stored; all others are zero.
///
/// INVARIANT: sector_keys is sorted in ascending order at all times.
/// Any operation that modifies sector_keys must restore this invariant.
pub struct BlockSparseTensor<T: Scalar, Q: BitPackable> {
    /// QIndex for each tensor leg. len() == rank.
    indices: Vec<QIndex<Q>>,
    /// Sorted sector keys (packed multi-leg quantum-number tuples).
    /// Parallel to sector_blocks.
    sector_keys: Vec<PackedSectorKey>,
    /// Dense sub-blocks, one per sector.
    /// sector_blocks[i] corresponds to sector_keys[i].
    sector_blocks: Vec<DenseTensor<T>>,
    /// Total charge of the tensor. Non-zero for e.g. creation operators.
    flux: Q,
}
```

### 7.2 Constructors

```rust
impl<T: Scalar, Q: BitPackable> BlockSparseTensor<T, Q> {
    /// Construct a zero tensor with the given leg bases and flux.
    /// Automatically enumerates all sectors satisfying the flux rule
    /// and allocates zero-filled DenseTensor blocks for each.
    pub fn zeros(indices: Vec<QIndex<Q>>, flux: Q) -> Self;

    /// Construct from an explicit list of (sector_key, block) pairs.
    /// Panics in debug mode if any block violates the flux rule,
    /// or if sector_keys are not unique.
    pub fn from_blocks(
        indices: Vec<QIndex<Q>>,
        flux: Q,
        blocks: Vec<(Vec<Q>, DenseTensor<T>)>,
    ) -> Self;
}
```

### 7.3 Sector Access

```rust
impl<T: Scalar, Q: BitPackable> BlockSparseTensor<T, Q> {
    /// O(log N) immutable block lookup. Returns None if the sector is absent
    /// (which means all elements in that sector are zero).
    #[inline(always)]
    pub fn get_block(&self, sector_qns: &[Q]) -> Option<&DenseTensor<T>> {
        let key = PackedSectorKey::pack(sector_qns);
        self.sector_keys
            .binary_search(&key)
            .ok()
            .map(|idx| &self.sector_blocks[idx])
    }

    /// O(log N) mutable block lookup.
    #[inline(always)]
    pub fn get_block_mut(&mut self, sector_qns: &[Q]) -> Option<&mut DenseTensor<T>>;

    /// Insert or overwrite a block. Maintains the sorted key invariant.
    pub fn insert_block(&mut self, sector_qns: Vec<Q>, block: DenseTensor<T>);

    /// Iterator over all non-zero (sector_qns, block) pairs.
    pub fn iter_blocks(&self) -> impl Iterator<Item = (SmallVec<[Q; 8]>, &DenseTensor<T>)>;

    pub fn rank(&self) -> usize { self.indices.len() }
    pub fn n_sectors(&self) -> usize { self.sector_blocks.len() }
    pub fn flux(&self) -> &Q { &self.flux }

    /// Total stored element count (sum of all block sizes).
    pub fn nnz(&self) -> usize;

    /// Maximum dimension across all sectors on one leg.
    pub fn max_sector_dim_on_leg(&self, leg: usize) -> usize;
}
```

### 7.4 Structural Operations

These operations return new `BlockSparseTensor`s with only metadata changes (strides, leg ordering) — no data is copied.

```rust
impl<T: Scalar, Q: BitPackable> BlockSparseTensor<T, Q> {
    /// Permute tensor legs. Returns a new tensor with rearranged QIndices
    /// and re-packed sector keys. Block data is permuted via DenseTensor::permute
    /// (zero-copy stride permutation).
    pub fn permute(&self, perm: &[usize]) -> Self;

    /// Fuse (combine) a contiguous range of legs into one combined leg.
    /// The combined QIndex has sectors given by all valid fused quantum numbers.
    /// Used to reshape MPS tensors before GEMM.
    pub fn fuse_legs(&self, legs: std::ops::Range<usize>) -> Self;

    /// Split one fused leg back into its component legs.
    /// Inverse of fuse_legs. Requires the original QIndex information.
    pub fn split_leg(&self, leg: usize, original_indices: Vec<QIndex<Q>>) -> Self;
}
```

### 7.5 Flux Rule Validation

```rust
/// Verify that a given multi-index sector satisfies the tensor's flux rule.
/// In debug/test builds, called on every block insertion and construction.
pub fn check_flux_rule<Q: QuantumNumber>(
    sector_qns: &[Q],
    expected_flux: &Q,
    leg_directions: &[LegDirection],  // Incoming or Outgoing
) -> bool {
    let fused = sector_qns
        .iter()
        .zip(leg_directions.iter())
        .fold(Q::identity(), |acc, (q, dir)| {
            match dir {
                LegDirection::Incoming => acc.fuse(q),
                LegDirection::Outgoing => acc.fuse(&q.dual()),
            }
        });
    fused == *expected_flux
}

pub enum LegDirection {
    Incoming,
    Outgoing,
}
```

---

## 8. Sparsity Format Registry

`tk-symmetry` tracks which sparsity strategy is appropriate for a given tensor, enabling `tk-linalg` to select the correct kernel.

```rust
/// Sparsity strategy for a tensor or matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparsityFormat {
    /// Contiguous 1-D buffer; standard BLAS path.
    Dense,
    /// Block-sparse with packed sector keys; Abelian symmetry path.
    BlockSparse,
    /// Compressed sparse row/column; for irregular geometry operators.
    ElementSparse,
    /// Diagonal only; for identity operators and local terms.
    Diagonal,
}
```

| Format | Storage | Application | When to use |
|:-------|:--------|:------------|:------------|
| `Dense` | Contiguous flat buffer | Non-symmetric or small matrices | When no symmetry is exploited |
| `BlockSparse` | Sorted block list (BSR) | U(1), Z₂, U(1)⊗Z₂ | Primary DMRG path |
| `ElementSparse` | CSR / CSC | Irregular Hamiltonians | Extreme sparsity beyond block structure |
| `Diagonal` | Single `Vec<T>` | Identity, on-site terms | Zero-overhead scalar multiplication |

---

## 9. Sector Enumeration

When constructing a zero tensor, valid sectors must be enumerated from the leg bases. This is non-trivial for high-rank tensors.

```rust
/// Enumerate all tuples (q₁, q₂, ..., qₙ) of quantum numbers — one per leg —
/// that satisfy the flux rule: fuse(q₁, q₂, ..., qₙ) == flux.
///
/// Uses backtracking with early pruning: after fixing the first k legs,
/// the remaining required charge is computed, and only basis states consistent
/// with that charge are explored on the remaining legs.
///
/// This function is called once at tensor construction; it is not on the
/// performance-critical path.
pub fn enumerate_valid_sectors<Q: QuantumNumber>(
    indices: &[QIndex<Q>],
    flux: &Q,
    leg_directions: &[LegDirection],
) -> Vec<Vec<Q>>;
```

---

## 10. Non-Abelian SU(2) Symmetry (Phase 5+)

Enabled only when `features = ["su2-symmetry"]`. The Abelian code path is **not affected** by this feature flag.

### 10.1 `SU2Irrep`

```rust
#[cfg(feature = "su2-symmetry")]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SU2Irrep {
    /// Twice the spin quantum number: twice_j = 2j, so j=0 → 0, j=1/2 → 1, j=1 → 2.
    /// Using twice_j avoids floating-point arithmetic in all comparisons.
    pub twice_j: u32,
}

#[cfg(feature = "su2-symmetry")]
impl QuantumNumber for SU2Irrep {
    fn identity() -> Self { SU2Irrep { twice_j: 0 } }

    /// Returns the *lowest* irrep in the tensor product decomposition:
    /// j₁ ⊗ j₂ = |j₁−j₂| ⊕ ... ⊕ (j₁+j₂).
    /// The full one-to-many fusion is handled by `fuse_all` below.
    fn fuse(&self, other: &Self) -> Self {
        let twice_j_min = (self.twice_j as i32 - other.twice_j as i32).unsigned_abs();
        SU2Irrep { twice_j: twice_j_min }
    }

    /// SU(2) irreps are self-dual (spin is its own conjugate representation).
    fn dual(&self) -> Self { *self }
}

#[cfg(feature = "su2-symmetry")]
impl SU2Irrep {
    /// Multiplet dimension: 2j + 1.
    pub fn dim(&self) -> usize { (self.twice_j + 1) as usize }

    /// Full tensor-product decomposition: all irreps j_c in j_a ⊗ j_b.
    /// Returns the range |j_a − j_b| ..= (j_a + j_b) in steps of 1.
    pub fn fuse_all(a: &Self, b: &Self) -> impl Iterator<Item = Self> {
        let min = (a.twice_j as i32 - b.twice_j as i32).unsigned_abs();
        let max = a.twice_j + b.twice_j;
        (min..=max).step_by(2).map(|twice_j| SU2Irrep { twice_j })
    }
}
```

**Note:** `SU2Irrep` intentionally does **not** implement `BitPackable`. The non-Abelian contraction path is dominated by Clebsch-Gordan evaluation, not sector lookup, so the bit-packing optimization is not applicable.

### 10.2 `ClebschGordanCache`

```rust
#[cfg(feature = "su2-symmetry")]
/// Thread-safe, lazily populated cache of Clebsch-Gordan coefficients.
/// Computing CG coefficients from scratch is expensive; caching amortizes
/// the cost across the thousands of contractions in a DMRG sweep.
pub struct ClebschGordanCache {
    /// Key: (j_a, j_b, j_c, m_a, m_b, m_c). Value: CG coefficient.
    cache: std::sync::RwLock<HashMap<CgKey, f64>>,
}

#[cfg(feature = "su2-symmetry")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CgKey {
    twice_ja: u32, twice_jb: u32, twice_jc: u32,
    twice_ma: i32, twice_mb: i32, twice_mc: i32,
}

#[cfg(feature = "su2-symmetry")]
impl ClebschGordanCache {
    pub fn new() -> Self;

    /// Retrieve ⟨j_a, m_a; j_b, m_b | j_c, m_c⟩, computing and caching on miss.
    pub fn get(
        &self,
        ja: SU2Irrep, ma: i32,
        jb: SU2Irrep, mb: i32,
        jc: SU2Irrep, mc: i32,
    ) -> f64;

    /// Pre-populate the cache for all CG coefficients up to a given j_max.
    /// Call once before a DMRG sweep to amortize initialization cost.
    pub fn prefill(&self, j_max: u32);
}
```

### 10.3 `WignerEckartTensor<T>`

```rust
#[cfg(feature = "su2-symmetry")]
/// An SU(2)-symmetric tensor in the Wigner-Eckart (reduced matrix element) form.
///
/// By the Wigner-Eckart theorem, T^{j_c}_{m_c} = ⟨j_a, m_a; j_b, m_b | j_c, m_c⟩ · T̃^{j_c}
/// where T̃^{j_c} is the reduced matrix element (scalar under rotation).
///
/// Only the reduced matrix elements are stored; the CG coefficients are
/// looked up from the cache at contraction time.
pub struct WignerEckartTensor<T: Scalar> {
    /// Reduced matrix elements, stored as a block-sparse tensor with SU2Irrep sectors.
    /// Uses SmallVec-keyed storage (not PackedSectorKey) because SU2Irrep
    /// is not BitPackable.
    reduced: HashMap<SmallVec<[SU2Irrep; 6]>, DenseTensor<T>>,
    /// Shared Clebsch-Gordan cache (Arc for multi-thread sharing across sweeps).
    cg_cache: std::sync::Arc<ClebschGordanCache>,
    /// Tensor flux (irrep of the operator).
    flux: SU2Irrep,
}
```

### 10.4 Known Refactoring Requirements

These items are scoped to the `su2-symmetry` feature flag and do not affect the Abelian code path:

**Fusion-rule multiplicity (task generation fan-out):** The Abelian block-sparse GEMM in `tk-linalg` assumes `compute_fusion_rule(key_a, key_b) -> Option<PackedSectorKey>` (one-to-one). For SU(2), j₁ ⊗ j₂ produces multiple output irreps. The `SectorGemmTask` generation loop must produce `Vec<SectorGemmTask>` per input pair, each weighted by the corresponding Clebsch-Gordan coefficient. The `structural_contraction` callback in `tk-contract` is the injection point for this coefficient evaluation.

**Output-sector collision hazard (map-reduce):** Multiple input pairs (j_a, j_b) can map to the same output sector j_c. Naive parallel dispatch creates a data race. Task generation must group tasks by output sector key and reduce (accumulate) partial contributions before writing. This is a structural change to the LPT scheduling phase in `tk-linalg`.

**Multiplet-aware SVD truncation:** Singular values in SU(2)-symmetric DMRG come in degenerate multiplets of dimension 2j+1. Truncation must snap to multiplet boundaries. The `svd_truncated` logic must implement a two-phase approach:
1. Sort singular values by magnitude.
2. Snap the truncation boundary to the nearest multiplet edge (never split a multiplet).
3. Weight discarded singular values by (2j+1)·σ_j² when computing truncation error.

---

## 11. Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum SymmetryError {
    #[error("flux rule violated: sector {sector:?} has fused charge {actual:?}, expected {expected:?}")]
    FluxRuleViolation {
        sector: Vec<String>,  // Display of quantum numbers
        actual: String,
        expected: String,
    },

    #[error("sector key overflow: rank {rank} × {bit_width} bits = {total} bits exceeds 64; use PackedSectorKey128")]
    SectorKeyOverflow { rank: usize, bit_width: usize, total: usize },

    #[error("sector not found: quantum numbers {qns:?} not present in this tensor")]
    SectorNotFound { qns: Vec<String> },

    #[error("leg dimension mismatch on leg {leg}: expected {expected}, got {got}")]
    LegDimensionMismatch { leg: usize, expected: usize, got: usize },

    #[error("incompatible quantum number types in operation")]
    QuantumNumberTypeMismatch,

    #[error(transparent)]
    Core(#[from] tk_core::TkError),
}

pub type SymResult<T> = Result<T, SymmetryError>;
```

---

## 12. Public API Surface (`lib.rs`)

```rust
// tk-symmetry/src/lib.rs

pub mod quantum_number;
pub mod builtins;
pub mod sector_key;
pub mod block_sparse;
pub mod formats;
pub mod flux;
pub mod error;

#[cfg(feature = "su2-symmetry")]
pub mod su2;

// Flat re-exports:
pub use quantum_number::{QuantumNumber, BitPackable, LegDirection};
pub use builtins::{U1, Z2, U1Z2, U1Wide};
pub use sector_key::{PackedSectorKey, PackedSectorKey128, QIndex};
pub use block_sparse::BlockSparseTensor;
pub use formats::SparsityFormat;
pub use error::{SymmetryError, SymResult};

#[cfg(feature = "su2-symmetry")]
pub use su2::{SU2Irrep, WignerEckartTensor, ClebschGordanCache};
```

---

## 13. Feature Flags

| Flag | Effect in tk-symmetry |
|:-----|:----------------------|
| `su2-symmetry` | Activates `SU2Irrep`, `WignerEckartTensor`, `ClebschGordanCache`; pulls in `lie-groups` crate |

No other `tk-symmetry`-specific feature flags. The `parallel` and backend flags are `tk-linalg` concerns.

---

## 14. Testing Requirements

### 14.1 Unit Tests

| Test | Description |
|:-----|:------------|
| `u1_fuse_identity` | `U1(3).fuse(U1::identity()) == U1(3)` |
| `u1_fuse_dual_is_identity` | `U1(n).fuse(U1(n).dual()) == U1::identity()` for all n in range |
| `u1_pack_round_trip` | `U1::unpack(U1(n).pack()) == U1(n)` for n in -128..=127 |
| `z2_fuse_xor` | `Z2(true).fuse(Z2(true)) == Z2(false)` |
| `z2_self_dual` | `Z2(b).dual() == Z2(b)` |
| `u1z2_pack_round_trip` | Round-trip pack/unpack for all (u1, z2) combinations |
| `packed_key_sort_order` | Keys packed for quantum-number-sorted inputs are in ascending order |
| `packed_key_overflow_debug_panic` | Debug build panics when rank × BIT_WIDTH > 64 |
| `block_sparse_get_block_present` | Lookup of a present sector returns the correct block |
| `block_sparse_get_block_absent` | Lookup of an absent sector returns None |
| `block_sparse_flux_rule_enforced` | Construction with flux-violating block panics in debug mode |
| `block_sparse_sector_key_sorted` | After `insert_block`, `sector_keys` is always sorted |
| `block_sparse_zeros_valid_sectors` | `zeros()` constructs blocks for all and only valid sectors |
| `block_sparse_permute_numel` | `permute()` preserves total element count |
| `enumerate_sectors_completeness` | All valid sectors for a rank-3 U1 tensor are found |
| `check_flux_rule_correct` | Correct sectors pass; flux-violating sectors fail |
| `qindex_offset_of` | `offset_of` returns correct cumulative offsets |

### 14.2 Property-Based Tests

```rust
proptest! {
    #[test]
    fn u1_fuse_associativity(a in -64i32..=64, b in -64i32..=64, c in -64i32..=64) {
        let qa = U1(a); let qb = U1(b); let qc = U1(c);
        prop_assert_eq!(qa.fuse(&qb).fuse(&qc), qa.fuse(&qb.fuse(&qc)));
    }

    #[test]
    fn u1_pack_roundtrip_all_range(n in -128i32..=127) {
        prop_assert_eq!(U1::unpack(U1(n).pack()), U1(n));
    }

    #[test]
    fn packed_key_binary_search_finds_inserted(
        // Bounded: rank 2..=6, dims 1..=8 per sector
    ) {
        // Build BlockSparseTensor, insert sectors at random valid quantum numbers,
        // verify get_block finds all of them.
    }
}
```

### 14.3 Invariant Checks

The `BlockSparseTensor` exposes a `#[cfg(debug_assertions)] fn assert_invariants(&self)` method that checks:
1. `sector_keys` is strictly sorted (ascending, no duplicates).
2. Each block's shape matches the dimensions specified by the corresponding `QIndex` sectors.
3. Each sector satisfies the flux rule.

This is called at the end of every constructor and mutation in debug/test builds.

---

## 15. Performance Invariants

| Operation | Invariant |
|:----------|:----------|
| `get_block` | O(log N) — binary search over sorted `Vec<u64>`; no allocations |
| `pack` for rank-8 U1 tensor | Single loop, ≤ 8 shifts and ORs — should compile to ~8 instructions |
| `BlockSparseTensor::zeros` construction | One-time cost; not on hot path |
| `SU2Irrep::fuse_all` | Returns an iterator; no heap allocation |

CI Criterion benchmarks must verify that `get_block` on a 100-sector tensor completes in < 10 ns.

---

## 16. Out of Scope

The following are explicitly **not** implemented in `tk-symmetry`:

- GEMM dispatch or LPT scheduling for block-sparse contractions (→ `tk-linalg`)
- DAG-based contraction path optimization (→ `tk-contract`)
- MPS / MPO data structures (→ `tk-dmrg`)
- SVD truncation, including multiplet-aware truncation (→ `tk-linalg`, `tk-dmrg`)
- Physical model definitions or lattice types (→ `tk-dsl`)

---

## 17. Open Questions

| # | Question | Status |
|:--|:---------|:-------|
| 1 | Should `BlockSparseTensor` support mixed symmetry groups on different legs (e.g., leg 0 carries `U1`, leg 1 carries `Z2`)? Current design requires all legs to share the same `Q` type. | Deferred; workaround is to use `U1Z2` composite type |
| 2 | Is `SmallVec<[SU2Irrep; 6]>` the right key for the `WignerEckartTensor` reduced-block map, or should it be a sorted `Vec` for a more complex rank pattern? | To be decided when SU(2) is actively implemented |
| 3 | Should `ClebschGordanCache` use `dashmap` for better concurrent write performance during `prefill`? | Profile under realistic multi-thread load first |
| 4 | Does `enumerate_valid_sectors` need memoization for high-rank tensors (rank > 6)? | Benchmark on rank-8 MPO construction before optimizing |
