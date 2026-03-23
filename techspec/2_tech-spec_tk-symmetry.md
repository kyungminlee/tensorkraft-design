# Technical Specification: `tk-symmetry`

**Crate:** `tensorkraft/crates/tk-symmetry`
**Version:** 0.2.0 (Post-Draft-Implementation)
**Status:** Draft
**Last Updated:** March 2026

---

## 1. Overview

`tk-symmetry` implements the quantum number types, block-sparse tensor format, and sector-lookup infrastructure that enable symmetry-exploiting algorithms throughout the tensorkraft workspace. It sits directly above `tk-core` in the dependency graph and is consumed by `tk-linalg`, `tk-contract`, and all higher-level crates.

**Core responsibility:** Represent the algebraic structure of physical symmetries (conservation laws) and expose tensors whose storage is partitioned into dense sub-blocks indexed by quantum number tuples (sectors). Only non-zero blocks are stored or computed. A dual-layout storage strategy supports both structural mutations (fragmented `Vec<DenseTensor<T>>`) and GPU-optimized DMA transfers (contiguous `FlatBlockStorage` packed into pinned arena memory).

**Performance motivation:** In a system with U(1) charge conservation, roughly 1/√N of tensor entries are non-zero at each charge sector. Exploiting block-sparsity yields O(1/√N) memory reduction and O(N^{1/2}) speedup in GEMM — order-of-magnitude gains for large bond dimensions.

**Dependencies:**
- `tk-core` — `Scalar`, `DenseTensor`, `TensorShape`, `TkError`, `SweepArena` (for arena-backed `flatten()`)
- `smallvec` — stack-allocated vectors for sector keys
- `hashbrown` — fast HashMap for sector metadata
- `thiserror` — error derive macros
- `dashmap` — thread-safe concurrent hash map for `ClebschGordanCache`

> **Implementation note:** The original spec listed `lie-groups` as an optional dependency for Clebsch-Gordan coefficients. The draft implementation instead uses a hand-rolled Racah formula with `DashMap`-based thread-safe lazy caching, removing the `lie-groups` dependency entirely.

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
    ├── flat_storage.rs     FlatBlockStorage, dual-layout flatten/unflatten
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
/// The `'static` bound is required because quantum number values are stored
/// in long-lived structs (e.g., `BlockSparseTensor`, `QIndex`) that outlive
/// any particular function scope.
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
/// `Copy` is required as a supertrait because downstream crates clone quantum
/// numbers pervasively (sector enumeration, key packing, block construction).
/// All built-in types (`U1`, `Z2`, `U1Z2`, `U1Wide`) are `Copy`. Requiring
/// `Copy` here eliminates excessive `.clone()` boilerplate throughout the
/// codebase.
pub trait BitPackable: QuantumNumber + Copy {
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
|:---------------|:------------|:---------------|:----------------|
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
    /// Direction (Incoming/Outgoing) for each tensor leg. len() == rank.
    /// Essential for flux rule validation: Incoming legs contribute q,
    /// Outgoing legs contribute q.dual() to the fused charge.
    leg_directions: Vec<LegDirection>,
    /// Sorted sector keys (packed multi-leg quantum-number tuples).
    /// Parallel to sector_blocks.
    sector_keys: Vec<PackedSectorKey>,
    /// Dense sub-blocks, one per sector.
    /// sector_blocks[i] corresponds to sector_keys[i].
    /// All blocks use `'static` lifetime: `Vec<DenseTensor<'static, T>>`.
    /// No arena-borrowed blocks; all block data is heap-owned.
    sector_blocks: Vec<DenseTensor<'static, T>>,
    /// Total charge of the tensor. Non-zero for e.g. creation operators.
    flux: Q,
}
```

### 7.2 Constructors

```rust
impl<T: Scalar, Q: BitPackable> BlockSparseTensor<T, Q> {
    /// Construct a zero tensor with the given leg bases, flux, and leg directions.
    /// Automatically enumerates all sectors satisfying the flux rule
    /// and allocates zero-filled DenseTensor blocks for each.
    ///
    /// `leg_directions` must have the same length as `indices`.
    pub fn zeros(
        indices: Vec<QIndex<Q>>,
        flux: Q,
        leg_directions: Vec<LegDirection>,
    ) -> Self;

    /// Construct from an explicit list of (sector_key, block) pairs.
    /// Panics in debug mode if any block violates the flux rule,
    /// or if sector_keys are not unique.
    ///
    /// `leg_directions` must have the same length as `indices`.
    pub fn from_blocks(
        indices: Vec<QIndex<Q>>,
        flux: Q,
        leg_directions: Vec<LegDirection>,
        blocks: Vec<(Vec<Q>, DenseTensor<T>)>,
    ) -> Self;

    /// Internal escape hatch: construct from pre-validated raw parts.
    /// Bypasses all invariant checks.
    /// Used by `unflatten` to reconstruct tensors from flat storage.
    pub(crate) fn from_raw_parts(
        indices: Vec<QIndex<Q>>,
        leg_directions: Vec<LegDirection>,
        sector_keys: Vec<PackedSectorKey>,
        sector_blocks: Vec<DenseTensor<'static, T>>,
        flux: Q,
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
    pub fn leg_directions(&self) -> &[LegDirection] { &self.leg_directions }

    /// Total stored element count (sum of all block sizes).
    pub fn nnz(&self) -> usize;

    /// Maximum dimension across all sectors on one leg.
    pub fn max_sector_dim_on_leg(&self, leg: usize) -> usize;

    /// Iterator over (packed_sector_key, block) pairs.
    /// Heavily used internally for structural operations (fuse, split, permute).
    pub fn iter_keyed_blocks(&self)
        -> impl Iterator<Item = (&PackedSectorKey, &DenseTensor<'static, T>)>;
}
```

### 7.4 Structural Operations

```rust
impl<T: Scalar, Q: BitPackable> BlockSparseTensor<T, Q> {
    /// Permute tensor legs. Returns a new tensor with rearranged QIndices,
    /// re-packed sector keys, and permuted leg directions.
    ///
    /// **Important:** For block-sparse tensors, `permute()` is NOT zero-copy.
    /// Each block requires `block.permute(perm).into_owned()` because BLAS
    /// kernels require contiguous (column-major or row-major) memory layout
    /// within each block. The "zero-copy stride permutation" claim applies
    /// only to dense tensors, not block-sparse tensors.
    ///
    /// Cost: O(nnz) — every stored element is copied once.
    pub fn permute(&self, perm: &[usize]) -> Self;

    /// Fuse (combine) a contiguous range of legs into one combined leg.
    /// The combined QIndex has sectors given by all valid fused quantum numbers.
    /// The fused quantum number for a combination (q_i, q_{i+1}, ...) is computed
    /// respecting the original leg directions: Incoming contributes q, Outgoing
    /// contributes q.dual(). The fused leg direction is always Incoming.
    /// Used to reshape MPS tensors before GEMM.
    ///
    /// **Algorithm complexity:** The fusion algorithm involves three phases:
    /// 1. **Cartesian product enumeration** of quantum numbers across the fused
    ///    legs to determine which combinations map to which fused quantum number.
    /// 2. **Offset map construction** using `BTreeMap` (not `HashMap`) to ensure
    ///    deterministic ordering of sectors within each fused block. The key is
    ///    the fused quantum number; the value tracks the row/column offset within
    ///    the fused dimension where each sub-block is placed.
    /// 3. **Block scatter** — each original block's data is copied into the
    ///    appropriate sub-region of the fused block at the offset determined
    ///    in phase 2.
    ///
    /// The `BTreeMap` is chosen over `HashMap` to guarantee reproducible sector
    /// ordering across runs, which is critical for numerical reproducibility in
    /// DMRG sweeps.
    pub fn fuse_legs(&self, legs: std::ops::Range<usize>) -> Self;

    /// Split one fused leg back into its component legs.
    /// Inverse of fuse_legs. Requires the original QIndex and direction information
    /// for each sub-leg.
    ///
    /// The `original_directions` parameter is essential: during fusion, per-sub-leg
    /// direction information is lost (the fused leg always has direction Incoming).
    /// To reconstruct the fuse map and determine which quantum-number combinations
    /// map to which fused quantum number and at what offset, the original directions
    /// must be provided.
    pub fn split_leg(
        &self,
        leg: usize,
        original_indices: Vec<QIndex<Q>>,
        original_directions: Vec<LegDirection>,
    ) -> Self;
}
```

### 7.5 Dual-Layout Block Storage (Phase 4)

The current `Vec<DenseTensor<T>>` per-block storage (the **mutation layout**) is adequate for CPU-only Phases 1–3. For GPU transfers in Phase 5, hundreds of individually-allocated blocks would require either hundreds of small `cudaMemcpyAsync` calls (terrible PCIe utilization) or manual gathering into a staging buffer.

The architecture distinguishes two storage layouts:

- **Mutation Layout (fragmented, default):** `Vec<DenseTensor<T>>` — each block is an independent heap allocation. Optimal for structural mutations (e.g., appending columns during TDVP subspace expansion is O(D_sector²) because only the affected block is reallocated).
- **Compute Layout (contiguous, read-only):** A single flat buffer with an offset table, optimized for GPU DMA and cache-friendly GEMM dispatch. Structural mutations on this layout are forbidden (would require O(D_total²) memory shift).

```rust
/// Compute-side read-only block storage: all sector data in one contiguous allocation.
/// Enables single-DMA GPU transfer of the entire tensor.
/// NOT used during structural mutations (subspace expansion); see mutation layout.
pub struct FlatBlockStorage<'a, T: Scalar> {
    /// Single contiguous buffer containing all sector blocks back-to-back.
    /// Allocated from SweepArena (pinned memory when backend-cuda is active),
    /// NOT from the pageable heap. This guarantees DMA-capable memory for
    /// GPU transfers without the NVIDIA driver's hidden pin-copy-unpin dance.
    ///
    /// **Safety:** The buffer is allocated via `alloc_slice_uninit` (unsafe)
    /// and immediately populated by `copy_from_slice` in the `flatten()` method.
    /// This is sound because every byte is written before any read occurs,
    /// but callers must not access the buffer between allocation and population.
    data: &'a mut [T],
    /// Start index of each sector block within `data`.
    /// offsets[i] is the start index of sector_keys[i]'s block data.
    offsets: Vec<usize>,
    /// Dimensions (rows, cols) of each sector block.
    ///
    /// **Limitation:** For blocks with rank != 2, the shape is stored as
    /// `(numel, 1)` — the original multi-dimensional shape is lost.
    /// This means `flatten -> unflatten` is NOT a round-trip for tensors
    /// with rank != 2: the unflattened blocks will have shape `(numel, 1)`
    /// instead of their original shape. This is acceptable because `flatten`
    /// is only used on the GEMM path where blocks are always rank-2 matrices.
    shapes: Vec<(usize, usize)>,
}

impl<T: Scalar, Q: BitPackable> BlockSparseTensor<T, Q> {
    /// Pack fragmented blocks into a contiguous flat buffer for GPU/GEMM.
    /// Called after structural mutations are complete, before dispatch.
    ///
    /// CRITICAL: The flat buffer is allocated from the SweepArena, NOT from
    /// fresh pageable heap memory. When `backend-cuda` is active, the arena
    /// uses pinned memory, so the resulting buffer is directly DMA-capable —
    /// no hidden staging copy by the NVIDIA driver.
    ///
    /// Cost: O(D_total²) — a single memcpy pass, negligible relative to
    /// the O(D³) GEMM it feeds.
    pub fn flatten<'a>(&self, arena: &'a SweepArena) -> FlatBlockStorage<'a, T> {
        let total_elems = self.sector_blocks.iter().map(|b| b.num_elements()).sum();
        let buf = arena.alloc_slice::<T>(total_elems);
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

**Transition flow during a TDVP step:** (1) subspace expansion mutates A_L in fragmented layout → (2) `A_L.flatten(&arena)` packs into the arena's pinned memory → (3) GEMM/GPU operates on the flat buffer (DMA-direct, no staging copy) → (4) results unflattened back to mutable layout for the next step → (5) `SweepArena::reset()` reclaims the flat buffer in O(1). The `SparseLinAlgBackend` trait takes `&BlockSparseTensor` as an opaque input, so the internal layout switching is invisible to callers.

### 7.6 Flux Rule Validation

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

/// Variants are `Incoming` and `Outgoing` (not abbreviated `In`/`Out`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
///
/// **Implementation note:** CG coefficients are computed using a hand-rolled
/// Racah formula (direct algebraic evaluation), NOT the `lie-groups` crate.
/// The cache uses `DashMap` for lock-free concurrent read/write access,
/// replacing the original `RwLock<HashMap>` design for better multi-thread
/// performance during `prefill` and concurrent `get` calls.
pub struct ClebschGordanCache {
    /// Key: (j_a, j_b, j_c, m_a, m_b, m_c). Value: CG coefficient.
    cache: DashMap<CgKey, f64>,
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
    /// Clebsch-Gordan / 6j / 9j structural coefficient evaluator.
    /// The core contraction engine includes an optional `structural_contraction`
    /// callback from day one: the Abelian code path passes a no-op (zero overhead);
    /// the SU(2) code path injects symbol evaluations via this cache.
    structural: ClebschGordanCache,
    /// Reduced matrix elements, stored as a block-sparse tensor with SU2Irrep sectors.
    /// Uses SmallVec-keyed storage (not PackedSectorKey) because SU2Irrep
    /// is not BitPackable.
    reduced: HashMap<SmallVec<[SU2Irrep; 6]>, DenseTensor<T>>,
    /// Tensor flux (irrep of the operator).
    flux: SU2Irrep,
}
```

### 10.4 Known Refactoring Requirements

These items are scoped to the `su2-symmetry` feature flag and do not affect the Abelian code path:

**Fusion-rule multiplicity (task generation fan-out):** The Abelian block-sparse GEMM in `tk-linalg` assumes `compute_fusion_rule(key_a, key_b) -> Option<PackedSectorKey>` (one-to-one). For SU(2), j₁ ⊗ j₂ produces multiple output irreps: j₁ ⊗ j₂ = |j₁−j₂| ⊕ (|j₁−j₂|+1) ⊕ ... ⊕ (j₁+j₂). The `SectorGemmTask` generation loop must produce `Vec<SectorGemmTask>` per input pair, each weighted by the corresponding Clebsch-Gordan coefficient. The `structural_contraction` callback in `tk-contract` is the injection point for this coefficient evaluation.

**Task generation memory bound:** The combinatorial fan-out of the map-reduce pattern is bounded by the physics. In SU(2)-symmetric DMRG, the number of distinct irreps at a given bond (D_reduced) is typically 10–50 even at large total bond dimensions, because many states share the same j label. The fan-out per input pair is at most (2·j_max + 1) output sectors. The total task count before reduction is therefore O(D_reduced² × j_max). For j_max = 10 and D_reduced = 50, this yields ~250,000 tasks at ~64 bytes each ≈ 16 MB — well within L3 cache and posing no allocation pressure. The task vector should be pre-allocated with `Vec::with_capacity(d_reduced * d_reduced * (2 * j_max + 1))` to avoid incremental reallocation during the generation loop. For exotic high-spin models with j_max > 50 or D_reduced > 200, a streaming/chunked reduction should be considered, but this is outside the scope of Phase 5 targets.

**Output-sector collision hazard (map-reduce):** Multiple input pairs (j_a, j_b) can map to the same output sector j_c. Naive parallel dispatch via `par_iter` creates a data race: two threads writing to the same output block. The SU(2) task generation must use a map-reduce pattern — group tasks by output sector key, then accumulate (reduce) partial contributions within each group before writing the final block. This is a structural change to the LPT scheduling phase in `tk-linalg` that does not affect the Abelian code path, where fusion is always one-to-one.

**Multiplet-aware SVD truncation:** Singular values in SU(2)-symmetric DMRG come in degenerate multiplets of dimension 2j+1. Truncation must keep or discard entire multiplets — splitting a multiplet explicitly breaks the symmetry and crashes the simulation. The `svd_truncated` logic must implement a two-phase approach:
1. Sort singular values by magnitude.
2. Snap the truncation boundary to the nearest multiplet edge (never split a multiplet).
3. Weight discarded singular values by (2j+1)·σ_j² when computing truncation error.

---

## 11. Error Handling

### 11.1 Error Propagation Strategy

`SymmetryError` wraps `tk_core::TkError` via the `#[from]` attribute, allowing any `TkError` raised by `tk-core` operations (e.g., `DenseTensor` construction or `SweepArena` allocation) to propagate transparently through `tk-symmetry` functions that return `SymResult<T>`. Downstream crates (`tk-linalg`, `tk-contract`, `tk-dmrg`, etc.) in turn wrap `SymmetryError` in their own error enums using the same `#[from]` pattern, forming a layered error chain that preserves the original error context across crate boundaries without requiring manual conversion logic.

**Draft implementation note:** The current draft implementation uses panics over `Result` in many code paths, particularly constructors (`zeros`, `from_blocks`) and invariant-checking code (`check_flux_rule`). These panics fire on programmer errors (e.g., mismatched leg counts, flux rule violations) rather than recoverable runtime errors. Whether to convert some of these to `SymResult` returns is an open question (see Open Questions).

### 11.2 Error Enum Definition

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
pub mod flat_storage;
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
pub use flat_storage::FlatBlockStorage;
pub use formats::SparsityFormat;
pub use error::{SymmetryError, SymResult};

#[cfg(feature = "su2-symmetry")]
pub use su2::{SU2Irrep, WignerEckartTensor, ClebschGordanCache};
```

---

## 13. Feature Flags

| Flag | Effect in tk-symmetry |
|:-----|:----------------------|
| `su2-symmetry` | Activates `SU2Irrep`, `WignerEckartTensor`, `ClebschGordanCache` (hand-rolled Racah formula; no external dependency) |

No other `tk-symmetry`-specific feature flags. The `parallel` and backend flags are `tk-linalg` concerns.

---

## 14. Dependencies and Integration

### 14.1 `Cargo.toml` Sketch

```toml
[dependencies]
tk-core = { path = "../tk-core" }
smallvec = "1"
hashbrown = "0.14"
thiserror = "1"
dashmap = "5"

[dev-dependencies]
proptest = "1"
trybuild = "1"

[features]
default = []
su2-symmetry = []
```

> **Implementation note:** The `lie-groups` dependency has been removed. CG coefficients are computed via a hand-rolled Racah formula. The `dashmap` dependency provides the thread-safe concurrent cache used by `ClebschGordanCache`.

### 14.2 Upstream Dependencies

- `tk-core` — `Scalar`, `DenseTensor`, `TensorShape`, `TkError`, `SweepArena`

### 14.3 Downstream Consumers

- `tk-linalg`
- `tk-contract`
- `tk-dsl`
- `tk-dmrg`
- `tk-dmft`
- `tk-python`

---

## 15. Testing Strategy

### 15.1 Unit Tests

The draft implementation has 40 passing tests, exceeding the original spec count of 19. The table below lists representative tests; the full test suite is in the implementation.

| Test | Description |
|:-----|:------------|
| `u1_fuse_identity` | `U1(3).fuse(U1::identity()) == U1(3)` |
| `u1_fuse_dual_is_identity` | `U1(n).fuse(U1(n).dual()) == U1::identity()` for all n in range |
| `u1_pack_round_trip` | `U1::unpack(U1(n).pack()) == U1(n)` for n in -128..=127 |
| `z2_fuse_xor` | `Z2(true).fuse(Z2(true)) == Z2(false)` |
| `z2_self_dual` | `Z2(b).dual() == Z2(b)` |
| `u1z2_pack_round_trip` | Round-trip pack/unpack for all (u1, z2) combinations |
| `packed_key_sort_order` | Keys packed for quantum-number-sorted inputs are in ascending order |
| `packed_key_overflow_debug_panic` | Debug build panics when rank x BIT_WIDTH > 64 |
| `block_sparse_get_block_present` | Lookup of a present sector returns the correct block |
| `block_sparse_get_block_absent` | Lookup of an absent sector returns None |
| `block_sparse_flux_rule_enforced` | Construction with flux-violating block panics in debug mode |
| `block_sparse_sector_key_sorted` | After `insert_block`, `sector_keys` is always sorted |
| `block_sparse_zeros_valid_sectors` | `zeros()` constructs blocks for all and only valid sectors |
| `block_sparse_zeros_leg_directions` | `zeros()` stores leg directions and uses them in flux validation |
| `block_sparse_permute_numel` | `permute()` preserves total element count |
| `block_sparse_permute_copies_data` | `permute()` produces owned contiguous blocks (not views) |
| `enumerate_sectors_completeness` | All valid sectors for a rank-3 U1 tensor are found |
| `check_flux_rule_correct` | Correct sectors pass; flux-violating sectors fail |
| `check_flux_rule_with_directions` | Flux rule correctly handles mixed Incoming/Outgoing legs |
| `qindex_total_dim` | `total_dim()` returns correct cumulative dimension |
| `qindex_offset_of` | `offset_of` returns correct cumulative offsets |
| `flatten_contiguous_data` | `flatten()` produces contiguous buffer matching element-by-element iteration over fragmented blocks |
| `unflatten_round_trip` | `unflatten(flatten(tensor))` recovers original blocks for rank-2 tensors |
| `flatten_non_rank2_shape_loss` | `flatten()` on non-rank-2 blocks stores shape as `(numel, 1)` |
| `flatten_offsets_correct` | Each offset in `FlatBlockStorage` matches the cumulative element count |
| `fuse_legs_cartesian_product` | Fused leg contains all valid quantum-number combinations |
| `fuse_legs_deterministic_order` | Fused blocks have deterministic sector ordering across runs |
| `split_leg_round_trip` | `split_leg(fuse_legs(t))` recovers the original tensor |
| `split_leg_requires_original_directions` | `split_leg` uses original directions to reconstruct fuse map |
| `iter_keyed_blocks_complete` | `iter_keyed_blocks()` yields all stored (key, block) pairs |
| `from_raw_parts_bypasses_validation` | `from_raw_parts()` constructs without invariant checks |

### 15.2 Property-Based Tests

**Status:** Property-based tests (`proptest`) are absent from the draft implementation. The following tests are planned but not yet implemented. Adding `proptest` coverage is a priority for the next iteration.

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

    #[test]
    fn permute_preserves_block_data(
        // Bounded: rank 2..=4, dims 1..=4 per sector
    ) {
        // Build BlockSparseTensor, permute, verify element values match
        // the permuted dense equivalent.
    }

    #[test]
    fn fuse_split_round_trip(
        // Bounded: rank 3..=5, dims 1..=4 per sector
    ) {
        // Build BlockSparseTensor, fuse two legs, split them back,
        // verify the result matches the original.
    }
}
```

### 15.3 Invariant Checks

The `BlockSparseTensor` exposes a `#[cfg(debug_assertions)] fn assert_invariants(&self)` method that checks:
1. `sector_keys` is strictly sorted (ascending, no duplicates).
2. Each block's shape matches the dimensions specified by the corresponding `QIndex` sectors.
3. Each sector satisfies the flux rule.

This is called at the end of every constructor and mutation in debug/test builds.

---

## 16. Performance Invariants

| Operation | Invariant |
|:----------|:----------|
| `get_block` | O(log N) — binary search over sorted `Vec<u64>`; no allocations |
| `pack` for rank-8 U1 tensor | Single loop, ≤ 8 shifts and ORs — should compile to ~8 instructions |
| `BlockSparseTensor::zeros` construction | One-time cost; not on hot path |
| `permute` | O(nnz) — copies every stored element (NOT zero-copy for block-sparse) |
| `fuse_legs` | O(nnz + S × P) where S = number of sectors, P = Cartesian product size of fused legs |
| `flatten` | O(D_total²) single memcpy pass; negligible relative to O(D³) GEMM it feeds |
| `SU2Irrep::fuse_all` | Returns an iterator; no heap allocation |
| `iter_keyed_blocks` | O(1) per iteration step; zero allocations |

CI Criterion benchmarks must verify that `get_block` on a 100-sector tensor completes in < 10 ns.

---

## 17. Implementation Notes and Design Decisions

### Note 1 — `leg_directions` Is a Required Field on `BlockSparseTensor`

The original architecture document omitted `leg_directions` from the `BlockSparseTensor` struct. The draft implementation revealed that `leg_directions: Vec<LegDirection>` is essential: `check_flux_rule` requires knowing which legs are `Incoming` vs `Outgoing` to correctly fuse quantum numbers. All constructors (`zeros`, `from_blocks`) now require `leg_directions` as a parameter. This field is stored on the tensor and propagated through `permute`, `fuse_legs`, and `split_leg`.

### Note 2 — `permute()` Copies Data for Block-Sparse Tensors

The original spec claimed structural operations are zero-copy. This is true for dense tensors (stride permutation), but NOT for block-sparse tensors. BLAS requires contiguous memory within each block, so `permute()` must call `block.permute(perm).into_owned()` on every block. The cost is O(nnz) — every stored element is copied once.

### Note 3 — `fuse_legs` Uses `BTreeMap` for Deterministic Ordering

The fusion algorithm uses `BTreeMap` (not `HashMap`) for the offset map to guarantee deterministic sector ordering. This is critical for numerical reproducibility: `HashMap` iteration order is randomized, which would cause non-deterministic block layouts and, consequently, non-bitwise-reproducible DMRG sweeps.

### Note 4 — All Blocks Use `'static` Lifetime

`BlockSparseTensor<T, Q>` stores `Vec<DenseTensor<'static, T>>`. There are no arena-borrowed blocks. This simplifies ownership semantics at the cost of requiring heap allocation for every block. Arena-backed blocks remain a future optimization for the GPU path.

### Note 5 — CG Coefficients Use Hand-Rolled Racah Formula

The `ClebschGordanCache` computes Clebsch-Gordan coefficients using a direct algebraic Racah formula implementation rather than the `lie-groups` external crate. This eliminates an external dependency and gives full control over numerical precision and caching strategy.

---

## 18. Security Considerations

### 18.1 Unsafe Code in `flatten()`

The `flatten()` method uses `alloc_slice_uninit` to allocate uninitialized memory from the `SweepArena`. This is sound because every byte is immediately populated by `copy_from_slice` before any read occurs. The unsafe block is confined to the `flatten` method and does not leak uninitialized memory to callers.

### 18.2 `from_raw_parts()` Bypasses Validation

The `pub(crate)` method `from_raw_parts()` constructs a `BlockSparseTensor` without checking invariants (sorted keys, flux rule, shape consistency). It is used only by `unflatten` where the data is known to satisfy all invariants. This method must not be made `pub` without adding safety documentation.

---

## 19. Out of Scope

The following are explicitly **not** implemented in `tk-symmetry`:

- GEMM dispatch or LPT scheduling for block-sparse contractions (-> `tk-linalg`)
- DAG-based contraction path optimization (-> `tk-contract`)
- MPS / MPO data structures (-> `tk-dmrg`)
- SVD truncation, including multiplet-aware truncation (-> `tk-linalg`, `tk-dmrg`)
- Physical model definitions or lattice types (-> `tk-dsl`)

---

## 20. Open Questions

| # | Question | Status |
|:--|:---------|:-------|
| 1 | Should `BlockSparseTensor` support mixed symmetry groups on different legs (e.g., leg 0 carries `U1`, leg 1 carries `Z2`)? Current design requires all legs to share the same `Q` type. | Deferred — workaround is to use `U1Z2` composite type |
| 2 | Is `SmallVec<[SU2Irrep; 6]>` the right key for the `WignerEckartTensor` reduced-block map, or should it be a sorted `Vec` for a more complex rank pattern? | Deferred — to be decided when SU(2) is actively implemented |
| 3 | Should `ClebschGordanCache` use `dashmap` for better concurrent write performance during `prefill`? | Resolved — draft implementation uses `DashMap` for thread-safe lazy caching |
| 4 | Does `enumerate_valid_sectors` need memoization for high-rank tensors (rank > 6)? | Open — benchmark on rank-8 MPO construction before optimizing |
| 5 | Should constructors return `SymResult` instead of panicking on invariant violations (e.g., mismatched leg counts, flux rule violations)? Panics are appropriate for programmer errors, but may be unfriendly for downstream crate consumers. | Open — draft uses panics; evaluate whether `Result` returns improve ergonomics |
| 6 | Should `FlatBlockStorage` store full multi-dimensional shapes for non-rank-2 blocks, or is the `(numel, 1)` fallback acceptable given that `flatten` is only used on the rank-2 GEMM path? | Open — current design loses shape for rank != 2 |
| 7 | Should `from_raw_parts` remain `pub(crate)`, or should it be promoted to a public unsafe constructor for advanced use cases? | Open — currently `pub(crate)` escape hatch for `unflatten` |
| 8 | Should property-based tests (`proptest`) be added as a priority before stabilizing the API? 40 unit tests pass, but no `proptest` coverage exists. | Open — planned for next iteration |
