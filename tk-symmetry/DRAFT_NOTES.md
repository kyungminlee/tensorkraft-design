# tk-symmetry Draft Notes

## Summary

This is a draft implementation of the `tk-symmetry` crate based on the tech spec
(`techspec/2_tech-spec_tk-symmetry.md`) and the architecture design document
(`tensorkraft_architecture_design_v8_4.md`).

**Status:** All source files compile. 72 tests pass (57 unit tests including SU(2) tests
behind the `su2-symmetry` feature flag, plus 15 property-based tests via proptest).
Criterion benchmarks are set up for `get_block` performance validation.

## What is implemented

| Module | Status | Description |
|:-------|:-------|:------------|
| `quantum_number.rs` | Complete | `QuantumNumber` trait, `BitPackable` trait, `LegDirection` enum |
| `builtins.rs` | Complete | `U1`, `Z2`, `U1Z2`, `U1Wide` with pack/unpack + unit tests |
| `sector_key.rs` | Complete | `PackedSectorKey`, `PackedSectorKey128`, `QIndex<Q>` with `assert_invariants` |
| `flux.rs` | Complete | `check_flux_rule`, `enumerate_valid_sectors` with backtracking + last-leg pruning |
| `block_sparse.rs` | Core done | `BlockSparseTensor<T, Q>` with constructors, sector access, insert, permute, fuse_legs, split_leg, debug invariants, fallible `try_from_blocks`/`try_insert_block` |
| `flat_storage.rs` | Core done | `FlatBlockStorage<'a, T>` with `flatten`/`unflatten` round-trip, `full_shapes` for higher-rank block preservation |
| `formats.rs` | Complete | `SparsityFormat` enum |
| `error.rs` | Complete | `SymmetryError`, `SymResult` |
| `su2/mod.rs` | Draft | `SU2Irrep`, `WignerEckartTensor<T>` |
| `su2/cg_cache.rs` | Draft | `ClebschGordanCache` with Racah formula, `prefill` |

## What was added in this iteration

### QIndex invariant checking

- Added `#[cfg(debug_assertions)] fn assert_invariants()` on `QIndex<Q>`, which
  verifies: (1) sectors are strictly sorted by quantum number, (2) all sector
  dimensions are non-zero, (3) `total_dim` equals the sum of sector dims.
- `QIndex::new()` now calls `assert_invariants` in debug builds.

### Fallible user-facing APIs

- Added `BlockSparseTensor::try_from_blocks()` — returns `SymResult` instead of
  panicking on flux rule violation, dimension mismatch, or duplicate sectors.
- Added `BlockSparseTensor::try_insert_block()` — returns `SymResult` instead of
  panicking on flux rule violation.
- The original panicking methods (`from_blocks`, `insert_block`) remain for
  internal use where correctness is guaranteed by construction.

### FlatBlockStorage shape preservation

- Added `full_shapes: Vec<Vec<usize>>` field to `FlatBlockStorage`, storing the
  original multi-dimensional shape of each block (not just the BLAS-compatible
  `(rows, cols)` view).
- `flatten()` populates `full_shapes` from the original block dimensions.
- `unflatten()` uses `full_shapes` to reconstruct original block shapes, correctly
  handling higher-rank blocks (e.g., rank-3 MPS tensors) through the round-trip.

### Property-based tests (proptest)

- Added `tests/proptest_symmetry.rs` with 15 property-based tests covering:
  - Group axioms (identity, inverse, associativity) for U1, Z2, U1Z2
  - Pack/unpack round-trips for U1, Z2, U1Z2
  - `PackedSectorKey` round-trip for arbitrary U1 vectors
  - Binary search finds inserted keys in sorted key lists
  - Fuse-legs nnz preservation on rank-4 tensors with random charge sets

### Criterion benchmarks

- Added `benches/get_block.rs` with Criterion benchmarks:
  - `get_block` hit/miss on a 100-sector tensor (tech spec target: < 10 ns)
  - `iter_keyed_blocks` full iteration
- Added `criterion` dev-dependency and `[[bench]]` section to `Cargo.toml`.

## Limitations and missing pieces

### BlockSparseTensor

- **`fuse_legs` / `split_leg`**: Implemented. `fuse_legs(Range<usize>)` fuses a
  contiguous range of legs into one combined leg; `split_leg` reverses it.
  The fused leg direction is always `Incoming`. `split_leg` takes an additional
  `original_directions: Vec<LegDirection>` parameter beyond the tech spec signature
  (needed to reconstruct the fuse map correctly for mixed-direction legs).
  Round-trip `fuse_legs → split_leg` preserves data exactly. Not yet benchmarked
  for large tensors.

### SU(2) / Non-Abelian

- ~~**ClebschGordanCache**: Not validated for higher spins~~ **(RESOLVED)**
  The Racah formula implementation is now validated for spins up to j=2 with
  comprehensive tests: spin-1/2 singlet and triplet coupling, spin-1 stretched
  states, spin-1/2 ⊗ spin-1 coupling, spin-2 stretched states, spin-2 → j=0
  coupling, and orthonormality sum rules for both spin-1/2 and spin-1. The
  internal Racah implementation is retained (no `lie-groups` dependency)
  as it passes all validation tests and avoids an external dependency.

- ~~**WignerEckartTensor**: Minimal API~~ **(RESOLVED)**
  `WignerEckartTensor<T>` now has a complete API:
  - `with_cache()` for shared CG cache construction
  - `get_reduced_mut()` for mutable block access
  - `nnz()` for total element count
  - `iter_reduced()` / `iter_reduced_mut()` for block iteration
  - `prefill_structural()` to pre-populate CG and 6j caches
  - `contains_sector()` / `remove_reduced()` for sector management
  - `cache()` for access to the structural CG cache
  Contraction callback machinery remains out-of-scope for tk-symmetry
  (belongs to `tk-contract`'s `structural_contraction` injection point).

- ~~**6j / 9j symbols**: Not implemented~~ **(RESOLVED)**
  - **Wigner 6j symbols** implemented in `ClebschGordanCache::sixj()` using
    the Racah formula with triangle coefficient factorization. Cached via
    `RwLock<HashMap>` with the same read-fast/write-on-miss pattern as CG
    coefficients. `prefill_sixj()` pre-populates up to `twice_j_max`.
  - **Wigner 9j symbols** implemented in `ClebschGordanCache::ninej()` via
    summation over 6j symbols using the standard identity. Triangle inequality
    bounds constrain the summation variable for efficiency.
  - Both validated with tests against known analytical values.

- **`lie-groups` dependency**: Decision made — keep the internal Racah formula
  implementation. It passes all CG, 6j, and 9j validation tests. The
  `lie-groups` crate would add an external dependency for no additional
  correctness benefit. For production, if higher-j numerical stability becomes
  a concern (j > 10), switching to log-factorial arithmetic is straightforward.

### General

- ~~**Sector enumeration**: No memoization~~ **(RESOLVED)**
  For rank > 4 tensors, `enumerate_valid_sectors` now uses partial-fusion
  memoization: the set of reachable suffix charges at each depth is precomputed
  from right-to-left, then used to prune branches whose partial fusion cannot
  reach the target flux. This reduces search cost from exponential in rank to
  polynomial in the number of distinct charges. The rank ≤ 4 path uses the
  original backtracking algorithm (which is already efficient for low rank).
  Validated with rank-5 (51 sectors) and rank-6 (141 sectors) tests.

## Test coverage

- 72 tests total (57 unit + 15 proptest, with `su2-symmetry` feature)
- Covers: quantum number group axioms (identity/inverse/associativity via proptest),
  pack/unpack round-trips (exhaustive + proptest), sector key ordering,
  binary search correctness (proptest), overflow detection, flux rule validation,
  sector enumeration (including memoized high-rank path),
  block-sparse construction/access/insert/permute/fuse_legs/split_leg,
  fuse nnz preservation (proptest), flatten/unflatten round-trip, SU(2) irrep
  algebra, CG coefficient computation (up to j=2 with orthonormality checks),
  6j symbols (analytical values, symmetry, triangle violations),
  9j symbols (trivial and non-trivial values)
- Criterion benchmarks: `get_block` (hit/miss) and `iter_keyed_blocks` on 100-sector tensors
