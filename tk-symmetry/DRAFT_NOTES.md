# tk-symmetry Draft Notes

## Summary

This is a draft implementation of the `tk-symmetry` crate based on the tech spec
(`techspec/2_tech-spec_tk-symmetry.md`) and the architecture design document
(`tensorkraft_architecture_design_v8_4.md`).

**Status:** All source files compile. 55 tests pass (40 unit tests including SU(2) tests
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

- **ClebschGordanCache**: The Racah formula implementation is a draft. It
  passes basic tests (trivial coupling, spin-1/2 coupling, selection rules,
  ⟨1,0;1,0|0,0⟩) but has not been validated against a reference library for
  higher spins (j > 2). Numerical stability for large j values is unverified.

- **WignerEckartTensor**: The struct is defined with the correct fields
  (`structural`, `reduced`, `flux`) but has minimal API beyond
  `new`/`insert_reduced`/`get_reduced`. No contraction or structural_contraction
  callback machinery is implemented.

- **6j / 9j symbols**: Not implemented. Required for recoupling in multi-site
  operations.

- **`lie-groups` dependency**: The tech spec lists `lie-groups` as an optional
  dependency for CG coefficients. The current draft uses a hand-rolled Racah
  formula instead. Should be evaluated whether to keep the internal
  implementation or switch to `lie-groups` for production.

### General

- **Sector enumeration**: Uses backtracking with last-leg pruning. No
  memoization for high-rank tensors (open question #4 in the spec).

## Test coverage

- 55 tests total (40 unit + 15 proptest, with `su2-symmetry` feature)
- Covers: quantum number group axioms (identity/inverse/associativity via proptest),
  pack/unpack round-trips (exhaustive + proptest), sector key ordering,
  binary search correctness (proptest), overflow detection, flux rule validation,
  sector enumeration, block-sparse construction/access/insert/permute/fuse_legs/split_leg,
  fuse nnz preservation (proptest), flatten/unflatten round-trip, SU(2) irrep
  algebra, CG coefficient computation
- Criterion benchmarks: `get_block` (hit/miss) and `iter_keyed_blocks` on 100-sector tensors
