# tk-symmetry Draft Notes

## Summary

This is a draft implementation of the `tk-symmetry` crate based on the tech spec
(`techspec/2_tech-spec_tk-symmetry.md`) and the architecture design document
(`tensorkraft_architecture_design_v8_4.md`).

**Status:** All source files compile. 40 unit tests pass (including SU(2) tests
behind the `su2-symmetry` feature flag).

## What is implemented

| Module | Status | Description |
|:-------|:-------|:------------|
| `quantum_number.rs` | Complete | `QuantumNumber` trait, `BitPackable` trait, `LegDirection` enum |
| `builtins.rs` | Complete | `U1`, `Z2`, `U1Z2`, `U1Wide` with pack/unpack + unit tests |
| `sector_key.rs` | Complete | `PackedSectorKey`, `PackedSectorKey128`, `QIndex<Q>` |
| `flux.rs` | Complete | `check_flux_rule`, `enumerate_valid_sectors` with backtracking + last-leg pruning |
| `block_sparse.rs` | Core done | `BlockSparseTensor<T, Q>` with constructors, sector access, insert, permute, fuse_legs, split_leg, debug invariants |
| `flat_storage.rs` | Core done | `FlatBlockStorage<'a, T>` with `flatten`/`unflatten` round-trip |
| `formats.rs` | Complete | `SparsityFormat` enum |
| `error.rs` | Complete | `SymmetryError`, `SymResult` |
| `su2/mod.rs` | Draft | `SU2Irrep`, `WignerEckartTensor<T>` |
| `su2/cg_cache.rs` | Draft | `ClebschGordanCache` with Racah formula, `prefill` |

## Limitations and missing pieces

### BlockSparseTensor

- **`fuse_legs` / `split_leg`**: Implemented. `fuse_legs(Range<usize>)` fuses a
  contiguous range of legs into one combined leg; `split_leg` reverses it.
  The fused leg direction is always `Incoming`. `split_leg` takes an additional
  `original_directions: Vec<LegDirection>` parameter beyond the tech spec signature
  (needed to reconstruct the fuse map correctly for mixed-direction legs).
  Round-trip `fuse_legs → split_leg` preserves data exactly. Not yet benchmarked
  for large tensors.

- **Property-based tests**: The tech spec calls for `proptest`-driven tests
  (e.g., `packed_key_binary_search_finds_inserted`, `u1_fuse_associativity`).
  These are not yet written; only deterministic unit tests are present.

- **CI benchmarks**: The spec requires Criterion benchmarks verifying
  `get_block` < 10 ns on a 100-sector tensor. Not yet set up.

### FlatBlockStorage

- **Block shape handling**: Currently treats blocks as rank-2 `(rows, cols)` or
  falls back to `(numel, 1)`. Higher-rank blocks (rank-3 MPS tensors) need
  proper shape preservation through the flatten/unflatten round-trip.

- **Integration with SparseLinAlgBackend**: The `flatten` method is implemented
  but not yet consumed by any GEMM dispatch code (that lives in `tk-linalg`).

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

- **No `#[cfg(debug_assertions)] fn assert_invariants`** on QIndex (only on
  BlockSparseTensor).

- **Sector enumeration**: Uses backtracking with last-leg pruning. No
  memoization for high-rank tensors (open question #4 in the spec).

- **Error paths**: Most error conditions use `debug_assert!` / panic rather
  than returning `SymmetryError`. Production code should use `Result` returns
  for user-facing operations.

## Test coverage

- 40 tests total (with `su2-symmetry` feature)
- Covers: quantum number axioms, pack/unpack round-trips, sector key ordering,
  overflow detection, flux rule validation, sector enumeration, block-sparse
  construction/access/insert/permute/fuse_legs/split_leg, flatten/unflatten round-trip, SU(2) irrep
  algebra, CG coefficient computation
