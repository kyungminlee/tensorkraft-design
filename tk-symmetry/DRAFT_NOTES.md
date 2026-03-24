# tk-symmetry Draft Notes

**Status:** All source files compile. 72 tests pass (57 unit tests including SU(2) tests
behind the `su2-symmetry` feature flag, plus 15 property-based tests via proptest).
Criterion benchmarks are set up for `get_block` performance validation.
**Based on:** `techspec/2_tech-spec_tk-symmetry.md` and `tensorkraft_architecture_design_v8_4.md`

---

## What is implemented

| Module | Status | Description |
|:-------|:-------|:------------|
| `quantum_number.rs` | Complete | `QuantumNumber` trait, `BitPackable` trait, `LegDirection` enum |
| `builtins.rs` | Complete | `U1`, `Z2`, `U1Z2`, `U1Wide` with pack/unpack + unit tests |
| `sector_key.rs` | Complete | `PackedSectorKey`, `PackedSectorKey128`, `QIndex<Q>` with `assert_invariants` |
| `flux.rs` | Complete | `check_flux_rule`, `enumerate_valid_sectors` with backtracking + last-leg pruning + partial-fusion memoization for rank > 4 |
| `block_sparse.rs` | Complete | `BlockSparseTensor<T, Q>` with constructors, sector access, insert, permute, fuse_legs, split_leg, debug invariants, fallible `try_from_blocks`/`try_insert_block` |
| `flat_storage.rs` | Complete | `FlatBlockStorage<'a, T>` with `flatten`/`unflatten` round-trip, `full_shapes` for higher-rank block preservation |
| `formats.rs` | Complete | `SparsityFormat` enum |
| `error.rs` | Complete | `SymmetryError`, `SymResult` |
| `su2/mod.rs` | Complete | `SU2Irrep`, `WignerEckartTensor<T>` with full API |
| `su2/cg_cache.rs` | Complete | `ClebschGordanCache` with Racah formula, `prefill`, 6j and 9j symbols |

### QIndex invariant checking (complete)

`#[cfg(debug_assertions)] fn assert_invariants()` on `QIndex<Q>` verifies: (1) sectors are strictly sorted by quantum number, (2) all sector dimensions are non-zero, (3) `total_dim` equals the sum of sector dims. `QIndex::new()` calls `assert_invariants` in debug builds.

### Fallible user-facing APIs (complete)

- `BlockSparseTensor::try_from_blocks()` — returns `SymResult` instead of panicking on flux rule violation, dimension mismatch, or duplicate sectors.
- `BlockSparseTensor::try_insert_block()` — returns `SymResult` instead of panicking on flux rule violation.
- Original panicking methods (`from_blocks`, `insert_block`) remain for internal use.

### FlatBlockStorage shape preservation (complete)

`full_shapes: Vec<Vec<usize>>` field stores original multi-dimensional block shapes (not just BLAS-compatible `(rows, cols)` view). `flatten()` populates from original block dimensions; `unflatten()` uses them to reconstruct, correctly handling higher-rank blocks (e.g., rank-3 MPS tensors).

### Sector enumeration memoization (complete)

For rank > 4 tensors, `enumerate_valid_sectors` uses partial-fusion memoization: reachable suffix charges at each depth are precomputed right-to-left, then used to prune branches. Reduces search cost from exponential in rank to polynomial in the number of distinct charges. Validated with rank-5 (51 sectors) and rank-6 (141 sectors) tests.

### ClebschGordan coefficient validation (complete)

Racah formula validated for spins up to j=2: spin-1/2 singlet and triplet coupling, spin-1 stretched states, spin-1/2 ⊗ spin-1 coupling, spin-2 stretched states, spin-2 → j=0 coupling, and orthonormality sum rules. Internal Racah implementation retained (no `lie-groups` dependency).

### WignerEckartTensor API (complete)

`WignerEckartTensor<T>` has a complete API: `with_cache()`, `get_reduced_mut()`, `nnz()`, `iter_reduced()` / `iter_reduced_mut()`, `prefill_structural()`, `contains_sector()` / `remove_reduced()`, `cache()`. Contraction callback machinery belongs to `tk-contract`.

### 6j / 9j symbols (complete)

- **Wigner 6j symbols** in `ClebschGordanCache::sixj()` using the Racah formula with triangle coefficient factorization. Cached via `RwLock<HashMap>`. `prefill_sixj()` pre-populates up to `twice_j_max`.
- **Wigner 9j symbols** in `ClebschGordanCache::ninej()` via summation over 6j symbols using the standard identity. Triangle inequality bounds constrain the summation variable.
- Both validated with tests against known analytical values.

---

## Testing status

72 tests total (57 unit + 15 proptest, with `su2-symmetry` feature):
- Quantum number group axioms (identity/inverse/associativity via proptest)
- Pack/unpack round-trips (exhaustive + proptest)
- Sector key ordering, binary search correctness (proptest)
- Overflow detection, flux rule validation
- Sector enumeration (including memoized high-rank path)
- Block-sparse construction/access/insert/permute/fuse_legs/split_leg
- Fuse nnz preservation (proptest)
- Flatten/unflatten round-trip
- SU(2) irrep algebra
- CG coefficient computation (up to j=2 with orthonormality checks)
- 6j symbols (analytical values, symmetry, triangle violations)
- 9j symbols (trivial and non-trivial values)

Criterion benchmarks: `get_block` (hit/miss) and `iter_keyed_blocks` on 100-sector tensors.

---

## Remaining limitations

1. **`fuse_legs` / `split_leg` not benchmarked for large tensors** — Round-trip preserves data exactly. `split_leg` takes an additional `original_directions: Vec<LegDirection>` parameter beyond the tech spec signature (needed to reconstruct the fuse map correctly for mixed-direction legs). Performance at large scale not yet validated.

2. **CG numerical stability for j > 10** — The Racah formula uses direct factorial arithmetic. For production use with higher spins, switching to log-factorial arithmetic is straightforward.

3. **`lie-groups` dependency** — Decision made: keep internal Racah formula. It passes all CG, 6j, and 9j validation tests. The `lie-groups` crate would add an external dependency for no additional correctness benefit.
