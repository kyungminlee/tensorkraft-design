# tk-dmrg Draft Implementation Notes

**Status:** Draft implementation — compiles, 16 tests pass, core type scaffolding functional with working Lanczos eigensolver. Sweep engine, environments, TDVP, and MPO compilation are skeletons.
**Date:** March 2026

---

## What Was Implemented

- **mps/mod.rs** — `MPS<T, Q, Gauge>` with typestate markers (`LeftCanonical`, `RightCanonical`, `MixedCanonical`, `BondCentered`), gauge transitions, canonicalization stubs
- **mpo/mod.rs** — `MPO<T, Q>` with accessors, `MpoCompiler` skeleton, `MpoCompressionConfig`
- **environments/mod.rs** — `Environment<T, Q>`, `Environments<T, Q>` with boundary construction, `build_heff_two_site`/`build_heff_single_site` stubs
- **eigensolver/mod.rs** — `IterativeEigensolver<T>` trait, `LanczosSolver` with working Lanczos + Sturm bisection, `DavidsonSolver` and `BlockDavidsonSolver` (delegating to Lanczos)
- **truncation/mod.rs** — `TruncationConfig`, `TruncationResult<T>`, `BondDimensionSchedule` with warmup/fixed/custom
- **sweep/mod.rs** — `DMRGEngine<T, Q, B>`, `DMRGConfig`, `DMRGStats`, `SweepSchedule`, `UpdateVariant`, cancellation support
- **tdvp/mod.rs** — `TdvpDriver<T, Q, B>`, `TdvpStabilizationConfig`, `exp_krylov` stub, soft D_max logic
- **excited/mod.rs** — `ExcitedStateConfig`, `build_heff_penalized` stub
- **idmrg/mod.rs** — `IDmrgConfig`, `run_idmrg` stub
- **checkpoint.rs** — `DMRGCheckpoint` skeleton
- **error.rs** — Full `DmrgError` enum with 14 variants

**Not implemented (skeletons only):**
- OpSum → MPO FSA + SVD compilation (`MpoCompiler::compile`)
- Environment contraction (grow_left/grow_right)
- H_eff matvec closure construction (build_heff_two_site/single_site)
- Actual DMRG step logic (eigensolve → SVD truncate → update MPS → update env)
- TDVP time evolution steps (Krylov exp, bond evolution, subspace expansion)
- Davidson preconditioned iteration (currently delegates to Lanczos)
- Block-Davidson block operations
- iDMRG unit-cell extension loop
- Checkpoint serialization (requires serde on BlockSparseTensor)

---

## Design Issues Discovered During Implementation

### 1. `BitPackable` does not require `Clone` — causes ergonomic friction everywhere

The `Q: BitPackable` bound is used throughout `MPS<T, Q, Gauge>`, `MPO<T, Q>`, and `Environment<T, Q>`. However, `BitPackable` does not require `Clone`. Since quantum number types (U1, Z2) need to be copied/cloned frequently (e.g., constructing boundary environments, moving between structs), every function that passes `Q` values needs explicit `.clone()` calls.

**Workaround used:** Added `.clone()` at every `Q` transfer point.

**Recommendation:** Either:
- Add `Clone` as a supertrait of `BitPackable` (most natural — all quantum number types are small Copy types)
- Add `Copy` as a supertrait (even better — U1(i32) and Z2(bool) are both Copy)

**Severity:** Medium — affects nearly every function in the crate.

### 2. `LegDirection` variants are `Incoming`/`Outgoing`, not `In`/`Out`

The tech spec and architecture doc use abbreviated names in code sketches (`In`, `Out`) that don't match the actual implementation (`Incoming`, `Outgoing`). This caused compilation errors during implementation.

**Recommendation:** Either update the specs to use the actual variant names, or add type aliases `const In: LegDirection = LegDirection::Incoming`.

**Severity:** Low — one-time fix.

### 3. `QIndex` has `total_dim()` not `dim()` — spec uses `dim()`

The tech spec refers to `QIndex::dim()` for getting the total dimension of an index. The actual API is `total_dim()`. This is a minor naming discrepancy but caused compilation errors.

**Severity:** Low.

### 4. `BlockSparseTensor` doesn't implement `Clone` — blocks MPS/MPO cloning

Same issue as discovered in tk-contract: `BlockSparseTensor<T, Q>` contains `Vec<DenseTensor<'static, T>>` blocks where `DenseTensor` doesn't implement `Clone`. This means `MPS` and `MPO` can't easily be cloned, which is needed for:
- Excited-state DMRG (storing penalized states)
- Checkpoint save/restore
- Energy variance computation (need H²|ψ⟩)
- iDMRG bootstrap (duplicating unit cell)

**Workaround used:** Avoided cloning in the draft by using references and move semantics where possible.

**Recommendation:** Implement `Clone` for `DenseTensor<'static, T>` in tk-core. This is the root cause of cloning issues across tk-contract, tk-dsl, and tk-dmrg.

**Severity:** High — blocks multiple key features.

### 5. `IterativeEigensolver<T>` trait is not object-safe if `T` is in generic position

The spec stores the eigensolver as `Box<dyn IterativeEigensolver<f64>>` in `DMRGConfig`. This works because `f64` is concrete. But for generic `T`, the trait is object-safe only if `T` is fixed. This means `DMRGConfig` is effectively non-generic over `T` for the eigensolver field.

**Workaround used:** Fixed `DMRGConfig::eigensolver` to `Box<dyn IterativeEigensolver<f64>>` as the spec recommends. Complex-valued TDVP would need a separate eigensolver field or a different approach.

**Assessment:** This is the right design for Phase 1-4 (f64 only). The spec's Open Question #2 correctly identifies this tension.

**Severity:** Low for now — will need addressing for complex TDVP.

### 6. `SweepArena` reset timing vs. ownership transfer is subtle

The spec says step 7 of the DMRG sweep is `arena.reset()`, but step 4 says "ownership transfer must complete before step 7." In practice, the SVD results (U, S, V†) are produced as arena-allocated tensors that must be copied to owned storage before the arena resets. The `into_owned()` method on `DenseTensor` handles this, but the ordering constraint is implicit and easy to violate.

**Recommendation:** Add a comment or assertion in `DMRGEngine::dmrg_step_two_site` documenting that all arena tensors must be `into_owned()` before `arena.reset()`. Consider a RAII guard that prevents arena use after reset.

**Severity:** Medium — correctness-critical but easy to get right with discipline.

### 7. `DMRGConfig` mixes static configuration with runtime state

`DMRGConfig` contains both static configuration (`max_sweeps`, `energy_tol`) and runtime-dynamic state (`bond_dim_schedule` changes over sweeps, `eigensolver` has mutable `diagonal` field on DavidsonSolver). This makes it awkward to take `&self` on the config while also needing to mutate the eigensolver's preconditioner.

**Recommendation:** Split into `DMRGConfig` (immutable) and `DMRGState` (mutable runtime state). Or make the eigensolver preconditioner injection a method on the engine rather than a field on the solver.

**Severity:** Medium — affects code clarity.

### 8. No `serde` integration yet — checkpoint is non-functional

The checkpoint module requires serializing `BlockSparseTensor<T, Q>`, which requires serde impls on types in `tk-symmetry`. This is an open question in the spec (OQ #8: derive serde vs. proxy types). Without this, checkpoint save/restore is non-functional.

**Recommendation:** Decide on the serde strategy early:
- `#[derive(Serialize, Deserialize)]` on `BlockSparseTensor` and `DenseTensor` (simplest)
- Proxy types in `tk-dmrg` (avoids serde in lower crates but more boilerplate)

**Severity:** Medium — needed for crash recovery but not for basic DMRG.

### 9. Lanczos eigensolver tridiagonal solve is naive

The draft Lanczos uses Sturm bisection for eigenvalue finding and a simple inverse iteration for the eigenvector. This is O(n²) for the tridiagonal problem where LAPACK's `dstev` is O(n). For small Krylov dimensions (20-100) this is fine, but for thick restarts with large Krylov spaces it would be slow.

**Recommendation:** Use LAPACK `dstev` via `tk-linalg` for the tridiagonal eigenvalue problem. The `eigh_lowest` method on `LinAlgBackend` could be used after constructing the tridiagonal matrix as a dense matrix.

**Severity:** Low for draft — would need fixing for production.

### 10. DMRGEngine owns both MPS and backend — limits flexibility

The engine takes ownership of the MPS, MPO, and backend. This means you can't easily share a backend across multiple engines (e.g., ground state + excited states) or inspect the MPS during a sweep without going through engine accessors.

**Recommendation:** Consider `&B` or `Arc<B>` for the backend, similar to the pattern discussed for `ContractionExecutor` in tk-contract. The MPS ownership is intentional (engine is the sole mutator during sweeps).

**Severity:** Low.

---

## Gaps Between Spec and Reality

| Spec Section | Issue | Severity |
|:-------------|:------|:---------|
| §3 (MPS) | `BlockSparseTensor` not cloneable; MPS can't be cloned for excited states or checkpointing | High |
| §5 (MPO) | FSA + SVD compilation not implemented; OpSum → MPO is a skeleton | High (but expected for draft) |
| §7 (Environments) | Environment contraction not implemented; build_heff returns no-op closures | High (but expected) |
| §8 (Eigensolvers) | Davidson/Block-Davidson delegate to Lanczos; no diagonal preconditioner | Medium |
| §12 (Checkpoint) | serde integration blocked by BlockSparseTensor lacking Serialize | Medium |
| §14 (Cargo.toml) | `backend-oxiblas` referenced in spec defaults but feature doesn't exist | Low |
| General | `BitPackable` doesn't require `Clone`; excessive `.clone()` calls | Medium |
| General | `QIndex::dim()` in spec vs `total_dim()` in reality | Low |

---

## Consistency Issues: Tech Spec vs. ARCHITECTURE.md

| Item | Tech Spec | Architecture | Assessment |
|:-----|:----------|:-------------|:-----------|
| DMRGConfig | `bond_dim_schedule: BondDimensionSchedule` | `max_bond_dim: usize` | Spec is more complete (per-sweep scheduling) |
| DMRGConfig | `update_variant: UpdateVariant` | Not present | Spec adds needed field |
| DMRGEngine | `arena: SweepArena` | `krylov_workspace: KrylovWorkspace<T>` | Different approaches; both needed |
| EigenResult | `matvec_count: usize` | `iterations: usize` | Different naming; matvec_count is more precise |
| DMRGConfig | `checkpoint_path: Option<PathBuf>` | `environment_offload: Option<PathBuf>` | Different purposes (checkpoint vs env disk cache) |
| Environments | Wrapped in `Environment<T,Q>` with `up_to` | Raw `Vec<BlockSparseTensor>` | Spec adds useful metadata |
| TdvpStabilizationConfig | `tikhonov_delta: f64` only | Adds `adaptive_tikhonov`, `tikhonov_delta_scale`, `tikhonov_delta_min` | Architecture is more detailed |
| DMFT | Out of scope for tk-dmrg | §8.4 places DMFT loop in tk-dmrg | Architecture mixes concerns; spec correctly separates |

---

## What Works Well

1. **Typestate canonical forms** — Compile-time gauge enforcement prevents entire categories of bugs (e.g., calling `dmrg_step` on a `LeftCanonical` MPS).

2. **`IterativeEigensolver<T>` trait** — Clean abstraction for swapping eigensolvers. The `Box<dyn IterativeEigensolver<f64>>` in `DMRGConfig` enables runtime solver selection.

3. **`BondDimensionSchedule`** — Geometric warmup ramp is the standard DMRG practice; having it built into the config prevents users from manually managing per-sweep D values.

4. **Lanczos with full reorthogonalization** works correctly on small test cases and the Sturm bisection eigenvalue finder is numerically stable.

5. **`SweepSchedule`** with left-to-right + right-to-left half-sweeps correctly implements the standard DMRG sweep pattern.

6. **TDVP soft D_max policy** — The exponential decay of bond dimension overshoot prevents oscillation while allowing temporary growth during subspace expansion.

7. **`DMRGStats`** accumulator — Tracking per-sweep energies, truncation errors, and bond dimensions is essential for monitoring convergence.

8. **Cancellation support** — `AtomicBool` cancellation flag enables clean interruption of long sweeps without data corruption.
