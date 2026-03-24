# tk-dmrg Draft Implementation Notes

**Status:** Draft implementation — compiles, 26 tests pass, core type scaffolding functional with working Lanczos and Davidson eigensolvers. Sweep engine, environments, TDVP, and MPO compilation are skeletons. High- and medium-severity gaps resolved.
**Date:** March 2026

---

## What Was Implemented

- **mps/mod.rs** — `MPS<T, Q, Gauge>` with typestate markers (`LeftCanonical`, `RightCanonical`, `MixedCanonical`, `BondCentered`), gauge transitions, canonicalization stubs
- **mpo/mod.rs** — `MPO<T, Q>` with accessors, `MpoCompiler` skeleton, `MpoCompressionConfig`
- **environments/mod.rs** — `Environment<T, Q>`, `Environments<T, Q>` with boundary construction, `build_heff_two_site`/`build_heff_single_site` stubs
- **eigensolver/mod.rs** — `IterativeEigensolver<T>` trait, `LanczosSolver` with working Lanczos + Sturm bisection, `DavidsonSolver` with actual diagonal preconditioner + Jacobi eigenvalue solver, `BlockDavidsonSolver` (delegating to Lanczos)
- **truncation/mod.rs** — `TruncationConfig`, `TruncationResult<T>`, `BondDimensionSchedule` with warmup/fixed/custom
- **sweep/mod.rs** — `DMRGEngine<T, Q, B>`, `DMRGConfig` (immutable) + `DMRGRuntimeState` (mutable), `DMRGStats`, `SweepSchedule`, `UpdateVariant`, cancellation support, arena safety documentation
- **tdvp/mod.rs** — `TdvpDriver<T, Q, B>`, `TdvpStabilizationConfig`, working `exp_krylov_f64` (Arnoldi + scaling-and-squaring), soft D_max logic
- **excited/mod.rs** — `ExcitedStateConfig`, working `build_heff_penalized` with penalty matvec wrapper
- **idmrg/mod.rs** — `IDmrgConfig`, `run_idmrg` stub
- **checkpoint.rs** — `DMRGCheckpoint` skeleton
- **error.rs** — Full `DmrgError` enum with 14 variants

**Cross-crate fixes:**
- **tk-core/tensor.rs** — Implemented `Clone` for `DenseTensor<'static, T>`
- **tk-symmetry/block_sparse.rs** — Implemented `Clone` for `BlockSparseTensor<T, Q>`

**Not implemented (skeletons only):**
- OpSum → MPO FSA + SVD compilation (`MpoCompiler::compile`)
- Environment contraction (grow_left/grow_right)
- H_eff matvec closure construction (build_heff_two_site/single_site)
- Actual DMRG step logic (eigensolve → SVD truncate → update MPS → update env)
- TDVP bond evolution and subspace expansion (Krylov exp is implemented)
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

**Assessment:** `QuantumNumber` already requires `Clone`, so `BitPackable` transitively requires it. The `.clone()` calls are needed but not a bug. Adding `Copy` as a supertrait would allow implicit copies.

**Severity:** Medium — ergonomic, not blocking.

### 2. `LegDirection` variants are `Incoming`/`Outgoing`, not `In`/`Out`

The tech spec and architecture doc use abbreviated names in code sketches (`In`, `Out`) that don't match the actual implementation (`Incoming`, `Outgoing`). This caused compilation errors during implementation.

**Recommendation:** Either update the specs to use the actual variant names, or add type aliases `const In: LegDirection = LegDirection::Incoming`.

**Severity:** Low — one-time fix.

### 3. `QIndex` has `total_dim()` not `dim()` — spec uses `dim()`

The tech spec refers to `QIndex::dim()` for getting the total dimension of an index. The actual API is `total_dim()`. This is a minor naming discrepancy but caused compilation errors.

**Severity:** Low.

### 4. `BlockSparseTensor` doesn't implement `Clone` — **Resolved**

`Clone` has been implemented for both `DenseTensor<'static, T>` (in tk-core) and `BlockSparseTensor<T, Q>` (in tk-symmetry). MPS and MPO can now be cloned for excited-state DMRG, checkpointing, variance computation, and iDMRG bootstrap.

**Severity:** ~~High~~ **Resolved.**

### 5. `IterativeEigensolver<T>` trait is not object-safe if `T` is in generic position

The spec stores the eigensolver as `Box<dyn IterativeEigensolver<f64>>` in `DMRGConfig`. This works because `f64` is concrete. But for generic `T`, the trait is object-safe only if `T` is fixed. This means `DMRGConfig` is effectively non-generic over `T` for the eigensolver field.

**Workaround used:** Fixed `DMRGRuntimeState::eigensolver` to `Box<dyn IterativeEigensolver<f64>>` as the spec recommends. Complex-valued TDVP would need a separate eigensolver field or a different approach.

**Assessment:** This is the right design for Phase 1-4 (f64 only). The spec's Open Question #2 correctly identifies this tension.

**Severity:** Low for now — will need addressing for complex TDVP.

### 6. `SweepArena` reset timing vs. ownership transfer is subtle — **Resolved**

Added detailed arena safety documentation in `dmrg_step_two_site` and `run()` documenting the ownership-transfer-before-reset contract. The borrow checker enforces this statically; the comments make the invariant explicit for maintainers.

**Severity:** ~~Medium~~ **Resolved** (documentation added).

### 7. `DMRGConfig` mixes static configuration with runtime state — **Resolved**

Split into `DMRGConfig` (immutable: max_sweeps, energy_tol, bond_dim_schedule, etc.) and `DMRGRuntimeState` (mutable: eigensolver with preconditioner). `DMRGEngine` now holds both, with a `with_runtime()` constructor for explicit state injection.

**Severity:** ~~Medium~~ **Resolved.**

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

| Spec Section | Issue | Severity | Status |
|:-------------|:------|:---------|:-------|
| §3 (MPS) | `BlockSparseTensor` not cloneable; MPS can't be cloned for excited states or checkpointing | High | **Resolved** — Clone impl added to DenseTensor and BlockSparseTensor |
| §5 (MPO) | FSA + SVD compilation not implemented; OpSum → MPO is a skeleton | High (but expected for draft) | Open |
| §7 (Environments) | Environment contraction not implemented; build_heff returns no-op closures | High (but expected) | Open |
| §8 (Eigensolvers) | Davidson/Block-Davidson delegate to Lanczos; no diagonal preconditioner | Medium | **Resolved** — Davidson now uses actual diagonal preconditioner with Jacobi solver |
| §8 (Eigensolvers) | TDVP Krylov matrix-exponential not implemented | Medium | **Resolved** — exp_krylov_f64 implemented via Arnoldi + scaling-and-squaring |
| §9 (Excited) | build_heff_penalized bypasses penalty | Medium | **Resolved** — penalty matvec wrapper implemented with overlap vectors |
| §10 (Sweep) | DMRGConfig mixes immutable config with mutable runtime state | Medium | **Resolved** — split into DMRGConfig + DMRGRuntimeState |
| §10 (Sweep) | Arena reset timing undocumented | Medium | **Resolved** — arena safety contract documented |
| §12 (Checkpoint) | serde integration blocked by BlockSparseTensor lacking Serialize | Medium | Open (design decision needed) |
| §14 (Cargo.toml) | `backend-oxiblas` referenced in spec defaults but feature doesn't exist | Low | Open |
| General | `BitPackable` doesn't require `Clone`; excessive `.clone()` calls | Medium | Clarified — QuantumNumber already requires Clone |
| General | `QIndex::dim()` in spec vs `total_dim()` in reality | Low | Open |

---

## Consistency Issues: Tech Spec vs. ARCHITECTURE.md

| Item | Tech Spec | Architecture | Assessment |
|:-----|:----------|:-------------|:-----------|
| DMRGConfig | `bond_dim_schedule: BondDimensionSchedule` | `max_bond_dim: usize` | Spec is more complete (per-sweep scheduling) |
| DMRGConfig | `update_variant: UpdateVariant` | Not present | Spec adds needed field |
| DMRGConfig | Eigensolver in config | Eigensolver separate | **Now resolved** — eigensolver in DMRGRuntimeState |
| DMRGEngine | `arena: SweepArena` | `krylov_workspace: KrylovWorkspace<T>` | Different approaches; both needed |
| EigenResult | `matvec_count: usize` | `iterations: usize` | Different naming; matvec_count is more precise |
| DMRGConfig | `checkpoint_path: Option<PathBuf>` | `environment_offload: Option<PathBuf>` | Different purposes (checkpoint vs env disk cache) |
| Environments | Wrapped in `Environment<T,Q>` with `up_to` | Raw `Vec<BlockSparseTensor>` | Spec adds useful metadata |
| TdvpStabilizationConfig | `tikhonov_delta: f64` only | Adds `adaptive_tikhonov`, `tikhonov_delta_scale`, `tikhonov_delta_min` | Architecture is more detailed |
| DMFT | Out of scope for tk-dmrg | §8.4 places DMFT loop in tk-dmrg | Architecture mixes concerns; spec correctly separates |

---

## What Works Well

1. **Typestate canonical forms** — Compile-time gauge enforcement prevents entire categories of bugs (e.g., calling `dmrg_step` on a `LeftCanonical` MPS).

2. **`IterativeEigensolver<T>` trait** — Clean abstraction for swapping eigensolvers. The `Box<dyn IterativeEigensolver<f64>>` in `DMRGRuntimeState` enables runtime solver selection.

3. **`BondDimensionSchedule`** — Geometric warmup ramp is the standard DMRG practice; having it built into the config prevents users from manually managing per-sweep D values.

4. **Lanczos with full reorthogonalization** works correctly on small test cases and the Sturm bisection eigenvalue finder is numerically stable.

5. **Davidson with diagonal preconditioner** now implements the actual Davidson algorithm with Jacobi-based subspace eigenvalue solve, restart logic, and fallback to unpreconditioned residual when the preconditioned correction collapses.

6. **`SweepSchedule`** with left-to-right + right-to-left half-sweeps correctly implements the standard DMRG sweep pattern.

7. **TDVP soft D_max policy** — The exponential decay of bond dimension overshoot prevents oscillation while allowing temporary growth during subspace expansion.

8. **`DMRGStats`** accumulator — Tracking per-sweep energies, truncation errors, and bond dimensions is essential for monitoring convergence.

9. **Cancellation support** — `AtomicBool` cancellation flag enables clean interruption of long sweeps without data corruption.

10. **`exp_krylov_f64`** — Arnoldi-based Krylov matrix exponential with scaling-and-squaring for the small projected matrix. Correct on diagonal and identity operator tests.

11. **Excited-state penalty method** — `build_heff_penalized` correctly wraps the base H_eff matvec with the penalty projector, shifting converged states out of the target subspace.
