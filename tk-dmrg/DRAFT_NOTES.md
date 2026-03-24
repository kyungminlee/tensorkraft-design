# tk-dmrg Draft Implementation Notes

**Status:** Draft implementation — compiles, 26 tests pass, core type scaffolding functional with working Lanczos and Davidson eigensolvers. Sweep engine, environments, TDVP, and MPO compilation are skeletons.
**Date:** March 2026

---

## What is implemented

### Core types
- **mps/mod.rs** — `MPS<T, Q, Gauge>` with typestate markers (`LeftCanonical`, `RightCanonical`, `MixedCanonical`, `BondCentered`), gauge transitions, canonicalization stubs
- **mpo/mod.rs** — `MPO<T, Q>` with accessors, `MpoCompiler` skeleton, `MpoCompressionConfig`
- **environments/mod.rs** — `Environment<T, Q>`, `Environments<T, Q>` with boundary construction, `build_heff_two_site`/`build_heff_single_site` stubs
- **truncation/mod.rs** — `TruncationConfig`, `TruncationResult<T>`, `BondDimensionSchedule` with warmup/fixed/custom

### Eigensolvers (functional)
- **eigensolver/mod.rs** — `IterativeEigensolver<T>` trait
- **`LanczosSolver`** — working Lanczos with full reorthogonalization + Sturm bisection for tridiagonal eigenvalue finding
- **`DavidsonSolver`** — actual diagonal preconditioner + Jacobi eigenvalue solver, restart logic, fallback to unpreconditioned residual
- **`BlockDavidsonSolver`** — delegates to Lanczos (block operations not yet implemented)

### TDVP (functional Krylov exp)
- **tdvp/mod.rs** — `TdvpDriver<T, Q, B>`, `TdvpStabilizationConfig`
- **`exp_krylov_f64`** — Arnoldi-based Krylov matrix exponential with scaling-and-squaring, soft D_max logic

### Excited states (functional)
- **excited/mod.rs** — `ExcitedStateConfig`, working `build_heff_penalized` with penalty matvec wrapper using overlap vectors

### Sweep engine
- **sweep/mod.rs** — `DMRGEngine<T, Q, B>`, `DMRGConfig` (immutable) + `DMRGRuntimeState` (mutable), `DMRGStats`, `SweepSchedule`, `UpdateVariant`, cancellation support, arena safety documentation

### Other
- **idmrg/mod.rs** — `IDmrgConfig`, `run_idmrg` stub
- **checkpoint.rs** — `DMRGCheckpoint` skeleton
- **error.rs** — Full `DmrgError` enum with 14 variants

### Cross-crate fixes
- **tk-core/tensor.rs** — Implemented `Clone` for `DenseTensor<'static, T>`
- **tk-symmetry/block_sparse.rs** — Implemented `Clone` for `BlockSparseTensor<T, Q>`

---

## Remaining limitations

### Not implemented (skeletons only)

| Component | Description |
|:----------|:------------|
| `MpoCompiler::compile` | OpSum → MPO FSA + SVD compilation |
| Environment contraction | `grow_left`/`grow_right` |
| H_eff matvec closures | `build_heff_two_site`/`build_heff_single_site` |
| DMRG step logic | eigensolve → SVD truncate → update MPS → update env |
| TDVP bond evolution | subspace expansion (Krylov exp is implemented) |
| Block-Davidson | block operations |
| iDMRG | unit-cell extension loop |
| Checkpoint serialization | requires serde on `BlockSparseTensor` |

### Design issues

1. **`BitPackable` does not require `Copy`** — All quantum number types (U1, Z2) are small Copy types, but `BitPackable` only transitively requires `Clone` via `QuantumNumber`. Adding `Copy` as a supertrait would eliminate verbose `.clone()` calls. **Severity:** Medium (ergonomic).

2. **`IterativeEigensolver<T>` not generic for complex TDVP** — Fixed to `Box<dyn IterativeEigensolver<f64>>` in `DMRGRuntimeState`. Complex-valued TDVP would need a separate eigensolver field. **Severity:** Low for now.

3. **Lanczos tridiagonal solve is naive** — Uses Sturm bisection (O(n²)) instead of LAPACK `dstev` (O(n)). Fine for small Krylov dimensions (20-100). **Severity:** Low for draft.

4. **`DMRGEngine` owns backend** — Limits sharing across multiple engines. Consider `&B` or `Arc<B>`. **Severity:** Low.

5. **No `serde` integration** — Checkpoint requires serializing `BlockSparseTensor<T, Q>`. Decision needed: `#[derive(Serialize, Deserialize)]` on types in tk-symmetry vs proxy types. **Severity:** Medium (needed for crash recovery).

6. **Naming discrepancies with spec** — `LegDirection::Incoming/Outgoing` (not `In/Out`), `QIndex::total_dim()` (not `dim()`). **Severity:** Low.

---

## Consistency issues: tech spec vs. architecture

| Item | Tech Spec | Architecture | Assessment |
|:-----|:----------|:-------------|:-----------|
| DMRGConfig | `bond_dim_schedule: BondDimensionSchedule` | `max_bond_dim: usize` | Spec is more complete |
| DMRGConfig | Eigensolver in config | Eigensolver separate | Resolved — eigensolver in DMRGRuntimeState |
| DMRGEngine | `arena: SweepArena` | `krylov_workspace: KrylovWorkspace<T>` | Different approaches; both needed |
| EigenResult | `matvec_count: usize` | `iterations: usize` | matvec_count is more precise |
| TdvpStabilization | `tikhonov_delta: f64` only | Adds `adaptive_tikhonov`, `tikhonov_delta_scale`, `tikhonov_delta_min` | Architecture is more detailed |
| DMFT | Out of scope for tk-dmrg | §8.4 places DMFT loop in tk-dmrg | Architecture mixes concerns; spec correctly separates |

---

## What works well

1. **Typestate canonical forms** — compile-time gauge enforcement prevents bugs
2. **`IterativeEigensolver<T>` trait** — clean abstraction for swapping eigensolvers
3. **`BondDimensionSchedule`** — geometric warmup ramp is standard DMRG practice
4. **Lanczos with full reorthogonalization** — works correctly on small test cases
5. **Davidson with diagonal preconditioner** — actual Davidson algorithm with restart and fallback
6. **`SweepSchedule`** — correct left-to-right + right-to-left half-sweep pattern
7. **TDVP soft D_max policy** — exponential decay of overshoot prevents oscillation
8. **`DMRGStats` accumulator** — tracks per-sweep energies, truncation errors, bond dimensions
9. **Cancellation support** — `AtomicBool` flag for clean sweep interruption
10. **`exp_krylov_f64`** — Arnoldi + scaling-and-squaring, correct on diagonal and identity tests
11. **Excited-state penalty method** — correctly wraps base H_eff matvec with penalty projector
