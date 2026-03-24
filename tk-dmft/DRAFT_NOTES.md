# tk-dmft Draft Implementation Notes

**Status:** Draft implementation — compiles, 39 tests pass. Core algorithms (Lanczos discretization, linear prediction, Chebyshev reconstruction, positivity restoration) are implemented and tested. DMFT self-consistency loop is skeleton (requires functional DMRGEngine from tk-dmrg).
**Date:** March 2026

---

## What is implemented

### Impurity Module (`src/impurity/`)

- **mod.rs** — `AndersonImpurityModel<T>` with bath management, `new()`, `discretize()`, `update_bath()`, `n_sites()`
- **bath.rs** — `BathParameters<T>` with `uniform()` initial guess, `hybridization_function()`, `hybridization_distance()` for convergence, `linear_mix()` for DMFT mixing
- **discretize.rs** — `BathDiscretizationConfig`, full `lanczos_tridiagonalize()` implementing weighted Lanczos recursion to extract (epsilon_k, V_k) from continuous Delta(omega)
- **hamiltonian.rs** — `build_aim_chain_hamiltonian()` constructing `OpSum<T>` for AIM in chain geometry. Uses `CustomOp` for n_up*n_down double occupancy

### Spectral Module (`src/spectral/`)

- **mod.rs** — `SpectralFunction` struct (omega grid + values), `SpectralSolverMode` enum (TdvpPrimary, ChebyshevPrimary, Adaptive)
- **linear_predict.rs** — `LinearPredictionConfig`, `ToeplitzSolver` enum, `solve_toeplitz_levinson_durbin()`, `linear_predict_regularized()`, `fft_to_spectral()`, `deconvolve_lorentzian()`
- **positivity.rs** — `restore_positivity()` with 4-step algorithm: negative weight diagnostic, Fermi-level recording, clamp + L1 renormalize, Fermi-level distortion check
- **chebyshev.rs** — `ChebyshevConfig`, `jackson_kernel()`, `reconstruct_from_moments()`
- **tdvp.rs** — `TdvpSpectralConfig`, `compute_greens_function_tdvp()` stub

### Loop Module (`src/loop/`)

- **mod.rs** — `DMFTLoop<T, Q, B>` driver with `solve()` (skeleton), `solve_with_cancel_flag()`, `weiss_field()` (stub), `apply_mixing()`, `validate_consistency()`, `DMFTCheckpoint` with atomic JSON write/read
- **config.rs** — `TimeEvolutionConfig`, `DMFTConfig`
- **mixing.rs** — `MixingScheme` (Linear, Broyden)
- **stats.rs** — `DmrgIterationSummary`, `DMFTStats` accumulator

### Other

- **error.rs** — `DmftError` with 11 variants, `DmftResult<T>` alias
- **mpi/mod.rs** — Stub for MPI node-budget detection
- **build.rs** — Feature conflict detection

---

## Remaining limitations

### Not implemented (stubs or skeletons)

| Component | Blocked on |
|:----------|:-----------|
| `DMFTLoop::solve()` | Functional DMRGEngine sweep + TdvpDriver time evolution |
| `compute_greens_function_tdvp()` | Functional TdvpDriver |
| `chebyshev_expand()` | `DMRGEngine::apply_hamiltonian_to_mps` |
| `weiss_field()` | Bethe lattice Weiss field via Kramers-Kronig / Hilbert transform |
| Broyden mixing | Falls back to linear mixing in draft |
| Full Delta validation | Delta_discretized vs Delta_target in Lanczos |

### Design issues

1. **`FermionOp` variant names differ from spec** — Spec uses `CDag`/`C`/`N`/`NPairInteraction`; actual is spin-resolved `CdagUp`/`CUp`/`Nup`/`Ndn`/`Ntotal`. No `NPairInteraction` exists. Double-occupancy `U * n_up * n_down` requires `CustomOp`. **Severity:** High — caused most rewriting.

2. **`ScaledOpProduct` lacks convenience constructors** — Spec assumes `::single()` and `::two_site()`. Local helper functions used as workaround. **Severity:** Medium.

3. **Operator overloading limited to `f64`** — `coeff * op(Sz, 0)` syntax doesn't work for generic `T: Scalar`. Use `ScaledOpProduct` struct literal instead. **Severity:** Medium.

4. **`DMRGConfig`/`DMRGStats`/`TdvpStabilizationConfig` lack `Clone`/`Debug`** — Blocks derives on containing types. Created `DmrgIterationSummary` as a workaround. **Severity:** Medium.

5. **tk-dmrg doesn't re-export dependencies** — Each upstream crate must be added as a direct dependency. **Severity:** Low.

6. **`Scalar` trait lacks imaginary unit constructor** — `hybridization_function()` returns `Vec<Complex<f64>>` instead of `Vec<T>`. **Severity:** Medium.

7. **Positivity restoration edge case** — Only rescale when both `original_sum` and `clamped_sum` are positive, to handle mostly-negative spectra gracefully. **Severity:** Low (fixed).

### Spec-vs-reality gaps

| Spec Section | Issue | Severity |
|:-------------|:------|:---------|
| §3.2 (AIM Hamiltonian) | `FermionOp::NPairInteraction` doesn't exist; need `CustomOp` | High |
| §3.2 (AIM Hamiltonian) | `ScaledOpProduct::single()`/`::two_site()` don't exist | Medium |
| §7 (Chebyshev) | `DMRGEngine::apply_hamiltonian_to_mps` doesn't exist yet | Expected |
| §8 (DMFT Loop) | Loop body is skeleton — depends on DMRGEngine + TdvpDriver | Expected |
| General | `OpSum::len()` in spec vs `n_terms()` in reality | Low |
| General | `CustomOp::new()` in spec vs struct literal in reality | Low |

---

## What works well

1. **Lanczos bath discretization** — weighted Lanczos recursion correctly extracts tridiagonal parameters. Tested with semicircular DOS.
2. **Levinson-Durbin solver** — efficient O(p²) Toeplitz solver for linear prediction coefficients.
3. **Jackson kernel damping** — monotone decreasing property verified by test.
4. **Spectral positivity restoration** — 4-step algorithm with idempotency and sum-rule preservation tested.
5. **Hybridization distance metric** — L-infinity relative distance on Delta(omega).
6. **Checkpoint roundtrip** — atomic write (write-tmp-then-rename) prevents corrupt checkpoints.
7. **`DMFTLoop<T, Q, B>` type structure** — correctly propagates all type parameters.
8. **Error types** — `DmftError` with `#[from] DmrgError` correctly chains error propagation.

---

## Recommendations for spec improvement

1. Use actual API names in code examples (pin to real API or add mapping table).
2. Document the `CustomOp` pattern for n_up*n_dn explicitly.
3. Specify edge cases in algorithms (positivity rescaling, Lanczos early breakdown, zero-hybridization).
4. Add a "Required Derives" section for upstream types lacking `Clone`/`Debug`.
5. Clarify dependency strategy (re-exports vs direct dependencies).
