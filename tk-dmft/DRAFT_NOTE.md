# tk-dmft Draft Implementation Notes

**Status:** Draft implementation — compiles, 39 tests pass, all module scaffolding functional. Core algorithms (Lanczos discretization, linear prediction, Chebyshev reconstruction, positivity restoration) are implemented and tested. DMFT self-consistency loop is skeleton (requires functional DMRGEngine from tk-dmrg).
**Date:** March 2026

---

## What Was Implemented

### Impurity Module (`src/impurity/`)

- **mod.rs** — `AndersonImpurityModel<T>` struct with bath management, `new()`, `discretize()`, `update_bath()`, `n_sites()`
- **bath.rs** — `BathParameters<T>` with `uniform()` initial guess, `hybridization_function()` (returns `Vec<Complex<f64>>`), `hybridization_distance()` for convergence, `linear_mix()` for DMFT mixing
- **discretize.rs** — `BathDiscretizationConfig`, full `lanczos_tridiagonalize()` implementing weighted Lanczos recursion to extract (epsilon_k, V_k) from continuous Delta(omega)
- **hamiltonian.rs** — `build_aim_chain_hamiltonian()` constructing `OpSum<T>` for AIM in chain geometry. Spin-resolved terms using `FermionOp` variants and `CustomOp` for n_up*n_down double occupancy

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
- **stats.rs** — `DmrgIterationSummary` (own type), `DMFTStats` accumulator

### Other

- **error.rs** — `DmftError` with 11 variants, `DmftResult<T>` alias
- **mpi/mod.rs** — Stub for MPI node-budget detection
- **build.rs** — Feature conflict detection

**Not implemented (stubs or skeletons):**
- `DMFTLoop::solve()` — requires functional DMRGEngine sweep + TdvpDriver time evolution
- `compute_greens_function_tdvp()` — requires functional TdvpDriver
- `chebyshev_expand()` — requires `DMRGEngine::apply_hamiltonian_to_mps`
- `weiss_field()` — Bethe lattice Weiss field via Kramers-Kronig / Hilbert transform
- Broyden mixing — falls back to linear mixing in draft
- Full Delta_discretized vs Delta_target validation in Lanczos

---

## Design Issues Discovered During Implementation

### 1. `FermionOp` variant names differ from spec — caused extensive rewriting

The tech spec uses abstract operator names (`CDag`, `C`, `N`, `NPairInteraction`) that don't match the actual `tk-dsl` implementation. The actual variants are spin-resolved: `CdagUp`, `CUp`, `CdagDn`, `CDn`, `Nup`, `Ndn`, `Ntotal`. There is no `NPairInteraction`.

**Impact:** The entire Hamiltonian construction had to be rewritten from the spec's design. The double-occupancy term `U * n_up * n_down` required a `CustomOp` workaround since no built-in operator exists for this.

**Workaround:** Created `nup_ndn_operator<T>()` building a 4x4 diagonal matrix `diag(0,0,0,1)` as a `CustomOp`.

**Recommendation:** Either:
- Add `NPairInteraction` (or `NupNdn`) to the `FermionOp` enum in tk-dsl (most useful for physics applications)
- Update the spec to use the actual spin-resolved variant names
- Document the `CustomOp` escape hatch pattern more prominently in the spec

**Severity:** High — caused the most rewriting of any single issue.

### 2. `ScaledOpProduct` has no convenience constructors — spec assumes they exist

The spec assumes `ScaledOpProduct::single(coeff, op, site)` and `ScaledOpProduct::two_site(coeff, op1, site1, op2, site2)` constructors. These don't exist. The only way to construct `ScaledOpProduct` is via struct literal or the `*` operator overloading — which only works for `f64`, not generic `T`.

**Workaround:** Created local helper functions `single_site_term()` and `two_site_term()` that directly construct `ScaledOpProduct` via struct literal with `SmallVec::from_elem()`.

**Recommendation:** Add these constructors to `ScaledOpProduct` in tk-dsl. They are universally needed by any downstream crate building Hamiltonians generically.

**Severity:** Medium — easy to work around but every consumer will hit this.

### 3. Operator overloading limited to `f64` — blocks generic Hamiltonian construction

The `impl Mul<FermionOp> for f64 -> ScaledOpProduct<f64>` overloading doesn't work for generic `T: Scalar`. This means the natural syntax `u_val * FermionOp::Ntotal` can't be used in generic code.

**Workaround:** Direct `ScaledOpProduct` struct literal construction.

**Recommendation:** Add blanket `impl<T: Scalar> Mul<FermionOp> for T -> ScaledOpProduct<T>`. Or at minimum add `impl Mul<FermionOp> for Complex<f64>`.

**Severity:** Medium — ergonomic friction for all generic consumers.

### 4. `DMRGConfig`, `DMRGStats`, `TdvpStabilizationConfig` lack `Clone`/`Debug`

These tk-dmrg types don't derive `Clone` or `Debug`, which blocks `#[derive(Clone, Debug)]` on any type that contains them. `DMFTConfig` and `TimeEvolutionConfig` had to drop these derives entirely.

**Impact:** Can't clone DMFT configs for parameter sweeps. Can't `{:?}` debug-print configs. Had to create own `DmrgIterationSummary` type instead of reusing `DMRGStats`.

**Workaround:**
- Removed derives from `DMFTConfig` and `TimeEvolutionConfig`
- Created `DmrgIterationSummary` as a thin owned type replacing `DMRGStats`

**Recommendation:** Add `#[derive(Clone, Debug)]` to `DMRGConfig`, `DMRGStats`, `TdvpStabilizationConfig` in tk-dmrg. The `Box<dyn IterativeEigensolver<f64>>` field in `DMRGConfig` blocks `Clone` — consider wrapping it in `Arc` or providing a `clone_config_without_solver()` method.

**Severity:** Medium — affects config management ergonomics throughout the DMFT loop.

### 5. tk-dmrg doesn't re-export its dependencies

Attempting `use tk_dmrg::tk_core::...` fails. Each upstream crate (`tk-core`, `tk-symmetry`, `tk-linalg`, `tk-contract`, `tk-dsl`) must be added as a direct path dependency in Cargo.toml.

**Impact:** The initial Cargo.toml only had `tk-dmrg` as a dependency, expecting transitive access. Had to add 5 more direct dependencies.

**Recommendation:** Either re-export key types from tk-dmrg (e.g., `pub use tk_core::Scalar;`) or document the direct-dependency requirement in ARCHITECTURE.md.

**Severity:** Low — one-time fix, but surprising for new crate authors.

### 6. `OpSum::n_terms()` not `len()` — naming inconsistency

The spec and standard Rust naming conventions would suggest `OpSum::len()`. The actual method is `n_terms()`. This is a minor naming discrepancy but caused a compilation error.

**Severity:** Low.

### 7. `CustomOp` has no `new()` constructor or `matrix_data()` accessor

The spec references `CustomOp::new(name, matrix)` and `op.matrix_data()`. Neither exists. You must construct via struct literal (`CustomOp { matrix, name }`) and access via `op.matrix.as_slice()`.

**Recommendation:** Add `CustomOp::new(name: &str, matrix: DenseTensor<T>) -> Self` for ergonomics.

**Severity:** Low.

### 8. `Scalar` trait lacks imaginary unit constructor

For computing complex Green's functions (Kramers-Kronig, hybridization function), you need `i * eta` terms. The `Scalar` trait has no `i()` or `from_imaginary()` method. You must use `num_complex::Complex::new(0.0, eta)` directly, breaking genericity.

**Impact:** `BathParameters::hybridization_function()` returns `Vec<Complex<f64>>` instead of `Vec<T>`, because constructing `T = Complex<f64>` generically requires the imaginary unit.

**Inherited from tk-dsl DRAFT_NOTE** — this was already identified there. Still unresolved.

**Severity:** Medium — will affect all complex-frequency Green's function code.

### 9. Positivity restoration rescaling needs care with mostly-negative spectra

The initial `restore_positivity()` implementation unconditionally rescaled clamped values to preserve the sum rule. This fails when the input spectrum is mostly negative (the clamped sum is much larger than the original, leading to very different rescaling on second application, breaking idempotency).

**Fix applied:** Only rescale when both `original_sum` and `clamped_sum` are positive. This preserves the sum rule for physical spectra while gracefully handling pathological inputs.

**Lesson:** The spec says "L1 renormalize" but doesn't specify the edge case where clamping dramatically changes the total weight. The conditional rescaling should be documented in the spec.

**Severity:** Low — edge case for unphysical inputs, but the test caught it.

---

## Gaps Between Spec and Reality

| Spec Section | Issue | Severity |
|:-------------|:------|:---------|
| §3.2 (AIM Hamiltonian) | `FermionOp::NPairInteraction` doesn't exist; need `CustomOp` for n_up*n_down | High |
| §3.2 (AIM Hamiltonian) | `ScaledOpProduct::single()`, `::two_site()` don't exist | Medium |
| §3.2 (AIM Hamiltonian) | Operator names `CDag`/`C`/`N` should be `CdagUp`/`CUp`/`Ntotal` etc. | Medium |
| §6 (Linear Prediction) | Algorithm works as specified; no gaps | — |
| §7 (Chebyshev) | `DMRGEngine::apply_hamiltonian_to_mps` doesn't exist yet; `chebyshev_expand` is stub | Expected |
| §8 (DMFT Loop) | Loop body is skeleton — depends on DMRGEngine + TdvpDriver being functional | Expected |
| §8.4.2 (Positivity) | Spec doesn't cover edge case of rescaling with mostly-negative input | Low |
| §10 (Errors) | Error enum implemented as specified; `InvalidHybridizationFunction` added | — |
| §16 (Testing) | All specified test categories have at least basic coverage | — |
| General | `OpSum::len()` in spec vs `n_terms()` in reality | Low |
| General | `CustomOp::new()` in spec vs struct literal in reality | Low |

---

## Consistency Issues: Tech Spec vs. ARCHITECTURE.md

| Item | Tech Spec | Architecture | Assessment |
|:-----|:----------|:-------------|:-----------|
| Operator names | `CDag`, `C`, `N`, `NPairInteraction` | Same abstractions | Both wrong — actual is spin-resolved |
| Bath discretization | Lanczos tridiagonalization (§4) | "Lanczos discretization" | Consistent |
| Spectral extraction | Two methods: TDVP+LP and Chebyshev (§5-7) | "Adaptive solver selection" | Consistent |
| DMFT loop | `DMFTLoop<T, Q, B>` (§8) | "Self-consistency driver" | Consistent |
| Mixing | Linear + Broyden (§8.3) | "Linear/Broyden" | Consistent |
| Cross-validation | Primary vs. cross spectral comparison (§8.2) | Mentioned | Consistent |
| Checkpointing | JSON-serialized `DMFTCheckpoint` (§8.5) | "Atomic checkpoint write" | Consistent |
| MPI | Stub only; spec §9 | "Multi-node MPI support (Phase 5+)" | Both agree it's future work |

---

## What Works Well

1. **Lanczos bath discretization** — The weighted Lanczos recursion correctly extracts tridiagonal parameters from a spectral weight function. Tested with semicircular density of states.

2. **Levinson-Durbin solver** — Efficient O(p^2) Toeplitz solver for the linear prediction coefficients. Correctly handles the trivial and single-coefficient cases.

3. **Jackson kernel damping** — Clean implementation of the Chebyshev reconstruction kernel. Monotone decreasing property verified by test.

4. **Spectral positivity restoration** — The 4-step algorithm with diagnostic logging faithfully implements the spec. Idempotency and sum-rule preservation are tested.

5. **Hybridization distance metric** — L-infinity relative distance on Delta(omega) is the right convergence metric for DMFT. The `hybridization_distance()` method is well-typed and efficient.

6. **Checkpoint roundtrip** — Atomic write (write-tmp-then-rename) prevents corrupt checkpoints. JSON format is human-inspectable.

7. **DMFTLoop type structure** — `DMFTLoop<T, Q, B>` correctly propagates all three type parameters and the `PhantomData<Q>` pattern avoids unused-type-parameter errors.

8. **Error types** — The `DmftError` enum with `#[from] DmrgError` correctly chains error propagation from the DMRG layer.

---

## Recommendations for Spec Improvement

1. **Use actual API names in code examples.** The spec's abstract operator names (`CDag`, `C`, `NPairInteraction`, `ScaledOpProduct::single()`) caused significant rewriting. Pin code examples to the real API or add a "spec name → actual name" mapping table.

2. **Document the `CustomOp` pattern for n_up*n_dn.** This is the most common workaround needed by physics consumers. Show the 4x4 matrix construction explicitly.

3. **Specify edge cases in algorithms.** The positivity restoration rescaling, Lanczos early breakdown, and zero-hybridization degenerate cases should be called out explicitly.

4. **Add a "Required Derives" section.** Document which upstream types lack `Clone`/`Debug` and what workarounds downstream crates should use.

5. **Clarify dependency strategy.** State explicitly whether downstream crates should depend on tk-dmrg alone (with re-exports) or on all upstream crates directly.
