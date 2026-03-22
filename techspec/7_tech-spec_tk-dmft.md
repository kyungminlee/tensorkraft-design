# Technical Specification: `tk-dmft`

**Crate:** `tensorkraft/crates/tk-dmft`
**Version:** 0.1.0 (Pre-Implementation)
**Status:** Specification
**Last Updated:** March 2026

---

## 1. Overview

`tk-dmft` is the Dynamical Mean-Field Theory (DMFT) solver crate for the tensorkraft workspace. It sits at the top of the dependency chain, consuming `tk-dmrg` (and transitively `tk-core`, `tk-symmetry`, `tk-linalg`, `tk-contract`, and `tk-dsl`). It is consumed directly only by `tk-python`.

**Core responsibilities:**

- **Anderson Impurity Model (AIM) mapping** — Represent the AIM in star geometry (impurity + bath sites) and compile it into a Hamiltonian via `tk-dsl`'s `hamiltonian!{}` macro. Maintain bath parameters (on-site energies ε_k and hybridizations V_k) as the mutable state updated at each DMFT iteration.
- **Bath discretization (Lanczos tridiagonalization)** — Map the continuous hybridization function Δ(ω) from the DMFT self-consistency equation onto a finite set of bath parameters via Lanczos tridiagonalization. This converts the star-geometry AIM into a 1D impurity chain suitable for DMRG.
- **DMRG ground state** — Delegate to `tk-dmrg`'s `DMRGEngine` to compute the ground state of the discretized AIM chain Hamiltonian.
- **Real-time Green's function via TDVP** — Use `tk-dmrg`'s `TdvpDriver` to compute G(t) = ⟨ψ₀|c(t) c†|ψ₀⟩ via real-time MPS evolution. Includes Tikhonov regularization, site-tensor subspace expansion, and soft D_max policy (all provided by `TdvpDriver`).
- **Chebyshev spectral function** — Compute A(ω) directly in frequency space via Chebyshev expansion of the Green's function, bypassing time-domain linear prediction instabilities.
- **Adaptive TDVP/Chebyshev solver selection** — Inspect the entanglement gap of the DMRG ground state to determine the primary spectral solver at runtime (design doc §8.4.1).
- **Linear prediction with Levinson-Durbin** — Extrapolate G(t) to larger times via regularized linear prediction (Levinson-Durbin recursion on the Toeplitz autocorrelation system), followed by FFT and regularized Lorentzian deconvolution.
- **Spectral positivity restoration** — Mandatory post-deconvolution pass: clamp negative spectral weight, L₁-renormalize to preserve the sum rule, and emit diagnostic warnings for Fermi-level distortion and large negative-weight fractions.
- **DMFT self-consistency loop** — Iterate bath parameters until the bath hybridization function converges, using linear or Broyden mixing schemes for stability.
- **MPI Mode B initialization** — At startup, query the MPI topology and divide the node's available pinned-memory budget evenly across co-resident ranks via `initialize_dmft_node_budget`.

**Key design principle:** `tk-dmft` is the integration point for quantum embedding. All tensor operations, linear algebra, and DMRG/TDVP algorithms are delegated to `tk-dmrg` and its dependencies. What `tk-dmft` owns is the physics layer: the DMFT feedback loop, bath parametrization, spectral function extraction, and the adaptive logic that selects between time-domain and frequency-domain solvers based on entanglement physics.

---

## 2. Module Structure

```
tk-dmft/
├── Cargo.toml
├── build.rs                    (feature conflict detection, MPI library detection)
└── src/
    ├── lib.rs                  re-exports all public items
    ├── impurity/
    │   ├── mod.rs              AndersonImpurityModel<T>: bath parameters, star geometry
    │   ├── bath.rs             BathParameters<T>: ε_k, V_k arrays and mixing update
    │   ├── discretize.rs       Lanczos tridiagonalization: Δ(ω) → bath parameters
    │   └── hamiltonian.rs      AIM Hamiltonian construction via tk-dsl OpSum + MPO compilation
    ├── spectral/
    │   ├── mod.rs              SpectralFunction, SpectralSolverMode, types
    │   ├── tdvp.rs             TDVP-based G(t) computation and linear prediction pipeline
    │   ├── chebyshev.rs        Chebyshev expansion of A(ω)
    │   ├── linear_predict.rs   Levinson-Durbin recursion, exponential windowing, deconvolution
    │   └── positivity.rs       restore_positivity, Fermi-level distortion diagnostic
    ├── loop/
    │   ├── mod.rs              DMFTLoop<T, Q, B>: self-consistency driver
    │   ├── mixing.rs           MixingScheme (linear, Broyden), convergence check
    │   ├── config.rs           DMFTConfig, TimeEvolutionConfig
    │   └── stats.rs            DMFTStats: per-iteration energies, spectral moments, wall times
    ├── mpi/
    │   └── mod.rs              initialize_dmft_node_budget (feature-gated)
    └── error.rs                DmftError, DmftResult<T>
```

---

## 3. Anderson Impurity Model

### 3.1 `BathParameters<T>`

```rust
/// Discretized bath parameters for the Anderson Impurity Model.
///
/// Represents a finite set of `n_bath` non-interacting bath orbitals
/// coupled to the impurity. In the star geometry, each bath site `k`
/// has an on-site energy `epsilon[k]` and hybridization amplitude `v[k]`.
///
/// The non-interacting hybridization function is:
///   Δ(ω) = Σ_k |V_k|² / (ω - ε_k + i0⁺)
///
/// Bath parameters are the mutable state updated at each DMFT iteration.
#[derive(Clone, Debug)]
pub struct BathParameters<T: Scalar> {
    /// On-site bath energies ε_k. Length = `n_bath`.
    pub epsilon: Vec<T::Real>,
    /// Hybridization amplitudes V_k (coupling impurity to bath site k).
    /// Length = `n_bath`.
    pub v: Vec<T>,
    /// Number of bath sites.
    pub n_bath: usize,
}

impl<T: Scalar> BathParameters<T> {
    /// Construct uniform bath: `n_bath` sites with energies linearly spaced
    /// in `[-bandwidth/2, bandwidth/2]` and uniform hybridization `v0`.
    ///
    /// Used as an initial guess before the first DMFT iteration.
    pub fn uniform(n_bath: usize, bandwidth: T::Real, v0: T) -> Self;

    /// Compute the discretized hybridization function at frequency grid `omega`.
    ///
    /// Δ(ω) = Σ_k |V_k|² / (ω - ε_k + i·broadening)
    ///
    /// # Parameters
    /// - `omega`: frequency grid points
    /// - `broadening`: Lorentzian broadening δ replacing the i0⁺ regulator
    pub fn hybridization_function(
        &self,
        omega: &[T::Real],
        broadening: T::Real,
    ) -> Vec<T>;

    /// Compute the relative L∞ distance ‖Δ_self(ω) - Δ_other(ω)‖_∞ / ‖Δ_other(ω)‖_∞
    /// for convergence assessment of the DMFT self-consistency loop.
    ///
    /// # Panics
    /// Panics if `omega.len()` is zero.
    pub fn hybridization_distance(
        &self,
        other: &Self,
        omega: &[T::Real],
        broadening: T::Real,
    ) -> T::Real;
}
```

### 3.2 `AndersonImpurityModel<T>`

```rust
/// Full Anderson Impurity Model: impurity + discretized bath.
///
/// Holds the current bath parameters and the model's physical parameters
/// (interaction U, impurity level ε_imp, inverse temperature β). Provides
/// methods to discretize the bath and to update parameters after a DMFT
/// iteration.
///
/// # Type Parameters
/// - `T`: scalar type (`f64` for real calculations, `Complex<f64>` for
///         complex hybridization functions)
#[derive(Clone, Debug)]
pub struct AndersonImpurityModel<T: Scalar> {
    /// On-site Coulomb interaction at the impurity.
    pub u: T::Real,
    /// Impurity site energy (relative to the chemical potential).
    pub epsilon_imp: T::Real,
    /// Inverse temperature. `None` for T=0 calculations.
    pub beta: Option<T::Real>,
    /// Current discretized bath parameters.
    pub bath: BathParameters<T>,
    /// Physical dimension of the impurity site (2 for single-orbital spin-1/2).
    pub impurity_local_dim: usize,
    /// Physical dimension of each bath site.
    pub bath_local_dim: usize,
}

impl<T: Scalar> AndersonImpurityModel<T>
where
    T::Real: Into<f64> + From<f64>,
{
    /// Construct a new AIM with `n_bath` bath sites initialized uniformly.
    ///
    /// # Parameters
    /// - `u`: Hubbard interaction
    /// - `epsilon_imp`: impurity level energy
    /// - `n_bath`: number of bath sites
    /// - `bandwidth`: initial bath bandwidth for uniform initialization
    /// - `v0`: initial uniform hybridization
    pub fn new(
        u: T::Real,
        epsilon_imp: T::Real,
        n_bath: usize,
        bandwidth: T::Real,
        v0: T,
    ) -> Self;

    /// Perform bath discretization: project the target hybridization function
    /// `delta_target` onto `n_bath` discrete bath parameters via Lanczos
    /// tridiagonalization.
    ///
    /// See §4 (bath discretization algorithm) for the full procedure.
    ///
    /// # Parameters
    /// - `delta_target`: target hybridization Δ(ω) sampled on `omega`
    /// - `omega`: real-frequency grid (uniform spacing assumed)
    /// - `config`: discretization configuration
    ///
    /// # Returns
    /// Updated `BathParameters`. Does NOT mutate `self`; caller decides
    /// whether to commit via `update_bath`.
    ///
    /// # Errors
    /// Returns `DmftError::BathDiscretizationFailed` if Lanczos tridiagonalization
    /// does not converge within `config.max_lanczos_steps`.
    pub fn discretize(
        &self,
        delta_target: &[T],
        omega: &[T::Real],
        config: &BathDiscretizationConfig,
    ) -> DmftResult<BathParameters<T>>;

    /// Update the bath parameters to `new_bath`.
    /// Called after each DMFT iteration's mixing step.
    pub fn update_bath(&mut self, new_bath: BathParameters<T>);

    /// Compile the current AIM into an MPO-ready chain Hamiltonian.
    ///
    /// Constructs the `OpSum` for the AIM in chain geometry via `tk-dsl`:
    ///
    ///   H = ε_imp Σ_σ n_{0σ} + U n_{0↑} n_{0↓}
    ///     + Σ_{k,σ} ε_k n_{kσ}
    ///     + Σ_{k,σ} V_k (c†_{0σ} c_{kσ} + h.c.)
    ///
    /// Site ordering: impurity at site 0, bath sites at 1..=n_bath.
    /// Uses `FermionOp` from `tk-dsl` for all operator terms.
    ///
    /// Delegates MPO compression to `tk-dmrg`'s `MpoCompiler`.
    ///
    /// # Errors
    /// Returns `DmftError::Dmrg` wrapping `DmrgError::OpSumCompilationFailed`
    /// if MPO compression fails.
    pub fn build_chain_hamiltonian<Q: BitPackable, B: LinAlgBackend<T>>(
        &self,
        backend: &B,
        mpo_config: &MpoCompressionConfig,
    ) -> DmftResult<MPO<T, Q>>;

    /// Total number of sites in the chain (1 impurity + n_bath bath sites).
    pub fn n_sites(&self) -> usize {
        1 + self.bath.n_bath
    }
}
```

### 3.3 `BathDiscretizationConfig`

```rust
/// Configuration for Lanczos tridiagonalization bath discretization.
#[derive(Clone, Debug)]
pub struct BathDiscretizationConfig {
    /// Maximum number of Lanczos recursion steps.
    /// Default: 0 (auto: `n_bath * 10`).
    pub max_lanczos_steps: usize,
    /// Convergence threshold on tridiagonalization residual. Default: 1e-12.
    pub lanczos_tol: f64,
    /// Number of frequency grid points for hybridization function evaluation.
    /// Default: 2000.
    pub n_omega_points: usize,
    /// Half-bandwidth of the frequency grid. Default: 10.0.
    pub bandwidth: f64,
    /// Lorentzian broadening for hybridization function evaluation. Default: 0.05.
    pub broadening: f64,
}

impl Default for BathDiscretizationConfig {
    fn default() -> Self {
        BathDiscretizationConfig {
            max_lanczos_steps: 0,
            lanczos_tol: 1e-12,
            n_omega_points: 2000,
            bandwidth: 10.0,
            broadening: 0.05,
        }
    }
}
```

---

## 4. Bath Discretization Algorithm

The Lanczos tridiagonalization procedure converts the continuous hybridization function Δ(ω) into `n_bath` discrete bath parameters (ε_k, V_k).

**Algorithm (`AndersonImpurityModel::discretize`, design doc §2.2 / §8.4):**

1. **Frequency grid:** Construct a real-frequency grid ω_j ∈ [-Ω, Ω] with `n_omega_points` points.
2. **Spectral weight vector:** Treat the imaginary part of Δ(ω) as a discrete measure: w_j = -Im[Δ(ω_j)] × Δω / π (must be non-negative for a valid hybridization function; emits `DmftError::InvalidHybridizationFunction` if any w_j < 0).
3. **Lanczos start vector:** φ₀(j) = 1 for all j (normalized by total weight W = Σ_j w_j). V₁ = √W is the first hybridization amplitude.
4. **Lanczos recursion:** Build a tridiagonal matrix T of dimension `n_bath` × `n_bath`. At step k, the diagonal element α_k = ⟨φ_k | ω | φ_k⟩_w (on-site energy ε_k) and the off-diagonal element β_{k+1} = ‖ω·φ_k - α_k·φ_k - β_k·φ_{k-1}‖_w (related to V_{k+1}). The inner product ⟨f|g⟩_w = Σ_j w_j f_j* g_j.
5. **Bath parameters:** ε_k = α_k, V_k = β_k for k ≥ 1, V₁ = √W. The resulting discretized hybridization satisfies Δ(ω) ≈ V₁²/(ω - ε₁ - V₂²/(ω - ε₂ - ...)) (continued-fraction expansion of the tridiagonal resolvent).
6. **Validation:** Check ‖Δ_discretized - Δ_target‖_∞ / ‖Δ_target‖_∞ < `lanczos_tol`. Return `DmftError::BathDiscretizationFailed { max_steps, residual }` if the residual exceeds tolerance.

**Complexity:** O(n_bath × n_omega_points) — linear in both bath size and frequency grid.

**Chain geometry note:** After discretization, the impurity site is placed at position 0 and the bath sites at positions 1..=n_bath in the DMRG chain. The Lanczos tridiagonalization is precisely the mapping from star geometry (impurity coupled to all bath sites simultaneously) to chain geometry (impurity coupled only to the first bath site of a tridiagonal chain).

---

## 5. Spectral Function Types

### 5.1 `SpectralFunction`

```rust
/// Real-frequency spectral function A(ω) = -Im[G(ω)] / π.
///
/// Defined on a uniform frequency grid `omega`. The spectral sum rule
/// requires ∫ A(ω) dω = 1 for a single-orbital impurity.
///
/// Invariant maintained after `restore_positivity`:
///   A(ω) ≥ 0 for all ω
#[derive(Clone, Debug)]
pub struct SpectralFunction {
    /// Frequency grid points (uniform spacing).
    pub omega: Vec<f64>,
    /// Spectral weight at each grid point. Same length as `omega`.
    pub values: Vec<f64>,
    /// Frequency spacing Δω (cached for integration).
    pub d_omega: f64,
}

impl SpectralFunction {
    /// Construct from frequency grid and spectral values.
    ///
    /// # Panics
    /// Panics if `omega.len() != values.len()` or if `omega` is empty.
    pub fn new(omega: Vec<f64>, values: Vec<f64>) -> Self;

    /// Spectral sum rule: ∫ A(ω) dω via the trapezoidal rule.
    pub fn sum_rule(&self) -> f64;

    /// Value at ω ≈ 0 (Fermi level). Interpolates linearly between the
    /// two grid points bracketing ω = 0.
    ///
    /// # Panics
    /// Panics if `omega` does not span ω = 0.
    pub fn value_at_omega_zero(&self) -> f64;

    /// The nth spectral moment: ∫ ωⁿ A(ω) dω via the trapezoidal rule.
    pub fn moment(&self, n: usize) -> f64;

    /// L∞ distance ‖self - other‖_∞ for convergence checks.
    ///
    /// # Panics
    /// Panics if `self.omega.len() != other.omega.len()`.
    pub fn max_distance(&self, other: &SpectralFunction) -> f64;
}
```

### 5.2 `SpectralSolverMode`

```rust
/// Controls which spectral function engine is designated as primary.
///
/// Both engines (TDVP + linear prediction, and Chebyshev) are always
/// computed at each DMFT iteration. Only their roles (primary vs.
/// cross-validation) change. Cross-validation consistency is checked
/// via `TimeEvolutionConfig::cross_validation_tol` (design doc §8.4.1).
#[derive(Clone, Debug)]
pub enum SpectralSolverMode {
    /// TDVP + linear prediction is primary; Chebyshev is cross-validation.
    /// Appropriate for gapped/insulating phases.
    TdvpPrimary,
    /// Chebyshev expansion is primary; TDVP + linear prediction is cross-validation.
    /// Appropriate for gapless/metallic phases. Selected automatically by
    /// `Adaptive` when the entanglement gap is below `gap_threshold`.
    ChebyshevPrimary,
    /// Automatically select based on the entanglement spectrum gap.
    ///
    /// After the DMRG ground state is computed, the entanglement gap at the
    /// center bond is inspected: gap = σ₁/σ₂ (ratio of the two largest Schmidt
    /// values at the bipartition).
    ///
    /// gap >= gap_threshold → gapped/insulating → `TdvpPrimary`
    /// gap <  gap_threshold → gapless/metallic  → `ChebyshevPrimary`
    ///
    /// Default `gap_threshold`: 0.1 (design doc §8.4.1)
    Adaptive { gap_threshold: f64 },
}

impl Default for SpectralSolverMode {
    fn default() -> Self {
        SpectralSolverMode::Adaptive { gap_threshold: 0.1 }
    }
}
```

---

## 6. Linear Prediction Pipeline

### 6.1 `ToeplitzSolver`

```rust
/// Solver for the P×P Toeplitz prediction system in linear prediction.
///
/// The autocorrelation matrix R (where R_{ij} = autocorr(|i-j|)) has
/// Toeplitz structure. Levinson-Durbin exploits this for O(P²) complexity.
///
/// Performance note (design doc §8.4.2): for typical P ≤ 200 the
/// performance difference between O(P²) and O(P³) is microseconds.
/// Levinson-Durbin is the default because it is the algorithmically correct
/// choice for Toeplitz systems, not for performance reasons.
#[derive(Clone, Debug)]
pub enum ToeplitzSolver {
    /// O(P²) Levinson-Durbin recursion with Tikhonov regularization.
    /// Applied as R → R + λI before the recursion.
    /// Default and recommended for all standard DMFT workflows.
    LevinsonDurbin { tikhonov_lambda: f64 },
    /// O(P³) SVD-based pseudo-inverse.
    /// Retained as fallback for non-Toeplitz extensions (e.g., non-uniform
    /// time grids where the prediction matrix is not exactly Toeplitz).
    SvdPseudoInverse { svd_noise_floor: f64 },
}

impl Default for ToeplitzSolver {
    fn default() -> Self {
        ToeplitzSolver::LevinsonDurbin { tikhonov_lambda: 1e-8 }
    }
}
```

### 6.2 `LinearPredictionConfig`

```rust
/// Configuration for the full linear prediction pipeline.
///
/// Pipeline stages (design doc §8.4.2):
///   1. Exponential windowing: G(t) → G(t) × exp(−η|t|)
///   2. Toeplitz prediction solve (Levinson-Durbin or SVD)
///   3. FFT → A_windowed(ω)
///   4. Regularized Lorentzian deconvolution → A_raw(ω)  [if η > 0]
///   5. Spectral positivity restoration → A(ω)  [always mandatory]
#[derive(Clone, Debug)]
pub struct LinearPredictionConfig {
    /// Solver for the Toeplitz prediction system.
    /// Default: `LevinsonDurbin { tikhonov_lambda: 1e-8 }`.
    pub toeplitz_solver: ToeplitzSolver,

    /// Prediction order P (number of past time points used). Default: 100.
    pub prediction_order: usize,

    /// Factor by which to extend G(t) beyond the TDVP simulation time.
    /// The extrapolated time window is `t_max × extrapolation_factor`.
    /// Default: 4.0.
    pub extrapolation_factor: f64,

    /// Exponential broadening parameter η for windowing G(t).
    /// W(t) = exp(−η|t|). Set to 0.0 to disable windowing (gapped phases).
    /// The Fourier transform of W(t) is a Lorentzian 2η/(η²+ω²) which must
    /// be deconvolved from A_windowed(ω) to recover A_true(ω).
    /// Default: 0.0 (disabled).
    pub broadening_eta: f64,

    /// Tikhonov regularization δ for the Lorentzian deconvolution denominator.
    ///
    /// Deconvolution factor: (η² + ω²) / (2η + δ × ω²)
    ///
    /// Without regularization, the naive factor (η²+ω²)/(2η) grows as ω²/2η,
    /// amplifying noise quadratically at high frequencies. The δ × ω² term in
    /// the denominator bounds amplification to 1/δ regardless of frequency.
    /// Default: 1e-3.
    pub deconv_tikhonov_delta: f64,

    /// Hard cutoff frequency for deconvolution. Beyond ω_max, the deconvolution
    /// factor is clamped to 1.0 (no correction applied). A(ω) is assumed
    /// negligible beyond this cutoff. In units of the AIM bandwidth.
    /// Default: 10.0 × bandwidth.
    pub deconv_omega_max: f64,

    /// Noise floor for spectral positivity clamping.
    /// A(ω) = max(A(ω), positivity_floor) after deconvolution.
    /// Default: 1e-15.
    pub positivity_floor: f64,

    /// Warning threshold: if clamped negative weight W_neg / W_total exceeds
    /// this fraction, emit a `SPECTRAL_POSITIVITY_WARNING` log event at the
    /// `tensorkraft::telemetry` target. Default: 0.05 (5%).
    pub positivity_warning_threshold: f64,

    /// Fermi-level distortion tolerance. If the global L₁ rescaling (for sum
    /// rule preservation) shifts A(ω=0) by more than this relative fraction,
    /// emit a `FERMI_LEVEL_DISTORTION` warning. Default: 0.01 (1%).
    ///
    /// Physical significance (design doc §8.1 revision notes): A(ω=0) determines
    /// the quasiparticle residue Z and the Luttinger pinning condition. A 2–3%
    /// unphysical shift can corrupt transport properties.
    pub fermi_level_shift_tolerance: f64,
}

impl Default for LinearPredictionConfig {
    fn default() -> Self {
        LinearPredictionConfig {
            toeplitz_solver: ToeplitzSolver::default(),
            prediction_order: 100,
            extrapolation_factor: 4.0,
            broadening_eta: 0.0,
            deconv_tikhonov_delta: 1e-3,
            deconv_omega_max: 10.0,
            positivity_floor: 1e-15,
            positivity_warning_threshold: 0.05,
            fermi_level_shift_tolerance: 0.01,
        }
    }
}
```

### 6.3 Linear Prediction Free Functions

```rust
/// Apply exponential windowing and run Toeplitz linear prediction to
/// extrapolate G(t) to `extrapolation_factor × t_max`.
///
/// Stages 1–2 of the linear prediction pipeline:
///   1. Apply W(t_k) = exp(−η|t_k|) to each sample G(t_k).
///   2. Compute autocorrelation R[k] = Σ_j G(t_j)* G(t_{j+k}).
///   3. Solve R·a = r for prediction coefficients a (Levinson-Durbin or SVD).
///   4. Extrapolate: G(t_{N+m}) = Σ_{k=1}^{P} a_k × G(t_{N+m-k}).
///
/// # Parameters
/// - `g_t`: complex Green's function samples at uniform time steps dt
/// - `dt`: physical time step (seconds or inverse energy)
/// - `config`: linear prediction configuration
///
/// # Returns
/// Extended time series with length ≈ `g_t.len() × config.extrapolation_factor`.
///
/// # Errors
/// Returns `DmftError::LinearPredictionFailed` if the Levinson-Durbin
/// recursion diverges (condition number estimate returned for diagnosis).
pub fn linear_predict_regularized(
    g_t: &[Complex<f64>],
    dt: f64,
    config: &LinearPredictionConfig,
) -> DmftResult<Vec<Complex<f64>>>;

/// FFT the extended G(t) to obtain A_windowed(ω) = -Im[G(ω)] / π.
///
/// Uses `rustfft::FftPlanner` for the DFT. The output is interpolated onto
/// the provided uniform `omega` grid via linear interpolation.
///
/// # Parameters
/// - `g_t_extended`: extrapolated Green's function from `linear_predict_regularized`
/// - `dt`: physical time step
/// - `omega`: target frequency grid (must be uniform)
///
/// # Returns
/// `SpectralFunction` with values = -Im[G(ω)] / π on the given `omega` grid.
pub fn fft_to_spectral(
    g_t_extended: &[Complex<f64>],
    dt: f64,
    omega: &[f64],
) -> SpectralFunction;

/// Apply regularized Lorentzian deconvolution to remove broadening η.
///
/// Regularized deconvolution formula (design doc §8.4.2):
///   A_true(ω) ≈ A_windowed(ω) × (η² + ω²) / (2η + δ × ω²)
///
/// For |ω| > `config.deconv_omega_max` (in units of bandwidth), the
/// correction factor is clamped to 1.0 (no correction; tail assumed negligible).
///
/// # Errors
/// Returns `DmftError::DeconvolutionFailed { eta }` if `broadening_eta == 0.0`
/// (deconvolution is a no-op and must not be called when η = 0; callers
/// should skip this step when `config.broadening_eta == 0.0`).
pub fn deconvolve_lorentzian(
    spectral: &SpectralFunction,
    config: &LinearPredictionConfig,
) -> DmftResult<SpectralFunction>;
```

### 6.4 `restore_positivity`

```rust
/// Clamp negative spectral weight and renormalize to preserve the sum rule.
///
/// Mandatory post-deconvolution pass (design doc §8.4.2). The four steps are:
///
/// **Step 1 — Diagnostic check:**
/// Compute W_neg = Σ_{A(ω)<0} |A(ω)| Δω and W_total = Σ |A(ω)| Δω.
/// If W_neg / W_total > `config.positivity_warning_threshold`, emit:
///   log::warn!(target: "tensorkraft::telemetry", "SPECTRAL_POSITIVITY_WARNING: ...")
///
/// **Step 2 — Record Fermi level before clamping:**
/// A_fermi_before = `spectral.value_at_omega_zero()`
///
/// **Step 3 — Clamp and L₁ renormalize:**
/// A(ω) = max(A(ω), `config.positivity_floor`)
/// scale = W_total_original / W_total_clamped
/// A(ω) *= scale  (preserves ∫A(ω)dω = original sum rule)
///
/// **Step 4 — Fermi-level distortion check:**
/// A_fermi_after = value at ω = 0 after clamping + rescaling.
/// If |A_fermi_after - A_fermi_before| / |A_fermi_before| > `fermi_level_shift_tolerance`,
/// emit: log::warn!(target: "tensorkraft::telemetry", "FERMI_LEVEL_DISTORTION: ...")
///
/// This function never fails. All diagnostic conditions emit log warnings only.
///
/// # Returns
/// `SpectralFunction` with A(ω) ≥ 0 everywhere and sum rule preserved.
///
/// # Example log events
/// ```text
/// [WARN tensorkraft::telemetry] SPECTRAL_POSITIVITY_WARNING: 7.3% of spectral
/// weight is negative (threshold: 5.0%). Deconvolution parameters (η=0.05,
/// δ=0.001, ω_max=10.0) may need adjustment.
///
/// [WARN tensorkraft::telemetry] FERMI_LEVEL_DISTORTION: A(ω=0) shifted by
/// 1.4% after positivity restoration (tolerance: 1.0%). Before: 3.142e-01,
/// After: 3.098e-01. This may corrupt the quasiparticle residue. Consider
/// reducing η or increasing ω_max to reduce tail ringing.
/// ```
pub fn restore_positivity(
    spectral: &SpectralFunction,
    config: &LinearPredictionConfig,
) -> SpectralFunction;
```

---

## 7. Chebyshev Spectral Expansion

### 7.1 `ChebyshevConfig`

```rust
/// Configuration for Chebyshev expansion of the spectral function.
///
/// Chebyshev expansion computes A(ω) directly in frequency space by
/// expanding the resolvent in Chebyshev polynomials rescaled to the
/// spectrum of H. This bypasses both Trotter error (TEBD) and the
/// time-domain linear prediction instabilities that afflict metallic phases.
///
/// Mandatory cross-validation for all DMFT runs regardless of phase
/// (design doc §8.4.3). Primary solver for gapless/metallic phases.
#[derive(Clone, Debug)]
pub struct ChebyshevConfig {
    /// Number of Chebyshev moments to compute. Default: 1000.
    /// Controls frequency resolution: δω ≈ (E_max - E_min) / n_moments.
    pub n_moments: usize,
    /// Lorentzian broadening η (Jackson kernel damping). Default: 0.05.
    pub broadening_eta: f64,
    /// Lower spectral bound E_min. Must satisfy E_min < E_ground.
    /// `None` = auto-detect from ground state energy ± 20%.
    pub e_min: Option<f64>,
    /// Upper spectral bound E_max. `None` = auto-detect.
    pub e_max: Option<f64>,
    /// Apply Jackson kernel to reduce Gibbs oscillations.
    /// Trades frequency resolution for smoothness. Default: true.
    pub jackson_kernel: bool,
}

impl Default for ChebyshevConfig {
    fn default() -> Self {
        ChebyshevConfig {
            n_moments: 1000,
            broadening_eta: 0.05,
            e_min: None,
            e_max: None,
            jackson_kernel: true,
        }
    }
}
```

### 7.2 `chebyshev_expand`

```rust
/// Compute the impurity spectral function via Chebyshev expansion.
///
/// **Algorithm (design doc §8.4.3):**
///
/// 1. Rescale H to H_tilde = (H - b) / a where:
///      a = (E_max - E_min) / (2 - ε)   (small ε for numerical safety)
///      b = (E_max + E_min) / 2
///    This maps the spectrum into (-1, 1).
///
/// 2. Construct |α⟩ = c†_{0σ}|ψ₀⟩ by applying the impurity creation
///    operator to the DMRG ground state.
///
/// 3. Compute Chebyshev moments via the three-term recursion:
///      |φ₀⟩ = |α⟩
///      |φ₁⟩ = H_tilde|α⟩
///      |φ_n⟩ = 2 H_tilde|φ_{n-1}⟩ − |φ_{n-2}⟩
///      μ_n = ⟨ψ₀|c_{0σ}|φ_n⟩
///
///    Each step calls `DMRGEngine::apply_hamiltonian` (H_eff matvec reuse).
///
/// 4. Apply Jackson kernel (if enabled):
///      g_n = [(N-n+1)cos(πn/(N+1)) + sin(πn/(N+1))cot(π/(N+1))] / (N+1)
///    where N = `config.n_moments`.
///
/// 5. Reconstruct A(ω):
///      A(ω) = (1/πa√(1-ω̃²)) × (μ₀ + 2 Σ_{n≥1} g_n μ_n T_n(ω̃))
///    where ω̃ = (ω - b) / a is the rescaled frequency.
///
/// # Complexity
/// O(`n_moments` × N × d × D² × w) — one H_eff matvec application per moment.
///
/// # Errors
/// Returns `DmftError::ChebyshevBandwidthError` if E_min >= E_max or if the
/// ground state energy lies outside [E_min, E_max].
pub fn chebyshev_expand<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>(
    engine: &DMRGEngine<T, Q, B>,
    mpo: &MPO<T, Q>,
    omega: &[f64],
    config: &ChebyshevConfig,
    backend: &B,
) -> DmftResult<SpectralFunction>;
```

---

## 8. DMFT Self-Consistency Loop

### 8.1 `TimeEvolutionConfig`

```rust
/// Configuration for TDVP-based real-time Green's function computation.
///
/// Controls the total simulation time, time step size, and both the
/// TDVP stabilization parameters and the Chebyshev cross-validation
/// configuration.
#[derive(Clone, Debug)]
pub struct TimeEvolutionConfig {
    /// Total simulation time t_max (in inverse energy units). Default: 20.0.
    pub t_max: f64,
    /// Physical time step dt. Default: 0.05.
    pub dt: f64,
    /// Maximum MPS bond dimension during time evolution. Default: 500.
    pub max_bond_dim: usize,
    /// TDVP numerical stabilization configuration.
    /// Contains Tikhonov parameters, soft D_max policy, and subspace expansion
    /// settings. See `TdvpStabilizationConfig` in `tk-dmrg`.
    pub tdvp_stabilization: TdvpStabilizationConfig,
    /// Chebyshev cross-validation configuration.
    pub chebyshev: ChebyshevConfig,
    /// Relative L∞ tolerance for TDVP/Chebyshev consistency check.
    /// If ‖A_primary − A_cross‖_∞ / ‖A_cross‖_∞ > this value,
    /// emit a `SPECTRAL_CROSS_VALIDATION_WARNING` log event.
    /// Default: 0.05 (5%).
    pub cross_validation_tol: f64,
}

impl Default for TimeEvolutionConfig {
    fn default() -> Self {
        TimeEvolutionConfig {
            t_max: 20.0,
            dt: 0.05,
            max_bond_dim: 500,
            tdvp_stabilization: TdvpStabilizationConfig::default(),
            chebyshev: ChebyshevConfig::default(),
            cross_validation_tol: 0.05,
        }
    }
}
```

### 8.2 `MixingScheme`

```rust
/// Bath-update mixing scheme for DMFT self-consistency convergence.
///
/// The self-consistency condition requires bath_new = f(G_imp[bath_old]).
/// Direct substitution (linear mixing α = 1.0) is often unstable for
/// strongly correlated phases. Broyden mixing uses quasi-Newton updates
/// of the residual history to accelerate convergence.
#[derive(Clone, Debug)]
pub enum MixingScheme {
    /// Linear mixing: bath_new = (1 - α) × bath_old + α × bath_from_spectral.
    /// Default α: 0.3.
    Linear { alpha: f64 },
    /// Broyden's first method (good Broyden) for quasi-Newton acceleration.
    /// Maintains a history of `history_depth` previous (input, residual) pairs.
    /// Default `alpha`: 0.5, default `history_depth`: 5.
    Broyden { alpha: f64, history_depth: usize },
}

impl Default for MixingScheme {
    fn default() -> Self {
        MixingScheme::Broyden { alpha: 0.5, history_depth: 5 }
    }
}
```

### 8.3 `DMFTConfig`

```rust
/// Top-level configuration for a DMFT self-consistency run.
#[derive(Clone, Debug)]
pub struct DMFTConfig {
    /// DMRG sweep configuration for ground-state computation at each iteration.
    pub dmrg_config: DMRGConfig,
    /// Time evolution and spectral function extraction configuration.
    pub time_evolution: TimeEvolutionConfig,
    /// Linear prediction pipeline configuration.
    pub linear_prediction: LinearPredictionConfig,
    /// Primary/fallback spectral solver selection strategy.
    /// Default: `Adaptive { gap_threshold: 0.1 }`.
    pub solver_mode: SpectralSolverMode,
    /// Bath update mixing scheme. Default: `Broyden { alpha: 0.5, history_depth: 5 }`.
    pub mixing: MixingScheme,
    /// DMFT convergence criterion: relative change in hybridization function.
    /// Converged when ‖Δ_new − Δ_old‖_∞ / ‖Δ_old‖_∞ < this value.
    /// Default: 1e-4.
    pub self_consistency_tol: f64,
    /// Maximum number of DMFT self-consistency iterations. Default: 50.
    pub max_iterations: usize,
    /// Bath discretization configuration.
    pub bath_discretization: BathDiscretizationConfig,
    /// Optional checkpoint path. When set, writes a `DMFTCheckpoint` after
    /// each completed iteration (atomically via temp file + rename).
    pub checkpoint_path: Option<std::path::PathBuf>,
}
```

### 8.4 `DMFTLoop<T, Q, B>`

```rust
/// The DMFT self-consistency driver.
///
/// Holds the current Anderson Impurity Model and drives the iterative loop:
///   bath → DMRG ground state → spectral function → new bath → ...
///
/// # Type Parameters
/// - `T`: scalar type (use `f64` for the standard single-orbital case)
/// - `Q`: quantum number type (typically `U1` for particle-number conservation)
/// - `B`: linear algebra backend (typically `DeviceFaer` for CPU)
pub struct DMFTLoop<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> {
    /// Current Anderson Impurity Model (bath updated each iteration).
    pub impurity: AndersonImpurityModel<T>,
    /// DMFT configuration (immutable for the duration of the run).
    pub config: DMFTConfig,
    /// Accumulated statistics.
    pub stats: DMFTStats,
    backend: B,
    _phantom: PhantomData<Q>,
}

impl<T, Q, B> DMFTLoop<T, Q, B>
where
    T: Scalar<Real = f64>,
    Q: BitPackable,
    B: LinAlgBackend<T>,
{
    /// Construct the DMFT loop from an initial AIM, configuration, and backend.
    pub fn new(
        impurity: AndersonImpurityModel<T>,
        config: DMFTConfig,
        backend: B,
    ) -> Self;

    /// Run the self-consistency loop until convergence or `max_iterations`.
    ///
    /// Returns the converged primary spectral function A(ω).
    ///
    /// # Self-consistency loop (design doc §8.4):
    ///
    /// ```text
    /// loop {
    ///     chain_mpo = impurity.build_chain_hamiltonian()
    ///     gs_engine = DMRGEngine::new(mps_init, chain_mpo, backend, dmrg_config)
    ///     gs_engine.run()?
    ///
    ///     // Adaptive solver selection (design doc §8.4.1):
    ///     let use_cheb_primary = match config.solver_mode {
    ///         TdvpPrimary       => false,
    ///         ChebyshevPrimary  => true,
    ///         Adaptive { gap_threshold } => {
    ///             gs_engine.mps.entanglement_gap_at_center() < gap_threshold
    ///         }
    ///     };
    ///
    ///     // Always compute both:
    ///     spectral_tdvp = self.tdvp_spectral(&gs_engine, &chain_mpo)?
    ///     spectral_cheb = chebyshev_expand(&gs_engine, &chain_mpo, ..)?
    ///
    ///     let (primary, cross) = if use_cheb_primary {
    ///         (&spectral_cheb, &spectral_tdvp)
    ///     } else {
    ///         (&spectral_tdvp, &spectral_cheb)
    ///     };
    ///
    ///     self.validate_consistency(primary, cross)
    ///
    ///     delta_new = self.weiss_field(primary)
    ///     bath_new  = impurity.discretize(&delta_new, ..)?
    ///     bath_mixed = apply_mixing(&self.impurity.bath, &bath_new, &config.mixing)
    ///     impurity.update_bath(bath_mixed)
    ///
    ///     if converged { return Ok(primary.clone()); }
    ///     if n_iters >= max_iterations {
    ///         return Err(DmftError::MaxIterationsExceeded { .. });
    ///     }
    /// }
    /// ```
    ///
    /// # Errors
    /// - `DmftError::Dmrg` wrapping any `DmrgError` from DMRG or TDVP
    /// - `DmftError::BathDiscretizationFailed` if Lanczos discretization fails
    /// - `DmftError::MaxIterationsExceeded { iterations, distance }` if
    ///   `max_iterations` is reached before convergence
    pub fn solve(&mut self) -> DmftResult<SpectralFunction>;

    /// Run with an `AtomicBool` cancellation flag.
    ///
    /// The cancel flag is checked once per complete DMFT iteration using a
    /// single `Relaxed` load. Returns `DmftError::Cancelled` immediately
    /// if the flag is set. Does not leave partially updated bath state.
    pub fn solve_with_cancel_flag(
        &mut self,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> DmftResult<SpectralFunction>;

    /// Whether the most recent iteration satisfied `self_consistency_tol`.
    pub fn converged(&self) -> bool;

    /// Number of completed self-consistency iterations.
    pub fn n_iterations(&self) -> usize;

    /// Reference to the current bath parameters.
    pub fn bath(&self) -> &BathParameters<T>;
}
```

### 8.5 Internal helpers (`pub(crate)`)

```rust
impl<T, Q, B> DMFTLoop<T, Q, B>
where
    T: Scalar<Real = f64>,
    Q: BitPackable,
    B: LinAlgBackend<T>,
{
    /// Run the full TDVP + linear prediction pipeline for one DMFT iteration.
    ///
    /// 1. Construct |α⟩ = c†_{0σ}|ψ₀⟩ from the ground-state MPS.
    /// 2. Run `TdvpDriver::run(n_steps, dt, max_bond_dim, cancel)`.
    /// 3. Sample G(t_k) = ⟨ψ₀|c_{0σ}|α(t_k)⟩ at each step.
    /// 4. Call `linear_predict_regularized`.
    /// 5. Call `fft_to_spectral`.
    /// 6. Call `deconvolve_lorentzian` (only if `broadening_eta > 0`).
    /// 7. Call `restore_positivity` (always).
    pub(crate) fn tdvp_spectral(
        &self,
        gs: &DMRGEngine<T, Q, B>,
        chain_mpo: &MPO<T, Q>,
    ) -> DmftResult<SpectralFunction>;

    /// Compute the non-interacting Weiss field from the impurity spectral function.
    ///
    /// For the Bethe lattice with coordination z and half-bandwidth W:
    ///   G₀⁻¹(ω) = ω + μ − (W²/z) × G_imp(ω)
    ///   Δ(ω) = ω + μ − G_imp⁻¹(ω) − G₀⁻¹_bare(ω)
    ///
    /// G_imp(ω) is reconstructed from A_imp(ω) via Kramers-Kronig.
    ///
    /// Returns the hybridization function Δ_new(ω) for bath discretization.
    pub(crate) fn weiss_field(
        &self,
        spectral: &SpectralFunction,
    ) -> Vec<Complex<f64>>;

    /// Apply the configured mixing scheme to produce the next bath parameters.
    ///
    /// Linear mixing: bath_next = (1 - α) × bath_current + α × bath_proposed
    /// Broyden: quasi-Newton update using history of (bath, residual) pairs
    pub(crate) fn apply_mixing(
        &mut self,
        bath_proposed: &BathParameters<T>,
    ) -> BathParameters<T>;

    /// Emit a cross-validation consistency warning if the two spectral functions
    /// disagree beyond `config.time_evolution.cross_validation_tol`.
    pub(crate) fn validate_consistency(
        &self,
        primary: &SpectralFunction,
        cross: &SpectralFunction,
    );
}
```

### 8.6 `DMFTStats`

```rust
/// Statistics accumulated across DMFT self-consistency iterations.
#[derive(Clone, Debug, Default)]
pub struct DMFTStats {
    /// DMRG ground-state energies at each iteration.
    pub ground_state_energies: Vec<f64>,
    /// Relative hybridization distances ‖Δ_new − Δ_old‖ / ‖Δ_old‖ per iteration.
    pub hybridization_distances: Vec<f64>,
    /// Spectral sum rule ∫A(ω)dω at each iteration (should be ≈ 1.0).
    pub spectral_sum_rules: Vec<f64>,
    /// Fraction of negative spectral weight clamped per iteration.
    pub positivity_clamped_fractions: Vec<f64>,
    /// Whether Chebyshev was the primary solver at each iteration.
    pub chebyshev_was_primary: Vec<bool>,
    /// Wall-clock seconds per iteration (DMRG + TDVP + Chebyshev).
    pub iteration_times_secs: Vec<f64>,
    /// DMRG sweep statistics per iteration.
    pub dmrg_stats: Vec<DMRGStats>,
}
```

---

## 9. MPI Mode B Integration

### 9.1 `initialize_dmft_node_budget`

```rust
/// Divide the node's pinned-memory budget across co-resident MPI ranks.
///
/// Called once at program startup before any `DMFTLoop::solve()` calls.
/// Queries system RAM, determines how many MPI ranks share the current
/// node (via shared-memory communicator split), and sets each rank's
/// per-process `PinnedMemoryTracker` budget to:
///   rank_budget = floor(0.60 × total_ram / n_local_ranks)
///
/// **MPI process-isolation semantics (design doc §10.2.1):**
/// `PinnedMemoryTracker` is a process-local atomic guard. Each rank
/// independently enforces its pre-negotiated slice using its own
/// process-local `AtomicUsize`. No inter-process synchronization occurs
/// after this call. Cross-rank coordination happens only at the DMFT
/// convergence check via `MPI_Allgather` in the calling application.
///
/// # Parameters
/// - `comm`: MPI communicator (typically `MPI_COMM_WORLD`)
///
/// # Safety
/// Must be called before any `SweepArena` construction.
/// Must be called by all ranks simultaneously (collective operation).
///
/// # Panics
/// Panics if `sys_info::mem_info()` fails (platform not supported).
#[cfg(all(feature = "backend-cuda", feature = "backend-mpi"))]
pub fn initialize_dmft_node_budget(comm: &MpiComm);
```

**Implementation (design doc §10.2.2):**

```rust
#[cfg(all(feature = "backend-cuda", feature = "backend-mpi"))]
pub fn initialize_dmft_node_budget(comm: &MpiComm) {
    let total_ram = sys_info::mem_info().unwrap().total as usize * 1024; // convert KB → bytes
    let local_comm = comm.split_by_shared_memory();
    let local_ranks = local_comm.size();
    let safe_node_limit = (total_ram as f64 * 0.60) as usize;
    let rank_budget = safe_node_limit / local_ranks;
    PinnedMemoryTracker::initialize_budget(rank_budget);
}
```

### 9.2 MPI Mode B Execution Pattern

In Mode B, each MPI rank runs an independent `DMFTLoop::solve_with_cancel_flag()` call on a different orbital or k-point sector. `tk-dmft` itself is MPI-unaware beyond `initialize_dmft_node_budget`. The application layer handles the `MPI_Allgather` convergence check:

```rust
// Application layer (not in tk-dmft):
let spectral = dmft_loop.solve_with_cancel_flag(&cancel)?;
let local_converged = dmft_loop.converged() as i32;

// MPI_Allreduce (all ranks check jointly):
let all_converged = comm.all_reduce_sum(&local_converged) == n_ranks as i32;
if all_converged { break; }
```

**Load-imbalance note (design doc §10.3):** For single-orbital Bethe lattice DMFT (Phase 4), iteration counts are relatively uniform across ranks and the barrier causes minimal idle time. For multi-orbital or cluster DMFT with heterogeneous bath parameters, iteration counts can vary by 2–5× across ranks. Async convergence checks and dynamic work-stealing are deferred to Phase 5+.

---

## 10. Error Handling

```rust
/// Error type for `tk-dmft`.
#[derive(Debug, thiserror::Error)]
pub enum DmftError {
    #[error("bath discretization failed: Lanczos tridiagonalization did not converge \
             within {max_steps} steps (residual = {residual:.2e})")]
    BathDiscretizationFailed { max_steps: usize, residual: f64 },

    #[error("linear prediction failed: Levinson-Durbin diverged \
             (estimated condition number = {condition:.2e}). \
             Consider increasing LinearPredictionConfig::toeplitz_solver.tikhonov_lambda.")]
    LinearPredictionFailed { condition: f64 },

    #[error("deconvolution requires broadening_eta > 0; got eta = {eta}. \
             Deconvolution must be skipped when eta = 0.0.")]
    DeconvolutionFailed { eta: f64 },

    #[error("Chebyshev bandwidth error: E_min ({e_min:.4}) >= E_max ({e_max:.4}), \
             or ground-state energy {e0:.4} outside [{e_min:.4}, {e_max:.4}]")]
    ChebyshevBandwidthError { e_min: f64, e_max: f64, e0: f64 },

    #[error("DMFT did not converge after {iterations} iterations \
             (final hybridization distance = {distance:.2e}, \
             threshold = {threshold:.2e})")]
    MaxIterationsExceeded { iterations: usize, distance: f64, threshold: f64 },

    #[error("spectral sum rule violated: integral A(ω) = {sum_rule:.6} (expected 1.0)")]
    SumRuleViolated { sum_rule: f64 },

    #[error("invalid hybridization function: -Im[Δ(ω)] < 0 at {n_negative} frequency \
             points. The hybridization function must have positive imaginary part.")]
    InvalidHybridizationFunction { n_negative: usize },

    #[error("checkpoint I/O error: {0}")]
    CheckpointIo(#[from] std::io::Error),

    #[error("checkpoint deserialization error: {0}")]
    CheckpointDeser(String),

    #[error("DMRG error")]
    Dmrg(#[from] DmrgError),

    #[error("computation cancelled")]
    Cancelled,
}

/// Result alias for `tk-dmft`.
pub type DmftResult<T> = Result<T, DmftError>;
```

**Error propagation:** `DmftError::Dmrg(#[from] DmrgError)` wraps the full `tk-dmrg` error hierarchy, which in turn wraps `LinAlgError`, `SymmetryError`, and `TkError`. DMFT-specific errors include quantitative diagnostics (residual, condition number, iteration count) to aid parameter tuning in production runs.

---

## 11. Feature Flags

| Feature Flag | Effect | Default |
|:-------------|:-------|:--------|
| `parallel` | Enables Rayon parallelism in FFT post-processing and Levinson-Durbin inner loops (propagated from `tk-linalg`). | Yes |
| `backend-cuda` | Activates pinned-memory budget tracking in `SweepArena` (propagated from `tk-dmrg`). Required for `initialize_dmft_node_budget`. | No |
| `backend-mpi` | Enables `initialize_dmft_node_budget` and the `MpiComm` type. Requires system MPI library. | No |
| `su2-symmetry` | Propagates SU(2) symmetry support into AIM Hamiltonian construction and DMRG engine. | No |

`backend-mkl` and `backend-openblas` are not declared in `tk-dmft`'s own `Cargo.toml`; they are propagated through `tk-dmrg`'s feature flags.

---

## 12. Build-Level Concerns

`tk-dmft/build.rs` performs two checks:

1. **MPI library detection:** When `backend-mpi` is enabled, probe for `mpicc` in PATH. Emit a `compile_error!` with installation instructions if not found.

2. **Feature conflict check (defense-in-depth):** Verify that `backend-mkl` and `backend-openblas` are not simultaneously active by checking environment variables set by `tk-linalg`'s `build.rs`. This mirrors the workspace-level enforcement.

---

## 13. Data Structures and Internal Representations

### 13.1 Time-Domain Green's Function

The retarded impurity Green's function at T=0:

```
G(t) = -i ⟨ψ₀| c_{0σ} exp(-iHt) c†_{0σ} |ψ₀⟩    (t > 0)
```

Stored as `Vec<Complex<f64>>` sampled at uniform times 0, dt, 2dt, ..., t_max. The FFT assumes uniform sampling; adaptive time-step TDVP must resample onto a uniform grid before calling `fft_to_spectral`.

### 13.2 Self-Consistency Equation (Bethe Lattice)

The DMFT self-consistency condition for the Bethe lattice (coordination z, half-bandwidth W):

```
G₀⁻¹(ω) = ω + μ - (W²/z) × G_imp(ω)
```

G_imp(ω) is recovered from A_imp(ω) via Kramers-Kronig. The hybridization function for the bath discretization step is:

```
Δ(ω) = ω + μ - G_imp⁻¹(ω) - ε_imp
```

The Weiss field is specialized to the Bethe lattice for Phase 4; the general-lattice Dyson equation path is marked `// TODO: Phase 5+`.

### 13.3 Tensor Leg Conventions

AIM chain Hamiltonian tensors follow the same conventions as `tk-dmrg` (design doc §9):

| Tensor | Rank | Leg order | Notes |
|:-------|:-----|:----------|:------|
| MPS site tensor | 3 | (σ, α_L, α_R) | impurity at site 0 |
| MPO site tensor | 4 | (σ_in, σ_out, w_L, w_R) | |
| Left environment | 3 | (α_bra, w, α_ket) | |
| Right environment | 3 | (α_bra, w, α_ket) | |

Physical dimensions: impurity site d = 4 (|0⟩, |↑⟩, |↓⟩, |↑↓⟩) for spin-1/2 single-orbital; bath sites d = 2 (|0⟩, |1⟩ per spin channel in the spinless basis) or d = 4 for spin-resolved bath.

### 13.4 Checkpoint Format

```rust
/// Serializable checkpoint for a DMFT run.
///
/// Written atomically after each DMFT iteration when
/// `DMFTConfig::checkpoint_path` is set (write to `{path}.tmp`, then rename).
/// Enables restart from the last completed iteration without rerunning
/// previous converged bath parameters.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct DMFTCheckpoint<T: Scalar>
where
    T: serde::Serialize + for<'de> serde::Deserialize<'de>,
    T::Real: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    /// Bath parameters from the last completed iteration.
    pub bath: BathParameters<T>,
    /// Primary spectral function from the last completed iteration.
    pub spectral: SpectralFunction,
    /// Iteration index at time of checkpoint.
    pub iteration: usize,
    /// Whether the loop had converged at this checkpoint.
    pub converged: bool,
    /// `DMFTConfig` serialized as JSON for human inspection.
    pub config_json: String,
    /// Statistics accumulated up to this checkpoint.
    pub stats: DMFTStats,
}

impl<T: Scalar> DMFTCheckpoint<T>
where
    T: serde::Serialize + for<'de> serde::Deserialize<'de>,
    T::Real: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    /// Write checkpoint atomically: write to `{path}.tmp`, then rename.
    ///
    /// # Errors
    /// Returns `DmftError::CheckpointIo` on any filesystem error.
    pub fn write_to_file(&self, path: &std::path::Path) -> DmftResult<()>;

    /// Load checkpoint from file.
    ///
    /// # Errors
    /// Returns `DmftError::CheckpointIo` on I/O failure.
    /// Returns `DmftError::CheckpointDeser` on deserialization failure.
    pub fn read_from_file(path: &std::path::Path) -> DmftResult<Self>;
}
```

---

## 14. Dependencies and Integration

### 14.1 Upstream Dependencies (`Cargo.toml`)

```toml
[dependencies]
tk-dmrg     = { path = "../tk-dmrg",     version = "0.1.0" }
# tk-dmrg re-exports the full dependency chain:
#   tk-core, tk-symmetry, tk-linalg, tk-contract, tk-dsl

num-complex = "0.4"
num-traits  = "0.2"
rustfft     = "6"         # FFT for spectral function extraction
thiserror   = "1"
log         = "0.4"
tracing     = "0.1"

serde       = { version = "1", features = ["derive"] }
bincode     = "2"
serde_json  = "1"

[dependencies.sys-info]
version  = "0.9"
optional = true           # gated on feature = "backend-mpi"

[dependencies.mpi]
version  = "0.7"
optional = true           # gated on feature = "backend-mpi"

[dev-dependencies]
proptest     = "1"
approx       = "0.5"
rand_chacha  = "0.3"
criterion    = { version = "0.5", optional = true }

[features]
default         = ["parallel"]
parallel        = ["tk-dmrg/parallel"]
backend-cuda    = ["tk-dmrg/backend-cuda"]
backend-mpi     = ["mpi", "sys-info", "tk-dmrg/backend-mpi"]
su2-symmetry    = ["tk-dmrg/su2-symmetry"]
```

### 14.2 Downstream Consumers

**`tk-python`** — Wraps `DMFTLoop<f64, U1, DeviceFaer>` in a `PyDMFTLoop` PyO3 class. Calls `solve_with_cancel_flag` inside a `py.allow_threads` closure so that Rayon workers inside `DMRGEngine` never touch the Python GIL (design doc §7.5). Spectral function output is zero-copied to NumPy via `rust-numpy`.

### 14.3 External Dependencies by Functionality

| Crate | Purpose | Feature Gate |
|:------|:--------|:-------------|
| `rustfft` | FFT in `fft_to_spectral` | always |
| `mpi` | `MpiComm` type in `initialize_dmft_node_budget` | `backend-mpi` |
| `sys-info` | System RAM query for pinned-budget calculation | `backend-mpi` |
| `tk-dmrg` | `DMRGEngine`, `TdvpDriver`, `MPO`, `MPS`, `TdvpStabilizationConfig`, `MpoCompiler`, `DmrgError` | always |

---

## 15. Testing Strategy

### 15.1 Unit Tests

| Test | Description |
|:-----|:------------|
| `bath_uniform_constructor` | Verify `BathParameters::uniform(n_bath=6, bw=10.0, v0=1.0)` produces energies in [-5, 5] (uniform), all V_k = 1.0. |
| `bath_hybridization_function` | Construct a 4-site bath with analytically known ε_k, V_k. Verify ‖Δ_computed - Δ_exact‖_∞ < 1e-12. |
| `bath_discretization_semicircular` | Discretize a semicircular Δ(ω) with n_bath=6. Verify ‖Δ_discretized - Δ_target‖_∞ / ‖Δ_target‖_∞ < 1e-8. |
| `bath_discretization_convergence` | Verify `DmftError::BathDiscretizationFailed` is returned when `max_lanczos_steps` is artificially reduced to 1. |
| `invalid_hybridization_function` | Verify `DmftError::InvalidHybridizationFunction` is returned when -Im[Δ(ω)] < 0 at some grid points. |
| `levinson_durbin_matches_svd` | For random Toeplitz systems of size P = 10, 50, 100: verify Levinson-Durbin solution matches SVD pseudo-inverse to within 1e-8. |
| `levinson_durbin_tikhonov_stability` | Verify Levinson-Durbin with `tikhonov_lambda = 1e-8` stabilizes a Toeplitz matrix with condition number > 10⁸ (constructed from near-degenerate autocorrelation). |
| `exponential_windowing_noop_gapped` | Construct a synthetic G(t) ∝ exp(-t). Verify deconvolved A(ω) with η > 0 matches A(ω) computed with η = 0 to within 1e-6 L∞. |
| `exponential_windowing_prevents_divergence` | For synthetic metallic G(t) ~ t⁻¹: verify prediction with η > 0 produces finite extrapolation; verify η = 0 produces diverging prediction coefficients (condition number > 10¹⁰). |
| `lorentzian_deconvolution_recovery` | Construct a known Lorentzian A(ω) = (η/π)/(ω²+η²), apply forward windowing+FFT, verify `deconvolve_lorentzian` recovers the original within 1% L∞ error. |
| `deconvolution_noise_amplification_bound` | Add white noise of amplitude ε to a smooth spectrum; verify that the noise in the deconvolved spectrum is bounded by ε/`deconv_tikhonov_delta`. |
| `deconvolution_omega_max_continuity` | Verify the deconvolution factor is continuous at ω = `deconv_omega_max` (no jump discontinuity). |
| `deconvolution_requires_eta` | Verify `DmftError::DeconvolutionFailed` is returned when `broadening_eta == 0.0`. |
| `positivity_restoration_nonneg` | For a random spectral function with some negative values: verify A(ω) ≥ 0 for all ω after `restore_positivity`. |
| `positivity_sum_rule_preserved` | Verify ∫A(ω)dω is equal before and after `restore_positivity` (to within 1e-12), for both mostly-positive and 4% negative input spectra. |
| `positivity_warning_fires` | Synthetically inject 6% negative weight. Verify `SPECTRAL_POSITIVITY_WARNING` log event fires. Verify no warning for 4% negative weight (below 5% threshold). |
| `positivity_warning_no_false_positive` | Verify no warning for a perfectly positive spectrum. |
| `fermi_level_distortion_diagnostic` | Construct spectrum with concentrated negative ringing at |ω| > 5 (forces large rescaling near ω=0). Verify `FERMI_LEVEL_DISTORTION` warning fires when relative shift > 1%. Verify no warning when negative weight is small and uniformly distributed. |
| `restore_positivity_idempotent` | Verify applying `restore_positivity` twice gives the same result as applying it once. |
| `chebyshev_sum_rule` | Run `chebyshev_expand` on a 6-site AIM. Verify ∫A(ω)dω = 1.0 ± 1e-4. |
| `chebyshev_jackson_kernel` | Verify Jackson kernel reduces spectral oscillation variance for a sharp peak, compared to no kernel. |
| `chebyshev_bandwidth_error` | Verify `DmftError::ChebyshevBandwidthError` fires when E_min >= E_max or when the DMRG ground state energy is outside [E_min, E_max]. |
| `spectral_function_moments` | Verify spectral moments ∫ωⁿ A(ω)dω (n = 0, 1, 2) match analytic results for a known Lorentzian spectrum. |
| `adaptive_solver_selects_chebyshev` | Build a metallic AIM. Verify `SpectralSolverMode::Adaptive` selects `ChebyshevPrimary` when entanglement gap < `gap_threshold`. |
| `adaptive_solver_selects_tdvp` | Build a Mott insulating AIM (large U). Verify `SpectralSolverMode::Adaptive` selects `TdvpPrimary` when entanglement gap >= `gap_threshold`. |
| `soft_dmax_physical_time_invariant` | Run TDVP on a 10-site AIM with dt=0.1 and dt=0.05. Verify D_target(t) curves match between the two runs to within ±1 (same physical time axis). |
| `advance_expansion_age` | Verify `TdvpDriver::advance_expansion_age(dt=0.1)` accumulates physical time across 100 steps; verify bond resets to `None` when effective D_max has decayed to hard D_max. |
| `dmft_checkpoint_roundtrip` | Write a `DMFTCheckpoint` with bincode, reload from file. Verify all bath parameter values and spectral function values match f64 round-trip precision. |
| `initialize_budget_mpi` | (gated `backend-mpi`) Mock two co-resident ranks; verify each rank's `PinnedMemoryTracker` budget = floor(0.60 × total_ram / 2). |

### 15.2 Integration Tests — Bethe Lattice Benchmarks

Gated behind the `integration-tests` feature flag (slow; excluded from standard `cargo test`).

Reference fixtures (from ED/NRG external frameworks):

```
fixtures/
├── bethe_z4_u0_metallic.json        # U=0, half-filling: Fermi liquid
├── bethe_z4_u4_correlated.json      # U=4W, half-filling: strongly correlated metal
├── bethe_z4_u8_mott.json            # U=8W, half-filling: Mott insulator
└── README.md                        # provenance, NRG code version, date
```

| Test | Description |
|:-----|:------------|
| `bethe_u0_sum_rule` | U=0 Bethe lattice. Verify ∫A(ω)dω = 1.0 ± 1e-3 after convergence. |
| `bethe_u4_convergence` | U=4W. Verify DMFT self-consistency converges within 30 iterations (hybridization distance < 1e-4). |
| `bethe_u4_spectral_match` | U=4W. Verify converged A(ω) matches reference fixture within L∞ error 0.01. |
| `bethe_u8_mott_gap` | U=8W. Verify Mott gap: A(ω) < 1e-4 for |ω| < 0.5W; verify upper and lower Hubbard band positions match reference. |
| `metallic_chebyshev_promoted` | U=0: verify `SpectralSolverMode::Adaptive` promotes Chebyshev to primary (entanglement gap < 0.1). |
| `mott_tdvp_primary` | U=8W: verify `SpectralSolverMode::Adaptive` keeps TDVP as primary (entanglement gap >= 0.1). |
| `positivity_metallic_no_kondo_corruption` | U=4W correlated metal. Verify A(ω=0) shift after positivity restoration < 1% (Kondo peak not corrupted). |
| `mode_b_two_rank_convergence` | (gated `backend-mpi`) Run two-rank Mode B DMFT on U=0 and U=4W. Verify both converge and spectral functions match single-rank runs to within 1e-4. |

### 15.3 Property-Based Tests

```rust
// Idempotency: restoring a positive spectrum is a no-op
proptest! {
    #[test]
    fn prop_positivity_idempotent(
        values in prop::collection::vec(0.0f64..=5.0, 100)
    ) {
        let omega: Vec<f64> = (0..100).map(|i| -5.0 + 0.1 * i as f64).collect();
        let spectral = SpectralFunction::new(omega, values);
        let config = LinearPredictionConfig::default();
        let r1 = restore_positivity(&spectral, &config);
        let r2 = restore_positivity(&r1, &config);
        for (a, b) in r1.values.iter().zip(r2.values.iter()) {
            approx::assert_abs_diff_eq!(a, b, epsilon = 1e-14);
        }
    }
}

// Levinson-Durbin and SVD agree on well-conditioned Toeplitz systems
proptest! {
    #[test]
    fn prop_levinson_matches_svd(
        entries in prop::collection::vec(0.5f64..=3.0, 10..=30)
    ) {
        // Build a positive Toeplitz matrix from entries (diagonals)
        let ld = solve_toeplitz_levinson_durbin(&entries, 1e-8);
        let sv = solve_toeplitz_svd(&entries, 1e-8);
        for (a, b) in ld.iter().zip(sv.iter()) {
            approx::assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
    }
}
```

**Proptest strategy bounds (consistent with workspace conventions):**
- Maximum bath sites in property tests: 8
- Maximum prediction order P: 50 (not 200)
- Maximum frequency grid points: 200
- Maximum 256 cases per test
- Deterministic seed via `PROPTEST_SEED`

### 15.4 Performance Benchmarks

```
// Criterion.rs — bare-metal only; iai instruction counting for CI gating
benchmarks:
  levinson_p100       : Levinson-Durbin, P=100 (target < 10 µs)
  levinson_p200       : Levinson-Durbin, P=200 (target < 50 µs)
  fft_4096            : rustfft on 4096-point complex time series (target < 100 µs)
  chebyshev_n1000_d200: 1000 Chebyshev moments, D=200, N=14-site AIM
  bath_discretize_n8  : Lanczos tridiagonalization, n_bath=8 (target < 1 ms)
  dmft_iteration_n6   : full single DMFT iteration, n_bath=6, D=200
```

CI gating: iai instruction counting, ±2% regression threshold (consistent with workspace-wide convention).

---

## 16. Implementation Notes

### Note 1 — Levinson-Durbin Recursion

The standard Levinson-Durbin algorithm for R·a = r (Toeplitz R, where R_{ij} = autocorr(|i-j|)):

```
Initialization (with Tikhonov: r₀ → r₀ + λ):
  α₁ = r₁ / (r₀ + λ)
  f = [1, α₁], b = [α₁, 1]  (forward/backward prediction vectors)

For n = 2..P:
  κ = (r_n + Σ_{k=1}^{n-1} f_k × r_{n-k}) / (r₀ + λ)
  f_new = f + κ × reverse(b)
  b_new = b + κ × reverse(f)
  f = f_new, b = b_new

Solution: a = [f₁, f₂, ..., f_P] (the prediction coefficients)
```

The inner accumulation `Σ_{k=1}^{n-1} f_k × r_{n-k}` is a dot product that can be vectorized with SIMD under the `parallel` feature.

### Note 2 — Soft D_max Physical Time (tk-dmrg delegation)

The architecture (v8.3 revision, design doc §8.1.1) changed `TdvpDriver`'s `expansion_age` from a discrete counter (`usize`) to accumulated physical time (`f64`), renamed `dmax_decay_steps` to `dmax_decay_time`, and replaced `tick_expansion_age` with `advance_expansion_age(dt: f64, hard_dmax: usize)`. This entire mechanism lives inside `tk-dmrg`'s `TdvpDriver`. `tk-dmft` does not duplicate this state; it simply passes the physical `dt` to each `TdvpDriver::step(dt, hard_dmax)` call.

### Note 3 — Adaptive Tikhonov in TdvpStabilizationConfig

The `TdvpStabilizationConfig` (v8.2 revision, design doc §8.1.1) added `adaptive_tikhonov: bool` (default: `true`), `tikhonov_delta_scale: f64` (default: 0.01), and `tikhonov_delta_min: f64` (default: 1e-14). When enabled, δ is scaled per-bond to `max(δ_min, scale × σ_discarded_max)` from the previous truncation step. This prevents the regularization floor from masking physics in near-product-state bonds during DMFT time evolution. The field is in `TimeEvolutionConfig::tdvp_stabilization` and exposed directly to users of `DMFTConfig`.

### Note 4 — Chebyshev H_eff Reuse

`chebyshev_expand` needs to apply the Hamiltonian H repeatedly to an MPS. It reuses the H_eff construction machinery from `tk-dmrg`'s environment infrastructure. This requires either a new `DMRGEngine::apply_hamiltonian_to_mps` method, or direct access to `build_heff_single_site` + `ContractionExecutor`. This is an open design point (see §18, Open Question 3).

### Note 5 — rustfft Integration

`fft_to_spectral` uses `rustfft::FftPlanner::<f64>::new()` to create a plan for the DFT of the extended G(t) array. The planner is constructed per call (no caching). Since `fft_to_spectral` is called at most once per DMFT iteration, planner construction overhead is negligible. For future optimization if called in a tight loop, the planner can be cached inside `DMFTLoop`.

### Note 6 — Weiss Field for General Lattices

The current `weiss_field` implementation is specialized to the Bethe lattice (Phase 4 validation target). The general-lattice formula requires the k-resolved non-interacting Green's function and the full lattice self-consistency equation. The general path is stubbed with `todo!("Phase 5+: implement general lattice Weiss field via Dyson equation")`.

---

## 17. Public API Surface (`lib.rs` re-exports)

```rust
// tk-dmft/src/lib.rs

// Anderson Impurity Model
pub use crate::impurity::mod_::{AndersonImpurityModel};
pub use crate::impurity::bath::{BathParameters};
pub use crate::impurity::discretize::{BathDiscretizationConfig};

// Spectral function types and free functions
pub use crate::spectral::mod_::{SpectralFunction, SpectralSolverMode};
pub use crate::spectral::linear_predict::{
    ToeplitzSolver,
    LinearPredictionConfig,
    linear_predict_regularized,
    fft_to_spectral,
    deconvolve_lorentzian,
};
pub use crate::spectral::positivity::restore_positivity;
pub use crate::spectral::chebyshev::{ChebyshevConfig, chebyshev_expand};

// DMFT loop and configuration
pub use crate::r#loop::config::{DMFTConfig, TimeEvolutionConfig};
pub use crate::r#loop::mixing::MixingScheme;
pub use crate::r#loop::mod_::DMFTLoop;
pub use crate::r#loop::stats::DMFTStats;

// Checkpointing
pub use crate::r#loop::mod_::DMFTCheckpoint;

// MPI budget initialization (feature-gated)
#[cfg(all(feature = "backend-cuda", feature = "backend-mpi"))]
pub use crate::mpi::initialize_dmft_node_budget;

// Error types
pub use crate::error::{DmftError, DmftResult};
```

---

## 18. Out of Scope

- **Cluster DMFT / multi-orbital DMFT** — Phase 5+. Current design handles single-orbital single-site DMFT only.
- **Finite-temperature DMFT (Matsubara formalism)** — Phase 5+. Imaginary-time TDVP is architecturally supported (T = f64, imaginary dt), but the full Matsubara self-consistency and analytic continuation are not in scope.
- **MPI Mode A (distributed block-sparse tensors)** — Phase 5+. Mode B (embarrassingly parallel independent DMRG runs) requires no core changes to `tk-dmrg`.
- **GPU-accelerated FFT (cuFFT)** — Phase 5+. The spectral pipeline uses CPU-only `rustfft`.
- **General non-Bethe-lattice Weiss fields** — The self-consistency equation is specialized to the Bethe lattice. General-lattice paths are marked `// TODO: Phase 5+`.
- **TEBD time evolution** — Mentioned in the design doc as a fallback but not specified for Phase 4 implementation. TDVP (primary) and Chebyshev (cross-validation) cover the Phase 4 use cases. TEBD would be added as a third `SpectralSolverMode` variant in Phase 5+ if needed.
- **Async MPI convergence checks and dynamic work-stealing** — Design doc §10.3 documents these as Phase 5+ mitigations for MPI Mode B load imbalance under multi-orbital DMFT.

---

## 19. Open Questions

| # | Question | Status |
|:--|:---------|:-------|
| 1 | Should `DMFTLoop::weiss_field` be a method or a pluggable `WeissFieldStrategy` trait to support non-Bethe lattices without forking `tk-dmft`? Phase 4 only needs the Bethe lattice. | Open — defer to Phase 5; annotate current impl as Bethe-only |
| 2 | Should `rustfft::FftPlanner` be cached inside `DMFTLoop` for repeated use? At one FFT per DMFT iteration the overhead is negligible, but a future spectral-function-only mode (no self-consistency loop) could call it frequently. | Low priority — profile first; defer to Phase 5 |
| 3 | `chebyshev_expand` requires applying H repeatedly to an MPS. Does this require a new `DMRGEngine::apply_hamiltonian_to_mps` method, or can it reuse `build_heff_single_site` from `tk-dmrg`'s environment module? | Requires coordination with `tk-dmrg` before implementation; resolve at start of Phase 4 |
| 4 | Broyden mixing stores a history of bath-parameter vectors. Should these be stored as `Vec<BathParameters<T>>` (clean API) or as flat `Vec<f64>` (cache-friendly Jacobian update)? | Recommend flat storage for numerical efficiency; confirm with implementer |
| 5 | `initialize_dmft_node_budget` uses `sys_info::mem_info()` which queries physical RAM. On HPC clusters with Slurm `--mem` constraints, the cgroup memory limit may be less than physical RAM. Should the function query the cgroup limit via `/proc/self/cgroup` instead? | Flag for HPC deployment review; use `sys_info::mem_info()` for Phase 4 (document limitation) |
| 6 | Should `DMFTCheckpoint` include the full MPS state (via `DMRGCheckpoint` from `tk-dmrg`) to enable DMRG warm-starting from the previous iteration's converged state? This would save 1–3 sweeps per iteration at the cost of larger checkpoint files. | Recommend bath-only checkpoint for Phase 4; MPS warm-start as Phase 5 optimization |
