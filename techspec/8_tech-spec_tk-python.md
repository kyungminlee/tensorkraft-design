# Technical Specification: `tk-python`

**Crate:** `tensorkraft/crates/tk-python`
**Version:** 0.1.0 (Pre-Implementation)
**Status:** Specification
**Last Updated:** March 2026

---

## 1. Overview

`tk-python` is the Python bindings crate for the tensorkraft workspace. It sits at the root of the dependency graph, consuming `tk-dmft` (and transitively all other crates). It is the only crate in the workspace that targets Python users rather than Rust.

**Core responsibilities:**

- **Type-erased PyO3 dispatch** — PyO3's `#[pyclass]` cannot be applied to generic structs. `tk-python` bridges Rust's monomorphization to Python's dynamic dispatch via `DmftLoopVariant`, a non-generic enum that wraps the concrete `DMFTLoop<T, Q, B>` instantiations that users actually need. Adding a new type combination is a one-line enum variant; no recompilation of the underlying stack is required.
- **Explicit GIL release on all compute paths** — `DMFTLoop::solve()` can run for hours. Holding the GIL for its duration freezes Jupyter kernels, prevents signal delivery, and starves OS context switching. Every compute-heavy `#[pymethods]` implementation releases the GIL via `py.allow_threads()` before entering Rust numerics.
- **Thread-safe `AtomicBool` cancellation with `mpsc`-guarded monitor lifecycle** — A dedicated monitor thread polls `py.check_signals()` every 100 ms to handle Ctrl+C. The monitor uses an `mpsc::channel` for lifecycle management. Critically, the monitor shutdown sequence (`done_tx.send()` + `monitor_handle.join()`) executes *inside* the `py.allow_threads` closure while the GIL is still released. This eliminates the AB/BA deadlock that arises if the main thread re-acquires the GIL while the monitor is blocked on `Python::with_gil`. Rayon workers inside `DMRGEngine` never touch the GIL under any circumstances.
- **Zero-copy NumPy interop via `rust-numpy`** — Spectral function output arrays (`omega`, `values`) and bath parameter arrays are shared directly between Rust and Python as NumPy arrays without element-wise copying. Memory is owned by Rust and the Python reference-counting system manages lifetime.
- **TRIQS Green's function interop** — When the optional `triqs` feature is enabled, `tk-python` accepts TRIQS `GfImFreq` and `GfReFreq` objects for bath initialization, and exports spectral functions in TRIQS-compatible format.
- **PyPI wheel packaging** — Pre-built wheels for PyPI are compiled exclusively with pure-Rust backends (`backend-faer`, `backend-oxiblas`) for cross-platform compatibility (Linux, macOS, Windows) without requiring BLAS installations.

**Key design principle:** `tk-python` is a thin translation layer. All physics logic, numerical algorithms, and memory management reside in the underlying Rust crates. What `tk-python` owns is the GIL boundary: where the GIL is released, how cancellation signals cross that boundary safely, and how Rust memory is presented to Python without copying. The golden rule is: Rayon workers must never touch the Python GIL, and the monitor thread must never be alive when the main thread holds the GIL.

---

## 2. Module Structure

```
tk-python/
├── Cargo.toml
├── pyproject.toml              (maturin build metadata)
├── build.rs                    (feature conflict detection, TRIQS detection)
└── src/
    ├── lib.rs                  #[pymodule] tensorkraft; re-exports all pyclass items
    ├── dispatch/
    │   ├── mod.rs              DmftLoopVariant enum, DefaultDevice type alias
    │   └── macros.rs           dispatch_variant!{} macro for match arm generation
    ├── dmft/
    │   ├── mod.rs              PyDmftLoop: #[pyclass] wrapping DmftLoopVariant
    │   ├── solve.rs            solve(), solve_cancelled() with GIL release + monitor thread
    │   ├── config.rs           PyDmftConfig, PyDmrgConfig, PyTimeEvolutionConfig
    │   └── stats.rs            PyDmftStats: per-iteration diagnostics exposed to Python
    ├── spectral/
    │   ├── mod.rs              PySpectralFunction: #[pyclass] with NumPy getters
    │   └── numpy.rs            zero-copy omega/values array construction via rust-numpy
    ├── bath/
    │   ├── mod.rs              PyBathParameters: #[pyclass] for bath energy/hybridization access
    │   └── triqs.rs            TRIQS GfImFreq/GfReFreq interop (feature-gated)
    ├── config/
    │   ├── mod.rs              PyLinearPredictionConfig, PyMixingScheme, PySpectralSolverMode
    │   └── defaults.rs         Default implementations forwarding to Rust defaults
    ├── monitor/
    │   └── mod.rs              CancellationMonitor: AtomicBool + mpsc lifecycle (internal)
    └── error.rs                PythonError: DmftError/DmrgError → PyErr conversions
```

---

## 3. Type-Erased Dispatch

### 3.1 `DefaultDevice` Type Alias

```rust
/// The default device for Python-exposed computations.
///
/// The common case is the pure-Rust Faer backend, which is always compiled
/// and requires no external BLAS installation. PyPI wheels use this alias.
///
/// Changing the alias to `DeviceMKL` or `DeviceCuda` requires rebuilding
/// from source with the corresponding feature flags enabled.
#[cfg(all(feature = "backend-faer", not(feature = "backend-mkl"), not(feature = "backend-openblas")))]
pub type DefaultDevice = DeviceFaer;

#[cfg(feature = "backend-mkl")]
pub type DefaultDevice = DeviceMKL;

#[cfg(all(feature = "backend-openblas", not(feature = "backend-mkl")))]
pub type DefaultDevice = DeviceOpenBLAS;
```

### 3.2 `DmftLoopVariant`

```rust
/// Type-erased wrapper over the supported concrete `DMFTLoop<T, Q, B>` instantiations.
///
/// PyO3's `#[pyclass]` cannot be applied to generic structs. This enum
/// explicitly enumerates the type combinations exposed to Python, bridging
/// Rust's monomorphization to Python's dynamic dispatch.
///
/// # Supported combinations
///
/// | Variant        | Scalar          | Symmetry | Use case                              |
/// |:---------------|:----------------|:---------|:--------------------------------------|
/// | `RealU1`       | `f64`           | `U1`     | Standard single-orbital DMFT          |
/// | `ComplexU1`    | `Complex<f64>`  | `U1`     | Complex hybridization, e.g., SOC      |
/// | `RealZ2`       | `f64`           | `Z2`     | Particle-hole symmetric models        |
///
/// # Extending
/// Add a new combination by appending one enum variant here and adding
/// a corresponding arm in the `dispatch_variant!{}` macro in `dispatch/macros.rs`.
/// No recompilation of the underlying `tk-dmrg` / `tk-dmft` stack is required.
pub(crate) enum DmftLoopVariant {
    RealU1(DMFTLoop<f64, U1, DefaultDevice>),
    ComplexU1(DMFTLoop<Complex<f64>, U1, DefaultDevice>),
    RealZ2(DMFTLoop<f64, Z2, DefaultDevice>),
}
```

### 3.3 `dispatch_variant!{}` Macro

To avoid boilerplate match arms across all `#[pymethods]` implementations, a declarative macro generates them:

```rust
/// Generate a `match &mut self.inner { ... }` expression that calls `$method`
/// on the inner solver for every `DmftLoopVariant`.
///
/// # Example
///
/// ```rust
/// // Instead of writing three identical match arms:
/// match &mut self.inner {
///     DmftLoopVariant::RealU1(s)    => s.solve_with_cancel_flag(&flag),
///     DmftLoopVariant::ComplexU1(s) => s.solve_with_cancel_flag(&flag),
///     DmftLoopVariant::RealZ2(s)    => s.solve_with_cancel_flag(&flag),
/// }
///
/// // Write:
/// dispatch_variant!(self.inner, solve_with_cancel_flag(&flag))
/// ```
macro_rules! dispatch_variant {
    ($inner:expr, $method:ident ( $($arg:expr),* )) => {
        match &mut $inner {
            DmftLoopVariant::RealU1(s)    => s.$method($($arg),*),
            DmftLoopVariant::ComplexU1(s) => s.$method($($arg),*),
            DmftLoopVariant::RealZ2(s)    => s.$method($($arg),*),
        }
    };
}
```

---

## 4. GIL Release and Cancellation Monitor

### 4.1 `CancellationMonitor` (Internal)

```rust
/// Internal helper that encapsulates the AtomicBool cancellation flag and
/// the mpsc-based monitor thread lifecycle.
///
/// The monitor thread polls `py.check_signals()` every 100 ms. On SIGINT,
/// it sets the cancellation flag and exits. On solver completion, it receives
/// a message on `done_rx` and exits cleanly.
///
/// # Critical invariant
/// `shutdown()` MUST be called inside the `py.allow_threads` closure,
/// while the GIL is still released. Calling `shutdown()` after
/// `allow_threads` returns will deadlock if the monitor thread is blocked
/// on `Python::with_gil`.
///
/// See architecture §7.5 for the formal deadlock proof.
pub(crate) struct CancellationMonitor {
    /// Shared with the solver. Set to `true` by the monitor on SIGINT.
    pub cancel_flag: Arc<AtomicBool>,
    /// Send `()` to signal the monitor thread to exit.
    done_tx: mpsc::SyncSender<()>,
    /// Join handle for the monitor thread. `None` after `shutdown()`.
    monitor_handle: Option<thread::JoinHandle<()>>,
}

impl CancellationMonitor {
    /// Spawn the monitor thread.
    ///
    /// The thread loops on `done_rx.recv_timeout(Duration::from_millis(100))`:
    /// - `Ok(())` → solver finished normally; exit.
    /// - `Err(Disconnected)` → `done_tx` dropped (solver panicked); exit.
    /// - `Err(Timeout)` → call `Python::with_gil(|py| py.check_signals())`;
    ///   on `Err`, store `true` into `cancel_flag` with `Release` ordering and exit.
    pub(crate) fn spawn() -> Self;

    /// Signal the monitor thread to exit and join it.
    ///
    /// Sends `()` on `done_tx`, then calls `monitor_handle.join()`.
    ///
    /// # Correctness requirement
    /// Must be called while the GIL is released (inside `py.allow_threads`).
    pub(crate) fn shutdown(mut self);

    /// A shared reference to the cancellation flag, for passing to the solver.
    pub(crate) fn flag(&self) -> &Arc<AtomicBool>;
}
```

### 4.2 The `solve` Pattern

The canonical GIL release + cancellation pattern used by every long-running `#[pymethods]` function:

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

#[pymethods]
impl PyDmftLoop {
    pub fn solve(&mut self, py: Python<'_>) -> PyResult<PySpectralFunction> {
        let monitor = CancellationMonitor::spawn();
        let cancel_flag = monitor.flag().clone();

        let result = py.allow_threads(|| {
            let r = dispatch_variant!(self.inner, solve_with_cancel_flag(&cancel_flag));

            // CRITICAL: shutdown while the GIL is still released.
            // If this line executed after allow_threads returns, the main
            // thread would hold the GIL while the monitor may be blocked on
            // Python::with_gil — AB/BA deadlock.
            monitor.shutdown();

            r
        });

        match result {
            Ok(spectral) => Ok(PySpectralFunction::from(spectral)),
            Err(DmftError::Cancelled) => {
                Err(PyErr::new::<PyKeyboardInterrupt, _>("DMFT cancelled by user"))
            }
            Err(e) => Err(PythonError::from(e).into()),
        }
    }
}
```

**Monitor thread exit conditions:** There are exactly three, all of which guarantee clean shutdown:

1. `done_rx.recv_timeout` returns `Ok(())` — the solver finished normally. The monitor exits before `allow_threads` returns.
2. `done_rx` returns `Err(Disconnected)` — `done_tx` was dropped (solver panicked). The monitor exits without segfaulting; the panic propagates via `py.allow_threads` → `std::panic::resume_unwind`.
3. `py.check_signals()` detects SIGINT — the monitor stores `true` into `cancel_flag` with `Release` ordering and exits. The solver checks the flag at each sweep step boundary with `Relaxed` ordering (a single atomic load, effectively free).

**Why `done_tx.send()` + `join()` inside `allow_threads`:** By the time `allow_threads` returns and the main thread re-acquires the GIL, the monitor thread is guaranteed to be dead. No deadlock is possible because the monitor can only call `Python::with_gil` when the GIL is released (the main thread is inside `allow_threads`), and `shutdown()` is called while the GIL is still released. (Architecture §7.5.)

---

## 5. Python-Exposed Classes

### 5.1 `PyDmftLoop` (`DMFTLoop` in Python)

```rust
/// Python-accessible DMFT self-consistency solver.
///
/// Wraps one of the concrete `DMFTLoop<T, Q, B>` instantiations via the
/// `DmftLoopVariant` type-erased enum.
///
/// # Python usage
///
/// ```python
/// import tensorkraft as tk
///
/// config = tk.DMFTConfig(
///     n_bath=6,
///     u=4.0,
///     epsilon_imp=0.0,
///     bandwidth=4.0,
///     max_iterations=30,
/// )
/// solver = tk.DMFTLoop(config)
/// spectral = solver.solve()          # releases GIL; responds to Ctrl+C
///
/// import numpy as np
/// print(f"Sum rule: {spectral.sum_rule():.6f}")
/// print(spectral.omega.shape)
/// ```
#[pyclass(name = "DMFTLoop")]
pub struct PyDmftLoop {
    pub(crate) inner: DmftLoopVariant,
}

#[pymethods]
impl PyDmftLoop {
    /// Construct a real-valued U(1) DMFT solver (the standard single-orbital case).
    ///
    /// For complex hybridization, use `DMFTLoop.complex_u1()`.
    /// For particle-hole symmetric models with Z₂ symmetry, use `DMFTLoop.real_z2()`.
    ///
    /// # Parameters
    /// - `config`: `DMFTConfig` controlling all solver parameters.
    ///
    /// # Raises
    /// `tensorkraft.ConfigError` if any configuration field is out of range.
    #[new]
    pub fn new(config: &PyDmftConfig) -> PyResult<Self>;

    /// Construct a complex-valued U(1) DMFT solver.
    ///
    /// Use when the hybridization function Δ(ω) is complex (e.g., spin-orbit
    /// coupling or non-Hermitian baths).
    #[staticmethod]
    pub fn complex_u1(config: &PyDmftConfig) -> PyResult<Self>;

    /// Construct a real-valued Z₂ DMFT solver.
    ///
    /// Use for particle-hole symmetric models where U(1) charge conservation
    /// is broken but particle-hole Z₂ is preserved.
    #[staticmethod]
    pub fn real_z2(config: &PyDmftConfig) -> PyResult<Self>;

    /// Run the DMFT self-consistency loop until convergence or `max_iterations`.
    ///
    /// Releases the GIL for the entire computation. Responds to Ctrl+C via
    /// a monitor thread that polls `check_signals()` every 100 ms without
    /// re-acquiring the GIL from Rayon workers.
    ///
    /// The cancellation flag is checked by the underlying `DMRGEngine` at the
    /// end of each sweep step — a single `Relaxed` atomic load, effectively free.
    ///
    /// # Returns
    /// Converged `SpectralFunction` A(ω).
    ///
    /// # Raises
    /// - `KeyboardInterrupt` — Ctrl+C received; cancellation is clean (no
    ///   partially updated bath state). Maps to `DmftError::Cancelled`.
    /// - `tensorkraft.DmftConvergenceError` — loop did not converge within
    ///   `max_iterations`. Maps to `DmftError::MaxIterationsExceeded`.
    /// - `tensorkraft.BathDiscretizationError` — Lanczos tridiagonalization failed.
    /// - `tensorkraft.DmrgError` — underlying DMRG or TDVP error.
    pub fn solve(&mut self, py: Python<'_>) -> PyResult<PySpectralFunction>;

    /// Whether the most recent call to `solve()` converged within `max_iterations`.
    ///
    /// Returns `False` before `solve()` has been called.
    pub fn converged(&self) -> bool;

    /// Number of completed self-consistency iterations in the most recent `solve()` call.
    pub fn n_iterations(&self) -> usize;

    /// Current bath parameters (read-only snapshot).
    ///
    /// Returns a `BathParameters` object. The `epsilon` and `v` arrays are
    /// copies of the internal Rust data (not live views — modifying them has
    /// no effect on the solver).
    pub fn bath(&self) -> PyResult<PyBathParameters>;

    /// Per-iteration statistics from the most recent `solve()` call.
    pub fn stats(&self) -> PyDmftStats;

    /// Human-readable string representation of the solver.
    pub fn __repr__(&self) -> String;
}
```

### 5.2 `PySpectralFunction` (`SpectralFunction` in Python)

```rust
/// Python-accessible spectral function A(ω) = -Im[G(ω)] / π.
///
/// Frequency grid (`omega`) and spectral values (`values`) are exposed as
/// NumPy arrays. The underlying data is owned by Rust; the arrays are
/// constructed via `into_pyarray(py)` which transfers buffer ownership to
/// CPython's reference-counting memory manager.
///
/// # Invariant
/// `A(ω) ≥ 0` for all ω (enforced by `restore_positivity` at the end of
/// every DMFT iteration in `tk-dmft`).
///
/// # Python usage
///
/// ```python
/// spectral = solver.solve()
///
/// # NumPy access:
/// import numpy as np
/// print(spectral.omega.shape)        # (n_omega,), dtype float64
/// print(spectral.values.shape)       # (n_omega,), dtype float64
///
/// # Physics queries:
/// print(spectral.sum_rule())         # should be ≈ 1.0
/// print(spectral.value_at_fermi_level())
/// print(spectral.moment(n=1))        # first spectral moment
///
/// # TRIQS export (requires triqs feature):
/// gf = spectral.to_triqs_gf_re_freq()
/// ```
#[pyclass(name = "SpectralFunction")]
pub struct PySpectralFunction {
    inner: SpectralFunction,
}

#[pymethods]
impl PySpectralFunction {
    /// Frequency grid ω as a NumPy array of shape `(n_omega,)`, dtype `float64`.
    ///
    /// The returned array owns its buffer (cloned from the internal `omega` Vec).
    /// Its lifetime is independent of this `SpectralFunction` object.
    #[getter]
    pub fn omega<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64>;

    /// Spectral weight A(ω) as a NumPy array of shape `(n_omega,)`, dtype `float64`.
    ///
    /// Invariant: all values ≥ 0.
    #[getter]
    pub fn values<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64>;

    /// Frequency grid spacing Δω (uniform).
    #[getter]
    pub fn d_omega(&self) -> f64;

    /// Spectral sum rule: ∫A(ω)dω via the trapezoidal rule.
    ///
    /// Should be ≈ 1.0 for a single-orbital impurity after convergence.
    pub fn sum_rule(&self) -> f64;

    /// Value of A(ω) at the Fermi level (ω = 0).
    ///
    /// Interpolates linearly between the two grid points bracketing ω = 0.
    /// Proportional to the quasiparticle residue Z in Fermi liquid theory.
    ///
    /// # Raises
    /// `ValueError` if the frequency grid does not span ω = 0.
    pub fn value_at_fermi_level(&self) -> PyResult<f64>;

    /// The nth spectral moment: ∫ωⁿ A(ω)dω via the trapezoidal rule.
    ///
    /// # Parameters
    /// - `n`: moment order (0 = sum rule, 1 = first moment ≡ mean energy, etc.)
    pub fn moment(&self, n: usize) -> f64;

    /// L∞ distance ‖self - other‖_∞ between two spectral functions.
    ///
    /// # Raises
    /// `ValueError` if `other` has a different number of frequency grid points.
    pub fn max_distance(&self, other: &PySpectralFunction) -> PyResult<f64>;

    /// Export to a TRIQS `GfReFreq` object.
    ///
    /// Requires the `triqs` feature flag and a TRIQS installation in the
    /// active Python environment.
    ///
    /// # Parameters
    /// - `mesh_size` (optional int): number of frequency points in the TRIQS
    ///   mesh. Default: same as the internal `omega` array length.
    ///
    /// # Raises
    /// - `ImportError` if TRIQS is not installed in the Python environment.
    /// - `RuntimeError` if the `triqs` feature was not compiled in.
    #[cfg(feature = "triqs")]
    pub fn to_triqs_gf_re_freq(
        &self,
        py: Python<'_>,
        mesh_size: Option<usize>,
    ) -> PyResult<PyObject>;

    pub fn __repr__(&self) -> String;

    /// Number of frequency grid points.
    pub fn __len__(&self) -> usize;
}
```

**NumPy implementation note:**

```rust
impl PySpectralFunction {
    #[getter]
    pub fn omega<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        // Clone the Vec<f64>, then transfer buffer ownership to NumPy.
        // The clone is O(n_omega) words. This is acceptable because:
        //   1. n_omega is typically 2000–10000 (16–80 KB).
        //   2. Getter calls occur only during post-processing, not in hot loops.
        // See §12.3 for the trade-off analysis against pinned-view alternatives.
        self.inner.omega.clone().into_pyarray(py)
    }

    #[getter]
    pub fn values<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.inner.values.clone().into_pyarray(py)
    }
}
```

### 5.3 `PyBathParameters` (`BathParameters` in Python)

```rust
/// Python-accessible discretized bath parameters.
///
/// Provides read and write access to the bath on-site energies (ε_k) and
/// hybridization amplitudes (V_k) as NumPy arrays. Array getters return
/// copies (not live views); modifying the returned arrays does not affect
/// the solver's internal state.
///
/// # Python usage
///
/// ```python
/// bath = solver.bath()
/// print(bath.epsilon)       # np.ndarray, shape (n_bath,), dtype float64
/// print(bath.v)             # np.ndarray, shape (n_bath,), dtype float64
/// print(bath.n_bath)        # int
///
/// # Initialize from TRIQS hybridization (requires triqs feature):
/// bath.update_from_triqs_delta(delta_gf, broadening=0.05)
/// ```
#[pyclass(name = "BathParameters")]
pub struct PyBathParameters {
    inner: BathParameters<f64>,
}

#[pymethods]
impl PyBathParameters {
    /// On-site bath energies ε_k as a NumPy array of shape `(n_bath,)`, dtype `float64`.
    #[getter]
    pub fn epsilon<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64>;

    /// Set bath energies from a NumPy array.
    ///
    /// # Raises
    /// `ValueError` if `values.len() != self.n_bath`.
    #[setter]
    pub fn set_epsilon(&mut self, values: PyReadonlyArray1<f64>) -> PyResult<()>;

    /// Hybridization amplitudes V_k as a NumPy array of shape `(n_bath,)`, dtype `float64`.
    #[getter]
    pub fn v<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64>;

    /// Set hybridization amplitudes from a NumPy array.
    ///
    /// # Raises
    /// `ValueError` if `values.len() != self.n_bath`.
    #[setter]
    pub fn set_v(&mut self, values: PyReadonlyArray1<f64>) -> PyResult<()>;

    /// Number of bath sites.
    #[getter]
    pub fn n_bath(&self) -> usize;

    /// Compute the discretized hybridization function on a frequency grid.
    ///
    /// Δ(ω) = Σ_k |V_k|² / (ω - ε_k + i·broadening)
    ///
    /// # Parameters
    /// - `omega`: NumPy array of frequency grid points, shape `(n_omega,)`.
    /// - `broadening`: Lorentzian broadening δ (replaces the i0⁺ regulator).
    ///
    /// # Returns
    /// Complex NumPy array of shape `(n_omega,)`, dtype `complex128`.
    pub fn hybridization_function<'py>(
        &self,
        py: Python<'py>,
        omega: PyReadonlyArray1<f64>,
        broadening: f64,
    ) -> PyResult<&'py PyArray1<Complex<f64>>>;

    /// Update bath parameters from a TRIQS imaginary-frequency Green's function.
    ///
    /// Extracts bath parameters (ε_k, V_k) from a TRIQS `GfImFreq` object by
    /// Lanczos tridiagonalization of its hybridization function. The array data
    /// inside the TRIQS object is accessed as a NumPy view without copying.
    ///
    /// # Parameters
    /// - `delta_gf`: TRIQS `GfImFreq` object representing Δ(iω_n).
    /// - `broadening`: Lorentzian broadening for real-frequency projection.
    ///
    /// # Raises
    /// - `ImportError` if TRIQS is not installed.
    /// - `RuntimeError` if the `triqs` feature was not compiled in.
    /// - `tensorkraft.BathDiscretizationError` if Lanczos tridiagonalization fails.
    #[cfg(feature = "triqs")]
    pub fn update_from_triqs_delta(
        &mut self,
        py: Python<'_>,
        delta_gf: &PyAny,
        broadening: f64,
    ) -> PyResult<()>;

    pub fn __repr__(&self) -> String;
}
```

### 5.4 Configuration Classes

#### `PyDmftConfig` (`DMFTConfig` in Python)

```rust
/// Top-level DMFT configuration. Maps to `tk_dmft::DMFTConfig`.
/// All fields have sensible defaults.
///
/// # Python usage
///
/// ```python
/// config = tk.DMFTConfig(
///     n_bath=6,
///     u=4.0,
///     epsilon_imp=0.0,
///     bandwidth=4.0,
///     max_iterations=30,
///     self_consistency_tol=1e-4,
/// )
///
/// # Access and modify nested configs:
/// config.dmrg.max_bond_dim = 200
/// config.dmrg.max_sweeps = 10
/// config.time_evolution.t_max = 30.0
/// config.linear_prediction.prediction_order = 150
/// ```
#[pyclass(name = "DMFTConfig")]
pub struct PyDmftConfig {
    pub(crate) inner: DMFTConfig,
    /// Number of bath sites (stored separately; used in DMFTLoop constructors).
    pub(crate) n_bath: usize,
    /// Initial bath bandwidth.
    pub(crate) bandwidth: f64,
}

#[pymethods]
impl PyDmftConfig {
    /// Construct a `DMFTConfig` with the given parameters; all are keyword-only.
    ///
    /// # Parameters
    /// - `n_bath` (int): number of bath sites. Default: 6.
    /// - `u` (float): Hubbard interaction at the impurity. Default: 0.0.
    /// - `epsilon_imp` (float): impurity level energy. Default: 0.0.
    /// - `bandwidth` (float): initial bath bandwidth for uniform initialization. Default: 4.0.
    /// - `max_iterations` (int): maximum self-consistency iterations. Default: 50.
    /// - `self_consistency_tol` (float): relative hybridization convergence threshold. Default: 1e-4.
    /// - `checkpoint_path` (str | None): path for atomic checkpoint writes after each iteration. Default: None.
    ///
    /// # Raises
    /// `tensorkraft.ConfigError` if `n_bath < 1`, `bandwidth <= 0`, or `self_consistency_tol <= 0`.
    #[new]
    #[pyo3(signature = (
        n_bath = 6,
        u = 0.0,
        epsilon_imp = 0.0,
        bandwidth = 4.0,
        max_iterations = 50,
        self_consistency_tol = 1e-4,
        checkpoint_path = None,
    ))]
    pub fn new(
        n_bath: usize,
        u: f64,
        epsilon_imp: f64,
        bandwidth: f64,
        max_iterations: usize,
        self_consistency_tol: f64,
        checkpoint_path: Option<&str>,
    ) -> PyResult<Self>;

    /// DMRG sweep configuration (nested `DMRGConfig` object, mutable).
    #[getter]
    pub fn dmrg(&self) -> PyDmrgConfig;

    /// Time evolution configuration (nested `TimeEvolutionConfig` object, mutable).
    #[getter]
    pub fn time_evolution(&self) -> PyTimeEvolutionConfig;

    /// Linear prediction pipeline configuration (nested object, mutable).
    #[getter]
    pub fn linear_prediction(&self) -> PyLinearPredictionConfig;

    /// Spectral solver mode: `"tdvp"`, `"chebyshev"`, or `"adaptive"`.
    ///
    /// Default: `"adaptive"` (inspects entanglement gap at center bond;
    /// promotes Chebyshev for metallic phases, keeps TDVP for gapped phases).
    /// See architecture §8.4.1.
    #[getter]
    pub fn solver_mode(&self) -> String;

    /// # Raises
    /// `ValueError` if the string is not `"tdvp"`, `"chebyshev"`, or `"adaptive"`.
    #[setter]
    pub fn set_solver_mode(&mut self, mode: &str) -> PyResult<()>;

    /// Bath mixing scheme: `"linear"` or `"broyden"`. Default: `"broyden"`.
    #[getter]
    pub fn mixing(&self) -> String;

    /// # Raises
    /// `ValueError` for unrecognized mixing scheme names.
    #[setter]
    pub fn set_mixing(&mut self, scheme: &str) -> PyResult<()>;

    /// Maximum number of self-consistency iterations.
    #[getter] pub fn max_iterations(&self) -> usize;
    #[setter] pub fn set_max_iterations(&mut self, val: usize);

    /// Relative hybridization convergence threshold. Default: 1e-4.
    #[getter] pub fn self_consistency_tol(&self) -> f64;
    #[setter] pub fn set_self_consistency_tol(&mut self, val: f64) -> PyResult<()>;

    /// Optional checkpoint file path. `None` disables checkpointing.
    #[getter] pub fn checkpoint_path(&self) -> Option<String>;
    #[setter] pub fn set_checkpoint_path(&mut self, path: Option<&str>);

    pub fn __repr__(&self) -> String;
}
```

#### `PyDmrgConfig` (`DMRGConfig` in Python)

```rust
/// DMRG sweep configuration, nested inside `DMFTConfig`.
///
/// # Python usage
///
/// ```python
/// config.dmrg.max_bond_dim = 300
/// config.dmrg.max_sweeps = 20
/// config.dmrg.svd_cutoff = 1e-10
/// config.dmrg.energy_tol = 1e-8
/// config.dmrg.eigensolver = "davidson"
/// ```
#[pyclass(name = "DMRGConfig")]
pub struct PyDmrgConfig {
    pub(crate) inner: DMRGConfig,
}

#[pymethods]
impl PyDmrgConfig {
    /// Maximum MPS bond dimension. Default: 200.
    #[getter] pub fn max_bond_dim(&self) -> usize;
    #[setter] pub fn set_max_bond_dim(&mut self, val: usize);

    /// Absolute singular value cutoff for SVD truncation. Default: 1e-10.
    #[getter] pub fn svd_cutoff(&self) -> f64;
    #[setter] pub fn set_svd_cutoff(&mut self, val: f64);

    /// Maximum number of DMRG sweeps per DMFT iteration. Default: 30.
    #[getter] pub fn max_sweeps(&self) -> usize;
    #[setter] pub fn set_max_sweeps(&mut self, val: usize);

    /// Energy convergence tolerance across consecutive sweeps. Default: 1e-8.
    #[getter] pub fn energy_tol(&self) -> f64;
    #[setter] pub fn set_energy_tol(&mut self, val: f64);

    /// Iterative eigensolver: `"lanczos"`, `"davidson"`, or `"block_davidson"`.
    /// Default: `"davidson"`.
    #[getter] pub fn eigensolver(&self) -> String;

    /// # Raises
    /// `ValueError` for unrecognized eigensolver names.
    #[setter] pub fn set_eigensolver(&mut self, name: &str) -> PyResult<()>;

    /// Optional directory path for environment block disk offload.
    /// `None` = full in-memory caching (default). Set to a path for
    /// memory-constrained nodes (uses `memmap2`-backed files).
    #[getter] pub fn environment_offload(&self) -> Option<String>;
    #[setter] pub fn set_environment_offload(&mut self, path: Option<&str>);

    pub fn __repr__(&self) -> String;
}
```

#### `PyTimeEvolutionConfig` (`TimeEvolutionConfig` in Python)

```rust
/// TDVP time evolution and Chebyshev cross-validation configuration.
///
/// # Python usage
///
/// ```python
/// config.time_evolution.t_max = 30.0
/// config.time_evolution.dt = 0.05
/// config.time_evolution.max_bond_dim = 400
/// config.time_evolution.adaptive_tikhonov = True
/// ```
#[pyclass(name = "TimeEvolutionConfig")]
pub struct PyTimeEvolutionConfig {
    pub(crate) inner: TimeEvolutionConfig,
}

#[pymethods]
impl PyTimeEvolutionConfig {
    /// Total TDVP simulation time t_max (inverse energy units). Default: 20.0.
    #[getter] pub fn t_max(&self) -> f64;
    #[setter] pub fn set_t_max(&mut self, val: f64);

    /// Physical time step dt. Default: 0.05.
    #[getter] pub fn dt(&self) -> f64;
    #[setter] pub fn set_dt(&mut self, val: f64);

    /// Maximum MPS bond dimension during TDVP time evolution. Default: 500.
    #[getter] pub fn max_bond_dim(&self) -> usize;
    #[setter] pub fn set_max_bond_dim(&mut self, val: usize);

    /// Relative L∞ tolerance for TDVP/Chebyshev cross-validation. Default: 0.05 (5%).
    ///
    /// If ‖A_primary − A_cross‖_∞ / ‖A_cross‖_∞ exceeds this, a
    /// `SPECTRAL_CROSS_VALIDATION_WARNING` log event is emitted.
    #[getter] pub fn cross_validation_tol(&self) -> f64;
    #[setter] pub fn set_cross_validation_tol(&mut self, val: f64);

    /// Static Tikhonov regularization δ for TDVP bond inversion. Default: 1e-10.
    ///
    /// Used as the minimum floor when `adaptive_tikhonov = True`.
    #[getter] pub fn tikhonov_delta(&self) -> f64;
    #[setter] pub fn set_tikhonov_delta(&mut self, val: f64);

    /// Whether adaptive Tikhonov scaling is enabled. Default: `True`.
    ///
    /// When `True`, δ is dynamically scaled to `max(tikhonov_delta, scale × σ_discarded_max)`
    /// per bond, preventing over-regularization in near-product-state bonds.
    #[getter] pub fn adaptive_tikhonov(&self) -> bool;
    #[setter] pub fn set_adaptive_tikhonov(&mut self, val: bool);

    /// Number of subspace expansion vectors per TDVP step. Default: 4.
    #[getter] pub fn expansion_vectors(&self) -> usize;
    #[setter] pub fn set_expansion_vectors(&mut self, val: usize);

    /// Soft D_max overshoot factor. Default: 1.1 (allow 10% bond-dimension overshoot).
    #[getter] pub fn soft_dmax_factor(&self) -> f64;
    #[setter] pub fn set_soft_dmax_factor(&mut self, val: f64);

    /// Physical time constant for soft D_max exponential decay. Default: 5.0.
    ///
    /// Expressed in the same physical time units as `dt`. The decay is invariant
    /// to adaptive time-stepping (see architecture §8.1.1).
    #[getter] pub fn dmax_decay_time(&self) -> f64;
    #[setter] pub fn set_dmax_decay_time(&mut self, val: f64);

    pub fn __repr__(&self) -> String;
}
```

#### `PyLinearPredictionConfig` (`LinearPredictionConfig` in Python)

```rust
/// Linear prediction pipeline configuration.
///
/// # Python usage
///
/// ```python
/// config.linear_prediction.prediction_order = 150
/// config.linear_prediction.extrapolation_factor = 5.0
/// config.linear_prediction.broadening_eta = 0.0     # 0.0 disables windowing
/// config.linear_prediction.toeplitz_solver = "levinson_durbin"
/// ```
#[pyclass(name = "LinearPredictionConfig")]
pub struct PyLinearPredictionConfig {
    pub(crate) inner: LinearPredictionConfig,
}

#[pymethods]
impl PyLinearPredictionConfig {
    /// Prediction order P (number of past time points used). Default: 100.
    #[getter] pub fn prediction_order(&self) -> usize;
    #[setter] pub fn set_prediction_order(&mut self, val: usize);

    /// Extrapolation factor: extend G(t) to `t_max × factor`. Default: 4.0.
    #[getter] pub fn extrapolation_factor(&self) -> f64;
    #[setter] pub fn set_extrapolation_factor(&mut self, val: f64);

    /// Exponential broadening η. Default: 0.0 (disabled for gapped phases).
    ///
    /// Set η > 0 for metallic phases to enforce G(t) → 0 at large t.
    /// When η > 0, Lorentzian deconvolution is automatically applied.
    /// When η = 0, deconvolution is skipped.
    #[getter] pub fn broadening_eta(&self) -> f64;
    #[setter] pub fn set_broadening_eta(&mut self, val: f64);

    /// Tikhonov δ for regularized Lorentzian deconvolution. Default: 1e-3.
    ///
    /// Bounds high-frequency noise amplification to 1/δ after deconvolution.
    #[getter] pub fn deconv_tikhonov_delta(&self) -> f64;
    #[setter] pub fn set_deconv_tikhonov_delta(&mut self, val: f64);

    /// Hard cutoff frequency for deconvolution (in units of bandwidth). Default: 10.0.
    #[getter] pub fn deconv_omega_max(&self) -> f64;
    #[setter] pub fn set_deconv_omega_max(&mut self, val: f64);

    /// Noise floor for spectral positivity clamping. Default: 1e-15.
    #[getter] pub fn positivity_floor(&self) -> f64;
    #[setter] pub fn set_positivity_floor(&mut self, val: f64);

    /// Warning threshold for negative spectral weight fraction. Default: 0.05 (5%).
    #[getter] pub fn positivity_warning_threshold(&self) -> f64;
    #[setter] pub fn set_positivity_warning_threshold(&mut self, val: f64);

    /// Fermi-level distortion tolerance. Default: 0.01 (1%).
    ///
    /// If A(ω=0) shifts by more than this relative fraction after
    /// positivity restoration, a `FERMI_LEVEL_DISTORTION` warning is emitted.
    #[getter] pub fn fermi_level_shift_tolerance(&self) -> f64;
    #[setter] pub fn set_fermi_level_shift_tolerance(&mut self, val: f64);

    /// Toeplitz prediction solver: `"levinson_durbin"` (default) or `"svd"`.
    ///
    /// `"levinson_durbin"` uses O(P²) Levinson-Durbin recursion with Tikhonov
    /// regularization. `"svd"` uses O(P³) SVD pseudo-inverse (fallback for
    /// non-Toeplitz prediction matrices).
    #[getter] pub fn toeplitz_solver(&self) -> String;

    /// # Raises
    /// `ValueError` for unrecognized solver names.
    #[setter] pub fn set_toeplitz_solver(&mut self, name: &str) -> PyResult<()>;

    pub fn __repr__(&self) -> String;
}
```

### 5.5 `PyDmftStats` (`DMFTStats` in Python)

```rust
/// Per-iteration DMFT statistics.
///
/// All fields are Python lists of scalars, one entry per completed
/// self-consistency iteration. Access after `solver.solve()` returns.
///
/// # Python usage
///
/// ```python
/// stats = solver.stats()
/// import matplotlib.pyplot as plt
/// plt.semilogy(stats.hybridization_distances)  # convergence curve
/// plt.title("DMFT self-consistency convergence")
/// print(f"Chebyshev was primary: {stats.chebyshev_was_primary}")
/// ```
#[pyclass(name = "DMFTStats")]
pub struct PyDmftStats {
    inner: DMFTStats,
}

#[pymethods]
impl PyDmftStats {
    /// DMRG ground-state energies at each iteration.
    #[getter] pub fn ground_state_energies(&self) -> Vec<f64>;

    /// Relative hybridization distance ‖Δ_new − Δ_old‖_∞ / ‖Δ_old‖_∞ per iteration.
    ///
    /// Convergence is declared when this drops below `DMFTConfig.self_consistency_tol`.
    #[getter] pub fn hybridization_distances(&self) -> Vec<f64>;

    /// Spectral sum rule ∫A(ω)dω per iteration. Should stay ≈ 1.0.
    #[getter] pub fn spectral_sum_rules(&self) -> Vec<f64>;

    /// Fraction of spectral weight clamped by positivity restoration per iteration.
    ///
    /// Values persistently above 0.05 indicate the linear prediction parameters
    /// need tuning.
    #[getter] pub fn positivity_clamped_fractions(&self) -> Vec<f64>;

    /// Whether Chebyshev expansion was the primary spectral solver at each iteration.
    ///
    /// In `Adaptive` mode, this reflects the entanglement gap decision at each step.
    #[getter] pub fn chebyshev_was_primary(&self) -> Vec<bool>;

    /// Wall-clock time in seconds per iteration (DMRG + TDVP + Chebyshev combined).
    #[getter] pub fn iteration_times_secs(&self) -> Vec<f64>;

    pub fn __repr__(&self) -> String;
}
```

---

## 6. TRIQS Integration

### 6.1 Feature Gate

TRIQS interop is controlled by the `triqs` Cargo feature flag. It is NOT compiled by default and must NOT appear in PyPI wheels.

```toml
[features]
triqs = []   # no additional Rust crate dependencies; uses runtime Python object inspection
```

When `triqs` is active, the following capabilities are added:
- `PyBathParameters::update_from_triqs_delta(delta_gf, broadening)` — extract (ε_k, V_k) from a TRIQS `GfImFreq`.
- `PySpectralFunction::to_triqs_gf_re_freq(mesh_size)` — export A(ω) as a TRIQS `GfReFreq`.
- `PyDmftLoop` is otherwise unchanged; TRIQS integration occurs at the bath parameter level.

### 6.2 Zero-Copy TRIQS/NumPy Interop

TRIQS Green's function objects expose their data as NumPy arrays via `GfImFreq.data` (C-contiguous `complex128`, shape `(n_freq, n_orb, n_orb)`). The interop layer extracts this array as a `PyReadonlyArray3<Complex<f64>>` — a zero-copy view:

```rust
/// Extract the imaginary-frequency hybridization function from a TRIQS GfImFreq.
///
/// # Protocol (zero-copy)
/// 1. Verify `delta_gf.__class__.__name__ == "GfImFreq"` via `getattr`.
/// 2. Extract `delta_gf.mesh.beta` (f64) for inverse temperature.
/// 3. Extract `delta_gf.data` as `PyReadonlyArray3<Complex<f64>>` (zero-copy view).
///    For single-orbital, `data[:, 0, 0]` gives Δ(iω_n) as a 1D view.
/// 4. Extract Matsubara frequencies `[m.value for m in delta_gf.mesh]` (f64 list).
/// 5. Return `(iω_n_vec, delta_values)` for bath discretization.
///
/// No element-wise copy of the `data` array occurs.
///
/// # Errors
/// Returns `PyErr` (TypeError) if `delta_gf` is not a `GfImFreq`.
/// Returns `PyErr` (ImportError) if TRIQS attributes are missing.
#[cfg(feature = "triqs")]
pub(crate) fn extract_gf_imfreq(
    py: Python<'_>,
    delta_gf: &PyAny,
) -> PyResult<(Vec<f64>, Vec<Complex<f64>>)>;
```

---

## 7. Monomorphization Control (Architecture §5.4)

`tk-python` is the sole crate in the workspace that determines which concrete `<T, Q, B>` combinations are compiled into the binary. The `DmftLoopVariant` enum enumerates exactly three:

| Variant | T | Q | B | Compiled |
|:--------|:---|:---|:---|:---------|
| `RealU1` | `f64` | `U1` | `DefaultDevice` | Always |
| `ComplexU1` | `Complex<f64>` | `U1` | `DefaultDevice` | Always |
| `RealZ2` | `f64` | `Z2` | `DefaultDevice` | Always |

All three variants are compiled regardless of feature flags, because Python users cannot opt out of type combinations at build time. `DefaultDevice` is a single concrete type (resolved at compile time by the active backend flag), so each variant generates exactly one machine-code path. For the PyPI wheel (`backend-faer` default), the binary contains three copies of the DMFT stack — not the theoretical maximum of 36 (3 scalars × 3 symmetries × 4 backends).

**Adding a new combination:**
1. Add one enum variant to `DmftLoopVariant`.
2. Add one match arm to `dispatch_variant!{}`.
3. Add one `#[staticmethod]` constructor to `PyDmftLoop`.

No changes to `tk-dmrg`, `tk-dmft`, or any other upstream crate are needed.

**Compile-time monitoring:** If any single `tk-python` compile step exceeds 60 seconds in release mode (architecture §5.4), `cargo-llvm-lines` is used to identify the largest generic expansions for targeted mitigation.

---

## 8. Error Handling

### 8.1 Error Enum

`tk-python` does not define its own Rust error enum. All domain errors originate as `DmftError` from `tk-dmft` (which transitively wraps `DmrgError` from `tk-dmrg`). The `PythonError` newtype in `error.rs` serves as the single conversion bridge from `DmftError` to PyO3's `PyErr`.

### 8.2 Result Type Alias

`tk-python` does not define a crate-level `Result` type alias. All fallible `#[pymethods]` return PyO3's `PyResult<T>` (i.e., `Result<T, PyErr>`) directly. The conversion from `DmftError` to `PyErr` is performed via `PythonError` at each call site.

### 8.3 Python Exception Hierarchy

```
tensorkraft.TensorkraftError  (base; subclass of Exception)
├── tensorkraft.DmftConvergenceError    ← DmftError::MaxIterationsExceeded
├── tensorkraft.BathDiscretizationError ← DmftError::BathDiscretizationFailed
│                                          DmftError::InvalidHybridizationFunction
├── tensorkraft.SpectralError           ← DmftError::DeconvolutionFailed
│                                          DmftError::ChebyshevBandwidthError
│                                          DmftError::SumRuleViolated
├── tensorkraft.LinearPredictionError   ← DmftError::LinearPredictionFailed
├── tensorkraft.DmrgError               ← DmftError::Dmrg(DmrgError::*)
├── tensorkraft.CheckpointError         ← DmftError::CheckpointIo
│                                          DmftError::CheckpointDeser
└── tensorkraft.ConfigError             ← validation failures in Python config classes

KeyboardInterrupt                        ← DmftError::Cancelled  (call-site conversion)
```

### 8.4 `PythonError` Conversion Type

```rust
/// Conversion bridge from `DmftError` to `PyErr`.
///
/// This is the single point of truth for Rust → Python error translation.
/// Never construct `PyErr` directly from a `DmftError` in `#[pymethods]`
/// bodies. `DmftError::Cancelled` must NOT be routed through this type;
/// it is converted to `KeyboardInterrupt` at the call site.
pub(crate) struct PythonError(DmftError);

impl From<DmftError> for PythonError {
    fn from(e: DmftError) -> Self { PythonError(e) }
}

impl From<PythonError> for PyErr {
    fn from(e: PythonError) -> PyErr {
        match e.0 {
            DmftError::MaxIterationsExceeded { iterations, distance, threshold } => {
                PyErr::new::<DmftConvergenceError, _>(format!(
                    "DMFT did not converge after {} iterations \
                     (distance = {:.2e}, threshold = {:.2e})",
                    iterations, distance, threshold
                ))
            }
            DmftError::BathDiscretizationFailed { max_steps, residual } => {
                PyErr::new::<BathDiscretizationError, _>(format!(
                    "Bath discretization failed: Lanczos did not converge in {} steps \
                     (residual = {:.2e}). Consider increasing n_bath or lanczos_tol.",
                    max_steps, residual
                ))
            }
            DmftError::InvalidHybridizationFunction { n_negative } => {
                PyErr::new::<BathDiscretizationError, _>(format!(
                    "Invalid hybridization function: -Im[Δ(ω)] < 0 at {} frequency points. \
                     The hybridization function must have non-negative imaginary part.",
                    n_negative
                ))
            }
            DmftError::LinearPredictionFailed { condition } => {
                PyErr::new::<LinearPredictionError, _>(format!(
                    "Linear prediction failed: Levinson-Durbin condition number = {:.2e}. \
                     Increase LinearPredictionConfig.toeplitz_solver tikhonov_lambda.",
                    condition
                ))
            }
            DmftError::Dmrg(inner) => {
                PyErr::new::<DmrgError, _>(format!("DMRG error: {}", inner))
            }
            DmftError::CheckpointIo(e) => {
                PyErr::new::<CheckpointError, _>(format!("Checkpoint I/O error: {}", e))
            }
            DmftError::CheckpointDeser(msg) => {
                PyErr::new::<CheckpointError, _>(
                    format!("Checkpoint deserialization failed: {}", msg)
                )
            }
            DmftError::Cancelled => {
                // This arm must never be reached. DmftError::Cancelled is
                // intercepted at the call site and converted to KeyboardInterrupt.
                unreachable!(
                    "DmftError::Cancelled must be converted to KeyboardInterrupt \
                     at the call site, not routed through PythonError."
                )
            }
            other => {
                PyErr::new::<SpectralError, _>(format!("{}", other))
            }
        }
    }
}
```

### 8.5 Error Propagation Strategy

All errors in `tk-python` originate from the upstream `tk-dmft` crate as `DmftError` variants. The propagation path is:

1. **Rust → `PythonError`**: Each `#[pymethods]` function calls into `tk-dmft` and receives a `Result<T, DmftError>`. On `Err`, the `DmftError` is converted to `PythonError` via `From<DmftError>`.
2. **`PythonError` → `PyErr`**: The `From<PythonError> for PyErr` implementation maps each `DmftError` variant to the appropriate Python exception class in the `tensorkraft.*` hierarchy.
3. **Special case — `DmftError::Cancelled`**: This variant is intercepted at the call site (in `solve()`) and converted directly to `KeyboardInterrupt`. It must never reach `PythonError`.
4. **Configuration validation errors**: `PyDmftConfig::new()` and `#[setter]` methods perform range checks and return `ConfigError` directly as `PyErr`, without going through `PythonError`.

No `#[from]` derives are used because `PythonError` is a simple newtype wrapper, not a `thiserror`-derived enum. The conversion is explicit and centralized in `error.rs`.

---

## 9. Public API Surface

```rust
/// The top-level Python module `tensorkraft`.
///
/// # Python usage
///
/// ```python
/// import tensorkraft as tk
/// print(tk.__version__)   # e.g., "0.1.0"
/// help(tk.DMFTLoop)
/// ```
///
/// Registers all public `#[pyclass]` types and exception subclasses.
/// The module version is set from the workspace `Cargo.toml` version field
/// via `env!("CARGO_PKG_VERSION")`.
#[pymodule]
fn tensorkraft(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Register classes
    m.add_class::<PyDmftLoop>()?;
    m.add_class::<PySpectralFunction>()?;
    m.add_class::<PyBathParameters>()?;
    m.add_class::<PyDmftConfig>()?;
    m.add_class::<PyDmrgConfig>()?;
    m.add_class::<PyTimeEvolutionConfig>()?;
    m.add_class::<PyLinearPredictionConfig>()?;
    m.add_class::<PyDmftStats>()?;

    // Register exception hierarchy (base → derived)
    let base = py.get_type::<pyo3::exceptions::PyException>();
    let tensorkraft_error = PyErr::new_type(
        py,
        "tensorkraft.TensorkraftError",
        Some(base),
        None,
    )?;
    m.add("TensorkraftError", &tensorkraft_error)?;

    macro_rules! add_exception {
        ($name:literal, $parent:expr) => {{
            let exc = PyErr::new_type(py, concat!("tensorkraft.", $name), Some($parent), None)?;
            m.add($name, &exc)?;
            exc
        }};
    }
    let dmft_convergence  = add_exception!("DmftConvergenceError",    &tensorkraft_error);
    let _bath_disc        = add_exception!("BathDiscretizationError", &tensorkraft_error);
    let _spectral         = add_exception!("SpectralError",           &tensorkraft_error);
    let _linear_pred      = add_exception!("LinearPredictionError",   &tensorkraft_error);
    let _dmrg             = add_exception!("DmrgError",               &tensorkraft_error);
    let _checkpoint       = add_exception!("CheckpointError",         &tensorkraft_error);
    let _config           = add_exception!("ConfigError",             &tensorkraft_error);
    let _ = (dmft_convergence,); // suppress unused-variable warning

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
```

The `lib.rs` file contains the following module declarations and re-exports:

```rust
pub mod dispatch;
pub mod dmft;
pub mod spectral;
pub mod bath;
pub mod config;
pub(crate) mod monitor;
pub(crate) mod error;

// The #[pymodule] function above is the sole public entry point.
// All #[pyclass] types are registered via m.add_class::<T>() rather
// than pub use re-exports, because PyO3 modules use runtime registration.
```

---

## 10. Feature Flags

| Flag | Effect in `tk-python` |
|:-----|:----------------------|
| `backend-faer` | Enables `DeviceFaer` as `DefaultDevice`; pure Rust, no system BLAS. Enabled by default and used in PyPI wheels. |
| `backend-oxiblas` | Enables `DeviceOxiblas` for sparse BSR/CSR operations. Enabled by default and used in PyPI wheels. |
| `backend-mkl` | Links Intel MKL; sets `DefaultDevice = DeviceMKL`. Build from source only; not included in PyPI wheels. |
| `backend-openblas` | Links OpenBLAS; sets `DefaultDevice = DeviceOpenBLAS`. Build from source only; not included in PyPI wheels. |
| `backend-cuda` | Enables `DeviceCuda` (requires CUDA toolkit). Build from source only; not included in PyPI wheels. |
| `backend-mpi` | Propagates MPI support from `tk-dmft`. Build from source only; not included in PyPI wheels. |
| `su2-symmetry` | Propagates SU(2) symmetry into the underlying stack. No new Python API in Phase 4; not included in PyPI wheels. |
| `parallel` | Rayon parallelism (propagated from `tk-linalg` via `tk-dmft`). Enabled by default and used in PyPI wheels. |
| `triqs` | TRIQS `GfImFreq`/`GfReFreq` interop via runtime Python object inspection. Not included in PyPI wheels. |
| `python-bindings` | Sentinel flag; activates `#[pymodule]` and `extension-module` in `pyo3`. Enabled by default and used in PyPI wheels. |

**PyPI wheel constraint (architecture §2.3):** Pre-built wheels use `default-features = true` which resolves to `{backend-faer, backend-oxiblas, parallel, python-bindings}`. No FFI BLAS symbols are linked. Users requiring MKL, OpenBLAS, CUDA, or TRIQS must build from source:

```sh
# Example: build with MKL and TRIQS
pip install maturin
maturin build --release --features backend-mkl,triqs
```

---

## 11. Build-Level Concerns

`tk-python/build.rs` performs three checks:

**1. FFI backend mutual exclusivity (defense-in-depth):**

```rust
// tk-python/build.rs
#[cfg(all(feature = "backend-mkl", feature = "backend-openblas"))]
compile_error!(
    "Features `backend-mkl` and `backend-openblas` are mutually exclusive in tk-python. \
     Both expose global BLAS symbols and cause linker collisions. \
     The canonical enforcement is in tk-linalg/build.rs; this check surfaces \
     the error closer to the build root."
);
```

**2. CUDA + Python GIL advisory warning:**

When `backend-cuda` is active, emit a `cargo:warning` reminding developers that CUDA kernel-launch calls must not occur on PyO3-managed threads. The `CancellationMonitor` design already prevents this (Rayon workers never touch the GIL); the warning is documentation in build output.

**3. TRIQS detection (informational):**

When the `triqs` feature is enabled, probe for a TRIQS installation via `python -c "import triqs"`. Emit a `cargo:warning` (not `compile_error!`) if not found, because TRIQS interop uses runtime Python object inspection — the crate compiles correctly even without TRIQS installed; the `ImportError` is raised at Python runtime.

**maturin `pyproject.toml`:**

```toml
[build-system]
requires = ["maturin>=1.4,<2"]
build-backend = "maturin"

[project]
name = "tensorkraft"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
]

[tool.maturin]
features    = ["python-bindings"]
python-source = "python"   # optional hand-written __init__.py and stubs
bindings    = "pyo3"
strip       = true          # strip debug symbols in release wheels
```

The `abi3-py38` tag on the `pyo3` dependency (`pyo3 = { version = "0.21", features = ["extension-module", "abi3-py38"] }`) ensures a single wheel file supports Python 3.8 through 3.x without recompilation.

---

## 12. Data Structures and Internal Representations

### 12.1 `DmftLoopVariant` Ownership and Thread Safety

`DmftLoopVariant` is stored directly inside `PyDmftLoop` (not behind `Box` or `Arc`). PyO3's `PyCell<PyDmftLoop>` manages mutation exclusivity: `solve()` takes `&mut self`, so concurrent Python-thread access raises `PyRuntimeError("Already borrowed")` before reaching Rust. No additional `Mutex` is needed.

This means a `PyDmftLoop` instance is inherently single-threaded from Python's perspective. Multiple independent DMFT loops require multiple `DMFTLoop(config)` instances — one per Python thread or `concurrent.futures` executor.

### 12.2 `CancellationMonitor` Lifecycle

The `CancellationMonitor` is constructed at the beginning of each `solve()` call and consumed by `shutdown()` inside `allow_threads`. It is not stored in `PyDmftLoop` between calls. A fresh monitor (new `AtomicBool`, new `mpsc::channel`, new thread) is created for each invocation. This avoids stale state from a previously interrupted run.

### 12.3 NumPy Memory Ownership Model

When `into_pyarray(py)` is called on a `Vec<f64>`, buffer ownership transfers to a Python `ndarray` object managed by CPython's reference counter. The `SpectralFunction` inside `PySpectralFunction` retains its own copy (via the clone in the getter), so dropping `PySpectralFunction` does not invalidate NumPy arrays previously returned to Python.

**Clone-vs-view trade-off:** The clone-on-getter strategy is chosen over a pinned-view approach because:
- Array sizes are small: `n_omega` ≤ 10,000 → ≤ 80 KB per clone.
- Pinned-view (`PyArray1::borrow_from_array`) requires the `PySpectralFunction` to be pinned (e.g., stored in an `Arc`), complicating the ownership model.
- Getter calls occur in post-processing, not in hot computation loops.

If a future benchmark shows clone overhead is material (e.g., for n_omega = 10⁶ output grids), the implementation can adopt `Arc<SpectralFunction>` + borrowed views without changing the Python API.

---

## 13. Dependencies and Integration

### 13.1 Upstream Dependencies (`Cargo.toml`)

```toml
[dependencies]
tk-dmft  = { path = "../tk-dmft", version = "0.1.0" }
# tk-dmft re-exports the full dependency chain:
#   tk-dmrg, tk-core, tk-symmetry, tk-linalg, tk-contract, tk-dsl

pyo3         = { version = "0.21", features = ["extension-module", "abi3-py38"] }
numpy        = "0.21"      # rust-numpy; zero-copy NumPy array construction
num-complex  = "0.4"
thiserror    = "1"
log          = "0.4"
tracing      = "0.1"

[dev-dependencies]
pyo3         = { version = "0.21", features = ["auto-initialize"] }  # for test harness
criterion    = { version = "0.5", optional = true }

[features]
default          = ["backend-faer", "backend-oxiblas", "parallel", "python-bindings"]
python-bindings  = ["pyo3/extension-module"]
backend-faer     = ["tk-dmft/backend-faer"]
backend-oxiblas  = ["tk-dmft/backend-oxiblas"]
backend-mkl      = ["tk-dmft/backend-mkl"]
backend-openblas = ["tk-dmft/backend-openblas"]
backend-cuda     = ["tk-dmft/backend-cuda"]
backend-mpi      = ["tk-dmft/backend-mpi"]
su2-symmetry     = ["tk-dmft/su2-symmetry"]
parallel         = ["tk-dmft/parallel"]
triqs            = []
```

### 13.2 Downstream Consumers

`tk-python` is the final leaf of the workspace dependency graph. No other Rust crate depends on it. It is consumed by:

- **Python application code** — via `import tensorkraft`.
- **TRIQS workflows** — when the `triqs` feature is enabled.
- **Jupyter notebooks** — the primary interactive target; the GIL release design is specifically motivated by Jupyter's single-threaded event loop.
- **MPI Mode B application scripts** — each MPI rank imports `tensorkraft` independently and runs `DMFTLoop.solve()` on a different orbital or k-point sector. `tk-python` does not expose MPI primitives; orchestration uses `mpi4py`.

### 13.3 External Dependencies by Functionality

| Crate / Package | Purpose | Feature gate |
|:----------------|:--------|:-------------|
| `pyo3` 0.21 | PyO3 extension-module framework; `#[pyclass]`, `#[pymethods]`, GIL management | `python-bindings` |
| `numpy` (rust-numpy) 0.21 | Zero-copy NumPy array construction via `into_pyarray` | always |
| `maturin` (build tool) | Wheel packaging and `pip install` support | build only |
| `tk-dmft` | All physics: DMRG, TDVP, Chebyshev, DMFT self-consistency loop, bath discretization | always |
| TRIQS Python library | `GfImFreq`/`GfReFreq` Green's function object protocol (runtime inspection) | `triqs` (runtime) |

---

## 14. Testing Strategy

### 14.1 Unit Tests

| Test | Description |
|:-----|:------------|
| `dispatch_variant_real_u1` | Construct `PyDmftLoop::new(config)` (RealU1 variant). Verify `inner` matches `DmftLoopVariant::RealU1`. Verify `__repr__` contains `"DMFTLoop(RealU1)"`. |
| `dispatch_variant_complex_u1` | Construct via `DMFTLoop.complex_u1(config)`. Verify `DmftLoopVariant::ComplexU1` is selected. |
| `dispatch_variant_real_z2` | Construct via `DMFTLoop.real_z2(config)`. Verify `DmftLoopVariant::RealZ2` is selected. |
| `dispatch_macro_n_iterations` | Call `dispatch_variant!(inner, n_iterations())` for each variant in a fresh solver. Verify 0 is returned in all three cases. |
| `config_defaults_match_rust` | Construct `PyDmftConfig::new(6, 0.0, ...)` with all defaults. Verify all nested fields match the values in `DMFTConfig::default()` from `tk-dmft`. |
| `config_nested_mutation_propagates` | Set `config.dmrg.max_bond_dim = 300`. Construct `PyDmftLoop`. Verify the underlying `DMFTConfig.dmrg_config.max_bond_dim == 300`. |
| `config_solver_mode_roundtrip` | Set `solver_mode = "chebyshev"`. Read back and assert `"chebyshev"`. Repeat for `"tdvp"` and `"adaptive"`. |
| `config_solver_mode_invalid` | Pass `solver_mode = "nonexistent"`. Verify `ValueError` is raised. |
| `config_eigensolver_roundtrip` | Set `config.dmrg.eigensolver = "lanczos"`. Read back and assert. Verify `"bad_solver"` raises `ValueError`. |
| `config_toeplitz_solver_roundtrip` | Set `config.linear_prediction.toeplitz_solver = "svd"`. Read back and assert. Verify `"bad_solver"` raises `ValueError`. |
| `config_mixing_roundtrip` | Set `mixing = "linear"`. Read back. Verify `"bad_scheme"` raises `ValueError`. |
| `spectral_function_numpy_dtype` | Construct `PySpectralFunction`. Verify `omega.dtype == np.float64` and `values.dtype == np.float64`. |
| `spectral_function_numpy_shape` | Verify `omega.shape == (n_omega,)` and `values.shape == (n_omega,)`. |
| `spectral_function_numpy_contiguous` | Verify `omega.flags['C_CONTIGUOUS']` and `values.flags['C_CONTIGUOUS']` are True. |
| `spectral_function_sum_rule` | Construct a Lorentzian A(ω) = (η/π)/(ω²+η²) with known integral 1.0. Verify `sum_rule()` matches within 1e-4. |
| `spectral_function_fermi_level` | Construct a Lorentzian centered at ω = 0. Verify `value_at_fermi_level()` matches the peak value within 1%. |
| `spectral_function_fermi_level_no_span` | Construct a spectral function with ω ∈ [1.0, 5.0] (does not span zero). Verify `ValueError` is raised. |
| `spectral_function_moment_n0` | Verify `moment(0)` equals `sum_rule()` within 1e-12. |
| `spectral_function_moment_n1` | Verify `moment(1)` matches the analytic first moment of a known Lorentzian. |
| `spectral_function_max_distance_identity` | Verify `sf.max_distance(sf) == 0.0`. |
| `spectral_function_max_distance_shape_mismatch` | Two spectral functions with different grid lengths. Verify `ValueError`. |
| `bath_parameters_getters` | Construct `PyBathParameters` from known `BathParameters`. Verify `epsilon.shape`, `v.shape`, and `n_bath`. |
| `bath_parameters_set_epsilon_valid` | Call `set_epsilon` with correct-length array. Verify internal `BathParameters.epsilon` is updated. |
| `bath_parameters_set_epsilon_wrong_size` | Pass array of length `n_bath + 1`. Verify `ValueError`. |
| `bath_parameters_set_v_wrong_size` | Pass array of length `n_bath - 1`. Verify `ValueError`. |
| `bath_parameters_hybridization_function_analytic` | 2-site bath with known ε_k, V_k. Verify `‖Δ_computed - Δ_exact‖_∞ < 1e-12`. |
| `dmft_stats_all_empty_before_solve` | Fresh `PyDmftLoop`. Verify all `stats()` list fields are empty. |
| `dmft_stats_lengths_match_n_iterations` | (requires a minimal solve) Verify all `stats()` list fields have length equal to `n_iterations()`. |
| `python_error_convergence` | Convert `DmftError::MaxIterationsExceeded { iterations: 50, distance: 1e-3, threshold: 1e-4 }`. Verify `DmftConvergenceError` is raised with correct message. |
| `python_error_bath_discretization` | Verify `BathDiscretizationError` for `DmftError::BathDiscretizationFailed`. |
| `python_error_dmrg` | Verify `DmrgError` Python exception wraps `DmftError::Dmrg(...)`. |
| `python_error_cancelled_is_unreachable` | Confirm the `DmftError::Cancelled` arm panics with `unreachable!` when reached via `PythonError`. (Call-site correctness: `Cancelled` must never reach `PythonError`.) |
| `dmft_loop_initial_state` | Verify `converged() == False` and `n_iterations() == 0` before `solve()`. |

### 14.2 GIL Correctness Tests

These tests verify the critical invariants of the GIL release and monitor thread protocol. They run the Rust code under a Python interpreter via `pyo3::Python::with_gil`.

| Test | Description |
|:-----|:------------|
| `monitor_spawn_and_shutdown_no_deadlock` | Spawn `CancellationMonitor`. Immediately call `shutdown()` inside a simulated `allow_threads` context. Verify join completes within 200 ms. |
| `monitor_shutdown_outside_allow_threads_deadlocks` | (skipped in standard CI; documented) Confirm that calling `shutdown()` while holding the GIL would deadlock, validating the correctness requirement in the spec. |
| `monitor_cancel_flag_on_simulated_sigint` | Override `check_signals` behavior to return `Err` on first call. Spawn monitor. Verify `cancel_flag` becomes `true` within 300 ms. |
| `monitor_exits_on_done_tx_drop` | Spawn monitor. Drop `done_tx` without sending. Verify monitor thread joins within 300 ms (Disconnected exit path). |
| `solve_releases_gil_verifiable` | Start `solver.solve()` (minimal n_bath=2, max_sweeps=1). While running, acquire the GIL from a separate OS thread and execute a Python `pass` statement. Assert no deadlock within 5 seconds. This proves the GIL is released. |
| `solve_cancelled_raises_keyboard_interrupt` | Configure a minimal solver. Register a Ctrl+C emulation via `signal.raise_signal(signal.SIGINT)` after 50 ms. Verify `KeyboardInterrupt` is raised and `solve()` returns within 500 ms. |
| `solve_panic_propagates_cleanly` | Inject a panic into the `RealU1` solver's first sweep. Verify the panic propagates as `PanicException` (not a segfault or hang) and the monitor thread has exited. |
| `rayon_workers_never_call_with_gil` | Instrument a small Rayon task inside `allow_threads` to call `Python::try_with_gil`. Verify it fails or returns `GilNotHeld` (not a hang or segfault). This is a best-effort safety net; the design-level guarantee comes from the AtomicBool pattern. |

### 14.3 NumPy Interop Tests

| Test | Description |
|:-----|:------------|
| `numpy_omega_dtype_float64` | Verify `omega.dtype == np.float64` (not float32 or object). |
| `numpy_values_dtype_float64` | Verify `values.dtype == np.float64`. |
| `numpy_hybridization_dtype_complex128` | Verify `bath.hybridization_function(omega, 0.05).dtype == np.complex128`. |
| `numpy_arrays_c_contiguous` | Verify `omega` and `values` are C-contiguous (required for TRIQS compatibility and BLAS interop). |
| `numpy_array_independence_from_solver` | Retrieve `solver.bath().epsilon`. Call `solver.solve()` with a minimal config. Verify the previously retrieved `epsilon` array is unchanged (getter returns a copy). |
| `numpy_spectral_lifetime_after_solver_drop` | Retrieve `spectral.omega`. Drop `spectral` from Python scope (`del spectral`). Force GC. Verify `omega` array is still valid and readable. |
| `numpy_bath_set_epsilon_roundtrip` | Set `bath.epsilon = np.array([...])`. Retrieve `bath.epsilon`. Verify the values round-trip correctly. |
| `numpy_bath_hybridization_analytic` | For a known 2-site bath, verify `hybridization_function` output matches the analytic formula within 1e-12. |

### 14.4 Integration Tests — End-to-End Python API

Gated behind the `integration-tests` feature flag (slow; excluded from standard `cargo test`). These tests exercise the complete Python API path against the same Bethe lattice fixtures used in `tk-dmft`'s integration tests.

| Test | Description |
|:-----|:------------|
| `python_bethe_u0_sum_rule` | `DMFTLoop(DMFTConfig(n_bath=6, u=0.0)).solve()`. Verify `spectral.sum_rule() ≈ 1.0 ± 1e-3`. |
| `python_bethe_u4_convergence` | U=4W. Verify `solver.converged() == True` and `n_iterations() < 30`. |
| `python_bethe_u4_stats_length` | After convergence, verify all `stats()` list fields have length equal to `n_iterations()`. |
| `python_config_checkpoint_restart` | Run U=4W with `checkpoint_path` set. Interrupt after 5 iterations via the cancellation flag. Reconstruct solver from checkpoint. Verify restart converges to the same final spectral function as a full run (L∞ < 1e-4). |
| `python_chebyshev_promoted_metallic` | U=0 metallic run. Verify `stats.chebyshev_was_primary` contains only `True`. |
| `python_tdvp_primary_mott` | U=8W Mott insulating run. Verify `stats.chebyshev_was_primary` contains predominantly `False`. |
| `python_real_z2_variant_runs` | `DMFTLoop.real_z2(config)`. Verify it produces a valid spectral function with `sum_rule() ≈ 1.0`. |
| `python_complex_u1_variant_runs` | `DMFTLoop.complex_u1(config)`. Verify it runs without error on a synthetic complex hybridization. |

### 14.5 TRIQS Interop Tests

Gated behind `features = ["triqs"]` and require a TRIQS installation. Run only in TRIQS-enabled CI environments.

| Test | Description |
|:-----|:------------|
| `triqs_gf_imfreq_extraction` | Construct a TRIQS `GfImFreq` with analytically known Δ(iω_n). Call `update_from_triqs_delta`. Verify extracted bath energies match known values. |
| `triqs_spectral_export_shape` | Run DMFT loop. Call `spectral.to_triqs_gf_re_freq()`. Verify the returned TRIQS object has a `GfReFreq` class and `data[:, 0, 0]` values match `spectral.values` within 1e-8. |
| `triqs_zero_copy_data_extraction` | Extract the `.data` buffer from a TRIQS `GfImFreq`. Verify `update_from_triqs_delta` does not copy it (buffer address in `data` matches before and after). |
| `triqs_import_error_graceful` | Remove TRIQS from `sys.path`. Call `update_from_triqs_delta`. Verify `ImportError` is raised with an instructive message. |
| `triqs_wrong_gf_type_raises` | Pass a `GfImTime` instead of `GfImFreq`. Verify `TypeError` is raised. |

### 14.6 Property-Based Tests

```rust
use proptest::prelude::*;

// For any valid (n_bath, u, bandwidth) tuple, constructing PyDmftLoop
// and reading initial state must not panic.
proptest! {
    #[test]
    fn prop_dmft_loop_construct_no_panic(
        n_bath in 1usize..=20,
        u in 0.0f64..=20.0,
        bandwidth in 0.1f64..=20.0,
    ) {
        pyo3::Python::with_gil(|py| {
            let config = PyDmftConfig::new(
                n_bath, u, 0.0, bandwidth, 50, 1e-4, None,
            ).expect("valid config should construct without error");
            let solver = PyDmftLoop::new(&config)
                .expect("valid solver should construct without error");
            assert_eq!(solver.n_iterations(), 0);
            assert!(!solver.converged());
        });
    }
}

// omega and values NumPy arrays always have matching shapes.
proptest! {
    #[test]
    fn prop_spectral_array_shapes_match(
        n_omega in 10usize..=10_000,
        d_omega in 0.001f64..=1.0,
    ) {
        pyo3::Python::with_gil(|py| {
            let omega: Vec<f64> = (0..n_omega).map(|i| i as f64 * d_omega).collect();
            let values = vec![1.0 / n_omega as f64; n_omega];
            let sf = SpectralFunction::new(omega, values);
            let py_sf = PySpectralFunction { inner: sf };
            assert_eq!(py_sf.omega(py).len(), n_omega);
            assert_eq!(py_sf.values(py).len(), n_omega);
        });
    }
}
```

### 14.7 Memory Leak Tests

Python reference counting errors cause memory leaks invisible to Rust's drop checker. These tests use `tracemalloc` to verify allocations do not grow monotonically across repeated solve/GC cycles.

```python
import tracemalloc
import gc
import tensorkraft as tk

def test_no_memory_leak_repeated_construction():
    """Repeated DMFTLoop construction and drop must not grow memory."""
    config = tk.DMFTConfig(n_bath=2, u=0.0, max_iterations=1)
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    for _ in range(20):
        solver = tk.DMFTLoop(config)
        del solver
    gc.collect()

    snapshot_after = tracemalloc.take_snapshot()
    growth_kb = sum(
        s.size_diff for s in snapshot_after.compare_to(snapshot_before, "lineno")
    ) / 1024
    assert growth_kb < 50, \
        f"Memory grew by {growth_kb:.1f} KB over 20 constructions (threshold: 50 KB)"

def test_no_memory_leak_spectral_function_arrays():
    """Repeated NumPy getter calls must not accumulate unreferenced buffers."""
    import numpy as np
    # Construct a large spectral function
    omega = np.linspace(-10, 10, 10_000)
    config = tk.DMFTConfig(n_bath=2, u=0.0, max_iterations=1)
    # ... (construct PySpectralFunction and call omega/values getters repeatedly)
```

---

## 15. Security Considerations

**Trust boundary:** `tk-python` accepts Python objects from user code (configuration dicts, NumPy arrays, TRIQS objects). When the `triqs` feature is active, TRIQS object inspection uses Python's attribute protocol (`getattr`), not raw FFI. No `unsafe` code touches user-provided Python objects.

**Input validation:** All configuration range checks (e.g., `n_bath >= 1`, `bandwidth > 0`, `self_consistency_tol > 0`) are performed in `PyDmftConfig::new()` and the `#[setter]` methods, returning `ConfigError` before values reach Rust. NumPy array sizes are validated by `set_epsilon` and `set_v` before calling into `BathParameters`.

**Panic safety in `allow_threads`:** If the DMFT solver panics inside `py.allow_threads`, PyO3 catches the Rust panic and converts it to `PanicException` (a subclass of `BaseException`). The monitor thread receives `Err(Disconnected)` on channel drop and exits cleanly — no monitor thread outlives a panicking solver.

**No `unsafe` code:** `tk-python` must contain no `unsafe` blocks. All raw memory access is handled by `pyo3` and `rust-numpy`. Any future `unsafe` use (e.g., pinned-view NumPy arrays) requires explicit review for CPython lifetime correctness before merging.

---

## 16. Out of Scope

- **MPI Mode B orchestration** — `tk-python` does not expose `initialize_dmft_node_budget` or MPI primitives. Multi-rank DMFT uses `mpi4py` at the application layer; each rank imports `tensorkraft` independently. `(-> application layer)`
- **Multi-GPU management** — The `backend-cuda` feature is propagated from `tk-dmft`, but no GPU-specific Python API is added. GPU selection is transparent to the Python user. `(-> tk-dmft)`
- **Interactive Jupyter widgets** — Progress bars and real-time convergence plots during `solve()` are out of scope. Users monitor convergence via `solver.stats()` after `solve()` returns. `(-> Phase 5+)`
- **Complex bath parameters via `PyBathParameters`** — `PyBathParameters` exposes `f64` arrays only. A `PyBathParametersComplex` variant for the `ComplexU1` solver is deferred (Open Question #1). `(-> Phase 5+)`
- **SU(2) Python API** — `su2-symmetry` is propagated into the underlying stack but no SU(2)-specific Python API (e.g., total spin constructors, multiplet-resolved bath parameters) is added in Phase 4. `(-> Phase 5+)`
- **Checkpoint load/reload API** — `DMFTCheckpoint` is not directly exposed. Checkpoint read/write is controlled via `DMFTConfig.checkpoint_path`; the binary format is opaque to Python users. `(-> Phase 5+)`
- **`tensorkraft-stubs` package** — Hand-written `.pyi` stubs for IDE autocompletion are out of scope for the initial spec (Open Question #5). `(-> Phase 5+)`

---

## 17. Open Questions

| # | Question | Status |
|:--|:---------|:-------|
| 1 | `PyBathParametersComplex` for `ComplexU1`: the complex hybridization variant is supported by `DmftLoopVariant::ComplexU1` but its bath parameter setter is unspecified. Should complex bath arrays be a separate class or a unified `PyBathParameters` with a `dtype` parameter? | Open |
| 2 | Clone-vs-pinned-view for NumPy getters: the clone-on-getter strategy is correct and simple (§12.3). Should we benchmark to confirm the ~80 KB clone is negligible for typical `n_omega`? If so, is this a CI benchmark or a one-time measurement? | Deferred — defer unless profiling shows it is an issue |
| 3 | PyO3 v0.22 migration: the spec targets v0.21's `&'py PyAny` lifetime style. The v0.22 `Bound<'py, T>` API is the long-term standard. Should this spec target v0.22 for Phase 4, or migrate in Phase 5? | Deferred — use v0.21 for Phase 4; migrate in Phase 5 |
| 4 | TRIQS duck-typing: §6.2 checks the TRIQS class by name. Should it instead check for a duck-typed protocol (presence of `.data`, `.mesh`, `.beta` attributes) to support compatible non-TRIQS Green's function objects? | Open |
| 5 | Python stubs (`.pyi` files) for IDE autocompletion: should maturin auto-generate stubs via `pyo3-stub-gen`, or should hand-written stubs be committed to a `python/tensorkraft.pyi` file? | Open |
| 6 | Config freeze-on-construct: if a Python user modifies `config.dmrg.max_bond_dim` while `solve()` is running on a different thread, PyO3 raises `PyRuntimeError("Already borrowed")`. Is this sufficient, or should `DMFTLoop` deep-copy the config on construction to make post-construction mutations a no-op? | Open — deep-copy on construction is safer; flag as design decision before Phase 4 implementation |
| 7 | Checkpoint restart API: should `DMFTLoop.solve()` accept an optional `resume_from_checkpoint: str` parameter to restart from a saved checkpoint, rather than requiring the user to reconstruct the entire loop? | Open |

---

## 18. Future Considerations

- **PyO3 v0.22 (`Bound<'py, T>`) migration** — The `Bound` API eliminates the implicit `py` lifetime on `&'py PyAny` and makes misuse a compile-time error rather than a runtime panic. Migration is non-breaking from the Python API perspective.
- **`async def solve_async()`** — An async wrapper using `asyncio.run_in_executor` would allow `solve()` to be `await`-ed from async Python code without blocking the event loop. Requires no changes to the GIL protocol but needs Python 3.10+ and a compatible executor.
- **Wheel portability matrix** — GitHub Actions CI matrix across `linux/amd64`, `linux/aarch64`, `macos/arm64`, and `windows/amd64` via `maturin build --release`. The `abi3-py38` tag ensures a single wheel per platform supports Python 3.8+.
- **`tensorkraft-stubs` PyPI package** — A companion package with hand-written `.pyi` stub files for full IDE autocompletion across all `#[pyclass]` types, distributed separately from the binary wheel.
- **SU(2) Python API** — When `su2-symmetry` reaches production readiness, expose `DMFTLoop.real_su2()` and a `PyBathParametersSU2` class with multiplet-resolved bath inspection (spin quantum numbers, Clebsch-Gordan weighting).
- **Streaming convergence callbacks** — A `on_iteration_complete` Python callback registered on `DMFTLoop` could provide iteration-by-iteration statistics without requiring the user to wait for `solve()` to return. Implementation requires a safe mechanism to call a Python callable from inside `allow_threads` — the same `Python::with_gil` channel used by the monitor thread.
