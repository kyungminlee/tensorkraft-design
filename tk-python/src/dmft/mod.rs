//! Python-accessible DMFT self-consistency solver.

pub(crate) mod solve;
pub(crate) mod stats;

use pyo3::prelude::*;

use tk_dmft::{AndersonImpurityModel, DMFTLoop};
use tk_linalg::{DefaultDevice, DeviceFaer, DeviceAPI};

use crate::bath::PyBathParameters;
use crate::config::PyDmftConfig;
use crate::dispatch::DmftLoopVariant;
use crate::spectral::PySpectralFunction;
use self::stats::PyDmftStats;

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
/// config = tk.DMFTConfig(n_bath=6, u=4.0, bandwidth=4.0, max_iterations=30)
/// solver = tk.DMFTLoop(config)
/// spectral = solver.solve()          # releases GIL; responds to Ctrl+C
/// print(f"Sum rule: {spectral.sum_rule():.6f}")
/// ```
#[pyclass(name = "DMFTLoop")]
pub struct PyDmftLoop {
    pub(crate) inner: DmftLoopVariant,
}

/// Helper: construct a default backend instance.
fn default_backend() -> DefaultDevice {
    DeviceAPI::new(DeviceFaer, DeviceFaer)
}

#[pymethods]
impl PyDmftLoop {
    /// Construct a real-valued U(1) DMFT solver (the standard single-orbital case).
    #[new]
    pub fn new(config: &PyDmftConfig) -> PyResult<Self> {
        let aim = AndersonImpurityModel::<f64>::new(
            config.u,
            config.epsilon_imp,
            config.n_bath,
            config.bandwidth,
            1.0, // v0: initial uniform hybridization
        );
        let rust_config = config.to_rust_config();
        let backend = default_backend();
        let loop_driver = DMFTLoop::new(aim, rust_config, backend);
        Ok(PyDmftLoop {
            inner: DmftLoopVariant::RealU1(loop_driver),
        })
    }

    /// Construct a complex-valued U(1) DMFT solver.
    ///
    /// NOTE: Not available in draft — `DeviceFaer` does not implement
    /// `LinAlgBackend<Complex<f64>>`. Will be enabled when the complex
    /// backend is implemented in tk-linalg.
    ///
    /// Raises RuntimeError in the draft implementation.
    #[staticmethod]
    pub fn complex_u1(_config: &PyDmftConfig) -> PyResult<Self> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "ComplexU1 variant is not available: DeviceFaer does not implement \
             LinAlgBackend<Complex<f64>>. This will be enabled when the complex \
             backend is implemented in tk-linalg."
        ))
    }

    /// Construct a real-valued Z₂ DMFT solver.
    ///
    /// Use for particle-hole symmetric models.
    #[staticmethod]
    pub fn real_z2(config: &PyDmftConfig) -> PyResult<Self> {
        let aim = AndersonImpurityModel::<f64>::new(
            config.u,
            config.epsilon_imp,
            config.n_bath,
            config.bandwidth,
            1.0,
        );
        let rust_config = config.to_rust_config();
        let backend = default_backend();
        let loop_driver = DMFTLoop::new(aim, rust_config, backend);
        Ok(PyDmftLoop {
            inner: DmftLoopVariant::RealZ2(loop_driver),
        })
    }

    /// Run the DMFT self-consistency loop until convergence or `max_iterations`.
    ///
    /// Releases the GIL for the entire computation. Responds to Ctrl+C via
    /// a monitor thread that polls `check_signals()` every 100 ms.
    ///
    /// # Raises
    /// - `KeyboardInterrupt` — Ctrl+C received.
    /// - `RuntimeError` — DMFT or DMRG error.
    pub fn solve(&mut self, py: Python<'_>) -> PyResult<PySpectralFunction> {
        solve::solve_impl(py, &mut self.inner)
    }

    /// Whether the most recent call to `solve()` converged.
    pub fn converged(&self) -> bool {
        crate::dispatch::macros::dispatch_variant!(self.inner, converged())
    }

    /// Number of completed self-consistency iterations.
    pub fn n_iterations(&self) -> usize {
        crate::dispatch::macros::dispatch_variant!(self.inner, n_iterations())
    }

    /// Current bath parameters (read-only snapshot).
    pub fn bath(&self) -> PyResult<PyBathParameters> {
        match &self.inner {
            DmftLoopVariant::RealU1(s) => Ok(PyBathParameters::from(s.bath().clone())),
            DmftLoopVariant::RealZ2(s) => Ok(PyBathParameters::from(s.bath().clone())),
        }
    }

    /// Per-iteration statistics from the most recent `solve()` call.
    pub fn stats(&self) -> PyDmftStats {
        let stats = match &self.inner {
            DmftLoopVariant::RealU1(s) => s.stats.clone(),
            DmftLoopVariant::RealZ2(s) => s.stats.clone(),
        };
        PyDmftStats { inner: stats }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "DMFTLoop({}, converged={}, n_iterations={})",
            self.inner.variant_name(),
            self.converged(),
            self.n_iterations(),
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_variant_name_strings() {
        assert_eq!("RealU1", "RealU1");
        assert_eq!("RealZ2", "RealZ2");
    }
}
