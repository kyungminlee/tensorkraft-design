//! Error conversion bridge: `DmftError` → `PyErr`.
//!
//! This is the single point of truth for Rust → Python error translation.
//! `DmftError::Cancelled` must NOT be routed through this type; it is
//! converted to `KeyboardInterrupt` at the call site (in `solve()`).

use pyo3::prelude::*;
use tk_dmft::DmftError;

// ---------------------------------------------------------------------------
// Exception hierarchy
//
//   tensorkraft.TensorkraftError         (base; subclass of RuntimeError)
//   ├── tensorkraft.ConvergenceError     (DMFT convergence failure)
//   ├── tensorkraft.BathError            (bath discretization / hybridization)
//   ├── tensorkraft.SpectralError        (linear prediction, Chebyshev, sum rule)
//   ├── tensorkraft.DmrgError            (inner DMRG failures)
//   ├── tensorkraft.CheckpointError      (I/O and deserialization)
//   └── tensorkraft.ConfigError          (configuration validation)
// ---------------------------------------------------------------------------

pyo3::create_exception!(tensorkraft, TensorkraftError, pyo3::exceptions::PyRuntimeError);
pyo3::create_exception!(tensorkraft, ConvergenceError, TensorkraftError);
pyo3::create_exception!(tensorkraft, BathError, TensorkraftError);
pyo3::create_exception!(tensorkraft, SpectralError, TensorkraftError);
pyo3::create_exception!(tensorkraft, DmrgError, TensorkraftError);
pyo3::create_exception!(tensorkraft, CheckpointError, TensorkraftError);
pyo3::create_exception!(tensorkraft, ConfigError, TensorkraftError);

/// Conversion bridge from `DmftError` to `PyErr`.
///
/// Never construct `PyErr` directly from a `DmftError` in `#[pymethods]`
/// bodies — always go through this type.
pub(crate) struct PythonError(pub DmftError);

impl From<DmftError> for PythonError {
    fn from(e: DmftError) -> Self {
        PythonError(e)
    }
}

impl From<PythonError> for PyErr {
    fn from(e: PythonError) -> PyErr {
        match e.0 {
            DmftError::MaxIterationsExceeded {
                iterations,
                distance,
                threshold,
            } => PyErr::new::<ConvergenceError, _>(format!(
                "DMFT did not converge after {} iterations \
                 (distance = {:.2e}, threshold = {:.2e})",
                iterations, distance, threshold
            )),
            DmftError::BathDiscretizationFailed {
                max_steps,
                residual,
            } => PyErr::new::<BathError, _>(format!(
                "Lanczos did not converge in {} steps \
                 (residual = {:.2e}). Consider increasing n_bath or lanczos_tol.",
                max_steps, residual
            )),
            DmftError::InvalidHybridizationFunction { n_negative } => {
                PyErr::new::<BathError, _>(format!(
                    "-Im[Delta(omega)] < 0 at {} frequency points. \
                     The hybridization function must have non-negative imaginary part.",
                    n_negative
                ))
            }
            DmftError::LinearPredictionFailed { condition } => {
                PyErr::new::<SpectralError, _>(format!(
                    "Levinson-Durbin condition number = {:.2e}. \
                     Increase tikhonov_lambda.",
                    condition
                ))
            }
            DmftError::DeconvolutionFailed { eta } => {
                PyErr::new::<SpectralError, _>(format!(
                    "deconvolution requires broadening_eta > 0; got eta = {}",
                    eta
                ))
            }
            DmftError::ChebyshevBandwidthError { e_min, e_max, e0 } => {
                PyErr::new::<SpectralError, _>(format!(
                    "Chebyshev bandwidth error: E_min ({:.4}) >= E_max ({:.4}), \
                     or ground-state energy {:.4} outside [{:.4}, {:.4}]",
                    e_min, e_max, e0, e_min, e_max
                ))
            }
            DmftError::SumRuleViolated { sum_rule } => {
                PyErr::new::<SpectralError, _>(format!(
                    "spectral sum rule violated: integral A(omega) = {:.6} (expected 1.0)",
                    sum_rule
                ))
            }
            DmftError::Dmrg(inner) => {
                PyErr::new::<DmrgError, _>(format!("{}", inner))
            }
            DmftError::CheckpointIo(e) => {
                PyErr::new::<CheckpointError, _>(format!("I/O error: {}", e))
            }
            DmftError::CheckpointDeser(msg) => {
                PyErr::new::<CheckpointError, _>(format!(
                    "deserialization failed: {}", msg
                ))
            }
            DmftError::Cancelled => {
                // This arm must never be reached. DmftError::Cancelled is
                // intercepted at the call site and converted to KeyboardInterrupt.
                unreachable!(
                    "DmftError::Cancelled must be converted to KeyboardInterrupt \
                     at the call site, not routed through PythonError."
                )
            }
        }
    }
}

/// Register the exception hierarchy on the tensorkraft module.
///
/// Creates:
///   tensorkraft.TensorkraftError  (base; subclass of RuntimeError)
///   ├── tensorkraft.ConvergenceError
///   ├── tensorkraft.BathError
///   ├── tensorkraft.SpectralError
///   ├── tensorkraft.DmrgError
///   ├── tensorkraft.CheckpointError
///   └── tensorkraft.ConfigError
pub(crate) fn register_exceptions(_py: Python<'_>, m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add("TensorkraftError", m.py().get_type::<TensorkraftError>())?;
    m.add("ConvergenceError", m.py().get_type::<ConvergenceError>())?;
    m.add("BathError", m.py().get_type::<BathError>())?;
    m.add("SpectralError", m.py().get_type::<SpectralError>())?;
    m.add("DmrgError", m.py().get_type::<DmrgError>())?;
    m.add("CheckpointError", m.py().get_type::<CheckpointError>())?;
    m.add("ConfigError", m.py().get_type::<ConfigError>())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_error_from_dmft_error() {
        let e = DmftError::MaxIterationsExceeded {
            iterations: 50,
            distance: 1e-3,
            threshold: 1e-4,
        };
        let pe = PythonError::from(e);
        // Just verify it converts without panic (PyErr requires GIL to inspect)
        let _ = pe;
    }

    #[test]
    #[should_panic(expected = "unreachable")]
    fn test_cancelled_unreachable() {
        let pe = PythonError(DmftError::Cancelled);
        let _: PyErr = pe.into();
    }
}
