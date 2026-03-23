//! Python-accessible spectral function with NumPy interop.

pub(crate) mod numpy_interop;

use pyo3::prelude::*;
use tk_dmft::SpectralFunction;

/// Python-accessible spectral function A(ω) = -Im[G(ω)] / π.
///
/// Frequency grid (`omega`) and spectral values (`values`) are exposed as
/// NumPy arrays via getters. The underlying data is owned by Rust; each
/// getter call clones the data into a new NumPy array.
///
/// # Python usage
///
/// ```python
/// spectral = solver.solve()
/// print(spectral.omega.shape)        # (n_omega,), dtype float64
/// print(spectral.values.shape)       # (n_omega,), dtype float64
/// print(spectral.sum_rule())         # should be ≈ 1.0
/// ```
#[pyclass(name = "SpectralFunction")]
pub struct PySpectralFunction {
    pub(crate) inner: SpectralFunction,
}

impl From<SpectralFunction> for PySpectralFunction {
    fn from(sf: SpectralFunction) -> Self {
        PySpectralFunction { inner: sf }
    }
}

#[pymethods]
impl PySpectralFunction {
    /// Frequency grid ω as a NumPy array of shape `(n_omega,)`, dtype `float64`.
    #[getter]
    pub fn omega<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        Ok(numpy::PyArray1::from_vec_bound(py, self.inner.omega.clone()))
    }

    /// Spectral weight A(ω) as a NumPy array of shape `(n_omega,)`, dtype `float64`.
    #[getter]
    pub fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        Ok(numpy::PyArray1::from_vec_bound(py, self.inner.values.clone()))
    }

    /// Frequency grid spacing Δω (uniform).
    #[getter]
    pub fn d_omega(&self) -> f64 {
        self.inner.d_omega
    }

    /// Spectral sum rule: ∫A(ω)dω via the trapezoidal rule.
    pub fn sum_rule(&self) -> f64 {
        self.inner.sum_rule()
    }

    /// Value of A(ω) at the Fermi level (ω = 0).
    ///
    /// Raises ValueError if the frequency grid does not span ω = 0.
    pub fn value_at_fermi_level(&self) -> PyResult<f64> {
        // Check that the grid spans zero before calling (which panics)
        let first = self.inner.omega.first().copied().unwrap_or(1.0);
        let last = self.inner.omega.last().copied().unwrap_or(-1.0);
        if first > 0.0 || last < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frequency grid does not span ω = 0",
            ));
        }
        Ok(self.inner.value_at_omega_zero())
    }

    /// The nth spectral moment: ∫ωⁿ A(ω)dω via the trapezoidal rule.
    pub fn moment(&self, n: usize) -> f64 {
        self.inner.moment(n)
    }

    /// L∞ distance ‖self - other‖_∞ between two spectral functions.
    ///
    /// Raises ValueError if `other` has a different number of frequency grid points.
    pub fn max_distance(&self, other: &PySpectralFunction) -> PyResult<f64> {
        if self.inner.omega.len() != other.inner.omega.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "spectral functions have different grid sizes: {} vs {}",
                self.inner.omega.len(),
                other.inner.omega.len()
            )));
        }
        Ok(self.inner.max_distance(&other.inner))
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SpectralFunction(n_omega={}, d_omega={:.6}, sum_rule={:.6})",
            self.inner.len(),
            self.inner.d_omega,
            self.inner.sum_rule()
        )
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_spectral_from_spectral() {
        let omega: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        let values: Vec<f64> = omega
            .iter()
            .map(|&w| {
                let eta = 0.5;
                eta / (std::f64::consts::PI * (w * w + eta * eta))
            })
            .collect();
        let sf = SpectralFunction::new(omega, values);
        let py_sf = PySpectralFunction::from(sf);
        assert_eq!(py_sf.__len__(), 101);
    }
}
