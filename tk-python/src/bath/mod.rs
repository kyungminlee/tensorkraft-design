//! Python-accessible bath parameters for the Anderson Impurity Model.

use num_complex::Complex;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use tk_dmft::BathParameters;

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
/// ```
#[pyclass(name = "BathParameters")]
pub struct PyBathParameters {
    pub(crate) inner: BathParameters<f64>,
}

impl From<BathParameters<f64>> for PyBathParameters {
    fn from(bp: BathParameters<f64>) -> Self {
        PyBathParameters { inner: bp }
    }
}

#[pymethods]
impl PyBathParameters {
    /// On-site bath energies ε_k as a NumPy array of shape `(n_bath,)`, dtype `float64`.
    #[getter]
    pub fn epsilon<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.epsilon.clone())
    }

    /// Set bath energies from a NumPy array.
    ///
    /// Raises ValueError if `values.len() != self.n_bath`.
    #[setter]
    pub fn set_epsilon(&mut self, values: PyReadonlyArray1<f64>) -> PyResult<()> {
        let slice = values.as_slice()?;
        if slice.len() != self.inner.n_bath {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected array of length {}, got {}",
                self.inner.n_bath,
                slice.len()
            )));
        }
        self.inner.epsilon = slice.to_vec();
        Ok(())
    }

    /// Hybridization amplitudes V_k as a NumPy array of shape `(n_bath,)`, dtype `float64`.
    #[getter]
    pub fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.v.clone())
    }

    /// Set hybridization amplitudes from a NumPy array.
    ///
    /// Raises ValueError if `values.len() != self.n_bath`.
    #[setter]
    pub fn set_v(&mut self, values: PyReadonlyArray1<f64>) -> PyResult<()> {
        let slice = values.as_slice()?;
        if slice.len() != self.inner.n_bath {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected array of length {}, got {}",
                self.inner.n_bath,
                slice.len()
            )));
        }
        self.inner.v = slice.to_vec();
        Ok(())
    }

    /// Number of bath sites.
    #[getter]
    pub fn n_bath(&self) -> usize {
        self.inner.n_bath
    }

    /// Compute the discretized hybridization function on a frequency grid.
    ///
    /// Δ(ω) = Σ_k |V_k|² / (ω - ε_k + i·broadening)
    ///
    /// Returns a complex NumPy array of shape `(n_omega,)`, dtype `complex128`.
    pub fn hybridization_function<'py>(
        &self,
        py: Python<'py>,
        omega: PyReadonlyArray1<f64>,
        broadening: f64,
    ) -> PyResult<Bound<'py, PyArray1<Complex<f64>>>> {
        let omega_slice = omega.as_slice()?;
        let delta = self.inner.hybridization_function(omega_slice, broadening);
        Ok(PyArray1::from_vec_bound(py, delta))
    }

    pub fn __repr__(&self) -> String {
        format!(
            "BathParameters(n_bath={}, epsilon=[{:.4}, ...], v=[{:.4}, ...])",
            self.inner.n_bath,
            self.inner.epsilon.first().copied().unwrap_or(0.0),
            self.inner.v.first().copied().unwrap_or(0.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_bath_from_bath() {
        let bath = BathParameters::<f64>::uniform(4, 10.0, 1.0);
        let py_bath = PyBathParameters::from(bath);
        assert_eq!(py_bath.n_bath(), 4);
    }
}
