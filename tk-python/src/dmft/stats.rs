//! Python-accessible DMFT per-iteration statistics.

use pyo3::prelude::*;
use tk_dmft::DMFTStats;

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
/// ```
#[pyclass(name = "DMFTStats")]
pub struct PyDmftStats {
    pub(crate) inner: DMFTStats,
}

#[pymethods]
impl PyDmftStats {
    /// DMRG ground-state energies at each iteration.
    #[getter]
    pub fn ground_state_energies(&self) -> Vec<f64> {
        self.inner.ground_state_energies.clone()
    }

    /// Relative hybridization distance per iteration.
    #[getter]
    pub fn hybridization_distances(&self) -> Vec<f64> {
        self.inner.hybridization_distances.clone()
    }

    /// Spectral sum rule per iteration (should be ~ 1.0).
    #[getter]
    pub fn spectral_sum_rules(&self) -> Vec<f64> {
        self.inner.spectral_sum_rules.clone()
    }

    /// Fraction of negative spectral weight clamped per iteration.
    #[getter]
    pub fn positivity_clamped_fractions(&self) -> Vec<f64> {
        self.inner.positivity_clamped_fractions.clone()
    }

    /// Whether Chebyshev was the primary solver at each iteration.
    #[getter]
    pub fn chebyshev_was_primary(&self) -> Vec<bool> {
        self.inner.chebyshev_was_primary.clone()
    }

    /// Wall-clock seconds per iteration.
    #[getter]
    pub fn iteration_times_secs(&self) -> Vec<f64> {
        self.inner.iteration_times_secs.clone()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "DMFTStats(n_iterations={})",
            self.inner.n_iterations()
        )
    }

    pub fn __len__(&self) -> usize {
        self.inner.n_iterations()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_default_empty() {
        let stats = PyDmftStats {
            inner: DMFTStats::default(),
        };
        assert_eq!(stats.__len__(), 0);
        assert!(stats.ground_state_energies().is_empty());
    }
}
