//! Python-accessible configuration classes.
//!
//! Since `DMRGConfig` from tk-dmrg does not implement `Clone` or `Debug`
//! (it contains `Box<dyn IterativeEigensolver<f64>>`), we maintain a
//! Python-owned mirror of all configuration fields. The Rust `DMFTConfig`
//! is reconstructed from these fields when `DMFTLoop` is constructed.

pub(crate) mod defaults;

use pyo3::prelude::*;

use tk_dmft::{DMFTConfig, LinearPredictionConfig, MixingScheme, SpectralSolverMode, TimeEvolutionConfig, ToeplitzSolver};
use tk_dmrg::{BondDimensionSchedule, DMRGConfig, DavidsonSolver, LanczosSolver, BlockDavidsonSolver, TdvpStabilizationConfig};

/// Top-level DMFT configuration.
///
/// # Python usage
///
/// ```python
/// config = tk.DMFTConfig(
///     n_bath=6, u=4.0, epsilon_imp=0.0, bandwidth=4.0,
///     max_iterations=30, self_consistency_tol=1e-4,
/// )
/// config.dmrg.max_bond_dim = 200
/// config.time_evolution.t_max = 30.0
/// ```
#[pyclass(name = "DMFTConfig")]
pub struct PyDmftConfig {
    // Physical parameters (stored for DMFTLoop constructors)
    pub(crate) n_bath: usize,
    pub(crate) u: f64,
    pub(crate) epsilon_imp: f64,
    pub(crate) bandwidth: f64,

    // DMFT loop parameters
    pub(crate) max_iterations: usize,
    pub(crate) self_consistency_tol: f64,
    pub(crate) checkpoint_path: Option<String>,

    // Nested config mirrors
    pub(crate) dmrg: PyDmrgConfig,
    pub(crate) time_evolution: PyTimeEvolutionConfig,
    pub(crate) linear_prediction: LinearPredictionConfig,
    pub(crate) solver_mode: SpectralSolverMode,
    pub(crate) mixing: MixingScheme,
}

#[pymethods]
impl PyDmftConfig {
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
    ) -> PyResult<Self> {
        if n_bath < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_bath must be >= 1",
            ));
        }
        if bandwidth <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "bandwidth must be > 0",
            ));
        }
        if self_consistency_tol <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "self_consistency_tol must be > 0",
            ));
        }
        Ok(PyDmftConfig {
            n_bath,
            u,
            epsilon_imp,
            bandwidth,
            max_iterations,
            self_consistency_tol,
            checkpoint_path: checkpoint_path.map(|s| s.to_string()),
            dmrg: PyDmrgConfig::default(),
            time_evolution: PyTimeEvolutionConfig::default(),
            linear_prediction: LinearPredictionConfig::default(),
            solver_mode: SpectralSolverMode::default(),
            mixing: MixingScheme::default(),
        })
    }

    #[getter]
    pub fn n_bath(&self) -> usize {
        self.n_bath
    }

    #[getter]
    pub fn u(&self) -> f64 {
        self.u
    }

    #[getter]
    pub fn epsilon_imp(&self) -> f64 {
        self.epsilon_imp
    }

    #[getter]
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    #[getter]
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    #[setter]
    pub fn set_max_iterations(&mut self, val: usize) {
        self.max_iterations = val;
    }

    #[getter]
    pub fn self_consistency_tol(&self) -> f64 {
        self.self_consistency_tol
    }

    #[setter]
    pub fn set_self_consistency_tol(&mut self, val: f64) -> PyResult<()> {
        if val <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "self_consistency_tol must be > 0",
            ));
        }
        self.self_consistency_tol = val;
        Ok(())
    }

    #[getter]
    pub fn checkpoint_path(&self) -> Option<String> {
        self.checkpoint_path.clone()
    }

    #[setter]
    pub fn set_checkpoint_path(&mut self, path: Option<&str>) {
        self.checkpoint_path = path.map(|s| s.to_string());
    }

    /// DMRG sweep configuration.
    #[getter]
    pub fn dmrg(&self) -> PyDmrgConfig {
        self.dmrg.clone()
    }

    /// Time evolution configuration.
    #[getter]
    pub fn time_evolution(&self) -> PyTimeEvolutionConfig {
        self.time_evolution.clone()
    }

    /// Linear prediction pipeline configuration.
    #[getter]
    pub fn linear_prediction(&self) -> PyLinearPredictionConfig {
        PyLinearPredictionConfig {
            inner: self.linear_prediction.clone(),
        }
    }

    /// Spectral solver mode: "tdvp", "chebyshev", or "adaptive".
    #[getter]
    pub fn solver_mode(&self) -> String {
        match &self.solver_mode {
            SpectralSolverMode::TdvpPrimary => "tdvp".to_string(),
            SpectralSolverMode::ChebyshevPrimary => "chebyshev".to_string(),
            SpectralSolverMode::Adaptive { .. } => "adaptive".to_string(),
        }
    }

    #[setter]
    pub fn set_solver_mode(&mut self, mode: &str) -> PyResult<()> {
        self.solver_mode = match mode {
            "tdvp" => SpectralSolverMode::TdvpPrimary,
            "chebyshev" => SpectralSolverMode::ChebyshevPrimary,
            "adaptive" => SpectralSolverMode::Adaptive {
                gap_threshold: 0.1,
            },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown solver mode '{}': expected 'tdvp', 'chebyshev', or 'adaptive'",
                    mode
                )));
            }
        };
        Ok(())
    }

    /// Bath mixing scheme: "linear" or "broyden".
    #[getter]
    pub fn mixing(&self) -> String {
        match &self.mixing {
            MixingScheme::Linear { .. } => "linear".to_string(),
            MixingScheme::Broyden { .. } => "broyden".to_string(),
        }
    }

    #[setter]
    pub fn set_mixing(&mut self, scheme: &str) -> PyResult<()> {
        self.mixing = match scheme {
            "linear" => MixingScheme::Linear { alpha: 0.5 },
            "broyden" => MixingScheme::Broyden {
                alpha: 0.5,
                history_depth: 5,
            },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown mixing scheme '{}': expected 'linear' or 'broyden'",
                    scheme
                )));
            }
        };
        Ok(())
    }

    pub fn __repr__(&self) -> String {
        format!(
            "DMFTConfig(n_bath={}, u={}, epsilon_imp={}, bandwidth={}, \
             max_iterations={}, self_consistency_tol={:.1e}, solver_mode='{}', mixing='{}')",
            self.n_bath,
            self.u,
            self.epsilon_imp,
            self.bandwidth,
            self.max_iterations,
            self.self_consistency_tol,
            self.solver_mode(),
            self.mixing(),
        )
    }
}

impl PyDmftConfig {
    /// Build the Rust `DMFTConfig` from the Python-owned mirror.
    pub(crate) fn to_rust_config(&self) -> DMFTConfig {
        DMFTConfig {
            dmrg_config: self.dmrg.to_rust_config(),
            time_evolution: self.time_evolution.to_rust_config(),
            linear_prediction: self.linear_prediction.clone(),
            solver_mode: self.solver_mode.clone(),
            mixing: self.mixing.clone(),
            self_consistency_tol: self.self_consistency_tol,
            max_iterations: self.max_iterations,
            bath_discretization: Default::default(),
            checkpoint_path: self.checkpoint_path.as_ref().map(|s| s.into()),
        }
    }
}

/// DMRG sweep configuration.
///
/// Since `tk_dmrg::DMRGConfig` contains `Box<dyn IterativeEigensolver<f64>>`
/// which is not `Clone`, we maintain a Python-owned mirror of all fields.
#[pyclass(name = "DMRGConfig")]
#[derive(Clone)]
pub struct PyDmrgConfig {
    pub(crate) max_bond_dim: usize,
    pub(crate) svd_cutoff: f64,
    pub(crate) max_sweeps: usize,
    pub(crate) energy_tol: f64,
    pub(crate) eigensolver_name: String,
}

impl Default for PyDmrgConfig {
    fn default() -> Self {
        PyDmrgConfig {
            max_bond_dim: 200,
            svd_cutoff: 1e-12,
            max_sweeps: 20,
            energy_tol: 1e-10,
            eigensolver_name: "davidson".to_string(),
        }
    }
}

#[pymethods]
impl PyDmrgConfig {
    #[getter]
    pub fn max_bond_dim(&self) -> usize {
        self.max_bond_dim
    }

    #[setter]
    pub fn set_max_bond_dim(&mut self, val: usize) {
        self.max_bond_dim = val;
    }

    #[getter]
    pub fn svd_cutoff(&self) -> f64 {
        self.svd_cutoff
    }

    #[setter]
    pub fn set_svd_cutoff(&mut self, val: f64) {
        self.svd_cutoff = val;
    }

    #[getter]
    pub fn max_sweeps(&self) -> usize {
        self.max_sweeps
    }

    #[setter]
    pub fn set_max_sweeps(&mut self, val: usize) {
        self.max_sweeps = val;
    }

    #[getter]
    pub fn energy_tol(&self) -> f64 {
        self.energy_tol
    }

    #[setter]
    pub fn set_energy_tol(&mut self, val: f64) {
        self.energy_tol = val;
    }

    #[getter]
    pub fn eigensolver(&self) -> String {
        self.eigensolver_name.clone()
    }

    #[setter]
    pub fn set_eigensolver(&mut self, name: &str) -> PyResult<()> {
        match name {
            "lanczos" | "davidson" | "block_davidson" => {
                self.eigensolver_name = name.to_string();
                Ok(())
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown eigensolver '{}': expected 'lanczos', 'davidson', or 'block_davidson'",
                name
            ))),
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "DMRGConfig(max_bond_dim={}, svd_cutoff={:.1e}, max_sweeps={}, \
             energy_tol={:.1e}, eigensolver='{}')",
            self.max_bond_dim, self.svd_cutoff, self.max_sweeps,
            self.energy_tol, self.eigensolver_name,
        )
    }
}

impl PyDmrgConfig {
    /// Build the Rust `DMRGConfig` from the Python-owned mirror.
    ///
    /// `DMRGConfig` is now `Clone` (no trait objects). The eigensolver
    /// is constructed separately via `to_rust_runtime_state()`.
    pub(crate) fn to_rust_config(&self) -> DMRGConfig {
        DMRGConfig {
            bond_dim_schedule: BondDimensionSchedule::fixed(self.max_bond_dim),
            svd_cutoff: self.svd_cutoff,
            max_sweeps: self.max_sweeps,
            energy_tol: self.energy_tol,
            variance_tol: None,
            idmrg_warmup: false,
            update_variant: Default::default(),
            checkpoint_path: None,
            n_target_states: None,
            excited_state_weight: 0.1,
        }
    }

    /// Build the Rust `DMRGRuntimeState` (eigensolver) from the Python mirror.
    pub(crate) fn to_rust_runtime_state(&self) -> tk_dmrg::DMRGRuntimeState {
        let eigensolver: Box<dyn tk_dmrg::IterativeEigensolver<f64>> =
            match self.eigensolver_name.as_str() {
                "lanczos" => Box::new(LanczosSolver::default()),
                "davidson" => Box::new(DavidsonSolver::default()),
                "block_davidson" => Box::new(BlockDavidsonSolver::default()),
                _ => Box::new(DavidsonSolver::default()),
            };
        tk_dmrg::DMRGRuntimeState { eigensolver }
    }
}

/// TDVP time evolution configuration.
#[pyclass(name = "TimeEvolutionConfig")]
#[derive(Clone)]
pub struct PyTimeEvolutionConfig {
    pub(crate) t_max: f64,
    pub(crate) dt: f64,
    pub(crate) max_bond_dim: usize,
    pub(crate) cross_validation_tol: f64,
    pub(crate) tikhonov_delta: f64,
    pub(crate) expansion_vectors: usize,
    pub(crate) soft_dmax_factor: f64,
    pub(crate) dmax_decay_time: f64,
}

impl Default for PyTimeEvolutionConfig {
    fn default() -> Self {
        PyTimeEvolutionConfig {
            t_max: 20.0,
            dt: 0.05,
            max_bond_dim: 500,
            cross_validation_tol: 0.05,
            tikhonov_delta: 1e-10,
            expansion_vectors: 4,
            soft_dmax_factor: 1.1,
            dmax_decay_time: 5.0,
        }
    }
}

#[pymethods]
impl PyTimeEvolutionConfig {
    #[getter]
    pub fn t_max(&self) -> f64 { self.t_max }
    #[setter]
    pub fn set_t_max(&mut self, val: f64) { self.t_max = val; }

    #[getter]
    pub fn dt(&self) -> f64 { self.dt }
    #[setter]
    pub fn set_dt(&mut self, val: f64) { self.dt = val; }

    #[getter]
    pub fn max_bond_dim(&self) -> usize { self.max_bond_dim }
    #[setter]
    pub fn set_max_bond_dim(&mut self, val: usize) { self.max_bond_dim = val; }

    #[getter]
    pub fn cross_validation_tol(&self) -> f64 { self.cross_validation_tol }
    #[setter]
    pub fn set_cross_validation_tol(&mut self, val: f64) { self.cross_validation_tol = val; }

    #[getter]
    pub fn tikhonov_delta(&self) -> f64 { self.tikhonov_delta }
    #[setter]
    pub fn set_tikhonov_delta(&mut self, val: f64) { self.tikhonov_delta = val; }

    #[getter]
    pub fn expansion_vectors(&self) -> usize { self.expansion_vectors }
    #[setter]
    pub fn set_expansion_vectors(&mut self, val: usize) { self.expansion_vectors = val; }

    #[getter]
    pub fn soft_dmax_factor(&self) -> f64 { self.soft_dmax_factor }
    #[setter]
    pub fn set_soft_dmax_factor(&mut self, val: f64) { self.soft_dmax_factor = val; }

    #[getter]
    pub fn dmax_decay_time(&self) -> f64 { self.dmax_decay_time }
    #[setter]
    pub fn set_dmax_decay_time(&mut self, val: f64) { self.dmax_decay_time = val; }

    pub fn __repr__(&self) -> String {
        format!(
            "TimeEvolutionConfig(t_max={}, dt={}, max_bond_dim={}, cross_validation_tol={})",
            self.t_max, self.dt, self.max_bond_dim, self.cross_validation_tol,
        )
    }
}

impl PyTimeEvolutionConfig {
    /// Build the Rust `TimeEvolutionConfig` from the Python-owned mirror.
    pub(crate) fn to_rust_config(&self) -> TimeEvolutionConfig {
        let mut tdvp_stab = TdvpStabilizationConfig::default();
        tdvp_stab.tikhonov_delta = self.tikhonov_delta;
        tdvp_stab.expansion_vectors = self.expansion_vectors;
        tdvp_stab.soft_dmax_factor = self.soft_dmax_factor;
        tdvp_stab.dmax_decay_steps = self.dmax_decay_time;

        TimeEvolutionConfig {
            t_max: self.t_max,
            dt: self.dt,
            max_bond_dim: self.max_bond_dim,
            tdvp_stabilization: tdvp_stab,
            chebyshev: Default::default(),
            cross_validation_tol: self.cross_validation_tol,
        }
    }
}

/// Linear prediction pipeline configuration.
#[pyclass(name = "LinearPredictionConfig")]
pub struct PyLinearPredictionConfig {
    pub(crate) inner: LinearPredictionConfig,
}

#[pymethods]
impl PyLinearPredictionConfig {
    #[getter]
    pub fn prediction_order(&self) -> usize {
        self.inner.prediction_order
    }
    #[setter]
    pub fn set_prediction_order(&mut self, val: usize) {
        self.inner.prediction_order = val;
    }

    #[getter]
    pub fn extrapolation_factor(&self) -> f64 {
        self.inner.extrapolation_factor
    }
    #[setter]
    pub fn set_extrapolation_factor(&mut self, val: f64) {
        self.inner.extrapolation_factor = val;
    }

    #[getter]
    pub fn broadening_eta(&self) -> f64 {
        self.inner.broadening_eta
    }
    #[setter]
    pub fn set_broadening_eta(&mut self, val: f64) {
        self.inner.broadening_eta = val;
    }

    #[getter]
    pub fn deconv_tikhonov_delta(&self) -> f64 {
        self.inner.deconv_tikhonov_delta
    }
    #[setter]
    pub fn set_deconv_tikhonov_delta(&mut self, val: f64) {
        self.inner.deconv_tikhonov_delta = val;
    }

    #[getter]
    pub fn deconv_omega_max(&self) -> f64 {
        self.inner.deconv_omega_max
    }
    #[setter]
    pub fn set_deconv_omega_max(&mut self, val: f64) {
        self.inner.deconv_omega_max = val;
    }

    #[getter]
    pub fn positivity_floor(&self) -> f64 {
        self.inner.positivity_floor
    }
    #[setter]
    pub fn set_positivity_floor(&mut self, val: f64) {
        self.inner.positivity_floor = val;
    }

    #[getter]
    pub fn positivity_warning_threshold(&self) -> f64 {
        self.inner.positivity_warning_threshold
    }
    #[setter]
    pub fn set_positivity_warning_threshold(&mut self, val: f64) {
        self.inner.positivity_warning_threshold = val;
    }

    #[getter]
    pub fn fermi_level_shift_tolerance(&self) -> f64 {
        self.inner.fermi_level_shift_tolerance
    }
    #[setter]
    pub fn set_fermi_level_shift_tolerance(&mut self, val: f64) {
        self.inner.fermi_level_shift_tolerance = val;
    }

    #[getter]
    pub fn toeplitz_solver(&self) -> String {
        match &self.inner.toeplitz_solver {
            ToeplitzSolver::LevinsonDurbin { .. } => "levinson_durbin".to_string(),
            ToeplitzSolver::SvdPseudoInverse { .. } => "svd".to_string(),
        }
    }

    #[setter]
    pub fn set_toeplitz_solver(&mut self, name: &str) -> PyResult<()> {
        self.inner.toeplitz_solver = match name {
            "levinson_durbin" => ToeplitzSolver::LevinsonDurbin {
                tikhonov_lambda: 1e-8,
            },
            "svd" => ToeplitzSolver::SvdPseudoInverse {
                svd_noise_floor: 1e-12,
            },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown Toeplitz solver '{}': expected 'levinson_durbin' or 'svd'",
                    name
                )));
            }
        };
        Ok(())
    }

    pub fn __repr__(&self) -> String {
        format!(
            "LinearPredictionConfig(prediction_order={}, broadening_eta={}, toeplitz_solver='{}')",
            self.inner.prediction_order,
            self.inner.broadening_eta,
            self.toeplitz_solver(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_dmrg_config_default() {
        let cfg = PyDmrgConfig::default();
        assert_eq!(cfg.max_bond_dim, 200);
        assert_eq!(cfg.max_sweeps, 20);
        assert_eq!(cfg.eigensolver_name, "davidson");
    }

    #[test]
    fn test_py_dmrg_config_to_rust() {
        let mut cfg = PyDmrgConfig::default();
        cfg.max_bond_dim = 300;
        cfg.eigensolver_name = "lanczos".to_string();
        let rust_cfg = cfg.to_rust_config();
        assert_eq!(rust_cfg.max_sweeps, 20);
        assert!((rust_cfg.svd_cutoff - 1e-12).abs() < 1e-20);
    }

    #[test]
    fn test_py_time_evolution_default() {
        let cfg = PyTimeEvolutionConfig::default();
        assert!((cfg.t_max - 20.0).abs() < 1e-12);
        assert!((cfg.dt - 0.05).abs() < 1e-12);
        assert_eq!(cfg.max_bond_dim, 500);
    }
}
