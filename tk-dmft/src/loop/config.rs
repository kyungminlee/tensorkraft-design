//! DMFT top-level configuration.

use tk_dmrg::DMRGConfig;
use tk_dmrg::TdvpStabilizationConfig;

use crate::impurity::discretize::BathDiscretizationConfig;
use crate::spectral::SpectralSolverMode;
use crate::spectral::chebyshev::ChebyshevConfig;
use crate::spectral::linear_predict::LinearPredictionConfig;

use super::mixing::MixingScheme;

/// Configuration for TDVP-based real-time Green's function computation.
///
/// Does not derive `Clone` or `Debug` because `TdvpStabilizationConfig`
/// from tk-dmrg doesn't implement them (it depends on types with
/// `Box<dyn ...>` fields).
pub struct TimeEvolutionConfig {
    /// Total simulation time t_max (in inverse energy units). Default: 20.0.
    pub t_max: f64,
    /// Physical time step dt. Default: 0.05.
    pub dt: f64,
    /// Maximum MPS bond dimension during time evolution. Default: 500.
    pub max_bond_dim: usize,
    /// TDVP numerical stabilization configuration.
    pub tdvp_stabilization: TdvpStabilizationConfig,
    /// Chebyshev cross-validation configuration.
    pub chebyshev: ChebyshevConfig,
    /// Relative L-infinity tolerance for TDVP/Chebyshev consistency check.
    /// Default: 0.05 (5%).
    pub cross_validation_tol: f64,
}

impl Default for TimeEvolutionConfig {
    fn default() -> Self {
        Self {
            t_max: 20.0,
            dt: 0.05,
            max_bond_dim: 500,
            tdvp_stabilization: TdvpStabilizationConfig::default(),
            chebyshev: ChebyshevConfig::default(),
            cross_validation_tol: 0.05,
        }
    }
}

/// Top-level configuration for a DMFT self-consistency run.
///
/// Does not derive `Clone` or `Debug` because `DMRGConfig` from tk-dmrg
/// contains `Box<dyn IterativeEigensolver<f64>>` which is not clonable.
pub struct DMFTConfig {
    /// DMRG sweep configuration for ground-state computation at each iteration.
    pub dmrg_config: DMRGConfig,
    /// Time evolution and spectral function extraction configuration.
    pub time_evolution: TimeEvolutionConfig,
    /// Linear prediction pipeline configuration.
    pub linear_prediction: LinearPredictionConfig,
    /// Primary/fallback spectral solver selection strategy.
    pub solver_mode: SpectralSolverMode,
    /// Bath update mixing scheme.
    pub mixing: MixingScheme,
    /// DMFT convergence criterion: relative change in hybridization function.
    /// Default: 1e-4.
    pub self_consistency_tol: f64,
    /// Maximum number of DMFT self-consistency iterations. Default: 50.
    pub max_iterations: usize,
    /// Bath discretization configuration.
    pub bath_discretization: BathDiscretizationConfig,
    /// Optional checkpoint path.
    pub checkpoint_path: Option<std::path::PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_evolution_config_default() {
        let config = TimeEvolutionConfig::default();
        assert!((config.t_max - 20.0).abs() < 1e-12);
        assert!((config.dt - 0.05).abs() < 1e-12);
        assert_eq!(config.max_bond_dim, 500);
        assert!((config.cross_validation_tol - 0.05).abs() < 1e-12);
    }
}
