//! DMFT self-consistency loop driver.
//!
//! Contains:
//! - `DMFTLoop<T, Q, B>` — the main self-consistency driver
//! - `DMFTConfig` — top-level configuration
//! - `MixingScheme` — bath-update mixing strategies
//! - `DMFTStats` — per-iteration statistics accumulator

pub mod config;
pub mod mixing;
pub mod stats;

use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

use num_complex::Complex;
use tk_core::Scalar;
use tk_linalg::LinAlgBackend;
use tk_symmetry::BitPackable;

use crate::error::{DmftError, DmftResult};
use crate::impurity::AndersonImpurityModel;
use crate::spectral::SpectralFunction;

use self::config::DMFTConfig;
use self::stats::DMFTStats;

/// The DMFT self-consistency driver.
///
/// Holds the current Anderson Impurity Model and drives the iterative loop:
///   bath -> DMRG ground state -> spectral function -> new bath -> ...
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
    /// Whether the most recent iteration converged.
    is_converged: bool,
    /// Number of completed iterations.
    n_iterations: usize,
    #[allow(dead_code)]
    backend: B,
    _phantom: PhantomData<Q>,
}

#[allow(dead_code)]

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
    ) -> Self {
        Self {
            impurity,
            config,
            stats: DMFTStats::default(),
            is_converged: false,
            n_iterations: 0,
            backend,
            _phantom: PhantomData,
        }
    }

    /// Run the self-consistency loop until convergence or `max_iterations`.
    ///
    /// Returns the converged primary spectral function A(omega).
    ///
    /// # Errors
    /// - `DmftError::Dmrg` wrapping any `DmrgError` from DMRG or TDVP
    /// - `DmftError::BathDiscretizationFailed` if Lanczos discretization fails
    /// - `DmftError::MaxIterationsExceeded` if `max_iterations` is reached
    pub fn solve(&mut self) -> DmftResult<SpectralFunction> {
        // TODO: Full implementation requires functional DMRGEngine and TdvpDriver.
        //
        // The self-consistency loop pseudocode:
        //
        // loop {
        //     chain_mpo = impurity.build_chain_hamiltonian()
        //     gs_engine = DMRGEngine::new(mps_init, chain_mpo, backend, dmrg_config)
        //     gs_engine.run()?
        //
        //     // Adaptive solver selection:
        //     let use_cheb_primary = match config.solver_mode { ... };
        //
        //     // Compute both spectral functions:
        //     spectral_tdvp = self.tdvp_spectral(&gs_engine, &chain_mpo)?
        //     spectral_cheb = chebyshev_expand(&gs_engine, &chain_mpo, ..)?
        //
        //     // Cross-validate:
        //     self.validate_consistency(primary, cross)
        //
        //     // Self-consistency update:
        //     delta_new = self.weiss_field(primary)
        //     bath_new = impurity.discretize(&delta_new, ..)?
        //     bath_mixed = apply_mixing(&impurity.bath, &bath_new, &config.mixing)
        //     impurity.update_bath(bath_mixed)
        //
        //     if converged { return Ok(primary.clone()); }
        // }

        unimplemented!(
            "DMFTLoop::solve() requires functional DMRGEngine sweep, \
             TdvpDriver time evolution, and environment contraction from tk-dmrg"
        )
    }

    /// Run with an `AtomicBool` cancellation flag.
    ///
    /// The cancel flag is checked once per complete DMFT iteration using a
    /// single `Relaxed` load.
    pub fn solve_with_cancel_flag(
        &mut self,
        cancel: &AtomicBool,
    ) -> DmftResult<SpectralFunction> {
        if cancel.load(Ordering::Relaxed) {
            return Err(DmftError::Cancelled);
        }
        self.solve()
    }

    /// Whether the most recent iteration satisfied `self_consistency_tol`.
    pub fn converged(&self) -> bool {
        self.is_converged
    }

    /// Number of completed self-consistency iterations.
    pub fn n_iterations(&self) -> usize {
        self.n_iterations
    }

    /// Reference to the current bath parameters.
    pub fn bath(&self) -> &crate::impurity::bath::BathParameters<T> {
        &self.impurity.bath
    }

    /// Compute the non-interacting Weiss field from the impurity spectral function.
    ///
    /// Specialized to the Bethe lattice for Phase 4.
    ///
    /// G_imp(omega) is reconstructed from A_imp(omega) via Kramers-Kronig.
    /// Delta(omega) = omega + mu - G_imp^-1(omega) - epsilon_imp
    pub(crate) fn weiss_field(
        &self,
        _spectral: &SpectralFunction,
    ) -> Vec<Complex<f64>> {
        // TODO: Phase 5+ — implement general lattice Weiss field via Dyson equation.
        // For Bethe lattice:
        //   G_0^-1(omega) = omega + mu - (W^2/z) * G_imp(omega)
        //   Delta(omega) = omega + mu - G_imp^-1(omega) - epsilon_imp
        //
        // Requires Kramers-Kronig transform to get G_imp(omega) from A_imp(omega):
        //   G_imp(omega) = integral A(omega') / (omega - omega' + i*0+) d_omega'
        //
        // This is a Hilbert transform that can be computed via FFT.
        unimplemented!("Bethe lattice Weiss field computation")
    }

    /// Apply the configured mixing scheme to produce the next bath parameters.
    pub(crate) fn apply_mixing(
        &mut self,
        bath_proposed: &crate::impurity::bath::BathParameters<T>,
    ) -> crate::impurity::bath::BathParameters<T> {
        match &self.config.mixing {
            mixing::MixingScheme::Linear { alpha } => {
                self.impurity.bath.linear_mix(bath_proposed, *alpha)
            }
            mixing::MixingScheme::Broyden {
                alpha,
                history_depth: _,
            } => {
                // TODO: Implement Broyden quasi-Newton mixing.
                // For draft: fall back to linear mixing.
                self.impurity.bath.linear_mix(bath_proposed, *alpha)
            }
        }
    }

    /// Emit a cross-validation consistency warning if the two spectral functions
    /// disagree beyond tolerance.
    pub(crate) fn validate_consistency(
        &self,
        primary: &SpectralFunction,
        cross: &SpectralFunction,
    ) {
        let distance = primary.max_distance(cross);
        let cross_max = cross
            .values
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        if cross_max > f64::EPSILON {
            let relative = distance / cross_max;
            if relative > self.config.time_evolution.cross_validation_tol {
                log::warn!(
                    target: "tensorkraft::telemetry",
                    "SPECTRAL_CROSS_VALIDATION_WARNING: primary and cross-validation \
                     spectral functions disagree by {:.1}% (tolerance: {:.1}%).",
                    100.0 * relative,
                    100.0 * self.config.time_evolution.cross_validation_tol,
                );
            }
        }
    }
}

/// Serializable checkpoint for a DMFT run.
///
/// Written atomically after each DMFT iteration when
/// `DMFTConfig::checkpoint_path` is set.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct DMFTCheckpoint {
    /// Iteration index at time of checkpoint.
    pub iteration: usize,
    /// Whether the loop had converged at this checkpoint.
    pub converged: bool,
    /// `DMFTConfig` serialized as JSON for human inspection.
    pub config_json: String,
    /// Bath epsilon values from the last completed iteration.
    pub bath_epsilon: Vec<f64>,
    /// Bath V values from the last completed iteration (stored as real parts).
    pub bath_v: Vec<f64>,
    /// Primary spectral function omega grid.
    pub spectral_omega: Vec<f64>,
    /// Primary spectral function values.
    pub spectral_values: Vec<f64>,
}

impl DMFTCheckpoint {
    /// Write checkpoint atomically: write to `{path}.tmp`, then rename.
    pub fn write_to_file(&self, path: &std::path::Path) -> DmftResult<()> {
        let tmp_path = path.with_extension("tmp");
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| DmftError::CheckpointDeser(e.to_string()))?;
        std::fs::write(&tmp_path, json)?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Load checkpoint from file.
    pub fn read_from_file(path: &std::path::Path) -> DmftResult<Self> {
        let data = std::fs::read_to_string(path)?;
        serde_json::from_str(&data).map_err(|e| DmftError::CheckpointDeser(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_roundtrip() {
        let cp = DMFTCheckpoint {
            iteration: 5,
            converged: false,
            config_json: "{}".to_string(),
            bath_epsilon: vec![-1.0, 0.0, 1.0],
            bath_v: vec![0.5, 0.5, 0.5],
            spectral_omega: vec![-1.0, 0.0, 1.0],
            spectral_values: vec![0.1, 0.3, 0.1],
        };

        let dir = std::env::temp_dir();
        let path = dir.join("test_dmft_checkpoint.json");
        cp.write_to_file(&path).unwrap();

        let loaded = DMFTCheckpoint::read_from_file(&path).unwrap();
        assert_eq!(loaded.iteration, 5);
        assert!(!loaded.converged);
        assert_eq!(loaded.bath_epsilon, vec![-1.0, 0.0, 1.0]);
        assert_eq!(loaded.spectral_values, vec![0.1, 0.3, 0.1]);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }
}
