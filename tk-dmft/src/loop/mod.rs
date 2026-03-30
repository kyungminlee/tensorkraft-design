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
    /// Broyden mixing state (used when MixingScheme::Broyden is selected).
    broyden_state: Option<mixing::BroydenState>,
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
        let broyden_state = match &config.mixing {
            mixing::MixingScheme::Broyden { history_depth, .. } => {
                Some(mixing::BroydenState::new(*history_depth))
            }
            _ => None,
        };
        Self {
            impurity,
            config,
            stats: DMFTStats::default(),
            is_converged: false,
            n_iterations: 0,
            backend,
            broyden_state,
            _phantom: PhantomData,
        }
    }

    /// Run the self-consistency loop until convergence or `max_iterations`.
    ///
    /// Returns the converged primary spectral function A(omega).
    ///
    /// The self-consistency loop:
    /// 1. Build AIM chain Hamiltonian (OpSum) from current bath
    /// 2. Run DMRG ground-state solver
    /// 3. Compute spectral functions via TDVP and/or Chebyshev
    /// 4. Adaptive solver selection (entanglement gap)
    /// 5. Cross-validate primary vs secondary
    /// 6. Compute Weiss field (Bethe lattice Dyson equation)
    /// 7. Discretize new bath via Lanczos tridiagonalization
    /// 8. Mix old and new bath (linear or Broyden)
    /// 9. Check convergence
    ///
    /// # Errors
    /// - `DmftError::Dmrg` wrapping any `DmrgError` from DMRG or TDVP
    /// - `DmftError::BathDiscretizationFailed` if Lanczos discretization fails
    /// - `DmftError::MaxIterationsExceeded` if `max_iterations` is reached
    /// - `DmftError::Cancelled` if the cancellation flag is set
    pub fn solve(&mut self) -> DmftResult<SpectralFunction> {
        // Build the frequency grid for spectral functions and Weiss field
        let n_omega = self.config.bath_discretization.n_omega_points;
        let bw = self.config.bath_discretization.bandwidth;
        let _omega: Vec<f64> = (0..n_omega)
            .map(|i| -bw + 2.0 * bw * i as f64 / (n_omega as f64 - 1.0))
            .collect();

        let mut _last_spectral: Option<SpectralFunction> = None;

        for iteration in 0..self.config.max_iterations {
            let _iter_start = std::time::Instant::now();

            // --- Step 1: Build AIM chain Hamiltonian ---
            let _opsum =
                crate::impurity::hamiltonian::build_aim_chain_hamiltonian(&self.impurity);

            // --- Step 2: DMRG ground state ---
            // This requires compiling the OpSum into an MPO and running DMRGEngine.
            // When tk-dmrg's MpoCompiler and DMRGEngine are fully functional:
            //
            //   let mpo = MpoCompiler::compile(&opsum, &backend)?;
            //   let mps = MPS::random(n_sites, d=4, D_init=10, &backend);
            //   let mut engine = DMRGEngine::new(mps, mpo, &backend, &config.dmrg_config);
            //   engine.run()?;
            //   let gs_energy = engine.energy();
            //   let gs_mps = engine.mps();

            // --- Step 3: Spectral functions ---
            // Adaptive solver selection based on entanglement gap:
            //   let use_cheb_primary = match self.config.solver_mode {
            //       SpectralSolverMode::TdvpPrimary => false,
            //       SpectralSolverMode::ChebyshevPrimary => true,
            //       SpectralSolverMode::Adaptive { gap_threshold } => {
            //           gs_mps.entanglement_gap_at_center() < gap_threshold
            //       }
            //   };
            //
            //   let spectral_tdvp = tdvp_spectral_pipeline(&tdvp_config, &lp_config, &omega)?;
            //   let spectral_cheb = chebyshev_expand(&omega, e_min, e_max, &cheb_config)?;
            //
            //   let (primary, cross) = if use_cheb_primary {
            //       (spectral_cheb, spectral_tdvp)
            //   } else {
            //       (spectral_tdvp, spectral_cheb)
            //   };

            // For now, return an error indicating DMRG engine is needed.
            // Once tk-dmrg provides a functional DMRGEngine with sweep(),
            // the above pseudocode should be uncommented and the error removed.
            return Err(DmftError::Dmrg(tk_dmrg::DmrgError::NotImplemented(
                format!(
                    "DMFT iteration {} requires functional DMRGEngine, \
                     MpoCompiler, and TdvpDriver from tk-dmrg. \
                     The self-consistency loop structure is complete; \
                     awaiting upstream integration.",
                    iteration
                ),
            )));

            // --- The following code documents the complete loop structure ---
            // --- and will execute once the DMRG engine is functional ---

            // // Step 4: Cross-validate
            // self.validate_consistency(&primary, &cross);

            // // Step 5: Weiss field (Bethe lattice)
            // let delta_new = self.weiss_field(&primary);

            // // Step 6: Discretize new bath
            // // Convert complex delta to the format expected by Lanczos
            // let delta_as_t: Vec<T> = delta_new.iter().map(|c| {
            //     T::from_real(T::Real::from(-c.im))
            // }).collect();
            // let omega_real: Vec<T::Real> = omega.iter().map(|&w| T::Real::from(w)).collect();
            // let bath_proposed = self.impurity.discretize(
            //     &delta_as_t, &omega_real, &self.config.bath_discretization
            // )?;

            // // Step 7: Mix
            // let bath_mixed = self.apply_mixing(&bath_proposed);

            // // Step 8: Convergence check
            // let omega_real: Vec<T::Real> = omega.iter().map(|&w| T::Real::from(w)).collect();
            // let broadening = T::Real::from(self.config.bath_discretization.broadening);
            // let distance = self.impurity.bath.hybridization_distance(
            //     &bath_mixed, &omega_real, broadening
            // );

            // // Step 9: Update bath and statistics
            // self.impurity.update_bath(bath_mixed);
            // self.n_iterations = iteration + 1;

            // let wall_time = iter_start.elapsed().as_secs_f64();
            // let dmrg_summary = stats::DmrgIterationSummary {
            //     energy: gs_energy,
            //     max_truncation_error: 0.0,
            //     max_bond_dim: 0,
            //     n_sweeps: 0,
            //     wall_time_secs: wall_time,
            // };

            // self.stats.push_iteration(
            //     gs_energy, distance, primary.sum_rule(),
            //     0.0, use_cheb_primary, wall_time, dmrg_summary,
            // );

            // // Checkpoint
            // if let Some(ref cp_path) = self.config.checkpoint_path {
            //     let checkpoint = DMFTCheckpoint { ... };
            //     checkpoint.write_to_file(cp_path)?;
            // }

            // if distance < self.config.self_consistency_tol {
            //     self.is_converged = true;
            //     return Ok(primary);
            // }

            // last_spectral = Some(primary);
        }

        // Max iterations exceeded
        let final_distance = self
            .stats
            .hybridization_distances
            .last()
            .copied()
            .unwrap_or(f64::INFINITY);
        Err(DmftError::MaxIterationsExceeded {
            iterations: self.config.max_iterations,
            distance: final_distance,
            threshold: self.config.self_consistency_tol,
        })
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

    /// Resume a DMFT run from a previously saved checkpoint.
    ///
    /// Restores bath parameters and iteration count from the checkpoint,
    /// allowing a crashed or interrupted run to continue from where it
    /// left off without recomputing earlier iterations.
    ///
    /// # Parameters
    /// - `checkpoint`: a `DMFTCheckpoint` loaded via `DMFTCheckpoint::read_from_file`
    ///
    /// # Panics
    /// Panics if the checkpoint bath size doesn't match the current AIM bath size.
    pub fn resume_from_checkpoint(&mut self, checkpoint: &DMFTCheckpoint) {
        let n = checkpoint.bath_epsilon.len();
        assert_eq!(
            n,
            checkpoint.bath_v.len(),
            "checkpoint bath_epsilon and bath_v must have the same length"
        );
        assert_eq!(
            n, self.impurity.bath.n_bath,
            "checkpoint bath size ({}) must match current AIM bath size ({})",
            n, self.impurity.bath.n_bath,
        );

        // Restore bath parameters
        let epsilon: Vec<T::Real> = checkpoint
            .bath_epsilon
            .iter()
            .map(|&e| T::Real::from(e))
            .collect();
        let v: Vec<T> = checkpoint
            .bath_v
            .iter()
            .map(|&v_val| T::from_real(T::Real::from(v_val)))
            .collect();

        self.impurity.update_bath(crate::impurity::bath::BathParameters {
            epsilon,
            v,
            n_bath: n,
        });

        // Restore iteration count and convergence state
        self.n_iterations = checkpoint.iteration;
        self.is_converged = checkpoint.converged;
    }

    /// Compute the non-interacting Weiss field from the impurity spectral function.
    ///
    /// Specialized to the Bethe lattice (infinite coordination).
    ///
    /// For the Bethe lattice with half-bandwidth W, the self-consistency
    /// relation simplifies to:
    ///   Delta(omega) = (W/2)^2 * G_imp(omega)
    ///
    /// where G_imp(omega) is reconstructed from A_imp(omega) via Kramers-Kronig:
    ///   G_imp(omega + i*eta) = integral A(omega') / (omega - omega' + i*eta) d_omega'
    ///
    /// The half-bandwidth W is taken from `bath_discretization.bandwidth` in the config.
    pub(crate) fn weiss_field(
        &self,
        spectral: &SpectralFunction,
    ) -> Vec<Complex<f64>> {
        let eta = self.config.bath_discretization.broadening;
        let half_bw = self.config.bath_discretization.bandwidth / 2.0;

        // Reconstruct G_imp(omega) from A(omega) via Kramers-Kronig (discrete Hilbert transform)
        // G_imp(omega_i) = sum_j A(omega_j) * d_omega / (omega_i - omega_j + i*eta)
        let d_omega = spectral.d_omega;
        let n = spectral.omega.len();

        let g_imp: Vec<Complex<f64>> = spectral
            .omega
            .iter()
            .map(|&w_i| {
                let mut g = Complex::new(0.0, 0.0);
                for j in 0..n {
                    let denom = Complex::new(w_i - spectral.omega[j], eta);
                    g = g + spectral.values[j] * d_omega / denom;
                }
                g
            })
            .collect();

        // Bethe lattice self-consistency: Delta(omega) = (W/2)^2 * G_imp(omega)
        let t_sq = half_bw * half_bw;
        g_imp.iter().map(|&g| t_sq * g).collect()
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
            mixing::MixingScheme::Broyden { alpha, .. } => {
                let alpha_val = *alpha;
                // Flatten current bath into parameter vector [epsilon..., |v|...]
                let n = self.impurity.bath.n_bath;
                let mut x_current = Vec::with_capacity(2 * n);
                for &e in &self.impurity.bath.epsilon {
                    x_current.push(e.into());
                }
                for &v in &self.impurity.bath.v {
                    let v_f64: f64 = v.abs_sq().into();
                    x_current.push(v_f64.sqrt());
                }

                let mut f_proposed = Vec::with_capacity(2 * n);
                for &e in &bath_proposed.epsilon {
                    f_proposed.push(e.into());
                }
                for &v in &bath_proposed.v {
                    let v_f64: f64 = v.abs_sq().into();
                    f_proposed.push(v_f64.sqrt());
                }

                let x_next = if let Some(ref mut state) = self.broyden_state {
                    state.update(&x_current, &f_proposed, alpha_val)
                } else {
                    // Fallback: linear mixing
                    x_current
                        .iter()
                        .zip(&f_proposed)
                        .map(|(x, f)| (1.0 - alpha_val) * x + alpha_val * f)
                        .collect()
                };

                // Reconstruct BathParameters from flattened vector
                let mut epsilon = Vec::with_capacity(n);
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    epsilon.push(T::Real::from(x_next[i]));
                }
                for i in 0..n {
                    v.push(T::from_real(T::Real::from(x_next[n + i])));
                }

                crate::impurity::bath::BathParameters {
                    epsilon,
                    v,
                    n_bath: n,
                }
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

    #[test]
    fn test_resume_from_checkpoint() {
        use crate::impurity::AndersonImpurityModel;
        use tk_linalg::DeviceFaer;
        use tk_symmetry::U1;

        let aim: AndersonImpurityModel<f64> =
            AndersonImpurityModel::new(4.0, -2.0, 3, 10.0, 1.0);
        let config = crate::r#loop::config::DMFTConfig::default();
        let backend = DeviceFaer;

        let mut dmft_loop: DMFTLoop<f64, U1, DeviceFaer> =
            DMFTLoop::new(aim, config, backend);

        // Create a checkpoint with known values
        let checkpoint = DMFTCheckpoint {
            iteration: 7,
            converged: false,
            config_json: "{}".to_string(),
            bath_epsilon: vec![-2.0, 0.0, 2.0],
            bath_v: vec![0.3, 0.5, 0.3],
            spectral_omega: vec![],
            spectral_values: vec![],
        };

        dmft_loop.resume_from_checkpoint(&checkpoint);

        assert_eq!(dmft_loop.n_iterations(), 7);
        assert!(!dmft_loop.converged());
        assert!((dmft_loop.bath().epsilon[0] - (-2.0)).abs() < 1e-12);
        assert!((dmft_loop.bath().epsilon[2] - 2.0).abs() < 1e-12);
        assert!((dmft_loop.bath().v[1] - 0.5).abs() < 1e-12);
    }
}
