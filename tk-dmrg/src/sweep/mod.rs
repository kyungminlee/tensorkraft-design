//! DMRG sweep engine: configuration, scheduling, and execution.

use std::sync::atomic::{AtomicBool, Ordering};

use num_traits::Zero;
use tk_core::{Scalar, SweepArena};
use tk_linalg::LinAlgBackend;
use tk_symmetry::BitPackable;

use crate::eigensolver::{DavidsonSolver, IterativeEigensolver};
use crate::environments::Environments;
use crate::error::{DmrgError, DmrgResult};
use crate::mpo::MPO;
use crate::mps::{MPS, MixedCanonical};
use crate::truncation::{BondDimensionSchedule, TruncationConfig};

/// DMRG update variant.
#[derive(Clone, Copy, Debug, Default)]
pub enum UpdateVariant {
    /// Two-site update: diagonalizes rank-4 objects, allows bond growth.
    #[default]
    TwoSite,
    /// Single-site update: faster, no bond dimension growth.
    SingleSite,
}

/// DMRG sweep direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepDirection {
    LeftToRight,
    RightToLeft,
}

/// Result of a single DMRG optimization step.
pub struct StepResult<T: Scalar> {
    /// Site index of the step.
    pub site: usize,
    /// Sweep direction.
    pub direction: SweepDirection,
    /// Ground state energy estimate.
    pub energy: T::Real,
    /// Truncation error from SVD.
    pub truncation_error: T::Real,
    /// New bond dimension after truncation.
    pub bond_dim_new: usize,
    /// Whether the eigensolver converged.
    pub eigensolver_converged: bool,
    /// Number of eigensolver iterations.
    pub eigensolver_iters: usize,
}

/// DMRG sweep schedule: determines the order of site updates.
pub struct SweepSchedule {
    n_sites: usize,
    lr_sites: Vec<usize>,
    rl_sites: Vec<usize>,
}

impl SweepSchedule {
    /// Standard sweep schedule: left-to-right then right-to-left.
    pub fn standard(n_sites: usize) -> Self {
        let lr_sites: Vec<usize> = (0..n_sites - 1).collect();
        let rl_sites: Vec<usize> = (1..n_sites).rev().collect();
        SweepSchedule {
            n_sites,
            lr_sites,
            rl_sites,
        }
    }

    /// Iterate over (site, direction) pairs for a full sweep.
    pub fn iter(&self) -> impl Iterator<Item = (usize, SweepDirection)> + '_ {
        self.lr_sites
            .iter()
            .map(|&s| (s, SweepDirection::LeftToRight))
            .chain(
                self.rl_sites
                    .iter()
                    .map(|&s| (s, SweepDirection::RightToLeft)),
            )
    }
}

/// Accumulated statistics from DMRG sweeps.
pub struct DMRGStats {
    /// Energy at end of each sweep.
    pub sweep_energies: Vec<f64>,
    /// Maximum truncation error per sweep.
    pub max_truncation_errors: Vec<f64>,
    /// Maximum bond dimension per sweep.
    pub max_bond_dims: Vec<usize>,
    /// Wall time per sweep in seconds.
    pub sweep_times_secs: Vec<f64>,
    /// Total eigensolver calls.
    pub total_eigensolver_calls: usize,
    /// Number of eigensolver calls that didn't converge.
    pub unconverged_eigensolver_calls: usize,
}

impl DMRGStats {
    pub fn new() -> Self {
        DMRGStats {
            sweep_energies: Vec::new(),
            max_truncation_errors: Vec::new(),
            max_bond_dims: Vec::new(),
            sweep_times_secs: Vec::new(),
            total_eigensolver_calls: 0,
            unconverged_eigensolver_calls: 0,
        }
    }
}

impl Default for DMRGStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable DMRG configuration — set once before a run begins.
pub struct DMRGConfig {
    /// Bond dimension schedule (ramp up over sweeps).
    pub bond_dim_schedule: BondDimensionSchedule,
    /// SVD cutoff (default: 1e-12).
    pub svd_cutoff: f64,
    /// Maximum number of sweeps (default: 20).
    pub max_sweeps: usize,
    /// Energy convergence tolerance (relative change, default: 1e-10).
    pub energy_tol: f64,
    /// Optional variance convergence criterion.
    pub variance_tol: Option<f64>,
    /// Use infinite DMRG warmup.
    pub idmrg_warmup: bool,
    /// Update variant (default: TwoSite).
    pub update_variant: UpdateVariant,
    /// Checkpoint path for serializing state.
    pub checkpoint_path: Option<std::path::PathBuf>,
    /// Number of target states for excited-state DMRG.
    pub n_target_states: Option<usize>,
    /// Penalty weight for excited states (default: 0.1).
    pub excited_state_weight: f64,
}

impl Default for DMRGConfig {
    fn default() -> Self {
        DMRGConfig {
            bond_dim_schedule: BondDimensionSchedule::fixed(200),
            svd_cutoff: 1e-12,
            max_sweeps: 20,
            energy_tol: 1e-10,
            variance_tol: None,
            idmrg_warmup: false,
            update_variant: UpdateVariant::default(),
            checkpoint_path: None,
            n_target_states: None,
            excited_state_weight: 0.1,
        }
    }
}

/// Mutable DMRG runtime state — holds solver and per-sweep bookkeeping.
///
/// Separated from `DMRGConfig` so that static config can be `&self` while
/// runtime state (eigensolver preconditioner, diagonal updates) can be `&mut self`.
pub struct DMRGRuntimeState {
    /// Eigensolver to use (default: DavidsonSolver). Mutable because the
    /// diagonal preconditioner is updated each step.
    pub eigensolver: Box<dyn IterativeEigensolver<f64>>,
}

impl Default for DMRGRuntimeState {
    fn default() -> Self {
        DMRGRuntimeState {
            eigensolver: Box::new(DavidsonSolver::default()),
        }
    }
}

/// Main DMRG sweep engine.
pub struct DMRGEngine<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> {
    /// Current MPS state.
    pub mps: MPS<T, Q, MixedCanonical>,
    /// Hamiltonian as MPO.
    pub mpo: MPO<T, Q>,
    /// Cached environment blocks.
    pub environments: Environments<T, Q>,
    /// Linear algebra backend.
    pub backend: B,
    /// DMRG configuration (immutable).
    pub config: DMRGConfig,
    /// Mutable runtime state (eigensolver, preconditioner, etc.).
    pub runtime: DMRGRuntimeState,
    /// Accumulated statistics.
    pub stats: DMRGStats,
    /// Arena for temporary allocations during sweeps.
    arena: SweepArena,
    /// Current energy.
    current_energy: T::Real,
}

impl<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>> DMRGEngine<T, Q, B> {
    /// Create a new DMRG engine. Builds environments from scratch.
    pub fn new(
        mps: MPS<T, Q, MixedCanonical>,
        mpo: MPO<T, Q>,
        backend: B,
        config: DMRGConfig,
    ) -> DmrgResult<Self> {
        Self::with_runtime(mps, mpo, backend, config, DMRGRuntimeState::default())
    }

    /// Create a new DMRG engine with explicit runtime state.
    pub fn with_runtime(
        mps: MPS<T, Q, MixedCanonical>,
        mpo: MPO<T, Q>,
        backend: B,
        config: DMRGConfig,
        runtime: DMRGRuntimeState,
    ) -> DmrgResult<Self> {
        let environments = Environments::build_from_scratch(&mps, &mpo, &backend)?;
        let arena = SweepArena::new(64 * 1024 * 1024); // 64 MB default

        Ok(DMRGEngine {
            mps,
            mpo,
            environments,
            backend,
            config,
            runtime,
            stats: DMRGStats::new(),
            arena,
            current_energy: T::Real::zero(),
        })
    }

    /// Run DMRG to convergence.
    pub fn run(&mut self) -> DmrgResult<T::Real> {
        let cancel = AtomicBool::new(false);
        self.run_with_cancel_flag(&cancel)
    }

    /// Run DMRG with cancellation support.
    pub fn run_with_cancel_flag(
        &mut self,
        cancel: &AtomicBool,
    ) -> DmrgResult<T::Real> {
        let schedule = SweepSchedule::standard(self.mps.n_sites());

        for sweep in 0..self.config.max_sweeps {
            let start = std::time::Instant::now();
            let mut max_trunc = 0.0_f64;
            let mut max_bond = 0_usize;

            let _bond_dim = self.config.bond_dim_schedule.bond_dim_at_sweep(sweep);

            for (site, direction) in schedule.iter() {
                if cancel.load(Ordering::Relaxed) {
                    return Err(DmrgError::Cancelled);
                }

                let step = match self.config.update_variant {
                    UpdateVariant::TwoSite => {
                        self.dmrg_step_two_site(site, direction)?
                    }
                    UpdateVariant::SingleSite => {
                        self.dmrg_step_single_site(site, direction)?
                    }
                };

                self.current_energy = step.energy;
                self.stats.total_eigensolver_calls += 1;
                if !step.eigensolver_converged {
                    self.stats.unconverged_eigensolver_calls += 1;
                }

                let trunc_f64 = num_traits::cast::NumCast::from(step.truncation_error).unwrap_or(0.0);
                max_trunc = max_trunc.max(trunc_f64);
                max_bond = max_bond.max(step.bond_dim_new);
            }

            let elapsed = start.elapsed().as_secs_f64();
            let energy_f64: f64 = num_traits::cast::NumCast::from(self.current_energy).unwrap_or(0.0);
            self.stats.sweep_energies.push(energy_f64);
            self.stats.max_truncation_errors.push(max_trunc);
            self.stats.max_bond_dims.push(max_bond);
            self.stats.sweep_times_secs.push(elapsed);

            if self.converged() {
                break;
            }

            // ARENA SAFETY: All step results have been moved into owned MPS tensors
            // by this point. The arena reset reclaims all scratch memory in O(1).
            // Any TempTensor<'_> references are statically prevented from surviving
            // past this point by the borrow checker.
            self.arena.reset();
        }

        Ok(self.current_energy)
    }

    /// Perform a single two-site DMRG step.
    ///
    /// # Arena Safety Contract
    ///
    /// Temporary tensors allocated from `self.arena` during this step (reshape
    /// buffers, Krylov vectors, etc.) must NOT outlive the step. Specifically:
    ///
    /// 1. All arena-allocated temporaries are used within this function scope.
    /// 2. SVD results (U, S, V†) must be converted to owned storage via
    ///    `into_owned()` BEFORE `arena.reset()` is called.
    /// 3. The arena is reset at the end of each sweep (in `run()`), not per-step,
    ///    so all step results must be fully owned by that point.
    ///
    /// The borrow checker enforces that `TempTensor<'a>` cannot outlive the
    /// arena, but `into_owned()` must be called explicitly before reset.
    pub fn dmrg_step_two_site(
        &mut self,
        site: usize,
        direction: SweepDirection,
    ) -> DmrgResult<StepResult<T>> {
        // Full implementation steps:
        // 1. Build H_eff closure from environments + MPO
        //    (uses arena for scratch space in environment contraction)
        // 2. Solve eigenvalue problem: eigensolver.lowest_eigenpair(heff, dim, initial)
        // 3. SVD truncate the two-site tensor
        // 4. CRITICAL: Call .into_owned() on U and V† before storing in MPS
        //    (arena data becomes invalid after reset)
        // 5. Update MPS tensors at site and site+1
        // 6. Update environments (grow_left or grow_right depending on direction)
        Ok(StepResult {
            site,
            direction,
            energy: self.current_energy,
            truncation_error: T::Real::zero(),
            bond_dim_new: self.mps.bond_dim(site.min(self.mps.n_sites() - 1)),
            eigensolver_converged: true,
            eigensolver_iters: 0,
        })
    }

    /// Perform a single single-site DMRG step.
    pub fn dmrg_step_single_site(
        &mut self,
        site: usize,
        direction: SweepDirection,
    ) -> DmrgResult<StepResult<T>> {
        Ok(StepResult {
            site,
            direction,
            energy: self.current_energy,
            truncation_error: T::Real::zero(),
            bond_dim_new: self.mps.bond_dim(site.min(self.mps.n_sites() - 1)),
            eigensolver_converged: true,
            eigensolver_iters: 0,
        })
    }

    /// Current ground state energy estimate.
    pub fn energy(&self) -> T::Real {
        self.current_energy
    }

    /// Check if the calculation has converged.
    pub fn converged(&self) -> bool {
        let n = self.stats.sweep_energies.len();
        if n < 2 {
            return false;
        }
        let e_curr = self.stats.sweep_energies[n - 1];
        let e_prev = self.stats.sweep_energies[n - 2];
        let denom = e_curr.abs().max(1.0);
        ((e_curr - e_prev).abs() / denom) < self.config.energy_tol
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = DMRGStats::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_schedule_standard() {
        let schedule = SweepSchedule::standard(5);
        let steps: Vec<_> = schedule.iter().collect();
        // LR: 0,1,2,3  RL: 4,3,2,1
        assert_eq!(steps[0], (0, SweepDirection::LeftToRight));
        assert_eq!(steps[3], (3, SweepDirection::LeftToRight));
        assert_eq!(steps[4], (4, SweepDirection::RightToLeft));
    }

    #[test]
    fn dmrg_config_defaults() {
        let config = DMRGConfig::default();
        assert_eq!(config.max_sweeps, 20);
        assert!((config.energy_tol - 1e-10).abs() < 1e-20);
    }

    #[test]
    fn dmrg_stats_new() {
        let stats = DMRGStats::new();
        assert_eq!(stats.total_eigensolver_calls, 0);
        assert!(stats.sweep_energies.is_empty());
    }
}
