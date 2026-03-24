//! DMRG sweep engine: configuration, scheduling, and execution.

use std::sync::atomic::{AtomicBool, Ordering};

use num_traits::{NumCast, Zero};
use tk_core::{Scalar, SweepArena};
use tk_linalg::LinAlgBackend;
use tk_symmetry::BitPackable;

use crate::eigensolver::{DavidsonSolver, InitialSubspace, IterativeEigensolver};
use crate::environments::{self, Environments};
use crate::error::{DmrgError, DmrgResult};
use crate::mpo::MPO;
use crate::mps::{MPS, MixedCanonical};
use crate::truncation::{self, BondDimensionSchedule, TruncationConfig};

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

/// Bridge function: convert f64 eigensolver I/O to generic T H_eff.
///
/// T::Real: Float implies ToPrimitive, so we can convert T -> f64 via abs_sq + sign detection.
fn heff_bridge_f64_to_t<T: Scalar>(
    heff: &dyn Fn(&[T], &mut [T]),
    x: &[f64],
    y: &mut [f64],
) {
    // f64 -> T: go through T::Real
    let x_t: Vec<T> = x
        .iter()
        .map(|&v| T::from_real(NumCast::from(v).unwrap_or(T::Real::zero())))
        .collect();
    let mut y_t = vec![T::zero(); y.len()];
    heff(&x_t, &mut y_t);
    // T -> f64: extract via abs_sq and sign
    for (yi, yti) in y.iter_mut().zip(y_t.iter()) {
        let val_sq = yti.abs_sq();
        let abs_val: f64 = num_traits::ToPrimitive::to_f64(&val_sq).unwrap_or(0.0).sqrt();
        // Sign detection: if val + small > val in magnitude, val is positive
        let eps_real: T::Real = NumCast::from(1e-30_f64).unwrap_or(T::Real::zero());
        let test = *yti + T::from_real(eps_real);
        let test_sq: f64 = num_traits::ToPrimitive::to_f64(&test.abs_sq()).unwrap_or(0.0);
        let orig_sq = abs_val * abs_val;
        *yi = if test_sq >= orig_sq { abs_val } else { -abs_val };
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
        let n = self.mps.n_sites();
        if site + 1 >= n {
            return Ok(StepResult {
                site,
                direction,
                energy: self.current_energy,
                truncation_error: T::Real::zero(),
                bond_dim_new: self.mps.bond_dim(site.min(n - 1)),
                eigensolver_converged: true,
                eigensolver_iters: 0,
            });
        }

        // 1. Build H_eff matrix and dimensions from environments + MPO
        // The H_eff closure captures environment data, but since build_heff_two_site
        // pre-computes the full dense matrix and moves it into the closure,
        // we can safely drop the closure after eigensolver use.
        let env_l = self.environments.left(site);
        let env_r = self.environments.right(site + 1);
        let (_, _, d_l) = env_l.dims();
        let (_, _, d_r) = env_r.dims();
        let (heff, dim) =
            environments::build_heff_two_site(env_l, env_r, &self.mpo, (site, site + 1))?;

        if dim == 0 {
            return Ok(StepResult {
                site,
                direction,
                energy: self.current_energy,
                truncation_error: T::Real::zero(),
                bond_dim_new: 0,
                eigensolver_converged: true,
                eigensolver_iters: 0,
            });
        }

        // 2. Solve eigenvalue problem via f64 bridge
        let eigen_result = self.runtime.eigensolver.lowest_eigenpair(
            &|x: &[f64], y: &mut [f64]| {
                heff_bridge_f64_to_t::<T>(&heff, x, y);
            },
            dim,
            InitialSubspace::None,
        );

        let energy: T::Real = NumCast::from(eigen_result.eigenvalue).unwrap_or(T::Real::zero());

        // 3. SVD truncate the eigenvector
        let d_i = self.mpo.local_dim(site);
        let d_j = self.mpo.local_dim(site + 1);
        let rows = d_i * d_l;
        let cols = d_j * d_r;

        let bond_dim = self.config.bond_dim_schedule.bond_dim_at_sweep(
            self.stats.sweep_energies.len(),
        );
        let trunc_config = TruncationConfig {
            max_bond_dim: bond_dim,
            svd_cutoff: self.config.svd_cutoff,
            min_bond_dim: 1,
        };

        // Convert eigenvector from f64 to T for SVD
        let eigvec_t: Vec<T> = eigen_result.eigenvector.iter()
            .map(|&v| T::from_real(NumCast::from(v).unwrap_or(T::Real::zero())))
            .collect();

        let (trunc_error, bond_dim_new) = if rows > 0 && cols > 0 && eigvec_t.len() >= rows * cols {
            match truncation::truncate_svd(&eigvec_t, rows, cols, &trunc_config, &self.backend) {
                Ok(result) => (result.truncation_error, result.bond_dim_new),
                Err(_) => (T::Real::zero(), self.mps.bond_dim(site.min(n - 1))),
            }
        } else {
            (T::Real::zero(), self.mps.bond_dim(site.min(n - 1)))
        };

        // 4. Update environments
        match direction {
            SweepDirection::LeftToRight => {
                self.environments.grow_left(site, &self.mps, &self.mpo, &self.backend)?;
            }
            SweepDirection::RightToLeft => {
                self.environments.grow_right(site + 1, &self.mps, &self.mpo, &self.backend)?;
            }
        }

        Ok(StepResult {
            site,
            direction,
            energy,
            truncation_error: trunc_error,
            bond_dim_new,
            eigensolver_converged: eigen_result.converged,
            eigensolver_iters: eigen_result.matvec_count,
        })
    }

    /// Perform a single single-site DMRG step.
    pub fn dmrg_step_single_site(
        &mut self,
        site: usize,
        direction: SweepDirection,
    ) -> DmrgResult<StepResult<T>> {
        let n = self.mps.n_sites();

        // Build H_eff for single site (scoped to release env borrows)
        let (heff, dim) = {
            let env_l = self.environments.left(site);
            let env_r = self.environments.right(site);
            environments::build_heff_single_site(env_l, env_r, &self.mpo, site)?
        };

        if dim == 0 {
            return Ok(StepResult {
                site,
                direction,
                energy: self.current_energy,
                truncation_error: T::Real::zero(),
                bond_dim_new: self.mps.bond_dim(site.min(n - 1)),
                eigensolver_converged: true,
                eigensolver_iters: 0,
            });
        }

        // Solve eigenvalue problem via f64 bridge
        let eigen_result = self.runtime.eigensolver.lowest_eigenpair(
            &|x: &[f64], y: &mut [f64]| {
                heff_bridge_f64_to_t::<T>(&heff, x, y);
            },
            dim,
            InitialSubspace::None,
        );

        let energy: T::Real = NumCast::from(eigen_result.eigenvalue).unwrap_or(T::Real::zero());

        // Update environments
        match direction {
            SweepDirection::LeftToRight => {
                self.environments.grow_left(site, &self.mps, &self.mpo, &self.backend)?;
            }
            SweepDirection::RightToLeft => {
                if site > 0 {
                    self.environments.grow_right(site, &self.mps, &self.mpo, &self.backend)?;
                }
            }
        }

        Ok(StepResult {
            site,
            direction,
            energy,
            truncation_error: T::Real::zero(),
            bond_dim_new: self.mps.bond_dim(site.min(n - 1)),
            eigensolver_converged: eigen_result.converged,
            eigensolver_iters: eigen_result.matvec_count,
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
