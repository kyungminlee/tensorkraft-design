//! `tk-dmrg` — DMRG algorithm engine for the tensorkraft workspace.
//!
//! This crate implements the complete DMRG infrastructure:
//! - **MPS/MPO types** with typestate canonical forms
//! - **Sweep engine** with two-site and single-site updates
//! - **Iterative eigensolvers**: Lanczos, Davidson, Block-Davidson
//! - **SVD truncation** with bond dimension scheduling
//! - **TDVP time evolution** with Tikhonov regularization and subspace expansion
//! - **Excited-state DMRG** via penalty method
//! - **Infinite DMRG** bootstrap for thermodynamic-limit initialization
//! - **Checkpointing** for crash recovery
//!
//! Consumed by `tk-dmft` for DMFT self-consistency loops.

pub mod checkpoint;
pub mod eigensolver;
pub mod environments;
pub mod error;
pub mod excited;
pub mod idmrg;
pub mod mpo;
pub mod mps;
pub mod sweep;
pub mod tdvp;
pub mod truncation;

// --- Public re-exports ---

// MPS types
pub use mps::{
    BondCentered, LeftCanonical, MPS, MixedCanonical, RightCanonical,
    left_canonicalize, mixed_canonicalize, mps_energy, mps_norm, mps_overlap,
    right_canonicalize,
};

// MPO
pub use mpo::{MPO, MpoCompiler, MpoCompressionConfig};

// Environments
pub use environments::{
    Environment, Environments, build_heff_single_site, build_heff_two_site,
};

// DMRG engine
pub use sweep::{
    DMRGConfig, DMRGEngine, DMRGRuntimeState, DMRGStats, StepResult, SweepDirection,
    SweepSchedule, UpdateVariant,
};

// Eigensolvers
pub use eigensolver::{
    BlockDavidsonSolver, DavidsonSolver, EigenResult, InitialSubspace,
    IterativeEigensolver, LanczosSolver,
};

// Truncation
pub use truncation::{
    BondDimensionSchedule, TruncationConfig, TruncationResult, truncate_svd,
};

// TDVP
pub use tdvp::{TdvpDriver, TdvpStabilizationConfig, TdvpStepResult, exp_krylov, exp_krylov_f64};

// Excited states
pub use excited::{ExcitedStateConfig, build_heff_penalized, build_heff_penalized_from_config};

// Infinite DMRG
pub use idmrg::{IDmrgConfig, run_idmrg};

// Checkpoint
pub use checkpoint::DMRGCheckpoint;

// Error
pub use error::{DmrgError, DmrgResult};
