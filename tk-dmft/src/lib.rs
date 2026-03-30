//! `tk-dmft` — DMFT self-consistency loop for the tensorkraft workspace.
//!
//! This crate implements the complete DMFT infrastructure:
//! - **Anderson Impurity Model** with bath parameters and Lanczos discretization
//! - **Spectral function extraction** via TDVP + linear prediction and Chebyshev expansion
//! - **Adaptive solver selection** between time-domain and frequency-domain methods
//! - **Self-consistency loop** with linear/Broyden mixing and convergence monitoring
//! - **Spectral positivity restoration** with diagnostic warnings
//! - **Checkpointing** for crash recovery of long DMFT runs
//!
//! Consumed by `tk-python` for PyO3 bindings.

pub mod error;
pub mod impurity;
pub mod r#loop;
pub mod mpi;
pub mod spectral;

// --- Public re-exports ---

// Error types
pub use error::{DmftError, DmftResult};

// Anderson Impurity Model
pub use impurity::AndersonImpurityModel;
pub use impurity::bath::BathParameters;
pub use impurity::discretize::BathDiscretizationConfig;
pub use impurity::hamiltonian::build_aim_chain_hamiltonian;

// Spectral function types and free functions
pub use spectral::{SpectralFunction, SpectralSolverMode};
pub use spectral::chebyshev::{ChebyshevConfig, chebyshev_expand, chebyshev_from_precomputed_moments, jackson_kernel, reconstruct_from_moments};
pub use spectral::linear_predict::{
    LinearPredictionConfig, ToeplitzSolver,
    deconvolve_lorentzian, fft_to_spectral, linear_predict_regularized,
    solve_toeplitz_levinson_durbin, solve_toeplitz_svd_pseudoinverse,
};
pub use spectral::positivity::restore_positivity;
pub use spectral::tdvp::{TdvpSpectralConfig, compute_greens_function_tdvp, tdvp_spectral_pipeline};

// DMFT loop and configuration
pub use r#loop::config::{DMFTConfig, TimeEvolutionConfig};
pub use r#loop::mixing::{BroydenState, MixingScheme};
pub use r#loop::stats::{DMFTStats, DmrgIterationSummary};
pub use r#loop::{DMFTCheckpoint, DMFTLoop};
