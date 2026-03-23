//! Error types for `tk-dmft`.

use tk_dmrg::DmrgError;

/// All errors that can arise from `tk-dmft` operations.
#[derive(Debug, thiserror::Error)]
pub enum DmftError {
    #[error("bath discretization failed: Lanczos tridiagonalization did not converge \
             within {max_steps} steps (residual = {residual:.2e})")]
    BathDiscretizationFailed { max_steps: usize, residual: f64 },

    #[error("linear prediction failed: Levinson-Durbin diverged \
             (estimated condition number = {condition:.2e}). \
             Consider increasing LinearPredictionConfig::toeplitz_solver.tikhonov_lambda.")]
    LinearPredictionFailed { condition: f64 },

    #[error("deconvolution requires broadening_eta > 0; got eta = {eta}. \
             Deconvolution must be skipped when eta = 0.0.")]
    DeconvolutionFailed { eta: f64 },

    #[error("Chebyshev bandwidth error: E_min ({e_min:.4}) >= E_max ({e_max:.4}), \
             or ground-state energy {e0:.4} outside [{e_min:.4}, {e_max:.4}]")]
    ChebyshevBandwidthError { e_min: f64, e_max: f64, e0: f64 },

    #[error("DMFT did not converge after {iterations} iterations \
             (final hybridization distance = {distance:.2e}, \
             threshold = {threshold:.2e})")]
    MaxIterationsExceeded {
        iterations: usize,
        distance: f64,
        threshold: f64,
    },

    #[error("spectral sum rule violated: integral A(omega) = {sum_rule:.6} (expected 1.0)")]
    SumRuleViolated { sum_rule: f64 },

    #[error("invalid hybridization function: -Im[Delta(omega)] < 0 at {n_negative} frequency \
             points. The hybridization function must have positive imaginary part.")]
    InvalidHybridizationFunction { n_negative: usize },

    #[error("checkpoint I/O error: {0}")]
    CheckpointIo(#[from] std::io::Error),

    #[error("checkpoint deserialization error: {0}")]
    CheckpointDeser(String),

    #[error("DMRG error")]
    Dmrg(#[from] DmrgError),

    #[error("computation cancelled")]
    Cancelled,
}

/// Result alias for `tk-dmft`.
pub type DmftResult<T> = Result<T, DmftError>;
