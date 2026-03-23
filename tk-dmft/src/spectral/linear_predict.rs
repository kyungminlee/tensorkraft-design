//! Linear prediction via Levinson-Durbin recursion, FFT, and Lorentzian deconvolution.
//!
//! Pipeline stages (design doc Section 8.4.2):
//!   1. Exponential windowing: G(t) -> G(t) * exp(-eta|t|)
//!   2. Toeplitz prediction solve (Levinson-Durbin or SVD)
//!   3. FFT -> A_windowed(omega)
//!   4. Regularized Lorentzian deconvolution -> A_raw(omega)  [if eta > 0]
//!   5. Spectral positivity restoration -> A(omega)  [always mandatory]

use num_complex::Complex;
use num_traits::Zero;
use rustfft::FftPlanner;

use crate::error::{DmftError, DmftResult};
use crate::spectral::SpectralFunction;

/// Solver for the P x P Toeplitz prediction system in linear prediction.
#[derive(Clone, Debug)]
pub enum ToeplitzSolver {
    /// O(P^2) Levinson-Durbin recursion with Tikhonov regularization.
    LevinsonDurbin { tikhonov_lambda: f64 },
    /// O(P^3) SVD-based pseudo-inverse. Fallback for non-Toeplitz extensions.
    SvdPseudoInverse { svd_noise_floor: f64 },
}

impl Default for ToeplitzSolver {
    fn default() -> Self {
        ToeplitzSolver::LevinsonDurbin {
            tikhonov_lambda: 1e-8,
        }
    }
}

/// Configuration for the full linear prediction pipeline.
#[derive(Clone, Debug)]
pub struct LinearPredictionConfig {
    /// Solver for the Toeplitz prediction system.
    pub toeplitz_solver: ToeplitzSolver,
    /// Prediction order P (number of past time points used). Default: 100.
    pub prediction_order: usize,
    /// Factor by which to extend G(t) beyond the TDVP simulation time.
    /// Default: 4.0.
    pub extrapolation_factor: f64,
    /// Exponential broadening parameter eta for windowing G(t).
    /// Default: 0.0 (disabled).
    pub broadening_eta: f64,
    /// Tikhonov regularization delta for the Lorentzian deconvolution denominator.
    /// Default: 1e-3.
    pub deconv_tikhonov_delta: f64,
    /// Hard cutoff frequency for deconvolution. Default: 10.0.
    pub deconv_omega_max: f64,
    /// Noise floor for spectral positivity clamping. Default: 1e-15.
    pub positivity_floor: f64,
    /// Warning threshold for negative spectral weight fraction. Default: 0.05 (5%).
    pub positivity_warning_threshold: f64,
    /// Fermi-level distortion tolerance. Default: 0.01 (1%).
    pub fermi_level_shift_tolerance: f64,
}

impl Default for LinearPredictionConfig {
    fn default() -> Self {
        Self {
            toeplitz_solver: ToeplitzSolver::default(),
            prediction_order: 100,
            extrapolation_factor: 4.0,
            broadening_eta: 0.0,
            deconv_tikhonov_delta: 1e-3,
            deconv_omega_max: 10.0,
            positivity_floor: 1e-15,
            positivity_warning_threshold: 0.05,
            fermi_level_shift_tolerance: 0.01,
        }
    }
}

/// Solve a Toeplitz system using Levinson-Durbin recursion.
///
/// Given autocorrelation values r[0..=P], solves R * a = r for prediction
/// coefficients a, where R_{ij} = r[|i-j|].
///
/// Returns the prediction coefficients a[1..=P].
///
/// # Parameters
/// - `autocorr`: autocorrelation r[0], r[1], ..., r[P] (length P+1)
/// - `tikhonov_lambda`: regularization added to diagonal
///
/// # Errors
/// Returns `DmftError::LinearPredictionFailed` if the recursion diverges.
pub fn solve_toeplitz_levinson_durbin(
    autocorr: &[Complex<f64>],
    tikhonov_lambda: f64,
) -> DmftResult<Vec<Complex<f64>>> {
    let p = autocorr.len() - 1;
    if p == 0 {
        return Ok(vec![]);
    }

    let r0 = autocorr[0] + Complex::new(tikhonov_lambda, 0.0);

    if r0.norm() < f64::EPSILON {
        return Err(DmftError::LinearPredictionFailed { condition: f64::INFINITY });
    }

    // Levinson-Durbin recursion
    let mut a: Vec<Complex<f64>> = vec![autocorr[1] / r0];
    let mut err = r0 - autocorr[1] * autocorr[1] / r0;

    for n in 1..p {
        // Compute reflection coefficient
        let mut num = autocorr[n + 1];
        for k in 0..n {
            num = num + a[k] * autocorr[n - k];
        }

        if err.norm() < f64::EPSILON * 1e-6 {
            return Err(DmftError::LinearPredictionFailed {
                condition: r0.norm() / err.norm(),
            });
        }

        let kappa = -num / err;

        // Update coefficients
        let mut a_new = vec![Complex::zero(); n + 1];
        for k in 0..n {
            a_new[k] = a[k] + kappa * a[n - 1 - k].conj();
        }
        a_new[n] = kappa;

        err = err * (Complex::new(1.0, 0.0) - kappa * kappa.conj());
        a = a_new;
    }

    Ok(a)
}

/// Apply exponential windowing and run Toeplitz linear prediction to
/// extrapolate G(t) to `extrapolation_factor * t_max`.
///
/// # Parameters
/// - `g_t`: complex Green's function samples at uniform time steps dt
/// - `dt`: physical time step
/// - `config`: linear prediction configuration
///
/// # Returns
/// Extended time series with length ~ `g_t.len() * config.extrapolation_factor`.
pub fn linear_predict_regularized(
    g_t: &[Complex<f64>],
    dt: f64,
    config: &LinearPredictionConfig,
) -> DmftResult<Vec<Complex<f64>>> {
    let n = g_t.len();
    if n == 0 {
        return Ok(vec![]);
    }

    let eta = config.broadening_eta;
    let p = config.prediction_order.min(n / 2);

    // Step 1: Apply exponential windowing
    let windowed: Vec<Complex<f64>> = g_t
        .iter()
        .enumerate()
        .map(|(k, &g)| {
            if eta > 0.0 {
                let t = k as f64 * dt;
                g * (-eta * t).exp()
            } else {
                g
            }
        })
        .collect();

    // Step 2: Compute autocorrelation for Toeplitz system
    let mut autocorr = vec![Complex::zero(); p + 1];
    for lag in 0..=p {
        for j in 0..(n - lag) {
            autocorr[lag] = autocorr[lag] + windowed[j].conj() * windowed[j + lag];
        }
    }

    // Step 3: Solve for prediction coefficients
    let coeffs = match &config.toeplitz_solver {
        ToeplitzSolver::LevinsonDurbin { tikhonov_lambda } => {
            solve_toeplitz_levinson_durbin(&autocorr, *tikhonov_lambda)?
        }
        ToeplitzSolver::SvdPseudoInverse { svd_noise_floor: _ } => {
            // Fallback: use Levinson-Durbin with default regularization
            // TODO: Implement SVD-based pseudo-inverse for non-Toeplitz systems
            solve_toeplitz_levinson_durbin(&autocorr, 1e-8)?
        }
    };

    // Step 4: Extrapolate
    let target_len = (n as f64 * config.extrapolation_factor) as usize;
    let mut extended = windowed;
    extended.reserve(target_len - n);

    for _ in n..target_len {
        let mut predicted = Complex::zero();
        let current_len = extended.len();
        for (k, &coeff) in coeffs.iter().enumerate() {
            if k < current_len {
                predicted = predicted + coeff * extended[current_len - 1 - k];
            }
        }
        extended.push(predicted);
    }

    Ok(extended)
}

/// FFT the extended G(t) to obtain A_windowed(omega) = -Im[G(omega)] / pi.
///
/// Uses `rustfft::FftPlanner` for the DFT.
///
/// # Parameters
/// - `g_t_extended`: extrapolated Green's function
/// - `dt`: physical time step
/// - `omega`: target frequency grid (must be uniform)
///
/// # Returns
/// `SpectralFunction` with values = -Im[G(omega)] / pi on the given `omega` grid.
pub fn fft_to_spectral(
    g_t_extended: &[Complex<f64>],
    dt: f64,
    omega: &[f64],
) -> SpectralFunction {
    let n = g_t_extended.len();
    if n == 0 {
        return SpectralFunction::new(omega.to_vec(), vec![0.0; omega.len()]);
    }

    // Perform FFT
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = g_t_extended.to_vec();
    fft.process(&mut buffer);

    // Convert FFT output to frequency-domain:
    // FFT gives G(omega_k) where omega_k = 2*pi*k/(N*dt) for k = 0..N-1
    // We need to map these onto the user's omega grid
    let d_omega_fft = 2.0 * std::f64::consts::PI / (n as f64 * dt);

    // Build spectral function by interpolating onto the target omega grid
    let pi = std::f64::consts::PI;
    let values: Vec<f64> = omega
        .iter()
        .map(|&w| {
            // Map omega to FFT bin index
            let fft_index_f = w / d_omega_fft;
            // Handle negative frequencies via periodicity
            let fft_index_wrapped = if fft_index_f >= 0.0 {
                fft_index_f
            } else {
                fft_index_f + n as f64
            };

            // Linear interpolation between bins
            let i0 = fft_index_wrapped.floor() as usize % n;
            let i1 = (i0 + 1) % n;
            let frac = fft_index_wrapped.fract();

            let g_w = buffer[i0] * (1.0 - frac) + buffer[i1] * frac;
            // A(omega) = -Im[G(omega)] / pi, scaled by dt for FFT normalization
            let spectral_val = -g_w.im * dt / pi;
            spectral_val
        })
        .collect();

    SpectralFunction::new(omega.to_vec(), values)
}

/// Apply regularized Lorentzian deconvolution to remove broadening eta.
///
/// Regularized deconvolution formula:
///   A_true(omega) ~ A_windowed(omega) * (eta^2 + omega^2) / (2*eta + delta * omega^2)
///
/// # Errors
/// Returns `DmftError::DeconvolutionFailed` if `broadening_eta == 0.0`.
pub fn deconvolve_lorentzian(
    spectral: &SpectralFunction,
    config: &LinearPredictionConfig,
) -> DmftResult<SpectralFunction> {
    let eta = config.broadening_eta;
    if eta <= 0.0 {
        return Err(DmftError::DeconvolutionFailed { eta });
    }

    let delta = config.deconv_tikhonov_delta;
    let omega_max = config.deconv_omega_max;

    let values: Vec<f64> = spectral
        .omega
        .iter()
        .zip(spectral.values.iter())
        .map(|(&w, &a)| {
            if w.abs() > omega_max {
                // Beyond cutoff: no correction
                a
            } else {
                // Deconvolution factor: (eta^2 + w^2) / (2*eta + delta * w^2)
                let num = eta * eta + w * w;
                let den = 2.0 * eta + delta * w * w;
                if den.abs() < f64::EPSILON {
                    a
                } else {
                    a * num / den
                }
            }
        })
        .collect();

    Ok(SpectralFunction::new(spectral.omega.clone(), values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levinson_durbin_trivial() {
        // Autocorrelation of a white noise: r[0]=1, r[k]=0 for k>0
        let autocorr = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let result = solve_toeplitz_levinson_durbin(&autocorr, 0.0);
        assert!(result.is_ok());
        let coeffs = result.unwrap();
        assert_eq!(coeffs.len(), 2);
        // For white noise, prediction coefficients should be zero
        for c in &coeffs {
            assert!(c.norm() < 1e-10, "coeff = {:?}", c);
        }
    }

    #[test]
    fn test_levinson_durbin_single_coefficient() {
        let autocorr = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.5, 0.0),
        ];
        let result = solve_toeplitz_levinson_durbin(&autocorr, 0.0);
        assert!(result.is_ok());
        let coeffs = result.unwrap();
        assert_eq!(coeffs.len(), 1);
        // a[0] = r[1]/r[0] = 0.5
        assert!((coeffs[0].re - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_deconvolution_requires_eta() {
        let spec = SpectralFunction::new(vec![0.0], vec![1.0]);
        let config = LinearPredictionConfig {
            broadening_eta: 0.0,
            ..Default::default()
        };
        let result = deconvolve_lorentzian(&spec, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_deconvolution_factor_at_zero() {
        let spec = SpectralFunction::new(vec![0.0], vec![1.0]);
        let eta = 0.1;
        let config = LinearPredictionConfig {
            broadening_eta: eta,
            deconv_tikhonov_delta: 1e-3,
            deconv_omega_max: 10.0,
            ..Default::default()
        };
        let result = deconvolve_lorentzian(&spec, &config).unwrap();
        // At omega=0: factor = eta^2 / (2*eta) = eta/2
        let expected = 1.0 * eta / 2.0;
        assert!((result.values[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_fft_to_spectral_empty() {
        let result = fft_to_spectral(&[], 0.1, &[0.0, 1.0]);
        assert_eq!(result.values, vec![0.0, 0.0]);
    }

    #[test]
    fn test_default_configs() {
        let config = LinearPredictionConfig::default();
        assert_eq!(config.prediction_order, 100);
        assert!((config.extrapolation_factor - 4.0).abs() < 1e-12);
        assert!((config.broadening_eta - 0.0).abs() < 1e-12);

        let solver = ToeplitzSolver::default();
        match solver {
            ToeplitzSolver::LevinsonDurbin { tikhonov_lambda } => {
                assert!((tikhonov_lambda - 1e-8).abs() < 1e-15);
            }
            _ => panic!("expected LevinsonDurbin"),
        }
    }
}
