//! Spectral positivity restoration.
//!
//! Clamps negative spectral weight and renormalizes to preserve the sum rule.
//! Mandatory post-deconvolution pass (design doc Section 8.4.2).

use crate::spectral::SpectralFunction;
use crate::spectral::linear_predict::LinearPredictionConfig;

/// Clamp negative spectral weight and renormalize to preserve the sum rule.
///
/// Steps:
/// 1. Compute fraction of negative spectral weight. Warn if > threshold.
/// 2. Record Fermi-level value before clamping.
/// 3. Clamp A(omega) >= positivity_floor and L1 renormalize.
/// 4. Check Fermi-level distortion. Warn if > tolerance.
///
/// This function never fails. All diagnostic conditions emit log warnings only.
pub fn restore_positivity(
    spectral: &SpectralFunction,
    config: &LinearPredictionConfig,
) -> SpectralFunction {
    let d_omega = spectral.d_omega;

    // Step 1: Diagnostic — measure negative weight
    let mut w_neg: f64 = 0.0;
    let mut w_total: f64 = 0.0;
    for &v in &spectral.values {
        w_total += v.abs() * d_omega;
        if v < 0.0 {
            w_neg += v.abs() * d_omega;
        }
    }

    if w_total > f64::EPSILON && w_neg / w_total > config.positivity_warning_threshold {
        log::warn!(
            target: "tensorkraft::telemetry",
            "SPECTRAL_POSITIVITY_WARNING: {:.1}% of spectral weight is negative \
             (threshold: {:.1}%). Deconvolution parameters (eta={}, delta={}, omega_max={}) \
             may need adjustment.",
            100.0 * w_neg / w_total,
            100.0 * config.positivity_warning_threshold,
            config.broadening_eta,
            config.deconv_tikhonov_delta,
            config.deconv_omega_max,
        );
    }

    // Step 2: Record Fermi-level value before clamping
    let a_fermi_before = if spectral.omega.first().copied().unwrap_or(1.0) <= 0.0
        && spectral.omega.last().copied().unwrap_or(-1.0) >= 0.0
    {
        spectral.value_at_omega_zero()
    } else {
        0.0
    };

    // Step 3: Clamp and L1 renormalize
    let mut clamped: Vec<f64> = spectral
        .values
        .iter()
        .map(|&v| v.max(config.positivity_floor))
        .collect();

    // Compute original sum rule: the signed integral (what we want to preserve)
    let original_sum = spectral.sum_rule();

    // Compute clamped sum (always positive since all values are >= positivity_floor)
    let clamped_spec_tmp = SpectralFunction::new(spectral.omega.clone(), clamped.clone());
    let clamped_sum = clamped_spec_tmp.sum_rule();

    // Only rescale if both sums are positive (which should be true for physical
    // spectral functions where the positive weight dominates)
    if clamped_sum > f64::EPSILON && original_sum > f64::EPSILON {
        let scale = original_sum / clamped_sum;
        for v in &mut clamped {
            *v *= scale;
        }
    }

    let result = SpectralFunction::new(spectral.omega.clone(), clamped);

    // Step 4: Fermi-level distortion check
    if a_fermi_before.abs() > f64::EPSILON
        && spectral.omega.first().copied().unwrap_or(1.0) <= 0.0
        && spectral.omega.last().copied().unwrap_or(-1.0) >= 0.0
    {
        let a_fermi_after = result.value_at_omega_zero();
        let relative_shift = (a_fermi_after - a_fermi_before).abs() / a_fermi_before.abs();
        if relative_shift > config.fermi_level_shift_tolerance {
            log::warn!(
                target: "tensorkraft::telemetry",
                "FERMI_LEVEL_DISTORTION: A(omega=0) shifted by {:.1}% after positivity \
                 restoration (tolerance: {:.1}%). Before: {:.3e}, After: {:.3e}. \
                 This may corrupt the quasiparticle residue. Consider reducing eta or \
                 increasing omega_max to reduce tail ringing.",
                100.0 * relative_shift,
                100.0 * config.fermi_level_shift_tolerance,
                a_fermi_before,
                a_fermi_after,
            );
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positivity_nonneg() {
        let omega: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        // Include some negative values
        let values: Vec<f64> = omega.iter().map(|&w| {
            if w.abs() < 1.0 {
                -0.1
            } else {
                0.5 / (w * w + 0.25)
            }
        }).collect();
        let spec = SpectralFunction::new(omega, values);
        let config = LinearPredictionConfig::default();
        let restored = restore_positivity(&spec, &config);
        for &v in &restored.values {
            assert!(v >= 0.0, "negative value found: {}", v);
        }
    }

    #[test]
    fn test_positivity_sum_rule_preserved() {
        let omega: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        let values: Vec<f64> = omega.iter().map(|&w| {
            let eta = 0.5;
            eta / (std::f64::consts::PI * (w * w + eta * eta))
        }).collect();
        let spec = SpectralFunction::new(omega, values);
        let original_sr = spec.sum_rule();
        let config = LinearPredictionConfig::default();
        let restored = restore_positivity(&spec, &config);
        let restored_sr = restored.sum_rule();
        assert!(
            (original_sr - restored_sr).abs() < 1e-10,
            "sum rule changed: {} -> {}",
            original_sr,
            restored_sr
        );
    }

    #[test]
    fn test_positivity_idempotent() {
        // Mostly positive Lorentzian with small negative ringing in the tails
        let omega: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        let eta = 0.5;
        let values: Vec<f64> = omega.iter().map(|&w| {
            let base = eta / (std::f64::consts::PI * (w * w + eta * eta));
            // Add small negative ringing at |w| ~ 3
            if (w.abs() - 3.0).abs() < 0.5 {
                base - 0.02
            } else {
                base
            }
        }).collect();
        let spec = SpectralFunction::new(omega, values);
        let config = LinearPredictionConfig::default();
        let r1 = restore_positivity(&spec, &config);
        let r2 = restore_positivity(&r1, &config);
        for (a, b) in r1.values.iter().zip(r2.values.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "not idempotent: {} vs {}",
                a,
                b
            );
        }
    }
}
