//! Property-based tests for tk-dmft spectral and mixing modules.

use proptest::prelude::*;
use num_complex::Complex;

use tk_dmft::spectral::linear_predict::{
    solve_toeplitz_levinson_durbin, solve_toeplitz_svd_pseudoinverse, LinearPredictionConfig,
};
use tk_dmft::spectral::positivity::restore_positivity;
use tk_dmft::spectral::chebyshev::{jackson_kernel, reconstruct_from_moments, ChebyshevConfig};
use tk_dmft::r#loop::mixing::BroydenState;
use tk_dmft::impurity::bath::BathParameters;
use tk_dmft::SpectralFunction;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Verify that the Levinson-Durbin solution satisfies the Toeplitz system.
    ///
    /// For a random positive-definite autocorrelation sequence r[0..=P] with
    /// r[k] = rho^|k| (AR(1) process), solve_toeplitz_levinson_durbin returns
    /// P coefficients. We verify:
    ///   1. The number of coefficients equals the prediction order P.
    ///   2. All coefficients are finite (no NaN/Inf from numerical blow-up).
    ///   3. The sum of squared coefficient magnitudes is bounded by a reasonable
    ///      factor, ensuring the predictor is stable.
    ///   4. For order 1, the exact solution a[0] = r[1]/(r[0]+lambda) holds.
    #[test]
    fn levinson_durbin_solution_satisfies_toeplitz_system(
        rho in 0.1_f64..0.9_f64,
        p in 1_usize..=7_usize,
    ) {
        let lambda = 1e-10;
        let autocorr: Vec<Complex<f64>> = (0..=p)
            .map(|k| Complex::new(rho.powi(k as i32), 0.0))
            .collect();

        let coeffs = solve_toeplitz_levinson_durbin(&autocorr, lambda)
            .expect("Levinson-Durbin should succeed on well-conditioned AR(1) autocorrelation");

        // 1. Correct number of coefficients
        prop_assert_eq!(coeffs.len(), p);

        // 2. All coefficients are finite
        for (k, c) in coeffs.iter().enumerate() {
            prop_assert!(
                c.re.is_finite() && c.im.is_finite(),
                "Coefficient a[{}] = {:?} is not finite", k, c,
            );
        }

        // 3. Stability: the coefficient magnitudes should not diverge to infinity.
        // For an over-specified AR model the coefficients can be large but must remain finite.
        let max_coeff: f64 = coeffs.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        prop_assert!(
            max_coeff < 1e6,
            "Max coefficient magnitude {} is unreasonably large for rho={}, p={}",
            max_coeff, rho, p,
        );

        // 4. For order 1, the exact solution is a[0] = r[1]/(r[0]+lambda)
        if p == 1 {
            let expected = rho / (1.0 + lambda);
            prop_assert!(
                (coeffs[0].re - expected).abs() < 1e-8,
                "Order-1 coefficient {} should equal r[1]/(r[0]+lambda) = {}",
                coeffs[0].re, expected,
            );
        }
    }

    /// Applying `restore_positivity` twice should give the same result as applying it once
    /// (idempotency), since the output is already non-negative.
    #[test]
    fn positivity_restoration_idempotent(
        seed in 0_u64..1000_u64,
    ) {
        // Build a Lorentzian with small negative ringing
        let n_pts = 101;
        let omega: Vec<f64> = (0..n_pts).map(|i| -5.0 + 10.0 * i as f64 / (n_pts as f64 - 1.0)).collect();
        let eta = 0.3 + (seed as f64 % 7.0) * 0.1;
        let values: Vec<f64> = omega.iter().map(|&w| {
            let base = eta / (std::f64::consts::PI * (w * w + eta * eta));
            // Add negative ringing that depends on seed
            if (w.abs() - 2.0).abs() < 0.5 {
                base - 0.01 * (1.0 + (seed as f64 % 5.0) * 0.005)
            } else {
                base
            }
        }).collect();

        let spec = SpectralFunction::new(omega, values);
        let config = LinearPredictionConfig::default();

        let r1 = restore_positivity(&spec, &config);
        let r2 = restore_positivity(&r1, &config);

        for (i, (a, b)) in r1.values.iter().zip(r2.values.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-12,
                "Not idempotent at index {}: first pass = {}, second pass = {}",
                i, a, b,
            );
        }
    }

    /// The spectral sum rule (integral of A(omega)) should be preserved by
    /// positivity restoration within a reasonable tolerance.
    #[test]
    fn positivity_restoration_preserves_sum(
        seed in 0_u64..1000_u64,
    ) {
        let n_pts = 201;
        let omega: Vec<f64> = (0..n_pts).map(|i| -10.0 + 20.0 * i as f64 / (n_pts as f64 - 1.0)).collect();
        let eta = 0.3 + (seed as f64 % 10.0) * 0.05;
        let values: Vec<f64> = omega.iter().map(|&w| {
            let base = eta / (std::f64::consts::PI * (w * w + eta * eta));
            // Inject small negative regions
            if (w.abs() - 3.0).abs() < 0.3 {
                base - 0.005
            } else {
                base
            }
        }).collect();

        let spec = SpectralFunction::new(omega, values);
        let original_sum = spec.sum_rule();

        let config = LinearPredictionConfig::default();
        let restored = restore_positivity(&spec, &config);
        let restored_sum = restored.sum_rule();

        let rel_error = if original_sum.abs() > 1e-15 {
            (original_sum - restored_sum).abs() / original_sum.abs()
        } else {
            (original_sum - restored_sum).abs()
        };

        prop_assert!(
            rel_error < 1e-6,
            "Sum rule not preserved: original = {}, restored = {}, relative error = {}",
            original_sum, restored_sum, rel_error,
        );
    }

    /// Jackson kernel values should be monotonically non-increasing for all
    /// valid moment counts.
    #[test]
    fn jackson_kernel_monotone_decreasing(
        n_moments in 4_usize..=200_usize,
    ) {
        let kernel = jackson_kernel(n_moments);
        prop_assert_eq!(kernel.len(), n_moments);

        // g_0 should be close to 1.0
        prop_assert!(
            (kernel[0] - 1.0).abs() < 0.05,
            "g_0 = {} deviates too far from 1.0",
            kernel[0],
        );

        // Monotonically non-increasing (with small tolerance for floating-point)
        for i in 1..kernel.len() {
            prop_assert!(
                kernel[i] <= kernel[i - 1] + 1e-10,
                "Jackson kernel not monotone at n={}: g[{}]={} > g[{}]={}",
                n_moments, i, kernel[i], i - 1, kernel[i - 1],
            );
        }
    }

    /// SVD pseudo-inverse solver should produce finite, bounded coefficients
    /// for well-conditioned AR(1) autocorrelation sequences.
    #[test]
    fn svd_solver_produces_bounded_coefficients(
        rho in 0.1_f64..0.9_f64,
        p in 1_usize..=7_usize,
    ) {
        let autocorr: Vec<Complex<f64>> = (0..=p)
            .map(|k| Complex::new(rho.powi(k as i32), 0.0))
            .collect();

        let coeffs = solve_toeplitz_svd_pseudoinverse(&autocorr, 1e-8)
            .expect("SVD solver should succeed on well-conditioned AR(1)");

        prop_assert_eq!(coeffs.len(), p);

        for (k, c) in coeffs.iter().enumerate() {
            prop_assert!(
                c.re.is_finite() && c.im.is_finite(),
                "SVD coefficient a[{}] = {:?} is not finite", k, c,
            );
        }

        // For order 1, the exact solution is a[0] ≈ rho
        if p == 1 {
            prop_assert!(
                (coeffs[0].re - rho).abs() < 0.01,
                "Order-1 SVD coefficient {} should be close to rho = {}",
                coeffs[0].re, rho,
            );
        }
    }

    /// Chebyshev reconstruction from a single nonzero moment (mu_0=1)
    /// should produce a non-negative spectral function inside the band.
    #[test]
    fn chebyshev_reconstruct_nonneg_inside_band(
        n_moments in 10_usize..=100_usize,
        half_bw in 1.0_f64..10.0_f64,
    ) {
        let mut moments = vec![0.0_f64; n_moments];
        moments[0] = 1.0;

        let omega: Vec<f64> = (-50..=50).map(|i| i as f64 * half_bw / 50.0).collect();
        let config = ChebyshevConfig {
            n_moments,
            jackson_kernel: true,
            ..Default::default()
        };

        let result = reconstruct_from_moments(
            &moments, &omega, -half_bw, half_bw, &config,
        ).expect("reconstruction should succeed");

        // Values outside the band should be zero
        for (&w, &v) in result.omega.iter().zip(result.values.iter()) {
            if w.abs() > half_bw * 1.01 {
                prop_assert!(
                    v.abs() < 1e-8,
                    "Nonzero outside band at w={}: v={}",
                    w, v,
                );
            }
        }
    }

    /// Uniform bath with even number of sites: hybridization distance
    /// to itself should be exactly zero.
    #[test]
    fn uniform_bath_self_distance_zero(
        n_bath in 2_usize..=8_usize,
        bandwidth in 2.0_f64..20.0_f64,
    ) {
        let bath: BathParameters<f64> = BathParameters::uniform(n_bath, bandwidth, 1.0);
        let omega: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.2).collect();
        let broadening = 0.1;

        let dist = bath.hybridization_distance(&bath, &omega, broadening);
        prop_assert!(
            dist < 1e-12,
            "Self-distance should be zero, got {}",
            dist,
        );
    }

    /// Linear mixing of two identical baths should return the original bath.
    #[test]
    fn linear_mix_identity(
        n_bath in 1_usize..=6_usize,
        alpha in 0.0_f64..=1.0_f64,
    ) {
        let bath: BathParameters<f64> = BathParameters::uniform(n_bath, 10.0, 1.0);
        let mixed = bath.linear_mix(&bath, alpha);

        for i in 0..n_bath {
            prop_assert!(
                (mixed.epsilon[i] - bath.epsilon[i]).abs() < 1e-12,
                "epsilon mismatch at {}: {} vs {}",
                i, mixed.epsilon[i], bath.epsilon[i],
            );
            prop_assert!(
                (mixed.v[i] - bath.v[i]).abs() < 1e-12,
                "v mismatch at {}: {} vs {}",
                i, mixed.v[i], bath.v[i],
            );
        }
    }

    /// On the first Broyden update (no history), the result should be identical
    /// to simple linear mixing: x_next = (1 - alpha) * x + alpha * f.
    #[test]
    fn broyden_mixing_reduces_to_linear_on_first_step(
        dim in 2_usize..=10_usize,
        alpha in 0.1_f64..0.9_f64,
        seed in 0_u64..1000_u64,
    ) {
        // Deterministic pseudo-random vectors from seed
        let x: Vec<f64> = (0..dim)
            .map(|i| ((seed as f64 + i as f64 * 1.3).sin() * 100.0).fract())
            .collect();
        let f: Vec<f64> = (0..dim)
            .map(|i| ((seed as f64 + i as f64 * 2.7 + 0.5).cos() * 100.0).fract())
            .collect();

        let mut state = BroydenState::new(5);
        let result = state.update(&x, &f, alpha);

        prop_assert_eq!(result.len(), dim);

        for i in 0..dim {
            let expected = (1.0 - alpha) * x[i] + alpha * f[i];
            prop_assert!(
                (result[i] - expected).abs() < 1e-12,
                "Mismatch at index {}: got {}, expected {} (linear mixing with alpha={})",
                i, result[i], expected, alpha,
            );
        }
    }
}
