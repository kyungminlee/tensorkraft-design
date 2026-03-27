//! Benchmarks for tk-dmft hot-path operations.
//!
//! Run with: cargo bench -p tk-dmft

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_complex::Complex;

use tk_dmft::spectral::chebyshev::{jackson_kernel, reconstruct_from_moments, ChebyshevConfig};
use tk_dmft::spectral::linear_predict::{
    fft_to_spectral, solve_toeplitz_levinson_durbin, solve_toeplitz_svd_pseudoinverse,
    LinearPredictionConfig,
};
use tk_dmft::spectral::positivity::restore_positivity;
use tk_dmft::impurity::bath::BathParameters;
use tk_dmft::SpectralFunction;

fn bench_levinson_durbin(c: &mut Criterion) {
    let p = 100;
    let rho = 0.7_f64;
    let autocorr: Vec<Complex<f64>> = (0..=p)
        .map(|k| Complex::new(rho.powi(k as i32), 0.0))
        .collect();

    c.bench_function("levinson_durbin_p100", |b| {
        b.iter(|| solve_toeplitz_levinson_durbin(black_box(&autocorr), 1e-8))
    });
}

fn bench_svd_pseudoinverse(c: &mut Criterion) {
    let p = 100;
    let rho = 0.7_f64;
    let autocorr: Vec<Complex<f64>> = (0..=p)
        .map(|k| Complex::new(rho.powi(k as i32), 0.0))
        .collect();

    c.bench_function("svd_pseudoinverse_p100", |b| {
        b.iter(|| solve_toeplitz_svd_pseudoinverse(black_box(&autocorr), 1e-8))
    });
}

fn bench_jackson_kernel(c: &mut Criterion) {
    c.bench_function("jackson_kernel_1000", |b| {
        b.iter(|| jackson_kernel(black_box(1000)))
    });
}

fn bench_chebyshev_reconstruct(c: &mut Criterion) {
    let n_moments = 1000;
    let mut moments = vec![0.0; n_moments];
    moments[0] = 1.0;
    // Add a few more moments for realism
    for k in 1..n_moments {
        moments[k] = (-0.01 * k as f64).exp() * (0.5_f64).powi(k as i32);
    }

    let omega: Vec<f64> = (-500..=500).map(|i| i as f64 * 0.02).collect();
    let config = ChebyshevConfig {
        n_moments,
        jackson_kernel: true,
        ..Default::default()
    };

    c.bench_function("chebyshev_reconstruct_1000_moments", |b| {
        b.iter(|| {
            reconstruct_from_moments(
                black_box(&moments),
                black_box(&omega),
                -10.0,
                10.0,
                &config,
            )
        })
    });
}

fn bench_positivity_restoration(c: &mut Criterion) {
    let n = 1001;
    let omega: Vec<f64> = (0..n)
        .map(|i| -10.0 + 20.0 * i as f64 / (n as f64 - 1.0))
        .collect();
    let eta = 0.5;
    let values: Vec<f64> = omega
        .iter()
        .map(|&w| {
            let base = eta / (std::f64::consts::PI * (w * w + eta * eta));
            // Inject small negative ringing
            if (w.abs() - 3.0).abs() < 0.5 {
                base - 0.02
            } else {
                base
            }
        })
        .collect();
    let spec = SpectralFunction::new(omega, values);
    let config = LinearPredictionConfig::default();

    c.bench_function("positivity_restore_1001pts", |b| {
        b.iter(|| restore_positivity(black_box(&spec), &config))
    });
}

fn bench_fft_to_spectral(c: &mut Criterion) {
    let n = 2000;
    let dt = 0.05;
    let g_t: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            let t = k as f64 * dt;
            Complex::new((-0.1 * t).exp() * (2.0 * t).cos(), (-0.1 * t).exp() * (2.0 * t).sin())
        })
        .collect();
    let omega: Vec<f64> = (-500..=500).map(|i| i as f64 * 0.02).collect();

    c.bench_function("fft_to_spectral_2000pts", |b| {
        b.iter(|| fft_to_spectral(black_box(&g_t), dt, &omega))
    });
}

fn bench_hybridization_function(c: &mut Criterion) {
    let bath: BathParameters<f64> = BathParameters::uniform(8, 10.0, 1.0);
    let omega: Vec<f64> = (0..2000)
        .map(|i| -10.0 + 20.0 * i as f64 / 1999.0)
        .collect();

    c.bench_function("hybridization_8bath_2000omega", |b| {
        b.iter(|| bath.hybridization_function(black_box(&omega), 0.05))
    });
}

criterion_group!(
    benches,
    bench_levinson_durbin,
    bench_svd_pseudoinverse,
    bench_jackson_kernel,
    bench_chebyshev_reconstruct,
    bench_positivity_restoration,
    bench_fft_to_spectral,
    bench_hybridization_function,
);
criterion_main!(benches);
