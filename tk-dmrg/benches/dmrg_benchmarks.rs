//! Criterion benchmarks for tk-dmrg components.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tk_dmrg::{
    BondDimensionSchedule, InitialSubspace, IterativeEigensolver, LanczosSolver, SweepSchedule,
};

/// Benchmark LanczosSolver on a 10x10 diagonal matrix.
fn lanczos_10x10(c: &mut Criterion) {
    let dim = 10;
    let diag: Vec<f64> = (0..dim).map(|i| (i + 1) as f64).collect();
    let solver = LanczosSolver {
        max_krylov_dim: dim,
        restart_vectors: 2,
        max_iter: 100,
        tol: 1e-10,
    };

    c.bench_function("lanczos_10x10", |b| {
        b.iter(|| {
            let matvec = |x: &[f64], y: &mut [f64]| {
                for (i, (xi, yi)) in x.iter().zip(y.iter_mut()).enumerate() {
                    *yi = diag[i] * xi;
                }
            };
            black_box(solver.lowest_eigenpair(
                black_box(&matvec),
                black_box(dim),
                InitialSubspace::None,
            ))
        })
    });
}

/// Benchmark BondDimensionSchedule::warmup construction.
fn bond_dim_schedule_warmup(c: &mut Criterion) {
    c.bench_function("bond_dim_schedule_warmup", |b| {
        b.iter(|| {
            black_box(BondDimensionSchedule::warmup(
                black_box(10),
                black_box(1000),
                black_box(20),
            ))
        })
    });
}

/// Benchmark SweepSchedule iteration for 100 sites.
fn sweep_schedule_iter(c: &mut Criterion) {
    let schedule = SweepSchedule::standard(100);

    c.bench_function("sweep_schedule_iter_100", |b| {
        b.iter(|| {
            let count: usize = black_box(&schedule).iter().count();
            black_box(count)
        })
    });
}

criterion_group!(benches, lanczos_10x10, bond_dim_schedule_warmup, sweep_schedule_iter);
criterion_main!(benches);
