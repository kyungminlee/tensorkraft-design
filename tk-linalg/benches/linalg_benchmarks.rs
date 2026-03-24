//! Criterion benchmarks for tk-linalg performance validation.
//!
//! Benchmarks:
//! - `gemm_f64_100x100`: GEMM throughput (tech spec: >= 90% peak DGEMM FLOP/s)
//! - `svd_truncated_f64_50x50`: SVD latency (tech spec: < 5ms for 200x200)
//! - `compute_fusion_rule`: Fusion rule lookup (tech spec: < 10 ns per call)
//! - `block_gemm_u1_10sectors`: Block-sparse GEMM with LPT scheduling

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tk_core::{MatMut, MatRef};
use tk_linalg::{DeviceFaer, LinAlgBackend, SparseLinAlgBackend};
use tk_symmetry::{BlockSparseTensor, LegDirection, QIndex, U1};

fn bench_gemm_f64(c: &mut Criterion) {
    let backend = DeviceFaer;
    let n = 100;
    let a_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.001).collect();
    let b_data: Vec<f64> = (0..n * n).map(|i| ((i * 7) as f64) * 0.001).collect();
    let mut c_data = vec![0.0_f64; n * n];

    c.bench_function("gemm_f64_100x100", |bench| {
        bench.iter(|| {
            let a = MatRef::from_slice(&a_data, n, n);
            let b = MatRef::from_slice(&b_data, n, n);
            let mut c_mat = MatMut::from_slice(&mut c_data, n, n);
            backend.gemm(1.0, &a, &b, 0.0, &mut c_mat);
            black_box(&c_data);
        })
    });
}

fn bench_svd_truncated_f64(c: &mut Criterion) {
    let backend = DeviceFaer;
    let m = 50;
    let n = 50;
    let data: Vec<f64> = (0..m * n).map(|i| (i as f64) * 0.01).collect();

    c.bench_function("svd_truncated_f64_50x50", |bench| {
        bench.iter(|| {
            let mat = MatRef::from_slice(&data, m, n);
            let result = backend.svd_truncated(&mat, 20, 1e-14).unwrap();
            black_box(result.rank);
        })
    });
}

fn bench_block_gemm_u1(c: &mut Criterion) {
    let backend = DeviceFaer;

    // 10 sectors, D=10 per sector (moderate size for benchmark)
    let sectors: Vec<(U1, usize)> = (0..10).map(|q| (U1(q), 10)).collect();
    let qindex = QIndex::new(sectors);

    let flux = U1(0);
    let dirs = vec![LegDirection::Incoming, LegDirection::Outgoing];

    let a = BlockSparseTensor::<f64, U1>::zeros(
        vec![qindex.clone(), qindex.clone()],
        flux.clone(),
        dirs.clone(),
    );
    let b = BlockSparseTensor::<f64, U1>::zeros(
        vec![qindex.clone(), qindex.clone()],
        flux.clone(),
        dirs.clone(),
    );

    c.bench_function("block_gemm_u1_10sectors_d10", |bench| {
        bench.iter(|| {
            let result = backend.block_gemm(&a, &b);
            black_box(result.n_sectors());
        })
    });
}

fn bench_threading_regime_select(c: &mut Criterion) {
    use tk_linalg::ThreadingRegime;

    let sectors: Vec<(U1, usize)> = (0..10).map(|q| (U1(q), 50)).collect();
    let qindex = QIndex::new(sectors);
    let flux = U1(0);
    let dirs = vec![LegDirection::Incoming, LegDirection::Outgoing];

    let tensor = BlockSparseTensor::<f64, U1>::zeros(
        vec![qindex.clone(), qindex.clone()],
        flux,
        dirs,
    );

    c.bench_function("threading_regime_select", |bench| {
        bench.iter(|| {
            let regime = ThreadingRegime::select(&tensor, 8);
            black_box(regime);
        })
    });
}

criterion_group!(
    benches,
    bench_gemm_f64,
    bench_svd_truncated_f64,
    bench_block_gemm_u1,
    bench_threading_regime_select,
);
criterion_main!(benches);
