//! Property-based tests for tk-linalg using proptest.
//!
//! Tests: GEMM associativity, SVD round-trip, regularized inverse monotonicity,
//! block_gemm output sector flux validity.

use proptest::prelude::*;
use tk_core::{DenseTensor, MatMut, MatRef, TensorShape};
use tk_linalg::{DeviceFaer, LinAlgBackend};
use tk_symmetry::QuantumNumber;

/// Simple LCG for deterministic matrix generation from a seed.
fn fill_matrix(data: &mut [f64], seed: u64) {
    let mut state = seed;
    for val in data.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *val = ((state >> 33) as i32 as f64) / (1 << 30) as f64;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// GEMM associativity: (A*B)*C == A*(B*C) within tolerance.
    #[test]
    fn gemm_associativity(
        m in 2..=8_usize,
        k in 2..=8_usize,
        n in 2..=8_usize,
        p in 2..=8_usize,
        seed in 0..10000_u64,
    ) {
        let backend = DeviceFaer;

        let mut a_data = vec![0.0_f64; m * k];
        let mut b_data = vec![0.0_f64; k * n];
        let mut c_data = vec![0.0_f64; n * p];
        fill_matrix(&mut a_data, seed);
        fill_matrix(&mut b_data, seed.wrapping_add(1));
        fill_matrix(&mut c_data, seed.wrapping_add(2));

        let a = MatRef::from_slice(&a_data, m, k);
        let b = MatRef::from_slice(&b_data, k, n);
        let c_mat = MatRef::from_slice(&c_data, n, p);

        // Compute A*B
        let mut ab_data = vec![0.0_f64; m * n];
        {
            let mut ab = MatMut::from_slice(&mut ab_data, m, n);
            backend.gemm(1.0, &a, &b, 0.0, &mut ab);
        }

        // Compute (A*B)*C
        let ab_ref = MatRef::from_slice(&ab_data, m, n);
        let mut abc_left = vec![0.0_f64; m * p];
        {
            let mut out = MatMut::from_slice(&mut abc_left, m, p);
            backend.gemm(1.0, &ab_ref, &c_mat, 0.0, &mut out);
        }

        // Compute B*C
        let mut bc_data = vec![0.0_f64; k * p];
        {
            let mut bc = MatMut::from_slice(&mut bc_data, k, p);
            backend.gemm(1.0, &b, &c_mat, 0.0, &mut bc);
        }

        // Compute A*(B*C)
        let bc_ref = MatRef::from_slice(&bc_data, k, p);
        let mut abc_right = vec![0.0_f64; m * p];
        {
            let mut out = MatMut::from_slice(&mut abc_right, m, p);
            backend.gemm(1.0, &a, &bc_ref, 0.0, &mut out);
        }

        // Compare
        let max_err: f64 = abc_left.iter()
            .zip(abc_right.iter())
            .map(|(l, r)| (l - r).abs())
            .fold(0.0, f64::max);

        let scale = (m * k * n * p) as f64;
        prop_assert!(max_err < 1e-8 * scale.sqrt(),
            "associativity error {max_err} exceeds tolerance for dims ({m},{k},{n},{p})");
    }

    /// SVD round-trip: ||A - U*diag(s)*Vt||_F / ||A||_F < tolerance.
    #[test]
    fn svd_round_trip(
        m in 2..=16_usize,
        n in 2..=16_usize,
        seed in 0..10000_u64,
    ) {
        let backend = DeviceFaer;

        let mut data = vec![0.0_f64; m * n];
        fill_matrix(&mut data, seed);

        let mat = MatRef::from_slice(&data, m, n);
        let result = backend.svd_truncated(&mat, m.min(n), 0.0)
            .expect("SVD should succeed");

        // Compute ||A||_F
        let norm_a: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-14 {
            return Ok(()); // skip near-zero matrices
        }

        // Reconstruct and compute residual
        let u_ref = result.u.as_mat_ref().unwrap();
        let vt_ref = result.vt.as_mat_ref().unwrap();

        let mut residual_sq = 0.0_f64;
        for r in 0..m {
            for c in 0..n {
                let original = data[r * n + c];
                let mut approx = 0.0;
                for kk in 0..result.rank {
                    approx += u_ref.get(r, kk) * result.singular_values[kk] * vt_ref.get(kk, c);
                }
                residual_sq += (original - approx) * (original - approx);
            }
        }
        let rel_err = residual_sq.sqrt() / norm_a;

        prop_assert!(rel_err < 1e-10,
            "SVD reconstruction relative error {rel_err} for {m}x{n} matrix");
    }

    /// Regularized inverse: smaller delta -> result closer to true inverse.
    #[test]
    fn regularized_inverse_decreasing_delta(
        s_val in 0.1..100.0_f64,
    ) {
        let backend = DeviceFaer;

        let s_values = vec![s_val];
        let u = DenseTensor::from_vec(TensorShape::row_major(&[1, 1]), vec![1.0_f64]);
        let vt = DenseTensor::from_vec(TensorShape::row_major(&[1, 1]), vec![1.0_f64]);

        let delta_large = 1.0;
        let delta_small = 1e-6;

        let inv_large = backend.regularized_svd_inverse(&s_values, &u, &vt, delta_large);
        let inv_small = backend.regularized_svd_inverse(&s_values, &u, &vt, delta_small);

        let val_large = inv_large.as_mat_ref().unwrap().get(0, 0);
        let val_small = inv_small.as_mat_ref().unwrap().get(0, 0);
        let true_inv = 1.0 / s_val;

        let err_large = (val_large - true_inv).abs();
        let err_small = (val_small - true_inv).abs();

        prop_assert!(err_small <= err_large + 1e-12,
            "smaller delta should give closer to true inverse: err_small={err_small}, err_large={err_large}");
    }

    /// Block GEMM output sectors are a subset of valid sectors (flux rule satisfied).
    #[test]
    fn block_gemm_output_sectors_valid(
        n_sectors in 2..=4_usize,
    ) {
        use tk_symmetry::{BlockSparseTensor, LegDirection, QIndex, U1};
        use tk_linalg::SparseLinAlgBackend;

        let backend = DeviceFaer;

        let sectors: Vec<(U1, usize)> = (0..n_sectors as i32)
            .map(|q| (U1(q), 3))
            .collect();
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

        let result = backend.block_gemm(&a, &b);

        for (qns, _block) in result.iter_blocks() {
            let fused = qns[0].fuse(&qns[1].dual());
            let target = result.flux();
            prop_assert_eq!(&fused, target,
                "output sector {:?} violates flux rule", qns);
        }
    }
}
