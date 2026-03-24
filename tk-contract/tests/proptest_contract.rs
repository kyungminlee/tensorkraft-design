//! Property-based tests for tk-contract using proptest.
//!
//! Tests:
//! - Contraction result matches manual GEMM reference for random M×K×N
//! - Optimizer cost is non-negative for random specs
//! - DP optimizer cost ≤ greedy optimizer cost (DP is optimal)
//! - ContractionSpec::validate_dimensions catches all mismatches

use hashbrown::HashMap;
use proptest::prelude::*;
use tk_contract::{
    ContractionExecutor, ContractionSpec, CostMetric, DPOptimizer, GreedyOptimizer,
    IndexId, IndexMap, IndexSpec, PathOptimizer, TensorId, TreeSAOptimizer,
};
use tk_core::{DenseTensor, MatMut, MatRef, TensorShape};
use tk_linalg::{DeviceFaer, LinAlgBackend};

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

    /// Contraction result matches manual GEMM reference for random M×K×N.
    #[test]
    fn contraction_result_equals_gemm_reference(
        m in 1usize..=8,
        k in 1usize..=8,
        n in 1usize..=8,
        seed in 0..10000_u64,
    ) {
        let backend = DeviceFaer;

        // Build tensors.
        let mut a_data = vec![0.0_f64; m * k];
        let mut b_data = vec![0.0_f64; k * n];
        fill_matrix(&mut a_data, seed);
        fill_matrix(&mut b_data, seed.wrapping_add(1));

        // Manual GEMM reference: C = A * B
        let mut c_ref = vec![0.0_f64; m * n];
        {
            let a_mat = MatRef::from_slice(&a_data, m, k);
            let b_mat = MatRef::from_slice(&b_data, k, n);
            let mut c_mat = MatMut::from_slice(&mut c_ref, m, n);
            backend.gemm(1.0, &a_mat, &b_mat, 0.0, &mut c_mat);
        }

        // Build contraction spec: A(i,j) * B(j,k) → C(i,k)
        let idx_i = IndexId::from_raw(10000);
        let idx_j = IndexId::from_raw(10001);
        let idx_k = IndexId::from_raw(10002);

        let spec = ContractionSpec::new(
            vec![
                (TensorId::new(0), vec![idx_i, idx_j]),
                (TensorId::new(1), vec![idx_j, idx_k]),
            ],
            vec![idx_i, idx_k],
        )
        .unwrap();

        let mut index_map = IndexMap::new();
        index_map.insert(
            TensorId::new(0),
            vec![
                IndexSpec { dim: m, is_contracted: false, is_contiguous: true },
                IndexSpec { dim: k, is_contracted: true, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(1),
            vec![
                IndexSpec { dim: k, is_contracted: true, is_contiguous: true },
                IndexSpec { dim: n, is_contracted: false, is_contiguous: true },
            ],
        );

        let graph = GreedyOptimizer
            .optimize(&spec, &index_map, &CostMetric::default(), None)
            .unwrap();

        let a = DenseTensor::from_vec(TensorShape::row_major(&[m, k]), a_data);
        let b = DenseTensor::from_vec(TensorShape::row_major(&[k, n]), b_data);

        let mut inputs = HashMap::new();
        inputs.insert(TensorId::new(0), &a);
        inputs.insert(TensorId::new(1), &b);

        let executor = ContractionExecutor::new(backend);
        let result = executor.execute(&graph, &inputs).unwrap();

        // Compare: result should match reference within machine epsilon.
        prop_assert_eq!(result.shape().dims(), &[m, n]);
        let result_data = result.as_slice();
        for idx in 0..m * n {
            let diff = (result_data[idx] - c_ref[idx]).abs();
            let scale = c_ref[idx].abs().max(1e-14);
            prop_assert!(
                diff / scale < 1e-10,
                "element {} mismatch: got {}, expected {}, diff={}",
                idx, result_data[idx], c_ref[idx], diff
            );
        }
    }

    /// Optimizer cost is always non-negative.
    #[test]
    fn optimizer_cost_nonnegative(
        d0 in 1usize..=16,
        d1 in 1usize..=16,
        d2 in 1usize..=16,
        d3 in 1usize..=16,
    ) {
        let i = IndexId::from_raw(11000);
        let j = IndexId::from_raw(11001);
        let k = IndexId::from_raw(11002);
        let l = IndexId::from_raw(11003);

        let spec = ContractionSpec::new(
            vec![
                (TensorId::new(0), vec![i, j]),
                (TensorId::new(1), vec![j, k]),
                (TensorId::new(2), vec![k, l]),
            ],
            vec![i, l],
        )
        .unwrap();

        let mut index_map = IndexMap::new();
        index_map.insert(
            TensorId::new(0),
            vec![
                IndexSpec { dim: d0, is_contracted: false, is_contiguous: true },
                IndexSpec { dim: d1, is_contracted: true, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(1),
            vec![
                IndexSpec { dim: d1, is_contracted: true, is_contiguous: true },
                IndexSpec { dim: d2, is_contracted: true, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(2),
            vec![
                IndexSpec { dim: d2, is_contracted: true, is_contiguous: true },
                IndexSpec { dim: d3, is_contracted: false, is_contiguous: true },
            ],
        );

        let metric = CostMetric::default();

        let greedy = GreedyOptimizer.optimize(&spec, &index_map, &metric, None).unwrap();
        prop_assert!(greedy.estimated_flops >= 0.0);
        prop_assert!(greedy.estimated_memory_bytes == 0 || greedy.estimated_memory_bytes > 0);

        let dp = DPOptimizer::default().optimize(&spec, &index_map, &metric, None).unwrap();
        prop_assert!(dp.estimated_flops >= 0.0);
    }

    /// DP optimizer cost ≤ greedy optimizer cost (DP finds the optimal).
    #[test]
    fn dp_cost_leq_greedy_cost(
        d0 in 2usize..=10,
        d1 in 2usize..=10,
        d2 in 2usize..=10,
        d3 in 2usize..=10,
    ) {
        let i = IndexId::from_raw(12000);
        let j = IndexId::from_raw(12001);
        let k = IndexId::from_raw(12002);
        let l = IndexId::from_raw(12003);

        let spec = ContractionSpec::new(
            vec![
                (TensorId::new(0), vec![i, j]),
                (TensorId::new(1), vec![j, k]),
                (TensorId::new(2), vec![k, l]),
            ],
            vec![i, l],
        )
        .unwrap();

        let mut index_map = IndexMap::new();
        index_map.insert(
            TensorId::new(0),
            vec![
                IndexSpec { dim: d0, is_contracted: false, is_contiguous: true },
                IndexSpec { dim: d1, is_contracted: true, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(1),
            vec![
                IndexSpec { dim: d1, is_contracted: true, is_contiguous: true },
                IndexSpec { dim: d2, is_contracted: true, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(2),
            vec![
                IndexSpec { dim: d2, is_contracted: true, is_contiguous: true },
                IndexSpec { dim: d3, is_contracted: false, is_contiguous: true },
            ],
        );

        let metric = CostMetric::default();
        let greedy = GreedyOptimizer.optimize(&spec, &index_map, &metric, None).unwrap();
        let dp = DPOptimizer::default().optimize(&spec, &index_map, &metric, None).unwrap();

        // DP is exact → its cost should be ≤ greedy's cost.
        prop_assert!(
            dp.estimated_flops <= greedy.estimated_flops + 1e-12,
            "DP cost {} > greedy cost {}",
            dp.estimated_flops,
            greedy.estimated_flops,
        );
    }

    /// validate_dimensions catches dimension mismatches on contracted indices.
    #[test]
    fn validate_dimensions_catches_mismatch(
        d_a in 1usize..=100,
        d_b in 1usize..=100,
    ) {
        let i = IndexId::from_raw(13000);
        let j = IndexId::from_raw(13001);

        let spec = ContractionSpec::new(
            vec![
                (TensorId::new(0), vec![i, j]),
                (TensorId::new(1), vec![j]),
            ],
            vec![i],
        )
        .unwrap();

        let mut index_map = IndexMap::new();
        index_map.insert(
            TensorId::new(0),
            vec![
                IndexSpec { dim: 10, is_contracted: false, is_contiguous: true },
                IndexSpec { dim: d_a, is_contracted: true, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(1),
            vec![
                IndexSpec { dim: d_b, is_contracted: true, is_contiguous: true },
            ],
        );

        let result = spec.validate_dimensions(&index_map);
        if d_a == d_b {
            prop_assert!(result.is_ok());
        } else {
            prop_assert!(result.is_err());
        }
    }
}
