//! Criterion benchmarks for tk-contract performance validation.
//!
//! Benchmarks:
//! - `greedy_optimizer_n5`: Greedy optimizer for 5-tensor DMRG-like contraction
//! - `dp_optimizer_n5`: DP optimizer for 5-tensor contraction (should match greedy)
//! - `executor_matmul_100x100`: Dense executor for 100×100 matrix multiply
//! - `executor_chain_3_tensors`: Dense executor for 3-tensor chain contraction
//! - `execution_plan_rebuild_check`: ExecutionPlan::needs_rebuild latency

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hashbrown::HashMap;
use tk_contract::{
    ContractionExecutor, ContractionSpec, CostMetric, DPOptimizer, ExecutionPlan,
    GreedyOptimizer, IndexId, IndexMap, IndexSpec, PathOptimizer, TensorId,
};
use tk_core::{DenseTensor, MatMut, MatRef, TensorShape};
use tk_linalg::{DeviceFaer, LinAlgBackend};

fn bench_greedy_optimizer_n5(c: &mut Criterion) {
    // 5-tensor chain: A(i,j) * B(j,k) * C(k,l) * D(l,m) * E(m,n)
    let indices: Vec<IndexId> = (0..6).map(|i| IndexId::from_raw(5000 + i)).collect();
    let spec = ContractionSpec::new(
        vec![
            (TensorId::new(0), vec![indices[0], indices[1]]),
            (TensorId::new(1), vec![indices[1], indices[2]]),
            (TensorId::new(2), vec![indices[2], indices[3]]),
            (TensorId::new(3), vec![indices[3], indices[4]]),
            (TensorId::new(4), vec![indices[4], indices[5]]),
        ],
        vec![indices[0], indices[5]],
    )
    .unwrap();

    let mut index_map = IndexMap::new();
    for t in 0..5 {
        let legs = &spec.tensors[t].1;
        let specs: Vec<IndexSpec> = (0..legs.len())
            .map(|_| IndexSpec {
                dim: 20,
                is_contracted: false,
                is_contiguous: true,
            })
            .collect();
        index_map.insert(TensorId::new(t as u32), specs);
    }

    let metric = CostMetric::default();

    c.bench_function("greedy_optimizer_n5", |bench| {
        bench.iter(|| {
            let graph = GreedyOptimizer
                .optimize(&spec, &index_map, &metric, None)
                .unwrap();
            black_box(graph.estimated_flops);
        })
    });
}

fn bench_dp_optimizer_n5(c: &mut Criterion) {
    let indices: Vec<IndexId> = (0..6).map(|i| IndexId::from_raw(5100 + i)).collect();
    let spec = ContractionSpec::new(
        vec![
            (TensorId::new(0), vec![indices[0], indices[1]]),
            (TensorId::new(1), vec![indices[1], indices[2]]),
            (TensorId::new(2), vec![indices[2], indices[3]]),
            (TensorId::new(3), vec![indices[3], indices[4]]),
            (TensorId::new(4), vec![indices[4], indices[5]]),
        ],
        vec![indices[0], indices[5]],
    )
    .unwrap();

    let mut index_map = IndexMap::new();
    for t in 0..5 {
        let legs = &spec.tensors[t].1;
        let specs: Vec<IndexSpec> = (0..legs.len())
            .map(|_| IndexSpec {
                dim: 20,
                is_contracted: false,
                is_contiguous: true,
            })
            .collect();
        index_map.insert(TensorId::new(t as u32), specs);
    }

    let metric = CostMetric::default();

    c.bench_function("dp_optimizer_n5", |bench| {
        bench.iter(|| {
            let graph = DPOptimizer::default()
                .optimize(&spec, &index_map, &metric, None)
                .unwrap();
            black_box(graph.estimated_flops);
        })
    });
}

fn bench_executor_matmul_100(c: &mut Criterion) {
    let backend = DeviceFaer;
    let n = 100;

    let i = IndexId::from_raw(5200);
    let j = IndexId::from_raw(5201);
    let k = IndexId::from_raw(5202);

    let spec = ContractionSpec::new(
        vec![
            (TensorId::new(0), vec![i, j]),
            (TensorId::new(1), vec![j, k]),
        ],
        vec![i, k],
    )
    .unwrap();

    let mut index_map = IndexMap::new();
    index_map.insert(
        TensorId::new(0),
        vec![
            IndexSpec { dim: n, is_contracted: false, is_contiguous: true },
            IndexSpec { dim: n, is_contracted: true, is_contiguous: true },
        ],
    );
    index_map.insert(
        TensorId::new(1),
        vec![
            IndexSpec { dim: n, is_contracted: true, is_contiguous: true },
            IndexSpec { dim: n, is_contracted: false, is_contiguous: true },
        ],
    );

    let graph = GreedyOptimizer
        .optimize(&spec, &index_map, &CostMetric::default(), None)
        .unwrap();

    let a_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.001).collect();
    let b_data: Vec<f64> = (0..n * n).map(|i| ((i * 7) as f64) * 0.001).collect();
    let a = DenseTensor::from_vec(TensorShape::row_major(&[n, n]), a_data);
    let b = DenseTensor::from_vec(TensorShape::row_major(&[n, n]), b_data);

    let mut inputs = HashMap::new();
    inputs.insert(TensorId::new(0), &a);
    inputs.insert(TensorId::new(1), &b);

    let executor = ContractionExecutor::new(backend);

    c.bench_function("executor_matmul_100x100", |bench| {
        bench.iter(|| {
            let result = executor.execute(&graph, &inputs).unwrap();
            black_box(result.numel());
        })
    });
}

fn bench_execution_plan_rebuild_check(c: &mut Criterion) {
    let i = IndexId::from_raw(5300);
    let j = IndexId::from_raw(5301);
    let k = IndexId::from_raw(5302);

    let spec = ContractionSpec::new(
        vec![
            (TensorId::new(0), vec![i, j]),
            (TensorId::new(1), vec![j, k]),
        ],
        vec![i, k],
    )
    .unwrap();

    let mut index_map = IndexMap::new();
    index_map.insert(
        TensorId::new(0),
        vec![
            IndexSpec { dim: 100, is_contracted: false, is_contiguous: true },
            IndexSpec { dim: 50, is_contracted: true, is_contiguous: true },
        ],
    );
    index_map.insert(
        TensorId::new(1),
        vec![
            IndexSpec { dim: 50, is_contracted: true, is_contiguous: true },
            IndexSpec { dim: 100, is_contracted: false, is_contiguous: true },
        ],
    );

    let plan = ExecutionPlan::<f64>::build(
        &spec,
        &index_map,
        &GreedyOptimizer,
        &CostMetric::default(),
        None,
    )
    .unwrap();

    c.bench_function("execution_plan_rebuild_check", |bench| {
        bench.iter(|| {
            let needs = plan.needs_rebuild(&index_map);
            black_box(needs);
        })
    });
}

criterion_group!(
    benches,
    bench_greedy_optimizer_n5,
    bench_dp_optimizer_n5,
    bench_executor_matmul_100,
    bench_execution_plan_rebuild_check,
);
criterion_main!(benches);
