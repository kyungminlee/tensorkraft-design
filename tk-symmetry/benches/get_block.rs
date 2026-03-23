//! Criterion benchmarks for `get_block` on a 100-sector tensor.
//!
//! Tech spec invariant: `get_block` < 10 ns on a 100-sector tensor.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use tk_symmetry::builtins::U1;
use tk_symmetry::quantum_number::{LegDirection, QuantumNumber};
use tk_symmetry::sector_key::QIndex;
use tk_symmetry::BlockSparseTensor;

/// Build a rank-2 tensor with ~100 sectors.
/// Uses charges -10..=10 (21 values) on each leg with identity flux,
/// giving sectors where q0 + q1 = 0 → 21 sectors per (q, -q) pair.
/// To get closer to 100 sectors, use a wider charge range.
fn make_100_sector_tensor() -> BlockSparseTensor<f64, U1> {
    // charges -50..=49 gives 100 values; with flux=0, incoming/outgoing
    // gives 100 valid sectors (one per charge value).
    let charges: Vec<(U1, usize)> = (-50..=49).map(|n| (U1(n), 1)).collect();
    let idx = QIndex::new(charges);
    let indices = vec![idx.clone(), idx.clone()];
    let dirs = vec![LegDirection::Incoming, LegDirection::Outgoing];
    BlockSparseTensor::<f64, U1>::zeros(indices, U1::identity(), dirs)
}

fn bench_get_block(c: &mut Criterion) {
    let tensor = make_100_sector_tensor();
    let n_sectors = tensor.n_sectors();

    c.bench_function(&format!("get_block ({n_sectors} sectors)"), |b| {
        b.iter(|| {
            // Look up a sector in the middle of the sorted key range
            black_box(tensor.get_block(&[U1(0), U1(0)]))
        })
    });

    c.bench_function(&format!("get_block_miss ({n_sectors} sectors)"), |b| {
        b.iter(|| {
            // Look up a sector that doesn't exist (flux mismatch)
            black_box(tensor.get_block(&[U1(0), U1(1)]))
        })
    });
}

fn bench_iter_keyed_blocks(c: &mut Criterion) {
    let tensor = make_100_sector_tensor();
    let n_sectors = tensor.n_sectors();

    c.bench_function(&format!("iter_keyed_blocks ({n_sectors} sectors)"), |b| {
        b.iter(|| {
            for (key, block) in tensor.iter_keyed_blocks() {
                black_box((key, block));
            }
        })
    });
}

criterion_group!(benches, bench_get_block, bench_iter_keyed_blocks);
criterion_main!(benches);
