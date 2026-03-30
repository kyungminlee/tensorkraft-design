//! Integration test: block-sparse matrix-vector multiply (spmv) with U1 quantum numbers.
//!
//! Verifies correctness of `SparseLinAlgBackend::spmv()` by comparing block-sparse
//! results against dense reference computations for various flux configurations.

use tk_core::{DenseTensor, TensorShape};
use tk_linalg::{DeviceFaer, SparseLinAlgBackend};
use tk_symmetry::{BlockSparseTensor, LegDirection, QIndex, QuantumNumber, U1};

/// Simple LCG for deterministic data generation.
fn lcg_next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as i32 as f64) / (1 << 30) as f64
}

/// Build a rank-2 block-sparse tensor with U(1) charges and non-trivial data.
fn build_test_matrix(
    row_sectors: &[(i32, usize)],
    col_sectors: &[(i32, usize)],
    flux: U1,
    seed: u64,
) -> BlockSparseTensor<f64, U1> {
    let row_index = QIndex::new(row_sectors.iter().map(|&(q, d)| (U1(q), d)).collect());
    let col_index = QIndex::new(col_sectors.iter().map(|&(q, d)| (U1(q), d)).collect());
    let dirs = vec![LegDirection::Incoming, LegDirection::Outgoing];

    let mut rng_state = seed;
    let mut blocks = Vec::new();

    for &(ref rq, rdim) in row_index.sectors() {
        for &(ref cq, cdim) in col_index.sectors() {
            let fused = rq.fuse(&cq.dual());
            if fused != flux {
                continue;
            }
            let mut data = vec![0.0_f64; rdim * cdim];
            for val in data.iter_mut() {
                *val = lcg_next(&mut rng_state);
            }
            let block = DenseTensor::from_vec(TensorShape::row_major(&[rdim, cdim]), data);
            blocks.push((vec![rq.clone(), cq.clone()], block));
        }
    }

    BlockSparseTensor::from_blocks(vec![row_index, col_index], flux, dirs, blocks)
}

/// Convert a block-sparse tensor to a dense matrix for reference comparison.
fn to_dense(tensor: &BlockSparseTensor<f64, U1>) -> (Vec<f64>, usize, usize) {
    let indices = tensor.indices();
    let row_index = &indices[0];
    let col_index = &indices[1];

    let total_rows = row_index.total_dim();
    let total_cols = col_index.total_dim();
    let mut dense = vec![0.0_f64; total_rows * total_cols];

    let mut row_offsets = std::collections::BTreeMap::new();
    let mut offset = 0;
    for &(ref q, dim) in row_index.sectors() {
        row_offsets.insert(q.clone(), offset);
        offset += dim;
    }

    let mut col_offsets = std::collections::BTreeMap::new();
    offset = 0;
    for &(ref q, dim) in col_index.sectors() {
        col_offsets.insert(q.clone(), offset);
        offset += dim;
    }

    for (qns, block) in tensor.iter_blocks() {
        let row_start = row_offsets[&qns[0]];
        let col_start = col_offsets[&qns[1]];
        let block_ref = block.as_mat_ref().expect("block should be rank-2");
        for i in 0..block_ref.rows {
            for j in 0..block_ref.cols {
                dense[(row_start + i) * total_cols + (col_start + j)] = block_ref.get(i, j);
            }
        }
    }

    (dense, total_rows, total_cols)
}

/// Generate a deterministic vector of length `n` using the LCG.
fn build_test_vector(n: usize, seed: u64) -> Vec<f64> {
    let mut rng_state = seed;
    (0..n).map(|_| lcg_next(&mut rng_state)).collect()
}

/// Dense matrix-vector multiply: y = A * x (row-major A).
fn dense_matvec(a: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), rows * cols);
    assert_eq!(x.len(), cols);
    let mut y = vec![0.0_f64; rows];
    for i in 0..rows {
        let mut sum = 0.0_f64;
        for j in 0..cols {
            sum += a[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
    y
}

#[test]
fn spmv_matches_dense_reference() {
    let backend = DeviceFaer;

    // Sectors: Sz = -1, 0, +1 with dims 2, 3, 2 => total 7
    let row_sectors = vec![(-1, 2), (0, 3), (1, 2)];
    let col_sectors = vec![(-1, 2), (0, 3), (1, 2)];

    let a = build_test_matrix(&row_sectors, &col_sectors, U1(0), 42);

    let total_rows = 7;
    let total_cols = 7;

    let x = build_test_vector(total_cols, 137);
    let mut y_sparse = vec![0.0_f64; total_rows];

    backend.spmv(&a, &x, &mut y_sparse);

    // Dense reference
    let (a_dense, nrows, ncols) = to_dense(&a);
    assert_eq!(nrows, total_rows);
    assert_eq!(ncols, total_cols);
    let y_dense = dense_matvec(&a_dense, nrows, ncols, &x);

    // Compare
    let mut max_err = 0.0_f64;
    for i in 0..total_rows {
        let err = (y_sparse[i] - y_dense[i]).abs();
        max_err = max_err.max(err);
    }

    assert!(
        max_err < 1e-12,
        "spmv vs dense matvec max error: {max_err}"
    );
}

#[test]
fn spmv_zero_flux() {
    let backend = DeviceFaer;

    // Zero flux: diagonal blocks only (q_row == q_col)
    let row_sectors = vec![(-1, 3), (0, 4), (1, 3)];
    let col_sectors = vec![(-1, 3), (0, 4), (1, 3)];

    let a = build_test_matrix(&row_sectors, &col_sectors, U1(0), 77);

    let total_rows: usize = row_sectors.iter().map(|(_, d)| d).sum();
    let total_cols: usize = col_sectors.iter().map(|(_, d)| d).sum();

    let x = build_test_vector(total_cols, 200);
    let mut y_sparse = vec![0.0_f64; total_rows];

    backend.spmv(&a, &x, &mut y_sparse);

    // Dense reference
    let (a_dense, nrows, ncols) = to_dense(&a);
    let y_dense = dense_matvec(&a_dense, nrows, ncols, &x);

    let mut max_err = 0.0_f64;
    for i in 0..total_rows {
        max_err = max_err.max((y_sparse[i] - y_dense[i]).abs());
    }

    assert!(
        max_err < 1e-12,
        "spmv_zero_flux: max error {max_err}"
    );

    // Verify flux is U1(0)
    assert_eq!(*a.flux(), U1(0), "flux should be zero");
}

#[test]
fn spmv_nonzero_flux() {
    let backend = DeviceFaer;

    // Creation operator pattern: flux = +1, maps charge q to q+1
    let row_sectors = vec![(0, 3), (1, 5)];
    let col_sectors = vec![(-1, 2), (0, 4)];

    let a = build_test_matrix(&row_sectors, &col_sectors, U1(1), 99);

    assert_eq!(*a.flux(), U1(1), "flux should be +1");

    let total_rows: usize = row_sectors.iter().map(|(_, d)| d).sum();
    let total_cols: usize = col_sectors.iter().map(|(_, d)| d).sum();

    let x = build_test_vector(total_cols, 300);
    let mut y_sparse = vec![0.0_f64; total_rows];

    backend.spmv(&a, &x, &mut y_sparse);

    // Dense reference
    let (a_dense, nrows, ncols) = to_dense(&a);
    let y_dense = dense_matvec(&a_dense, nrows, ncols, &x);

    let mut max_err = 0.0_f64;
    for i in 0..total_rows {
        max_err = max_err.max((y_sparse[i] - y_dense[i]).abs());
    }

    assert!(
        max_err < 1e-12,
        "spmv_nonzero_flux: max error {max_err}"
    );

    // Verify that output sectors respect flux rule
    for (qns, _block) in a.iter_blocks() {
        let fused = qns[0].fuse(&qns[1].dual());
        assert_eq!(fused, *a.flux(), "sector {:?} violates flux rule", qns);
    }
}

#[test]
fn spmv_empty_tensor() {
    let backend = DeviceFaer;

    // Build a tensor with no valid blocks (empty sectors list)
    let row_index = QIndex::<U1>::new(vec![]);
    let col_index = QIndex::<U1>::new(vec![]);
    let dirs = vec![LegDirection::Incoming, LegDirection::Outgoing];

    let a = BlockSparseTensor::<f64, U1>::from_blocks(
        vec![row_index, col_index],
        U1(0),
        dirs,
        vec![],
    );

    assert_eq!(a.n_sectors(), 0, "tensor should have zero sectors");

    // Both x and y are empty (total_dim == 0)
    let x: Vec<f64> = vec![];
    let mut y: Vec<f64> = vec![];

    backend.spmv(&a, &x, &mut y);

    // y should remain empty (no elements to check, just verify no panic)
    assert!(y.is_empty(), "y should remain empty for zero-dim tensor");
}
