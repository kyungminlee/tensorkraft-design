//! Integration test: block-sparse GEMM with realistic U1 quantum numbers.
//!
//! Simulates a typical DMRG-like scenario where two rank-2 block-sparse tensors
//! with U(1) charge conservation are contracted. Verifies correctness by comparing
//! block-sparse GEMM output against dense reference computation.

use tk_core::{DenseTensor, MatMut, MatRef, TensorShape};
use tk_linalg::{DeviceFaer, LinAlgBackend, SparseLinAlgBackend};
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

    // Enumerate valid sectors and build blocks with deterministic data
    let mut rng_state = seed;
    let mut blocks = Vec::new();

    for &(ref rq, rdim) in row_index.sectors() {
        for &(ref cq, cdim) in col_index.sectors() {
            // Check flux rule: rq.fuse(cq.dual()) == flux
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

    // Build offset maps
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

#[test]
fn block_gemm_matches_dense_reference() {
    let backend = DeviceFaer;

    // Realistic U(1) charges for a spin chain with Sz conservation:
    // Sectors correspond to Sz = -1, 0, +1 with varying bond dimensions.
    let row_sectors = vec![(-1, 4), (0, 8), (1, 4)]; // 16 total
    let col_sectors = vec![(-1, 3), (0, 6), (1, 3)]; // 12 total

    let a = build_test_matrix(&row_sectors, &col_sectors, U1(0), 42);
    let b = build_test_matrix(&col_sectors, &row_sectors, U1(0), 137);

    // Block-sparse GEMM
    let result_sparse = backend.block_gemm(&a, &b);

    // Dense reference computation
    let (a_dense, a_rows, a_cols) = to_dense(&a);
    let (b_dense, b_rows, b_cols) = to_dense(&b);
    assert_eq!(a_cols, b_rows, "inner dimensions must match");

    let a_ref = MatRef::from_slice(&a_dense, a_rows, a_cols);
    let b_ref = MatRef::from_slice(&b_dense, b_rows, b_cols);
    let mut c_dense = vec![0.0_f64; a_rows * b_cols];
    {
        let mut c_mat = MatMut::from_slice(&mut c_dense, a_rows, b_cols);
        backend.gemm(1.0, &a_ref, &b_ref, 0.0, &mut c_mat);
    }

    // Convert sparse result to dense for comparison
    let (result_dense, r_rows, r_cols) = to_dense(&result_sparse);
    assert_eq!(r_rows, a_rows);
    assert_eq!(r_cols, b_cols);

    // Compare element-by-element
    let mut max_err = 0.0_f64;
    for i in 0..a_rows {
        for j in 0..b_cols {
            let sparse_val = result_dense[i * b_cols + j];
            let dense_val = c_dense[i * b_cols + j];
            max_err = max_err.max((sparse_val - dense_val).abs());
        }
    }

    assert!(
        max_err < 1e-10,
        "block_gemm vs dense GEMM max error: {max_err}"
    );
}

#[test]
fn block_gemm_nonzero_flux() {
    let backend = DeviceFaer;

    // Test with non-trivial flux (creation operator-like tensor)
    let row_sectors = vec![(0, 3), (1, 5)];
    let col_sectors = vec![(-1, 2), (0, 4)];

    // A has flux +1 (maps charge q to q+1)
    let a = build_test_matrix(&row_sectors, &col_sectors, U1(1), 99);

    // B has flux -1
    let b = build_test_matrix(&col_sectors, &row_sectors, U1(-1), 200);

    let result = backend.block_gemm(&a, &b);

    // Result should have flux 0 (= +1 + (-1))
    assert_eq!(*result.flux(), U1(0), "output flux should be 0");

    // Verify all output sectors satisfy the flux rule
    for (qns, _block) in result.iter_blocks() {
        let fused = qns[0].fuse(&qns[1].dual());
        assert_eq!(fused, *result.flux(),
            "output sector {:?} violates flux rule", qns);
    }
}

#[test]
fn block_gemm_sector_count_bounded() {
    let backend = DeviceFaer;

    // 5-sector tensor: output should have at most 5 sectors
    let sectors: Vec<(i32, usize)> = (0..5).map(|q| (q, 2)).collect();
    let qindex = QIndex::new(sectors.iter().map(|&(q, d)| (U1(q), d)).collect());

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

    // Output sectors ≤ min(sectors_a, sectors_b)
    assert!(
        result.n_sectors() <= 5,
        "output has {} sectors, expected at most 5",
        result.n_sectors()
    );
}
