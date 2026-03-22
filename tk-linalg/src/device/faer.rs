//! `DeviceFaer` — Pure-Rust dense backend using the `faer` crate.
//!
//! Default backend when `backend-faer` feature is active (enabled by default).
//! Provides state-of-the-art multithreaded SVD and GEMM with native lazy
//! conjugation support.

use tk_core::{DenseTensor, MatMut, MatRef, TensorShape};
use tk_symmetry::{BitPackable, BlockSparseTensor, PackedSectorKey};

use crate::error::{LinAlgError, LinAlgResult};
use crate::results::{EighResult, QrResult, SvdConvergenceError, SvdResult};
use crate::tasks::{compute_fusion_rule, compute_output_indices, lpt_sort, SectorGemmTask};
use crate::traits::{LinAlgBackend, SparseLinAlgBackend};

/// Pure-Rust dense backend using the `faer` crate.
///
/// Supported scalar types: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
pub struct DeviceFaer;

// ---------------------------------------------------------------------------
// Helper: convert tk-core MatRef to faer MatRef
// ---------------------------------------------------------------------------

/// Convert a tk-core `MatRef<f64>` to a faer `MatRef<f64>`.
///
/// # Safety
/// The data pointer and strides from the tk-core MatRef are assumed valid.
fn to_faer_mat_ref<'a>(mat: &'a MatRef<'a, f64>) -> faer::MatRef<'a, f64> {
    unsafe {
        faer::mat::from_raw_parts(
            mat.data.as_ptr(),
            mat.rows,
            mat.cols,
            mat.row_stride,
            mat.col_stride,
        )
    }
}

/// Convert a tk-core `MatMut<f64>` to a faer `MatMut<f64>`.
/// Create a faer MatMut from raw parts of a tk-core MatMut.
/// This is a macro to avoid lifetime issues with the borrow checker.
macro_rules! faer_mat_mut {
    ($mat:expr) => {
        unsafe {
            faer::mat::from_raw_parts_mut(
                $mat.data.as_mut_ptr(),
                $mat.rows,
                $mat.cols,
                $mat.row_stride,
                $mat.col_stride,
            )
        }
    };
}

/// Copy a tk-core MatRef into a faer::Mat (owned).
fn tk_mat_to_faer_owned(mat: &MatRef<'_, f64>) -> faer::Mat<f64> {
    let mut faer_mat = faer::Mat::<f64>::zeros(mat.rows, mat.cols);
    for i in 0..mat.rows {
        for j in 0..mat.cols {
            faer_mat.write(i, j, mat.get(i, j));
        }
    }
    faer_mat
}

// ---------------------------------------------------------------------------
// Truncation helper
// ---------------------------------------------------------------------------

/// Given thin SVD components (singular values in descending order, U and V matrices),
/// truncate to at most `max_rank` values above `cutoff * sigma_max`.
fn truncate_svd_f64(
    u_ref: faer::MatRef<'_, f64>,
    s_values: &[f64],
    v_ref: faer::MatRef<'_, f64>,
    max_rank: usize,
    cutoff: f64,
) -> SvdResult<f64> {
    let sigma_max = s_values.first().copied().unwrap_or(0.0);
    let threshold = cutoff * sigma_max;

    // Determine retained rank.
    let mut rank = 0;
    for &s in s_values {
        if rank >= max_rank || s < threshold {
            break;
        }
        rank += 1;
    }

    // Compute truncation error: sum of squares of discarded singular values.
    let truncation_error: f64 = s_values[rank..].iter().map(|&s| s * s).sum();

    let singular_values: Vec<f64> = s_values[..rank].to_vec();

    // Extract U[:, :rank]
    let m = u_ref.nrows();
    let mut u_data = vec![0.0_f64; m * rank];
    for j in 0..rank {
        for i in 0..m {
            u_data[i * rank + j] = u_ref.read(i, j);
        }
    }
    let u = DenseTensor::from_vec(TensorShape::row_major(&[m, rank]), u_data);

    // Extract Vt[:rank, :] — faer returns V, not V†. We need Vt = V†.
    // For real f64, V† = V^T.
    let n = v_ref.nrows(); // V is (n, k), so V^T is (k, n).
    let mut vt_data = vec![0.0_f64; rank * n];
    for i in 0..rank {
        for j in 0..n {
            // Vt[i, j] = V[j, i]
            vt_data[i * n + j] = v_ref.read(j, i);
        }
    }
    let vt = DenseTensor::from_vec(TensorShape::row_major(&[rank, n]), vt_data);

    SvdResult {
        u,
        singular_values,
        vt,
        rank,
        truncation_error,
    }
}

// ---------------------------------------------------------------------------
// LinAlgBackend<f64> implementation for DeviceFaer
// ---------------------------------------------------------------------------

impl LinAlgBackend<f64> for DeviceFaer {
    fn gemm(
        &self,
        alpha: f64,
        a: &MatRef<'_, f64>,
        b: &MatRef<'_, f64>,
        beta: f64,
        c: &mut MatMut<'_, f64>,
    ) {
        debug_assert_eq!(
            a.cols, b.rows,
            "gemm: a.cols ({}) != b.rows ({})",
            a.cols, b.rows
        );
        debug_assert_eq!(
            c.rows, a.rows,
            "gemm: c.rows ({}) != a.rows ({})",
            c.rows, a.rows
        );
        debug_assert_eq!(
            c.cols, b.cols,
            "gemm: c.cols ({}) != b.cols ({})",
            c.cols, b.cols
        );

        let faer_a = to_faer_mat_ref(a);
        let faer_b = to_faer_mat_ref(b);
        let mut faer_c = faer_mat_mut!(c);

        // faer's lazy conjugation: .conjugate() flips one bit in the view struct.
        // For real f64, conjugation is always a no-op, but we include it for
        // consistency — the compiler will elide it for real types.
        let a_op = if a.is_conjugated {
            faer_a.conjugate()
        } else {
            faer_a.as_ref()
        };
        let b_op = if b.is_conjugated {
            faer_b.conjugate()
        } else {
            faer_b.as_ref()
        };

        faer::linalg::matmul::matmul(
            faer_c.as_mut(),
            a_op,
            b_op,
            Some(beta),
            alpha,
            faer::Parallelism::Rayon(0),
        );
    }

    fn svd_truncated_gesdd(
        &self,
        mat: &MatRef<'_, f64>,
        max_rank: usize,
        cutoff: f64,
    ) -> Result<SvdResult<f64>, SvdConvergenceError> {
        let k = mat.rows.min(mat.cols);

        // Use faer's high-level thin_svd API which handles workspace internally.
        let faer_mat = tk_mat_to_faer_owned(mat);
        let svd = faer_mat.thin_svd();

        // Extract singular values from the ColRef and sort descending.
        let s_col = svd.s_diagonal();
        let s_values: Vec<f64> = (0..k).map(|i| s_col.read(i)).collect();

        // faer may return them in arbitrary order; sort descending.
        // Build index permutation.
        let mut indices: Vec<usize> = (0..k).collect();
        indices.sort_unstable_by(|&i, &j| {
            s_values[j]
                .partial_cmp(&s_values[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let s_sorted: Vec<f64> = indices.iter().map(|&i| s_values[i]).collect();

        // Build permuted U and V (faer returns V, not V†).
        let u_full = svd.u();
        let v_full = svd.v();
        let m = u_full.nrows();
        let n = v_full.nrows();

        let mut u_sorted = faer::Mat::<f64>::zeros(m, k);
        let mut v_sorted = faer::Mat::<f64>::zeros(n, k);
        for (new_col, &old_col) in indices.iter().enumerate() {
            for row in 0..m {
                u_sorted.write(row, new_col, u_full.read(row, old_col));
            }
            for row in 0..n {
                v_sorted.write(row, new_col, v_full.read(row, old_col));
            }
        }

        Ok(truncate_svd_f64(
            u_sorted.as_ref(),
            &s_sorted,
            v_sorted.as_ref(),
            max_rank,
            cutoff,
        ))
    }

    fn svd_truncated_gesvd(
        &self,
        mat: &MatRef<'_, f64>,
        max_rank: usize,
        cutoff: f64,
    ) -> Result<SvdResult<f64>, SvdConvergenceError> {
        // For the faer backend, gesdd and gesvd use the same underlying routine.
        // faer does not expose separate LAPACK-style algorithm selection.
        // This method exists to satisfy the trait contract and serves as the fallback.
        self.svd_truncated_gesdd(mat, max_rank, cutoff)
    }

    fn eigh_lowest(
        &self,
        mat: &MatRef<'_, f64>,
        k: usize,
    ) -> LinAlgResult<EighResult<f64>> {
        debug_assert!(mat.is_square(), "eigh: matrix must be square");
        let n = mat.rows;

        if k > n {
            return Err(LinAlgError::EighKTooLarge { k, n });
        }

        // Use faer's high-level selfadjoint_eigendecomposition API.
        let faer_mat = tk_mat_to_faer_owned(mat);
        let evd = faer_mat.selfadjoint_eigendecomposition(faer::Side::Lower);

        // faer returns eigenvalues in ascending order via .s() diagonal.
        let s_diag = evd.s();
        let u_mat = evd.u();

        let eig_vals: Vec<f64> = (0..k).map(|i| s_diag.column_vector().read(i)).collect();

        // Extract first k eigenvector columns.
        let mut evec_data = vec![0.0_f64; n * k];
        for j in 0..k {
            for i in 0..n {
                evec_data[i * k + j] = u_mat.read(i, j);
            }
        }
        let eigvecs = DenseTensor::from_vec(TensorShape::row_major(&[n, k]), evec_data);

        Ok(EighResult {
            eigenvalues: eig_vals,
            eigenvectors: eigvecs,
        })
    }

    fn qr(&self, mat: &MatRef<'_, f64>) -> LinAlgResult<QrResult<f64>> {
        let m = mat.rows;
        let n = mat.cols;
        let k = m.min(n);

        debug_assert!(m > 0 && n > 0, "qr: matrix must have nonzero dimensions");

        // Use faer's high-level QR API.
        let faer_mat = tk_mat_to_faer_owned(mat);
        let qr_decomp = faer_mat.qr();

        // Extract thin Q (m×k) and thin R (k×n).
        let q_faer = qr_decomp.compute_thin_q();
        let r_faer = qr_decomp.compute_thin_r();

        // Copy Q into tk-core DenseTensor.
        let mut q_data = vec![0.0_f64; m * k];
        for j in 0..k {
            for i in 0..m {
                q_data[i * k + j] = q_faer.read(i, j);
            }
        }
        let q = DenseTensor::from_vec(TensorShape::row_major(&[m, k]), q_data);

        // Copy R into tk-core DenseTensor.
        let r_rows = r_faer.nrows();
        let r_cols = r_faer.ncols();
        let mut r_data = vec![0.0_f64; r_rows * r_cols];
        for i in 0..r_rows {
            for j in 0..r_cols {
                r_data[i * r_cols + j] = r_faer.read(i, j);
            }
        }
        let r = DenseTensor::from_vec(TensorShape::row_major(&[r_rows, r_cols]), r_data);

        Ok(QrResult { q, r })
    }
}

// ---------------------------------------------------------------------------
// SparseLinAlgBackend stub for DeviceFaer
// ---------------------------------------------------------------------------
//
// DeviceFaer provides a naive (non-optimized) SparseLinAlgBackend implementation
// for use before the oxiblas backend is integrated. Each sector GEMM is dispatched
// sequentially using the dense GEMM routine.

impl<Q: BitPackable> SparseLinAlgBackend<f64, Q> for DeviceFaer {
    fn spmv(
        &self,
        a: &BlockSparseTensor<f64, Q>,
        x: &[f64],
        y: &mut [f64],
    ) {
        // Naive implementation: iterate over sectors and apply dense matvec.
        debug_assert_eq!(a.rank(), 2, "spmv: A must be rank-2");

        let indices = a.indices();
        let row_index = &indices[0];
        let col_index = &indices[1];

        // Zero the output.
        for val in y.iter_mut() {
            *val = 0.0;
        }

        // Build offset maps for row and column quantum numbers.
        let mut row_offset_map = std::collections::BTreeMap::new();
        let mut col_offset_map = std::collections::BTreeMap::new();
        {
            let mut offset = 0;
            for &(ref q, dim) in row_index.sectors() {
                row_offset_map.insert(q.clone(), offset);
                offset += dim;
            }
        }
        {
            let mut offset = 0;
            for &(ref q, dim) in col_index.sectors() {
                col_offset_map.insert(q.clone(), offset);
                offset += dim;
            }
        }

        for (qns, block) in a.iter_blocks() {
            let row_q = &qns[0];
            let col_q = &qns[1];

            let row_start = row_offset_map[row_q];
            let col_start = col_offset_map[col_q];

            let block_ref = block.as_mat_ref().expect("block should be rank-2");
            let rows = block_ref.rows;
            let cols = block_ref.cols;

            for i in 0..rows {
                let mut sum = 0.0;
                for j in 0..cols {
                    sum += block_ref.get(i, j) * x[col_start + j];
                }
                y[row_start + i] += sum;
            }
        }
    }

    fn block_gemm(
        &self,
        a: &BlockSparseTensor<f64, Q>,
        b: &BlockSparseTensor<f64, Q>,
    ) -> BlockSparseTensor<f64, Q> {
        debug_assert_eq!(a.rank(), 2, "block_gemm: A must be rank-2");
        debug_assert_eq!(b.rank(), 2, "block_gemm: B must be rank-2");

        let target_flux = a.flux().fuse(b.flux());

        // Phase 1: Task generation.
        let a_keys = a.sector_keys();
        let a_blocks = a.sector_blocks();
        let b_keys = b.sector_keys();
        let b_blocks = b.sector_blocks();

        let mut tasks: Vec<SectorGemmTask<f64>> = Vec::new();
        for (i, key_a) in a_keys.iter().enumerate() {
            for (j, key_b) in b_keys.iter().enumerate() {
                if let Some(out_key) = compute_fusion_rule(
                    *key_a,
                    *key_b,
                    a.rank(),
                    b.rank(),
                    &target_flux,
                    a.indices(),
                    b.indices(),
                ) {
                    let ba = &a_blocks[i];
                    let bb = &b_blocks[j];
                    let flops =
                        ba.shape().dims()[0] * bb.shape().dims()[1] * ba.shape().dims()[1];
                    tasks.push(SectorGemmTask {
                        out_key,
                        block_a: ba,
                        block_b: bb,
                        flops,
                    });
                }
            }
        }

        // Phase 2: LPT scheduling — sort by descending FLOP cost.
        lpt_sort(&mut tasks);

        // Execute GEMMs (sequential for now; Rayon parallelism deferred).
        let mut results: Vec<(PackedSectorKey, DenseTensor<'static, f64>)> =
            Vec::with_capacity(tasks.len());

        for task in &tasks {
            let m = task.block_a.shape().dims()[0];
            let n = task.block_b.shape().dims()[1];

            let mut out = DenseTensor::<f64>::zeros(TensorShape::row_major(&[m, n]));

            let a_ref = task.block_a.as_mat_ref().expect("block_a should be rank-2");
            let b_ref = task.block_b.as_mat_ref().expect("block_b should be rank-2");
            let mut out_ref = out.as_mat_mut().expect("out should be rank-2");

            self.gemm(1.0, &a_ref, &b_ref, 0.0, &mut out_ref);

            // Check if we already have an entry for this output key (accumulate).
            if let Some(existing) = results.iter_mut().find(|(k, _)| *k == task.out_key) {
                let existing_slice = existing.1.as_mut_slice();
                let out_slice = out.as_slice();
                for (e, &o) in existing_slice.iter_mut().zip(out_slice.iter()) {
                    *e = *e + o;
                }
            } else {
                results.push((task.out_key, out));
            }
        }

        // Phase 3: Structural restoration — re-sort by key.
        results.sort_unstable_by_key(|(key, _)| *key);

        let out_indices = compute_output_indices(a.indices(), b.indices());
        let out_directions = vec![a.leg_directions()[0], b.leg_directions()[b.rank() - 1]];

        let out_rank = 2;
        let blocks: Vec<(Vec<Q>, DenseTensor<'static, f64>)> = results
            .into_iter()
            .map(|(key, block)| {
                let qns: Vec<Q> = key.unpack::<Q>(out_rank).into_vec();
                (qns, block)
            })
            .collect();

        BlockSparseTensor::from_blocks(out_indices, target_flux, out_directions, blocks)
    }
}

// ---------------------------------------------------------------------------
// Macro for generating impls for f32, C32, C64 (stub for now)
// ---------------------------------------------------------------------------

// TODO: Use macro_rules! to generate LinAlgBackend<f32>, LinAlgBackend<C32>,
// LinAlgBackend<C64> implementations from a template. The f64 implementation
// above serves as the canonical reference. Complex types require adapting
// the faer type conversions (faer::c64, faer::c32).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_identity_f64() {
        let backend = DeviceFaer;

        // A = [[1, 2], [3, 4]]
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let a = MatRef::from_slice(&a_data, 2, 2);

        // I = identity
        let i_data = vec![1.0, 0.0, 0.0, 1.0];
        let b = MatRef::from_slice(&i_data, 2, 2);

        let mut c_data = vec![0.0; 4];
        let mut c = MatMut::from_slice(&mut c_data, 2, 2);

        backend.gemm(1.0, &a, &b, 0.0, &mut c);

        assert!((c_data[0] - 1.0).abs() < 1e-12);
        assert!((c_data[1] - 2.0).abs() < 1e-12);
        assert!((c_data[2] - 3.0).abs() < 1e-12);
        assert!((c_data[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn gemm_alpha_beta() {
        let backend = DeviceFaer;

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let a = MatRef::from_slice(&a_data, 2, 2);

        let b_data = vec![1.0, 0.0, 0.0, 1.0];
        let b = MatRef::from_slice(&b_data, 2, 2);

        // C starts as [[10, 20], [30, 40]]
        let mut c_data = vec![10.0, 20.0, 30.0, 40.0];
        let mut c = MatMut::from_slice(&mut c_data, 2, 2);

        // C = 2·A·B + 3·C = 2·A + 3·C
        backend.gemm(2.0, &a, &b, 3.0, &mut c);

        assert!((c_data[0] - 32.0).abs() < 1e-12);
        assert!((c_data[1] - 64.0).abs() < 1e-12);
        assert!((c_data[2] - 96.0).abs() < 1e-12);
        assert!((c_data[3] - 128.0).abs() < 1e-12);
    }

    #[test]
    fn svd_reconstruction_f64() {
        let backend = DeviceFaer;

        // 3×2 matrix
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat = MatRef::from_slice(&data, 3, 2);

        let result = backend
            .svd_truncated(&mat, 2, 0.0)
            .expect("SVD should succeed");

        assert_eq!(result.rank, 2);
        assert_eq!(result.singular_values.len(), 2);
        assert!(result.singular_values[0] >= result.singular_values[1]);

        // Verify reconstruction: A ≈ U · diag(σ) · Vt
        let m = 3;
        let n = 2;
        let u_ref = result.u.as_mat_ref().unwrap();
        let vt_ref = result.vt.as_mat_ref().unwrap();

        let mut max_err = 0.0_f64;
        for r in 0..m {
            for c in 0..n {
                let original = data[r * n + c];
                let mut approx = 0.0;
                for k in 0..result.rank {
                    approx +=
                        u_ref.get(r, k) * result.singular_values[k] * vt_ref.get(k, c);
                }
                max_err = max_err.max((original - approx).abs());
            }
        }
        assert!(max_err < 1e-10, "reconstruction error: {max_err}");
    }

    #[test]
    fn svd_truncation_max_rank() {
        let backend = DeviceFaer;

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat = MatRef::from_slice(&data, 3, 2);

        let result = backend
            .svd_truncated(&mat, 1, 0.0)
            .expect("SVD should succeed");

        assert_eq!(result.rank, 1);
        assert_eq!(result.singular_values.len(), 1);
        assert!(result.truncation_error > 0.0);
    }

    #[test]
    fn eigh_lowest_symmetric() {
        let backend = DeviceFaer;

        // Symmetric 3×3: [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
        // Eigenvalues: 2 - 2cos(kπ/4) for k=1,2,3
        let data = vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let mat = MatRef::from_slice(&data, 3, 3);

        let result = backend.eigh_lowest(&mat, 2).expect("eigh should succeed");

        assert_eq!(result.eigenvalues.len(), 2);
        assert!(result.eigenvalues[0] <= result.eigenvalues[1]);
        // Smallest eigenvalue: 2 - sqrt(2) ≈ 0.5858
        assert!(
            (result.eigenvalues[0] - (2.0 - 2.0_f64.sqrt())).abs() < 1e-10,
            "eigenvalue[0] = {}, expected {}",
            result.eigenvalues[0],
            2.0 - 2.0_f64.sqrt()
        );
    }

    #[test]
    fn qr_reconstruction() {
        let backend = DeviceFaer;

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat = MatRef::from_slice(&data, 3, 2);

        let result = backend.qr(&mat).expect("QR should succeed");

        assert_eq!(result.q.shape().dims(), &[3, 2]);
        assert_eq!(result.r.shape().dims(), &[2, 2]);

        // Verify Q·R ≈ A
        let q_ref = result.q.as_mat_ref().unwrap();
        let r_ref = result.r.as_mat_ref().unwrap();

        let mut max_err = 0.0_f64;
        for i in 0..3 {
            for j in 0..2 {
                let original = data[i * 2 + j];
                let mut approx = 0.0;
                for k in 0..2 {
                    approx += q_ref.get(i, k) * r_ref.get(k, j);
                }
                max_err = max_err.max((original - approx).abs());
            }
        }
        assert!(max_err < 1e-10, "QR reconstruction error: {max_err}");
    }

    #[test]
    fn regularized_inverse_large_s() {
        let backend = DeviceFaer;

        let s_values = vec![5.0, 3.0, 1.0];
        let delta = 1e-10;

        let eye3 = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let u = DenseTensor::from_vec(TensorShape::row_major(&[3, 3]), eye3.clone());
        let vt = DenseTensor::from_vec(TensorShape::row_major(&[3, 3]), eye3);

        let result = backend.regularized_svd_inverse(&s_values, &u, &vt, delta);

        let r = result.as_mat_ref().unwrap();
        assert!((r.get(0, 0) - 0.2).abs() < 1e-8);
        assert!((r.get(1, 1) - 1.0 / 3.0).abs() < 1e-8);
        assert!((r.get(2, 2) - 1.0).abs() < 1e-8);
    }

    #[test]
    fn regularized_inverse_zero_s() {
        let backend = DeviceFaer;

        let s_values = vec![0.0];
        let delta = 1e-8;

        let u = DenseTensor::from_vec(TensorShape::row_major(&[1, 1]), vec![1.0]);
        let vt = DenseTensor::from_vec(TensorShape::row_major(&[1, 1]), vec![1.0]);

        let result = backend.regularized_svd_inverse(&s_values, &u, &vt, delta);

        let r = result.as_mat_ref().unwrap();
        assert!(r.get(0, 0).abs() < 1e-12, "Should be zero, not NaN or Inf");
        assert!(!r.get(0, 0).is_nan());
        assert!(!r.get(0, 0).is_infinite());
    }
}
