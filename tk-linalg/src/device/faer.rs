//! `DeviceFaer` — Pure-Rust dense backend using the `faer` crate.
//!
//! Default backend when `backend-faer` feature is active (enabled by default).
//! Provides state-of-the-art multithreaded SVD and GEMM with native lazy
//! conjugation support.
//!
//! Implementations for all four scalar types (`f32`, `f64`, `C32`, `C64`) are
//! generated via `macro_rules!`. Real types (f32, f64) use zero-copy faer views
//! for GEMM; complex types (C32, C64) use copy-based conversion to handle faer's
//! split-storage internal representation.

use tk_core::{DenseTensor, MatMut, MatRef, Scalar, TensorShape, C32, C64};
use tk_symmetry::{BitPackable, BlockSparseTensor, PackedSectorKey};

use crate::error::{LinAlgError, LinAlgResult};
use crate::results::{EighResult, QrResult, SvdConvergenceError, SvdResult};
use crate::tasks::{compute_fusion_rule, compute_output_indices, lpt_sort, SectorGemmTask};
use crate::traits::{LinAlgBackend, SparseLinAlgBackend};

/// Pure-Rust dense backend using the `faer` crate.
///
/// Supported scalar types: `f32`, `f64`, `Complex<f32>` (C32), `Complex<f64>` (C64).
pub struct DeviceFaer;

// ---------------------------------------------------------------------------
// Helper trait: extract the real part from a scalar value
// ---------------------------------------------------------------------------

/// faer returns singular values and eigenvalues as the full scalar type
/// (e.g., `Complex<f64>` for Hermitian eigenvalues that are mathematically
/// real). This trait extracts the `Real` component uniformly.
trait RealPart: Scalar {
    fn real_part(self) -> Self::Real;
}

impl RealPart for f32 {
    #[inline(always)]
    fn real_part(self) -> f32 {
        self
    }
}

impl RealPart for f64 {
    #[inline(always)]
    fn real_part(self) -> f64 {
        self
    }
}

impl RealPart for C32 {
    #[inline(always)]
    fn real_part(self) -> f32 {
        self.re
    }
}

impl RealPart for C64 {
    #[inline(always)]
    fn real_part(self) -> f64 {
        self.re
    }
}

// ---------------------------------------------------------------------------
// Zero-copy helpers for real scalar types (f32, f64)
// ---------------------------------------------------------------------------

/// Convert a tk-core `MatRef<T>` to a faer `MatRef<T>` (zero-copy, real types only).
///
/// Complex types use split storage in faer and cannot be zero-copy converted.
fn to_faer_mat_ref_real<'a, T: Scalar + faer::SimpleEntity>(
    mat: &'a MatRef<'a, T>,
) -> faer::MatRef<'a, T> {
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

/// Create a faer `MatMut` from a tk-core `MatMut` (zero-copy, real types only).
macro_rules! faer_mat_mut_real {
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

// ---------------------------------------------------------------------------
// Copy-based conversion for all scalar types (including complex)
// ---------------------------------------------------------------------------

/// Copy a tk-core `MatRef` into a `faer::Mat` (owned).
///
/// Works for all scalar types. Applies `is_conjugated` flag via `mat.get()`.
/// For complex types, this also handles the interleaved-to-split storage conversion.
fn tk_mat_to_faer_owned<T: Scalar + faer::ComplexField>(mat: &MatRef<'_, T>) -> faer::Mat<T> {
    let mut faer_mat = faer::Mat::<T>::zeros(mat.rows, mat.cols);
    for i in 0..mat.rows {
        for j in 0..mat.cols {
            faer_mat.write(i, j, mat.get(i, j));
        }
    }
    faer_mat
}

// ---------------------------------------------------------------------------
// SVD truncation helper (macro to avoid associated-type ambiguity)
// ---------------------------------------------------------------------------

/// Given thin SVD components (singular values in descending order, U and V matrices),
/// truncate to at most `max_rank` values above `cutoff * sigma_max`.
///
/// V† is computed with conjugate-transpose: `Vt[i,j] = conj(V[j,i])`.
/// For real types, `conj()` is a no-op, so this reduces to plain transpose.
macro_rules! truncate_svd {
    ($scalar:ty, $real:ty, $u_mat:expr, $s_sorted:expr, $v_mat:expr, $max_rank:expr, $cutoff:expr) => {{
        let u_ref = $u_mat;
        let s_values: &[$real] = $s_sorted;
        let v_ref = $v_mat;
        let max_rank: usize = $max_rank;
        let cutoff: $real = $cutoff;

        let sigma_max: $real = s_values
            .first()
            .copied()
            .unwrap_or(<$real as num_traits::Zero>::zero());
        let threshold = cutoff * sigma_max;

        let mut rank = 0usize;
        for &s in s_values {
            if rank >= max_rank || s < threshold {
                break;
            }
            rank += 1;
        }

        let truncation_error: $real = s_values[rank..].iter().map(|&s| s * s).sum();
        let singular_values: Vec<$real> = s_values[..rank].to_vec();

        // Extract U[:, :rank]
        let m = u_ref.nrows();
        let mut u_data = vec![<$scalar as num_traits::Zero>::zero(); m * rank];
        for j in 0..rank {
            for i in 0..m {
                u_data[i * rank + j] = u_ref.read(i, j);
            }
        }
        let u = DenseTensor::from_vec(TensorShape::row_major(&[m, rank]), u_data);

        // Extract Vt[:rank, :] = conj(V[:, :rank])^T.
        // Vt[i,j] = conj(V[j,i]). For real types, conj is no-op.
        let n = v_ref.nrows();
        let mut vt_data = vec![<$scalar as num_traits::Zero>::zero(); rank * n];
        for i in 0..rank {
            for j in 0..n {
                vt_data[i * n + j] = Scalar::conj(v_ref.read(j, i));
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
    }};
}

// ---------------------------------------------------------------------------
// Shared LinAlgBackend methods (SVD, eigh, QR)
// ---------------------------------------------------------------------------

/// Common SVD, eigendecomposition, and QR implementations for all scalar types.
/// Invoked inside each `LinAlgBackend<$scalar>` impl block.
macro_rules! impl_common_linalg {
    ($scalar:ty, $real:ty) => {
        fn svd_truncated_gesdd(
            &self,
            mat: &MatRef<'_, $scalar>,
            max_rank: usize,
            cutoff: $real,
        ) -> Result<SvdResult<$scalar>, SvdConvergenceError> {
            let k = mat.rows.min(mat.cols);

            let faer_mat = tk_mat_to_faer_owned(mat);
            let svd = faer_mat.thin_svd();

            // Extract singular values (always real) and sort descending.
            // faer returns them as the full scalar type; RealPart extracts .re.
            let s_col = svd.s_diagonal();
            let s_values: Vec<$real> = (0..k)
                .map(|i| <$scalar as RealPart>::real_part(s_col.read(i)))
                .collect();

            let mut indices: Vec<usize> = (0..k).collect();
            indices.sort_unstable_by(|&i, &j| {
                s_values[j]
                    .partial_cmp(&s_values[i])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let s_sorted: Vec<$real> = indices.iter().map(|&i| s_values[i]).collect();

            // Build permuted U and V.
            let u_full = svd.u();
            let v_full = svd.v();
            let m = u_full.nrows();
            let n = v_full.nrows();

            let mut u_sorted = faer::Mat::<$scalar>::zeros(m, k);
            let mut v_sorted = faer::Mat::<$scalar>::zeros(n, k);
            for (new_col, &old_col) in indices.iter().enumerate() {
                for row in 0..m {
                    u_sorted.write(row, new_col, u_full.read(row, old_col));
                }
                for row in 0..n {
                    v_sorted.write(row, new_col, v_full.read(row, old_col));
                }
            }

            Ok(truncate_svd!(
                $scalar, $real,
                &u_sorted, &s_sorted, &v_sorted,
                max_rank, cutoff
            ))
        }

        fn svd_truncated_gesvd(
            &self,
            mat: &MatRef<'_, $scalar>,
            max_rank: usize,
            cutoff: $real,
        ) -> Result<SvdResult<$scalar>, SvdConvergenceError> {
            // faer does not expose separate gesdd/gesvd algorithms.
            // This method exists to satisfy the trait contract and serves as the fallback.
            self.svd_truncated_gesdd(mat, max_rank, cutoff)
        }

        fn eigh_lowest(
            &self,
            mat: &MatRef<'_, $scalar>,
            k: usize,
        ) -> LinAlgResult<EighResult<$scalar>> {
            debug_assert!(mat.is_square(), "eigh: matrix must be square");
            let n = mat.rows;

            if k > n {
                return Err(LinAlgError::EighKTooLarge { k, n });
            }

            let faer_mat = tk_mat_to_faer_owned(mat);
            let evd = faer_mat.selfadjoint_eigendecomposition(faer::Side::Lower);

            // faer returns eigenvalues in ascending order.
            let s_diag = evd.s();
            let u_mat = evd.u();

            // Eigenvalues are always real for symmetric/Hermitian matrices.
            // faer returns them as the full scalar type; RealPart extracts .re.
            let eig_vals: Vec<$real> = (0..k)
                .map(|i| <$scalar as RealPart>::real_part(s_diag.column_vector().read(i)))
                .collect();

            let mut evec_data = vec![<$scalar as num_traits::Zero>::zero(); n * k];
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

        fn qr(&self, mat: &MatRef<'_, $scalar>) -> LinAlgResult<QrResult<$scalar>> {
            let m = mat.rows;
            let n = mat.cols;
            let k = m.min(n);

            debug_assert!(m > 0 && n > 0, "qr: matrix must have nonzero dimensions");

            let faer_mat = tk_mat_to_faer_owned(mat);
            let qr_decomp = faer_mat.qr();

            let q_faer = qr_decomp.compute_thin_q();
            let r_faer = qr_decomp.compute_thin_r();

            let mut q_data = vec![<$scalar as num_traits::Zero>::zero(); m * k];
            for j in 0..k {
                for i in 0..m {
                    q_data[i * k + j] = q_faer.read(i, j);
                }
            }
            let q = DenseTensor::from_vec(TensorShape::row_major(&[m, k]), q_data);

            let r_rows = r_faer.nrows();
            let r_cols = r_faer.ncols();
            let mut r_data = vec![<$scalar as num_traits::Zero>::zero(); r_rows * r_cols];
            for i in 0..r_rows {
                for j in 0..r_cols {
                    r_data[i * r_cols + j] = r_faer.read(i, j);
                }
            }
            let r = DenseTensor::from_vec(TensorShape::row_major(&[r_rows, r_cols]), r_data);

            Ok(QrResult { q, r })
        }
    };
}

// ---------------------------------------------------------------------------
// Shared SparseLinAlgBackend methods (spmv, block_gemm)
// ---------------------------------------------------------------------------

/// Common sparse backend methods for all scalar types.
/// Invoked inside each `SparseLinAlgBackend<$scalar, Q>` impl block.
macro_rules! impl_common_sparse {
    ($scalar:ty) => {
        fn spmv(
            &self,
            a: &BlockSparseTensor<$scalar, Q>,
            x: &[$scalar],
            y: &mut [$scalar],
        ) {
            debug_assert_eq!(a.rank(), 2, "spmv: A must be rank-2");

            let indices = a.indices();
            let row_index = &indices[0];
            let col_index = &indices[1];

            for val in y.iter_mut() {
                *val = <$scalar as num_traits::Zero>::zero();
            }

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
                    let mut sum = <$scalar as num_traits::Zero>::zero();
                    for j in 0..cols {
                        sum = sum + block_ref.get(i, j) * x[col_start + j];
                    }
                    y[row_start + i] = y[row_start + i] + sum;
                }
            }
        }

        fn block_gemm(
            &self,
            a: &BlockSparseTensor<$scalar, Q>,
            b: &BlockSparseTensor<$scalar, Q>,
        ) -> BlockSparseTensor<$scalar, Q> {
            debug_assert_eq!(a.rank(), 2, "block_gemm: A must be rank-2");
            debug_assert_eq!(b.rank(), 2, "block_gemm: B must be rank-2");

            let target_flux = a.flux().fuse(b.flux());

            // Phase 1: Task generation.
            let a_keys = a.sector_keys();
            let a_blocks = a.sector_blocks();
            let b_keys = b.sector_keys();
            let b_blocks = b.sector_blocks();

            let mut tasks: Vec<SectorGemmTask<$scalar>> = Vec::new();
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

            // Phase 3: Execute GEMMs and accumulate results.
            let mut results: Vec<(PackedSectorKey, DenseTensor<'static, $scalar>)> =
                Vec::with_capacity(tasks.len());

            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;

                // Parallel execution (FragmentedSectors regime): each Rayon
                // task computes one dense GEMM using single-threaded faer.
                let partial: Vec<(PackedSectorKey, DenseTensor<'static, $scalar>)> =
                    tasks
                        .par_iter()
                        .map(|task| {
                            let m = task.block_a.shape().dims()[0];
                            let n = task.block_b.shape().dims()[1];
                            let mut out = DenseTensor::<$scalar>::zeros(
                                TensorShape::row_major(&[m, n]),
                            );

                            {
                                let a_ref = task
                                    .block_a
                                    .as_mat_ref()
                                    .expect("block_a should be rank-2");
                                let b_ref = task
                                    .block_b
                                    .as_mat_ref()
                                    .expect("block_b should be rank-2");
                                let mut out_mat =
                                    out.as_mat_mut().expect("out should be rank-2");

                                // Use copy-based faer conversion for thread safety.
                                let faer_a = tk_mat_to_faer_owned(&a_ref);
                                let faer_b = tk_mat_to_faer_owned(&b_ref);
                                let mut faer_c =
                                    faer::Mat::<$scalar>::zeros(out_mat.rows, out_mat.cols);

                                // Single-threaded BLAS per task; parallelism is
                                // at the Rayon task level.
                                faer::linalg::matmul::matmul(
                                    faer_c.as_mut(),
                                    faer_a.as_ref(),
                                    faer_b.as_ref(),
                                    None,
                                    <$scalar as num_traits::One>::one(),
                                    faer::Parallelism::None,
                                );

                                // Copy result back.
                                for i in 0..out_mat.rows {
                                    for j in 0..out_mat.cols {
                                        out_mat.set(i, j, faer_c.read(i, j));
                                    }
                                }
                            }

                            (task.out_key, out)
                        })
                        .collect();

                // Sequential accumulation by sector key.
                for (key, block) in partial {
                    if let Some(existing) =
                        results.iter_mut().find(|(k, _)| *k == key)
                    {
                        let existing_slice = existing.1.as_mut_slice();
                        let out_slice = block.as_slice();
                        for (e, &o) in
                            existing_slice.iter_mut().zip(out_slice.iter())
                        {
                            *e = *e + o;
                        }
                    } else {
                        results.push((key, block));
                    }
                }
            }

            #[cfg(not(feature = "parallel"))]
            {
                // Sequential execution: use multithreaded BLAS per GEMM call.
                for task in &tasks {
                    let m = task.block_a.shape().dims()[0];
                    let n = task.block_b.shape().dims()[1];

                    let mut out = DenseTensor::<$scalar>::zeros(
                        TensorShape::row_major(&[m, n]),
                    );

                    let a_ref = task.block_a
                        .as_mat_ref()
                        .expect("block_a should be rank-2");
                    let b_ref = task.block_b
                        .as_mat_ref()
                        .expect("block_b should be rank-2");
                    let mut out_ref = out
                        .as_mat_mut()
                        .expect("out should be rank-2");

                    self.gemm(
                        <$scalar as num_traits::One>::one(),
                        &a_ref,
                        &b_ref,
                        <$scalar as num_traits::Zero>::zero(),
                        &mut out_ref,
                    );

                    if let Some(existing) =
                        results.iter_mut().find(|(k, _)| *k == task.out_key)
                    {
                        let existing_slice = existing.1.as_mut_slice();
                        let out_slice = out.as_slice();
                        for (e, &o) in
                            existing_slice.iter_mut().zip(out_slice.iter())
                        {
                            *e = *e + o;
                        }
                    } else {
                        results.push((task.out_key, out));
                    }
                }
            }

            // Phase 4: Structural restoration — re-sort by key.
            results.sort_unstable_by_key(|(key, _)| *key);

            let out_indices = compute_output_indices(a.indices(), b.indices());
            let out_directions = vec![a.leg_directions()[0], b.leg_directions()[b.rank() - 1]];

            let out_rank = 2;
            let blocks: Vec<(Vec<Q>, DenseTensor<'static, $scalar>)> = results
                .into_iter()
                .map(|(key, block)| {
                    let qns: Vec<Q> = key.unpack::<Q>(out_rank).into_vec();
                    (qns, block)
                })
                .collect();

            BlockSparseTensor::from_blocks(out_indices, target_flux, out_directions, blocks)
        }
    };
}

// ===========================================================================
// Real scalar backend (f32, f64): zero-copy GEMM via faer pointer views
// ===========================================================================

/// Generate `LinAlgBackend` and `SparseLinAlgBackend` for a real scalar type.
///
/// Real types support zero-copy faer `MatRef` conversion since they use identity
/// storage (no split real/imaginary). Conjugation is always a no-op but is included
/// for consistency — the compiler elides it.
macro_rules! impl_faer_real_backend {
    ($scalar:ty) => {
        impl LinAlgBackend<$scalar> for DeviceFaer {
            fn gemm(
                &self,
                alpha: $scalar,
                a: &MatRef<'_, $scalar>,
                b: &MatRef<'_, $scalar>,
                beta: $scalar,
                c: &mut MatMut<'_, $scalar>,
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

                let faer_a = to_faer_mat_ref_real(a);
                let faer_b = to_faer_mat_ref_real(b);
                let mut faer_c = faer_mat_mut_real!(c);

                // For real types, Conj<T> = T, so both branches produce the same type.
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

            impl_common_linalg!($scalar, $scalar);
        }

        impl<Q: BitPackable> SparseLinAlgBackend<$scalar, Q> for DeviceFaer {
            impl_common_sparse!($scalar);
        }
    };
}

// ===========================================================================
// Complex scalar backend (C32, C64): copy-based GEMM
// ===========================================================================

/// Generate `LinAlgBackend` and `SparseLinAlgBackend` for a complex scalar type.
///
/// Complex types use faer's split real/imaginary internal storage, so zero-copy
/// pointer conversion is not possible. Inputs and outputs are copied to/from
/// faer-owned `Mat<T>`. The O(mn) copy overhead is negligible compared to the
/// O(mn*k) or O(n^3) computation cost of GEMM/SVD.
///
/// Conjugation is applied during the copy via `MatRef::get()`, which respects
/// the `is_conjugated` flag. This eliminates the need for faer's conjugation-
/// aware view types in the `matmul` call.
macro_rules! impl_faer_complex_backend {
    ($scalar:ty, $real:ty) => {
        impl LinAlgBackend<$scalar> for DeviceFaer {
            fn gemm(
                &self,
                alpha: $scalar,
                a: &MatRef<'_, $scalar>,
                b: &MatRef<'_, $scalar>,
                beta: $scalar,
                c: &mut MatMut<'_, $scalar>,
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

                // Copy inputs to faer-owned Mats. get() applies conjugation.
                let faer_a = tk_mat_to_faer_owned(a);
                let faer_b = tk_mat_to_faer_owned(b);

                // Copy existing C for beta scaling.
                let m = c.rows;
                let n = c.cols;
                let mut faer_c = faer::Mat::<$scalar>::zeros(m, n);
                for i in 0..m {
                    for j in 0..n {
                        faer_c.write(i, j, c.get(i, j));
                    }
                }

                // C = beta * C + alpha * A * B
                // (A and B are already conjugated if needed from the copy step)
                faer::linalg::matmul::matmul(
                    faer_c.as_mut(),
                    faer_a.as_ref(),
                    faer_b.as_ref(),
                    Some(beta),
                    alpha,
                    faer::Parallelism::Rayon(0),
                );

                // Copy result back to tk-core MatMut.
                for i in 0..m {
                    for j in 0..n {
                        c.set(i, j, faer_c.read(i, j));
                    }
                }
            }

            impl_common_linalg!($scalar, $real);
        }

        impl<Q: BitPackable> SparseLinAlgBackend<$scalar, Q> for DeviceFaer {
            impl_common_sparse!($scalar);
        }
    };
}

// ===========================================================================
// Generate implementations for all four scalar types
// ===========================================================================

impl_faer_real_backend!(f32);
impl_faer_real_backend!(f64);
impl_faer_complex_backend!(C32, f32);
impl_faer_complex_backend!(C64, f64);

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- f64 tests ---

    #[test]
    fn gemm_identity_f64() {
        let backend = DeviceFaer;

        let a_data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let a = MatRef::from_slice(&a_data, 2, 2);

        let i_data = vec![1.0_f64, 0.0, 0.0, 1.0];
        let b = MatRef::from_slice(&i_data, 2, 2);

        let mut c_data = vec![0.0_f64; 4];
        let mut c = MatMut::from_slice(&mut c_data, 2, 2);

        backend.gemm(1.0_f64, &a, &b, 0.0_f64, &mut c);

        assert!((c_data[0] - 1.0_f64).abs() < 1e-12);
        assert!((c_data[1] - 2.0_f64).abs() < 1e-12);
        assert!((c_data[2] - 3.0_f64).abs() < 1e-12);
        assert!((c_data[3] - 4.0_f64).abs() < 1e-12);
    }

    #[test]
    fn gemm_alpha_beta() {
        let backend = DeviceFaer;

        let a_data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let a = MatRef::from_slice(&a_data, 2, 2);

        let b_data = vec![1.0_f64, 0.0, 0.0, 1.0];
        let b = MatRef::from_slice(&b_data, 2, 2);

        let mut c_data = vec![10.0_f64, 20.0, 30.0, 40.0];
        let mut c = MatMut::from_slice(&mut c_data, 2, 2);

        // C = 2*A*B + 3*C = 2*A + 3*C
        backend.gemm(2.0_f64, &a, &b, 3.0_f64, &mut c);

        assert!((c_data[0] - 32.0_f64).abs() < 1e-12);
        assert!((c_data[1] - 64.0_f64).abs() < 1e-12);
        assert!((c_data[2] - 96.0_f64).abs() < 1e-12);
        assert!((c_data[3] - 128.0_f64).abs() < 1e-12);
    }

    #[test]
    fn svd_reconstruction_f64() {
        let backend = DeviceFaer;

        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat = MatRef::from_slice(&data, 3, 2);

        let result = backend
            .svd_truncated(&mat, 2, 0.0_f64)
            .expect("SVD should succeed");

        assert_eq!(result.rank, 2);
        assert_eq!(result.singular_values.len(), 2);
        assert!(result.singular_values[0] >= result.singular_values[1]);

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

        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat = MatRef::from_slice(&data, 3, 2);

        let result = backend
            .svd_truncated(&mat, 1, 0.0_f64)
            .expect("SVD should succeed");

        assert_eq!(result.rank, 1);
        assert_eq!(result.singular_values.len(), 1);
        assert!(result.truncation_error > 0.0);
    }

    #[test]
    fn eigh_lowest_symmetric() {
        let backend = DeviceFaer;

        // Symmetric 3x3: [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
        let data = vec![2.0_f64, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let mat = MatRef::from_slice(&data, 3, 3);

        let result = backend.eigh_lowest(&mat, 2).expect("eigh should succeed");

        assert_eq!(result.eigenvalues.len(), 2);
        assert!(result.eigenvalues[0] <= result.eigenvalues[1]);
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

        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat = MatRef::from_slice(&data, 3, 2);

        let result = backend.qr(&mat).expect("QR should succeed");

        assert_eq!(result.q.shape().dims(), &[3, 2]);
        assert_eq!(result.r.shape().dims(), &[2, 2]);

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

        let s_values = vec![5.0_f64, 3.0, 1.0];
        let delta = 1e-10_f64;

        let eye3 = vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
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

        let s_values = vec![0.0_f64];
        let delta = 1e-8_f64;

        let u = DenseTensor::from_vec(TensorShape::row_major(&[1, 1]), vec![1.0_f64]);
        let vt = DenseTensor::from_vec(TensorShape::row_major(&[1, 1]), vec![1.0_f64]);

        let result = backend.regularized_svd_inverse(&s_values, &u, &vt, delta);

        let r = result.as_mat_ref().unwrap();
        assert!(r.get(0, 0).abs() < 1e-12, "Should be zero, not NaN or Inf");
        assert!(!r.get(0, 0).is_nan());
        assert!(!r.get(0, 0).is_infinite());
    }

    // --- f32 tests ---

    #[test]
    fn gemm_identity_f32() {
        let backend = DeviceFaer;

        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let a = MatRef::from_slice(&a_data, 2, 2);

        let i_data = vec![1.0f32, 0.0, 0.0, 1.0];
        let b = MatRef::from_slice(&i_data, 2, 2);

        let mut c_data = vec![0.0f32; 4];
        let mut c = MatMut::from_slice(&mut c_data, 2, 2);

        backend.gemm(1.0, &a, &b, 0.0, &mut c);

        assert!((c_data[0] - 1.0).abs() < 1e-6);
        assert!((c_data[1] - 2.0).abs() < 1e-6);
        assert!((c_data[2] - 3.0).abs() < 1e-6);
        assert!((c_data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn svd_reconstruction_f32() {
        let backend = DeviceFaer;

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat = MatRef::from_slice(&data, 3, 2);

        let result = backend
            .svd_truncated(&mat, 2, 0.0)
            .expect("SVD should succeed");

        assert_eq!(result.rank, 2);

        let m = 3;
        let n = 2;
        let u_ref = result.u.as_mat_ref().unwrap();
        let vt_ref = result.vt.as_mat_ref().unwrap();

        let mut max_err = 0.0_f32;
        for r in 0..m {
            for c in 0..n {
                let original = data[r * n + c];
                let mut approx = 0.0_f32;
                for k in 0..result.rank {
                    approx +=
                        u_ref.get(r, k) * result.singular_values[k] * vt_ref.get(k, c);
                }
                max_err = max_err.max((original - approx).abs());
            }
        }
        assert!(max_err < 1e-4, "reconstruction error: {max_err}");
    }

    #[test]
    fn eigh_lowest_f32() {
        let backend = DeviceFaer;

        let data = vec![2.0f32, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let mat = MatRef::from_slice(&data, 3, 3);

        let result = backend.eigh_lowest(&mat, 2).expect("eigh should succeed");

        assert_eq!(result.eigenvalues.len(), 2);
        assert!(result.eigenvalues[0] <= result.eigenvalues[1]);
        assert!(
            (result.eigenvalues[0] - (2.0 - 2.0_f32.sqrt())).abs() < 1e-5,
            "eigenvalue[0] = {}",
            result.eigenvalues[0],
        );
    }

    // --- C64 tests ---

    #[test]
    fn gemm_identity_c64() {
        let backend = DeviceFaer;

        let a_data = vec![
            C64::new(1.0, 0.5),
            C64::new(2.0, -1.0),
            C64::new(3.0, 0.0),
            C64::new(4.0, 1.5),
        ];
        let a = MatRef::from_slice(&a_data, 2, 2);

        let i_data = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
        ];
        let b = MatRef::from_slice(&i_data, 2, 2);

        let mut c_data = vec![C64::new(0.0, 0.0); 4];
        let mut c = MatMut::from_slice(&mut c_data, 2, 2);

        backend.gemm(C64::new(1.0, 0.0), &a, &b, C64::new(0.0, 0.0), &mut c);

        for i in 0..4 {
            assert!(
                (c_data[i] - a_data[i]).norm() < 1e-12,
                "A*I should equal A at index {i}"
            );
        }
    }

    #[test]
    fn gemm_conjugated_c64() {
        let backend = DeviceFaer;

        // A = [[1+i]], conjugated -> conj(A) = [[1-i]]
        let a_data = vec![C64::new(1.0, 1.0)];
        let a = MatRef::from_slice(&a_data, 1, 1).conjugate();

        let b_data = vec![C64::new(2.0, 0.0)];
        let b = MatRef::from_slice(&b_data, 1, 1);

        let mut c_data = vec![C64::new(0.0, 0.0)];
        let mut c = MatMut::from_slice(&mut c_data, 1, 1);

        // C = 1 * conj(A) * B = (1-i) * 2 = 2-2i
        backend.gemm(C64::new(1.0, 0.0), &a, &b, C64::new(0.0, 0.0), &mut c);

        assert!(
            (c_data[0] - C64::new(2.0, -2.0)).norm() < 1e-12,
            "expected 2-2i, got {:?}",
            c_data[0]
        );
    }

    #[test]
    fn svd_reconstruction_c64() {
        let backend = DeviceFaer;

        let data = vec![
            C64::new(1.0, 0.5),
            C64::new(2.0, -1.0),
            C64::new(3.0, 0.0),
            C64::new(4.0, 1.5),
            C64::new(5.0, -0.5),
            C64::new(6.0, 0.0),
        ];
        let mat = MatRef::from_slice(&data, 3, 2);

        let result = backend
            .svd_truncated(&mat, 2, 0.0)
            .expect("SVD should succeed");

        assert_eq!(result.rank, 2);
        assert!(result.singular_values[0] >= result.singular_values[1]);

        let m = 3;
        let n = 2;
        let u_ref = result.u.as_mat_ref().unwrap();
        let vt_ref = result.vt.as_mat_ref().unwrap();

        let mut max_err = 0.0_f64;
        for r in 0..m {
            for c in 0..n {
                let original = data[r * n + c];
                let mut approx = C64::new(0.0, 0.0);
                for k in 0..result.rank {
                    approx = approx
                        + u_ref.get(r, k)
                            * C64::from_real(result.singular_values[k])
                            * vt_ref.get(k, c);
                }
                max_err = max_err.max((original - approx).norm());
            }
        }
        assert!(max_err < 1e-10, "reconstruction error: {max_err}");
    }

    #[test]
    fn eigh_hermitian_c64() {
        let backend = DeviceFaer;

        // Hermitian 2x2: [[2, 1-i], [1+i, 3]]
        // Eigenvalues: (5 +/- sqrt(9))/2 = 1 and 4
        let data = vec![
            C64::new(2.0, 0.0),
            C64::new(1.0, -1.0),
            C64::new(1.0, 1.0),
            C64::new(3.0, 0.0),
        ];
        let mat = MatRef::from_slice(&data, 2, 2);

        let result = backend.eigh_lowest(&mat, 2).expect("eigh should succeed");

        assert_eq!(result.eigenvalues.len(), 2);
        assert!(result.eigenvalues[0] <= result.eigenvalues[1]);
        assert!(
            (result.eigenvalues[0] - 1.0).abs() < 1e-10,
            "eigenvalue[0] = {}, expected 1.0",
            result.eigenvalues[0]
        );
        assert!(
            (result.eigenvalues[1] - 4.0).abs() < 1e-10,
            "eigenvalue[1] = {}, expected 4.0",
            result.eigenvalues[1]
        );
    }

    #[test]
    fn qr_reconstruction_c64() {
        let backend = DeviceFaer;

        let data = vec![
            C64::new(1.0, 0.5),
            C64::new(2.0, -1.0),
            C64::new(3.0, 0.0),
            C64::new(4.0, 1.5),
            C64::new(5.0, -0.5),
            C64::new(6.0, 0.0),
        ];
        let mat = MatRef::from_slice(&data, 3, 2);

        let result = backend.qr(&mat).expect("QR should succeed");

        assert_eq!(result.q.shape().dims(), &[3, 2]);
        assert_eq!(result.r.shape().dims(), &[2, 2]);

        let q_ref = result.q.as_mat_ref().unwrap();
        let r_ref = result.r.as_mat_ref().unwrap();

        let mut max_err = 0.0_f64;
        for i in 0..3 {
            for j in 0..2 {
                let original = data[i * 2 + j];
                let mut approx = C64::new(0.0, 0.0);
                for k in 0..2 {
                    approx = approx + q_ref.get(i, k) * r_ref.get(k, j);
                }
                max_err = max_err.max((original - approx).norm());
            }
        }
        assert!(max_err < 1e-10, "QR reconstruction error: {max_err}");
    }

    // --- C32 tests ---

    #[test]
    fn gemm_identity_c32() {
        let backend = DeviceFaer;

        let a_data = vec![
            C32::new(1.0, 0.5),
            C32::new(2.0, -1.0),
            C32::new(3.0, 0.0),
            C32::new(4.0, 1.5),
        ];
        let a = MatRef::from_slice(&a_data, 2, 2);

        let i_data = vec![
            C32::new(1.0, 0.0),
            C32::new(0.0, 0.0),
            C32::new(0.0, 0.0),
            C32::new(1.0, 0.0),
        ];
        let b = MatRef::from_slice(&i_data, 2, 2);

        let mut c_data = vec![C32::new(0.0, 0.0); 4];
        let mut c = MatMut::from_slice(&mut c_data, 2, 2);

        backend.gemm(C32::new(1.0, 0.0), &a, &b, C32::new(0.0, 0.0), &mut c);

        for i in 0..4 {
            assert!(
                (c_data[i] - a_data[i]).norm() < 1e-6,
                "A*I should equal A at index {i}"
            );
        }
    }

    #[test]
    fn svd_reconstruction_c32() {
        let backend = DeviceFaer;

        let data = vec![
            C32::new(1.0, 0.5),
            C32::new(2.0, -1.0),
            C32::new(3.0, 0.0),
            C32::new(4.0, 1.5),
            C32::new(5.0, -0.5),
            C32::new(6.0, 0.0),
        ];
        let mat = MatRef::from_slice(&data, 3, 2);

        let result = backend
            .svd_truncated(&mat, 2, 0.0)
            .expect("SVD should succeed");

        assert_eq!(result.rank, 2);

        let m = 3;
        let n = 2;
        let u_ref = result.u.as_mat_ref().unwrap();
        let vt_ref = result.vt.as_mat_ref().unwrap();

        let mut max_err = 0.0_f32;
        for r in 0..m {
            for c in 0..n {
                let original = data[r * n + c];
                let mut approx = C32::new(0.0, 0.0);
                for k in 0..result.rank {
                    approx = approx
                        + u_ref.get(r, k)
                            * C32::from_real(result.singular_values[k])
                            * vt_ref.get(k, c);
                }
                max_err = max_err.max((original - approx).norm());
            }
        }
        assert!(max_err < 1e-4, "reconstruction error: {max_err}");
    }
}
