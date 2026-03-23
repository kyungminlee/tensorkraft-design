//! Core trait definitions: `LinAlgBackend<T>` and `SparseLinAlgBackend<T, Q>`.

use tk_core::{DenseTensor, MatMut, MatRef, Scalar, TensorShape};
use tk_symmetry::{BitPackable, BlockSparseTensor};

use crate::error::{LinAlgError, LinAlgResult};
use crate::results::{EighResult, QrResult, SvdConvergenceError, SvdResult};

/// Object-safe linear algebra backend, parameterized over a single scalar type.
///
/// Implementations provide: GEMM (conjugation-aware), truncated SVD (gesdd/gesvd),
/// lowest eigenvalue/eigenvector (for reference only; DMRG uses in-house Krylov),
/// QR decomposition, and Tikhonov-regularized pseudo-inverse for TDVP gauge shifts.
///
/// # Object Safety
///
/// This trait is object-safe: `Box<dyn LinAlgBackend<f64>>` compiles. The scalar
/// type T is a trait-level parameter, not a per-method generic. This differs from
/// the pre-v5.0 design which had per-method generics and violated E0038.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync`. BLAS handles (MKL, OpenBLAS) must be
/// initialized once and wrapped in types that satisfy this contract.
pub trait LinAlgBackend<T: Scalar>: Send + Sync {
    // -----------------------------------------------------------------------
    // SVD
    // -----------------------------------------------------------------------

    /// Compute truncated SVD of `mat`, retaining at most `max_rank` singular values
    /// with singular value ≥ `cutoff` (relative to the largest singular value).
    ///
    /// **Algorithm selection:** defaults to divide-and-conquer (`gesdd`) for speed.
    /// Falls back to QR-iteration (`gesvd`) on convergence failure.
    ///
    /// **Residual validation (debug only):** after `gesdd` returns, asserts
    /// `‖A − U·Σ·V†‖_F / ‖A‖_F < 1e-10`. Compiled out in `--release`.
    fn svd_truncated(
        &self,
        mat: &MatRef<T>,
        max_rank: usize,
        cutoff: T::Real,
    ) -> LinAlgResult<SvdResult<T>> {
        match self.svd_truncated_gesdd(mat, max_rank, cutoff) {
            Ok(result) => {
                #[cfg(debug_assertions)]
                {
                    // Only check residual when the SVD is full-rank (no truncation).
                    // When truncated, reconstruction error is expected and is captured
                    // by `truncation_error`.
                    let full_rank = mat.rows.min(mat.cols);
                    if result.rank == full_rank {
                        let residual = svd_reconstruction_error(mat, &result);
                        let norm = frobenius_norm(mat);
                        let eps = <T::Real as num_traits::Float>::epsilon();
                        if norm > eps {
                            let rel = residual / norm;
                            // Use 1e-10 for f64 (64-bit) and eps^(3/4) for f32 (32-bit).
                            // eps^(3/4) ≈ 4e-6 for f32, ≈ 5e-12 for f64.
                            let eps_threshold =
                                num_traits::Float::sqrt(eps) * num_traits::Float::sqrt(num_traits::Float::sqrt(eps));
                            let threshold = <T::Real as num_traits::NumCast>::from(1e-10_f64)
                                .unwrap_or(eps_threshold);
                            let threshold = if threshold < eps_threshold {
                                eps_threshold
                            } else {
                                threshold
                            };
                            debug_assert!(
                                rel < threshold,
                                "gesdd SVD reconstruction residual exceeds tolerance",
                            );
                        }
                    }
                }
                Ok(result)
            }
            Err(_convergence_err) => {
                log::warn!(
                    target: "tensorkraft::linalg",
                    "gesdd failed to converge on {}×{} matrix; falling back to gesvd",
                    mat.rows, mat.cols,
                );
                self.svd_truncated_gesvd(mat, max_rank, cutoff)
                    .map_err(|_| LinAlgError::SvdConvergence)
            }
        }
    }

    /// Truncated SVD using divide-and-conquer (`gesdd`).
    fn svd_truncated_gesdd(
        &self,
        mat: &MatRef<T>,
        max_rank: usize,
        cutoff: T::Real,
    ) -> Result<SvdResult<T>, SvdConvergenceError>;

    /// Truncated SVD using QR iteration (`gesvd`).
    fn svd_truncated_gesvd(
        &self,
        mat: &MatRef<T>,
        max_rank: usize,
        cutoff: T::Real,
    ) -> Result<SvdResult<T>, SvdConvergenceError>;

    // -----------------------------------------------------------------------
    // GEMM
    // -----------------------------------------------------------------------

    /// Conjugation-aware GEMM: C = α·op(A)·op(B) + β·C.
    ///
    /// The `is_conjugated` flag on each `MatRef` determines the operation applied
    /// to each input before multiplication. For real `T`, conjugation is a no-op.
    fn gemm(
        &self,
        alpha: T,
        a: &MatRef<T>,
        b: &MatRef<T>,
        beta: T,
        c: &mut MatMut<T>,
    );

    // -----------------------------------------------------------------------
    // Eigenvalue decomposition
    // -----------------------------------------------------------------------

    /// Compute the lowest `k` eigenvalues and corresponding eigenvectors of a
    /// symmetric/Hermitian matrix.
    ///
    /// This is a **full dense EVD** suitable for small matrices (e.g., small
    /// auxiliary matrices in MPO compression). For the iterative ground-state
    /// eigenvalue problem in DMRG sweeps, `LanczosSolver` / `DavidsonSolver`
    /// in `tk-dmrg` must be used instead.
    fn eigh_lowest(
        &self,
        mat: &MatRef<T>,
        k: usize,
    ) -> LinAlgResult<EighResult<T>>;

    // -----------------------------------------------------------------------
    // QR decomposition
    // -----------------------------------------------------------------------

    /// Thin QR decomposition: mat = Q · R.
    ///
    /// Returns `(Q, R)` where Q has orthonormal columns (dimensions m×k where
    /// k = min(m,n)) and R is upper-triangular (dimensions k×n).
    fn qr(&self, mat: &MatRef<T>) -> LinAlgResult<QrResult<T>>;

    // -----------------------------------------------------------------------
    // Tikhonov-regularized pseudo-inverse (TDVP gauge restoration)
    // -----------------------------------------------------------------------

    /// Compute the Tikhonov-regularized pseudo-inverse of a matrix given its
    /// pre-computed SVD factors.
    ///
    /// For each singular value `s_i`, computes `s_i / (s_i² + δ²)` instead
    /// of the naive `1 / s_i`. This prevents NaN when singular values approach
    /// machine zero during TDVP backward bond evolution.
    ///
    /// The reconstruction is: V · diag(s_i / (s_i² + δ²)) · U†
    fn regularized_svd_inverse(
        &self,
        s_values: &[T::Real],
        u: &DenseTensor<'static, T>,
        vt: &DenseTensor<'static, T>,
        delta: T::Real,
    ) -> DenseTensor<'static, T>
    where
        Self: Sized,
    {
        use num_traits::Zero;
        debug_assert!(
            delta > T::Real::zero(),
            "regularized_svd_inverse: delta must be positive",
        );
        let delta_sq = delta * delta;
        let inv_s: Vec<T::Real> = s_values
            .iter()
            .map(|&s| s / (s * s + delta_sq))
            .collect();
        construct_regularized_inverse(self, u, &inv_s, vt)
    }
}

/// Object-safe sparse backend, parameterized over both scalar and quantum number.
///
/// Extends `LinAlgBackend<T>` with operations that exploit block-sparse structure:
/// sparse matrix-vector multiply (`spmv`) and block-sparse GEMM with LPT scheduling.
pub trait SparseLinAlgBackend<T: Scalar, Q: BitPackable>: LinAlgBackend<T> {
    /// Block-sparse matrix-vector multiply: y = A · x.
    fn spmv(
        &self,
        a: &BlockSparseTensor<T, Q>,
        x: &[T],
        y: &mut [T],
    );

    /// Block-sparse GEMM with LPT scheduling and automatic threading regime selection.
    fn block_gemm(
        &self,
        a: &BlockSparseTensor<T, Q>,
        b: &BlockSparseTensor<T, Q>,
    ) -> BlockSparseTensor<T, Q>;
}

// ---------------------------------------------------------------------------
// Internal helper functions
// ---------------------------------------------------------------------------

/// Compute the Frobenius norm of a matrix: sqrt(Σ |a_ij|²).
pub(crate) fn frobenius_norm<T: Scalar>(mat: &MatRef<T>) -> T::Real {
    use num_traits::{Float, Zero};
    let mut sum = <T::Real as Zero>::zero();
    for r in 0..mat.rows {
        for c in 0..mat.cols {
            let val = mat.get(r, c).abs_sq();
            sum = sum + val;
        }
    }
    sum.sqrt()
}

/// Compute the SVD reconstruction residual: ‖A − U·diag(σ)·V†‖_F.
/// Called inside the debug_assert in svd_truncated. Compiled out in release.
#[cfg(debug_assertions)]
pub(crate) fn svd_reconstruction_error<T: Scalar>(
    mat: &MatRef<T>,
    result: &SvdResult<T>,
) -> T::Real {
    use num_traits::{Float, Zero};

    let m = mat.rows;
    let n = mat.cols;
    let rank = result.rank;

    let u_ref = result.u.as_mat_ref().expect("U should be rank-2");
    let vt_ref = result.vt.as_mat_ref().expect("Vt should be rank-2");

    // Reconstruct A_approx = U · diag(σ) · Vt element by element.
    // This is O(m·n·rank) but only runs in debug builds.
    let mut sum_sq = <T::Real as Zero>::zero();
    for r in 0..m {
        for c in 0..n {
            let original = mat.get(r, c);
            let mut approx = T::zero();
            for k in 0..rank {
                let sigma = T::from_real(result.singular_values[k]);
                approx = approx + u_ref.get(r, k) * sigma * vt_ref.get(k, c);
            }
            let diff = original - approx;
            let abs_sq_val = diff.abs_sq();
            sum_sq = sum_sq + abs_sq_val;
        }
    }
    sum_sq.sqrt()
}

/// Reconstruct the regularized pseudo-inverse: V · diag(inv_s) · U†.
///
/// Given U (m×k), inv_s (length k), and Vt (k×n), computes:
///   result = Vt† · diag(inv_s) · U† = V · diag(inv_s) · U†
/// which has shape (n, m).
pub(crate) fn construct_regularized_inverse<T: Scalar>(
    backend: &dyn LinAlgBackend<T>,
    u: &DenseTensor<'static, T>,
    inv_s: &[T::Real],
    vt: &DenseTensor<'static, T>,
) -> DenseTensor<'static, T> {
    let u_ref = u.as_mat_ref().expect("U should be rank-2");
    let vt_ref = vt.as_mat_ref().expect("Vt should be rank-2");

    let m = u_ref.rows;
    let k = inv_s.len();
    let n = vt_ref.cols;

    // Step 1: S_inv_Ut = diag(inv_s) · U†  (k × m)
    // We compute this as element-wise scaling of U† rows.
    let mut s_inv_ut_data = vec![T::zero(); k * m];
    for i in 0..k {
        let s = T::from_real(inv_s[i]);
        for j in 0..m {
            // U†[i, j] = conj(U[j, i])
            s_inv_ut_data[i * m + j] = s * u_ref.get(j, i).conj();
        }
    }
    let s_inv_ut = DenseTensor::from_vec(TensorShape::row_major(&[k, m]), s_inv_ut_data);

    // Step 2: result = V · (S_inv_Ut) = Vt† · (S_inv_Ut)  (n × m)
    // Vt† = V has shape (n, k), S_inv_Ut has shape (k, m), result is (n, m).
    let mut result = DenseTensor::<T>::zeros(TensorShape::row_major(&[n, m]));
    {
        let vt_adj = vt_ref.adjoint(); // (n × k)
        let s_inv_ut_ref = s_inv_ut.as_mat_ref().expect("S_inv_Ut should be rank-2");
        let mut result_mut = result.as_mat_mut().expect("result should be rank-2");
        backend.gemm(
            T::one(),
            &vt_adj,
            &s_inv_ut_ref,
            T::zero(),
            &mut result_mut,
        );
    }
    result
}

/// Set the BLAS internal thread count for MKL/OpenBLAS FFI backends.
/// No-op for DeviceFaer (Rayon controls its own thread pool).
pub(crate) fn set_blas_num_threads(_n: usize) {
    // TODO: implement for MKL/OpenBLAS when those backends are enabled.
    // #[cfg(feature = "backend-mkl")]
    // unsafe { intel_mkl_sys::mkl_set_num_threads(_n as i32); }
    // #[cfg(feature = "backend-openblas")]
    // unsafe { openblas_src::openblas_set_num_threads(_n as i32); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tk_core::C64;

    #[test]
    fn frobenius_norm_identity_2x2() {
        // Identity 2x2: Frobenius norm = sqrt(2)
        let data = vec![1.0_f64, 0.0, 0.0, 1.0];
        let m = MatRef::from_slice(&data, 2, 2);
        let norm = frobenius_norm(&m);
        assert!((norm - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn frobenius_norm_complex() {
        // Single element: [[3+4i]] → norm = sqrt(25) = 5
        let data = vec![C64::new(3.0, 4.0)];
        let m = MatRef::from_slice(&data, 1, 1);
        let norm: f64 = frobenius_norm(&m);
        assert!((norm - 5.0).abs() < 1e-12);
    }
}
