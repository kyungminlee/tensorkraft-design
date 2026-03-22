//! Concrete backend implementations and the composite `DeviceAPI` type.

#[cfg(feature = "backend-faer")]
pub mod faer;

// TODO: Uncomment when backend crates are available.
// #[cfg(feature = "backend-oxiblas")]
// pub mod oxiblas;
// #[cfg(feature = "backend-mkl")]
// pub mod mkl;
// #[cfg(feature = "backend-openblas")]
// pub mod openblas;
// #[cfg(feature = "backend-cuda")]
// pub mod cuda;

use tk_core::{MatMut, MatRef, Scalar};
use tk_symmetry::{BitPackable, BlockSparseTensor};

use crate::error::LinAlgResult;
use crate::results::{EighResult, QrResult, SvdConvergenceError, SvdResult};
use crate::traits::{LinAlgBackend, SparseLinAlgBackend};

/// Composite backend pairing a dense backend `D` with a sparse backend `S`.
///
/// `D` handles: GEMM, SVD, QR, eigh, regularized_svd_inverse.
/// `S` handles: spmv, block_gemm (when sectors are in play).
///
/// The default configuration is `DeviceAPI<DeviceFaer, DeviceFaer>` (until
/// oxiblas is integrated, DeviceFaer serves as both dense and sparse backend).
pub struct DeviceAPI<D, S> {
    pub dense: D,
    pub sparse: S,
}

impl<D, S> DeviceAPI<D, S> {
    /// Construct a composite backend from a dense and sparse component.
    pub fn new(dense: D, sparse: S) -> Self {
        DeviceAPI { dense, sparse }
    }
}

/// `DeviceAPI` delegates `LinAlgBackend<T>` to the dense component.
impl<T, D, S> LinAlgBackend<T> for DeviceAPI<D, S>
where
    T: Scalar,
    D: LinAlgBackend<T>,
    S: Send + Sync,
{
    fn gemm(&self, alpha: T, a: &MatRef<T>, b: &MatRef<T>, beta: T, c: &mut MatMut<T>) {
        self.dense.gemm(alpha, a, b, beta, c)
    }

    fn svd_truncated_gesdd(
        &self,
        mat: &MatRef<T>,
        max_rank: usize,
        cutoff: T::Real,
    ) -> Result<SvdResult<T>, SvdConvergenceError> {
        self.dense.svd_truncated_gesdd(mat, max_rank, cutoff)
    }

    fn svd_truncated_gesvd(
        &self,
        mat: &MatRef<T>,
        max_rank: usize,
        cutoff: T::Real,
    ) -> Result<SvdResult<T>, SvdConvergenceError> {
        self.dense.svd_truncated_gesvd(mat, max_rank, cutoff)
    }

    fn eigh_lowest(&self, mat: &MatRef<T>, k: usize) -> LinAlgResult<EighResult<T>> {
        self.dense.eigh_lowest(mat, k)
    }

    fn qr(&self, mat: &MatRef<T>) -> LinAlgResult<QrResult<T>> {
        self.dense.qr(mat)
    }
}

/// `DeviceAPI` delegates `SparseLinAlgBackend<T, Q>` to the sparse component.
impl<T, Q, D, S> SparseLinAlgBackend<T, Q> for DeviceAPI<D, S>
where
    T: Scalar,
    Q: BitPackable,
    D: LinAlgBackend<T>,
    S: SparseLinAlgBackend<T, Q>,
{
    fn spmv(&self, a: &BlockSparseTensor<T, Q>, x: &[T], y: &mut [T]) {
        self.sparse.spmv(a, x, y)
    }

    fn block_gemm(
        &self,
        a: &BlockSparseTensor<T, Q>,
        b: &BlockSparseTensor<T, Q>,
    ) -> BlockSparseTensor<T, Q> {
        self.sparse.block_gemm(a, b)
    }
}

/// Default concrete backend when `backend-faer` is active.
/// Uses DeviceFaer for both dense and sparse operations.
#[cfg(feature = "backend-faer")]
pub type DefaultDevice = DeviceAPI<faer::DeviceFaer, faer::DeviceFaer>;

/// Threshold in matrix dimension below which GEMM is routed to the CPU backend.
/// Configurable; default 500 based on empirical cuBLAS launch overhead measurements.
#[cfg(feature = "backend-cuda")]
pub const GPU_DISPATCH_THRESHOLD: usize = 500;
