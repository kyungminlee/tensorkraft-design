//! `tk-linalg` — Linear algebra backend abstraction layer for the tensorkraft workspace.
//!
//! This crate sits directly above `tk-core` and `tk-symmetry` in the dependency
//! graph and is consumed by every higher-level crate that performs numerical
//! computation: `tk-contract`, `tk-dmrg`, and `tk-dmft`.
//!
//! # Responsibilities
//! - **Backend trait abstraction** — `LinAlgBackend<T>` and `SparseLinAlgBackend<T, Q>`
//! - **Conjugation-aware GEMM** — Propagates `MatRef::is_conjugated` to backends
//! - **SVD with algorithm selection** — `gesdd` with `gesvd` fallback
//! - **Tikhonov-regularized pseudo-inverse** — For TDVP gauge restoration
//! - **LPT-scheduled block-sparse GEMM** — Longest Processing Time heuristic
//! - **Hybrid threading regime** — FatSectors vs FragmentedSectors selection
//! - **Concrete backends** — DeviceFaer (default), with stubs for MKL/OpenBLAS/CUDA

pub mod error;
pub mod results;
pub mod traits;
pub mod threading;
pub mod device;

pub(crate) mod tasks;

// Flat re-exports for ergonomic downstream use:
pub use traits::{LinAlgBackend, SparseLinAlgBackend};
pub use results::{SvdResult, EighResult, QrResult, SvdConvergenceError};
pub use threading::ThreadingRegime;
pub use error::{LinAlgError, LinAlgResult};

// Device re-exports (conditional on feature flags):
#[cfg(feature = "backend-faer")]
pub use device::faer::DeviceFaer;

#[cfg(feature = "backend-faer")]
pub use device::{DeviceAPI, DefaultDevice};

// TODO: Uncomment when backends are integrated.
// #[cfg(feature = "backend-oxiblas")]
// pub use device::oxiblas::DeviceOxiblas;
// #[cfg(feature = "backend-mkl")]
// pub use device::mkl::DeviceMKL;
// #[cfg(feature = "backend-openblas")]
// pub use device::openblas::DeviceOpenBLAS;
// #[cfg(feature = "backend-cuda")]
// pub use device::cuda::DeviceCuda;
