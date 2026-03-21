//! `tk-core` — Core tensor data structures, memory management, and scalar
//! traits for the tensorkraft workspace.
//!
//! This is the leaf crate of the workspace. Every other crate depends on it,
//! so it must remain stable, minimal, and free of mathematical logic.
//!
//! # Responsibilities
//! - **Dimensional metadata** — shape/stride management for zero-copy views
//! - **Memory management** — arena allocators, Copy-on-Write storage, pinned-memory tracking
//! - **Matrix view types** — `MatRef`/`MatMut` with lazy conjugation flags
//! - **Element-type abstraction** — the `Scalar` trait hierarchy
//! - **Shared error types** — `TkError` and `TkResult`

pub mod arena;
pub mod error;
pub mod matview;
pub mod scalar;
pub mod shape;
pub mod storage;
pub mod tensor;

#[cfg(feature = "backend-cuda")]
pub mod pinned;

// Flat re-exports for ergonomic downstream use:
pub use arena::SweepArena;
pub use error::{TkError, TkResult};
pub use matview::{MatMut, MatRef};
pub use scalar::{Scalar, C32, C64};
pub use shape::TensorShape;
pub use storage::TensorStorage;
pub use tensor::{DenseTensor, TempTensor};

#[cfg(feature = "backend-cuda")]
pub use pinned::PinnedMemoryTracker;
