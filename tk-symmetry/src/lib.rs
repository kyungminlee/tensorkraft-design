//! `tk-symmetry` — Quantum number types, block-sparse tensor format, and
//! sector-lookup infrastructure for symmetry-exploiting algorithms.
//!
//! This crate sits directly above `tk-core` in the dependency graph and is
//! consumed by `tk-linalg`, `tk-contract`, and all higher-level crates.
//!
//! # Responsibilities
//! - Quantum number traits (`QuantumNumber`, `BitPackable`) and built-in types
//! - Block-sparse tensors (`BlockSparseTensor<T, Q>`) with sorted-key O(log N) lookup
//! - Dual-layout storage: fragmented mutation layout + contiguous `FlatBlockStorage`
//! - Flux rule validation and sector enumeration
//! - SU(2) non-Abelian support behind the `su2-symmetry` feature flag

pub mod block_sparse;
pub mod builtins;
pub mod error;
pub mod flat_storage;
pub mod flux;
pub mod formats;
pub mod quantum_number;
pub mod sector_key;

#[cfg(feature = "su2-symmetry")]
pub mod su2;

// Flat re-exports:
pub use block_sparse::BlockSparseTensor;
pub use builtins::{U1, U1Wide, U1Z2, Z2};
pub use error::{SymResult, SymmetryError};
pub use flat_storage::FlatBlockStorage;
pub use formats::SparsityFormat;
pub use quantum_number::{BitPackable, LegDirection, QuantumNumber};
pub use sector_key::{PackedSectorKey, PackedSectorKey128, QIndex};

#[cfg(feature = "su2-symmetry")]
pub use su2::{ClebschGordanCache, SU2Irrep, WignerEckartTensor};
