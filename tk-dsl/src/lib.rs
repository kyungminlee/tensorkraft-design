//! `tk-dsl` — Ergonomic DSL for constructing Hamiltonians and operator sums.
//!
//! This crate provides:
//! - **Typed operator enums** (`SpinOp`, `FermionOp`, `BosonOp`) with `CustomOp` escape hatch
//! - **`OpSum` builder** for accumulating weighted operator products
//! - **Lattice abstractions** (`Chain`, `Square`, `Triangular`, `BetheLattice`, `StarGeometry`)
//! - **Intelligent indices** with unique IDs, prime levels, and automatic contraction
//! - **`IndexedTensor<T>`** with ITensor-style named-index contraction
//!
//! `tk-dsl` produces only `OpSum` structures. MPO compilation (SVD, FSA
//! minimization) lives in `tk-dmrg`.
//!
//! **No dependency on `tk-linalg` or `tk-dmrg`** — prevents cyclic dependencies.

pub mod error;
pub mod index;
pub mod indexed_tensor;
pub mod lattice;
pub mod operators;
pub mod opsum;
pub mod opterm;

// Re-exports for ergonomic use
pub use error::{DslError, DslResult};
pub use index::{Index, IndexDirection, IndexRegistry};
pub use indexed_tensor::{IndexedTensor, contract};
pub use operators::{BosonOp, CustomOp, FermionOp, SiteOperator, SpinOp};
pub use opsum::{HermitianConjugate, OpSum, OpSumPair, OpSumTerm, hc};
pub use opterm::{OpProduct, OpTerm, ScaledOpProduct, op};
pub use lattice::{BetheLattice, Chain, Lattice, Square, StarGeometry, Triangular, snake_path};

// Re-export IndexId from tk-contract for convenience
pub use tk_contract::IndexId;
