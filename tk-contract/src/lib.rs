//! `tk-contract` — DAG-based tensor contraction engine for the tensorkraft workspace.
//!
//! This crate sits directly above `tk-linalg` (and transitively `tk-core` and
//! `tk-symmetry`) in the dependency graph and is consumed by `tk-dmrg` and `tk-dmft`.
//!
//! # Core responsibilities
//! - **Index notation parsing and validation** — `ContractionSpec` with `IndexId`/`TensorId`
//! - **Contraction graph construction** — Binary DAG (`ContractionGraph`)
//! - **Path optimization** — Pluggable `PathOptimizer` implementations (greedy, DP, TreeSA)
//! - **Contraction execution** — Reshape-to-matrix + GEMM dispatch via `LinAlgBackend`
//! - **Symmetry-aware contraction** — Block-sparse path via `SparseLinAlgBackend`
//! - **SU(2) structural hook** — Extension point for Clebsch-Gordan coefficient injection
//!
//! # Scope note
//! The contraction engine operates with **bosonic tensor legs only**. Fermionic
//! sign conventions (Jordan-Wigner strings) are the MPO's responsibility in `tk-dmrg`.

pub mod cost;
pub mod error;
pub mod executor;
pub mod graph;
pub mod index;
pub mod optimizer;
pub mod reshape;
pub mod sparse;
pub mod structural;

// Flat re-exports for ergonomic downstream use:
pub use cost::CostMetric;
pub use error::{ContractionError, ContractResult};
pub use executor::{ContractionExecutor, DenseExecutionPlan, ExecutionPlan};
pub use graph::{ContractionGraph, ContractionNode};
pub use index::{ContractionSpec, IndexId, IndexMap, IndexSpec, TensorId};
pub use optimizer::{GreedyOptimizer, PathOptimizer};
pub use sparse::SparseContractionExecutor;
pub use structural::{AbelianHook, StructuralContractionHook};
