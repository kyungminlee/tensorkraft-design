//! Path optimizer trait and implementations.

pub mod dp;
pub mod greedy;
pub mod treesa;

pub use dp::DPOptimizer;
pub use greedy::GreedyOptimizer;
pub use treesa::TreeSAOptimizer;

use crate::cost::CostMetric;
use crate::error::ContractResult;
use crate::graph::ContractionGraph;
use crate::index::{ContractionSpec, IndexMap};

/// Trait for contraction path optimizers.
///
/// An optimizer takes the contraction specification and the `IndexMap`
/// (containing dimension and stride information) and returns an optimized
/// `ContractionGraph`. It does NOT access tensor data.
///
/// All implementations must be `Send + Sync` so they can be shared across
/// threads (e.g., pre-computed once and used inside Rayon closures).
pub trait PathOptimizer: Send + Sync {
    /// Compute an optimized contraction order.
    ///
    /// # Parameters
    /// - `spec`: The full contraction specification.
    /// - `index_map`: Dimension and stride metadata for cost estimation.
    /// - `cost`: Weights for the composite FLOP + bandwidth metric.
    /// - `max_memory_bytes`: Optional hard constraint on peak intermediate memory.
    fn optimize(
        &self,
        spec: &ContractionSpec,
        index_map: &IndexMap,
        cost: &CostMetric,
        max_memory_bytes: Option<usize>,
    ) -> ContractResult<ContractionGraph>;

    /// Name of this optimizer (for diagnostic logging and benchmark labeling).
    fn name(&self) -> &str;
}
