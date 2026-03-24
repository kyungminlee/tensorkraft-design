//! Greedy pairwise optimizer.
//!
//! At each step, picks the pair of remaining tensors whose contraction has
//! the lowest composite cost. Time complexity: O(n³) where n = number of
//! input tensors.

use crate::cost::{CostEstimator, CostMetric};
use crate::error::{ContractionError, ContractResult};
use crate::graph::{ContractionGraph, ContractionNode};
use crate::index::{ContractionSpec, IndexId, IndexMap, TensorId};
use crate::optimizer::PathOptimizer;

/// Greedy pairwise optimizer.
///
/// Default for DMRG use cases where n ≤ 5. Its O(n³) scaling is completely
/// irrelevant at n ≤ 5 but produces a near-optimal result because the DMRG
/// contraction graph is a known binary tree pattern.
pub struct GreedyOptimizer;

impl PathOptimizer for GreedyOptimizer {
    fn optimize(
        &self,
        spec: &ContractionSpec,
        index_map: &IndexMap,
        cost: &CostMetric,
        max_memory_bytes: Option<usize>,
    ) -> ContractResult<ContractionGraph> {
        let n = spec.n_tensors();
        if n == 0 {
            return Err(ContractionError::EmptySpec);
        }

        // Build initial node set: one leaf per input tensor.
        let mut nodes: Vec<Option<NodeEntry>> = spec
            .tensors
            .iter()
            .map(|(tid, legs)| {
                let dims: Vec<usize> = (0..legs.len())
                    .map(|pos| index_map.dim(*tid, pos).unwrap_or(1))
                    .collect();
                Some(NodeEntry {
                    tensor_id: *tid,
                    node: ContractionNode::Input {
                        tensor_id: *tid,
                        indices: legs.clone(),
                    },
                    indices: legs.clone(),
                    dims,
                })
            })
            .collect();

        let mut total_flops = 0.0;
        let total_memory_bytes: usize = 0;
        let mut max_intermediate_size: usize = 0;
        let mut next_intermediate = n as u32;

        // Greedy loop: contract n-1 pairs.
        for _ in 0..n - 1 {
            let best = find_best_pair(&nodes, cost)?;

            // Check memory constraint.
            if let Some(limit) = max_memory_bytes {
                let bytes = best.intermediate_size * 8; // assume f64
                if bytes > limit {
                    return Err(ContractionError::OptimizerFailed {
                        optimizer: "greedy".to_string(),
                        reason: format!(
                            "intermediate tensor size {} bytes exceeds limit {} bytes",
                            bytes, limit
                        ),
                    });
                }
            }

            let left_entry = nodes[best.left_idx].take().unwrap();
            let right_entry = nodes[best.right_idx].take().unwrap();

            let result_id = TensorId::new(next_intermediate);
            next_intermediate += 1;

            let new_node = ContractionNode::Contraction {
                left: Box::new(left_entry.node),
                right: Box::new(right_entry.node),
                contracted_indices: best.contracted,
                result_indices: best.result_indices.clone(),
            };

            total_flops += best.cost;
            max_intermediate_size = max_intermediate_size.max(best.intermediate_size);

            // Place the new node in the first available slot, carrying
            // the computed dimensions for use in subsequent steps.
            nodes[best.left_idx] = Some(NodeEntry {
                tensor_id: result_id,
                node: new_node,
                indices: best.result_indices,
                dims: best.result_dims,
            });
        }

        // Extract the final remaining node.
        let final_entry = nodes.into_iter().flatten().next().unwrap();

        Ok(ContractionGraph {
            inputs: spec.tensors.iter().map(|(tid, _)| *tid).collect(),
            root: final_entry.node,
            estimated_flops: total_flops,
            estimated_memory_bytes: total_memory_bytes,
            max_intermediate_size,
        })
    }

    fn name(&self) -> &str {
        "greedy"
    }
}

/// Working data for a node during greedy optimization.
///
/// Each entry tracks its own dimension map so that intermediate results
/// (whose TensorIds are not in the original IndexMap) can still report
/// correct per-leg dimensions to subsequent optimization steps.
struct NodeEntry {
    tensor_id: TensorId,
    node: ContractionNode,
    indices: Vec<IndexId>,
    /// Per-leg dimensions for this node's output tensor.
    /// For input nodes, populated from the original `IndexMap`.
    /// For intermediate nodes, computed from the contracting pair's dims.
    dims: Vec<usize>,
}

/// Result of finding the best pair to contract.
struct BestPair {
    left_idx: usize,
    right_idx: usize,
    cost: f64,
    contracted: Vec<IndexId>,
    result_indices: Vec<IndexId>,
    result_dims: Vec<usize>,
    intermediate_size: usize,
}

/// Find the pair of active nodes with the lowest contraction cost.
///
/// Uses per-node dimension caches rather than the original `IndexMap`,
/// so intermediate tensor dimensions are correctly tracked.
fn find_best_pair(
    nodes: &[Option<NodeEntry>],
    cost: &CostMetric,
) -> ContractResult<BestPair> {
    let mut best: Option<BestPair> = None;

    let active: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| n.as_ref().map(|_| i))
        .collect();

    for (ai, &i) in active.iter().enumerate() {
        for &j in &active[ai + 1..] {
            let left = nodes[i].as_ref().unwrap();
            let right = nodes[j].as_ref().unwrap();

            // Find shared indices between left and right.
            let mut contracted = Vec::new();
            for idx_l in &left.indices {
                if right.indices.contains(idx_l) {
                    contracted.push(*idx_l);
                }
            }

            // Compute M, K, N dimensions from node-local dims (not IndexMap).
            let mut m: usize = 1;
            let mut k: usize = 1;
            let mut n: usize = 1;

            let mut result_indices = Vec::new();
            let mut result_dims = Vec::new();

            for (pos, idx) in left.indices.iter().enumerate() {
                let dim = left.dims[pos];
                if contracted.contains(idx) {
                    k *= dim;
                } else {
                    m *= dim;
                    result_indices.push(*idx);
                    result_dims.push(dim);
                }
            }

            for (pos, idx) in right.indices.iter().enumerate() {
                let dim = right.dims[pos];
                if contracted.contains(idx) {
                    // Already in k
                } else {
                    n *= dim;
                    result_indices.push(*idx);
                    result_dims.push(dim);
                }
            }

            let intermediate_size = m * n;
            let pair_cost = CostEstimator::step_cost::<f64>(m, k, n, true, true, cost);

            let is_better = best.as_ref().map_or(true, |b| pair_cost < b.cost);

            if is_better {
                best = Some(BestPair {
                    left_idx: i,
                    right_idx: j,
                    cost: pair_cost,
                    contracted,
                    result_indices,
                    result_dims,
                    intermediate_size,
                });
            }
        }
    }

    best.ok_or(ContractionError::EmptySpec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexSpec;

    #[test]
    fn greedy_two_tensor_matmul() {
        let i = IndexId::from_raw(0);
        let j = IndexId::from_raw(1);
        let k = IndexId::from_raw(2);

        let spec = ContractionSpec::new(
            vec![
                (TensorId::new(0), vec![i, j]),
                (TensorId::new(1), vec![j, k]),
            ],
            vec![i, k],
        )
        .unwrap();

        let mut index_map = IndexMap::new();
        index_map.insert(
            TensorId::new(0),
            vec![
                IndexSpec { dim: 4, is_contiguous: true },
                IndexSpec { dim: 5, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(1),
            vec![
                IndexSpec { dim: 5, is_contiguous: true },
                IndexSpec { dim: 6, is_contiguous: true },
            ],
        );

        let graph = GreedyOptimizer
            .optimize(&spec, &index_map, &CostMetric::default(), None)
            .unwrap();

        assert_eq!(graph.n_pairwise_steps(), 1);
        assert_eq!(graph.inputs.len(), 2);
        // M=4, K=5, N=6 → flops = 2*4*5*6 = 240
        assert!((graph.estimated_flops - 240.0).abs() < 1e-12);
    }

    #[test]
    fn greedy_three_tensor_chain() {
        let i = IndexId::from_raw(0);
        let j = IndexId::from_raw(1);
        let k = IndexId::from_raw(2);
        let l = IndexId::from_raw(3);

        // A(i,j) * B(j,k) * C(k,l) → D(i,l)
        let spec = ContractionSpec::new(
            vec![
                (TensorId::new(0), vec![i, j]),
                (TensorId::new(1), vec![j, k]),
                (TensorId::new(2), vec![k, l]),
            ],
            vec![i, l],
        )
        .unwrap();

        let mut index_map = IndexMap::new();
        index_map.insert(
            TensorId::new(0),
            vec![
                IndexSpec { dim: 2, is_contiguous: true },
                IndexSpec { dim: 10, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(1),
            vec![
                IndexSpec { dim: 10, is_contiguous: true },
                IndexSpec { dim: 10, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(2),
            vec![
                IndexSpec { dim: 10, is_contiguous: true },
                IndexSpec { dim: 2, is_contiguous: true },
            ],
        );

        let graph = GreedyOptimizer
            .optimize(&spec, &index_map, &CostMetric::default(), None)
            .unwrap();

        assert_eq!(graph.n_pairwise_steps(), 2);
        // Greedy should pick the cheapest pair first.
        // A*B cost: 2*10*10*2MKN → depends on which pair.
        // A*B: M=2,K=10,N=10 → 400 flops
        // B*C: M=10,K=10,N=2 → 400 flops
        // A*C: no shared indices → outer product (not useful)
        // So greedy picks one of A*B or B*C first.
        assert!(graph.estimated_flops > 0.0);
    }

    #[test]
    fn greedy_memory_constraint_rejects() {
        let i = IndexId::from_raw(0);
        let j = IndexId::from_raw(1);
        let k = IndexId::from_raw(2);

        let spec = ContractionSpec::new(
            vec![
                (TensorId::new(0), vec![i, j]),
                (TensorId::new(1), vec![j, k]),
            ],
            vec![i, k],
        )
        .unwrap();

        let mut index_map = IndexMap::new();
        index_map.insert(
            TensorId::new(0),
            vec![
                IndexSpec { dim: 1000, is_contiguous: true },
                IndexSpec { dim: 1000, is_contiguous: true },
            ],
        );
        index_map.insert(
            TensorId::new(1),
            vec![
                IndexSpec { dim: 1000, is_contiguous: true },
                IndexSpec { dim: 1000, is_contiguous: true },
            ],
        );

        // Intermediate is 1000*1000 = 1M elements * 8 bytes = 8MB.
        // Set limit to 1MB → should fail.
        let result = GreedyOptimizer.optimize(
            &spec,
            &index_map,
            &CostMetric::default(),
            Some(1_000_000),
        );
        assert!(matches!(result, Err(ContractionError::OptimizerFailed { .. })));
    }
}
