//! Dynamic programming optimizer (Held-Karp variant for tensor networks).
//!
//! Finds the globally optimal contraction order for n input tensors by
//! evaluating all 2^n subsets. Exact and reproducible.
//!
//! Time complexity: O(3^n) — exponential, practical only for n ≤ 15.
//! Memory: O(2^n) memoization table.

use hashbrown::HashMap;

use crate::cost::{CostEstimator, CostMetric};
use crate::error::{ContractionError, ContractResult};
use crate::graph::{ContractionGraph, ContractionNode};
use crate::index::{ContractionSpec, IndexId, IndexMap, TensorId};
use crate::optimizer::PathOptimizer;

/// Dynamic programming optimizer (Held-Karp variant for tensor networks).
///
/// Finds the globally optimal contraction order for n input tensors by
/// evaluating all 2^n subsets. Exact and reproducible.
///
/// Time complexity: O(3^n) — exponential, practical only for n ≤ 15.
/// Memory: O(2^n) memoization table.
///
/// `max_width` limits the maximum intermediate tensor dimension.
/// If a candidate path would produce an intermediate exceeding this limit,
/// the path is pruned.
pub struct DPOptimizer {
    /// Maximum allowed number of elements in any intermediate tensor.
    /// Default: 2^30 (about 1 billion elements, ~8 GB for f64).
    pub max_width: usize,
}

impl Default for DPOptimizer {
    fn default() -> Self {
        DPOptimizer { max_width: 1 << 30 }
    }
}

/// Entry in the DP memoization table.
#[derive(Clone)]
struct DPEntry {
    /// Best contraction node for this subset.
    node: ContractionNode,
    /// Output indices of this subset's result tensor.
    indices: Vec<IndexId>,
    /// Per-leg dimensions.
    dims: Vec<usize>,
    /// Total cost to contract this subset.
    cost: f64,
    /// Maximum intermediate size seen so far in this subtree.
    max_intermediate: usize,
}

impl PathOptimizer for DPOptimizer {
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
        if n > 20 {
            return Err(ContractionError::OptimizerFailed {
                optimizer: "dp".to_string(),
                reason: format!("n={n} exceeds practical DP limit of 20 tensors"),
            });
        }

        // Build leaf entries for each single-tensor subset.
        let mut memo: HashMap<u32, DPEntry> = HashMap::new();
        for i in 0..n {
            let (tid, legs) = &spec.tensors[i];
            let dims: Vec<usize> = (0..legs.len())
                .map(|pos| index_map.dim(*tid, pos).unwrap_or(1))
                .collect();
            let mask = 1u32 << i;
            memo.insert(
                mask,
                DPEntry {
                    node: ContractionNode::Input {
                        tensor_id: *tid,
                        indices: legs.clone(),
                    },
                    indices: legs.clone(),
                    dims,
                    cost: 0.0,
                    max_intermediate: 0,
                },
            );
        }

        let full_mask = (1u32 << n) - 1;

        // Iterate over subsets of increasing size (2..=n).
        for size in 2..=n {
            // Enumerate all subsets of `size` elements from n.
            for subset in SubsetIter::new(n as u32, size as u32) {
                let best = self.find_best_split(subset, n, spec, cost, max_memory_bytes, &memo)?;
                if let Some(entry) = best {
                    memo.insert(subset, entry);
                }
            }
        }

        let final_entry = memo
            .remove(&full_mask)
            .ok_or(ContractionError::OptimizerFailed {
                optimizer: "dp".to_string(),
                reason: "no valid contraction path found (memory constraint too tight?)".to_string(),
            })?;

        Ok(ContractionGraph {
            inputs: spec.tensors.iter().map(|(tid, _)| *tid).collect(),
            root: final_entry.node,
            estimated_flops: final_entry.cost,
            estimated_memory_bytes: 0,
            max_intermediate_size: final_entry.max_intermediate,
        })
    }

    fn name(&self) -> &str {
        "dp"
    }
}

impl DPOptimizer {
    /// Find the best way to split a subset into two non-empty complementary subsets.
    fn find_best_split(
        &self,
        subset: u32,
        n: usize,
        spec: &ContractionSpec,
        cost: &CostMetric,
        max_memory_bytes: Option<usize>,
        memo: &HashMap<u32, DPEntry>,
    ) -> ContractResult<Option<DPEntry>> {
        let mut best: Option<DPEntry> = None;

        // Enumerate all proper non-empty subsets of `subset`.
        // For each split (left, right) where left | right == subset and left & right == 0,
        // we only consider left < right to avoid duplicates.
        let mut left = (subset - 1) & subset;
        while left > 0 {
            let right = subset ^ left;
            // Only consider each split once (left < right).
            if left < right {
                if let (Some(l_entry), Some(r_entry)) = (memo.get(&left), memo.get(&right)) {
                    if let Some(entry) = self.try_contract(
                        l_entry,
                        r_entry,
                        n,
                        spec,
                        cost,
                        max_memory_bytes,
                    ) {
                        let is_better = best.as_ref().map_or(true, |b| entry.cost < b.cost);
                        if is_better {
                            best = Some(entry);
                        }
                    }
                }
            }
            left = (left - 1) & subset;
        }

        Ok(best)
    }

    /// Try contracting left and right entries, returning None if constraints violated.
    fn try_contract(
        &self,
        left: &DPEntry,
        right: &DPEntry,
        _n: usize,
        _spec: &ContractionSpec,
        cost: &CostMetric,
        max_memory_bytes: Option<usize>,
    ) -> Option<DPEntry> {
        // Find shared (contracted) indices between left and right.
        let mut contracted = Vec::new();
        for idx in &left.indices {
            if right.indices.contains(idx) {
                contracted.push(*idx);
            }
        }

        // Compute M, K, N.
        let mut m: usize = 1;
        let mut k: usize = 1;
        let mut n_dim: usize = 1;
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
            if !contracted.contains(idx) {
                n_dim *= dim;
                result_indices.push(*idx);
                result_dims.push(dim);
            }
        }

        let intermediate_size = m * n_dim;

        // Check max_width constraint.
        if intermediate_size > self.max_width {
            return None;
        }

        // Check memory constraint.
        if let Some(limit) = max_memory_bytes {
            let bytes = intermediate_size * 8; // assume f64
            if bytes > limit {
                return None;
            }
        }

        let step_cost = CostEstimator::step_cost::<f64>(m, k, n_dim, true, true, cost);
        let total_cost = left.cost + right.cost + step_cost;
        let max_intermediate = left
            .max_intermediate
            .max(right.max_intermediate)
            .max(intermediate_size);

        let node = ContractionNode::Contraction {
            left: Box::new(left.node.clone()),
            right: Box::new(right.node.clone()),
            contracted_indices: contracted,
            result_indices: result_indices.clone(),
        };

        Some(DPEntry {
            node,
            indices: result_indices,
            dims: result_dims,
            cost: total_cost,
            max_intermediate,
        })
    }
}

/// Iterator over all bitmask subsets of a given size from n elements.
struct SubsetIter {
    n: u32,
    size: u32,
    current: u32,
    done: bool,
}

impl SubsetIter {
    fn new(n: u32, size: u32) -> Self {
        if size == 0 || size > n {
            return SubsetIter {
                n,
                size,
                current: 0,
                done: true,
            };
        }
        // Start with the smallest subset of the given size: the lowest `size` bits set.
        let first = (1u32 << size) - 1;
        SubsetIter {
            n,
            size,
            current: first,
            done: false,
        }
    }
}

impl Iterator for SubsetIter {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.done {
            return None;
        }
        let result = self.current;
        // Gosper's hack: next combination of `size` bits in n-bit space.
        let c = self.current;
        let lowest = c & c.wrapping_neg();
        let ripple = c + lowest;
        let ones = ((c ^ ripple) >> 2) / lowest;
        let next = ripple | ones;
        if next >= (1u32 << self.n) || next <= c {
            self.done = true;
        } else {
            self.current = next;
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexSpec;

    #[test]
    fn dp_two_tensor_matmul() {
        let i = IndexId::from_raw(1000);
        let j = IndexId::from_raw(1001);
        let k = IndexId::from_raw(1002);

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

        let graph = DPOptimizer::default()
            .optimize(&spec, &index_map, &CostMetric::default(), None)
            .unwrap();

        assert_eq!(graph.n_pairwise_steps(), 1);
        assert_eq!(graph.inputs.len(), 2);
        // M=4, K=5, N=6 → flops = 2*4*5*6 = 240
        assert!((graph.estimated_flops - 240.0).abs() < 1e-12);
    }

    #[test]
    fn dp_three_tensor_chain() {
        let i = IndexId::from_raw(1100);
        let j = IndexId::from_raw(1101);
        let k = IndexId::from_raw(1102);
        let l = IndexId::from_raw(1103);

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

        let graph = DPOptimizer::default()
            .optimize(&spec, &index_map, &CostMetric::default(), None)
            .unwrap();

        assert_eq!(graph.n_pairwise_steps(), 2);
        assert!(graph.estimated_flops > 0.0);
    }

    #[test]
    fn dp_recovers_greedy_at_small_n() {
        // For n=2, DP should give the same result as greedy.
        let i = IndexId::from_raw(1200);
        let j = IndexId::from_raw(1201);
        let k = IndexId::from_raw(1202);

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

        let metric = CostMetric::default();
        let dp_graph = DPOptimizer::default()
            .optimize(&spec, &index_map, &metric, None)
            .unwrap();
        let greedy_graph = crate::optimizer::GreedyOptimizer
            .optimize(&spec, &index_map, &metric, None)
            .unwrap();

        // DP cost should be <= greedy cost (DP is optimal).
        assert!(dp_graph.estimated_flops <= greedy_graph.estimated_flops + 1e-12);
    }

    #[test]
    fn dp_memory_constraint_rejects() {
        let i = IndexId::from_raw(1300);
        let j = IndexId::from_raw(1301);
        let k = IndexId::from_raw(1302);

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

        // Intermediate is 1M elements * 8 bytes = 8MB. Set limit to 1MB.
        let result = DPOptimizer::default().optimize(
            &spec,
            &index_map,
            &CostMetric::default(),
            Some(1_000_000),
        );
        assert!(matches!(result, Err(ContractionError::OptimizerFailed { .. })));
    }

    #[test]
    fn subset_iter_generates_correct_subsets() {
        // All 2-element subsets of 4: {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
        let subsets: Vec<u32> = SubsetIter::new(4, 2).collect();
        assert_eq!(subsets.len(), 6); // C(4,2) = 6
        for s in &subsets {
            assert_eq!(s.count_ones(), 2);
            assert!(*s < 16); // < 2^4
        }
    }
}
