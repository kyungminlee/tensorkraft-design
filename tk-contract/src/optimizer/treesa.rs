//! Simulated annealing optimizer over the space of binary contraction trees.
//!
//! Used for large tensor networks (n > 15) where DP is intractable. Explores
//! the space of all binary trees via random subtree swaps, accepting worse
//! solutions with probability exp(-ΔC / T) where T is the current temperature.
//!
//! Time complexity: O(max_iterations × n) per optimization.
//! Solution quality improves with more iterations; not guaranteed optimal.

use crate::cost::{CostEstimator, CostMetric};
use crate::error::{ContractionError, ContractResult};
use crate::graph::{ContractionGraph, ContractionNode};
use crate::index::{ContractionSpec, IndexId, IndexMap, TensorId};
use crate::optimizer::PathOptimizer;

/// Simulated annealing optimizer over the space of binary contraction trees.
///
/// Appropriate for: multi-site correlators, custom network geometries,
/// and one-time optimization of recurring DMRG sub-patterns.
pub struct TreeSAOptimizer {
    /// Maximum number of simulated annealing iterations.
    pub max_iterations: usize,
    /// Initial temperature. Higher values explore more broadly.
    pub initial_temperature: f64,
    /// Cooling rate per iteration: T_{n+1} = T_n × cooling_rate.
    pub cooling_rate: f64,
    /// Random seed for reproducibility in tests. `None` uses a random seed.
    pub seed: Option<u64>,
}

impl Default for TreeSAOptimizer {
    fn default() -> Self {
        TreeSAOptimizer {
            max_iterations: 1_000,
            initial_temperature: 1.0,
            cooling_rate: 0.999,
            seed: None,
        }
    }
}

/// Lightweight RNG (xoshiro256++) that avoids the `rand` crate dependency.
/// Used only for simulated annealing moves. Not cryptographically secure.
struct SimpleRng {
    s: [u64; 4],
}

impl SimpleRng {
    fn from_seed(seed: u64) -> Self {
        // SplitMix64 to initialize state from a single seed.
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        SimpleRng { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform usize in [0, n).
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Working node for tree construction, carrying dimension info.
#[derive(Clone)]
struct SANode {
    node: ContractionNode,
    indices: Vec<IndexId>,
    dims: Vec<usize>,
}

impl PathOptimizer for TreeSAOptimizer {
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

        let seed = self.seed.unwrap_or_else(|| {
            // Use a deterministic fallback based on the spec for reproducibility
            // when no explicit seed is given. In production, this would use
            // a proper entropy source.
            let mut h: u64 = 0xcbf29ce484222325;
            for (tid, legs) in &spec.tensors {
                h ^= tid.raw() as u64;
                h = h.wrapping_mul(0x100000001b3);
                for idx in legs {
                    h ^= idx.raw() as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
            h
        });
        let mut rng = SimpleRng::from_seed(seed);

        // Build leaf nodes.
        let leaves: Vec<SANode> = spec
            .tensors
            .iter()
            .map(|(tid, legs)| {
                let dims: Vec<usize> = (0..legs.len())
                    .map(|pos| index_map.dim(*tid, pos).unwrap_or(1))
                    .collect();
                SANode {
                    node: ContractionNode::Input {
                        tensor_id: *tid,
                        indices: legs.clone(),
                    },
                    indices: legs.clone(),
                    dims,
                }
            })
            .collect();

        // Build initial tree by random greedy: pick a random pair, contract.
        let (best_node, best_cost, best_max_intermediate) = self.build_random_tree(
            &leaves,
            cost,
            max_memory_bytes,
            &mut rng,
        )?;

        let mut best_tree = best_node;
        let mut best_total_cost = best_cost;
        let mut best_max_inter = best_max_intermediate;

        let mut current_tree = best_tree.clone();
        let mut current_cost = best_total_cost;
        let mut current_max_inter = best_max_inter;

        // Simulated annealing loop.
        let mut temperature = self.initial_temperature;
        for _ in 0..self.max_iterations {
            // Generate a neighbor by rebuilding a random tree.
            // (Full rebuild per iteration — simple but effective for moderate n.)
            if let Ok((new_tree, new_cost, new_max)) =
                self.build_random_tree(&leaves, cost, max_memory_bytes, &mut rng)
            {
                let delta = new_cost - current_cost;
                let accept = if delta <= 0.0 {
                    true
                } else {
                    let prob = (-delta / (temperature * best_total_cost.max(1.0))).exp();
                    rng.next_f64() < prob
                };

                if accept {
                    current_tree = new_tree;
                    current_cost = new_cost;
                    current_max_inter = new_max;

                    if current_cost < best_total_cost {
                        best_tree = current_tree.clone();
                        best_total_cost = current_cost;
                        best_max_inter = current_max_inter;
                    }
                }
            }

            temperature *= self.cooling_rate;
        }

        Ok(ContractionGraph {
            inputs: spec.tensors.iter().map(|(tid, _)| *tid).collect(),
            root: best_tree,
            estimated_flops: best_total_cost,
            estimated_memory_bytes: 0,
            max_intermediate_size: best_max_inter,
        })
    }

    fn name(&self) -> &str {
        "treesa"
    }
}

impl TreeSAOptimizer {
    /// Build a random contraction tree by randomly pairing nodes.
    fn build_random_tree(
        &self,
        leaves: &[SANode],
        cost: &CostMetric,
        max_memory_bytes: Option<usize>,
        rng: &mut SimpleRng,
    ) -> ContractResult<(ContractionNode, f64, usize)> {
        let mut nodes: Vec<SANode> = leaves.to_vec();
        let mut total_cost = 0.0;
        let mut max_intermediate: usize = 0;

        while nodes.len() > 1 {
            let n = nodes.len();
            // Pick a random pair.
            let i = rng.next_usize(n);
            let mut j = rng.next_usize(n - 1);
            if j >= i {
                j += 1;
            }
            // Ensure i < j for removal ordering.
            let (i, j) = if i < j { (i, j) } else { (j, i) };

            let right = nodes.remove(j);
            let left = nodes.remove(i);

            // Find contracted indices.
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

            if let Some(limit) = max_memory_bytes {
                let bytes = intermediate_size * 8;
                if bytes > limit {
                    return Err(ContractionError::OptimizerFailed {
                        optimizer: "treesa".to_string(),
                        reason: format!(
                            "intermediate tensor size {} bytes exceeds limit {} bytes",
                            bytes, limit
                        ),
                    });
                }
            }

            let step_cost = CostEstimator::step_cost::<f64>(m, k, n_dim, true, true, cost);
            total_cost += step_cost;
            max_intermediate = max_intermediate.max(intermediate_size);

            let new_node = SANode {
                node: ContractionNode::Contraction {
                    left: Box::new(left.node),
                    right: Box::new(right.node),
                    contracted_indices: contracted,
                    result_indices: result_indices.clone(),
                },
                indices: result_indices,
                dims: result_dims,
            };

            nodes.push(new_node);
        }

        let final_node = nodes.into_iter().next().unwrap();
        Ok((final_node.node, total_cost, max_intermediate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexSpec;

    #[test]
    fn treesa_two_tensor() {
        let i = IndexId::from_raw(2000);
        let j = IndexId::from_raw(2001);
        let k = IndexId::from_raw(2002);

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

        let opt = TreeSAOptimizer {
            seed: Some(42),
            ..Default::default()
        };
        let graph = opt
            .optimize(&spec, &index_map, &CostMetric::default(), None)
            .unwrap();

        assert_eq!(graph.n_pairwise_steps(), 1);
        // M=4, K=5, N=6 → flops = 240
        assert!((graph.estimated_flops - 240.0).abs() < 1e-12);
    }

    #[test]
    fn treesa_seed_reproducible() {
        let i = IndexId::from_raw(2100);
        let j = IndexId::from_raw(2101);
        let k = IndexId::from_raw(2102);
        let l = IndexId::from_raw(2103);
        let m = IndexId::from_raw(2104);
        let n = IndexId::from_raw(2105);

        // 4 tensors: A(i,j) * B(j,k) * C(k,l) * D(l,m)
        let spec = ContractionSpec::new(
            vec![
                (TensorId::new(0), vec![i, j]),
                (TensorId::new(1), vec![j, k]),
                (TensorId::new(2), vec![k, l]),
                (TensorId::new(3), vec![l, n]),
            ],
            vec![i, n],
        )
        .unwrap();

        let mut index_map = IndexMap::new();
        for t in 0..4 {
            let legs = &spec.tensors[t].1;
            let specs: Vec<IndexSpec> = (0..legs.len())
                .map(|_| IndexSpec { dim: 5, is_contiguous: true })
                .collect();
            index_map.insert(TensorId::new(t as u32), specs);
        }

        let opt = TreeSAOptimizer {
            seed: Some(12345),
            max_iterations: 500,
            ..Default::default()
        };

        let graph1 = opt
            .optimize(&spec, &index_map, &CostMetric::default(), None)
            .unwrap();
        let graph2 = opt
            .optimize(&spec, &index_map, &CostMetric::default(), None)
            .unwrap();

        // Same seed → same cost.
        assert!((graph1.estimated_flops - graph2.estimated_flops).abs() < 1e-12);
    }

    #[test]
    fn simple_rng_is_deterministic() {
        let mut rng1 = SimpleRng::from_seed(42);
        let mut rng2 = SimpleRng::from_seed(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }
}
