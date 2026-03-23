//! Contraction graph: binary DAG representation of multi-tensor contractions.

use smallvec::SmallVec;

use crate::index::{IndexId, TensorId};

/// A node in the contraction DAG.
///
/// The tree is binary: every non-leaf node is a pairwise contraction of
/// exactly two children. Multi-tensor contractions are decomposed by the
/// `PathOptimizer` into a sequence of pairwise steps.
#[derive(Clone, Debug)]
pub enum ContractionNode {
    /// Leaf: one of the original input tensors.
    Input {
        tensor_id: TensorId,
        /// Current logical index ordering.
        indices: Vec<IndexId>,
    },
    /// Interior: pairwise contraction of two subtrees.
    Contraction {
        left: Box<ContractionNode>,
        right: Box<ContractionNode>,
        /// Contracted index IDs. Each entry is a single `IndexId` shared
        /// by both the left and right subtrees that is summed over.
        contracted_indices: Vec<IndexId>,
        /// Indices of the result tensor, in the order they appear after
        /// contraction. Free indices from `left` first, then from `right`.
        result_indices: Vec<IndexId>,
    },
}

impl ContractionNode {
    /// Ordered indices of the tensor this node produces.
    pub fn output_indices(&self) -> &[IndexId] {
        match self {
            ContractionNode::Input { indices, .. } => indices,
            ContractionNode::Contraction { result_indices, .. } => result_indices,
        }
    }

    /// Estimated rank of the output tensor.
    pub fn output_rank(&self) -> usize {
        self.output_indices().len()
    }
}

/// A fully-specified, cost-annotated contraction DAG.
///
/// Produced by `PathOptimizer::optimize`. Consumed by `ContractionExecutor::execute`.
#[derive(Clone, Debug)]
pub struct ContractionGraph {
    /// Original input tensors in registration order.
    pub inputs: Vec<TensorId>,
    /// Root node of the binary DAG.
    pub root: ContractionNode,
    /// Total estimated arithmetic cost (multiply-add operations).
    pub estimated_flops: f64,
    /// Estimated memory traffic (bytes moved for transpose/reshape operations).
    pub estimated_memory_bytes: usize,
    /// Maximum size (elements) of any intermediate tensor in the DAG.
    pub max_intermediate_size: usize,
}

impl ContractionGraph {
    /// Compute the optimal arena capacity needed to execute this graph
    /// without reallocation.
    pub fn arena_capacity_bytes<T>(&self) -> usize {
        self.max_intermediate_size * std::mem::size_of::<T>()
    }

    /// Number of pairwise contractions: `n_inputs - 1` for a tree DAG.
    pub fn n_pairwise_steps(&self) -> usize {
        if self.inputs.is_empty() {
            0
        } else {
            self.inputs.len() - 1
        }
    }

    /// Walk the DAG in post-order (children before parent) and return a
    /// flattened list of `PairwiseStep` structs ready for sequential execution.
    pub fn execution_order(&self) -> Vec<PairwiseStep> {
        let mut steps = Vec::new();
        let mut next_intermediate_id = self.inputs.iter().map(|t| t.raw()).max().unwrap_or(0) + 1;
        self.collect_steps(&self.root, &mut steps, &mut next_intermediate_id);
        steps
    }

    fn collect_steps(
        &self,
        node: &ContractionNode,
        steps: &mut Vec<PairwiseStep>,
        next_id: &mut u32,
    ) -> TensorId {
        match node {
            ContractionNode::Input { tensor_id, .. } => *tensor_id,
            ContractionNode::Contraction {
                left,
                right,
                contracted_indices,
                result_indices,
            } => {
                let left_id = self.collect_steps(left, steps, next_id);
                let right_id = self.collect_steps(right, steps, next_id);

                let result_id = TensorId::new(*next_id);
                *next_id += 1;

                // Determine which leg positions on each side are contracted.
                let left_indices = left.output_indices();
                let right_indices = right.output_indices();

                let mut left_contracted_legs = SmallVec::new();
                let mut right_contracted_legs = SmallVec::new();

                for c_idx in contracted_indices {
                    if let Some(pos) = left_indices.iter().position(|i| i == c_idx) {
                        left_contracted_legs.push(pos);
                    }
                    if let Some(pos) = right_indices.iter().position(|i| i == c_idx) {
                        right_contracted_legs.push(pos);
                    }
                }

                steps.push(PairwiseStep {
                    left_tensor_id: left_id,
                    right_tensor_id: right_id,
                    left_contracted_legs,
                    right_contracted_legs,
                    output_indices: result_indices.clone(),
                    result_tensor_id: result_id,
                });

                result_id
            }
        }
    }
}

/// A single pairwise contraction step extracted from the DAG.
#[derive(Clone, Debug)]
pub struct PairwiseStep {
    pub left_tensor_id: TensorId,
    pub right_tensor_id: TensorId,
    /// Legs of `left` that are summed (positions in left's output_indices).
    pub left_contracted_legs: SmallVec<[usize; 6]>,
    /// Legs of `right` that are summed (parallel to `left_contracted_legs`).
    pub right_contracted_legs: SmallVec<[usize; 6]>,
    /// Leg ordering of the output tensor.
    pub output_indices: Vec<IndexId>,
    /// Result tensor assigned this ID for subsequent steps.
    pub result_tensor_id: TensorId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_contraction_execution_order() {
        let i = IndexId::from_raw(0);
        let j = IndexId::from_raw(1);
        let k = IndexId::from_raw(2);

        let graph = ContractionGraph {
            inputs: vec![TensorId::new(0), TensorId::new(1)],
            root: ContractionNode::Contraction {
                left: Box::new(ContractionNode::Input {
                    tensor_id: TensorId::new(0),
                    indices: vec![i, j],
                }),
                right: Box::new(ContractionNode::Input {
                    tensor_id: TensorId::new(1),
                    indices: vec![j, k],
                }),
                contracted_indices: vec![j],
                result_indices: vec![i, k],
            },
            estimated_flops: 0.0,
            estimated_memory_bytes: 0,
            max_intermediate_size: 0,
        };

        let steps = graph.execution_order();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].left_tensor_id, TensorId::new(0));
        assert_eq!(steps[0].right_tensor_id, TensorId::new(1));
        assert_eq!(steps[0].left_contracted_legs.as_slice(), &[1]); // j is at position 1 in left
        assert_eq!(steps[0].right_contracted_legs.as_slice(), &[0]); // j is at position 0 in right
    }

    #[test]
    fn chain_contraction_execution_order() {
        let i = IndexId::from_raw(0);
        let j = IndexId::from_raw(1);
        let k = IndexId::from_raw(2);
        let l = IndexId::from_raw(3);

        // (A(i,j) * B(j,k)) * C(k,l) → D(i,l)
        let graph = ContractionGraph {
            inputs: vec![TensorId::new(0), TensorId::new(1), TensorId::new(2)],
            root: ContractionNode::Contraction {
                left: Box::new(ContractionNode::Contraction {
                    left: Box::new(ContractionNode::Input {
                        tensor_id: TensorId::new(0),
                        indices: vec![i, j],
                    }),
                    right: Box::new(ContractionNode::Input {
                        tensor_id: TensorId::new(1),
                        indices: vec![j, k],
                    }),
                    contracted_indices: vec![j],
                    result_indices: vec![i, k],
                }),
                right: Box::new(ContractionNode::Input {
                    tensor_id: TensorId::new(2),
                    indices: vec![k, l],
                }),
                contracted_indices: vec![k],
                result_indices: vec![i, l],
            },
            estimated_flops: 0.0,
            estimated_memory_bytes: 0,
            max_intermediate_size: 0,
        };

        let steps = graph.execution_order();
        assert_eq!(steps.len(), 2);
        // First step: A*B
        assert_eq!(steps[0].left_tensor_id, TensorId::new(0));
        assert_eq!(steps[0].right_tensor_id, TensorId::new(1));
        // Second step: (A*B) * C
        assert_eq!(steps[1].right_tensor_id, TensorId::new(2));
    }
}
