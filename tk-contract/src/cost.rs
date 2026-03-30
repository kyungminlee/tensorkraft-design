//! Cost metric and estimator for contraction path optimization.

use tk_core::Scalar;

use crate::index::{IndexMap, TensorId};

/// Composite cost metric for contraction path optimization.
///
/// The total cost of a contraction path is:
///   C_total = α · FLOPs + β · Bytes_Moved
///
/// where β >> α ensures that transpose costs dominate the ordering decision
/// whenever they exist. The default values (α=1, β=50) encode the empirical
/// observation that one byte transferred is approximately as expensive as
/// one floating-point multiply-add on current hardware.
///
/// Conjugation metadata from `MatRef::is_conjugated` contributes ZERO to
/// `Bytes_Moved` because Hermitian conjugates are realized via flag-flip.
#[derive(Clone, Debug)]
pub struct CostMetric {
    /// Weight α for arithmetic operations (one unit = one FMA).
    pub flop_weight: f64,
    /// Weight β for memory traffic (one unit = one byte transferred).
    pub bandwidth_weight: f64,
}

impl Default for CostMetric {
    fn default() -> Self {
        CostMetric {
            flop_weight: 1.0,
            bandwidth_weight: 50.0,
        }
    }
}

/// Stateless utilities for estimating the cost of a pairwise contraction.
pub struct CostEstimator;

impl CostEstimator {
    /// Estimate FLOP count for a pairwise GEMM of shapes (M×K) @ (K×N).
    /// Returns `2 * M * K * N` (multiply + add per element of output).
    #[inline]
    pub fn flop_count(m: usize, k: usize, n: usize) -> f64 {
        2.0 * m as f64 * k as f64 * n as f64
    }

    /// Estimate bytes moved for a transpose of the given number of elements.
    ///
    /// Rules:
    /// - Contiguous input or Hermitian conjugate (flag flip) → 0 cost.
    /// - Non-contiguous requires explicit out-of-place transpose → `numel * sizeof::<T>()`.
    pub fn transpose_cost_bytes<T: Scalar>(numel: usize, is_contiguous: bool) -> usize {
        if is_contiguous {
            0
        } else {
            numel * std::mem::size_of::<T>()
        }
    }

    /// Composite cost for one pairwise contraction step.
    ///
    /// Given two tensors being contracted, computes the total cost as
    /// `flop_weight * flops + bandwidth_weight * bytes_moved`.
    pub fn step_cost<T: Scalar>(
        m: usize,
        k: usize,
        n: usize,
        left_contiguous: bool,
        right_contiguous: bool,
        metric: &CostMetric,
    ) -> f64 {
        let flops = Self::flop_count(m, k, n);
        let left_numel = m * k;
        let right_numel = k * n;
        let bytes = Self::transpose_cost_bytes::<T>(left_numel, left_contiguous)
            + Self::transpose_cost_bytes::<T>(right_numel, right_contiguous);
        metric.flop_weight * flops + metric.bandwidth_weight * bytes as f64
    }

    /// Estimate the cost of contracting two tensors identified by their IDs.
    ///
    /// Uses the `IndexMap` to look up dimensions and determine the M, K, N
    /// sizes for the GEMM operation.
    pub fn pairwise_cost<T: Scalar>(
        tensor_a: TensorId,
        legs_a: &[usize],
        contracted_legs_a: &[usize],
        tensor_b: TensorId,
        legs_b: &[usize],
        contracted_legs_b: &[usize],
        index_map: &IndexMap,
        metric: &CostMetric,
    ) -> f64 {
        // M = product of free legs of A
        // K = product of contracted legs
        // N = product of free legs of B
        let mut m: usize = 1;
        let mut k: usize = 1;
        let mut n: usize = 1;

        for (pos, _leg) in legs_a.iter().enumerate() {
            let dim = index_map.dim(tensor_a, pos).unwrap_or(1);
            if contracted_legs_a.contains(&pos) {
                k *= dim;
            } else {
                m *= dim;
            }
        }

        for (pos, _leg) in legs_b.iter().enumerate() {
            let dim = index_map.dim(tensor_b, pos).unwrap_or(1);
            if contracted_legs_b.contains(&pos) {
                // Already counted in k
            } else {
                n *= dim;
            }
        }

        // For cost estimation, assume non-contiguous (conservative).
        // The actual contiguity check happens at execution time.
        Self::step_cost::<T>(m, k, n, true, true, metric)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flop_count_basic() {
        assert!((CostEstimator::flop_count(4, 5, 6) - 240.0).abs() < 1e-12);
    }

    #[test]
    fn zero_conjugation_cost() {
        // Contiguous → zero cost
        assert_eq!(CostEstimator::transpose_cost_bytes::<f64>(100, true), 0);
    }

    #[test]
    fn noncontiguous_has_cost() {
        let cost = CostEstimator::transpose_cost_bytes::<f64>(100, false);
        assert_eq!(cost, 100 * 8); // 100 elements * 8 bytes per f64
    }

    #[test]
    fn step_cost_contiguous() {
        let metric = CostMetric::default();
        // M=4, K=5, N=6, both contiguous → only FLOP cost
        let cost = CostEstimator::step_cost::<f64>(4, 5, 6, true, true, &metric);
        assert!((cost - 240.0).abs() < 1e-12);
    }

    #[test]
    fn step_cost_with_transpose() {
        let metric = CostMetric::default();
        // M=4, K=5, N=6, left non-contiguous
        let cost = CostEstimator::step_cost::<f64>(4, 5, 6, false, true, &metric);
        let expected = 240.0 + 50.0 * (4 * 5 * 8) as f64;
        assert!((cost - expected).abs() < 1e-12);
    }

    #[test]
    fn cost_metric_custom_weights() {
        let metric = CostMetric {
            flop_weight: 2.0,
            bandwidth_weight: 100.0,
        };
        // M=4, K=5, N=6, both contiguous → only FLOP cost with weight 2
        let cost = CostEstimator::step_cost::<f64>(4, 5, 6, true, true, &metric);
        assert!((cost - 480.0).abs() < 1e-12); // 2 * 240
    }

    #[test]
    fn flop_count_edge_cases() {
        // Zero dimension → zero flops
        assert!((CostEstimator::flop_count(0, 5, 6) - 0.0).abs() < 1e-12);
        assert!((CostEstimator::flop_count(4, 0, 6) - 0.0).abs() < 1e-12);
        assert!((CostEstimator::flop_count(4, 5, 0) - 0.0).abs() < 1e-12);
        // Single element
        assert!((CostEstimator::flop_count(1, 1, 1) - 2.0).abs() < 1e-12);
    }
}
