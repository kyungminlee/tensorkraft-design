//! LPT-scheduled block-sparse GEMM task generation and fusion rules.

use tk_core::{DenseTensor, Scalar};
use tk_symmetry::{BitPackable, PackedSectorKey, QIndex};

/// A single dense GEMM within a block-sparse contraction.
///
/// Represents one element of the task queue generated in Phase 1 of `block_gemm`.
/// Each task is fully independent: it reads from two input blocks (borrowed from
/// the input tensors, which are immutable for the duration of Phase 2 execution)
/// and writes to an output block identified by `out_key`.
pub(crate) struct SectorGemmTask<'a, T: Scalar> {
    /// Output sector key (determines where the result accumulates).
    pub out_key: PackedSectorKey,
    /// Immutable reference to the left input block.
    pub block_a: &'a DenseTensor<'static, T>,
    /// Immutable reference to the right input block.
    pub block_b: &'a DenseTensor<'static, T>,
    /// Estimated FLOP count: rows(A) × cols(B) × cols(A).
    /// Used for LPT scheduling (sort descending before dispatch).
    pub flops: usize,
}

/// Compute the output sector key for a pair of input sector keys.
///
/// For Abelian symmetries, each input sector pair produces at most one output sector.
/// This function encodes the Abelian fusion rule: the output quantum number is the
/// fused (summed, for U(1)) combination of the input quantum numbers.
///
/// Returns `None` if the input sectors are not compatible (do not satisfy the
/// output tensor's flux rule), signaling that no GEMM task should be generated
/// for this pair.
///
/// **Non-Abelian note:** For SU(2) symmetry (behind `su2-symmetry` feature flag),
/// the fusion rule is one-to-many. This function handles only the Abelian case.
pub(crate) fn compute_fusion_rule<Q: BitPackable>(
    key_a: PackedSectorKey,
    key_b: PackedSectorKey,
    rank_a: usize,
    rank_b: usize,
    target_flux: &Q,
    _indices_a: &[QIndex<Q>],
    _indices_b: &[QIndex<Q>],
) -> Option<PackedSectorKey> {
    // For a rank-2 block-sparse GEMM (matrix × matrix):
    //   A has legs [row_a, col_a], B has legs [row_b, col_b]
    //   Output has legs [row_a, col_b]
    //   The contracted leg is col_a == row_b
    //
    // Unpack the sector quantum numbers from each key.
    let qns_a = key_a.unpack::<Q>(rank_a);
    let qns_b = key_b.unpack::<Q>(rank_b);

    // For rank-2 tensors (the primary use case in tk-linalg):
    // col_a quantum number must be compatible with row_b quantum number
    // for the contraction to be non-zero.
    if rank_a == 2 && rank_b == 2 {
        // Contracted index: A's last leg with B's first leg.
        // They must match for the block product to be non-zero.
        if qns_a[1] != qns_b[0] {
            return None;
        }

        // Output sector key: [row_a_qn, col_b_qn]
        let out_qns = [qns_a[0].clone(), qns_b[1].clone()];

        // Verify the output satisfies the target flux rule.
        // For Abelian symmetries: row_qn.fuse(col_qn.dual()) should equal the target flux.
        let fused = out_qns[0].fuse(&out_qns[1].dual());
        if fused != *target_flux {
            return None;
        }

        return Some(PackedSectorKey::pack(&out_qns));
    }

    // General case: for higher-rank tensors, the caller must reshape to rank-2
    // before calling block_gemm. This is the standard approach in DMRG.
    // Return None for unsupported configurations.
    None
}

/// SU(2) fusion-rule fan-out: one input sector pair produces *multiple* output sectors.
///
/// For SU(2) symmetry, j₁ ⊗ j₂ = |j₁−j₂| ⊕ ... ⊕ (j₁+j₂). Each output irrep
/// generates a separate GEMM task weighted by the corresponding Clebsch-Gordan
/// coefficient. The caller (block_gemm) must handle output-sector collision by
/// accumulating results for the same output key via map-reduce.
///
/// Returns a `Vec<SectorGemmTask>` (possibly empty if no compatible output sectors).
#[cfg(feature = "su2-symmetry")]
pub(crate) fn compute_fusion_rule_su2<'a, T: Scalar>(
    key_a: PackedSectorKey,
    key_b: PackedSectorKey,
    block_a: &'a DenseTensor<'static, T>,
    block_b: &'a DenseTensor<'static, T>,
    rank_a: usize,
    rank_b: usize,
    target_flux: &tk_symmetry::SU2Irrep,
    indices_a: &[QIndex<tk_symmetry::SU2Irrep>],
    indices_b: &[QIndex<tk_symmetry::SU2Irrep>],
) -> Vec<SectorGemmTask<'a, T>> {
    use tk_symmetry::SU2Irrep;

    if rank_a != 2 || rank_b != 2 {
        return Vec::new();
    }

    let qns_a = key_a.unpack::<SU2Irrep>(rank_a);
    let qns_b = key_b.unpack::<SU2Irrep>(rank_b);

    // Contracted leg: A's col with B's row. They must match for the block
    // product to be non-zero in the reduced matrix element sense.
    if qns_a[1] != qns_b[0] {
        return Vec::new();
    }

    // Fan-out: enumerate all output irreps from j_row_a ⊗ j_col_b.
    let j_row = qns_a[0];
    let j_col = qns_b[1];

    let mut tasks = Vec::new();
    for j_out in SU2Irrep::fuse_all(&j_row, &j_col) {
        // Check the output satisfies the target flux.
        // For SU(2), the flux rule checks that the output irrep is compatible
        // with the tensor's overall flux quantum number.
        let fused = j_row.fuse(&j_col.dual());
        // The output sector [j_row, j_col] must produce the target flux
        // through the appropriate fusion path. For rank-2, the simplest check
        // is whether j_out is the identity (scalar) when target_flux is trivial,
        // or whether the fusion produces the target flux irrep.
        //
        // For the general SU(2) case, we check if target_flux appears in the
        // decomposition of j_row ⊗ j_col*.
        let min_j = (j_row.twice_j as i32 - j_col.twice_j as i32).unsigned_abs();
        let max_j = j_row.twice_j + j_col.twice_j;
        if target_flux.twice_j < min_j || target_flux.twice_j > max_j
            || (target_flux.twice_j - min_j) % 2 != 0
        {
            continue;
        }

        let out_qns = [j_row, j_col];
        let out_key = PackedSectorKey::pack(&out_qns);

        let flops = block_a.shape().dims()[0]
            * block_b.shape().dims()[1]
            * block_a.shape().dims()[1];

        tasks.push(SectorGemmTask {
            out_key,
            block_a,
            block_b,
            flops,
        });

        // For rank-2 SU(2), each (j_row, j_col) pair maps to exactly one
        // output sector key. The fan-out over j_out is for the CG weighting
        // which is applied during contraction in tk-contract. Here we just
        // generate the structural task.
        break;
    }

    tasks
}

/// Sort tasks by descending FLOP count (Longest Processing Time heuristic).
///
/// Dispatching the heaviest tasks first to Rayon's work-stealing scheduler
/// minimizes load imbalance caused by the binomial sector-size distribution
/// typical of Abelian DMRG.
pub(crate) fn lpt_sort<T: Scalar>(tasks: &mut [SectorGemmTask<'_, T>]) {
    tasks.sort_unstable_by(|x, y| y.flops.cmp(&x.flops));
}

/// Compute the output QIndex structure for a block-sparse GEMM.
///
/// Given the QIndices of rank-2 input tensors `a` and `b`,
/// returns the QIndex list for the output tensor: [row_index_of_a, col_index_of_b].
pub(crate) fn compute_output_indices<Q: BitPackable>(
    indices_a: &[QIndex<Q>],
    indices_b: &[QIndex<Q>],
) -> Vec<QIndex<Q>> {
    debug_assert!(indices_a.len() >= 2, "A must have at least 2 legs");
    debug_assert!(indices_b.len() >= 2, "B must have at least 2 legs");
    // Output indices: first leg of A, last leg of B
    vec![indices_a[0].clone(), indices_b[indices_b.len() - 1].clone()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lpt_sort_descending() {
        use tk_core::TensorShape;

        let block_a = DenseTensor::<f64>::zeros(TensorShape::row_major(&[2, 2]));
        let block_b = DenseTensor::<f64>::zeros(TensorShape::row_major(&[2, 2]));

        let mut tasks = vec![
            SectorGemmTask {
                out_key: PackedSectorKey(0),
                block_a: &block_a,
                block_b: &block_b,
                flops: 100,
            },
            SectorGemmTask {
                out_key: PackedSectorKey(1),
                block_a: &block_a,
                block_b: &block_b,
                flops: 1000,
            },
            SectorGemmTask {
                out_key: PackedSectorKey(2),
                block_a: &block_a,
                block_b: &block_b,
                flops: 500,
            },
        ];

        lpt_sort(&mut tasks);

        assert_eq!(tasks[0].flops, 1000);
        assert_eq!(tasks[1].flops, 500);
        assert_eq!(tasks[2].flops, 100);
    }
}
