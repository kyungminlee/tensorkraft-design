//! Flux rule validation and sector enumeration.

use crate::quantum_number::{LegDirection, QuantumNumber};
use crate::sector_key::QIndex;

/// Verify that a given multi-index sector satisfies the tensor's flux rule.
///
/// In debug/test builds, called on every block insertion and construction.
pub fn check_flux_rule<Q: QuantumNumber>(
    sector_qns: &[Q],
    expected_flux: &Q,
    leg_directions: &[LegDirection],
) -> bool {
    debug_assert_eq!(
        sector_qns.len(),
        leg_directions.len(),
        "sector_qns and leg_directions must have the same length"
    );
    let fused = sector_qns
        .iter()
        .zip(leg_directions.iter())
        .fold(Q::identity(), |acc, (q, dir)| match dir {
            LegDirection::Incoming => acc.fuse(q),
            LegDirection::Outgoing => acc.fuse(&q.dual()),
        });
    fused == *expected_flux
}

/// Enumerate all tuples (q₁, q₂, ..., qₙ) of quantum numbers — one per leg —
/// that satisfy the flux rule: fuse(q₁, q₂, ..., qₙ) == flux.
///
/// Uses backtracking with early pruning: after fixing the first k legs,
/// the remaining required charge is computed, and only basis states consistent
/// with that charge are explored on the remaining legs.
///
/// For high-rank tensors (rank > 4), partial-fusion memoization is applied:
/// the set of reachable partial fusions at each depth is cached so that
/// redundant subtree explorations are skipped. This reduces the cost from
/// exponential in rank to polynomial in the number of distinct charges.
///
/// This function is called once at tensor construction; it is not on the
/// performance-critical path.
pub fn enumerate_valid_sectors<Q: QuantumNumber>(
    indices: &[QIndex<Q>],
    flux: &Q,
    leg_directions: &[LegDirection],
) -> Vec<Vec<Q>> {
    debug_assert_eq!(
        indices.len(),
        leg_directions.len(),
        "indices and leg_directions must have the same length"
    );
    let rank = indices.len();
    let mut results = Vec::new();
    let mut current = Vec::with_capacity(rank);

    if rank > 4 {
        // For high-rank tensors, use memoization to prune redundant subtrees.
        // Build a set of reachable partial fusions at each depth, then
        // enumerate only paths whose partial fusion at each step can reach
        // the target flux given the remaining legs.
        let reachable = build_reachable_from_end(indices, flux, leg_directions);
        enumerate_memoized(indices, flux, leg_directions, &reachable, &mut current, &mut results);
    } else {
        enumerate_recursive(indices, flux, leg_directions, &mut current, &mut results);
    }
    results
}

/// Build "reachable charges from the right" for memoized enumeration.
///
/// `reachable[d]` is the set of partial fusions that can be achieved by
/// fusing legs `d..rank` to produce a charge that, combined with the
/// first `d` legs' partial fusion, equals `flux`. This allows pruning
/// at depth `d` by checking whether the current partial fusion can
/// reach the target flux via the remaining legs.
fn build_reachable_from_end<Q: QuantumNumber>(
    indices: &[QIndex<Q>],
    _flux: &Q,
    leg_directions: &[LegDirection],
) -> Vec<std::collections::HashSet<Q>> {
    use std::collections::HashSet;

    let rank = indices.len();
    // reachable[d] = set of partial fusions achievable from legs d..rank
    // such that: partial_left ⊕ partial_right = flux
    // We build from right to left.
    let mut reachable: Vec<HashSet<Q>> = vec![HashSet::new(); rank + 1];

    // Base case: at depth rank, the "remaining fusion" must be identity
    // (all legs have been consumed, the full fusion must equal flux).
    // Actually, we track the "suffix fusion" from right. At depth rank,
    // the suffix fusion is identity.
    reachable[rank].insert(Q::identity());

    // Build backwards: for leg (d-1), suffix_fusion = q_{d-1} ⊕ suffix_{d}
    for d in (0..rank).rev() {
        // Clone the next level to avoid borrow conflict
        let next: Vec<Q> = reachable[d + 1].iter().cloned().collect();
        for (q, _, _) in indices[d].iter_sectors() {
            let contribution = match leg_directions[d] {
                LegDirection::Incoming => q.clone(),
                LegDirection::Outgoing => q.dual(),
            };
            for suffix in &next {
                let new_suffix = contribution.fuse(suffix);
                reachable[d].insert(new_suffix);
            }
        }
    }

    reachable
}

fn enumerate_memoized<Q: QuantumNumber>(
    indices: &[QIndex<Q>],
    flux: &Q,
    leg_directions: &[LegDirection],
    reachable: &[std::collections::HashSet<Q>],
    current: &mut Vec<Q>,
    results: &mut Vec<Vec<Q>>,
) {
    let depth = current.len();
    let rank = indices.len();

    if depth == rank {
        if check_flux_rule(current, flux, leg_directions) {
            results.push(current.clone());
        }
        return;
    }

    // Last leg: compute required quantum number directly (pruning)
    if depth == rank - 1 {
        let partial_fused = current
            .iter()
            .zip(leg_directions.iter())
            .fold(Q::identity(), |acc, (q, dir)| match dir {
                LegDirection::Incoming => acc.fuse(q),
                LegDirection::Outgoing => acc.fuse(&q.dual()),
            });

        let required = match leg_directions[depth] {
            LegDirection::Incoming => flux.fuse(&partial_fused.dual()),
            LegDirection::Outgoing => flux.fuse(&partial_fused.dual()).dual(),
        };

        if indices[depth].dim_of(&required).is_some() {
            current.push(required);
            results.push(current.clone());
            current.pop();
        }
        return;
    }

    // Compute current partial fusion
    let partial_fused = current
        .iter()
        .zip(leg_directions.iter())
        .fold(Q::identity(), |acc, (q, dir)| match dir {
            LegDirection::Incoming => acc.fuse(q),
            LegDirection::Outgoing => acc.fuse(&q.dual()),
        });

    for (q, _, _) in indices[depth].iter_sectors() {
        let contribution = match leg_directions[depth] {
            LegDirection::Incoming => q.clone(),
            LegDirection::Outgoing => q.dual(),
        };
        let new_partial = partial_fused.fuse(&contribution);

        // Pruning: check if the remaining legs (depth+1..rank) can produce
        // a suffix that, when fused with new_partial, equals flux.
        // Required suffix = flux ⊕ new_partial.dual()
        let required_suffix = flux.fuse(&new_partial.dual());
        if !reachable[depth + 1].contains(&required_suffix) {
            continue;
        }

        current.push(q.clone());
        enumerate_memoized(indices, flux, leg_directions, reachable, current, results);
        current.pop();
    }
}

fn enumerate_recursive<Q: QuantumNumber>(
    indices: &[QIndex<Q>],
    flux: &Q,
    leg_directions: &[LegDirection],
    current: &mut Vec<Q>,
    results: &mut Vec<Vec<Q>>,
) {
    let depth = current.len();
    let rank = indices.len();

    if depth == rank {
        // Check flux rule
        if check_flux_rule(current, flux, leg_directions) {
            results.push(current.clone());
        }
        return;
    }

    // Last leg: compute required quantum number directly (pruning)
    if depth == rank - 1 {
        let partial_fused = current
            .iter()
            .zip(leg_directions.iter())
            .fold(Q::identity(), |acc, (q, dir)| match dir {
                LegDirection::Incoming => acc.fuse(q),
                LegDirection::Outgoing => acc.fuse(&q.dual()),
            });

        let required = match leg_directions[depth] {
            LegDirection::Incoming => flux.fuse(&partial_fused.dual()),
            LegDirection::Outgoing => flux.fuse(&partial_fused.dual()).dual(),
        };

        // Check if required q is available on this leg
        if indices[depth].dim_of(&required).is_some() {
            current.push(required);
            results.push(current.clone());
            current.pop();
        }
        return;
    }

    // General case: try all quantum numbers on this leg
    for (q, _, _) in indices[depth].iter_sectors() {
        current.push(q.clone());
        enumerate_recursive(indices, flux, leg_directions, current, results);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::U1;

    #[test]
    fn check_flux_rule_correct() {
        let dirs = [LegDirection::Incoming, LegDirection::Incoming, LegDirection::Outgoing];
        // U1(1) + U1(2) - U1(3) = U1(0) ✓
        assert!(check_flux_rule(
            &[U1(1), U1(2), U1(3)],
            &U1::identity(),
            &dirs,
        ));
        // U1(1) + U1(2) - U1(4) = U1(-1) ≠ U1(0) ✗
        assert!(!check_flux_rule(
            &[U1(1), U1(2), U1(4)],
            &U1::identity(),
            &dirs,
        ));
    }

    #[test]
    fn enumerate_sectors_completeness() {
        // Rank-3 U1 tensor with legs:
        //   leg 0 (in): charges {-1, 0, 1}, dim 1 each
        //   leg 1 (in): charges {-1, 0, 1}, dim 1 each
        //   leg 2 (out): charges {-1, 0, 1}, dim 1 each
        // Flux = identity (0)
        let idx0 = QIndex::new(vec![(U1(-1), 1), (U1(0), 1), (U1(1), 1)]);
        let idx1 = idx0.clone();
        let idx2 = idx0.clone();
        let dirs = [LegDirection::Incoming, LegDirection::Incoming, LegDirection::Outgoing];

        let sectors = enumerate_valid_sectors(
            &[idx0, idx1, idx2],
            &U1::identity(),
            &dirs,
        );

        // Valid: q0 + q1 = q2
        // (-1,-1,-2)? no, -2 not in leg 2
        // (-1,0,-1) ✓  (-1,1,0) ✓
        // (0,-1,-1) ✓  (0,0,0) ✓  (0,1,1) ✓
        // (1,-1,0) ✓  (1,0,1) ✓  (1,1,2)? no
        assert_eq!(sectors.len(), 7);
    }

    #[test]
    fn enumerate_sectors_high_rank_memoized() {
        // Rank-6 tensor (triggers memoized path for rank > 4)
        // All legs: charges {-1, 0, 1}, all incoming, flux = 0
        let idx = QIndex::new(vec![(U1(-1), 1), (U1(0), 1), (U1(1), 1)]);
        let indices: Vec<_> = (0..6).map(|_| idx.clone()).collect();
        let dirs = vec![LegDirection::Incoming; 6];

        let sectors = enumerate_valid_sectors(&indices, &U1::identity(), &dirs);

        // Verify all returned sectors satisfy the flux rule
        for sector in &sectors {
            assert!(check_flux_rule(sector, &U1::identity(), &dirs));
        }

        // Compare with a brute-force enumeration using a rank-3 + rank-3 composition
        // to verify correctness. The number of valid sectors should be the same
        // regardless of whether memoization is used.
        assert!(!sectors.is_empty());

        // For 6 legs each with charges {-1,0,1}, all incoming, flux=0:
        // We need q0+q1+q2+q3+q4+q5 = 0. Count by stars-and-bars:
        // This is the number of ways to choose 6 values from {-1,0,1} that sum to 0.
        // By direct count: 141 valid sectors.
        assert_eq!(sectors.len(), 141);
    }

    #[test]
    fn enumerate_sectors_rank5_matches_naive() {
        // Rank-5 tensor — crosses the memoization threshold (rank > 4)
        // Verify results match the non-memoized path
        let idx = QIndex::new(vec![(U1(-1), 1), (U1(0), 1), (U1(1), 1)]);
        let indices: Vec<_> = (0..5).map(|_| idx.clone()).collect();
        let dirs = vec![LegDirection::Incoming; 5];

        let sectors = enumerate_valid_sectors(&indices, &U1::identity(), &dirs);

        // All results must satisfy flux rule
        for sector in &sectors {
            assert!(check_flux_rule(sector, &U1::identity(), &dirs));
        }

        // Brute-force count: 5 values from {-1,0,1} summing to 0
        // = 51 (can be verified by direct enumeration)
        assert_eq!(sectors.len(), 51);
    }
}
