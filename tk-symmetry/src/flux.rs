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
    let mut results = Vec::new();
    let mut current = Vec::with_capacity(indices.len());
    enumerate_recursive(indices, flux, leg_directions, &mut current, &mut results);
    results
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

        // Required q on the last leg:
        // Incoming: partial_fused ⊕ q = flux  =>  q = flux ⊕ partial_fused.dual()
        // Outgoing: partial_fused ⊕ q.dual() = flux  =>  q = (flux ⊕ partial_fused.dual()).dual()
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
}
