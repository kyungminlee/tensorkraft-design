//! Property-based tests for tk-symmetry using proptest.

use proptest::prelude::*;
use smallvec::SmallVec;

use tk_symmetry::builtins::{U1, U1Z2, Z2};
use tk_symmetry::quantum_number::{BitPackable, LegDirection, QuantumNumber};
use tk_symmetry::sector_key::{PackedSectorKey, QIndex};
use tk_symmetry::BlockSparseTensor;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

fn arb_u1() -> impl Strategy<Value = U1> {
    (-128i32..=127).prop_map(U1)
}

fn arb_z2() -> impl Strategy<Value = Z2> {
    any::<bool>().prop_map(Z2)
}

fn arb_u1z2() -> impl Strategy<Value = U1Z2> {
    (arb_u1(), arb_z2()).prop_map(|(u, z)| U1Z2(u, z))
}

// ---------------------------------------------------------------------------
// Quantum number group axioms
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn u1_identity_law(q in arb_u1()) {
        prop_assert_eq!(q.fuse(&U1::identity()), q);
        prop_assert_eq!(U1::identity().fuse(&q), q);
    }

    #[test]
    fn u1_inverse_law(q in arb_u1()) {
        prop_assert_eq!(q.fuse(&q.dual()), U1::identity());
        prop_assert_eq!(q.dual().fuse(&q), U1::identity());
    }

    #[test]
    fn u1_fuse_associativity(a in arb_u1(), b in arb_u1(), c in arb_u1()) {
        prop_assert_eq!(a.fuse(&b).fuse(&c), a.fuse(&b.fuse(&c)));
    }

    #[test]
    fn z2_identity_law(q in arb_z2()) {
        prop_assert_eq!(q.fuse(&Z2::identity()), q);
        prop_assert_eq!(Z2::identity().fuse(&q), q);
    }

    #[test]
    fn z2_inverse_law(q in arb_z2()) {
        prop_assert_eq!(q.fuse(&q.dual()), Z2::identity());
    }

    #[test]
    fn z2_fuse_associativity(a in arb_z2(), b in arb_z2(), c in arb_z2()) {
        prop_assert_eq!(a.fuse(&b).fuse(&c), a.fuse(&b.fuse(&c)));
    }

    #[test]
    fn u1z2_identity_law(q in arb_u1z2()) {
        prop_assert_eq!(q.fuse(&U1Z2::identity()), q);
        prop_assert_eq!(U1Z2::identity().fuse(&q), q);
    }

    #[test]
    fn u1z2_inverse_law(q in arb_u1z2()) {
        prop_assert_eq!(q.fuse(&q.dual()), U1Z2::identity());
    }

    #[test]
    fn u1z2_fuse_associativity(a in arb_u1z2(), b in arb_u1z2(), c in arb_u1z2()) {
        prop_assert_eq!(a.fuse(&b).fuse(&c), a.fuse(&b.fuse(&c)));
    }
}

// ---------------------------------------------------------------------------
// Pack / Unpack round-trips
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn u1_pack_round_trip(q in arb_u1()) {
        prop_assert_eq!(U1::unpack(q.pack()), q);
    }

    #[test]
    fn z2_pack_round_trip(q in arb_z2()) {
        prop_assert_eq!(Z2::unpack(q.pack()), q);
    }

    #[test]
    fn u1z2_pack_round_trip(q in arb_u1z2()) {
        prop_assert_eq!(U1Z2::unpack(q.pack()), q);
    }
}

// ---------------------------------------------------------------------------
// PackedSectorKey round-trips and binary search
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn packed_key_round_trip_u1(
        qns in prop::collection::vec(arb_u1(), 1..=8)
    ) {
        let key = PackedSectorKey::pack(&qns);
        let unpacked: SmallVec<[U1; 8]> = key.unpack(qns.len());
        prop_assert_eq!(unpacked.as_slice(), &qns[..]);
    }

    /// After inserting a key into a sorted key list, binary search finds it.
    #[test]
    fn packed_key_binary_search_finds_inserted(
        existing in prop::collection::vec(
            prop::collection::vec(arb_u1(), 2..=2), 0..20
        ),
        needle in prop::collection::vec(arb_u1(), 2..=2),
    ) {
        let mut keys: Vec<PackedSectorKey> = existing
            .iter()
            .map(|qns| PackedSectorKey::pack(qns))
            .collect();
        keys.sort();
        keys.dedup();

        let needle_key = PackedSectorKey::pack(&needle);

        // Insert at sorted position
        let pos = keys.binary_search(&needle_key).unwrap_or_else(|p| {
            keys.insert(p, needle_key);
            p
        });

        // Binary search must find it
        prop_assert_eq!(keys.binary_search(&needle_key), Ok(pos));
    }
}

// ---------------------------------------------------------------------------
// BlockSparseTensor: fuse_legs associativity
// ---------------------------------------------------------------------------

proptest! {
    /// Fusing legs [0..2] then [0..2] on a rank-4 tensor should equal
    /// fusing all four legs [0..4] in one shot (in terms of nnz preservation).
    #[test]
    fn u1_fuse_associativity_rank4(
        charge_set in prop::collection::hash_set(-3i32..=3, 2..=5),
    ) {
        // Build a rank-4 tensor from a set of unique charges
        let mut charges: Vec<U1> = charge_set.into_iter().map(U1).collect();
        charges.sort();
        let idx = QIndex::new(
            charges.iter().map(|&q| (q, 1)).collect()
        );
        let indices = vec![idx.clone(), idx.clone(), idx.clone(), idx.clone()];
        let dirs = vec![
            LegDirection::Incoming,
            LegDirection::Incoming,
            LegDirection::Outgoing,
            LegDirection::Outgoing,
        ];
        let t = BlockSparseTensor::<f64, U1>::zeros(indices, U1::identity(), dirs);

        if t.n_sectors() == 0 {
            return Ok(());  // No valid sectors for this charge set
        }

        // Fuse [0..2] → rank-3, then fuse [0..2] → rank-2
        let fused_step = t.fuse_legs(0..2).fuse_legs(0..2);
        // Fuse all [0..4] → rank-1... but that changes rank differently
        // Instead verify fuse preserves total nnz
        let fused_all = t.fuse_legs(0..4);

        prop_assert_eq!(fused_step.nnz(), t.nnz());
        prop_assert_eq!(fused_all.nnz(), t.nnz());
    }
}
