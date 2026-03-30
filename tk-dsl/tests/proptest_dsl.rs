//! Property-based integration tests for tk-dsl.

use proptest::prelude::*;
use tk_dsl::lattice::Lattice;
use tk_dsl::{op, BosonOp, Chain, OpSum, SpinOp, Square};

/// Build an OpSum with N terms added in a random permutation order;
/// verify that n_terms() always equals the number of terms pushed.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn opsum_term_accumulation_order_independent(
        n in 2usize..=6,
        seed in any::<u64>(),
    ) {
        // Build terms in a deterministic-but-randomised order based on seed.
        // We create `n` terms each on a different site, then shuffle via a
        // simple Fisher-Yates driven by the seed.
        let mut indices: Vec<usize> = (0..n).collect();
        // Cheap deterministic shuffle using the seed.
        let mut s = seed;
        for i in (1..indices.len()).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (s as usize) % (i + 1);
            indices.swap(i, j);
        }

        let mut opsum = OpSum::<f64>::new();
        for &site in &indices {
            opsum += 1.0 * op(SpinOp::Sz, site);
        }
        prop_assert_eq!(opsum.n_terms(), n);
    }

    /// Chain(n) must have exactly n-1 nearest-neighbour bonds.
    #[test]
    fn chain_lattice_bond_count(n in 2usize..=20) {
        let chain = Chain::new(n, 2);
        prop_assert_eq!(chain.bonds().len(), n - 1);
    }

    /// Square(rows, cols) must have the correct number of bonds:
    ///   horizontal = rows * (cols - 1)
    ///   vertical   = (rows - 1) * cols
    #[test]
    fn square_lattice_bond_count(rows in 2usize..=5, cols in 2usize..=5) {
        let sq = Square::new(cols, rows, 2);
        let expected_horizontal = rows * (cols - 1);
        let expected_vertical = (rows - 1) * cols;
        prop_assert_eq!(sq.bonds().len(), expected_horizontal + expected_vertical);
    }

    /// adjoint(adjoint(op)) == op for every SpinOp variant.
    #[test]
    fn spin_operator_adjoint_involution(variant in 0u8..6) {
        let spin_op = match variant {
            0 => SpinOp::SPlus,
            1 => SpinOp::SMinus,
            2 => SpinOp::Sz,
            3 => SpinOp::Sx,
            4 => SpinOp::Sy,
            5 => SpinOp::Identity,
            _ => unreachable!(),
        };
        let double_adjoint = spin_op.adjoint().adjoint();
        prop_assert_eq!(double_adjoint, spin_op);
    }

    /// BosonOp::NPairInteraction matrix is diagonal with entries n*(n-1).
    #[test]
    fn boson_op_n_pair_diagonal(n_max in 1usize..=6) {
        let mat = BosonOp::NPairInteraction.matrix::<f64>(n_max);
        let d = n_max + 1;
        for r in 0..d {
            for c in 0..d {
                let val = mat[r * d + c];
                if r == c {
                    let expected = (r * r.saturating_sub(1)) as f64;
                    prop_assert!(
                        (val - expected).abs() < 1e-12,
                        "diagonal entry [{r},{c}] = {val}, expected {expected}"
                    );
                } else {
                    prop_assert!(
                        val.abs() < 1e-12,
                        "off-diagonal entry [{r},{c}] = {val}, expected 0"
                    );
                }
            }
        }
    }
}
