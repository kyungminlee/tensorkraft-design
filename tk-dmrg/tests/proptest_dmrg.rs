//! Property-based tests for tk-dmrg using proptest.
//!
//! Tests: Lanczos eigenvalue accuracy, bond dimension schedule monotonicity,
//! sweep schedule coverage, truncation config bond dimension bounds.

use proptest::prelude::*;
use tk_dmrg::{
    BondDimensionSchedule, LanczosSolver, IterativeEigensolver, InitialSubspace,
    SweepSchedule, SweepDirection, TruncationConfig, truncate_svd,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Lanczos on random diagonal matrices: finds the minimum eigenvalue.
    /// Diagonal matrices are guaranteed to converge with Lanczos.
    #[test]
    fn lanczos_eigenvalue_accuracy(
        dim in 3..=20_usize,
        seed in 0..10000_u64,
    ) {
        // Generate random diagonal entries via LCG
        let mut state = seed.wrapping_add(42);
        let diag: Vec<f64> = (0..dim).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as i32 as f64) / (1 << 30) as f64
        }).collect();

        let true_min = diag.iter().cloned().fold(f64::INFINITY, f64::min);

        let matvec = |x: &[f64], y: &mut [f64]| {
            for (i, (xi, yi)) in x.iter().zip(y.iter_mut()).enumerate() {
                *yi = diag[i] * xi;
            }
        };

        let solver = LanczosSolver::default();
        let result = solver.lowest_eigenpair(&matvec, dim, InitialSubspace::None);

        prop_assert!(result.eigenvalue.is_finite(),
            "eigenvalue must be finite, got {}", result.eigenvalue);

        // Should find the minimum within tolerance
        prop_assert!(
            (result.eigenvalue - true_min).abs() < 1e-6,
            "eigenvalue {} differs from true min {} for dim={dim}, seed={seed}",
            result.eigenvalue, true_min,
        );
    }

    /// BondDimensionSchedule::warmup produces monotonically non-decreasing dims.
    #[test]
    fn bond_dim_schedule_monotone(
        d_init in 1..=50_usize,
        d_max in 50..=500_usize,
        n_warmup in 1..=20_usize,
    ) {
        let schedule = BondDimensionSchedule::warmup(d_init, d_max, n_warmup);

        let mut prev = schedule.bond_dim_at_sweep(0);
        for i in 1..=n_warmup {
            let curr = schedule.bond_dim_at_sweep(i);
            prop_assert!(curr >= prev,
                "bond dim decreased from {prev} to {curr} at sweep {i}");
            prev = curr;
        }

        // Final value should be d_max
        let final_dim = schedule.bond_dim_at_sweep(n_warmup);
        prop_assert_eq!(final_dim, d_max);
    }

    /// SweepSchedule::standard visits all sites in both directions.
    #[test]
    fn sweep_schedule_covers_all_sites(
        n_sites in 3..=50_usize,
    ) {
        let schedule = SweepSchedule::standard(n_sites);
        let steps: Vec<(usize, SweepDirection)> = schedule.iter().collect();

        // Collect sites visited in each direction
        let lr_sites: Vec<usize> = steps.iter()
            .filter(|(_, d)| *d == SweepDirection::LeftToRight)
            .map(|(s, _)| *s)
            .collect();
        let rl_sites: Vec<usize> = steps.iter()
            .filter(|(_, d)| *d == SweepDirection::RightToLeft)
            .map(|(s, _)| *s)
            .collect();

        // LR should cover 0..n_sites-1
        for site in 0..n_sites - 1 {
            prop_assert!(lr_sites.contains(&site),
                "LR sweep missing site {site} for n_sites={n_sites}");
        }

        // RL should cover 1..n_sites (reversed)
        for site in 1..n_sites {
            prop_assert!(rl_sites.contains(&site),
                "RL sweep missing site {site} for n_sites={n_sites}");
        }

        // LR sites should be in ascending order
        for i in 1..lr_sites.len() {
            prop_assert!(lr_sites[i] > lr_sites[i - 1],
                "LR sites not ascending at index {i}");
        }

        // RL sites should be in descending order
        for i in 1..rl_sites.len() {
            prop_assert!(rl_sites[i] < rl_sites[i - 1],
                "RL sites not descending at index {i}");
        }
    }

    /// TruncationConfig with max_bond_dim produces results within bounds.
    #[test]
    fn truncation_config_max_bond_respected(
        max_bond in 1..=5_usize,
        seed in 0..10000_u64,
    ) {
        use tk_linalg::DeviceFaer;

        let rows = 6_usize;
        let cols = 6_usize;
        let mut data = vec![0.0_f64; rows * cols];
        // Fill with a deterministic non-zero matrix
        let mut state = seed;
        for val in data.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *val = ((state >> 33) as i32 as f64) / (1 << 30) as f64;
        }

        let config = TruncationConfig {
            max_bond_dim: max_bond,
            svd_cutoff: 1e-15,
            min_bond_dim: 1,
        };
        let backend = DeviceFaer;
        let result = truncate_svd(&data, rows, cols, &config, &backend)
            .expect("truncate_svd should succeed");

        prop_assert!(result.bond_dim_new <= max_bond,
            "bond_dim_new {} exceeds max_bond_dim {}", result.bond_dim_new, max_bond);
        prop_assert!(result.bond_dim_new >= 1,
            "bond_dim_new {} less than min 1", result.bond_dim_new);
        prop_assert_eq!(result.u_cols, result.bond_dim_new);
        prop_assert_eq!(result.vt_rows, result.bond_dim_new);
        prop_assert_eq!(result.singular_values.len(), result.bond_dim_new);
    }
}
