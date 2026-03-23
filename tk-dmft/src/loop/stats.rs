//! DMFT per-iteration statistics accumulator.

/// Per-iteration DMRG statistics snapshot.
///
/// Separate from `tk_dmrg::DMRGStats` because that type doesn't
/// implement `Clone` or `Debug`. This stores the summary values
/// we care about for DMFT convergence monitoring.
#[derive(Clone, Debug, Default)]
pub struct DmrgIterationSummary {
    /// Ground-state energy from the DMRG sweep.
    pub energy: f64,
    /// Maximum truncation error across bonds.
    pub max_truncation_error: f64,
    /// Maximum bond dimension used.
    pub max_bond_dim: usize,
    /// Number of sweeps performed.
    pub n_sweeps: usize,
    /// Wall time in seconds.
    pub wall_time_secs: f64,
}

/// Statistics accumulated across DMFT self-consistency iterations.
#[derive(Clone, Debug, Default)]
pub struct DMFTStats {
    /// DMRG ground-state energies at each iteration.
    pub ground_state_energies: Vec<f64>,
    /// Relative hybridization distances per iteration.
    pub hybridization_distances: Vec<f64>,
    /// Spectral sum rule at each iteration (should be ~ 1.0).
    pub spectral_sum_rules: Vec<f64>,
    /// Fraction of negative spectral weight clamped per iteration.
    pub positivity_clamped_fractions: Vec<f64>,
    /// Whether Chebyshev was the primary solver at each iteration.
    pub chebyshev_was_primary: Vec<bool>,
    /// Wall-clock seconds per iteration.
    pub iteration_times_secs: Vec<f64>,
    /// Per-iteration DMRG summary statistics.
    pub dmrg_summaries: Vec<DmrgIterationSummary>,
}

impl DMFTStats {
    /// Number of completed iterations.
    pub fn n_iterations(&self) -> usize {
        self.ground_state_energies.len()
    }

    /// Record one iteration's results.
    pub fn push_iteration(
        &mut self,
        energy: f64,
        hyb_distance: f64,
        sum_rule: f64,
        pos_clamped: f64,
        cheb_primary: bool,
        wall_time: f64,
        dmrg_summary: DmrgIterationSummary,
    ) {
        self.ground_state_energies.push(energy);
        self.hybridization_distances.push(hyb_distance);
        self.spectral_sum_rules.push(sum_rule);
        self.positivity_clamped_fractions.push(pos_clamped);
        self.chebyshev_was_primary.push(cheb_primary);
        self.iteration_times_secs.push(wall_time);
        self.dmrg_summaries.push(dmrg_summary);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_default() {
        let stats = DMFTStats::default();
        assert_eq!(stats.n_iterations(), 0);
    }

    #[test]
    fn test_stats_push() {
        let mut stats = DMFTStats::default();
        stats.push_iteration(
            -5.0, 0.01, 1.0, 0.0, false, 10.5,
            DmrgIterationSummary::default(),
        );
        assert_eq!(stats.n_iterations(), 1);
        assert!((stats.ground_state_energies[0] - (-5.0)).abs() < 1e-12);
    }
}
