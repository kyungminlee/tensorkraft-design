//! Bath-update mixing schemes for DMFT self-consistency convergence.

/// Bath-update mixing scheme for DMFT self-consistency convergence.
///
/// The self-consistency condition requires bath_new = f(G_imp[bath_old]).
/// Direct substitution (linear mixing alpha = 1.0) is often unstable for
/// strongly correlated phases.
#[derive(Clone, Debug)]
pub enum MixingScheme {
    /// Linear mixing: bath_new = (1 - alpha) * bath_old + alpha * bath_from_spectral.
    /// Default alpha: 0.3.
    Linear { alpha: f64 },
    /// Broyden's first method (good Broyden) for quasi-Newton acceleration.
    /// Default alpha: 0.5, default history_depth: 5.
    Broyden { alpha: f64, history_depth: usize },
}

impl Default for MixingScheme {
    fn default() -> Self {
        MixingScheme::Broyden {
            alpha: 0.5,
            history_depth: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_mixing() {
        let mix = MixingScheme::default();
        match mix {
            MixingScheme::Broyden {
                alpha,
                history_depth,
            } => {
                assert!((alpha - 0.5).abs() < 1e-12);
                assert_eq!(history_depth, 5);
            }
            _ => panic!("expected Broyden"),
        }
    }
}
