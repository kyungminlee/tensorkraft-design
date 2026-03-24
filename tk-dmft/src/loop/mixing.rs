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

/// Mutable state for Broyden's first method (good Broyden).
///
/// Maintains a history of input vectors x_n and residual vectors F(x_n) - x_n,
/// and an approximate inverse Jacobian via the Sherman-Morrison rank-1 update.
///
/// The update rule is:
///   dx_n = x_n - x_{n-1}
///   df_n = F(x_n) - F(x_{n-1})
///   J^{-1}_{n+1} = J^{-1}_n + (dx_n - J^{-1}_n df_n) (dx_n^T J^{-1}_n) / (dx_n^T J^{-1}_n df_n)
///   x_{n+1} = x_n - alpha * J^{-1}_n * (F(x_n) - x_n)
///
/// For bath parameters, x is the concatenation of [epsilon_0..epsilon_{N-1}, v_0..v_{N-1}].
#[derive(Clone, Debug)]
pub struct BroydenState {
    /// Previous input vector (bath parameters concatenated).
    prev_input: Option<Vec<f64>>,
    /// Previous residual vector F(x) - x.
    prev_residual: Option<Vec<f64>>,
    /// Approximate inverse Jacobian (stored as dense matrix, row-major).
    /// Initialized as -alpha * I.
    inv_jacobian: Option<Vec<f64>>,
    /// Dimension of the parameter vector.
    dim: usize,
    /// Number of updates applied.
    n_updates: usize,
    /// Maximum history depth (controls when to reset).
    max_history: usize,
}

impl BroydenState {
    /// Create a new empty Broyden state.
    pub fn new(history_depth: usize) -> Self {
        Self {
            prev_input: None,
            prev_residual: None,
            inv_jacobian: None,
            dim: 0,
            n_updates: 0,
            max_history: history_depth,
        }
    }

    /// Apply Broyden's method to compute the next bath parameter vector.
    ///
    /// `x_current` is the current bath parameters (flattened),
    /// `f_current` is the proposed bath parameters from the solver (flattened).
    /// `alpha` is the initial mixing parameter.
    ///
    /// Returns the next bath parameter vector.
    pub fn update(&mut self, x_current: &[f64], f_current: &[f64], alpha: f64) -> Vec<f64> {
        let dim = x_current.len();
        assert_eq!(dim, f_current.len());

        // Residual: F(x) - x (difference between proposed and current)
        let residual: Vec<f64> = f_current.iter().zip(x_current).map(|(f, x)| f - x).collect();

        // First iteration or after reset: use linear mixing and initialize Jacobian
        if self.prev_input.is_none() || self.dim != dim || self.n_updates >= self.max_history {
            self.dim = dim;
            self.n_updates = 0;
            // Initialize J^{-1} = -alpha * I
            let mut j_inv = vec![0.0; dim * dim];
            for i in 0..dim {
                j_inv[i * dim + i] = -alpha;
            }
            self.inv_jacobian = Some(j_inv);
            self.prev_input = Some(x_current.to_vec());
            self.prev_residual = Some(residual.clone());

            // x_next = x - alpha * residual = (1-alpha)*x + alpha*f
            return x_current
                .iter()
                .zip(f_current)
                .map(|(x, f)| (1.0 - alpha) * x + alpha * f)
                .collect();
        }

        let prev_x = self.prev_input.as_ref().unwrap();
        let prev_r = self.prev_residual.as_ref().unwrap();
        let j_inv = self.inv_jacobian.as_mut().unwrap();

        // dx = x_current - x_prev
        let dx: Vec<f64> = x_current.iter().zip(prev_x).map(|(a, b)| a - b).collect();
        // df = residual - prev_residual
        let df: Vec<f64> = residual.iter().zip(prev_r).map(|(a, b)| a - b).collect();

        // j_inv_df = J^{-1} * df
        let j_inv_df: Vec<f64> = (0..dim)
            .map(|i| (0..dim).map(|j| j_inv[i * dim + j] * df[j]).sum::<f64>())
            .collect();

        // numerator = dx - j_inv_df
        let num: Vec<f64> = dx.iter().zip(&j_inv_df).map(|(a, b)| a - b).collect();

        // dx^T * j_inv (row vector)
        let dx_t_jinv: Vec<f64> = (0..dim)
            .map(|j| (0..dim).map(|i| dx[i] * j_inv[i * dim + j]).sum::<f64>())
            .collect();

        // denominator = dx^T * j_inv * df
        let denom: f64 = dx_t_jinv.iter().zip(&df).map(|(a, b)| a * b).sum();

        // Only update Jacobian if denominator is not too small
        if denom.abs() > 1e-15 {
            // Rank-1 update: J^{-1} += (num * dx^T * J^{-1}) / denom
            for i in 0..dim {
                for j in 0..dim {
                    j_inv[i * dim + j] += num[i] * dx_t_jinv[j] / denom;
                }
            }
        }

        // x_next = x_current - J^{-1} * residual
        let j_inv = self.inv_jacobian.as_ref().unwrap();
        let x_next: Vec<f64> = (0..dim)
            .map(|i| {
                let correction: f64 = (0..dim).map(|j| j_inv[i * dim + j] * residual[j]).sum();
                x_current[i] - correction
            })
            .collect();

        self.prev_input = Some(x_current.to_vec());
        self.prev_residual = Some(residual);
        self.n_updates += 1;

        x_next
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

    #[test]
    fn test_broyden_first_iteration_is_linear() {
        let mut state = BroydenState::new(5);
        let x = vec![1.0, 2.0, 3.0];
        let f = vec![1.5, 2.5, 3.5];
        let alpha = 0.3;
        let result = state.update(&x, &f, alpha);
        // First iteration: (1-alpha)*x + alpha*f
        for i in 0..3 {
            let expected = (1.0 - alpha) * x[i] + alpha * f[i];
            assert!(
                (result[i] - expected).abs() < 1e-12,
                "mismatch at {}: {} vs {}",
                i,
                result[i],
                expected,
            );
        }
    }

    #[test]
    fn test_broyden_second_iteration_differs() {
        let mut state = BroydenState::new(5);
        let alpha = 0.5;
        // First iteration
        let x1 = vec![1.0, 2.0];
        let f1 = vec![1.5, 2.5];
        let x2 = state.update(&x1, &f1, alpha);
        // Second iteration — should use Jacobian update, not just linear mixing
        let f2 = vec![x2[0] + 0.1, x2[1] + 0.1];
        let x3 = state.update(&x2, &f2, alpha);
        // Should differ from simple linear mixing
        let x3_linear: Vec<f64> = x2
            .iter()
            .zip(&f2)
            .map(|(x, f)| (1.0 - alpha) * x + alpha * f)
            .collect();
        let differs = x3.iter().zip(&x3_linear).any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(differs, "Broyden should differ from linear mixing after first iteration");
    }

    #[test]
    fn test_broyden_resets_after_history_depth() {
        let mut state = BroydenState::new(2);
        let alpha = 0.5;
        // Fill history
        let mut x = vec![1.0, 2.0];
        for _ in 0..3 {
            let f: Vec<f64> = x.iter().map(|v| v + 0.1).collect();
            x = state.update(&x, &f, alpha);
        }
        // After exceeding history_depth, should reset
        assert!(state.n_updates <= 2);
    }
}
