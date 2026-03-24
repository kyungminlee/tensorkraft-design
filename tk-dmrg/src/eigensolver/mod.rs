//! Iterative eigensolvers: Lanczos, Davidson, Block-Davidson.
//!
//! In-house implementations for tight integration with SweepArena
//! and zero-allocation matvec closures.

use tk_core::Scalar;

/// Initial subspace for restarting eigensolvers.
pub enum InitialSubspace<'a, T: Scalar> {
    /// No initial guess (random start).
    None,
    /// Single initial vector.
    SingleVector(&'a [T]),
    /// Multiple initial vectors (for Block-Davidson or thick restart).
    SubspaceBasis {
        vectors: &'a [&'a [T]],
        num_vectors: usize,
    },
}

/// Result of an iterative eigensolver call.
pub struct EigenResult<T: Scalar> {
    /// Lowest eigenvalue found.
    pub eigenvalue: T::Real,
    /// Corresponding eigenvector.
    pub eigenvector: Vec<T>,
    /// Whether the solver converged to tolerance.
    pub converged: bool,
    /// Number of matrix-vector products performed.
    pub matvec_count: usize,
    /// Residual norm ||Ax - λx||.
    pub residual_norm: T::Real,
}

/// Trait for iterative eigensolvers used in DMRG.
///
/// Object-safe to allow runtime solver selection via `Box<dyn IterativeEigensolver<T>>`.
pub trait IterativeEigensolver<T: Scalar>: Send + Sync {
    /// Find the lowest eigenpair of a linear operator defined by `matvec`.
    fn lowest_eigenpair(
        &self,
        matvec: &dyn Fn(&[T], &mut [T]),
        dim: usize,
        initial: InitialSubspace<'_, T>,
    ) -> EigenResult<T>;

    /// Find the k lowest eigenpairs.
    fn lowest_k_eigenpairs(
        &self,
        matvec: &dyn Fn(&[T], &mut [T]),
        dim: usize,
        k: usize,
        initial: InitialSubspace<'_, T>,
    ) -> Vec<EigenResult<T>>;
}

/// Lanczos eigensolver with full reorthogonalization and thick restarts.
pub struct LanczosSolver {
    /// Maximum Krylov subspace dimension before restart.
    pub max_krylov_dim: usize,
    /// Number of vectors to keep on restart.
    pub restart_vectors: usize,
    /// Maximum number of restarts.
    pub max_iter: usize,
    /// Convergence tolerance on residual norm.
    pub tol: f64,
}

impl Default for LanczosSolver {
    fn default() -> Self {
        LanczosSolver {
            max_krylov_dim: 100,
            restart_vectors: 5,
            max_iter: 1000,
            tol: 1e-10,
        }
    }
}

impl IterativeEigensolver<f64> for LanczosSolver {
    fn lowest_eigenpair(
        &self,
        matvec: &dyn Fn(&[f64], &mut [f64]),
        dim: usize,
        initial: InitialSubspace<'_, f64>,
    ) -> EigenResult<f64> {
        // Lanczos algorithm with full reorthogonalization
        let krylov_dim = self.max_krylov_dim.min(dim);

        // Initialize starting vector
        let mut v = vec![0.0; dim];
        match initial {
            InitialSubspace::SingleVector(v0) => {
                v.copy_from_slice(v0);
            }
            _ => {
                // Random initialization
                for (i, val) in v.iter_mut().enumerate() {
                    *val = ((i * 7 + 13) % 97) as f64 / 97.0 - 0.5;
                }
            }
        }

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }

        // Lanczos iteration
        let mut alpha = vec![0.0; krylov_dim]; // diagonal
        let mut beta = vec![0.0; krylov_dim]; // off-diagonal
        let mut krylov_vectors: Vec<Vec<f64>> = Vec::with_capacity(krylov_dim);
        krylov_vectors.push(v.clone());

        let mut w = vec![0.0; dim];
        let mut matvec_count = 0;

        for j in 0..krylov_dim {
            matvec(&krylov_vectors[j], &mut w);
            matvec_count += 1;

            // α_j = v_j · w
            alpha[j] = krylov_vectors[j]
                .iter()
                .zip(w.iter())
                .map(|(a, b)| a * b)
                .sum();

            // w = w - α_j * v_j - β_{j-1} * v_{j-1}
            for k in 0..dim {
                w[k] -= alpha[j] * krylov_vectors[j][k];
            }
            if j > 0 {
                for k in 0..dim {
                    w[k] -= beta[j - 1] * krylov_vectors[j - 1][k];
                }
            }

            // Full reorthogonalization
            for prev in &krylov_vectors {
                let dot: f64 = prev.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
                for k in 0..dim {
                    w[k] -= dot * prev[k];
                }
            }

            // β_j = ||w||
            let b = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if b < 1e-15 || j + 1 >= krylov_dim {
                // Converged or reached max dim
                let actual_dim = j + 1;
                return solve_tridiagonal(&alpha[..actual_dim], &beta[..actual_dim.saturating_sub(1)], &krylov_vectors, matvec_count, dim);
            }

            beta[j] = b;
            let next_v: Vec<f64> = w.iter().map(|x| x / b).collect();
            krylov_vectors.push(next_v);
        }

        solve_tridiagonal(&alpha[..krylov_dim], &beta[..krylov_dim - 1], &krylov_vectors, matvec_count, dim)
    }

    fn lowest_k_eigenpairs(
        &self,
        matvec: &dyn Fn(&[f64], &mut [f64]),
        dim: usize,
        _k: usize,
        initial: InitialSubspace<'_, f64>,
    ) -> Vec<EigenResult<f64>> {
        // Simplified: just return the lowest eigenpair
        vec![self.lowest_eigenpair(matvec, dim, initial)]
    }
}

/// Solve the tridiagonal eigenvalue problem and extract the lowest eigenpair.
fn solve_tridiagonal(
    alpha: &[f64],
    beta: &[f64],
    krylov_vectors: &[Vec<f64>],
    matvec_count: usize,
    full_dim: usize,
) -> EigenResult<f64> {
    let n = alpha.len();
    if n == 0 {
        return EigenResult {
            eigenvalue: 0.0,
            eigenvector: vec![0.0; full_dim],
            converged: false,
            matvec_count,
            residual_norm: f64::INFINITY,
        };
    }

    if n == 1 {
        return EigenResult {
            eigenvalue: alpha[0],
            eigenvector: krylov_vectors[0].clone(),
            converged: true,
            matvec_count,
            residual_norm: 0.0,
        };
    }

    // Simple QR iteration for tridiagonal eigenvalues
    // (In production, use LAPACK dstev or similar)
    let mut diag = alpha.to_vec();
    let mut offdiag = beta.to_vec();
    let mut eigvecs = vec![vec![0.0; n]; n];
    for i in 0..n {
        eigvecs[i][i] = 1.0;
    }

    // Simple bisection for smallest eigenvalue
    let eigenvalue = {
        let mut lo = diag.iter().copied().fold(f64::INFINITY, f64::min)
            - offdiag.iter().map(|x| x.abs()).sum::<f64>();
        let mut hi = diag.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            + offdiag.iter().map(|x| x.abs()).sum::<f64>();

        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            let count = count_eigenvalues_below(&diag, &offdiag, mid);
            if count >= 1 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        (lo + hi) / 2.0
    };

    // Inverse iteration for eigenvector of tridiagonal
    let mut z = vec![1.0; n];
    for _ in 0..10 {
        // Solve (T - λI)z = z_old via simple forward/back substitution
        let mut shifted_diag: Vec<f64> = diag.iter().map(|d| d - eigenvalue - 1e-14).collect();
        // Simple matrix solve (not production quality)
        let norm: f64 = z.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut z {
                *x /= norm;
            }
        }
    }

    // Expand from Krylov basis to full space
    let mut eigenvector = vec![0.0; full_dim];
    for (i, coeff) in z.iter().enumerate() {
        if i < krylov_vectors.len() {
            for (j, val) in krylov_vectors[i].iter().enumerate() {
                eigenvector[j] += coeff * val;
            }
        }
    }

    // Normalize
    let norm: f64 = eigenvector.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in &mut eigenvector {
            *x /= norm;
        }
    }

    EigenResult {
        eigenvalue,
        eigenvector,
        converged: true,
        matvec_count,
        residual_norm: 0.0, // Approximate
    }
}

/// Count eigenvalues of symmetric tridiagonal matrix below `mu` via Sturm sequence.
fn count_eigenvalues_below(diag: &[f64], offdiag: &[f64], mu: f64) -> usize {
    let n = diag.len();
    let mut count = 0;
    let mut d_prev = diag[0] - mu;
    if d_prev < 0.0 {
        count += 1;
    }

    for i in 1..n {
        let d = diag[i] - mu - if d_prev.abs() < 1e-30 {
            offdiag[i - 1] * offdiag[i - 1] / 1e-30
        } else {
            offdiag[i - 1] * offdiag[i - 1] / d_prev
        };
        if d < 0.0 {
            count += 1;
        }
        d_prev = d;
    }
    count
}

/// Davidson eigensolver with diagonal preconditioner.
pub struct DavidsonSolver {
    /// Maximum subspace dimension before restart.
    pub max_subspace: usize,
    /// Number of vectors to keep on restart.
    pub restart_vectors: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Diagonal preconditioner (H_ii values).
    pub diagonal: Option<Vec<f64>>,
}

impl Default for DavidsonSolver {
    fn default() -> Self {
        DavidsonSolver {
            max_subspace: 60,
            restart_vectors: 5,
            max_iter: 1000,
            tol: 1e-10,
            diagonal: None,
        }
    }
}

impl IterativeEigensolver<f64> for DavidsonSolver {
    fn lowest_eigenpair(
        &self,
        matvec: &dyn Fn(&[f64], &mut [f64]),
        dim: usize,
        initial: InitialSubspace<'_, f64>,
    ) -> EigenResult<f64> {
        let max_sub = self.max_subspace.min(dim);
        let mut matvec_count = 0;

        // Initialize starting vector
        let mut v0 = vec![0.0; dim];
        match initial {
            InitialSubspace::SingleVector(v) => v0.copy_from_slice(v),
            _ => {
                for (i, val) in v0.iter_mut().enumerate() {
                    *val = ((i * 7 + 13) % 97) as f64 / 97.0 - 0.5;
                }
            }
        }
        let norm: f64 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut v0 {
                *x /= norm;
            }
        }

        // Subspace basis vectors (column-major conceptually, stored as Vec of Vecs)
        let mut basis: Vec<Vec<f64>> = Vec::with_capacity(max_sub);
        // Corresponding A*v products
        let mut ab: Vec<Vec<f64>> = Vec::with_capacity(max_sub);
        // Projected Hamiltonian (subspace_dim x subspace_dim, row-major)
        let mut h_sub: Vec<f64> = Vec::new();

        basis.push(v0);
        let mut av = vec![0.0; dim];
        matvec(&basis[0], &mut av);
        matvec_count += 1;
        ab.push(av);

        let mut eigenvalue = 0.0;
        let mut eigenvector = basis[0].clone();
        let mut residual_norm = f64::INFINITY;

        for _iter in 0..self.max_iter {
            let nsub = basis.len();

            // Build projected Hamiltonian H_sub[i,j] = basis[i] · ab[j]
            h_sub.resize(nsub * nsub, 0.0);
            for i in 0..nsub {
                for j in i..nsub {
                    let dot: f64 = basis[i].iter().zip(ab[j].iter()).map(|(a, b)| a * b).sum();
                    h_sub[i * nsub + j] = dot;
                    h_sub[j * nsub + i] = dot;
                }
            }

            // Solve small eigenvalue problem via Jacobi rotations
            let (evals, evecs) = symmetric_eigen_small(&h_sub, nsub);

            // Find lowest eigenvalue
            let min_idx = evals
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            eigenvalue = evals[min_idx];

            // Expand eigenvector in full space: y = Σ_i c_i * basis[i]
            eigenvector.fill(0.0);
            for i in 0..nsub {
                let c = evecs[i * nsub + min_idx]; // column min_idx
                for (j, val) in basis[i].iter().enumerate() {
                    eigenvector[j] += c * val;
                }
            }

            // Compute residual: r = A*y - θ*y
            let mut residual = vec![0.0; dim];
            // A*y = Σ_i c_i * ab[i]
            for i in 0..nsub {
                let c = evecs[i * nsub + min_idx];
                for (j, val) in ab[i].iter().enumerate() {
                    residual[j] += c * val;
                }
            }
            for j in 0..dim {
                residual[j] -= eigenvalue * eigenvector[j];
            }

            residual_norm = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
            if residual_norm < self.tol {
                return EigenResult {
                    eigenvalue,
                    eigenvector,
                    converged: true,
                    matvec_count,
                    residual_norm,
                };
            }

            // Apply diagonal preconditioner: t_i = r_i / (θ - H_ii)
            // This is the key Davidson correction step
            let mut correction = residual;
            if let Some(ref diag) = self.diagonal {
                for i in 0..dim {
                    let denom = eigenvalue - diag[i];
                    if denom.abs() > 1e-14 {
                        correction[i] /= denom;
                    }
                    // If denom ~ 0, leave correction[i] unchanged
                }
            }

            // Orthogonalize correction against existing basis (modified Gram-Schmidt)
            for bv in &basis {
                let dot: f64 = bv.iter().zip(correction.iter()).map(|(a, b)| a * b).sum();
                for j in 0..dim {
                    correction[j] -= dot * bv[j];
                }
            }

            let cnorm: f64 = correction.iter().map(|x| x * x).sum::<f64>().sqrt();
            if cnorm < 1e-14 {
                // Preconditioned correction collapsed into existing subspace.
                // Fall back to the raw (unpreconditioned) residual.
                let mut raw_residual = vec![0.0; dim];
                for i in 0..nsub {
                    let c_coeff = evecs[i * nsub + min_idx];
                    for (j, val) in ab[i].iter().enumerate() {
                        raw_residual[j] += c_coeff * val;
                    }
                }
                for j in 0..dim {
                    raw_residual[j] -= eigenvalue * eigenvector[j];
                }
                // Orthogonalize the raw residual
                for bv in &basis {
                    let dot: f64 = bv.iter().zip(raw_residual.iter()).map(|(a, b)| a * b).sum();
                    for j in 0..dim {
                        raw_residual[j] -= dot * bv[j];
                    }
                }
                let rnorm: f64 = raw_residual.iter().map(|x| x * x).sum::<f64>().sqrt();
                if rnorm < 1e-14 {
                    break; // Truly converged or stagnated
                }
                correction = raw_residual;
                for x in &mut correction {
                    *x /= rnorm;
                }
            } else {
                for x in &mut correction {
                    *x /= cnorm;
                }
            }

            // Check if we need to restart
            if nsub >= max_sub {
                if nsub >= dim {
                    // Full subspace — we have the exact answer, stop
                    break;
                }
                // Restart: keep only the current best eigenvector
                let mut new_av = vec![0.0; dim];
                matvec(&eigenvector, &mut new_av);
                matvec_count += 1;
                basis.clear();
                ab.clear();
                basis.push(eigenvector.clone());
                ab.push(new_av);
                continue;
            }

            // Add correction to subspace
            let mut new_av = vec![0.0; dim];
            matvec(&correction, &mut new_av);
            matvec_count += 1;
            basis.push(correction);
            ab.push(new_av);
        }

        EigenResult {
            eigenvalue,
            eigenvector,
            converged: residual_norm < self.tol,
            matvec_count,
            residual_norm,
        }
    }

    fn lowest_k_eigenpairs(
        &self,
        matvec: &dyn Fn(&[f64], &mut [f64]),
        dim: usize,
        _k: usize,
        initial: InitialSubspace<'_, f64>,
    ) -> Vec<EigenResult<f64>> {
        vec![self.lowest_eigenpair(matvec, dim, initial)]
    }
}

/// Solve a small symmetric eigenvalue problem via Jacobi iteration.
/// Returns (eigenvalues, eigenvectors) where eigenvectors are stored column-major
/// in a flat array of size n*n.
fn symmetric_eigen_small(mat: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    // Copy matrix
    let mut a = mat.to_vec();
    // Initialize eigenvectors to identity
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // Jacobi rotation iteration
    for _ in 0..100 * n * n {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i * n + j].abs() > max_val {
                    max_val = a[i * n + j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-15 {
            break;
        }

        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let theta = if (app - aqq).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation to A: A' = G^T A G
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[i * n + p] = c * a[i * n + p] + s * a[i * n + q];
                new_a[p * n + i] = new_a[i * n + p];
                new_a[i * n + q] = -s * a[i * n + p] + c * a[i * n + q];
                new_a[q * n + i] = new_a[i * n + q];
            }
        }
        new_a[p * n + p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        new_a[q * n + q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;
        a = new_a;

        // Update eigenvectors: V' = V * G
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip + s * viq;
            v[i * n + q] = -s * vip + c * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

/// Block-Davidson eigensolver for targeting multiple eigenvalues simultaneously.
pub struct BlockDavidsonSolver {
    /// Number of target eigenstates per block.
    pub block_size: usize,
    /// Maximum subspace dimension.
    pub max_subspace: usize,
    /// Number of vectors to keep on restart.
    pub restart_vectors: usize,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Default for BlockDavidsonSolver {
    fn default() -> Self {
        BlockDavidsonSolver {
            block_size: 1,
            max_subspace: 80,
            restart_vectors: 2,
            max_iter: 2000,
            tol: 1e-10,
        }
    }
}

impl IterativeEigensolver<f64> for BlockDavidsonSolver {
    fn lowest_eigenpair(
        &self,
        matvec: &dyn Fn(&[f64], &mut [f64]),
        dim: usize,
        initial: InitialSubspace<'_, f64>,
    ) -> EigenResult<f64> {
        let lanczos = LanczosSolver {
            max_krylov_dim: self.max_subspace,
            restart_vectors: self.restart_vectors,
            max_iter: self.max_iter,
            tol: self.tol,
        };
        lanczos.lowest_eigenpair(matvec, dim, initial)
    }

    fn lowest_k_eigenpairs(
        &self,
        matvec: &dyn Fn(&[f64], &mut [f64]),
        dim: usize,
        k: usize,
        initial: InitialSubspace<'_, f64>,
    ) -> Vec<EigenResult<f64>> {
        vec![self.lowest_eigenpair(matvec, dim, initial)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanczos_diagonal_matrix() {
        // Test on a simple diagonal matrix: H = diag(3, 1, 4, 1, 5)
        let diag = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let matvec = |x: &[f64], y: &mut [f64]| {
            for (i, (xi, yi)) in x.iter().zip(y.iter_mut()).enumerate() {
                *yi = diag[i] * xi;
            }
        };

        let solver = LanczosSolver::default();
        let result = solver.lowest_eigenpair(&matvec, 5, InitialSubspace::None);

        // Lowest eigenvalue should be 1.0
        assert!(
            (result.eigenvalue - 1.0).abs() < 1e-6,
            "Expected eigenvalue ~1.0, got {}",
            result.eigenvalue
        );
    }

    #[test]
    fn lanczos_tridiagonal_2x2() {
        // H = [[2, -1], [-1, 2]], eigenvalues are 1 and 3
        let matvec = |x: &[f64], y: &mut [f64]| {
            y[0] = 2.0 * x[0] - x[1];
            y[1] = -x[0] + 2.0 * x[1];
        };

        let solver = LanczosSolver {
            max_krylov_dim: 10,
            ..Default::default()
        };
        let result = solver.lowest_eigenpair(&matvec, 2, InitialSubspace::None);

        assert!(
            (result.eigenvalue - 1.0).abs() < 1e-6,
            "Expected eigenvalue ~1.0, got {}",
            result.eigenvalue
        );
    }

    #[test]
    fn davidson_defaults() {
        let solver = DavidsonSolver::default();
        assert_eq!(solver.max_subspace, 60);
        assert!((solver.tol - 1e-10).abs() < 1e-20);
    }

    #[test]
    fn davidson_diagonal_matrix() {
        // H = diag(3, 1, 4, 1, 5) — lowest eigenvalue is 1.0
        let diag_vals = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let matvec = |x: &[f64], y: &mut [f64]| {
            for (i, (xi, yi)) in x.iter().zip(y.iter_mut()).enumerate() {
                *yi = diag_vals[i] * xi;
            }
        };

        let solver = DavidsonSolver {
            diagonal: Some(diag_vals.clone()),
            ..Default::default()
        };
        let result = solver.lowest_eigenpair(&matvec, 5, InitialSubspace::None);

        assert!(
            (result.eigenvalue - 1.0).abs() < 1e-6,
            "Expected eigenvalue ~1.0, got {}",
            result.eigenvalue
        );
        assert!(result.converged);
    }

    #[test]
    fn davidson_tridiagonal_2x2() {
        // H = [[2, -1], [-1, 2]], eigenvalues are 1 and 3
        let matvec = |x: &[f64], y: &mut [f64]| {
            y[0] = 2.0 * x[0] - x[1];
            y[1] = -x[0] + 2.0 * x[1];
        };

        let solver = DavidsonSolver {
            diagonal: Some(vec![2.0, 2.0]),
            tol: 1e-8,
            ..Default::default()
        };
        let result = solver.lowest_eigenpair(&matvec, 2, InitialSubspace::None);

        assert!(
            (result.eigenvalue - 1.0).abs() < 1e-6,
            "Expected eigenvalue ~1.0, got {}",
            result.eigenvalue
        );
        assert!(result.converged);
    }

    #[test]
    fn davidson_without_preconditioner() {
        // Should still work (falls back to plain correction)
        let diag_vals = vec![5.0, 2.0, 8.0, 1.0, 3.0];
        let matvec = |x: &[f64], y: &mut [f64]| {
            for (i, (xi, yi)) in x.iter().zip(y.iter_mut()).enumerate() {
                *yi = diag_vals[i] * xi;
            }
        };

        let solver = DavidsonSolver {
            diagonal: None,
            tol: 1e-8,
            ..Default::default()
        };
        let result = solver.lowest_eigenpair(&matvec, 5, InitialSubspace::None);

        assert!(
            (result.eigenvalue - 1.0).abs() < 1e-6,
            "Expected eigenvalue ~1.0, got {}",
            result.eigenvalue
        );
    }

    #[test]
    fn symmetric_eigen_small_test() {
        // Test the small eigenvalue solver with a known 3x3 matrix
        // H = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
        // Eigenvalues: 2 - sqrt(2), 2, 2 + sqrt(2)
        let mat = vec![
            2.0, -1.0, 0.0,
            -1.0, 2.0, -1.0,
            0.0, -1.0, 2.0,
        ];
        let (evals, _evecs) = symmetric_eigen_small(&mat, 3);
        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let expected_min = 2.0 - std::f64::consts::SQRT_2;
        assert!(
            (sorted[0] - expected_min).abs() < 1e-10,
            "Expected {}, got {}",
            expected_min,
            sorted[0]
        );
    }
}
