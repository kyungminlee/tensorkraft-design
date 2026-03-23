//! Anderson Impurity Model representation for DMFT.
//!
//! Contains the bath parameters, the full AIM struct, and the
//! bath discretization algorithm (Lanczos tridiagonalization).

pub mod bath;
pub mod discretize;
pub mod hamiltonian;

pub use bath::BathParameters;
pub use discretize::BathDiscretizationConfig;
pub use hamiltonian::build_aim_chain_hamiltonian;

use tk_core::Scalar;

use crate::error::DmftResult;
use crate::impurity::bath::BathParameters as Bath;
use crate::impurity::discretize::BathDiscretizationConfig as DiscConfig;

/// Full Anderson Impurity Model: impurity + discretized bath.
///
/// Holds the current bath parameters and the model's physical parameters
/// (interaction U, impurity level epsilon_imp, inverse temperature beta). Provides
/// methods to discretize the bath and to update parameters after a DMFT
/// iteration.
///
/// # Type Parameters
/// - `T`: scalar type (`f64` for real calculations, `Complex<f64>` for
///         complex hybridization functions)
#[derive(Clone, Debug)]
pub struct AndersonImpurityModel<T: Scalar> {
    /// On-site Coulomb interaction at the impurity.
    pub u: T::Real,
    /// Impurity site energy (relative to the chemical potential).
    pub epsilon_imp: T::Real,
    /// Inverse temperature. `None` for T=0 calculations.
    pub beta: Option<T::Real>,
    /// Current discretized bath parameters.
    pub bath: Bath<T>,
    /// Physical dimension of the impurity site (4 for single-orbital spin-1/2:
    /// |0>, |up>, |down>, |up,down>).
    pub impurity_local_dim: usize,
    /// Physical dimension of each bath site.
    pub bath_local_dim: usize,
}

impl<T: Scalar> AndersonImpurityModel<T>
where
    T::Real: Into<f64> + From<f64>,
{
    /// Construct a new AIM with `n_bath` bath sites initialized uniformly.
    ///
    /// # Parameters
    /// - `u`: Hubbard interaction
    /// - `epsilon_imp`: impurity level energy
    /// - `n_bath`: number of bath sites
    /// - `bandwidth`: initial bath bandwidth for uniform initialization
    /// - `v0`: initial uniform hybridization
    pub fn new(
        u: T::Real,
        epsilon_imp: T::Real,
        n_bath: usize,
        bandwidth: T::Real,
        v0: T,
    ) -> Self {
        Self {
            u,
            epsilon_imp,
            beta: None,
            bath: Bath::uniform(n_bath, bandwidth, v0),
            impurity_local_dim: 4,
            bath_local_dim: 4,
        }
    }

    /// Perform bath discretization: project the target hybridization function
    /// `delta_target` onto `n_bath` discrete bath parameters via Lanczos
    /// tridiagonalization.
    ///
    /// # Parameters
    /// - `delta_target`: target hybridization Delta(omega) sampled on `omega`
    /// - `omega`: real-frequency grid (uniform spacing assumed)
    /// - `config`: discretization configuration
    ///
    /// # Returns
    /// Updated `BathParameters`. Does NOT mutate `self`; caller decides
    /// whether to commit via `update_bath`.
    ///
    /// # Errors
    /// Returns `DmftError::BathDiscretizationFailed` if Lanczos tridiagonalization
    /// does not converge within `config.max_lanczos_steps`.
    pub fn discretize(
        &self,
        delta_target: &[T],
        omega: &[T::Real],
        config: &DiscConfig,
    ) -> DmftResult<Bath<T>> {
        discretize::lanczos_tridiagonalize(
            delta_target,
            omega,
            self.bath.n_bath,
            config,
        )
    }

    /// Update the bath parameters to `new_bath`.
    /// Called after each DMFT iteration's mixing step.
    pub fn update_bath(&mut self, new_bath: Bath<T>) {
        self.bath = new_bath;
    }

    /// Total number of sites in the chain (1 impurity + n_bath bath sites).
    pub fn n_sites(&self) -> usize {
        1 + self.bath.n_bath
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aim_construction() {
        let aim: AndersonImpurityModel<f64> =
            AndersonImpurityModel::new(4.0, -2.0, 6, 10.0, 1.0);
        assert_eq!(aim.n_sites(), 7);
        assert_eq!(aim.bath.n_bath, 6);
        assert_eq!(aim.impurity_local_dim, 4);
        assert_eq!(aim.u, 4.0);
        assert_eq!(aim.epsilon_imp, -2.0);
    }

    #[test]
    fn test_aim_update_bath() {
        let mut aim: AndersonImpurityModel<f64> =
            AndersonImpurityModel::new(4.0, -2.0, 6, 10.0, 1.0);
        let new_bath = BathParameters::uniform(4, 8.0, 0.5);
        aim.update_bath(new_bath);
        assert_eq!(aim.bath.n_bath, 4);
        assert_eq!(aim.n_sites(), 5);
    }
}
