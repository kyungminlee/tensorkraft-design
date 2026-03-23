//! Lattice abstractions for defining geometry and connectivity.

mod chain;
mod square;
mod triangular;
mod bethe;
mod star;

pub use chain::Chain;
pub use square::{Square, snake_path};
pub use triangular::Triangular;
pub use bethe::BetheLattice;
pub use star::StarGeometry;

/// Trait for lattice geometries used by the Hamiltonian DSL.
///
/// Object-safe: all methods return owned data or slices.
pub trait Lattice: std::fmt::Debug + Send + Sync + LatticeClone {
    /// Total number of lattice sites.
    fn n_sites(&self) -> usize;

    /// Nearest-neighbour bonds as (i, j) pairs with i < j.
    fn bonds(&self) -> &[(usize, usize)];

    /// Site ordering for DMRG sweeps (identity for 1D, snake-path for 2D).
    fn dmrg_ordering(&self) -> Vec<usize>;

    /// Local Hilbert space dimension, if uniform across all sites.
    fn local_dim(&self) -> Option<usize> {
        None
    }
}

/// Helper trait for cloning boxed Lattice objects.
pub trait LatticeClone {
    fn clone_box(&self) -> Box<dyn Lattice>;
}

impl<T: Lattice + Clone + 'static> LatticeClone for T {
    fn clone_box(&self) -> Box<dyn Lattice> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Lattice> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
