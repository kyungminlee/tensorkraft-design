//! 1D chain lattice.

use super::Lattice;

/// A linear chain of `n` sites with nearest-neighbour bonds.
#[derive(Clone, Debug)]
pub struct Chain {
    pub n: usize,
    pub d: usize,
    bonds_cache: Vec<(usize, usize)>,
}

impl Chain {
    pub fn new(n: usize, d: usize) -> Self {
        assert!(n >= 2, "Chain requires at least 2 sites");
        let bonds_cache: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        Chain { n, d, bonds_cache }
    }
}

impl Lattice for Chain {
    fn n_sites(&self) -> usize {
        self.n
    }

    fn bonds(&self) -> &[(usize, usize)] {
        &self.bonds_cache
    }

    fn dmrg_ordering(&self) -> Vec<usize> {
        (0..self.n).collect()
    }

    fn local_dim(&self) -> Option<usize> {
        Some(self.d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_bonds_count() {
        let chain = Chain::new(10, 2);
        assert_eq!(chain.bonds().len(), 9);
    }

    #[test]
    fn chain_dmrg_ordering_identity() {
        let chain = Chain::new(5, 2);
        assert_eq!(chain.dmrg_ordering(), vec![0, 1, 2, 3, 4]);
    }
}
