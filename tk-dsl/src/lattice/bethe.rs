//! Bethe lattice (Cayley tree).

use super::Lattice;

/// A Bethe lattice with coordination number `z` and `depth` shells.
///
/// Site 0 is the root. Sites are numbered in breadth-first order.
#[derive(Clone, Debug)]
pub struct BetheLattice {
    pub z: usize,
    pub depth: usize,
    pub d: usize,
    sites_count: usize,
    bonds_cache: Vec<(usize, usize)>,
}

impl BetheLattice {
    pub fn new(z: usize, depth: usize, d: usize) -> Self {
        assert!(z >= 2, "Bethe lattice requires coordination number z >= 2");
        assert!(depth >= 1, "Bethe lattice requires depth >= 1");

        let mut bonds = Vec::new();
        let mut total_sites = 1usize; // root
        let mut current_shell_start = 0usize;
        let mut current_shell_size = 1usize;

        for shell in 0..depth {
            let branching = if shell == 0 { z } else { z - 1 };
            let next_shell_start = total_sites;

            for i in 0..current_shell_size {
                let parent = current_shell_start + i;
                for _b in 0..branching {
                    let child = total_sites;
                    let (lo, hi) = if parent < child {
                        (parent, child)
                    } else {
                        (child, parent)
                    };
                    bonds.push((lo, hi));
                    total_sites += 1;
                }
            }

            current_shell_start = next_shell_start;
            current_shell_size = total_sites - next_shell_start;
        }

        BetheLattice {
            z,
            depth,
            d,
            sites_count: total_sites,
            bonds_cache: bonds,
        }
    }
}

impl Lattice for BetheLattice {
    fn n_sites(&self) -> usize {
        self.sites_count
    }

    fn bonds(&self) -> &[(usize, usize)] {
        &self.bonds_cache
    }

    fn dmrg_ordering(&self) -> Vec<usize> {
        // Breadth-first ordering (already the natural numbering)
        (0..self.sites_count).collect()
    }

    fn local_dim(&self) -> Option<usize> {
        Some(self.d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bethe_z3_depth1() {
        // z=3, depth=1: root + 3 children = 4 sites, 3 bonds
        let b = BetheLattice::new(3, 1, 2);
        assert_eq!(b.n_sites(), 4);
        assert_eq!(b.bonds().len(), 3);
    }

    #[test]
    fn bethe_z3_depth2() {
        // z=3, depth=2: 1 + 3 + 3*2 = 10 sites, 9 bonds
        let b = BetheLattice::new(3, 2, 2);
        assert_eq!(b.n_sites(), 10);
        assert_eq!(b.bonds().len(), 9);
    }
}
