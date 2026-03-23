//! Star geometry for Anderson Impurity Model (AIM).

use super::Lattice;

/// Star geometry: one impurity site connected to `n_bath` bath sites.
///
/// Site 0 is the impurity. Sites 1..=n_bath are bath orbitals.
/// No bath-bath bonds exist.
#[derive(Clone, Debug)]
pub struct StarGeometry {
    pub n_bath: usize,
    pub d: usize,
    bonds_cache: Vec<(usize, usize)>,
}

impl StarGeometry {
    pub fn new(n_bath: usize, d: usize) -> Self {
        assert!(n_bath >= 1, "StarGeometry requires at least 1 bath site");
        let bonds_cache: Vec<(usize, usize)> = (1..=n_bath).map(|b| (0, b)).collect();
        StarGeometry {
            n_bath,
            d,
            bonds_cache,
        }
    }
}

impl Lattice for StarGeometry {
    fn n_sites(&self) -> usize {
        self.n_bath + 1
    }

    fn bonds(&self) -> &[(usize, usize)] {
        &self.bonds_cache
    }

    /// DMRG ordering: impurity at center, bath sites split left/right.
    ///
    /// For n_bath=6: [bath_3, bath_2, bath_1, impurity, bath_4, bath_5, bath_6]
    /// This places the impurity at the center of the 1D chain for optimal DMRG.
    fn dmrg_ordering(&self) -> Vec<usize> {
        let half = self.n_bath / 2;
        let mut ordering = Vec::with_capacity(self.n_bath + 1);

        // Left bath sites (reversed for symmetry)
        for i in (1..=half).rev() {
            ordering.push(i);
        }

        // Impurity at center
        ordering.push(0);

        // Right bath sites
        for i in (half + 1)..=self.n_bath {
            ordering.push(i);
        }

        ordering
    }

    fn local_dim(&self) -> Option<usize> {
        Some(self.d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn star_geometry_bonds_star_only() {
        let star = StarGeometry::new(4, 4);
        assert_eq!(star.bonds().len(), 4);
        // All bonds connect to impurity (site 0)
        for &(a, b) in star.bonds() {
            assert!(a == 0 || b == 0);
        }
    }

    #[test]
    fn star_geometry_dmrg_ordering_center() {
        let star = StarGeometry::new(6, 4);
        let ordering = star.dmrg_ordering();
        assert_eq!(ordering.len(), 7);
        // Impurity should be at center position (index 3)
        assert_eq!(ordering[3], 0);
    }

    #[test]
    fn star_geometry_dmrg_ordering_bijection() {
        let star = StarGeometry::new(6, 4);
        let ordering = star.dmrg_ordering();
        let mut sorted = ordering.clone();
        sorted.sort();
        assert_eq!(sorted, (0..7).collect::<Vec<_>>());
    }
}
