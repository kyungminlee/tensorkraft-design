//! 2D triangular lattice.

use super::Lattice;
use super::square::snake_path;

/// A triangular lattice with `lx × ly` sites.
///
/// Has horizontal, vertical, and diagonal (lower-left to upper-right)
/// nearest-neighbour bonds.
#[derive(Clone, Debug)]
pub struct Triangular {
    pub lx: usize,
    pub ly: usize,
    pub d: usize,
    bonds_cache: Vec<(usize, usize)>,
}

impl Triangular {
    pub fn new(lx: usize, ly: usize, d: usize) -> Self {
        assert!(lx >= 1 && ly >= 1, "Triangular lattice requires lx, ly >= 1");
        let mut bonds = Vec::new();

        for y in 0..ly {
            for x in 0..lx {
                let site = y * lx + x;
                // Horizontal bond
                if x + 1 < lx {
                    bonds.push((site, site + 1));
                }
                // Vertical bond
                if y + 1 < ly {
                    let below = (y + 1) * lx + x;
                    bonds.push((site, below));
                }
                // Diagonal bond (lower-left to upper-right): (x,y) → (x+1,y+1)
                if x + 1 < lx && y + 1 < ly {
                    let diag = (y + 1) * lx + (x + 1);
                    bonds.push((site, diag));
                }
            }
        }

        Triangular {
            lx,
            ly,
            d,
            bonds_cache: bonds,
        }
    }
}

impl Lattice for Triangular {
    fn n_sites(&self) -> usize {
        self.lx * self.ly
    }

    fn bonds(&self) -> &[(usize, usize)] {
        &self.bonds_cache
    }

    fn dmrg_ordering(&self) -> Vec<usize> {
        snake_path(self.lx, self.ly)
    }

    fn local_dim(&self) -> Option<usize> {
        Some(self.d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangular_bonds_count() {
        // 3×3: 6 horizontal + 6 vertical + 4 diagonal = 16
        let tri = Triangular::new(3, 3, 2);
        assert_eq!(tri.bonds().len(), 16);
    }
}
