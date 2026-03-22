//! 2D square lattice.

use super::Lattice;

/// A square lattice with `lx × ly` sites.
#[derive(Clone, Debug)]
pub struct Square {
    pub lx: usize,
    pub ly: usize,
    pub d: usize,
    bonds_cache: Vec<(usize, usize)>,
}

impl Square {
    pub fn new(lx: usize, ly: usize, d: usize) -> Self {
        assert!(lx >= 1 && ly >= 1, "Square lattice requires lx, ly >= 1");
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
            }
        }

        Square {
            lx,
            ly,
            d,
            bonds_cache: bonds,
        }
    }
}

impl Lattice for Square {
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

/// Boustrophedon (snake) ordering for mapping a 2D grid to 1D.
///
/// Even rows go left-to-right, odd rows go right-to-left.
pub fn snake_path(lx: usize, ly: usize) -> Vec<usize> {
    let mut path = Vec::with_capacity(lx * ly);
    for y in 0..ly {
        if y % 2 == 0 {
            // Left to right
            for x in 0..lx {
                path.push(y * lx + x);
            }
        } else {
            // Right to left
            for x in (0..lx).rev() {
                path.push(y * lx + x);
            }
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_bonds_count() {
        // 3×3 grid: 6 horizontal + 6 vertical = 12 bonds
        let sq = Square::new(3, 3, 2);
        assert_eq!(sq.bonds().len(), 12);
    }

    #[test]
    fn snake_path_correctness() {
        let path = snake_path(3, 2);
        // Row 0: 0, 1, 2 (left to right)
        // Row 1: 5, 4, 3 (right to left)
        assert_eq!(path, vec![0, 1, 2, 5, 4, 3]);
    }

    #[test]
    fn snake_path_bijection_3x3() {
        let path = snake_path(3, 3);
        assert_eq!(path.len(), 9);
        let mut sorted = path.clone();
        sorted.sort();
        assert_eq!(sorted, (0..9).collect::<Vec<_>>());
    }
}
