//! SU(2) non-Abelian symmetry support.
//!
//! Enabled only when `features = ["su2-symmetry"]`.

mod cg_cache;

pub use cg_cache::ClebschGordanCache;

use smallvec::SmallVec;

use hashbrown::HashMap;
use tk_core::{DenseTensor, Scalar};

use crate::quantum_number::QuantumNumber;

/// SU(2) irreducible representation, labeled by spin j.
///
/// Uses `twice_j = 2j` to avoid floating-point arithmetic.
/// j=0 → 0, j=1/2 → 1, j=1 → 2, j=3/2 → 3, etc.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SU2Irrep {
    /// Twice the spin quantum number.
    pub twice_j: u32,
}

impl QuantumNumber for SU2Irrep {
    fn identity() -> Self {
        SU2Irrep { twice_j: 0 }
    }

    /// Returns the *lowest* irrep in the tensor product decomposition:
    /// j₁ ⊗ j₂ = |j₁−j₂| ⊕ ... ⊕ (j₁+j₂).
    /// The full one-to-many fusion is handled by `fuse_all`.
    fn fuse(&self, other: &Self) -> Self {
        let twice_j_min = (self.twice_j as i32 - other.twice_j as i32).unsigned_abs();
        SU2Irrep {
            twice_j: twice_j_min,
        }
    }

    /// SU(2) irreps are self-dual (spin is its own conjugate representation).
    fn dual(&self) -> Self {
        *self
    }
}

impl SU2Irrep {
    /// Multiplet dimension: 2j + 1.
    pub fn dim(&self) -> usize {
        (self.twice_j + 1) as usize
    }

    /// Full tensor-product decomposition: all irreps j_c in j_a ⊗ j_b.
    /// Returns the range |j_a − j_b| ..= (j_a + j_b) in steps of 2 (in twice_j units).
    pub fn fuse_all(a: &Self, b: &Self) -> impl Iterator<Item = Self> {
        let min = (a.twice_j as i32 - b.twice_j as i32).unsigned_abs();
        let max = a.twice_j + b.twice_j;
        (min..=max)
            .step_by(2)
            .map(|twice_j| SU2Irrep { twice_j })
    }
}

/// An SU(2)-symmetric tensor in the Wigner-Eckart (reduced matrix element) form.
///
/// By the Wigner-Eckart theorem, T^{j_c}_{m_c} = ⟨j_a, m_a; j_b, m_b | j_c, m_c⟩ · T̃^{j_c}
/// where T̃^{j_c} is the reduced matrix element (scalar under rotation).
///
/// Only the reduced matrix elements are stored; the CG coefficients are
/// looked up from the cache at contraction time.
pub struct WignerEckartTensor<T: Scalar> {
    /// Clebsch-Gordan / 6j / 9j structural coefficient evaluator.
    /// The core contraction engine includes an optional `structural_contraction`
    /// callback from day one: the Abelian code path passes a no-op (zero overhead);
    /// the SU(2) code path injects symbol evaluations via this cache.
    pub structural: ClebschGordanCache,
    /// Reduced matrix elements, stored as a block-sparse tensor with SU2Irrep sectors.
    /// Uses SmallVec-keyed storage (not PackedSectorKey) because SU2Irrep
    /// is not BitPackable.
    pub reduced: HashMap<SmallVec<[SU2Irrep; 6]>, DenseTensor<'static, T>>,
    /// Tensor flux (irrep of the operator).
    pub flux: SU2Irrep,
}

impl<T: Scalar> WignerEckartTensor<T> {
    /// Construct a new Wigner-Eckart tensor with the given flux and
    /// a fresh CG cache.
    pub fn new(flux: SU2Irrep) -> Self {
        WignerEckartTensor {
            structural: ClebschGordanCache::new(),
            reduced: HashMap::new(),
            flux,
        }
    }

    /// Insert a reduced matrix element block for a given sector.
    pub fn insert_reduced(
        &mut self,
        sector: SmallVec<[SU2Irrep; 6]>,
        block: DenseTensor<'static, T>,
    ) {
        self.reduced.insert(sector, block);
    }

    /// Get a reduced matrix element block.
    pub fn get_reduced(&self, sector: &[SU2Irrep]) -> Option<&DenseTensor<'static, T>> {
        let key: SmallVec<[SU2Irrep; 6]> = SmallVec::from_slice(sector);
        self.reduced.get(&key)
    }

    /// Number of reduced blocks.
    pub fn n_sectors(&self) -> usize {
        self.reduced.len()
    }
}

impl<T: Scalar> std::fmt::Debug for WignerEckartTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WignerEckartTensor")
            .field("flux", &self.flux)
            .field("n_sectors", &self.n_sectors())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn su2_identity() {
        assert_eq!(SU2Irrep::identity(), SU2Irrep { twice_j: 0 });
    }

    #[test]
    fn su2_fuse_lowest() {
        let j1 = SU2Irrep { twice_j: 2 }; // j=1
        let j2 = SU2Irrep { twice_j: 4 }; // j=2
        // Lowest: |1-2| = 1, so twice_j = 2
        assert_eq!(j1.fuse(&j2), SU2Irrep { twice_j: 2 });
    }

    #[test]
    fn su2_fuse_all() {
        let j1 = SU2Irrep { twice_j: 2 }; // j=1
        let j2 = SU2Irrep { twice_j: 4 }; // j=2
        let all: Vec<_> = SU2Irrep::fuse_all(&j1, &j2).collect();
        // j=1 ⊗ j=2 = j=1 ⊕ j=2 ⊕ j=3
        assert_eq!(
            all,
            vec![
                SU2Irrep { twice_j: 2 },
                SU2Irrep { twice_j: 4 },
                SU2Irrep { twice_j: 6 },
            ]
        );
    }

    #[test]
    fn su2_dim() {
        assert_eq!(SU2Irrep { twice_j: 0 }.dim(), 1);
        assert_eq!(SU2Irrep { twice_j: 1 }.dim(), 2);
        assert_eq!(SU2Irrep { twice_j: 2 }.dim(), 3);
        assert_eq!(SU2Irrep { twice_j: 4 }.dim(), 5);
    }

    #[test]
    fn su2_self_dual() {
        let j = SU2Irrep { twice_j: 3 };
        assert_eq!(j.dual(), j);
    }

    #[test]
    fn wigner_eckart_new() {
        let wet = WignerEckartTensor::<f64>::new(SU2Irrep { twice_j: 0 });
        assert_eq!(wet.n_sectors(), 0);
        assert_eq!(wet.flux, SU2Irrep { twice_j: 0 });
    }
}
