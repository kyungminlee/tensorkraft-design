//! Structural contraction hook for SU(2) extension point.
//!
//! For Abelian symmetries this is a zero-cost no-op. For SU(2) (Phase 5+),
//! this injects Clebsch-Gordan coefficients into the contraction.

use std::marker::PhantomData;

use smallvec::{smallvec, SmallVec};
use tk_core::Scalar;
use tk_symmetry::BitPackable;

/// Callback trait for injecting structural coefficients into a pairwise
/// block-sparse contraction step.
///
/// For Abelian symmetries, the fusion rule maps one input sector pair to
/// at most one output sector, with coefficient 1. Zero overhead.
///
/// For SU(2) symmetries (Phase 5+), the Wigner-Eckart theorem requires
/// weighting each output block by a Clebsch-Gordan coefficient.
///
/// This trait is defined in `tk-contract` (not `tk-linalg`) because
/// evaluating CG coefficients requires types from `tk-symmetry`.
pub trait StructuralContractionHook<T: Scalar, Q: BitPackable>: Send + Sync {
    /// Called once per sector-pair during pairwise block-sparse contraction.
    ///
    /// **Semantics:** `sector_a` and `sector_b` represent the quantum numbers
    /// of the *free* (non-contracted) legs only. The caller pre-extracts free-leg
    /// quantum numbers before invoking this hook. Contracted legs have already
    /// been matched by the sector-pair selection logic.
    ///
    /// # Returns
    /// A `SmallVec` of `(output_sector, coefficient)` pairs.
    /// - For Abelian: exactly one element with coefficient 1.
    /// - For SU(2): may return multiple elements (fusion multiplicity).
    fn compute_output_sectors(
        &self,
        sector_a: &[Q],
        sector_b: &[Q],
    ) -> SmallVec<[(SmallVec<[Q; 8]>, T); 4]>;
}

/// No-op Abelian implementation (zero runtime overhead).
pub struct AbelianHook<Q: BitPackable> {
    _phantom: PhantomData<Q>,
}

impl<Q: BitPackable> AbelianHook<Q> {
    pub fn new() -> Self {
        AbelianHook {
            _phantom: PhantomData,
        }
    }
}

impl<Q: BitPackable> Default for AbelianHook<Q> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar, Q: BitPackable> StructuralContractionHook<T, Q> for AbelianHook<Q> {
    #[inline(always)]
    fn compute_output_sectors(
        &self,
        sector_a: &[Q],
        sector_b: &[Q],
    ) -> SmallVec<[(SmallVec<[Q; 8]>, T); 4]> {
        // Abelian fusion: element-wise fuse of free-leg quantum numbers.
        // Always produces exactly one output sector with coefficient 1.
        let output: SmallVec<[Q; 8]> = sector_a
            .iter()
            .zip(sector_b.iter())
            .map(|(qa, qb)| qa.fuse(qb))
            .collect();
        smallvec![(output, T::one())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tk_symmetry::U1;

    #[test]
    fn abelian_hook_single_output() {
        let hook = AbelianHook::<U1>::new();
        let result: SmallVec<[(SmallVec<[U1; 8]>, f64); 4]> =
            hook.compute_output_sectors(
                &[U1(1), U1(2)],
                &[U1(3), U1(-1)],
            );
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn abelian_hook_fuses_correctly() {
        let hook = AbelianHook::<U1>::new();
        let result: SmallVec<[(SmallVec<[U1; 8]>, f64); 4]> =
            hook.compute_output_sectors(
                &[U1(1), U1(2)],
                &[U1(3), U1(-1)],
            );
        // 1+3=4, 2+(-1)=1
        assert_eq!(result[0].0[0].0, 4);
        assert_eq!(result[0].0[1].0, 1);
        assert!((result[0].1 - 1.0).abs() < 1e-12);
    }
}
