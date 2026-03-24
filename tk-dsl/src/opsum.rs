//! OpSum: runtime accumulator of weighted operator products.
//!
//! `OpSum<T>` is the central output of `tk-dsl`. It holds a list of
//! `(coefficient, operator_product)` terms that represent a Hamiltonian
//! or other operator. It is consumed by `tk-dmrg` for MPO compilation.

use tk_core::Scalar;

use crate::lattice::Lattice;
use crate::opterm::{OpProduct, ScaledOpProduct};

/// A single term in an OpSum: coefficient × product of site operators.
#[derive(Clone, Debug)]
pub struct OpSumTerm<T: Scalar> {
    pub coeff: T,
    pub product: OpProduct<T>,
}

/// Marker type for Hermitian conjugate syntax: `term + hc()`.
pub struct HermitianConjugate;

/// Construct the h.c. marker.
pub fn hc() -> HermitianConjugate {
    HermitianConjugate
}

/// A pair of forward + backward (h.c.) terms, produced by `term + hc()`.
pub struct OpSumPair<T: Scalar> {
    pub forward: ScaledOpProduct<T>,
    pub backward: ScaledOpProduct<T>,
}

// ScaledOpProduct + HermitianConjugate → OpSumPair
impl<T: Scalar> std::ops::Add<HermitianConjugate> for ScaledOpProduct<T> {
    type Output = OpSumPair<T>;

    fn add(self, _: HermitianConjugate) -> OpSumPair<T> {
        let backward = self.hermitian_conjugate();
        OpSumPair {
            forward: self,
            backward,
        }
    }
}

/// Runtime accumulator of weighted operator products.
///
/// This is the only output that `tk-dsl` produces. No SVD, no MPO
/// allocation — those happen in `tk-dmrg`.
#[derive(Clone, Debug)]
pub struct OpSum<T: Scalar> {
    terms: Vec<OpSumTerm<T>>,
    lattice: Option<Box<dyn Lattice>>,
}

impl<T: Scalar> OpSum<T> {
    /// Create an empty OpSum with no lattice context.
    pub fn new() -> Self {
        OpSum {
            terms: Vec::new(),
            lattice: None,
        }
    }

    /// Create an empty OpSum with an associated lattice.
    pub fn with_lattice(lattice: impl Lattice + 'static) -> Self {
        OpSum {
            terms: Vec::new(),
            lattice: Some(Box::new(lattice)),
        }
    }

    /// Add a scaled operator product as a new term.
    ///
    /// If a lattice is set, validates that all site indices are in bounds.
    pub fn push_term(&mut self, term: ScaledOpProduct<T>) -> crate::error::DslResult<()> {
        if let Some(ref lattice) = self.lattice {
            let n = lattice.n_sites();
            for factor in &term.product.factors {
                if factor.site >= n {
                    return Err(crate::error::DslError::SiteOutOfBounds {
                        site: factor.site,
                        n_sites: n,
                    });
                }
            }
        }
        self.terms.push(OpSumTerm {
            coeff: term.coeff,
            product: term.product,
        });
        Ok(())
    }

    /// Number of terms.
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Iterate over all terms.
    pub fn iter_terms(&self) -> impl Iterator<Item = &OpSumTerm<T>> {
        self.terms.iter()
    }

    /// Compute the Hermitian conjugate of the entire OpSum.
    pub fn hc(&self) -> OpSum<T> {
        let hc_terms = self
            .terms
            .iter()
            .map(|t| {
                let sp = ScaledOpProduct {
                    coeff: t.coeff,
                    product: t.product.clone(),
                };
                let hc = sp.hermitian_conjugate();
                OpSumTerm {
                    coeff: hc.coeff,
                    product: hc.product,
                }
            })
            .collect();
        OpSum {
            terms: hc_terms,
            lattice: self.lattice.clone(),
        }
    }

    /// Scale all terms by a factor.
    pub fn scale(&mut self, factor: T) {
        for term in &mut self.terms {
            term.coeff = term.coeff * factor;
        }
    }

    /// Extend this OpSum with all terms from another.
    pub fn extend(&mut self, other: OpSum<T>) {
        self.terms.extend(other.terms);
    }

    /// Get the associated lattice, if any.
    pub fn lattice(&self) -> Option<&dyn Lattice> {
        self.lattice.as_deref()
    }
}

impl<T: Scalar> Default for OpSum<T> {
    fn default() -> Self {
        Self::new()
    }
}

// OpSum += ScaledOpProduct
impl<T: Scalar> std::ops::AddAssign<ScaledOpProduct<T>> for OpSum<T> {
    fn add_assign(&mut self, rhs: ScaledOpProduct<T>) {
        self.push_term(rhs).expect("site out of bounds in OpSum += ScaledOpProduct");
    }
}

// OpSum += OpSumPair (forward + h.c.)
impl<T: Scalar> std::ops::AddAssign<OpSumPair<T>> for OpSum<T> {
    fn add_assign(&mut self, rhs: OpSumPair<T>) {
        *self += rhs.forward;
        *self += rhs.backward;
    }
}

// OpSum + OpSum → OpSum
impl<T: Scalar> std::ops::Add<OpSum<T>> for OpSum<T> {
    type Output = OpSum<T>;

    fn add(mut self, rhs: OpSum<T>) -> OpSum<T> {
        self.extend(rhs);
        self
    }
}

// OpSum * T → OpSum (generic, works for any Scalar)
impl<T: Scalar> std::ops::Mul<T> for OpSum<T> {
    type Output = OpSum<T>;

    fn mul(mut self, rhs: T) -> OpSum<T> {
        self.scale(rhs);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::Chain;
    use crate::operators::SpinOp;
    use crate::opterm::op;

    #[test]
    fn opsum_push_in_bounds() {
        let mut opsum = OpSum::with_lattice(Chain::new(4, 2));
        let result = opsum.push_term(ScaledOpProduct {
            coeff: 1.0_f64,
            product: crate::opterm::OpProduct {
                factors: smallvec::smallvec![crate::opterm::OpTerm {
                    operator: SpinOp::Sz.into(),
                    site: 3,
                }],
            },
        });
        assert!(result.is_ok());
    }

    #[test]
    fn opsum_push_out_of_bounds() {
        let mut opsum = OpSum::with_lattice(Chain::new(4, 2));
        let result = opsum.push_term(ScaledOpProduct {
            coeff: 1.0_f64,
            product: crate::opterm::OpProduct {
                factors: smallvec::smallvec![crate::opterm::OpTerm {
                    operator: SpinOp::Sz.into(),
                    site: 4, // out of bounds
                }],
            },
        });
        assert!(result.is_err());
    }

    #[test]
    fn opsum_hc_spin_plus_minus() {
        let mut opsum = OpSum::<f64>::new();
        opsum += 1.0 * op(SpinOp::SPlus, 0) * op(SpinOp::SMinus, 1);
        let hc = opsum.hc();
        assert_eq!(hc.n_terms(), 1);
        // h.c. of SPlus(0)*SMinus(1) = SPlus(1)*SMinus(0) with sites reversed
        let term = &hc.terms[0];
        assert_eq!(term.product.factors[0].site, 1);
        assert_eq!(term.product.factors[1].site, 0);
    }

    #[test]
    fn opsum_hc_involution() {
        let mut opsum = OpSum::<f64>::new();
        opsum += 2.0 * op(SpinOp::SPlus, 0);
        let hc2 = opsum.hc().hc();
        assert_eq!(hc2.n_terms(), 1);
        assert!((hc2.terms[0].coeff - 2.0).abs() < 1e-12);
    }

    #[test]
    fn opsum_scale() {
        let mut opsum = OpSum::<f64>::new();
        opsum += 1.0 * op(SpinOp::Sz, 0);
        opsum.scale(3.0);
        assert!((opsum.terms[0].coeff - 3.0).abs() < 1e-12);
    }

    #[test]
    fn opsum_extend() {
        let mut a = OpSum::<f64>::new();
        a += 1.0 * op(SpinOp::Sz, 0);
        let mut b = OpSum::<f64>::new();
        b += 2.0 * op(SpinOp::SPlus, 1);
        a.extend(b);
        assert_eq!(a.n_terms(), 2);
    }

    #[test]
    fn hc_marker_adds_two_terms() {
        let mut opsum = OpSum::<f64>::new();
        opsum += 1.0 * op(SpinOp::SPlus, 0) * op(SpinOp::SMinus, 1) + hc();
        assert_eq!(opsum.n_terms(), 2);
    }
}
