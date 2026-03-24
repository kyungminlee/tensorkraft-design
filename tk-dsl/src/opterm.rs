//! Operator term and product types for building Hamiltonian expressions.
//!
//! The `op()` constructor creates an `OpTerm`, and operator overloading
//! (`*`, scalar multiplication) builds `OpProduct` and `ScaledOpProduct`.

use smallvec::SmallVec;
use tk_core::Scalar;

use crate::operators::SiteOperator;

/// A single operator acting on a specific site.
#[derive(Clone, Debug)]
pub struct OpTerm<T: Scalar> {
    pub operator: SiteOperator<T>,
    pub site: usize,
}

/// Type-safe constructor: `op(SpinOp::Sz, 0)`.
pub fn op<T: Scalar>(operator: impl Into<SiteOperator<T>>, site: usize) -> OpTerm<T> {
    OpTerm {
        operator: operator.into(),
        site,
    }
}

/// A product of operators on different sites.
#[derive(Clone, Debug)]
pub struct OpProduct<T: Scalar> {
    pub factors: SmallVec<[OpTerm<T>; 4]>,
}

impl<T: Scalar> OpProduct<T> {
    /// Number of operator factors.
    pub fn len(&self) -> usize {
        self.factors.len()
    }

    /// Whether the product is empty (should never be in practice).
    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }
}

/// An operator product multiplied by a scalar coefficient.
#[derive(Clone, Debug)]
pub struct ScaledOpProduct<T: Scalar> {
    pub coeff: T,
    pub product: OpProduct<T>,
}

impl<T: Scalar> ScaledOpProduct<T> {
    /// Compute the Hermitian conjugate: conjugate the coefficient,
    /// reverse the factor order, and adjoint each operator.
    pub fn hermitian_conjugate(&self) -> Self {
        let conj_coeff = self.coeff.conj();
        let reversed_factors: SmallVec<[OpTerm<T>; 4]> = self
            .product
            .factors
            .iter()
            .rev()
            .map(|term| OpTerm {
                operator: term.operator.adjoint(),
                site: term.site,
            })
            .collect();
        ScaledOpProduct {
            coeff: conj_coeff,
            product: OpProduct {
                factors: reversed_factors,
            },
        }
    }
}

// --- Generic scaling methods (work for any T: Scalar) ---

impl<T: Scalar> OpTerm<T> {
    /// Scale this single-operator term by a coefficient.
    /// Works for any `T: Scalar`, unlike the `f64`-only `Mul` overload.
    pub fn scale(self, coeff: T) -> ScaledOpProduct<T> {
        ScaledOpProduct {
            coeff,
            product: OpProduct {
                factors: SmallVec::from_elem(self, 1),
            },
        }
    }
}

impl<T: Scalar> OpProduct<T> {
    /// Scale this operator product by a coefficient.
    /// Works for any `T: Scalar`, unlike the `f64`-only `Mul` overload.
    pub fn scale(self, coeff: T) -> ScaledOpProduct<T> {
        ScaledOpProduct {
            coeff,
            product: self,
        }
    }
}

// --- Operator overloading ---

// OpTerm * OpTerm → OpProduct
impl<T: Scalar> std::ops::Mul<OpTerm<T>> for OpTerm<T> {
    type Output = OpProduct<T>;

    fn mul(self, rhs: OpTerm<T>) -> OpProduct<T> {
        let mut factors = SmallVec::new();
        factors.push(self);
        factors.push(rhs);
        OpProduct { factors }
    }
}

// OpProduct * OpTerm → OpProduct
impl<T: Scalar> std::ops::Mul<OpTerm<T>> for OpProduct<T> {
    type Output = OpProduct<T>;

    fn mul(mut self, rhs: OpTerm<T>) -> OpProduct<T> {
        self.factors.push(rhs);
        self
    }
}

// OpTerm * T → ScaledOpProduct
impl std::ops::Mul<f64> for OpTerm<f64> {
    type Output = ScaledOpProduct<f64>;

    fn mul(self, rhs: f64) -> ScaledOpProduct<f64> {
        ScaledOpProduct {
            coeff: rhs,
            product: OpProduct {
                factors: SmallVec::from_elem(self, 1),
            },
        }
    }
}

// T * OpTerm → ScaledOpProduct
impl std::ops::Mul<OpTerm<f64>> for f64 {
    type Output = ScaledOpProduct<f64>;

    fn mul(self, rhs: OpTerm<f64>) -> ScaledOpProduct<f64> {
        ScaledOpProduct {
            coeff: self,
            product: OpProduct {
                factors: SmallVec::from_elem(rhs, 1),
            },
        }
    }
}

// OpProduct * T → ScaledOpProduct
impl std::ops::Mul<f64> for OpProduct<f64> {
    type Output = ScaledOpProduct<f64>;

    fn mul(self, rhs: f64) -> ScaledOpProduct<f64> {
        ScaledOpProduct {
            coeff: rhs,
            product: self,
        }
    }
}

// T * OpProduct → ScaledOpProduct
impl std::ops::Mul<OpProduct<f64>> for f64 {
    type Output = ScaledOpProduct<f64>;

    fn mul(self, rhs: OpProduct<f64>) -> ScaledOpProduct<f64> {
        ScaledOpProduct {
            coeff: self,
            product: rhs,
        }
    }
}

// ScaledOpProduct * OpTerm → ScaledOpProduct (extend product)
impl<T: Scalar> std::ops::Mul<OpTerm<T>> for ScaledOpProduct<T> {
    type Output = ScaledOpProduct<T>;

    fn mul(mut self, rhs: OpTerm<T>) -> ScaledOpProduct<T> {
        self.product.factors.push(rhs);
        self
    }
}

// T * ScaledOpProduct → ScaledOpProduct (rescale)
impl std::ops::Mul<ScaledOpProduct<f64>> for f64 {
    type Output = ScaledOpProduct<f64>;

    fn mul(self, mut rhs: ScaledOpProduct<f64>) -> ScaledOpProduct<f64> {
        rhs.coeff = rhs.coeff * self;
        rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::SpinOp;

    #[test]
    fn op_product_length() {
        let p = op::<f64>(SpinOp::Sz, 0) * op(SpinOp::Sz, 1);
        assert_eq!(p.len(), 2);
        let p2 = p * op(SpinOp::SPlus, 2);
        assert_eq!(p2.len(), 3);
    }

    #[test]
    fn op_scaled_by_scalar() {
        let scaled = 0.5 * op::<f64>(SpinOp::Sz, 0);
        assert!((scaled.coeff - 0.5).abs() < 1e-12);
        assert_eq!(scaled.product.len(), 1);
    }

    #[test]
    fn op_term_scale_generic() {
        use tk_core::C64;
        let coeff = C64::new(1.0, 2.0);
        let scaled = op::<C64>(SpinOp::Sz, 0).scale(coeff);
        assert_eq!(scaled.coeff, coeff);
        assert_eq!(scaled.product.len(), 1);
    }

    #[test]
    fn op_product_scale_generic() {
        use tk_core::C64;
        let coeff = C64::new(0.5, -0.3);
        let product = op::<C64>(SpinOp::Sz, 0) * op(SpinOp::Sz, 1);
        let scaled = product.scale(coeff);
        assert_eq!(scaled.coeff, coeff);
        assert_eq!(scaled.product.len(), 2);
    }

    #[test]
    fn scaled_product_hermitian_conjugate() {
        let sp = ScaledOpProduct {
            coeff: 2.0_f64,
            product: OpProduct {
                factors: SmallVec::from_vec(vec![
                    OpTerm {
                        operator: SpinOp::SPlus.into(),
                        site: 0,
                    },
                    OpTerm {
                        operator: SpinOp::Sz.into(),
                        site: 1,
                    },
                ]),
            },
        };
        let hc = sp.hermitian_conjugate();
        assert!((hc.coeff - 2.0).abs() < 1e-12); // real coefficient unchanged
        assert_eq!(hc.product.factors[0].site, 1); // reversed order
        assert_eq!(hc.product.factors[1].site, 0);
    }
}
