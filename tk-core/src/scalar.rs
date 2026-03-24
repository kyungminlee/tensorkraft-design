//! The `Scalar` trait hierarchy and element-type abstractions.

use num_complex::Complex;
use num_traits::{Float, One, Zero};
use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};

/// Convenience alias for `Complex<f64>`.
pub type C64 = Complex<f64>;
/// Convenience alias for `Complex<f32>`.
pub type C32 = Complex<f32>;

/// Sealed marker for element types supported by tensorkraft.
///
/// Implemented for: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`,
/// and optionally `f128` when feature `backend-oxiblas` is active.
pub trait Scalar:
    Copy
    + Clone
    + Send
    + Sync
    + Zero
    + One
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Debug
    + 'static
{
    /// The underlying real type. `f64` for both `f64` and `Complex<f64>`.
    type Real: Scalar<Real = Self::Real> + Float + PartialOrd;

    /// Complex conjugate. No-op for real types.
    fn conj(self) -> Self;

    /// Squared absolute value: |z|².
    fn abs_sq(self) -> Self::Real;

    /// Embed a real value into this scalar type.
    fn from_real(r: Self::Real) -> Self;

    /// Construct a scalar from real and imaginary parts.
    ///
    /// For real types (`f32`, `f64`), the imaginary part is ignored and
    /// only `re` is returned. For complex types, returns `re + im*i`.
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self;

    /// Returns true iff complex conjugation is a no-op (i.e., T is real).
    /// Used by the contraction engine to skip conjugation-flag propagation
    /// in tight loops over real-valued models.
    fn is_real() -> bool;
}

// ---------------------------------------------------------------------------
// Implementations for f32
// ---------------------------------------------------------------------------

impl Scalar for f32 {
    type Real = f32;

    #[inline(always)]
    fn conj(self) -> Self {
        self
    }

    #[inline(always)]
    fn abs_sq(self) -> Self::Real {
        self * self
    }

    #[inline(always)]
    fn from_real(r: Self::Real) -> Self {
        r
    }

    #[inline(always)]
    fn from_real_imag(re: Self::Real, _im: Self::Real) -> Self {
        re
    }

    #[inline(always)]
    fn is_real() -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Implementations for f64
// ---------------------------------------------------------------------------

impl Scalar for f64 {
    type Real = f64;

    #[inline(always)]
    fn conj(self) -> Self {
        self
    }

    #[inline(always)]
    fn abs_sq(self) -> Self::Real {
        self * self
    }

    #[inline(always)]
    fn from_real(r: Self::Real) -> Self {
        r
    }

    #[inline(always)]
    fn from_real_imag(re: Self::Real, _im: Self::Real) -> Self {
        re
    }

    #[inline(always)]
    fn is_real() -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Implementations for Complex<f32>
// ---------------------------------------------------------------------------

impl Scalar for Complex<f32> {
    type Real = f32;

    #[inline(always)]
    fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }

    #[inline(always)]
    fn abs_sq(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline(always)]
    fn from_real(r: Self::Real) -> Self {
        Complex::new(r, 0.0)
    }

    #[inline(always)]
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self {
        Complex::new(re, im)
    }

    #[inline(always)]
    fn is_real() -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Implementations for Complex<f64>
// ---------------------------------------------------------------------------

impl Scalar for Complex<f64> {
    type Real = f64;

    #[inline(always)]
    fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }

    #[inline(always)]
    fn abs_sq(self) -> Self::Real {
        self.re * self.re + self.im * self.im
    }

    #[inline(always)]
    fn from_real(r: Self::Real) -> Self {
        Complex::new(r, 0.0)
    }

    #[inline(always)]
    fn from_real_imag(re: Self::Real, im: Self::Real) -> Self {
        Complex::new(re, im)
    }

    #[inline(always)]
    fn is_real() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_conj_complex() {
        let z = C64::new(3.0, 4.0);
        let zc = z.conj();
        assert_eq!(zc.re, 3.0);
        assert_eq!(zc.im, -4.0);
    }

    #[test]
    fn scalar_is_real_f64() {
        assert!(f64::is_real());
    }

    #[test]
    fn scalar_is_real_c64() {
        assert!(!C64::is_real());
    }

    #[test]
    fn scalar_abs_sq() {
        let z = C64::new(3.0, 4.0);
        assert!((z.abs_sq() - 25.0).abs() < 1e-12);
    }

    #[test]
    fn scalar_from_real() {
        let z = C64::from_real(5.0);
        assert_eq!(z.re, 5.0);
        assert_eq!(z.im, 0.0);
    }
}
