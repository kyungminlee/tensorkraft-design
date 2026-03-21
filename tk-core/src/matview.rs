//! 2-D matrix views with lazy conjugation for zero-copy Hermitian transposes.

use crate::scalar::Scalar;

/// An immutable 2-D view into a contiguous-or-strided buffer.
///
/// Carries lazy conjugation semantics for zero-copy Hermitian transposes.
/// The `is_conjugated` flag instructs the backend to treat the underlying
/// data as complex-conjugated without touching memory. For real `T`, this
/// flag has no effect.
#[derive(Clone, Copy, Debug)]
pub struct MatRef<'a, T: Scalar> {
    pub data: &'a [T],
    pub rows: usize,
    pub cols: usize,
    /// Offset in elements to advance by one row.
    pub row_stride: isize,
    /// Offset in elements to advance by one column.
    pub col_stride: isize,
    /// If true, the backend treats each element as its complex conjugate
    /// during GEMM/SVD. For real T, this flag has no effect.
    pub is_conjugated: bool,
}

/// A mutable 2-D view. No conjugation flag — writes are always literal.
#[derive(Debug)]
pub struct MatMut<'a, T: Scalar> {
    pub data: &'a mut [T],
    pub rows: usize,
    pub cols: usize,
    pub row_stride: isize,
    pub col_stride: isize,
}

// ---------------------------------------------------------------------------
// MatRef constructors
// ---------------------------------------------------------------------------

impl<'a, T: Scalar> MatRef<'a, T> {
    /// Row-major contiguous matrix (C layout).
    pub fn from_slice(data: &'a [T], rows: usize, cols: usize) -> Self {
        debug_assert!(
            data.len() >= rows * cols,
            "slice too small for {}x{} matrix",
            rows,
            cols
        );
        MatRef {
            data,
            rows,
            cols,
            row_stride: cols as isize,
            col_stride: 1,
            is_conjugated: false,
        }
    }

    /// Column-major contiguous matrix (Fortran layout).
    pub fn from_slice_col_major(data: &'a [T], rows: usize, cols: usize) -> Self {
        debug_assert!(
            data.len() >= rows * cols,
            "slice too small for {}x{} matrix",
            rows,
            cols
        );
        MatRef {
            data,
            rows,
            cols,
            row_stride: 1,
            col_stride: rows as isize,
            is_conjugated: false,
        }
    }

    /// Arbitrary strides.
    pub fn from_slice_with_strides(
        data: &'a [T],
        rows: usize,
        cols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> Self {
        MatRef {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
            is_conjugated: false,
        }
    }
}

// ---------------------------------------------------------------------------
// MatRef adjoint / conjugate / transpose
// ---------------------------------------------------------------------------

impl<'a, T: Scalar> MatRef<'a, T> {
    /// Returns a zero-copy view of A† (Hermitian conjugate).
    /// Swaps rows <-> cols and row_stride <-> col_stride.
    /// Flips `is_conjugated`.
    /// For real T: equivalent to transpose.
    #[inline(always)]
    pub fn adjoint(&self) -> MatRef<'a, T> {
        MatRef {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            is_conjugated: !self.is_conjugated,
        }
    }

    /// Returns a zero-copy conjugated view without transposing.
    #[inline(always)]
    pub fn conjugate(&self) -> MatRef<'a, T> {
        MatRef {
            is_conjugated: !self.is_conjugated,
            ..*self
        }
    }

    /// Returns a zero-copy transposed view (not conjugated).
    #[inline(always)]
    pub fn transpose(&self) -> MatRef<'a, T> {
        MatRef {
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            ..*self
        }
    }

    /// Element access by (row, col).
    ///
    /// If `is_conjugated` is true, the returned value is the complex conjugate
    /// of the stored element.
    pub fn get(&self, row: usize, col: usize) -> T {
        debug_assert!(row < self.rows, "row index out of bounds");
        debug_assert!(col < self.cols, "col index out of bounds");
        let idx = (row as isize * self.row_stride + col as isize * self.col_stride) as usize;
        let val = self.data[idx];
        if self.is_conjugated {
            val.conj()
        } else {
            val
        }
    }

    /// True if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }
}

// ---------------------------------------------------------------------------
// MatMut constructors
// ---------------------------------------------------------------------------

impl<'a, T: Scalar> MatMut<'a, T> {
    /// Row-major contiguous mutable matrix (C layout).
    pub fn from_slice(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        debug_assert!(
            data.len() >= rows * cols,
            "slice too small for {}x{} matrix",
            rows,
            cols
        );
        MatMut {
            data,
            rows,
            cols,
            row_stride: cols as isize,
            col_stride: 1,
        }
    }

    /// Element access by (row, col).
    pub fn get(&self, row: usize, col: usize) -> T {
        debug_assert!(row < self.rows, "row index out of bounds");
        debug_assert!(col < self.cols, "col index out of bounds");
        let idx = (row as isize * self.row_stride + col as isize * self.col_stride) as usize;
        self.data[idx]
    }

    /// Mutable element access by (row, col).
    pub fn set(&mut self, row: usize, col: usize, val: T) {
        debug_assert!(row < self.rows, "row index out of bounds");
        debug_assert!(col < self.cols, "col index out of bounds");
        let idx = (row as isize * self.row_stride + col as isize * self.col_stride) as usize;
        self.data[idx] = val;
    }

    /// True if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::C64;
    use num_complex::Complex;

    #[test]
    fn matref_adjoint_zero_copy() {
        let data = vec![1.0_f64; 12];
        let m = MatRef::from_slice(&data, 3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert_eq!(m.row_stride, 4);
        assert_eq!(m.col_stride, 1);
        assert!(!m.is_conjugated);

        let adj = m.adjoint();
        assert_eq!(adj.rows, 4);
        assert_eq!(adj.cols, 3);
        assert_eq!(adj.row_stride, 1);
        assert_eq!(adj.col_stride, 4);
        assert!(adj.is_conjugated);
        // Data pointer unchanged — zero copy
        assert!(std::ptr::eq(m.data, adj.data));
    }

    #[test]
    fn matref_adjoint_real_type() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let m = MatRef::from_slice(&data, 2, 2);
        let adj = m.adjoint();
        assert!(adj.is_conjugated);
        // For real types, conjugation is a no-op
        assert!(f64::is_real());
    }

    #[test]
    fn matref_adjoint_roundtrip() {
        let data: Vec<C64> = (0..12)
            .map(|i| Complex::new(i as f64, (i as f64) * 0.1))
            .collect();
        let m = MatRef::from_slice(&data, 3, 4);

        let roundtrip = m.adjoint().adjoint();
        assert_eq!(roundtrip.rows, m.rows);
        assert_eq!(roundtrip.cols, m.cols);
        assert_eq!(roundtrip.row_stride, m.row_stride);
        assert_eq!(roundtrip.col_stride, m.col_stride);
        assert_eq!(roundtrip.is_conjugated, m.is_conjugated);
    }

    #[test]
    fn matref_conjugate() {
        let data = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let m = MatRef::from_slice(&data, 1, 2);
        assert!(!m.is_conjugated);
        let c = m.conjugate();
        assert!(c.is_conjugated);
        assert_eq!(c.rows, 1);
        assert_eq!(c.cols, 2);
        // strides unchanged
        assert_eq!(c.row_stride, m.row_stride);
        assert_eq!(c.col_stride, m.col_stride);
    }

    #[test]
    fn matref_transpose() {
        let data = vec![1.0_f64; 6];
        let m = MatRef::from_slice(&data, 2, 3);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        // transpose does NOT flip conjugation
        assert_eq!(t.is_conjugated, m.is_conjugated);
    }

    #[test]
    fn matref_get_with_conjugation() {
        let data = vec![Complex::new(3.0, 4.0)];
        let m = MatRef::from_slice(&data, 1, 1);
        assert_eq!(m.get(0, 0), Complex::new(3.0, 4.0));

        let c = m.conjugate();
        assert_eq!(c.get(0, 0), Complex::new(3.0, -4.0));
    }

    #[test]
    fn matmut_set_get() {
        let mut data = vec![0.0_f64; 4];
        let mut m = MatMut::from_slice(&mut data, 2, 2);
        m.set(0, 1, 5.0);
        m.set(1, 0, 7.0);
        assert_eq!(m.get(0, 1), 5.0);
        assert_eq!(m.get(1, 0), 7.0);
    }
}
