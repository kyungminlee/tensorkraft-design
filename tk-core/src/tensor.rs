//! Primary dense tensor type with Copy-on-Write storage.

use crate::error::{TkError, TkResult};
use crate::matview::{MatMut, MatRef};
use crate::scalar::Scalar;
use crate::shape::TensorShape;
use crate::storage::{TensorCow, TensorStorage};

/// The primary N-dimensional dense tensor.
///
/// Shape metadata is always owned; storage is Copy-on-Write.
/// Arena-allocated tensors use a shorter lifetime via the `TempTensor` alias.
pub struct DenseTensor<'a, T: Scalar> {
    shape: TensorShape,
    storage: TensorCow<'a, T>,
}

/// Convenience alias: a `DenseTensor` whose storage borrows from an arena.
/// The `'a` lifetime is tied to the `SweepArena`'s current allocation epoch.
pub type TempTensor<'a, T> = DenseTensor<'a, T>;

impl<'a, T: Scalar> DenseTensor<'a, T> {
    /// Allocate a zero-filled owned tensor.
    pub fn zeros(shape: TensorShape) -> DenseTensor<'static, T> {
        let n = shape.numel();
        DenseTensor {
            shape,
            storage: TensorCow::Owned(TensorStorage::zeros(n)),
        }
    }

    /// Create from a flat `Vec` with the given shape.
    ///
    /// # Panics
    /// Panics if `data.len() != shape.numel()`.
    pub fn from_vec(shape: TensorShape, data: Vec<T>) -> DenseTensor<'static, T> {
        assert_eq!(
            data.len(),
            shape.numel(),
            "data length {} does not match shape numel {}",
            data.len(),
            shape.numel()
        );
        DenseTensor {
            shape,
            storage: TensorCow::Owned(TensorStorage::from_vec(data)),
        }
    }

    /// Create a tensor that borrows from existing storage.
    pub fn borrowed(shape: TensorShape, storage: &'a TensorStorage<T>) -> Self {
        debug_assert_eq!(storage.len(), shape.numel());
        DenseTensor {
            shape,
            storage: TensorCow::Borrowed(storage),
        }
    }

    /// Return a zero-copy transposed view by permuting strides.
    /// The returned tensor borrows the same storage.
    pub fn permute(&self, perm: &[usize]) -> DenseTensor<'_, T> {
        DenseTensor {
            shape: self.shape.permute(perm),
            storage: match &self.storage {
                TensorCow::Borrowed(s) => TensorCow::Borrowed(s),
                TensorCow::Owned(s) => TensorCow::Borrowed(s),
            },
        }
    }

    /// Reshape to new dims. Returns Err if non-contiguous or numel mismatch.
    pub fn reshape(&self, new_dims: &[usize]) -> TkResult<DenseTensor<'_, T>> {
        let new_shape = self.shape.reshape(new_dims)?;
        Ok(DenseTensor {
            shape: new_shape,
            storage: match &self.storage {
                TensorCow::Borrowed(s) => TensorCow::Borrowed(s),
                TensorCow::Owned(s) => TensorCow::Borrowed(s),
            },
        })
    }

    /// Slice along one axis.
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> DenseTensor<'_, T> {
        let (new_shape, _offset) = self.shape.slice_axis(axis, start, end);
        // NOTE: In a full implementation, we would adjust the data pointer
        // by `offset` elements. For now we return a view with the sliced shape.
        // The offset tracking requires pointer arithmetic on the underlying storage.
        DenseTensor {
            shape: new_shape,
            storage: match &self.storage {
                TensorCow::Borrowed(s) => TensorCow::Borrowed(s),
                TensorCow::Owned(s) => TensorCow::Borrowed(s),
            },
        }
    }

    /// Materialize into heap-allocated owned storage.
    ///
    /// Must be called before `SweepArena::reset()` for any tensor whose
    /// data must survive past the current sweep step.
    pub fn into_owned(self) -> DenseTensor<'static, T> {
        match self.storage {
            TensorCow::Owned(s) => DenseTensor {
                shape: self.shape,
                storage: TensorCow::Owned(s),
            },
            TensorCow::Borrowed(storage) => DenseTensor {
                shape: self.shape,
                storage: TensorCow::Owned(storage.clone()),
            },
        }
    }

    /// View as a 2-D immutable matrix (row-major). Errors if rank != 2.
    pub fn as_mat_ref(&self) -> TkResult<MatRef<'_, T>> {
        if self.shape.rank() != 2 {
            return Err(TkError::RankError {
                expected: 2,
                got: self.shape.rank(),
            });
        }
        let rows = self.shape.dims()[0];
        let cols = self.shape.dims()[1];
        Ok(MatRef::from_slice_with_strides(
            self.as_slice(),
            rows,
            cols,
            self.shape.strides()[0] as isize,
            self.shape.strides()[1] as isize,
        ))
    }

    /// View as a 2-D mutable matrix (row-major). Errors if rank != 2.
    pub fn as_mat_mut(&mut self) -> TkResult<MatMut<'_, T>> {
        if self.shape.rank() != 2 {
            return Err(TkError::RankError {
                expected: 2,
                got: self.shape.rank(),
            });
        }
        let rows = self.shape.dims()[0];
        let cols = self.shape.dims()[1];
        let row_stride = self.shape.strides()[0] as isize;
        let col_stride = self.shape.strides()[1] as isize;
        let data = self.as_mut_slice();
        Ok(MatMut {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        })
    }

    /// Read-only access to the flat data slice.
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Mutable access to the flat data slice.
    ///
    /// # Panics
    /// Panics if the storage is `Borrowed` (cannot mutate borrowed data).
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match &mut self.storage {
            TensorCow::Owned(s) => s.as_mut_slice(),
            TensorCow::Borrowed(_) => {
                panic!("cannot mutably access borrowed tensor storage; call .into_owned() first")
            }
        }
    }

    /// Read-only access to the shape.
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }
}

impl<T: Scalar> std::fmt::Debug for DenseTensor<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseTensor")
            .field("shape", &self.shape)
            .field("numel", &self.numel())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_zeros() {
        let t = DenseTensor::<f64>::zeros(TensorShape::row_major(&[3, 4]));
        assert_eq!(t.numel(), 12);
        assert_eq!(t.rank(), 2);
        assert!(t.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn tensor_from_vec() {
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let t = DenseTensor::from_vec(TensorShape::row_major(&[3, 4]), data);
        assert_eq!(t.as_slice()[5], 5.0);
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn tensor_from_vec_mismatch() {
        DenseTensor::from_vec(TensorShape::row_major(&[3, 4]), vec![0.0_f64; 10]);
    }

    #[test]
    fn tensor_into_owned() {
        let t = DenseTensor::<f64>::zeros(TensorShape::row_major(&[2, 3]));
        let storage = TensorStorage::<f64>::zeros(6);
        let borrowed = DenseTensor::borrowed(TensorShape::row_major(&[2, 3]), &storage);
        assert!(borrowed.storage.is_borrowed());
        let owned = borrowed.into_owned();
        assert!(owned.storage.is_owned());

        // Already owned => no-op
        let still_owned = t.into_owned();
        assert!(still_owned.storage.is_owned());
    }

    #[test]
    fn tensor_as_mat_ref() {
        let data: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let t = DenseTensor::from_vec(TensorShape::row_major(&[2, 3]), data);
        let m = t.as_mat_ref().unwrap();
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.get(1, 2), 5.0);
    }

    #[test]
    fn tensor_as_mat_ref_rank_error() {
        let t = DenseTensor::<f64>::zeros(TensorShape::row_major(&[2, 3, 4]));
        assert!(matches!(
            t.as_mat_ref(),
            Err(TkError::RankError {
                expected: 2,
                got: 3
            })
        ));
    }
}
