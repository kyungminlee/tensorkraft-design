//! Primary dense tensor type with Copy-on-Write storage.

use crate::error::{TkError, TkResult};
use crate::matview::{MatMut, MatRef};
use crate::scalar::Scalar;
use crate::shape::TensorShape;
use crate::storage::TensorStorage;

/// The primary N-dimensional dense tensor.
///
/// Shape metadata is always owned; storage is Copy-on-Write.
/// Arena-allocated tensors use a shorter lifetime via the `TempTensor` alias.
///
/// The `offset` field tracks where this tensor's data begins within the
/// underlying storage buffer. It is zero for freshly constructed tensors
/// and nonzero for sliced views (produced by `slice_axis`).
pub struct DenseTensor<'a, T: Scalar> {
    shape: TensorShape,
    storage: TensorStorage<'a, T>,
    /// Element offset into the storage buffer where this tensor's data begins.
    /// Used by `slice_axis` to create sub-views without copying.
    offset: usize,
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
            storage: TensorStorage::zeros(n),
            offset: 0,
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
            storage: TensorStorage::from_vec(data),
            offset: 0,
        }
    }

    /// Create a tensor that borrows from a slice.
    pub fn borrowed(shape: TensorShape, data: &'a [T]) -> Self {
        debug_assert!(data.len() >= shape.numel());
        DenseTensor {
            shape,
            storage: TensorStorage::from_slice(data),
            offset: 0,
        }
    }

    /// Return a zero-copy transposed view by permuting strides.
    /// The returned tensor borrows the same storage and preserves the offset.
    pub fn permute(&self, perm: &[usize]) -> DenseTensor<'_, T> {
        DenseTensor {
            shape: self.shape.permute(perm),
            storage: self.borrow_storage(),
            offset: self.offset,
        }
    }

    /// Reshape to new dims. Returns Err if non-contiguous or numel mismatch.
    pub fn reshape(&self, new_dims: &[usize]) -> TkResult<DenseTensor<'_, T>> {
        let new_shape = self.shape.reshape(new_dims)?;
        Ok(DenseTensor {
            shape: new_shape,
            storage: self.borrow_storage(),
            offset: self.offset,
        })
    }

    /// Slice along one axis. Returns a zero-copy view into the same buffer
    /// with the offset advanced to the start of the sliced region.
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> DenseTensor<'_, T> {
        let (new_shape, element_offset) = self.shape.slice_axis(axis, start, end);
        DenseTensor {
            shape: new_shape,
            storage: self.borrow_storage(),
            offset: self.offset + element_offset,
        }
    }

    /// Materialize into heap-allocated owned storage.
    ///
    /// Must be called before `SweepArena::reset()` for any tensor whose
    /// data must survive past the current sweep step.
    ///
    /// If the tensor has a nonzero offset or non-contiguous strides,
    /// the owned copy contains only the logical elements (contiguous,
    /// row-major, offset reset to 0).
    pub fn into_owned(self) -> DenseTensor<'static, T> {
        // Fast path: already owned, contiguous, at offset 0, and tight.
        let can_move = self.offset == 0
            && self.shape.is_contiguous()
            && matches!(&self.storage, TensorStorage::Owned(v) if v.len() == self.shape.numel());

        if can_move {
            return DenseTensor {
                shape: self.shape,
                storage: self.storage.into_owned(),
                offset: 0,
            };
        }

        // Slow path: gather elements into a fresh contiguous buffer.
        let data = self.gather_elements();
        let new_shape = TensorShape::row_major(self.shape.dims());
        DenseTensor {
            shape: new_shape,
            storage: TensorStorage::from_vec(data),
            offset: 0,
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
            self.data_slice(),
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
        let data = self.data_slice_mut();
        Ok(MatMut {
            data,
            rows,
            cols,
            row_stride,
            col_stride,
        })
    }

    /// Read-only access to the flat data slice, offset-adjusted.
    ///
    /// The returned slice starts at `self.offset` and extends to the end
    /// of the underlying buffer. For contiguous tensors with offset 0 this
    /// is the entire buffer. For sliced or strided views, callers must use
    /// `shape.offset(index)` relative to the start of this slice.
    pub fn as_slice(&self) -> &[T] {
        self.data_slice()
    }

    /// Mutable access to the flat data slice, offset-adjusted.
    ///
    /// # Panics
    /// Panics if the storage is `Borrowed` (cannot mutate borrowed data).
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data_slice_mut()
    }

    /// Read-only access to the shape.
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// The element offset into the underlying storage buffer.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    // -- internal helpers --

    /// Borrow the underlying storage (regardless of Owned/Borrowed variant).
    fn borrow_storage(&self) -> TensorStorage<'_, T> {
        self.storage.borrow()
    }

    /// Offset-adjusted immutable data slice.
    fn data_slice(&self) -> &[T] {
        &self.storage.as_slice()[self.offset..]
    }

    /// Offset-adjusted mutable data slice.
    ///
    /// # Panics
    /// Panics if the storage is `Borrowed`.
    fn data_slice_mut(&mut self) -> &mut [T] {
        let offset = self.offset;
        &mut self.storage.as_mut_slice()[offset..]
    }

    /// Gather all logical elements into a contiguous Vec in row-major order.
    /// Handles arbitrary offsets and non-contiguous strides.
    fn gather_elements(&self) -> Vec<T> {
        let numel = self.shape.numel();
        if numel == 0 {
            return Vec::new();
        }
        let data = self.data_slice();
        let rank = self.shape.rank();

        // Fast path: contiguous layout — just memcpy the relevant region.
        if self.shape.is_contiguous() {
            return data[..numel].to_vec();
        }

        // General path: iterate over all multi-indices.
        let dims = self.shape.dims();
        let strides = self.shape.strides();
        let mut result = Vec::with_capacity(numel);
        let mut index = vec![0usize; rank];

        for _ in 0..numel {
            let linear: usize = index.iter().zip(strides).map(|(&i, &s)| i * s).sum();
            result.push(data[linear]);

            // Increment multi-index (row-major order: last axis fastest).
            for d in (0..rank).rev() {
                index[d] += 1;
                if index[d] < dims[d] {
                    break;
                }
                index[d] = 0;
            }
        }
        result
    }
}

impl<T: Scalar> std::fmt::Debug for DenseTensor<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseTensor")
            .field("shape", &self.shape)
            .field("offset", &self.offset)
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
        assert_eq!(t.offset(), 0);
        assert!(t.as_slice().iter().take(12).all(|&x| x == 0.0));
    }

    #[test]
    fn tensor_from_vec() {
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let t = DenseTensor::from_vec(TensorShape::row_major(&[3, 4]), data);
        assert_eq!(t.as_slice()[5], 5.0);
        assert_eq!(t.offset(), 0);
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn tensor_from_vec_mismatch() {
        DenseTensor::from_vec(TensorShape::row_major(&[3, 4]), vec![0.0_f64; 10]);
    }

    #[test]
    fn tensor_into_owned() {
        let t = DenseTensor::<f64>::zeros(TensorShape::row_major(&[2, 3]));
        let data = vec![0.0_f64; 6];
        let borrowed = DenseTensor::borrowed(TensorShape::row_major(&[2, 3]), &data);
        assert!(matches!(borrowed.storage, TensorStorage::Borrowed(_)));
        let owned = borrowed.into_owned();
        assert!(matches!(owned.storage, TensorStorage::Owned(_)));
        assert_eq!(owned.offset(), 0);

        // Already owned => no-op
        let still_owned = t.into_owned();
        assert!(matches!(still_owned.storage, TensorStorage::Owned(_)));
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

    // -----------------------------------------------------------------------
    // slice_axis tests
    // -----------------------------------------------------------------------

    #[test]
    fn slice_axis_offset_correct() {
        // 3x4 row-major tensor: [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let t = DenseTensor::from_vec(TensorShape::row_major(&[3, 4]), data);

        // Slice rows 1..3 → should see [[4,5,6,7],[8,9,10,11]]
        let sliced = t.slice_axis(0, 1, 3);
        assert_eq!(sliced.shape().dims(), &[2, 4]);
        assert_eq!(sliced.offset(), 4); // row 1 starts at element 4
        // First element of sliced view is element 4 of the original
        assert_eq!(sliced.as_slice()[0], 4.0);
        // Element at [1,2] of the sliced view = original [2,2] = 10
        let idx_offset = sliced.shape().offset(&[1, 2]);
        assert_eq!(sliced.as_slice()[idx_offset], 10.0);
    }

    #[test]
    fn slice_axis_cols() {
        // 3x4 row-major: [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let t = DenseTensor::from_vec(TensorShape::row_major(&[3, 4]), data);

        // Slice cols 1..3 → shape [3,2], strides [4,1], offset 1
        let sliced = t.slice_axis(1, 1, 3);
        assert_eq!(sliced.shape().dims(), &[3, 2]);
        assert_eq!(sliced.offset(), 1);
        // Element [0,0] of sliced = original element 1
        assert_eq!(sliced.as_slice()[0], 1.0);
        // Element [1,0] of sliced = original[1,1] = 5, at stride offset 4
        let idx_offset = sliced.shape().offset(&[1, 0]);
        assert_eq!(sliced.as_slice()[idx_offset], 5.0);
        // Element [2,1] of sliced = original[2,2] = 10
        let idx_offset = sliced.shape().offset(&[2, 1]);
        assert_eq!(sliced.as_slice()[idx_offset], 10.0);
    }

    #[test]
    fn slice_axis_into_owned_gathers_elements() {
        // 3x4 row-major, slice rows 1..3 → 2x4
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let t = DenseTensor::from_vec(TensorShape::row_major(&[3, 4]), data);
        let sliced = t.slice_axis(0, 1, 3);

        let owned = sliced.into_owned();
        assert_eq!(owned.offset(), 0);
        assert_eq!(owned.numel(), 8);
        assert!(owned.shape().is_contiguous());
        assert_eq!(owned.as_slice()[..8], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn slice_axis_chained() {
        // 4x4 row-major
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let t = DenseTensor::from_vec(TensorShape::row_major(&[4, 4]), data);

        // Slice rows 1..3, then cols 1..3 → 2x2 sub-matrix
        let s1 = t.slice_axis(0, 1, 3);    // offset 4
        let s2 = s1.slice_axis(1, 1, 3);   // offset 4+1 = 5
        assert_eq!(s2.shape().dims(), &[2, 2]);
        assert_eq!(s2.offset(), 5);
        // Element [0,0] = original[1,1] = 5
        assert_eq!(s2.as_slice()[0], 5.0);
        // Element [1,1] = original[2,2] = 10
        let idx_offset = s2.shape().offset(&[1, 1]);
        assert_eq!(s2.as_slice()[idx_offset], 10.0);
    }

    #[test]
    fn slice_axis_mat_ref() {
        // 3x4, slice rows 1..3 → 2x4 matrix view
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let t = DenseTensor::from_vec(TensorShape::row_major(&[3, 4]), data);
        let sliced = t.slice_axis(0, 1, 3);
        let m = sliced.as_mat_ref().unwrap();
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 4);
        assert_eq!(m.get(0, 0), 4.0);  // original[1,0]
        assert_eq!(m.get(1, 3), 11.0); // original[2,3]
    }
}
