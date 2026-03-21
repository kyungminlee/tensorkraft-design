//! Dimensional metadata: shapes and strides for zero-copy tensor views.

use smallvec::SmallVec;

use crate::error::{TkError, TkResult};

/// Shape and stride metadata for an N-dimensional tensor.
/// Stores no data — only the logical layout.
///
/// The internal `SmallVec<[usize; 6]>` avoids heap allocation for tensors
/// up to rank 6, covering every tensor that appears in DMRG (rank-3 MPS
/// tensors, rank-4 MPO tensors, rank-6 environment blocks).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorShape {
    /// Extent of each dimension.
    dims: SmallVec<[usize; 6]>,
    /// Element-offset multiplier per unit step along each dimension.
    /// Default: row-major (C-order) layout.
    strides: SmallVec<[usize; 6]>,
}

impl TensorShape {
    /// Create a row-major (C-order) shape from extents.
    pub fn row_major(dims: &[usize]) -> Self {
        let strides = Self::compute_row_major_strides(dims);
        TensorShape {
            dims: SmallVec::from_slice(dims),
            strides,
        }
    }

    /// Create a column-major (Fortran-order) shape from extents.
    pub fn col_major(dims: &[usize]) -> Self {
        let strides = Self::compute_col_major_strides(dims);
        TensorShape {
            dims: SmallVec::from_slice(dims),
            strides,
        }
    }

    /// Create a shape with explicit strides (e.g., for a non-contiguous view).
    ///
    /// # Panics
    /// In debug mode, panics if any stride is zero while the corresponding
    /// dimension is > 1.
    pub fn with_strides(dims: &[usize], strides: &[usize]) -> Self {
        assert_eq!(
            dims.len(),
            strides.len(),
            "dims and strides must have the same length"
        );
        debug_assert!(
            dims.iter()
                .zip(strides.iter())
                .all(|(&d, &s)| d <= 1 || s > 0),
            "stride must be non-zero for dimensions > 1"
        );
        TensorShape {
            dims: SmallVec::from_slice(dims),
            strides: SmallVec::from_slice(strides),
        }
    }

    /// Total number of elements: product of all dims.
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Read-only access to dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Read-only access to strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Linear offset for a multi-index: `sum_i index[i] * strides[i]`.
    ///
    /// # Panics
    /// In debug mode, panics if any index component is out of bounds.
    pub fn offset(&self, index: &[usize]) -> usize {
        debug_assert_eq!(index.len(), self.dims.len(), "index rank mismatch");
        debug_assert!(
            index
                .iter()
                .zip(self.dims.iter())
                .all(|(&i, &d)| i < d),
            "index out of bounds: index={index:?}, dims={:?}",
            &self.dims[..]
        );
        index
            .iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum()
    }

    /// True if data is stored contiguously in row-major order.
    pub fn is_contiguous(&self) -> bool {
        let expected = Self::compute_row_major_strides(&self.dims);
        self.strides == expected
    }

    /// Returns a new `TensorShape` with dimensions permuted by `perm`.
    /// Zero-copy: only strides are rearranged.
    ///
    /// # Panics
    /// Panics if `perm` is not a valid permutation of `0..rank()`.
    pub fn permute(&self, perm: &[usize]) -> TensorShape {
        assert_eq!(perm.len(), self.rank(), "permutation length mismatch");
        debug_assert!(
            {
                let mut sorted = SmallVec::<[usize; 6]>::from_slice(perm);
                sorted.sort_unstable();
                sorted.iter().enumerate().all(|(i, &v)| v == i)
            },
            "perm must be a valid permutation"
        );

        let mut new_dims = SmallVec::with_capacity(self.rank());
        let mut new_strides = SmallVec::with_capacity(self.rank());
        for &p in perm {
            new_dims.push(self.dims[p]);
            new_strides.push(self.strides[p]);
        }
        TensorShape {
            dims: new_dims,
            strides: new_strides,
        }
    }

    /// Returns the shape after a reshape to `new_dims`.
    ///
    /// Errors if the total element count differs or if the current
    /// layout is non-contiguous (reshape requires contiguous memory).
    pub fn reshape(&self, new_dims: &[usize]) -> TkResult<TensorShape> {
        if !self.is_contiguous() {
            return Err(TkError::NonContiguous);
        }
        let new_numel: usize = new_dims.iter().product();
        if new_numel != self.numel() {
            return Err(TkError::ReshapeError {
                numel_src: self.numel(),
                dims_dst: new_dims.to_vec(),
            });
        }
        Ok(TensorShape::row_major(new_dims))
    }

    /// Slice along one axis: returns the shape and data-pointer offset
    /// for the sub-tensor at `axis` in `start..end`.
    ///
    /// # Panics
    /// Panics if `axis >= rank()` or `end > dims[axis]` or `start > end`.
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> (TensorShape, usize) {
        assert!(axis < self.rank(), "axis out of bounds");
        assert!(end <= self.dims[axis], "slice end out of bounds");
        assert!(start <= end, "slice start > end");

        let mut new_dims = self.dims.clone();
        new_dims[axis] = end - start;
        let offset = start * self.strides[axis];
        let new_shape = TensorShape {
            dims: new_dims,
            strides: self.strides.clone(),
        };
        (new_shape, offset)
    }

    // -- internal helpers --

    fn compute_row_major_strides(dims: &[usize]) -> SmallVec<[usize; 6]> {
        let rank = dims.len();
        let mut strides = SmallVec::from_elem(0, rank);
        if rank == 0 {
            return strides;
        }
        strides[rank - 1] = 1;
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    fn compute_col_major_strides(dims: &[usize]) -> SmallVec<[usize; 6]> {
        let rank = dims.len();
        let mut strides = SmallVec::from_elem(0, rank);
        if rank == 0 {
            return strides;
        }
        strides[0] = 1;
        for i in 1..rank {
            strides[i] = strides[i - 1] * dims[i - 1];
        }
        strides
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy: generate dims of rank 2..=6, each dimension 1..=8.
    fn dims_strategy() -> impl Strategy<Value = Vec<usize>> {
        prop::collection::vec(1usize..=8, 2..=6)
    }

    proptest! {
        #[test]
        fn prop_offset_within_bounds(dims in dims_strategy()) {
            let shape = TensorShape::row_major(&dims);
            let numel = shape.numel();
            if numel == 0 {
                return Ok(());
            }
            // Generate a valid multi-index: each component in [0, dim).
            let index: Vec<usize> = dims.iter().map(|&d| d - 1).collect();
            let off = shape.offset(&index);
            prop_assert!(off < numel, "offset {} >= numel {} for index {:?}, dims {:?}", off, numel, index, dims);
        }

        #[test]
        fn prop_permute_preserves_numel(dims in dims_strategy()) {
            let shape = TensorShape::row_major(&dims);
            let rank = shape.rank();
            // Use a reversed permutation as a simple valid perm.
            let perm: Vec<usize> = (0..rank).rev().collect();
            let permuted = shape.permute(&perm);
            prop_assert_eq!(shape.numel(), permuted.numel());
        }

        #[test]
        fn prop_permute_roundtrip(dims in dims_strategy()) {
            let shape = TensorShape::row_major(&dims);
            let rank = shape.rank();
            let perm: Vec<usize> = (0..rank).rev().collect();
            let permuted = shape.permute(&perm);
            let restored = permuted.permute(&perm); // reverse of reverse = identity
            prop_assert_eq!(shape.dims(), restored.dims());
            prop_assert_eq!(shape.strides(), restored.strides());
        }

        #[test]
        fn prop_reshape_roundtrip(dims in dims_strategy()) {
            let shape = TensorShape::row_major(&dims);
            let numel = shape.numel();
            if numel == 0 {
                return Ok(());
            }
            // Flatten to 1-D and reshape back.
            let flat = shape.reshape(&[numel]).unwrap();
            prop_assert_eq!(flat.numel(), numel);
            let restored = flat.reshape(&dims).unwrap();
            prop_assert_eq!(restored.dims(), shape.dims());
        }

        #[test]
        fn prop_slice_axis_numel(dims in dims_strategy()) {
            let shape = TensorShape::row_major(&dims);
            // Slice the first axis at 0..1 (single element along that axis).
            let (sliced, _offset) = shape.slice_axis(0, 0, 1);
            let expected_numel = shape.numel() / dims[0];
            prop_assert_eq!(sliced.numel(), expected_numel);
        }

        #[test]
        fn prop_col_major_same_numel(dims in dims_strategy()) {
            let row = TensorShape::row_major(&dims);
            let col = TensorShape::col_major(&dims);
            prop_assert_eq!(row.numel(), col.numel());
            prop_assert_eq!(row.dims(), col.dims());
        }
    }

    #[test]
    fn shape_row_major_strides() {
        let s = TensorShape::row_major(&[3, 4, 5]);
        assert_eq!(s.strides(), &[20, 5, 1]);
        assert_eq!(s.numel(), 60);
        assert_eq!(s.rank(), 3);
    }

    #[test]
    fn shape_col_major_strides() {
        let s = TensorShape::col_major(&[3, 4, 5]);
        assert_eq!(s.strides(), &[1, 3, 12]);
        assert_eq!(s.numel(), 60);
    }

    #[test]
    fn shape_permute_strides() {
        let s = TensorShape::row_major(&[3, 4, 5]);
        let p = s.permute(&[2, 0, 1]);
        assert_eq!(p.dims(), &[5, 3, 4]);
        assert_eq!(p.strides(), &[1, 20, 5]);
        assert_eq!(p.numel(), s.numel());
    }

    #[test]
    fn shape_reshape_ok() {
        let s = TensorShape::row_major(&[3, 4, 5]);
        let r = s.reshape(&[60]).unwrap();
        assert_eq!(r.dims(), &[60]);
        assert!(r.is_contiguous());

        let r2 = s.reshape(&[4, 15]).unwrap();
        assert_eq!(r2.dims(), &[4, 15]);
    }

    #[test]
    fn shape_reshape_noncontiguous_err() {
        let s = TensorShape::row_major(&[3, 4, 5]);
        let p = s.permute(&[2, 0, 1]); // non-contiguous
        assert!(!p.is_contiguous());
        let result = p.reshape(&[60]);
        assert!(matches!(result, Err(TkError::NonContiguous)));
    }

    #[test]
    fn shape_reshape_numel_mismatch() {
        let s = TensorShape::row_major(&[3, 4, 5]);
        let result = s.reshape(&[10, 10]);
        assert!(matches!(result, Err(TkError::ReshapeError { .. })));
    }

    #[test]
    fn shape_offset() {
        let s = TensorShape::row_major(&[3, 4, 5]);
        // index [1, 2, 3] => 1*20 + 2*5 + 3*1 = 33
        assert_eq!(s.offset(&[1, 2, 3]), 33);
    }

    #[test]
    fn shape_slice_axis() {
        let s = TensorShape::row_major(&[3, 4, 5]);
        let (sliced, offset) = s.slice_axis(1, 1, 3);
        assert_eq!(sliced.dims(), &[3, 2, 5]);
        assert_eq!(offset, 5); // start=1, stride[1]=5
    }

    #[test]
    fn shape_is_contiguous() {
        let s = TensorShape::row_major(&[3, 4, 5]);
        assert!(s.is_contiguous());

        let p = s.permute(&[2, 0, 1]);
        assert!(!p.is_contiguous());
    }
}
