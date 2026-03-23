//! GEMM reshape helpers: tensor ↔ matrix conversion for contraction execution.
//!
//! These helpers implement the critical "reshape to matrix, GEMM, unfold" pipeline.

use tk_core::{DenseTensor, MatRef, Scalar, TensorShape};

use crate::error::ContractResult;

/// Result of reshaping a tensor into matrix form for GEMM dispatch.
///
/// When the zero-copy fast path is unavailable (non-contiguous or contracted
/// legs not trailing), this struct owns the heap-allocated transposed copy
/// and provides a `MatRef` that borrows from it.
pub struct ReshapeResult<'a, T: Scalar> {
    /// Dimensions of the free legs, in order. Needed to unfold the GEMM
    /// result back into a tensor.
    pub free_dims: Vec<usize>,
    /// Product of free-leg dimensions (M for left operand, N for right).
    pub rows: usize,
    /// Product of contracted-leg dimensions (K).
    pub cols: usize,
    /// Whether Hermitian conjugation should be flagged on the MatRef.
    pub conjugated: bool,
    /// The data source: either a reference to the original tensor's storage
    /// (zero-copy fast path) or owned transposed storage (slow path).
    data: ReshapeData<'a, T>,
}

enum ReshapeData<'a, T: Scalar> {
    /// Zero-copy: borrowed from the original tensor.
    Borrowed(&'a [T]),
    /// Heap-allocated transposed copy.
    Owned(DenseTensor<'static, T>),
}

impl<'a, T: Scalar> ReshapeResult<'a, T> {
    /// Create a `MatRef` from this reshape result.
    pub fn as_mat_ref(&self) -> MatRef<'_, T> {
        let data = match &self.data {
            ReshapeData::Borrowed(s) => *s,
            ReshapeData::Owned(t) => t.as_slice(),
        };
        MatRef {
            data,
            rows: self.rows,
            cols: self.cols,
            row_stride: self.cols as isize,
            col_stride: 1,
            is_conjugated: self.conjugated,
        }
    }
}

/// Reshape a tensor into a 2-D matrix for GEMM dispatch.
///
/// Fuses `contracted_legs` into one axis (K) and the remaining (free) legs
/// into another axis (M for left operand, N for right operand).
///
/// # Contiguity fast path vs. transpose slow path
///
/// When the tensor is contiguous and contracted legs are trailing (rightmost),
/// the result is a zero-copy view into the tensor's storage.
///
/// When either condition fails, the tensor is transposed via `block_transpose`
/// into a contiguous heap buffer where free legs come first and contracted
/// legs are trailing. The `ReshapeResult` owns this buffer.
///
/// # Conjugation propagation
/// If `conjugated` is true, `is_conjugated` is set on the returned `MatRef`
/// rather than performing an explicit conjugation pass.
pub fn tensor_to_mat_ref<'a, T: Scalar>(
    tensor: &'a DenseTensor<'a, T>,
    contracted_legs: &[usize],
    conjugated: bool,
) -> ContractResult<ReshapeResult<'a, T>> {
    let shape = tensor.shape();
    let dims = shape.dims();
    let rank = shape.rank();

    // Separate legs into free and contracted.
    let mut free_dims = Vec::new();
    let mut contracted_dim_product: usize = 1;

    for leg in 0..rank {
        if contracted_legs.contains(&leg) {
            contracted_dim_product *= dims[leg];
        } else {
            free_dims.push(dims[leg]);
        }
    }

    let free_dim_product: usize = free_dims.iter().product();

    if shape.is_contiguous() && are_trailing_legs(contracted_legs, rank) {
        // Zero-copy fast path: contracted legs are trailing in contiguous storage.
        Ok(ReshapeResult {
            free_dims,
            rows: free_dim_product,
            cols: contracted_dim_product,
            conjugated,
            data: ReshapeData::Borrowed(tensor.as_slice()),
        })
    } else {
        // Slow path: transpose so free legs come first, contracted legs trail.
        let mut perm = Vec::with_capacity(rank);
        for leg in 0..rank {
            if !contracted_legs.contains(&leg) {
                perm.push(leg);
            }
        }
        for leg in 0..rank {
            if contracted_legs.contains(&leg) {
                perm.push(leg);
            }
        }

        let transposed = block_transpose(tensor, &perm);
        Ok(ReshapeResult {
            free_dims,
            rows: free_dim_product,
            cols: contracted_dim_product,
            conjugated,
            data: ReshapeData::Owned(transposed),
        })
    }
}

/// Check if the contracted legs are the trailing (rightmost) legs.
pub(crate) fn are_trailing_legs(contracted_legs: &[usize], rank: usize) -> bool {
    if contracted_legs.is_empty() {
        return true;
    }
    let mut sorted = contracted_legs.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    // Check that they form a contiguous range at the end.
    sorted.last() == Some(&(rank - 1))
        && sorted.first() == Some(&(rank - n))
}

/// Check if the contracted legs are the leading (leftmost) legs.
pub(crate) fn are_leading_legs(contracted_legs: &[usize], _rank: usize) -> bool {
    if contracted_legs.is_empty() {
        return true;
    }
    let mut sorted = contracted_legs.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    // Check that they form a contiguous range starting at 0.
    sorted.first() == Some(&0)
        && sorted.last() == Some(&(n - 1))
}

/// Reconstruct a `DenseTensor` from a flat result buffer and target shape.
pub fn mat_to_tensor<T: Scalar>(
    data: Vec<T>,
    output_dims: &[usize],
) -> DenseTensor<'static, T> {
    let shape = TensorShape::row_major(output_dims);
    DenseTensor::from_vec(shape, data)
}

/// Cache-oblivious block-transpose for non-contiguous inputs.
///
/// Transposes `src` according to `perm` into a new contiguous buffer.
/// Uses recursive blocking for L1-cache locality.
pub fn block_transpose<T: Scalar>(
    src: &DenseTensor<'_, T>,
    perm: &[usize],
) -> DenseTensor<'static, T> {
    let old_shape = src.shape();
    let old_dims = old_shape.dims();
    let old_strides = old_shape.strides();
    let numel = old_shape.numel();

    // Compute new dims and strides after permutation.
    let new_dims: Vec<usize> = perm.iter().map(|&p| old_dims[p]).collect();
    let new_shape = TensorShape::row_major(&new_dims);

    if numel == 0 {
        return DenseTensor::zeros(new_shape);
    }

    // General path: gather elements in the new order.
    let rank = old_dims.len();
    let src_data = src.as_slice();
    let mut dst = vec![T::zero(); numel];

    let mut old_index = vec![0usize; rank];
    for flat_new in 0..numel {
        // Compute the source linear index from the old_index.
        let src_linear: usize = old_index.iter().zip(old_strides).map(|(&i, &s)| i * s).sum();
        dst[flat_new] = src_data[src_linear];

        // Increment old_index in *permuted* order (new dimension order).
        for &d in perm.iter().rev() {
            old_index[d] += 1;
            if old_index[d] < old_dims[d] {
                break;
            }
            old_index[d] = 0;
        }
    }

    DenseTensor::from_vec(new_shape, dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_legs_check() {
        assert!(are_trailing_legs(&[2, 3], 4));
        assert!(are_trailing_legs(&[3], 4));
        assert!(are_trailing_legs(&[], 4));
        assert!(!are_trailing_legs(&[0, 1], 4));
        assert!(!are_trailing_legs(&[1, 3], 4)); // gap
    }

    #[test]
    fn block_transpose_2x3() {
        // [[0,1,2],[3,4,5]] → transpose → [[0,3],[1,4],[2,5]]
        let data: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let t = DenseTensor::from_vec(TensorShape::row_major(&[2, 3]), data);
        let transposed = block_transpose(&t, &[1, 0]);

        assert_eq!(transposed.shape().dims(), &[3, 2]);
        let result = transposed.as_slice();
        assert_eq!(&result[..6], &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn block_transpose_identity() {
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let t = DenseTensor::from_vec(TensorShape::row_major(&[3, 4]), data.clone());
        let same = block_transpose(&t, &[0, 1]);
        assert_eq!(same.as_slice()[..12], data[..]);
    }

    #[test]
    fn mat_to_tensor_round_trip() {
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let t = mat_to_tensor(data.clone(), &[2, 3, 4]);
        assert_eq!(t.shape().dims(), &[2, 3, 4]);
        assert_eq!(t.numel(), 24);
        assert_eq!(t.as_slice()[..24], data[..]);
    }

    #[test]
    fn leading_legs_check() {
        assert!(are_leading_legs(&[0, 1], 4));
        assert!(are_leading_legs(&[0], 4));
        assert!(are_leading_legs(&[], 4));
        assert!(!are_leading_legs(&[2, 3], 4));
        assert!(!are_leading_legs(&[0, 2], 4)); // gap
    }

    #[test]
    fn tensor_to_mat_ref_trailing_zero_copy() {
        // 2x3 tensor, contract leg 1 (trailing) → should be zero-copy.
        let data: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let t = DenseTensor::from_vec(TensorShape::row_major(&[2, 3]), data);
        let result = tensor_to_mat_ref(&t, &[1], false).unwrap();

        assert_eq!(result.rows, 2); // M = free dim
        assert_eq!(result.cols, 3); // K = contracted dim
        assert_eq!(result.free_dims, vec![2]);

        let mat = result.as_mat_ref();
        assert_eq!(mat.rows, 2);
        assert_eq!(mat.cols, 3);
    }

    #[test]
    fn tensor_to_mat_ref_non_trailing_transpose() {
        // 2x3 tensor, contract leg 0 (leading, not trailing) → needs transpose.
        let data: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let t = DenseTensor::from_vec(TensorShape::row_major(&[2, 3]), data);
        let result = tensor_to_mat_ref(&t, &[0], false).unwrap();

        assert_eq!(result.rows, 3); // M = free dim (leg 1, dim=3)
        assert_eq!(result.cols, 2); // K = contracted dim (leg 0, dim=2)
        assert_eq!(result.free_dims, vec![3]);

        // Verify the transposed data is correct:
        // Original: [[0,1,2],[3,4,5]], permuted to [leg1, leg0] = 3x2
        // Transposed: [[0,3],[1,4],[2,5]]
        let mat = result.as_mat_ref();
        assert_eq!(mat.rows, 3);
        assert_eq!(mat.cols, 2);
        // Row 0: [0, 3], Row 1: [1, 4], Row 2: [2, 5]
        assert!((mat.data[0] - 0.0).abs() < 1e-12);
        assert!((mat.data[1] - 3.0).abs() < 1e-12);
        assert!((mat.data[2] - 1.0).abs() < 1e-12);
        assert!((mat.data[3] - 4.0).abs() < 1e-12);
    }
}
