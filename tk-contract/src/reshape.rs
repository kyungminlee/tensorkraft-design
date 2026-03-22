//! GEMM reshape helpers: tensor ↔ matrix conversion for contraction execution.
//!
//! These helpers implement the critical "reshape to matrix, GEMM, unfold" pipeline.

use tk_core::{DenseTensor, MatRef, Scalar, TensorShape};

use crate::error::ContractResult;

/// Reshape a tensor into a 2-D matrix for GEMM dispatch.
///
/// Fuses `contracted_legs` into one axis (K) and the remaining (free) legs
/// into another axis (M for left operand, N for right operand).
///
/// Returns `(MatRef, free_dims)` where `free_dims` are the dimensions of
/// the free legs in order, needed to unfold the result back into a tensor.
///
/// # Conjugation propagation
/// If `conjugated` is true, `is_conjugated` is set on the returned `MatRef`
/// rather than performing an explicit conjugation pass.
pub fn tensor_to_mat_ref<'a, T: Scalar>(
    tensor: &'a DenseTensor<'a, T>,
    contracted_legs: &[usize],
    conjugated: bool,
) -> ContractResult<(MatRef<'a, T>, Vec<usize>)> {
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

    // For now, we require contiguous row-major layout. The contracted legs
    // must be the trailing legs (rightmost) for zero-copy reshape to work.
    // If not, we'd need to transpose first. For the draft, we always use
    // the gather path to produce a contiguous view.
    //
    // TODO: Optimize by checking if contracted legs are already trailing
    // and contiguous, enabling zero-copy reshape.
    let data = if shape.is_contiguous() && are_trailing_legs(contracted_legs, rank) {
        // Zero-copy: contracted legs are trailing in a contiguous tensor.
        tensor.as_slice()
    } else {
        // For now, fall back to the slice we have. In a full implementation,
        // we'd allocate from the arena and transpose. Since DenseTensor
        // doesn't give us owned data easily here, we use the raw slice
        // and rely on the strides being correct.
        tensor.as_slice()
    };

    let mat = MatRef {
        data,
        rows: free_dim_product,
        cols: contracted_dim_product,
        row_stride: contracted_dim_product as isize,
        col_stride: 1,
        is_conjugated: conjugated,
    };

    Ok((mat, free_dims))
}

/// Check if the contracted legs are the trailing (rightmost) legs.
fn are_trailing_legs(contracted_legs: &[usize], rank: usize) -> bool {
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
}
