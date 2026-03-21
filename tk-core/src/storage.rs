//! Contiguous memory buffers and Copy-on-Write storage.

use crate::scalar::Scalar;

/// Owned, contiguous, 1-D memory buffer.
///
/// Invariant: `data.len() == shape.numel()` for any `DenseTensor` using
/// this storage. `TensorStorage` has no shape knowledge — that lives
/// exclusively in `TensorShape`.
#[derive(Clone, Debug)]
pub struct TensorStorage<T: Scalar> {
    data: Vec<T>,
}

impl<T: Scalar> TensorStorage<T> {
    /// Create a zero-filled storage of `n` elements.
    pub fn zeros(n: usize) -> Self {
        TensorStorage {
            data: vec![T::zero(); n],
        }
    }

    /// Create from an existing `Vec`.
    pub fn from_vec(data: Vec<T>) -> Self {
        TensorStorage { data }
    }

    /// Read-only slice access.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable slice access.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the storage is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Copy-on-Write wrapper for tensor storage.
///
/// Borrows a view when possible; materializes into owned storage on demand.
/// Shape operations (permute, reshape, slice) always return `Borrowed`.
/// Data is cloned into `Owned` only when strict contiguity is required
/// (e.g., BLAS kernel input with non-unit strides).
#[derive(Debug)]
pub enum TensorCow<'a, T: Scalar> {
    /// Zero-copy view: points into an existing buffer (arena or heap).
    Borrowed(&'a TensorStorage<T>),
    /// Heap-allocated owned data.
    Owned(TensorStorage<T>),
}

impl<'a, T: Scalar> TensorCow<'a, T> {
    /// Read-only slice access regardless of variant.
    pub fn as_slice(&self) -> &[T] {
        match self {
            TensorCow::Borrowed(s) => s.as_slice(),
            TensorCow::Owned(s) => s.as_slice(),
        }
    }

    /// Clone data into owned storage if borrowed.
    pub fn into_owned(self) -> TensorStorage<T> {
        match self {
            TensorCow::Owned(s) => s,
            TensorCow::Borrowed(s) => s.clone(),
        }
    }

    /// Returns `true` if this is the `Owned` variant.
    pub fn is_owned(&self) -> bool {
        matches!(self, TensorCow::Owned(_))
    }

    /// Returns `true` if this is the `Borrowed` variant.
    pub fn is_borrowed(&self) -> bool {
        matches!(self, TensorCow::Borrowed(_))
    }
}

impl<T: Scalar> Clone for TensorCow<'_, T> {
    fn clone(&self) -> Self {
        match self {
            TensorCow::Borrowed(s) => TensorCow::Borrowed(s),
            TensorCow::Owned(s) => TensorCow::Owned(s.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensorcow_borrowed_no_clone() {
        let storage = TensorStorage::<f64>::zeros(10);
        let cow = TensorCow::Borrowed(&storage);
        assert!(cow.is_borrowed());
        // Accessing slice doesn't clone
        assert_eq!(cow.as_slice().len(), 10);
        assert!(cow.is_borrowed());
    }

    #[test]
    fn tensorcow_into_owned_clones() {
        let storage = TensorStorage::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let cow = TensorCow::Borrowed(&storage);
        let owned = cow.into_owned();
        // The owned storage is a separate allocation
        assert_eq!(owned.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn tensorcow_owned_passthrough() {
        let cow = TensorCow::Owned(TensorStorage::from_vec(vec![1.0_f64, 2.0]));
        assert!(cow.is_owned());
        let owned = cow.into_owned();
        assert_eq!(owned.as_slice(), &[1.0, 2.0]);
    }
}
