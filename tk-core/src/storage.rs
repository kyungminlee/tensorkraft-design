//! Contiguous memory buffers with Copy-on-Write semantics.
//!
//! `TensorStorage` is the single storage type for all tensors. It is either:
//! - `Owned(Vec<T>)` — heap-allocated, mutable, can be moved freely
//! - `Borrowed(&'a [T])` — zero-copy view into arena or other buffer, immutable
//!
//! Shape operations (permute, reshape, slice) always produce `Borrowed` views.
//! Data is cloned into `Owned` only when strict ownership is required
//! (e.g., persisting past an arena reset, or mutating in-place).

use crate::scalar::Scalar;

/// Contiguous memory buffer with Copy-on-Write semantics.
///
/// `TensorStorage` has no shape knowledge — that lives exclusively in
/// `TensorShape`. This strict separation means shape-manipulation
/// operations never touch the data buffer.
#[derive(Debug)]
pub enum TensorStorage<'a, T: Scalar> {
    /// Heap-allocated owned data. Mutable and freely movable.
    Owned(Vec<T>),
    /// Zero-copy view into an existing buffer (arena, another tensor, etc.).
    /// Immutable — must call `into_owned()` before mutation.
    Borrowed(&'a [T]),
}

impl<'a, T: Scalar> TensorStorage<'a, T> {
    /// Create a zero-filled owned storage of `n` elements.
    pub fn zeros(n: usize) -> TensorStorage<'static, T> {
        TensorStorage::Owned(vec![T::zero(); n])
    }

    /// Create owned storage from an existing `Vec`.
    pub fn from_vec(data: Vec<T>) -> TensorStorage<'static, T> {
        TensorStorage::Owned(data)
    }

    /// Create a borrowed view from a slice.
    pub fn from_slice(data: &'a [T]) -> Self {
        TensorStorage::Borrowed(data)
    }

    /// Read-only slice access regardless of variant.
    pub fn as_slice(&self) -> &[T] {
        match self {
            TensorStorage::Owned(v) => v,
            TensorStorage::Borrowed(s) => s,
        }
    }

    /// Mutable slice access.
    ///
    /// # Panics
    /// Panics if the storage is `Borrowed`.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            TensorStorage::Owned(v) => v,
            TensorStorage::Borrowed(_) => {
                panic!("cannot mutably access borrowed storage; call .into_owned() first")
            }
        }
    }

    /// Number of elements in the buffer.
    pub fn len(&self) -> usize {
        match self {
            TensorStorage::Owned(v) => v.len(),
            TensorStorage::Borrowed(s) => s.len(),
        }
    }

    /// Whether the storage is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clone data into an owned `Vec<T>`, consuming self.
    /// If already owned, returns the `Vec` without copying.
    pub fn into_owned_vec(self) -> Vec<T> {
        match self {
            TensorStorage::Owned(v) => v,
            TensorStorage::Borrowed(s) => s.to_vec(),
        }
    }

    /// Clone data into owned storage. If already owned, no-op.
    pub fn into_owned(self) -> TensorStorage<'static, T> {
        match self {
            TensorStorage::Owned(v) => TensorStorage::Owned(v),
            TensorStorage::Borrowed(s) => TensorStorage::Owned(s.to_vec()),
        }
    }

    /// Returns `true` if this is the `Owned` variant.
    pub fn is_owned(&self) -> bool {
        matches!(self, TensorStorage::Owned(_))
    }

    /// Returns `true` if this is the `Borrowed` variant.
    pub fn is_borrowed(&self) -> bool {
        matches!(self, TensorStorage::Borrowed(_))
    }

    /// Borrow the data as a `Borrowed` variant (regardless of current variant).
    /// The returned storage borrows from `self`.
    pub fn borrow(&self) -> TensorStorage<'_, T> {
        TensorStorage::Borrowed(self.as_slice())
    }
}

impl<T: Scalar> Clone for TensorStorage<'_, T> {
    fn clone(&self) -> Self {
        match self {
            TensorStorage::Owned(v) => TensorStorage::Owned(v.clone()),
            TensorStorage::Borrowed(s) => TensorStorage::Borrowed(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_zeros() {
        let s = TensorStorage::<f64>::zeros(10);
        assert!(s.is_owned());
        assert_eq!(s.len(), 10);
        assert!(s.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn storage_from_vec() {
        let s = TensorStorage::from_vec(vec![1.0_f64, 2.0, 3.0]);
        assert!(s.is_owned());
        assert_eq!(s.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn storage_from_slice() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let s = TensorStorage::from_slice(&data);
        assert!(s.is_borrowed());
        assert_eq!(s.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn storage_borrowed_no_clone_on_access() {
        let data = vec![0.0_f64; 10];
        let s = TensorStorage::from_slice(&data);
        assert!(s.is_borrowed());
        assert_eq!(s.as_slice().len(), 10);
        assert!(s.is_borrowed());
    }

    #[test]
    fn storage_into_owned_clones_borrowed() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let s = TensorStorage::from_slice(&data);
        let owned = s.into_owned();
        assert!(owned.is_owned());
        assert_eq!(owned.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn storage_into_owned_passthrough_owned() {
        let s = TensorStorage::from_vec(vec![1.0_f64, 2.0]);
        assert!(s.is_owned());
        let owned = s.into_owned();
        assert!(owned.is_owned());
        assert_eq!(owned.as_slice(), &[1.0, 2.0]);
    }

    #[test]
    fn storage_borrow() {
        let s = TensorStorage::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = s.borrow();
        assert!(b.is_borrowed());
        assert_eq!(b.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "cannot mutably access")]
    fn storage_borrowed_mut_panics() {
        let data = vec![1.0_f64];
        let mut s = TensorStorage::from_slice(&data);
        let _ = s.as_mut_slice();
    }
}
