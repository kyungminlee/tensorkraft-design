//! The `QuantumNumber` and `BitPackable` traits.

use std::fmt::Debug;
use std::hash::Hash;

/// Abstract quantum number: the label attached to one leg of a symmetric tensor.
///
/// A quantum number encodes a conservation law. `fuse` combines two quantum
/// numbers on adjacent legs; `dual` gives the outgoing quantum number when
/// an incoming one is fixed by the flux rule.
///
/// All implementations must satisfy the group axioms:
///   - `q.fuse(Q::identity()) == q` (identity element)
///   - `q.fuse(q.dual()) == Q::identity()` (inverse)
///   - `a.fuse(b.fuse(c)) == a.fuse(b).fuse(c)` (associativity)
pub trait QuantumNumber: Clone + Eq + Hash + Ord + Debug + Send + Sync + 'static {
    /// The group identity (additive zero, parity-even, trivial irrep, etc.).
    fn identity() -> Self;

    /// Group product / fusion: combine two quantum numbers into one.
    /// For U(1): addition. For Z₂: XOR. For SU(2): tensor-product irrep
    /// (one-to-many — see `SU2Irrep::fuse_all` for the full decomposition).
    fn fuse(&self, other: &Self) -> Self;

    /// Group inverse / dual: the quantum number that cancels this one.
    /// For U(1): negation. For Z₂: identity (self-dual). For SU(2): same irrep.
    fn dual(&self) -> Self;
}

/// Extension of `QuantumNumber`: compresses the quantum number into a
/// fixed-width bitfield for register-speed sector lookup.
///
/// Only Abelian symmetries implement this trait. Non-Abelian symmetries
/// (SU(2)) use a separate SmallVec-keyed storage path.
pub trait BitPackable: QuantumNumber + Copy {
    /// Number of bits required to encode one quantum number.
    /// Must be a compile-time constant.
    const BIT_WIDTH: usize;

    /// Compress into the lower `BIT_WIDTH` bits of a `u64`.
    /// Implementors are responsible for lossless round-trip:
    ///   `Self::unpack(self.pack()) == *self`
    fn pack(&self) -> u64;

    /// Reconstruct from the lower `BIT_WIDTH` bits of a `u64`.
    fn unpack(bits: u64) -> Self;
}

/// Direction of a tensor leg for flux rule evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegDirection {
    Incoming,
    Outgoing,
}
