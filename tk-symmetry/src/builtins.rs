//! Built-in quantum number types: `U1`, `Z2`, `U1Z2`, `U1Wide`.

use crate::quantum_number::{BitPackable, QuantumNumber};

// ---------------------------------------------------------------------------
// U1 — Charge / Particle-Number Conservation
// ---------------------------------------------------------------------------

/// U(1) additive quantum number (e.g., particle number, total Sz).
/// Range: -128..=127 (8-bit signed, packed as u8 wrapping arithmetic).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct U1(pub i32);

impl QuantumNumber for U1 {
    fn identity() -> Self {
        U1(0)
    }

    fn fuse(&self, other: &Self) -> Self {
        U1(self.0 + other.0)
    }

    fn dual(&self) -> Self {
        U1(-self.0)
    }
}

impl BitPackable for U1 {
    const BIT_WIDTH: usize = 8;

    #[inline(always)]
    fn pack(&self) -> u64 {
        (self.0 as u8) as u64
    }

    #[inline(always)]
    fn unpack(bits: u64) -> Self {
        U1(((bits & 0xFF) as u8) as i8 as i32)
    }
}

// ---------------------------------------------------------------------------
// Z2 — Parity Conservation
// ---------------------------------------------------------------------------

/// Z₂ parity quantum number (e.g., fermion parity: even/odd electron count).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Z2(pub bool);

impl QuantumNumber for Z2 {
    fn identity() -> Self {
        Z2(false)
    }

    fn fuse(&self, other: &Self) -> Self {
        Z2(self.0 ^ other.0)
    }

    fn dual(&self) -> Self {
        *self // Z₂ is self-dual
    }
}

impl BitPackable for Z2 {
    const BIT_WIDTH: usize = 1;

    #[inline(always)]
    fn pack(&self) -> u64 {
        self.0 as u64
    }

    #[inline(always)]
    fn unpack(bits: u64) -> Self {
        Z2(bits & 1 == 1)
    }
}

// ---------------------------------------------------------------------------
// U1Z2 — Composite Symmetry
// ---------------------------------------------------------------------------

/// Product symmetry: U(1) charge ⊗ Z₂ parity.
/// Common in fermionic Hubbard models where both particle number
/// and fermion parity are conserved.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct U1Z2(pub U1, pub Z2);

impl QuantumNumber for U1Z2 {
    fn identity() -> Self {
        U1Z2(U1::identity(), Z2::identity())
    }

    fn fuse(&self, other: &Self) -> Self {
        U1Z2(self.0.fuse(&other.0), self.1.fuse(&other.1))
    }

    fn dual(&self) -> Self {
        U1Z2(self.0.dual(), self.1.dual())
    }
}

impl BitPackable for U1Z2 {
    const BIT_WIDTH: usize = 9; // 8 bits U1 + 1 bit Z2

    #[inline(always)]
    fn pack(&self) -> u64 {
        self.0.pack() | (self.1.pack() << U1::BIT_WIDTH)
    }

    #[inline(always)]
    fn unpack(bits: u64) -> Self {
        U1Z2(U1::unpack(bits), Z2::unpack(bits >> U1::BIT_WIDTH))
    }
}

// ---------------------------------------------------------------------------
// U1Wide — Extended Range U(1)
// ---------------------------------------------------------------------------

/// U(1) with 16-bit packing for systems with more than 127 sites.
/// Supports charges -32768..=+32767.
/// Uses u128 sector keys; see `PackedSectorKey128`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct U1Wide(pub i32);

impl QuantumNumber for U1Wide {
    fn identity() -> Self {
        U1Wide(0)
    }

    fn fuse(&self, other: &Self) -> Self {
        U1Wide(self.0 + other.0)
    }

    fn dual(&self) -> Self {
        U1Wide(-self.0)
    }
}

impl BitPackable for U1Wide {
    const BIT_WIDTH: usize = 16;

    #[inline(always)]
    fn pack(&self) -> u64 {
        (self.0 as u16) as u64
    }

    #[inline(always)]
    fn unpack(bits: u64) -> Self {
        U1Wide(((bits & 0xFFFF) as u16) as i16 as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- U1 tests --

    #[test]
    fn u1_fuse_identity() {
        assert_eq!(U1(3).fuse(&U1::identity()), U1(3));
    }

    #[test]
    fn u1_fuse_dual_is_identity() {
        for n in -128..=127 {
            assert_eq!(U1(n).fuse(&U1(n).dual()), U1::identity());
        }
    }

    #[test]
    fn u1_pack_round_trip() {
        for n in -128..=127 {
            assert_eq!(U1::unpack(U1(n).pack()), U1(n));
        }
    }

    // -- Z2 tests --

    #[test]
    fn z2_fuse_xor() {
        assert_eq!(Z2(true).fuse(&Z2(true)), Z2(false));
        assert_eq!(Z2(true).fuse(&Z2(false)), Z2(true));
        assert_eq!(Z2(false).fuse(&Z2(false)), Z2(false));
    }

    #[test]
    fn z2_self_dual() {
        assert_eq!(Z2(true).dual(), Z2(true));
        assert_eq!(Z2(false).dual(), Z2(false));
    }

    // -- U1Z2 tests --

    #[test]
    fn u1z2_pack_round_trip() {
        for n in -128..=127i32 {
            for &p in &[false, true] {
                let q = U1Z2(U1(n), Z2(p));
                assert_eq!(U1Z2::unpack(q.pack()), q);
            }
        }
    }

    // -- U1Wide tests --

    #[test]
    fn u1wide_pack_round_trip() {
        for n in [-32768, -1, 0, 1, 127, 128, 32767] {
            assert_eq!(U1Wide::unpack(U1Wide(n).pack()), U1Wide(n));
        }
    }
}
