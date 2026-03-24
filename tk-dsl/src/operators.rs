//! Strongly-typed operator enums for spin, fermion, and boson models.
//!
//! Typos are caught at compile time — no runtime string matching.

use smallstr::SmallString;
use tk_core::{DenseTensor, Scalar, TensorShape};

/// Spin-1/2 operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpinOp {
    /// S⁺ = |↑⟩⟨↓|
    SPlus,
    /// S⁻ = |↓⟩⟨↑|
    SMinus,
    /// Sᶻ = ½(|↑⟩⟨↑| − |↓⟩⟨↓|)
    Sz,
    /// Sˣ = ½(S⁺ + S⁻)
    Sx,
    /// Sʸ = −i/2(S⁺ − S⁻). Requires T = Complex.
    Sy,
    /// 2×2 identity.
    Identity,
}

impl SpinOp {
    pub const fn local_dim(self) -> usize {
        2
    }

    /// Produce the matrix elements in row-major order [d×d].
    pub fn matrix<T: Scalar>(self) -> Vec<T> {
        let zero = T::zero();
        let one = T::one();
        let half = T::from_real(<T::Real as num_traits::cast::NumCast>::from(0.5).unwrap());
        let neg_half = -half;

        match self {
            // [0, 1; 0, 0]
            SpinOp::SPlus => vec![zero, one, zero, zero],
            // [0, 0; 1, 0]
            SpinOp::SMinus => vec![zero, zero, one, zero],
            // [0.5, 0; 0, -0.5]
            SpinOp::Sz => vec![half, zero, zero, neg_half],
            // [0, 0.5; 0.5, 0]
            SpinOp::Sx => vec![zero, half, half, zero],
            // Sy = -i/2 * (S+ - S-)  →  [0, -i/2; i/2, 0]
            // For real types this is all zeros (Sy has no real representation).
            // For complex types, from_real_imag constructs the imaginary entries.
            SpinOp::Sy => {
                let real_zero = <T::Real as num_traits::cast::NumCast>::from(0.0).unwrap();
                let real_half = <T::Real as num_traits::cast::NumCast>::from(0.5).unwrap();
                let neg_i_half = T::from_real_imag(real_zero, -real_half); // -i/2
                let pos_i_half = T::from_real_imag(real_zero, real_half);  //  i/2
                vec![zero, neg_i_half, pos_i_half, zero]
            }
            // [1, 0; 0, 1]
            SpinOp::Identity => vec![one, zero, zero, one],
        }
    }

    /// Whether this operator preserves U(1) charge (Sz).
    pub fn preserves_u1(self) -> bool {
        matches!(self, SpinOp::Sz | SpinOp::Identity)
    }

    /// Change in Sz quantum number.
    pub fn delta_sz(self) -> i32 {
        match self {
            SpinOp::SPlus => 1,
            SpinOp::SMinus => -1,
            SpinOp::Sz | SpinOp::Sx | SpinOp::Sy | SpinOp::Identity => 0,
        }
    }

    /// Return the Hermitian adjoint operator.
    pub fn adjoint(self) -> Self {
        match self {
            SpinOp::SPlus => SpinOp::SMinus,
            SpinOp::SMinus => SpinOp::SPlus,
            SpinOp::Sz => SpinOp::Sz,
            SpinOp::Sx => SpinOp::Sx,
            SpinOp::Sy => SpinOp::Sy,
            SpinOp::Identity => SpinOp::Identity,
        }
    }
}

/// Spinful fermion operators (4-dimensional local Hilbert space: |0⟩, |↑⟩, |↓⟩, |↑↓⟩).
///
/// Jordan-Wigner strings are NOT embedded here; they are handled at MPO
/// construction time in `tk-dmrg`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FermionOp {
    /// c†_↑
    CdagUp,
    /// c_↑
    CUp,
    /// c†_↓
    CdagDn,
    /// c_↓
    CDn,
    /// n_↑ = c†_↑ c_↑
    Nup,
    /// n_↓ = c†_↓ c_↓
    Ndn,
    /// n_↑ + n_↓
    Ntotal,
    /// 4×4 identity.
    Identity,
}

impl FermionOp {
    pub const fn local_dim(self) -> usize {
        4
    }

    /// Produce the 4×4 matrix in row-major order.
    /// Basis ordering: |0⟩, |↑⟩, |↓⟩, |↑↓⟩
    pub fn matrix<T: Scalar>(self) -> Vec<T> {
        let z = T::zero();
        let o = T::one();

        match self {
            // c†_↑: |0⟩→|↑⟩, |↓⟩→|↑↓⟩
            // Row 0 (⟨0|): all zero
            // Row 1 (⟨↑|): col 0 = 1
            // Row 2 (⟨↓|): all zero
            // Row 3 (⟨↑↓|): col 2 = 1
            FermionOp::CdagUp => vec![
                z, z, z, z,
                o, z, z, z,
                z, z, z, z,
                z, z, o, z,
            ],
            // c_↑: transpose of c†_↑
            FermionOp::CUp => vec![
                z, o, z, z,
                z, z, z, z,
                z, z, z, o,
                z, z, z, z,
            ],
            // c†_↓: |0⟩→|↓⟩, |↑⟩→|↑↓⟩
            FermionOp::CdagDn => vec![
                z, z, z, z,
                z, z, z, z,
                o, z, z, z,
                z, o, z, z,
            ],
            // c_↓: transpose of c†_↓
            FermionOp::CDn => vec![
                z, z, o, z,
                z, z, z, o,
                z, z, z, z,
                z, z, z, z,
            ],
            // n_↑: diag(0, 1, 0, 1)
            FermionOp::Nup => vec![
                z, z, z, z,
                z, o, z, z,
                z, z, z, z,
                z, z, z, o,
            ],
            // n_↓: diag(0, 0, 1, 1)
            FermionOp::Ndn => vec![
                z, z, z, z,
                z, z, z, z,
                z, z, o, z,
                z, z, z, o,
            ],
            // n_total: diag(0, 1, 1, 2)
            FermionOp::Ntotal => {
                let two = o + o;
                vec![
                    z, z, z, z,
                    z, o, z, z,
                    z, z, o, z,
                    z, z, z, two,
                ]
            }
            // Identity
            FermionOp::Identity => vec![
                o, z, z, z,
                z, o, z, z,
                z, z, o, z,
                z, z, z, o,
            ],
        }
    }

    /// Total particle number change.
    pub fn delta_n(self) -> i32 {
        match self {
            FermionOp::CdagUp | FermionOp::CdagDn => 1,
            FermionOp::CUp | FermionOp::CDn => -1,
            _ => 0,
        }
    }

    /// Spin-up particle number change.
    pub fn delta_n_up(self) -> i32 {
        match self {
            FermionOp::CdagUp => 1,
            FermionOp::CUp => -1,
            _ => 0,
        }
    }

    /// Spin-down particle number change.
    pub fn delta_n_dn(self) -> i32 {
        match self {
            FermionOp::CdagDn => 1,
            FermionOp::CDn => -1,
            _ => 0,
        }
    }

    /// Return the Hermitian adjoint operator.
    pub fn adjoint(self) -> Self {
        match self {
            FermionOp::CdagUp => FermionOp::CUp,
            FermionOp::CUp => FermionOp::CdagUp,
            FermionOp::CdagDn => FermionOp::CDn,
            FermionOp::CDn => FermionOp::CdagDn,
            other => other, // Nup, Ndn, Ntotal, Identity are Hermitian
        }
    }
}

/// Bosonic operators. The local Hilbert space dimension is `n_max + 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BosonOp {
    /// b† (creation)
    BDag,
    /// b (annihilation)
    B,
    /// n = b†b (number)
    N,
    /// n(n-1) for Bose-Hubbard interaction U/2 · n(n-1)
    NPairInteraction,
    /// Identity
    Identity,
}

impl BosonOp {
    /// Produce the (n_max+1)×(n_max+1) matrix in row-major order.
    pub fn matrix<T: Scalar>(self, n_max: usize) -> Vec<T> {
        let d = n_max + 1;
        let mut mat = vec![T::zero(); d * d];

        match self {
            // b†: (b†)|n⟩ = √(n+1)|n+1⟩ → matrix[n+1, n] = √(n+1)
            BosonOp::BDag => {
                for n in 0..n_max {
                    let val = ((n + 1) as f64).sqrt();
                    mat[(n + 1) * d + n] = T::from_real(
                        <T::Real as num_traits::cast::NumCast>::from(val).unwrap(),
                    );
                }
            }
            // b: (b)|n⟩ = √n|n-1⟩ → matrix[n-1, n] = √n
            BosonOp::B => {
                for n in 1..=n_max {
                    let val = (n as f64).sqrt();
                    mat[(n - 1) * d + n] = T::from_real(
                        <T::Real as num_traits::cast::NumCast>::from(val).unwrap(),
                    );
                }
            }
            // n: diagonal with n
            BosonOp::N => {
                for n in 0..=n_max {
                    mat[n * d + n] = T::from_real(
                        <T::Real as num_traits::cast::NumCast>::from(n as f64).unwrap(),
                    );
                }
            }
            // n(n-1): diagonal with n*(n-1)
            BosonOp::NPairInteraction => {
                for n in 0..=n_max {
                    let val = (n * n.saturating_sub(1)) as f64;
                    mat[n * d + n] = T::from_real(
                        <T::Real as num_traits::cast::NumCast>::from(val).unwrap(),
                    );
                }
            }
            // Identity
            BosonOp::Identity => {
                for n in 0..=n_max {
                    mat[n * d + n] = T::one();
                }
            }
        }

        mat
    }

    /// Return the Hermitian adjoint operator.
    pub fn adjoint(self) -> Self {
        match self {
            BosonOp::BDag => BosonOp::B,
            BosonOp::B => BosonOp::BDag,
            other => other, // N, NPairInteraction, Identity are Hermitian
        }
    }
}

/// A user-defined operator represented as an explicit matrix.
#[derive(Debug)]
pub struct CustomOp<T: Scalar> {
    /// Square matrix [d, d] in row-major storage.
    pub matrix: DenseTensor<'static, T>,
    /// Human-readable name for diagnostics.
    pub name: SmallString<[u8; 32]>,
}

impl<T: Scalar> Clone for CustomOp<T> {
    fn clone(&self) -> Self {
        CustomOp {
            matrix: DenseTensor::from_vec(
                self.matrix.shape().clone(),
                self.matrix.as_slice().to_vec(),
            ),
            name: self.name.clone(),
        }
    }
}

impl<T: Scalar> CustomOp<T> {
    pub fn local_dim(&self) -> usize {
        self.matrix.shape().dims()[0]
    }
}

/// Unified operator type that can hold any standard or custom operator.
pub enum SiteOperator<T: Scalar> {
    Spin(SpinOp),
    Fermion(FermionOp),
    Boson { op: BosonOp, n_max: usize },
    Custom(CustomOp<T>),
}

impl<T: Scalar> SiteOperator<T> {
    pub fn local_dim(&self) -> usize {
        match self {
            SiteOperator::Spin(op) => op.local_dim(),
            SiteOperator::Fermion(op) => op.local_dim(),
            SiteOperator::Boson { n_max, .. } => n_max + 1,
            SiteOperator::Custom(op) => op.local_dim(),
        }
    }

    /// Produce the matrix elements as a flat Vec in row-major order.
    pub fn matrix(&self) -> Vec<T> {
        match self {
            SiteOperator::Spin(op) => op.matrix::<T>(),
            SiteOperator::Fermion(op) => op.matrix::<T>(),
            SiteOperator::Boson { op, n_max } => op.matrix::<T>(*n_max),
            SiteOperator::Custom(op) => op.matrix.as_slice().to_vec(),
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &str {
        match self {
            SiteOperator::Spin(op) => match op {
                SpinOp::SPlus => "SPlus",
                SpinOp::SMinus => "SMinus",
                SpinOp::Sz => "Sz",
                SpinOp::Sx => "Sx",
                SpinOp::Sy => "Sy",
                SpinOp::Identity => "SpinId",
            },
            SiteOperator::Fermion(op) => match op {
                FermionOp::CdagUp => "CdagUp",
                FermionOp::CUp => "CUp",
                FermionOp::CdagDn => "CdagDn",
                FermionOp::CDn => "CDn",
                FermionOp::Nup => "Nup",
                FermionOp::Ndn => "Ndn",
                FermionOp::Ntotal => "Ntotal",
                FermionOp::Identity => "FermionId",
            },
            SiteOperator::Boson { op, .. } => match op {
                BosonOp::BDag => "BDag",
                BosonOp::B => "B",
                BosonOp::N => "N",
                BosonOp::NPairInteraction => "NPair",
                BosonOp::Identity => "BosonId",
            },
            SiteOperator::Custom(op) => &op.name,
        }
    }

    /// Return the adjoint (Hermitian conjugate) of the operator.
    pub fn adjoint(&self) -> SiteOperator<T> {
        match self {
            SiteOperator::Spin(op) => SiteOperator::Spin(op.adjoint()),
            SiteOperator::Fermion(op) => SiteOperator::Fermion(op.adjoint()),
            SiteOperator::Boson { op, n_max } => SiteOperator::Boson {
                op: op.adjoint(),
                n_max: *n_max,
            },
            SiteOperator::Custom(custom) => {
                // Transpose + conjugate the matrix
                let d = custom.local_dim();
                let src = custom.matrix.as_slice();
                let mut adjoint_data = vec![T::zero(); d * d];
                for r in 0..d {
                    for c in 0..d {
                        adjoint_data[c * d + r] = src[r * d + c].conj();
                    }
                }
                let shape = TensorShape::row_major(&[d, d]);
                SiteOperator::Custom(CustomOp {
                    matrix: DenseTensor::from_vec(shape, adjoint_data),
                    name: custom.name.clone(),
                })
            }
        }
    }
}

impl<T: Scalar> Clone for SiteOperator<T> {
    fn clone(&self) -> Self {
        match self {
            SiteOperator::Spin(op) => SiteOperator::Spin(*op),
            SiteOperator::Fermion(op) => SiteOperator::Fermion(*op),
            SiteOperator::Boson { op, n_max } => SiteOperator::Boson {
                op: *op,
                n_max: *n_max,
            },
            SiteOperator::Custom(op) => SiteOperator::Custom(op.clone()),
        }
    }
}

impl<T: Scalar> std::fmt::Debug for SiteOperator<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SiteOperator::{}", self.name())
    }
}

impl<T: Scalar> From<SpinOp> for SiteOperator<T> {
    fn from(op: SpinOp) -> Self {
        SiteOperator::Spin(op)
    }
}

impl<T: Scalar> From<FermionOp> for SiteOperator<T> {
    fn from(op: FermionOp) -> Self {
        SiteOperator::Fermion(op)
    }
}

impl<T: Scalar> From<CustomOp<T>> for SiteOperator<T> {
    fn from(op: CustomOp<T>) -> Self {
        SiteOperator::Custom(op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spin_op_matrix_splus() {
        let mat = SpinOp::SPlus.matrix::<f64>();
        // [0, 1; 0, 0]
        assert_eq!(mat, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn spin_op_matrix_sz() {
        let mat = SpinOp::Sz.matrix::<f64>();
        assert!((mat[0] - 0.5).abs() < 1e-12);
        assert!((mat[3] + 0.5).abs() < 1e-12);
    }

    #[test]
    fn fermion_op_matrix_nup() {
        let mat = FermionOp::Nup.matrix::<f64>();
        // diag(0, 1, 0, 1)
        assert_eq!(mat[0], 0.0);
        assert_eq!(mat[5], 1.0);  // [1,1]
        assert_eq!(mat[10], 0.0); // [2,2]
        assert_eq!(mat[15], 1.0); // [3,3]
    }

    #[test]
    fn boson_op_matrix_bdag_n3() {
        let mat = BosonOp::BDag.matrix::<f64>(3);
        // b†: [n+1, n] = √(n+1)
        // d=4: matrix[1*4+0] = √1, [2*4+1] = √2, [3*4+2] = √3
        assert!((mat[4] - 1.0).abs() < 1e-12);
        assert!((mat[9] - 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((mat[14] - 3.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn custom_op_local_dim() {
        let data = vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let shape = TensorShape::row_major(&[3, 3]);
        let op = CustomOp {
            matrix: DenseTensor::from_vec(shape, data),
            name: SmallString::from("test3x3"),
        };
        assert_eq!(op.local_dim(), 3);
    }

    #[test]
    fn spin_adjoint_splus_sminus() {
        assert_eq!(SpinOp::SPlus.adjoint(), SpinOp::SMinus);
        assert_eq!(SpinOp::SMinus.adjoint(), SpinOp::SPlus);
        assert_eq!(SpinOp::Sz.adjoint(), SpinOp::Sz);
    }

    #[test]
    fn fermion_adjoint_creation() {
        assert_eq!(FermionOp::CdagUp.adjoint(), FermionOp::CUp);
        assert_eq!(FermionOp::CUp.adjoint(), FermionOp::CdagUp);
    }

    #[test]
    fn spin_op_sy_real_zeros() {
        // For real types, Sy has no representation — entries are zero
        // because from_real_imag ignores the imaginary part.
        let mat = SpinOp::Sy.matrix::<f64>();
        assert_eq!(mat, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn spin_op_sy_complex_correct() {
        use tk_core::C64;
        let mat = SpinOp::Sy.matrix::<C64>();
        // Sy = [0, -i/2; i/2, 0]
        let zero = C64::new(0.0, 0.0);
        let neg_i_half = C64::new(0.0, -0.5);
        let pos_i_half = C64::new(0.0, 0.5);
        assert_eq!(mat[0], zero);       // (0,0)
        assert_eq!(mat[1], neg_i_half); // (0,1)
        assert_eq!(mat[2], pos_i_half); // (1,0)
        assert_eq!(mat[3], zero);       // (1,1)
    }
}
