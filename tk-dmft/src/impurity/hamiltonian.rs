//! AIM Hamiltonian construction via tk-dsl OpSum.
//!
//! Constructs the Anderson Impurity Model Hamiltonian in chain geometry:
//!
//!   H = epsilon_imp * n_total(0) + U * n_up(0) * n_down(0)
//!     + Sum_k epsilon_k * n_total(k)
//!     + Sum_{k,sigma} V_k * (c^dag_{sigma}(0) * c_{sigma}(k) + h.c.)
//!
//! Site ordering: impurity at site 0, bath sites at 1..=n_bath.
//! Basis per site: |0>, |up>, |down>, |up,down> (d=4).

use smallvec::SmallVec;
use tk_core::{DenseTensor, Scalar, TensorShape};
use tk_dsl::opterm::{OpProduct, OpTerm, ScaledOpProduct};
use tk_dsl::operators::{CustomOp, FermionOp, SiteOperator};
use tk_dsl::OpSum;

use crate::impurity::AndersonImpurityModel;

/// Helper: create a ScaledOpProduct with one operator on one site.
fn single_site_term<T: Scalar>(
    coeff: T,
    op: impl Into<SiteOperator<T>>,
    site: usize,
) -> ScaledOpProduct<T> {
    ScaledOpProduct {
        coeff,
        product: OpProduct {
            factors: SmallVec::from_elem(
                OpTerm {
                    operator: op.into(),
                    site,
                },
                1,
            ),
        },
    }
}

/// Helper: create a ScaledOpProduct with operators on two sites.
fn two_site_term<T: Scalar>(
    coeff: T,
    op1: impl Into<SiteOperator<T>>,
    site1: usize,
    op2: impl Into<SiteOperator<T>>,
    site2: usize,
) -> ScaledOpProduct<T> {
    let mut factors = SmallVec::new();
    factors.push(OpTerm {
        operator: op1.into(),
        site: site1,
    });
    factors.push(OpTerm {
        operator: op2.into(),
        site: site2,
    });
    ScaledOpProduct {
        coeff,
        product: OpProduct { factors },
    }
}

/// Build a CustomOp for n_up * n_down on a single site.
///
/// In the 4-state basis |0>, |up>, |down>, |up,down>:
/// n_up * n_down = diag(0, 0, 0, 1)
fn nup_ndn_operator<T: Scalar>() -> CustomOp<T> {
    let z = T::zero();
    let o = T::one();
    // 4x4 matrix in row-major: only (3,3) element is 1
    let data = vec![
        z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, o,
    ];
    let shape = TensorShape::row_major(&[4, 4]);
    let matrix = DenseTensor::from_vec(shape, data);
    CustomOp {
        matrix,
        name: "NupNdn".into(),
    }
}

/// Build the AIM Hamiltonian as an `OpSum` in chain geometry.
///
/// This constructs the operator sum that can be compiled into an MPO
/// by `tk-dmrg`'s `MpoCompiler`.
///
/// # Parameters
/// - `aim`: the Anderson Impurity Model with current bath parameters
///
/// # Returns
/// An `OpSum<T>` representing the full AIM Hamiltonian.
pub fn build_aim_chain_hamiltonian<T: Scalar>(aim: &AndersonImpurityModel<T>) -> OpSum<T>
where
    T::Real: Into<f64> + From<f64>,
{
    let mut h = OpSum::<T>::new();
    let n_bath = aim.bath.n_bath;

    // --- Impurity terms at site 0 ---

    // epsilon_imp * (n_up + n_down) = epsilon_imp * n_total
    let eps_imp = T::from_real(aim.epsilon_imp);
    h += single_site_term(eps_imp, FermionOp::Ntotal, 0);

    // U * n_up * n_down (double occupancy penalty)
    let u_val = T::from_real(aim.u);
    h += single_site_term(u_val, nup_ndn_operator::<T>(), 0);

    // --- Bath on-site energies ---
    for k in 0..n_bath {
        let site = k + 1;
        let eps_k = T::from_real(aim.bath.epsilon[k]);
        h += single_site_term(eps_k, FermionOp::Ntotal, site);
    }

    // --- Hybridization: V_k * (c^dag_sigma(0) c_sigma(k) + h.c.) for each spin ---
    for k in 0..n_bath {
        let site = k + 1;
        let v_k = aim.bath.v[k];

        // Spin-up hopping: V_k * c^dag_up(0) c_up(k) + h.c.
        h += two_site_term(v_k, FermionOp::CdagUp, 0, FermionOp::CUp, site);
        h += two_site_term(v_k.conj(), FermionOp::CdagUp, site, FermionOp::CUp, 0);

        // Spin-down hopping: V_k * c^dag_dn(0) c_dn(k) + h.c.
        h += two_site_term(v_k, FermionOp::CdagDn, 0, FermionOp::CDn, site);
        h += two_site_term(v_k.conj(), FermionOp::CdagDn, site, FermionOp::CDn, 0);
    }

    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impurity::bath::BathParameters;

    #[test]
    fn test_aim_hamiltonian_term_count() {
        let aim = AndersonImpurityModel::<f64> {
            u: 4.0,
            epsilon_imp: -2.0,
            beta: None,
            bath: BathParameters {
                epsilon: vec![-1.0, 0.0, 1.0],
                v: vec![0.5, 0.5, 0.5],
                n_bath: 3,
            },
            impurity_local_dim: 4,
            bath_local_dim: 4,
        };

        let h = build_aim_chain_hamiltonian(&aim);
        // Terms:
        //   1 (epsilon_imp * Ntotal) + 1 (U * NupNdn) = 2 impurity terms
        //   3 (bath energies)
        //   3 * 4 = 12 hopping terms (2 spin channels x (forward + h.c.) x 3 bath sites)
        // Total = 2 + 3 + 12 = 17
        assert_eq!(h.n_terms(), 17);
    }

    #[test]
    fn test_aim_hamiltonian_zero_bath() {
        let aim = AndersonImpurityModel::<f64> {
            u: 4.0,
            epsilon_imp: -2.0,
            beta: None,
            bath: BathParameters {
                epsilon: vec![],
                v: vec![],
                n_bath: 0,
            },
            impurity_local_dim: 4,
            bath_local_dim: 4,
        };

        let h = build_aim_chain_hamiltonian(&aim);
        // Only impurity terms: epsilon_imp * Ntotal + U * NupNdn = 2
        assert_eq!(h.n_terms(), 2);
    }

    #[test]
    fn test_nup_ndn_matrix() {
        let op = nup_ndn_operator::<f64>();
        let mat = op.matrix.as_slice();
        // diag(0, 0, 0, 1)
        assert!((mat[0] - 0.0).abs() < 1e-12);
        assert!((mat[5] - 0.0).abs() < 1e-12);
        assert!((mat[10] - 0.0).abs() < 1e-12);
        assert!((mat[15] - 1.0).abs() < 1e-12);
    }
}
