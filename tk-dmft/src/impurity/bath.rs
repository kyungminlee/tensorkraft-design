//! Bath parameters for the Anderson Impurity Model.

use num_complex::Complex;
use num_traits::{One, Zero};
use tk_core::Scalar;

/// Discretized bath parameters for the Anderson Impurity Model.
///
/// Represents a finite set of `n_bath` non-interacting bath orbitals
/// coupled to the impurity. In the star geometry, each bath site `k`
/// has an on-site energy `epsilon[k]` and hybridization amplitude `v[k]`.
///
/// The non-interacting hybridization function is:
///   Delta(omega) = Sum_k |V_k|^2 / (omega - epsilon_k + i*0+)
///
/// Bath parameters are the mutable state updated at each DMFT iteration.
#[derive(Clone, Debug)]
pub struct BathParameters<T: Scalar> {
    /// On-site bath energies epsilon_k. Length = `n_bath`.
    pub epsilon: Vec<T::Real>,
    /// Hybridization amplitudes V_k (coupling impurity to bath site k).
    /// Length = `n_bath`.
    pub v: Vec<T>,
    /// Number of bath sites.
    pub n_bath: usize,
}

impl<T: Scalar> BathParameters<T>
where
    T::Real: Into<f64> + From<f64>,
{
    /// Construct uniform bath: `n_bath` sites with energies linearly spaced
    /// in `[-bandwidth/2, bandwidth/2]` and uniform hybridization `v0`.
    ///
    /// Used as an initial guess before the first DMFT iteration.
    pub fn uniform(n_bath: usize, bandwidth: T::Real, v0: T) -> Self {
        let half_bw = bandwidth / T::Real::from(2.0);
        let epsilon = if n_bath <= 1 {
            vec![T::Real::zero(); n_bath]
        } else {
            let n = n_bath as f64;
            (0..n_bath)
                .map(|k| {
                    let frac = k as f64 / (n - 1.0);
                    let val: f64 = -half_bw.into() + frac * bandwidth.into();
                    T::Real::from(val)
                })
                .collect()
        };
        let v = vec![v0; n_bath];
        Self { epsilon, v, n_bath }
    }

    /// Compute the discretized hybridization function at frequency grid `omega`.
    ///
    /// Delta(omega) = Sum_k |V_k|^2 / (omega - epsilon_k + i*broadening)
    ///
    /// # Parameters
    /// - `omega`: frequency grid points
    /// - `broadening`: Lorentzian broadening delta replacing the i*0+ regulator
    pub fn hybridization_function(
        &self,
        omega: &[T::Real],
        broadening: T::Real,
    ) -> Vec<Complex<f64>> {
        omega
            .iter()
            .map(|&w| {
                let mut delta = Complex::<f64>::zero();
                let w_f64: f64 = w.into();
                let broad_f64: f64 = broadening.into();
                for k in 0..self.n_bath {
                    let eps_f64: f64 = self.epsilon[k].into();
                    let v_abs_sq: f64 = self.v[k].abs_sq().into();
                    let denom = Complex::new(w_f64 - eps_f64, broad_f64);
                    delta = delta + v_abs_sq / denom;
                }
                delta
            })
            .collect()
    }

    /// Compute the relative L-infinity distance
    /// ||Delta_self(omega) - Delta_other(omega)||_inf / ||Delta_other(omega)||_inf
    /// for convergence assessment of the DMFT self-consistency loop.
    ///
    /// # Panics
    /// Panics if `omega.len()` is zero.
    pub fn hybridization_distance(
        &self,
        other: &Self,
        omega: &[T::Real],
        broadening: T::Real,
    ) -> f64 {
        assert!(!omega.is_empty(), "omega must not be empty");
        let delta_self = self.hybridization_function(omega, broadening);
        let delta_other = other.hybridization_function(omega, broadening);

        let mut max_diff: f64 = 0.0;
        let mut max_ref: f64 = 0.0;
        for (ds, dr) in delta_self.iter().zip(delta_other.iter()) {
            let diff = (ds - dr).norm();
            let refn = dr.norm();
            if diff > max_diff {
                max_diff = diff;
            }
            if refn > max_ref {
                max_ref = refn;
            }
        }
        if max_ref < f64::EPSILON {
            return max_diff;
        }
        max_diff / max_ref
    }

    /// Mix two sets of bath parameters: result = (1-alpha)*self + alpha*other.
    /// Used for linear mixing in the DMFT self-consistency loop.
    pub fn linear_mix(&self, other: &Self, alpha: T::Real) -> Self
    where
        T::Real: Copy,
    {
        assert_eq!(self.n_bath, other.n_bath, "bath sizes must match for mixing");
        let one_minus_alpha = T::Real::one() - alpha;
        let epsilon = self
            .epsilon
            .iter()
            .zip(other.epsilon.iter())
            .map(|(&a, &b)| one_minus_alpha * a + alpha * b)
            .collect();

        // For hybridization amplitudes, mix the absolute values and keep the phase.
        // In the real case (T = f64) this is simple linear interpolation.
        let v = self
            .v
            .iter()
            .zip(other.v.iter())
            .map(|(&a, &b)| {
                let a_scaled = T::from_real(one_minus_alpha) * a;
                let b_scaled = T::from_real(alpha) * b;
                a_scaled + b_scaled
            })
            .collect();

        Self {
            epsilon,
            v,
            n_bath: self.n_bath,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_bath() {
        let bath: BathParameters<f64> = BathParameters::uniform(6, 10.0, 1.0);
        assert_eq!(bath.n_bath, 6);
        assert_eq!(bath.epsilon.len(), 6);
        assert_eq!(bath.v.len(), 6);
        // First energy should be -5.0, last should be 5.0
        assert!((bath.epsilon[0] - (-5.0)).abs() < 1e-12);
        assert!((bath.epsilon[5] - 5.0).abs() < 1e-12);
        // All V_k should be 1.0
        for &vk in &bath.v {
            assert!((vk - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_uniform_bath_single_site() {
        let bath: BathParameters<f64> = BathParameters::uniform(1, 10.0, 1.0);
        assert_eq!(bath.n_bath, 1);
        assert!((bath.epsilon[0] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_hybridization_function_known_result() {
        // Two-site bath with known parameters
        let bath = BathParameters::<f64> {
            epsilon: vec![-1.0, 1.0],
            v: vec![0.5, 0.5],
            n_bath: 2,
        };
        let omega = vec![0.0];
        let broadening = 0.1;
        let delta = bath.hybridization_function(&omega, broadening);
        // Delta(0) = |0.5|^2 / (0 - (-1) + 0.1i) + |0.5|^2 / (0 - 1 + 0.1i)
        //          = 0.25 / (1 + 0.1i) + 0.25 / (-1 + 0.1i)
        let expected = Complex::new(0.25, 0.0) / Complex::new(1.0, 0.1)
            + Complex::new(0.25, 0.0) / Complex::new(-1.0, 0.1);
        assert!((delta[0] - expected).norm() < 1e-12);
    }

    #[test]
    fn test_hybridization_distance_zero_for_same() {
        let bath: BathParameters<f64> = BathParameters::uniform(4, 10.0, 1.0);
        let omega: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.2).collect();
        let dist = bath.hybridization_distance(&bath, &omega, 0.05);
        assert!(dist < 1e-12);
    }

    #[test]
    fn test_linear_mix() {
        let b1: BathParameters<f64> = BathParameters {
            epsilon: vec![-1.0, 1.0],
            v: vec![1.0, 1.0],
            n_bath: 2,
        };
        let b2: BathParameters<f64> = BathParameters {
            epsilon: vec![-2.0, 2.0],
            v: vec![0.5, 0.5],
            n_bath: 2,
        };
        let mixed = b1.linear_mix(&b2, 0.3);
        // epsilon: 0.7*(-1) + 0.3*(-2) = -1.3, 0.7*1 + 0.3*2 = 1.3
        assert!((mixed.epsilon[0] - (-1.3)).abs() < 1e-12);
        assert!((mixed.epsilon[1] - 1.3).abs() < 1e-12);
        // v: 0.7*1.0 + 0.3*0.5 = 0.85
        assert!((mixed.v[0] - 0.85).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "omega must not be empty")]
    fn test_hybridization_distance_empty_omega() {
        let bath: BathParameters<f64> = BathParameters::uniform(4, 10.0, 1.0);
        bath.hybridization_distance(&bath, &[], 0.05);
    }
}
