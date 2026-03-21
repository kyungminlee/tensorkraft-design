//! Lazily populated cache of Clebsch-Gordan coefficients.

use std::sync::RwLock;

use hashbrown::HashMap;

use super::SU2Irrep;

/// Key for a single CG coefficient lookup.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CgKey {
    twice_ja: u32,
    twice_jb: u32,
    twice_jc: u32,
    twice_ma: i32,
    twice_mb: i32,
    twice_mc: i32,
}

/// Thread-safe, lazily populated cache of Clebsch-Gordan coefficients.
///
/// Computing CG coefficients from scratch is expensive; caching amortizes
/// the cost across the thousands of contractions in a DMRG sweep.
pub struct ClebschGordanCache {
    cache: RwLock<HashMap<CgKey, f64>>,
}

impl ClebschGordanCache {
    /// Construct an empty cache.
    pub fn new() -> Self {
        ClebschGordanCache {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Retrieve ⟨j_a, m_a; j_b, m_b | j_c, m_c⟩, computing and caching on miss.
    ///
    /// All angular momentum quantum numbers are given as "twice" values
    /// (e.g., j=1/2 → twice_j=1) to avoid floating-point arithmetic.
    pub fn get(
        &self,
        ja: SU2Irrep,
        twice_ma: i32,
        jb: SU2Irrep,
        twice_mb: i32,
        jc: SU2Irrep,
        twice_mc: i32,
    ) -> f64 {
        let key = CgKey {
            twice_ja: ja.twice_j,
            twice_jb: jb.twice_j,
            twice_jc: jc.twice_j,
            twice_ma,
            twice_mb,
            twice_mc,
        };

        // Fast path: read lock
        {
            let cache = self.cache.read().unwrap();
            if let Some(&val) = cache.get(&key) {
                return val;
            }
        }

        // Slow path: compute and insert
        let val = Self::compute_cg(
            ja.twice_j, twice_ma, jb.twice_j, twice_mb, jc.twice_j, twice_mc,
        );

        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(key, val);
        }

        val
    }

    /// Pre-populate the cache for all CG coefficients up to a given j_max.
    /// Call once before a DMRG sweep to amortize initialization cost.
    ///
    /// `j_max` is the maximum twice_j value to precompute (e.g., 20 for j=10).
    pub fn prefill(&self, twice_j_max: u32) {
        let mut cache = self.cache.write().unwrap();

        for twice_ja in 0..=twice_j_max {
            for twice_jb in 0..=twice_j_max {
                let twice_jc_min = (twice_ja as i32 - twice_jb as i32).unsigned_abs();
                let twice_jc_max = twice_ja + twice_jb;
                for twice_jc in (twice_jc_min..=twice_jc_max).step_by(2) {
                    let ma_range = -(twice_ja as i32)..=(twice_ja as i32);
                    for twice_ma in ma_range.step_by(2) {
                        let mb_range = -(twice_jb as i32)..=(twice_jb as i32);
                        for twice_mb in mb_range.step_by(2) {
                            let twice_mc = twice_ma + twice_mb;
                            if twice_mc.unsigned_abs() > twice_jc {
                                continue;
                            }
                            let key = CgKey {
                                twice_ja,
                                twice_jb,
                                twice_jc,
                                twice_ma,
                                twice_mb,
                                twice_mc,
                            };
                            if !cache.contains_key(&key) {
                                let val = Self::compute_cg(
                                    twice_ja, twice_ma, twice_jb, twice_mb, twice_jc, twice_mc,
                                );
                                cache.insert(key, val);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Compute a single Clebsch-Gordan coefficient using the Racah formula.
    ///
    /// ⟨j₁ m₁ ; j₂ m₂ | J M⟩
    /// All arguments are "twice" values.
    fn compute_cg(
        twice_j1: u32,
        twice_m1: i32,
        twice_j2: u32,
        twice_m2: i32,
        twice_j: u32,
        twice_m: i32,
    ) -> f64 {
        // Selection rules
        if twice_m1 + twice_m2 != twice_m {
            return 0.0;
        }
        if twice_m.unsigned_abs() > twice_j {
            return 0.0;
        }
        if twice_m1.unsigned_abs() > twice_j1 {
            return 0.0;
        }
        if twice_m2.unsigned_abs() > twice_j2 {
            return 0.0;
        }

        let j1_plus_j2 = twice_j1 + twice_j2;
        let j1_minus_j2 = (twice_j1 as i32 - twice_j2 as i32).unsigned_abs();

        // Triangle inequality
        if twice_j > j1_plus_j2 || twice_j < j1_minus_j2 {
            return 0.0;
        }
        // Parity check: j1 + j2 + J must be even (all twice values)
        if (twice_j1 + twice_j2 + twice_j) % 2 != 0 {
            return 0.0;
        }

        // Use the Racah formula (sum over k)
        // Convert to half-integer arithmetic: all "twice" values
        // We work with integers that are twice the actual quantum numbers.

        // Helper: factorial with f64
        fn factorial(n: i64) -> f64 {
            if n <= 1 {
                return 1.0;
            }
            (2..=n).fold(1.0, |acc, i| acc * i as f64)
        }

        let j1 = twice_j1 as i64;
        let j2 = twice_j2 as i64;
        let j = twice_j as i64;
        let m1 = twice_m1 as i64;
        let m2 = twice_m2 as i64;
        let m = twice_m as i64;

        // All these must be even for the factorials to make sense
        // (since we're working with twice values).
        // Convert to standard half-integer indices:
        let a1 = (j1 + j2 - j) / 2;
        let a2 = (j1 - j2 + j) / 2;
        let a3 = (-j1 + j2 + j) / 2;
        let a4 = (j1 + j2 + j) / 2 + 1;

        let b1 = (j1 + m1) / 2;
        let b2 = (j1 - m1) / 2;
        let b3 = (j2 + m2) / 2;
        let b4 = (j2 - m2) / 2;
        let b5 = (j + m) / 2;
        let b6 = (j - m) / 2;

        // Prefactor
        let prefactor = ((j + 1) as f64
            * factorial(a1)
            * factorial(a2)
            * factorial(a3)
            / factorial(a4)
            * factorial(b1)
            * factorial(b2)
            * factorial(b3)
            * factorial(b4)
            * factorial(b5)
            * factorial(b6))
            .sqrt();

        // Sum over k: constrained by all factorial arguments being non-negative.
        // d4 = (j-j2+m1)/2 + k ≥ 0  =>  k ≥ -(j-j2+m1)/2 = (j2-j-m1)/2
        // d5 = (j-j1-m2)/2 + k ≥ 0  =>  k ≥ -(j-j1-m2)/2 = (j1+m2-j)/2
        // d1 = a1-k ≥ 0, d2 = b2-k ≥ 0, d3 = b3-k ≥ 0
        let k_min = [0i64, (j2 - j - m1) / 2, (j1 + m2 - j) / 2]
            .into_iter()
            .max()
            .unwrap()
            .max(0);
        let k_max = [a1, b2, b3].into_iter().min().unwrap();

        let mut sum = 0.0;
        for k in k_min..=k_max {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };

            // Standard Racah formula denominator terms (half-integer indices):
            let d0 = k;
            let d1 = a1 - k; // (j1+j2-J)/2 - k
            let d2 = b2 - k; // (j1-m1)/2 - k
            let d3 = b3 - k; // (j2+m2)/2 - k
            let d4 = (j - j2 + m1) / 2 + k; // (J-j2+m1)/2 + k
            let d5 = (j - j1 - m2) / 2 + k; // (J-j1-m2)/2 + k

            if d0 < 0 || d1 < 0 || d2 < 0 || d3 < 0 || d4 < 0 || d5 < 0 {
                continue;
            }

            let denom_val = factorial(d0)
                * factorial(d1)
                * factorial(d2)
                * factorial(d3)
                * factorial(d4)
                * factorial(d5);

            sum += sign / denom_val;
        }

        prefactor * sum
    }
}

impl Default for ClebschGordanCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cg_trivial_coupling() {
        let cache = ClebschGordanCache::new();
        // ⟨0 0 ; 0 0 | 0 0⟩ = 1
        let val = cache.get(
            SU2Irrep { twice_j: 0 },
            0,
            SU2Irrep { twice_j: 0 },
            0,
            SU2Irrep { twice_j: 0 },
            0,
        );
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cg_spin_half_coupling() {
        let cache = ClebschGordanCache::new();
        // ⟨1/2 1/2 ; 1/2 -1/2 | 1 0⟩ = 1/√2
        let val = cache.get(
            SU2Irrep { twice_j: 1 },
            1,
            SU2Irrep { twice_j: 1 },
            -1,
            SU2Irrep { twice_j: 2 },
            0,
        );
        assert!((val - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn cg_selection_rule_m() {
        let cache = ClebschGordanCache::new();
        // m1 + m2 ≠ M → 0
        let val = cache.get(
            SU2Irrep { twice_j: 2 },
            2,
            SU2Irrep { twice_j: 2 },
            2,
            SU2Irrep { twice_j: 2 },
            0,
        );
        assert!((val).abs() < 1e-12);
    }

    #[test]
    fn cg_prefill_and_lookup() {
        let cache = ClebschGordanCache::new();
        cache.prefill(2); // up to j=1
        // After prefill, lookups should be cache hits
        let val = cache.get(
            SU2Irrep { twice_j: 2 },
            0,
            SU2Irrep { twice_j: 2 },
            0,
            SU2Irrep { twice_j: 0 },
            0,
        );
        // ⟨1 0 ; 1 0 | 0 0⟩ = -1/√3
        assert!((val + 1.0 / 3.0_f64.sqrt()).abs() < 1e-12);
    }
}
