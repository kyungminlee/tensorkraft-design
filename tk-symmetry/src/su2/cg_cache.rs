//! Lazily populated cache of Clebsch-Gordan coefficients, 6j and 9j symbols.

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

/// Key for a 6j symbol lookup.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct SixJKey {
    twice_j1: u32,
    twice_j2: u32,
    twice_j3: u32,
    twice_j4: u32,
    twice_j5: u32,
    twice_j6: u32,
}

/// Thread-safe, lazily populated cache of Clebsch-Gordan coefficients
/// and Wigner 6j/9j symbols.
///
/// Computing CG coefficients and recoupling symbols from scratch is expensive;
/// caching amortizes the cost across the thousands of contractions in a DMRG sweep.
pub struct ClebschGordanCache {
    cg_cache: RwLock<HashMap<CgKey, f64>>,
    sixj_cache: RwLock<HashMap<SixJKey, f64>>,
}

impl ClebschGordanCache {
    /// Construct an empty cache.
    pub fn new() -> Self {
        ClebschGordanCache {
            cg_cache: RwLock::new(HashMap::new()),
            sixj_cache: RwLock::new(HashMap::new()),
        }
    }

    // -----------------------------------------------------------------------
    // Clebsch-Gordan coefficients
    // -----------------------------------------------------------------------

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
            let cache = self.cg_cache.read().unwrap();
            if let Some(&val) = cache.get(&key) {
                return val;
            }
        }

        // Slow path: compute and insert
        let val = Self::compute_cg(
            ja.twice_j, twice_ma, jb.twice_j, twice_mb, jc.twice_j, twice_mc,
        );

        {
            let mut cache = self.cg_cache.write().unwrap();
            cache.insert(key, val);
        }

        val
    }

    /// Pre-populate the cache for all CG coefficients up to a given j_max.
    /// Call once before a DMRG sweep to amortize initialization cost.
    ///
    /// `twice_j_max` is the maximum twice_j value to precompute (e.g., 20 for j=10).
    pub fn prefill(&self, twice_j_max: u32) {
        let mut cache = self.cg_cache.write().unwrap();

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

    /// Number of cached CG coefficients.
    pub fn cg_cache_size(&self) -> usize {
        self.cg_cache.read().unwrap().len()
    }

    // -----------------------------------------------------------------------
    // Wigner 6j symbols
    // -----------------------------------------------------------------------

    /// Compute the Wigner 6j symbol:
    ///
    /// ```text
    /// { j1  j2  j3 }
    /// { j4  j5  j6 }
    /// ```
    ///
    /// All arguments are "twice" values (e.g., j=1/2 → 1).
    /// The 6j symbol is required for recoupling in multi-site operations.
    ///
    /// Uses the Racah formula with caching for repeated lookups.
    pub fn sixj(
        &self,
        j1: SU2Irrep,
        j2: SU2Irrep,
        j3: SU2Irrep,
        j4: SU2Irrep,
        j5: SU2Irrep,
        j6: SU2Irrep,
    ) -> f64 {
        let key = SixJKey {
            twice_j1: j1.twice_j,
            twice_j2: j2.twice_j,
            twice_j3: j3.twice_j,
            twice_j4: j4.twice_j,
            twice_j5: j5.twice_j,
            twice_j6: j6.twice_j,
        };

        // Fast path: read lock
        {
            let cache = self.sixj_cache.read().unwrap();
            if let Some(&val) = cache.get(&key) {
                return val;
            }
        }

        // Slow path: compute and insert
        let val = Self::compute_sixj(
            j1.twice_j, j2.twice_j, j3.twice_j,
            j4.twice_j, j5.twice_j, j6.twice_j,
        );

        {
            let mut cache = self.sixj_cache.write().unwrap();
            cache.insert(key, val);
        }

        val
    }

    /// Number of cached 6j symbols.
    pub fn sixj_cache_size(&self) -> usize {
        self.sixj_cache.read().unwrap().len()
    }

    // -----------------------------------------------------------------------
    // Wigner 9j symbols
    // -----------------------------------------------------------------------

    /// Compute the Wigner 9j symbol:
    ///
    /// ```text
    /// { j1  j2  j3 }
    /// { j4  j5  j6 }
    /// { j7  j8  j9 }
    /// ```
    ///
    /// All arguments are "twice" values.
    /// Computed via summation over 6j symbols:
    ///
    /// ```text
    /// {j1 j2 j3}   = Σ_x (2x+1) {j1 j4 j7} {j2 j5 j8} {j3 j6 j9}
    /// {j4 j5 j6}                  {j8 j9 x } {j4 x  j6} {x  j1 j2}
    /// {j7 j8 j9}
    /// ```
    ///
    /// where x ranges over all values satisfying the triangle inequalities
    /// of the three 6j symbols.
    pub fn ninej(
        &self,
        j1: SU2Irrep, j2: SU2Irrep, j3: SU2Irrep,
        j4: SU2Irrep, j5: SU2Irrep, j6: SU2Irrep,
        j7: SU2Irrep, j8: SU2Irrep, j9: SU2Irrep,
    ) -> f64 {
        // x must satisfy triangle inequalities from all three 6j symbols:
        // From {j1,j4,j7; j8,j9,x}: triangle(j1,j4,j7), triangle(j8,j9,x), triangle(j1,j9,x)(?), etc.
        // The summation variable x must satisfy:
        //   triangle(j1, j9, x), triangle(j4, x, j6), triangle(j2, x, j8)
        // which gives: x ∈ [x_min, x_max] where
        //   x_min = max(|j1-j9|, |j4-j6|, |j2-j8|)    (in twice values)
        //   x_max = min(j1+j9, j4+j6, j2+j8)

        let twice_x_min = [
            (j1.twice_j as i32 - j9.twice_j as i32).unsigned_abs(),
            (j4.twice_j as i32 - j6.twice_j as i32).unsigned_abs(),
            (j2.twice_j as i32 - j8.twice_j as i32).unsigned_abs(),
        ].into_iter().max().unwrap();

        let twice_x_max = [
            j1.twice_j + j9.twice_j,
            j4.twice_j + j6.twice_j,
            j2.twice_j + j8.twice_j,
        ].into_iter().min().unwrap();

        if twice_x_min > twice_x_max {
            return 0.0;
        }

        // Determine step: x must have the same integer/half-integer parity
        // as determined by the triangle constraints. The parity is fixed by
        // the sum j1+j9 (mod 2) — all three constraints must agree.
        let parity = (j1.twice_j + j9.twice_j + twice_x_min) % 2;
        let start = if parity == 0 { twice_x_min } else { twice_x_min + 1 };

        let mut sum = 0.0;
        let mut twice_x = start;
        while twice_x <= twice_x_max {
            let x = SU2Irrep { twice_j: twice_x };
            let weight = (twice_x + 1) as f64; // (2x+1) in twice units is twice_x+1

            let s1 = self.sixj(j1, j4, j7, j8, j9, x);
            let s2 = self.sixj(j2, j5, j8, j4, x, j6);
            let s3 = self.sixj(j3, j6, j9, x, j1, j2);

            let sign = if twice_x % 2 == 0 { 1.0 } else { -1.0 };
            sum += sign * weight * s1 * s2 * s3;

            twice_x += 2;
        }

        sum
    }

    // -----------------------------------------------------------------------
    // Pre-populate 6j cache
    // -----------------------------------------------------------------------

    /// Pre-populate the 6j cache for all valid combinations up to `twice_j_max`.
    pub fn prefill_sixj(&self, twice_j_max: u32) {
        for j1 in 0..=twice_j_max {
            for j2 in 0..=twice_j_max {
                for j3 in (Self::tri_min(j1, j2)..=Self::tri_max(j1, j2)).step_by(2) {
                    for j4 in 0..=twice_j_max {
                        for j5 in (Self::tri_min(j2, j4)..=Self::tri_max(j2, j4)).step_by(2) {
                            if j5 > twice_j_max { break; }
                            let j6_min1 = Self::tri_min(j1, j5);
                            let j6_max1 = Self::tri_max(j1, j5);
                            let j6_min2 = Self::tri_min(j3, j4);
                            let j6_max2 = Self::tri_max(j3, j4);
                            let j6_min = j6_min1.max(j6_min2);
                            let j6_max = j6_max1.min(j6_max2);
                            if j6_min > j6_max { continue; }
                            for j6 in (j6_min..=j6_max).step_by(2) {
                                if j6 > twice_j_max { break; }
                                let _ = self.sixj(
                                    SU2Irrep { twice_j: j1 },
                                    SU2Irrep { twice_j: j2 },
                                    SU2Irrep { twice_j: j3 },
                                    SU2Irrep { twice_j: j4 },
                                    SU2Irrep { twice_j: j5 },
                                    SU2Irrep { twice_j: j6 },
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Internal computation routines
    // -----------------------------------------------------------------------

    /// Triangle inequality minimum: |a - b|
    fn tri_min(a: u32, b: u32) -> u32 {
        (a as i32 - b as i32).unsigned_abs()
    }

    /// Triangle inequality maximum: a + b
    fn tri_max(a: u32, b: u32) -> u32 {
        a + b
    }

    /// Check whether (a, b, c) satisfy the triangle inequality.
    /// All values are "twice" values.
    fn triangle_ok(a: u32, b: u32, c: u32) -> bool {
        let sum = a + b;
        let diff = (a as i32 - b as i32).unsigned_abs();
        c <= sum && c >= diff && (a + b + c) % 2 == 0
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

        let j1 = twice_j1 as i64;
        let j2 = twice_j2 as i64;
        let j = twice_j as i64;
        let m1 = twice_m1 as i64;
        let m2 = twice_m2 as i64;
        let m = twice_m as i64;

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

        let k_min = [0i64, (j2 - j - m1) / 2, (j1 + m2 - j) / 2]
            .into_iter()
            .max()
            .unwrap()
            .max(0);
        let k_max = [a1, b2, b3].into_iter().min().unwrap();

        let mut sum = 0.0;
        for k in k_min..=k_max {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };

            let d0 = k;
            let d1 = a1 - k;
            let d2 = b2 - k;
            let d3 = b3 - k;
            let d4 = (j - j2 + m1) / 2 + k;
            let d5 = (j - j1 - m2) / 2 + k;

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

    /// Compute the Wigner 6j symbol using the Racah formula.
    ///
    /// ```text
    /// { j1  j2  j3 }
    /// { j4  j5  j6 }
    /// ```
    ///
    /// All arguments are "twice" values.
    /// Uses the formula:
    ///   {j1 j2 j3; j4 j5 j6} = Δ(j1,j2,j3) Δ(j1,j5,j6) Δ(j4,j2,j6) Δ(j4,j5,j3)
    ///                            × Σ_k (-1)^k (k+1)! / [denominator products]
    fn compute_sixj(
        twice_j1: u32,
        twice_j2: u32,
        twice_j3: u32,
        twice_j4: u32,
        twice_j5: u32,
        twice_j6: u32,
    ) -> f64 {
        // Check all four triangle inequalities
        if !Self::triangle_ok(twice_j1, twice_j2, twice_j3) { return 0.0; }
        if !Self::triangle_ok(twice_j1, twice_j5, twice_j6) { return 0.0; }
        if !Self::triangle_ok(twice_j4, twice_j2, twice_j6) { return 0.0; }
        if !Self::triangle_ok(twice_j4, twice_j5, twice_j3) { return 0.0; }

        let j1 = twice_j1 as i64;
        let j2 = twice_j2 as i64;
        let j3 = twice_j3 as i64;
        let j4 = twice_j4 as i64;
        let j5 = twice_j5 as i64;
        let j6 = twice_j6 as i64;

        // Triangle coefficients Δ(a,b,c) = sqrt[(a+b-c)/2)! ((a-b+c)/2)! ((-a+b+c)/2)! / ((a+b+c)/2+1)!]
        let delta = |a: i64, b: i64, c: i64| -> f64 {
            let n1 = (a + b - c) / 2;
            let n2 = (a - b + c) / 2;
            let n3 = (-a + b + c) / 2;
            let n4 = (a + b + c) / 2 + 1;
            (factorial(n1) * factorial(n2) * factorial(n3) / factorial(n4)).sqrt()
        };

        let d1 = delta(j1, j2, j3);
        let d2 = delta(j1, j5, j6);
        let d3 = delta(j4, j2, j6);
        let d4 = delta(j4, j5, j3);

        // The four "triad sums" used in the summation bounds:
        let t1 = (j1 + j2 + j3) / 2;
        let t2 = (j1 + j5 + j6) / 2;
        let t3 = (j4 + j2 + j6) / 2;
        let t4 = (j4 + j5 + j3) / 2;

        // Three "quad sums":
        let q1 = (j1 + j2 + j4 + j5) / 2;
        let q2 = (j2 + j3 + j5 + j6) / 2;
        let q3 = (j1 + j3 + j4 + j6) / 2;

        let k_min = [t1, t2, t3, t4].into_iter().max().unwrap();
        let k_max = [q1, q2, q3].into_iter().min().unwrap();

        let mut sum = 0.0;
        for k in k_min..=k_max {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };

            let num = factorial(k + 1);
            let den = factorial(k - t1)
                * factorial(k - t2)
                * factorial(k - t3)
                * factorial(k - t4)
                * factorial(q1 - k)
                * factorial(q2 - k)
                * factorial(q3 - k);

            sum += sign * num / den;
        }

        d1 * d2 * d3 * d4 * sum
    }
}

/// Factorial of a non-negative integer, computed as f64.
fn factorial(n: i64) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    (2..=n).fold(1.0, |acc, i| acc * i as f64)
}

impl Default for ClebschGordanCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn j(twice_j: u32) -> SU2Irrep {
        SU2Irrep { twice_j }
    }

    // -------------------------------------------------------------------
    // CG coefficient tests
    // -------------------------------------------------------------------

    #[test]
    fn cg_trivial_coupling() {
        let cache = ClebschGordanCache::new();
        // ⟨0 0 ; 0 0 | 0 0⟩ = 1
        let val = cache.get(j(0), 0, j(0), 0, j(0), 0);
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cg_spin_half_coupling() {
        let cache = ClebschGordanCache::new();
        // ⟨1/2 1/2 ; 1/2 -1/2 | 1 0⟩ = 1/√2
        let val = cache.get(j(1), 1, j(1), -1, j(2), 0);
        assert!((val - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn cg_spin_half_singlet() {
        let cache = ClebschGordanCache::new();
        // ⟨1/2 1/2 ; 1/2 -1/2 | 0 0⟩ = 1/√2
        let val = cache.get(j(1), 1, j(1), -1, j(0), 0);
        assert!((val - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn cg_selection_rule_m() {
        let cache = ClebschGordanCache::new();
        // m1 + m2 ≠ M → 0
        let val = cache.get(j(2), 2, j(2), 2, j(2), 0);
        assert!(val.abs() < 1e-12);
    }

    #[test]
    fn cg_prefill_and_lookup() {
        let cache = ClebschGordanCache::new();
        cache.prefill(2); // up to j=1
        // ⟨1 0 ; 1 0 | 0 0⟩ = -1/√3
        let val = cache.get(j(2), 0, j(2), 0, j(0), 0);
        assert!((val + 1.0 / 3.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn cg_spin1_coupling_j2() {
        let cache = ClebschGordanCache::new();
        // ⟨1 1 ; 1 1 | 2 2⟩ = 1 (stretched state)
        let val = cache.get(j(2), 2, j(2), 2, j(4), 4);
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cg_spin1_coupling_j1() {
        let cache = ClebschGordanCache::new();
        // ⟨1 1 ; 1 0 | 1 1⟩ = 1/√2
        let val = cache.get(j(2), 2, j(2), 0, j(2), 2);
        assert!((val - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn cg_spin1_half_plus_spin1() {
        let cache = ClebschGordanCache::new();
        // ⟨1/2 1/2 ; 1 0 | 3/2 1/2⟩ = √(2/3)
        let val = cache.get(j(1), 1, j(2), 0, j(3), 1);
        assert!((val - (2.0 / 3.0_f64).sqrt()).abs() < 1e-12,
            "got {val}, expected {}", (2.0 / 3.0_f64).sqrt());
    }

    #[test]
    fn cg_spin2_stretched_state() {
        let cache = ClebschGordanCache::new();
        // ⟨2 2 ; 2 2 | 4 4⟩ = 1 (stretched state for j1=j2=2, J=4)
        let val = cache.get(j(4), 4, j(4), 4, j(8), 8);
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cg_spin2_coupling_j0() {
        let cache = ClebschGordanCache::new();
        // ⟨2 0 ; 2 0 | 0 0⟩ = 1/√5
        let val = cache.get(j(4), 0, j(4), 0, j(0), 0);
        assert!((val - 1.0 / 5.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn cg_orthonormality_spin_half() {
        // Verify sum_{m1,m2} |⟨j1 m1; j2 m2 | J M⟩|² = 1 for fixed (J, M)
        let cache = ClebschGordanCache::new();
        // j1 = j2 = 1/2, J = 1, M = 0
        let mut sum_sq = 0.0;
        for twice_m1 in [-1i32, 1] {
            let twice_m2 = 0 - twice_m1; // M = 0
            let c = cache.get(j(1), twice_m1, j(1), twice_m2, j(2), 0);
            sum_sq += c * c;
        }
        assert!((sum_sq - 1.0).abs() < 1e-12, "orthonormality: sum = {sum_sq}");
    }

    #[test]
    fn cg_orthonormality_spin1() {
        // j1 = j2 = 1, J = 2, M = 0
        let cache = ClebschGordanCache::new();
        let mut sum_sq = 0.0;
        for twice_m1 in (-2i32..=2).step_by(2) {
            let twice_m2 = 0 - twice_m1;
            if twice_m2.unsigned_abs() > 2 { continue; }
            let c = cache.get(j(2), twice_m1, j(2), twice_m2, j(4), 0);
            sum_sq += c * c;
        }
        assert!((sum_sq - 1.0).abs() < 1e-12, "orthonormality: sum = {sum_sq}");
    }

    // -------------------------------------------------------------------
    // 6j symbol tests
    // -------------------------------------------------------------------

    #[test]
    fn sixj_with_j0() {
        let cache = ClebschGordanCache::new();
        // {j1, j2, j3; j4, j5, 0} requires j1=j5, j4=j2
        // = δ_{j1,j5} δ_{j4,j2} (-1)^{j1+j2+j3} / √((2j1+1)(2j2+1))
        // {1/2, 1, 3/2; 1, 1/2, 0}:
        //   j1=1/2, j5=1/2 ✓; j4=1, j2=1 ✓
        //   (-1)^{1/2+1+3/2} = (-1)^3 = -1
        //   1/√(2·3) = 1/√6
        //   Result: -1/√6
        let val = cache.sixj(j(1), j(2), j(3), j(2), j(1), j(0));
        let expected = -1.0 / 6.0_f64.sqrt();
        assert!((val - expected).abs() < 1e-10,
            "6j with j0: got {val}, expected {expected}");
    }

    #[test]
    fn sixj_spin_half() {
        let cache = ClebschGordanCache::new();
        // {1/2, 1/2, 0; 1/2, 1/2, 0}
        // Using the j6=0 formula: j1=j5=1/2, j4=j2=1/2
        // (-1)^{1/2+1/2+0} = (-1)^1 = -1
        // 1/√(2·2) = 1/2
        // Result: -1/2
        let val = cache.sixj(j(1), j(1), j(0), j(1), j(1), j(0));
        assert!((val - (-0.5)).abs() < 1e-10,
            "6j spin-half: got {val}, expected -0.5");
    }

    #[test]
    fn sixj_triangle_violation() {
        let cache = ClebschGordanCache::new();
        // Triangle inequality violation: {2, 2, 2; 2, 2, 10} fails triangle(2,2,10)
        let val = cache.sixj(j(4), j(4), j(4), j(4), j(4), j(20));
        assert!((val).abs() < 1e-12, "6j triangle violation should be 0, got {val}");
    }

    #[test]
    fn sixj_symmetry() {
        let cache = ClebschGordanCache::new();
        // 6j symbols are invariant under column permutations
        let v1 = cache.sixj(j(2), j(4), j(4), j(2), j(2), j(2));
        let v2 = cache.sixj(j(4), j(2), j(4), j(2), j(2), j(2));
        assert!((v1 - v2).abs() < 1e-10,
            "6j column permutation symmetry: {v1} != {v2}");
    }

    // -------------------------------------------------------------------
    // 9j symbol tests
    // -------------------------------------------------------------------

    #[test]
    fn ninej_trivial_j0_row() {
        let cache = ClebschGordanCache::new();
        // {j1, j2, j3; 0, 0, 0; j1, j2, j3} with all j=0 => a known simple value
        // For all zeros:
        // {0 0 0; 0 0 0; 0 0 0} = 1
        let val = cache.ninej(j(0), j(0), j(0), j(0), j(0), j(0), j(0), j(0), j(0));
        assert!((val - 1.0).abs() < 1e-10, "9j all-zero: got {val}, expected 1.0");
    }

    #[test]
    fn ninej_simple_spin_half() {
        let cache = ClebschGordanCache::new();
        // A simple non-trivial 9j symbol:
        // {1/2, 1/2, 0; 1/2, 1/2, 0; 0, 0, 0} = 1/2
        let val = cache.ninej(
            j(1), j(1), j(0),
            j(1), j(1), j(0),
            j(0), j(0), j(0),
        );
        assert!((val - 0.5).abs() < 1e-10,
            "9j spin-half: got {val}, expected 0.5");
    }
}
