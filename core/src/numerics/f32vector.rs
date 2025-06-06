use std::simd::{num::SimdFloat, Simd};

pub const SIMD_LANECOUNT: usize = 8;
type SimdF32 = Simd<f32, SIMD_LANECOUNT>;

#[derive(Debug, Clone)]
pub struct F32Vector<'a> {
    array: &'a [f32],
}

impl AsRef<[f32]> for F32Vector<'_> {
    fn as_ref(&self) -> &[f32] {
        self.array
    }
}

impl<'a> F32Vector<'a> {
    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    /// # Usage
    /// Computes the **SQUARED** L2 distance between two vectors.
    /// This is typically useful when comparing two distances :
    ///
    /// dist(u,v) < dist(w, x) ⇔ dist(u,v) ** 2 < dist(w,x) ** 2
    ///
    /// We are usually interested in the left side of the equivalence,
    /// but the right side is cheaper to compute.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the two vectors have different lengths.
    /// In release mode, the longest vector will be silently truncated.
    #[inline]
    pub fn l2_dist_squared(&self, othr: &F32Vector<'a>) -> f32 {
        debug_assert!(self.len() == othr.len());
        debug_assert!(self.len() % SIMD_LANECOUNT == 0);

        let mut intermediate_sum_x8 = Simd::<f32, SIMD_LANECOUNT>::splat(0.0);

        let self_chunks = self.array.chunks_exact(SIMD_LANECOUNT);
        let othr_chunks = othr.array.chunks_exact(SIMD_LANECOUNT);

        for (slice_self, slice_othr) in self_chunks.zip(othr_chunks) {
            let f32x8_slf = SimdF32::from_slice(slice_self);
            let f32x8_oth = SimdF32::from_slice(slice_othr);
            let diff = f32x8_slf - f32x8_oth;
            intermediate_sum_x8 += diff * diff;
        }

        intermediate_sum_x8.reduce_sum() // 8-to-1 sum
    }

    pub fn dot(&self, othr: &F32Vector<'a>) -> f32 {
        debug_assert!(self.len() == othr.len());
        debug_assert!(self.len() % SIMD_LANECOUNT == 0);

        // accumulator vector of zeroes
        let mut accumulated = Simd::<f32, SIMD_LANECOUNT>::splat(0.0);

        let self_chunks = self.array.chunks_exact(SIMD_LANECOUNT);
        let othr_chunks = othr.array.chunks_exact(SIMD_LANECOUNT);

        for (slice_self, slice_othr) in self_chunks.zip(othr_chunks) {
            // load each chunk into a SIMD register
            let vx = SimdF32::from_slice(slice_self);
            let vy = SimdF32::from_slice(slice_othr);
            // multiply-and-accumulate
            accumulated += vx * vy;
        }

        // horizontal sum across lanes
        accumulated.reduce_sum()
    }

    /// # Usage
    /// Computes the L2 distance between two vectors.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the two vectors have different lengths.
    /// In release mode, the longest vector will be silently truncated.
    #[inline]
    pub fn l2_dist(&self, other: &F32Vector<'a>) -> f32 {
        self.l2_dist_squared(other).sqrt()
    }

    /// Returns a new Vec<f32> containing `self` divided by its L2-norm.
    /// If the norm is zero, returns a zero‐filled Vec.
    pub fn normalized(&self) -> Vec<f32> {
        let norm = self.dot(self).sqrt();
        if norm == 0.0 {
            // avoid division by zero; return zero vector
            return vec![0.0; self.len()];
        }
        let inv_norm = 1.0 / norm;

        let mut out = Vec::with_capacity(self.len());

        for chunk in self.array.chunks_exact(SIMD_LANECOUNT) {
            let v = SimdF32::from_slice(chunk);
            let scaled = v * SimdF32::splat(inv_norm);
            out.extend_from_slice(&scaled.to_array());
        }

        out
    }
}

impl<'a> From<&'a [f32]> for F32Vector<'a> {
    fn from(value: &'a [f32]) -> Self {
        F32Vector { array: value }
    }
}

impl<'a> From<F32Vector<'a>> for &'a [f32] {
    fn from(value: F32Vector<'a>) -> Self {
        value.array
    }
}

impl PartialEq for F32Vector<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.array
            .iter()
            .zip(other.array.iter())
            .all(|(&a, &b)| a == b)
    }
}

impl Eq for F32Vector<'_> {}

impl std::hash::Hash for F32Vector<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Iterate through each element of the slice and hash it
        for &value in self.array {
            value.to_bits().hash(state); // Convert `f32` to its bit representation for consistent hashing
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{QuickCheck, TestResult};

    const TOLERANCE: f32 = 1e-8;

    fn close(actual: f32, target: f32) -> bool {
        (target - actual).abs() < TOLERANCE
    }

    fn is_valid_l2(suspect: f32) -> bool {
        suspect.is_finite() && suspect >= 0.0
    }

    fn l2_spec<'a>(v1: F32Vector<'a>, v2: F32Vector<'a>) -> f32 {
        v1.array
            .iter()
            .zip(v2.array.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum()
    }

    #[test]
    fn self_sim_is_zero() {
        fn qc_self_sim_is_zero(totest: Vec<f32>) -> TestResult {
            let usable_length = totest.len() / 8 * 8;
            if totest[0..usable_length].iter().any(|x| !x.is_finite()) {
                return TestResult::discard();
            }
            let testvec = F32Vector::from(&totest[0..usable_length]);
            let selfsim = testvec.l2_dist(&testvec);
            let to_check = is_valid_l2(selfsim) && close(selfsim, 0.0);
            return TestResult::from_bool(to_check);
        }

        QuickCheck::new()
            .tests(10_000)
            // force that less than 90% of tests are discarded due to precondition violations
            // i.e. at least 10% of inputs should be valid so that we cover a good range
            .min_tests_passed(1_000)
            .quickcheck(qc_self_sim_is_zero as fn(Vec<f32>) -> TestResult);
    }

    #[test]
    // verifies the claim in the documentation of l2_dist_squared
    // i.e. dist(u,v) < dist(w, x) ⇔ dist(u,v) ** 2 < dist(w,x) ** 2
    fn squared_invariant() {
        fn qc_squared_invariant(u: Vec<f32>, v: Vec<f32>, w: Vec<f32>, x: Vec<f32>) -> TestResult {
            let all_vecs = [u, v, w, x]; //no need to check for NaNs in this case
            let min_length = all_vecs.iter().map(|x| x.len()).min().unwrap() / 8 * 8;
            let all_vectors: Vec<F32Vector> = all_vecs
                .iter()
                .map(|vec| F32Vector::from(&vec[..min_length]))
                .collect();

            let d1_squared = all_vectors[0].l2_dist_squared(&all_vectors[1]);
            let d2_squared = all_vectors[2].l2_dist_squared(&all_vectors[3]);

            let d1_root = all_vectors[0].l2_dist(&all_vectors[1]);
            let d2_root = all_vectors[2].l2_dist(&all_vectors[3]);

            let sanity_check1 = (d1_squared < d2_squared) == (d1_root < d2_root);
            let sanity_check2 = (d1_squared <= d2_squared) == (d1_root <= d2_root);
            TestResult::from_bool(sanity_check1 && sanity_check2)
        }

        QuickCheck::new().tests(10_000).quickcheck(
            qc_squared_invariant as fn(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) -> TestResult,
        );
    }

    #[test]
    fn simd_matches_spec() {
        fn qc_simd_matches_spec(u: Vec<f32>, v: Vec<f32>) -> TestResult {
            let min_length = u.len().min(v.len()) / 8 * 8;
            let (u_f32v, v_f32v) = (
                F32Vector::from(&u[0..min_length]),
                F32Vector::from(&v[0..min_length]),
            );
            let simd = u_f32v.l2_dist_squared(&v_f32v);
            let spec = l2_spec(u_f32v, v_f32v);

            if simd.is_infinite() {
                TestResult::from_bool(spec.is_infinite())
            } else if simd.is_nan() {
                TestResult::from_bool(spec.is_nan())
            } else {
                TestResult::from_bool(close(simd, spec))
            }
        }

        QuickCheck::new()
            .tests(10_000)
            .quickcheck(qc_simd_matches_spec as fn(Vec<f32>, Vec<f32>) -> TestResult);
    }
}
