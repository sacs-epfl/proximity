pub struct F32Vector<'a> {
    array: &'a [f32],
}

impl<'a> F32Vector<'a> {
    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// # Usage
    /// Computes the **SQUARED** L2 distance between two vectors.
    /// This is cheaper to compute than the regular L2 distance.
    /// This is typically useful when comparing two distances :
    ///
    /// dist(u,v) < dist(w, x) ⇔ dist(u,v) ** 2 < dist(w,x) ** 2
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the two vectors have different lengths.
    /// In release mode, the longest vector will be silently truncated.
    #[inline]
    pub fn l2_dist_squared(&self, other: &F32Vector<'a>) -> f32 {
        debug_assert!(self.len() == other.len());

        self.array
            .iter()
            .zip(other.array)
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
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
}

impl<'a> From<&'a [f32]> for F32Vector<'a> {
    fn from(value: &'a [f32]) -> Self {
        F32Vector { array: value }
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

    #[test]
    fn self_sim_is_zero() {
        fn qc_self_sim_is_zero(totest: Vec<f32>) -> TestResult {
            if totest.iter().any(|x| !x.is_finite()) {
                return TestResult::discard();
            }
            let testvec = F32Vector::from(&totest[..]);
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
            let min_length = all_vecs.iter().map(|x| x.len()).min().unwrap();
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
}
