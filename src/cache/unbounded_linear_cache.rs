use crate::cache::approximate_cache::*;
use crate::numerics::comp::*;

/// A cache implementation that checks all entries one-by-one, without eviction
/// ## Generic Types
/// The types K and V are used for the cache keys and values respectively.
///
/// K should be ApproxComparable, i.e. the compiler should know how to
/// decide that two K's are 'close enough' given a certain tolerance.
///
/// V should be Clone so that the user can do whatever they want with a returned
/// value without messing with the actual cache line.
///
/// ## Constructors
/// Use the ```from``` method to create a new cache. You will be asked to provide a
/// tolerance for the search and (optionally) an initial allocated capacity in memory.
/// ```tolerance``` indicates the searching sensitivity (see ApproxComparable),
/// which is a constant w.r.t. to the queried K (for now).
struct UnboundedLinearCache<K, V>
where
    K: ApproxComparable,
    V: Clone,
{
    lines: Vec<(K, V)>,
    tolerance: f32,
}

impl<K, V> UnboundedLinearCache<K, V>
where
    K: ApproxComparable,
    V: Clone,
{
    pub fn new(tolerance: f32) -> Self {
        UnboundedLinearCache {
            lines: Vec::new(),
            tolerance,
        }
    }

    pub fn with_initial_capacity(tolerance: f32, capacity: usize) -> Self {
        UnboundedLinearCache {
            lines: Vec::with_capacity(capacity),
            tolerance,
        }
    }
}

impl<K, V> ApproximateCache<K, V> for UnboundedLinearCache<K, V>
where
    K: ApproxComparable,
    V: Clone,
{
    // to find a match in an unbounded cache, iterate over all cache lines
    // and return early if you have something
    fn find(&self, key: &K) -> Option<V> {
        let x = self
            .lines
            .iter()
            .find(|&(k, _v)| key.roughly_matches(k, self.tolerance));

        x.map(|(_u, v)| v.clone())
    }

    // inserting a new value in a linear cache == pushing it at the end for future scans
    fn insert(&mut self, key: K, value: V) {
        self.lines.push((key, value));
    }

    fn len(&self) -> usize {
        self.lines.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{QuickCheck, TestResult};

    const TEST_TOLERANCE : f32 = 1e-8;

    #[test]
    fn identity_always_matches() {
        fn qc_identity_always_matches(start_state: Vec<(f32, u8)>, key : f32, value : u8) -> TestResult {
            let mut ulc = UnboundedLinearCache::<f32, u8>::new(TEST_TOLERANCE);
            if !key.is_finite() || start_state.len() > 100 {
                return TestResult::discard()
            }

            ulc.insert(key, value);
            for (k, v) in start_state {
                ulc.insert(k, v);
            }

            if let Some(x) = ulc.find(&key) {
                TestResult::from_bool(x == value)
            } else {
                TestResult::failed()
            }
        }

        QuickCheck::new()
            .tests(10_000)
            .min_tests_passed(1_000)
            .quickcheck(qc_identity_always_matches as fn(Vec<(f32, u8)>, f32, u8) -> TestResult);
    }
}