use crate::caching::approximate_cache::ApproximateCache;
use crate::numerics::comp::ApproxComparable;
/// A cache implementation that checks all entries one-by-one, without eviction
/// ## Generic Types
/// The types K and V are used for the cache keys and values respectively.
///
/// K should be `ApproxComparable`, i.e. the compiler should know how to
/// decide that two K's are 'close enough' given a certain tolerance.
///
/// V should be `Clone` so that the user can do whatever they want with a returned
/// value without messing with the actual cache line.
///
/// ## Constructors
/// Use the ```from``` method to create a new cache. You will be asked to provide a
/// tolerance for the search and (optionally) an initial allocated capacity in memory.
/// ```tolerance``` indicates the searching sensitivity (see `ApproxComparable`),
/// which is a constant w.r.t. to the queried K (for now).
struct UnboundedLinearCache<K, V>
where
    K: ApproxComparable,
    V: Clone,
{
    keys: Vec<K>,
    values: Vec<V>,
    tolerance: f32,
}

impl<K, V> UnboundedLinearCache<K, V>
where
    K: ApproxComparable,
    V: Clone,
{
    pub fn new(tolerance: f32) -> Self {
        UnboundedLinearCache {
            keys: Vec::new(),
            values: Vec::new(),
            tolerance,
        }
    }

    pub fn with_initial_capacity(tolerance: f32, capacity: usize) -> Self {
        UnboundedLinearCache {
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
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
    fn find(&self, to_find: &K) -> Option<V> {
        let potential_match = self
            .keys
            .iter()
            .position(|key| to_find.roughly_matches(key, self.tolerance));

        potential_match.map(|i| self.values[i].clone())
    }

    // inserting a new value in a linear cache == pushing it at the end for future scans
    fn insert(&mut self, key: K, value: V) {
        self.keys.push(key);
        self.values.push(value);
    }

    fn len(&self) -> usize {
        self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::caching::approximate_cache::COMPTIME_CACHE_SIZE;

    use super::*;
    use quickcheck::{QuickCheck, TestResult};

    const TEST_TOLERANCE: f32 = 1e-8;
    const TEST_MAX_SIZE: usize = COMPTIME_CACHE_SIZE;

    #[test]
    fn start_always_matches_exactly() {
        fn qc_start_always_matches_exactly(
            start_state: Vec<(f32, u8)>,
            key: f32,
            value: u8,
        ) -> TestResult {
            let mut ulc = UnboundedLinearCache::<f32, u8>::new(TEST_TOLERANCE);
            if !key.is_finite() || start_state.len() > TEST_MAX_SIZE {
                return TestResult::discard();
            }

            ulc.insert(key, value);
            for &(k, v) in start_state.iter() {
                ulc.insert(k, v);
            }

            assert!(ulc.len() == start_state.len() + 1);

            if let Some(x) = ulc.find(&key) {
                TestResult::from_bool(x == value)
            } else {
                TestResult::failed()
            }
        }

        QuickCheck::new()
            .tests(10_000)
            .min_tests_passed(1_000)
            .quickcheck(
                qc_start_always_matches_exactly as fn(Vec<(f32, u8)>, f32, u8) -> TestResult,
            );
    }

    #[test]
    fn middle_always_matches() {
        fn qc_middle_always_matches(
            start_state: Vec<(f32, u8)>,
            key: f32,
            value: u8,
            end_state: Vec<(f32, u8)>,
        ) -> TestResult {
            let mut ulc = UnboundedLinearCache::<f32, u8>::new(TEST_TOLERANCE);
            if !key.is_finite() || start_state.len() > TEST_MAX_SIZE {
                return TestResult::discard();
            }

            for &(k, v) in start_state.iter() {
                ulc.insert(k, v);
            }
            ulc.insert(key, value);
            for &(k, v) in end_state.iter() {
                ulc.insert(k, v);
            }

            assert!(ulc.len() == start_state.len() + end_state.len() + 1);

            // we should match on something but we can't know on what
            TestResult::from_bool(ulc.find(&key).is_some())
        }

        QuickCheck::new()
            .tests(10_000)
            .min_tests_passed(1_000)
            .quickcheck(
                qc_middle_always_matches
                    as fn(Vec<(f32, u8)>, f32, u8, Vec<(f32, u8)>) -> TestResult,
            );
    }
}
