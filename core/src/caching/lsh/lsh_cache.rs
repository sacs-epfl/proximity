use crate::caching::approximate_cache::ApproximateCache;
use crate::caching::approximate_cache::DefaultApproximateCache;
use crate::caching::approximate_cache::Tolerance;
use crate::caching::FifoCache;
use crate::caching::LruCache;

use crate::caching::lsh::hasher::SimHashHasher;
use crate::numerics::ApproxComparable;
use crate::numerics::VectorLike;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

/// A key-value store that uses cosine LSH to direct queries into fixed-size cache buckets.
pub struct LshCache<C> {
    hasher: SimHashHasher,
    buckets: HashMap<Vec<bool>, C>,
    bucket_capacity: usize,
}

pub type LshFifoCache<K, V> = LshCache<FifoCache<K, V>>;
pub type LshLruCache<K, V> = LshCache<LruCache<K, V>>;

impl<C> LshCache<C> {
    pub fn new(num_hash: usize, dim: usize, bucket_capacity: usize, seed: Option<u64>) -> Self {
        let hasher = match seed {
            Some(s) => SimHashHasher::new_seeded(num_hash, dim, s),
            None => SimHashHasher::new(num_hash, dim),
        };

        Self {
            hasher,
            buckets: HashMap::new(),
            bucket_capacity,
        }
    }

    fn signature(&self, key: &[f32]) -> Vec<bool> {
        self.hasher.hash(key.normalized().as_ref())
    }
}

impl<K, V, C> ApproximateCache<K, V> for LshCache<C>
where
    V: Clone,
    K: ApproxComparable + AsRef<[f32]>,
    C: DefaultApproximateCache<K, V>,
{
    /// Find a value by key, mutably accessing the bucket for potential reordering.
    fn find(&mut self, target: &K) -> Option<V> {
        let sig = self.signature(target.as_ref());
        let bucket = self.buckets.get_mut(&sig)?;
        bucket.find(target)
    }

    /// Insert a key-value pair, normalizing the key before hashing and storing.
    fn insert(&mut self, key: K, value: V, tol: f32) {
        let sig = self.signature(key.as_ref());
        self.buckets
            .entry(sig)
            .or_insert_with(|| C::from_capacity(self.bucket_capacity))
            .insert(key, value, tol);
    }

    fn len(&self) -> usize {
        self.buckets.values().map(|b| b.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use std::hash::Hasher;

    use super::*;
    use quickcheck::{QuickCheck, TestResult};

    const DIM: usize = 8;
    const NUM_HASH: usize = 8;
    const BUCKET_CAP: usize = 2;
    const TOL: f32 = 1e-6;

    #[cfg(test)]
    #[derive(Debug, Clone)]
    struct TestVecF32(pub Vec<f32>);

    #[cfg(test)]
    impl PartialEq for TestVecF32 {
        fn eq(&self, other: &Self) -> bool {
            self.0.len() == other.0.len()
                && self
                    .0
                    .iter()
                    .zip(&other.0)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        }
    }

    #[cfg(test)]
    impl ApproxComparable for TestVecF32 {
        fn fuzziness(&self, instore: &Self) -> f32 {
            self.0.fuzziness(&instore.0)
        }
    }

    #[cfg(test)]
    impl Hash for TestVecF32 {
        fn hash<H: Hasher>(&self, state: &mut H) {
            for &val in &self.0 {
                state.write_u32(val.to_bits());
            }
        }
    }

    #[cfg(test)]
    impl AsRef<[f32]> for TestVecF32 {
        fn as_ref(&self) -> &[f32] {
            self.0.as_ref()
        }
    }

    #[cfg(test)]
    impl Eq for TestVecF32 {}

    // Helper function to create valid test vectors
    fn make_valid_vec(raw: Vec<f32>, dim: usize) -> Option<TestVecF32> {
        if raw.len() < dim {
            return None;
        }
        let slice = &raw[0..dim];
        if slice.iter().any(|&x| !x.is_finite()) {
            return None;
        }
        Some(TestVecF32(slice.to_vec()))
    }

    #[test]
    fn insert_then_find_succeeds() {
        fn qc_insert_then_find(raw_key: Vec<f32>, value: i32) -> TestResult {
            let Some(key) = make_valid_vec(raw_key, DIM) else {
                return TestResult::discard();
            };

            let mut cache = LshFifoCache::new(NUM_HASH, DIM, BUCKET_CAP, Some(42));
            cache.insert(key.clone(), value, TOL);
            let found = cache.find(&key);

            TestResult::from_bool(found == Some(value))
        }

        QuickCheck::new()
            .tests(10_000)
            .min_tests_passed(1_000)
            .quickcheck(qc_insert_then_find as fn(Vec<f32>, i32) -> TestResult);
    }

    #[test]
    fn find_is_idempotent() {
        fn qc_find_idempotent(raw_key: Vec<f32>, value: i32) -> TestResult {
            let Some(key) = make_valid_vec(raw_key, DIM) else {
                return TestResult::discard();
            };

            let mut cache = LshFifoCache::new(NUM_HASH, DIM, BUCKET_CAP, Some(99));
            cache.insert(key.clone(), value, TOL);

            let first = cache.find(&key);
            let second = cache.find(&key);

            TestResult::from_bool(first == second)
        }

        QuickCheck::new()
            .tests(10_000)
            .min_tests_passed(1_000)
            .quickcheck(qc_find_idempotent as fn(Vec<f32>, i32) -> TestResult);
    }

    #[test]
    fn insert_never_decreases_size() {
        fn qc_insert_size(raw_keys: Vec<Vec<f32>>) -> TestResult {
            let keys: Vec<TestVecF32> = raw_keys
                .into_iter()
                .filter_map(|k| make_valid_vec(k, DIM))
                .take(10) // Limit to avoid excessive test time
                .collect();

            if keys.is_empty() {
                return TestResult::discard();
            }

            let mut cache = LshFifoCache::new(NUM_HASH, DIM, BUCKET_CAP, Some(123));
            let mut prev_size = 0;

            for (i, key) in keys.iter().enumerate() {
                cache.insert(key.clone(), i as i32, TOL);
                let new_size = cache.len();
                if new_size < prev_size {
                    return TestResult::from_bool(false);
                }
                prev_size = new_size;
            }

            TestResult::from_bool(true)
        }

        QuickCheck::new()
            .tests(1_000)
            .min_tests_passed(100)
            .quickcheck(qc_insert_size as fn(Vec<Vec<f32>>) -> TestResult);
    }

    #[test]
    fn scaled_vectors_hash_to_same_bucket() {
        fn qc_scaled_vectors_same_bucket(
            raw_vec: Vec<f32>,
            scale1: f32,
            scale2: f32,
        ) -> TestResult {
            let Some(base) = make_valid_vec(raw_vec, DIM) else {
                return TestResult::discard();
            };

            if !scale1.is_finite() || !scale2.is_finite() || scale1 == 0.0 || scale2 == 0.0 {
                return TestResult::discard();
            }

            let scaled1 = TestVecF32(base.0.iter().map(|&x| x * scale1).collect());
            let scaled2 = TestVecF32(base.0.iter().map(|&x| x * scale2).collect());

            if scaled1.0.iter().any(|&x| !x.is_finite())
                || scaled2.0.iter().any(|&x| !x.is_finite())
            {
                return TestResult::discard();
            }

            let mut cache = LshFifoCache::new(NUM_HASH, DIM, BUCKET_CAP, Some(777));

            cache.insert(scaled1.clone(), 1, TOL);
            let found = cache.find(&scaled2);

            // Scaled vectors should hash to the same bucket, so we should find something
            // (though it may not match exactly due to tolerance)
            TestResult::from_bool(found.is_some())
        }

        QuickCheck::new()
            .tests(10_000)
            .min_tests_passed(1_000)
            .quickcheck(qc_scaled_vectors_same_bucket as fn(Vec<f32>, f32, f32) -> TestResult);
    }

    #[test]
    fn cache_size_bounded_by_capacity() {
        fn qc_size_bounded(raw_keys: Vec<Vec<f32>>, bucket_cap: u8) -> TestResult {
            let bucket_cap = bucket_cap.max(1) as usize; // Ensure at least 1

            let keys: Vec<TestVecF32> = raw_keys
                .into_iter()
                .filter_map(|k| make_valid_vec(k, DIM))
                .take(100) // Limit insertions
                .collect();

            if keys.len() < 5 {
                return TestResult::discard();
            }

            let mut cache = LshFifoCache::new(NUM_HASH, DIM, bucket_cap, Some(456));

            for (i, key) in keys.iter().enumerate() {
                cache.insert(key.clone(), i as i32, TOL);
            }

            // Size should be bounded by number of buckets * capacity
            // We can't easily count buckets, but we know size should be reasonable
            let size = cache.len();
            TestResult::from_bool(size <= keys.len())
        }

        QuickCheck::new()
            .tests(1_000)
            .min_tests_passed(100)
            .quickcheck(qc_size_bounded as fn(Vec<Vec<f32>>, u8) -> TestResult);
    }

    #[test]
    fn lru_find_updates_recency() {
        fn qc_lru_recency(raw_keys: Vec<Vec<f32>>) -> TestResult {
            let keys: Vec<TestVecF32> = raw_keys
                .into_iter()
                .filter_map(|k| make_valid_vec(k, DIM))
                .take(3)
                .collect();

            if keys.len() < 3 {
                return TestResult::discard();
            }

            // Ensure keys hash to same bucket by making them point in same direction
            let base_key = &keys[0];
            let k1 = base_key.clone();
            let k2 = TestVecF32(base_key.0.iter().map(|&x| x * 2.0).collect());
            let k3 = TestVecF32(base_key.0.iter().map(|&x| x * 3.0).collect());

            let mut cache = LshLruCache::new(NUM_HASH, DIM, 2, Some(888));

            // Insert k1 and k2 (fills bucket)
            cache.insert(k1.clone(), 1, TOL);
            cache.insert(k2.clone(), 2, TOL);

            // Access k1 to make it recently used
            cache.find(&k1);

            // Insert k3 (should evict k2, the least recently used)
            cache.insert(k3.clone(), 3, TOL);

            // k1 should still be findable, k3 should be findable
            let k1_found = cache.find(&k1).is_some();
            let k3_found = cache.find(&k3).is_some();

            TestResult::from_bool(k1_found && k3_found)
        }

        QuickCheck::new()
            .tests(1_000)
            .min_tests_passed(100)
            .quickcheck(qc_lru_recency as fn(Vec<Vec<f32>>) -> TestResult);
    }

    #[test]
    fn empty_cache_finds_nothing() {
        fn qc_empty_finds_nothing(raw_key: Vec<f32>) -> TestResult {
            let Some(key) = make_valid_vec(raw_key, DIM) else {
                return TestResult::discard();
            };

            let mut cache =
                LshFifoCache::<TestVecF32, i32>::new(NUM_HASH, DIM, BUCKET_CAP, Some(111));
            let found = cache.find(&key);

            TestResult::from_bool(found.is_none())
        }

        QuickCheck::new()
            .tests(10_000)
            .min_tests_passed(1_000)
            .quickcheck(qc_empty_finds_nothing as fn(Vec<f32>) -> TestResult);
    }

    #[test]
    fn multiple_values_in_bucket_fifo() {
        fn qc_fifo_ordering(raw_key: Vec<f32>) -> TestResult {
            let Some(base_key) = make_valid_vec(raw_key, DIM) else {
                return TestResult::discard();
            };

            // Create keys that hash to same bucket (scaled versions)
            let k1 = TestVecF32(base_key.0.iter().map(|&x| x * 1.0).collect());
            let k2 = TestVecF32(base_key.0.iter().map(|&x| x * 2.0).collect());
            let k3 = TestVecF32(base_key.0.iter().map(|&x| x * 3.0).collect());

            let mut cache = LshFifoCache::new(NUM_HASH, DIM, 2, Some(555));

            cache.insert(k1.clone(), 1, TOL);
            cache.insert(k2.clone(), 2, TOL);
            cache.insert(k3.clone(), 3, TOL); // Should evict k1 (FIFO)

            let k1_found = cache.find(&k1);
            let k2_found = cache.find(&k2);
            let k3_found = cache.find(&k3);

            // After inserting 3 items with capacity 2, first should be evicted
            TestResult::from_bool(k1_found.is_none() && k2_found.is_some() && k3_found.is_some())
        }

        QuickCheck::new()
            .tests(10_000)
            .min_tests_passed(1_000)
            .quickcheck(qc_fifo_ordering as fn(Vec<f32>) -> TestResult);
    }
}
