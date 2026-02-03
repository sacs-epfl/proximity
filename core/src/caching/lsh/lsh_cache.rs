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

    #[test]
    fn test_lsh_fifo_cache_basic() {
        let mut cache = LshFifoCache::new(NUM_HASH, DIM, BUCKET_CAP, Some(42));

        let k1 = TestVecF32(vec![0.1; DIM]);
        let k2 = TestVecF32(vec![-0.2; DIM]);
        let k3 = TestVecF32(vec![0.1; DIM]); // same direction as k1

        cache.insert(k1.clone(), 1, TOL);
        cache.insert(k2, 2, TOL);
        assert_eq!(cache.find(&k1), Some(1));
        cache.insert(k3.clone(), 3, TOL);
        // Bucket for k1 now has entries [k1=1, k3=3], matches equally
        assert!(cache.find(&k3) == Some(3) || cache.find(&k3) == Some(1));
    }

    #[test]
    fn test_lsh_fifo_cache_eviction_order() {
        let mut cache = LshFifoCache::new(NUM_HASH, DIM, 2, Some(123));

        let k2 = TestVecF32(vec![1.0; DIM]);
        let k3 = TestVecF32(vec![2.0; DIM]);
        let k4 = TestVecF32(vec![3.0; DIM]);

        cache.insert(k2.clone(), 20, TOL);
        cache.insert(k3.clone(), 30, TOL);
        cache.insert(k4.clone(), 40, TOL);

        // One of the earlier keys must have been evicted (depending on bucket)
        let hits = vec![cache.find(&k2), cache.find(&k3), cache.find(&k4)];

        assert_eq!(hits, vec![None, Some(30), Some(40)]);
    }

    #[test]
    fn test_lsh_fifo_cache_overwrite_behavior() {
        let mut cache = LshFifoCache::new(NUM_HASH, DIM, 2, Some(77));

        let k = TestVecF32(vec![1.0; DIM]);
        cache.insert(k.clone(), 111, TOL);
        assert_eq!(cache.find(&k), Some(111));

        cache.insert(k.clone(), 999, TOL);
        let val = cache.find(&k).unwrap();
        assert!(
            val == 111,
            "FIFO-LSH will match both with a preference for oldest"
        );

        cache.insert(TestVecF32(vec![2.0; DIM]), 222, TOL); // Will evict one version of k, vectors that point in the same direction have the same hash
        let maybe = cache.find(&k);
        assert!(maybe == Some(999), "LSH will find the closest match");
    }

    #[test]
    fn test_lsh_fifo_cache_capacity_one() {
        let mut cache = LshFifoCache::new(NUM_HASH, DIM, 1, Some(321));

        let k1 = TestVecF32(vec![2.0; DIM]);
        let k2 = TestVecF32(vec![1.0; DIM]);
        cache.insert(k1.clone(), 1, TOL);
        assert_eq!(cache.find(&k1), Some(1));

        cache.insert(k2.clone(), 2, TOL);

        let f1 = cache.find(&k1);
        let f2 = cache.find(&k2);
        let hits = vec![f1, f2].into_iter().filter(|x| x.is_some()).count();
        assert_eq!(hits, 1, "Only one key should be in cache due to capacity 1");
    }

    #[test]
    fn test_lsh_lru_cache_basic() {
        let mut cache = LshLruCache::new(NUM_HASH, DIM, BUCKET_CAP, Some(99));

        let k1 = TestVecF32(vec![0.1; DIM]);
        let k2 = TestVecF32(vec![-0.1; DIM]);
        let k3 = TestVecF32(vec![0.1; DIM]); // Same direction as k1

        cache.insert(k1.clone(), 10, TOL);
        cache.insert(k2.clone(), 20, TOL);
        assert_eq!(cache.find(&k1), Some(10));
        cache.insert(k3.clone(), 30, TOL);

        let found = cache.find(&k3);
        assert!(found == Some(30) || found == Some(10));
    }

    #[test]
    fn test_lsh_lru_cache_eviction_order() {
        let mut cache = LshLruCache::new(NUM_HASH, DIM, 2, Some(202));

        let k1 = TestVecF32(vec![1.0; DIM]);
        let k2 = TestVecF32(vec![2.0; DIM]);
        let k3 = TestVecF32(vec![3.0; DIM]);

        cache.insert(k1.clone(), 1, TOL);
        cache.insert(k2.clone(), 2, TOL);
        // Access k1 to make it most recently used
        cache.find(&k1);
        cache.insert(k3.clone(), 3, TOL);

        // k2 should be evicted (least recently used)
        assert_eq!(cache.find(&k1), Some(1));
        assert_eq!(cache.find(&k2), None);
        assert_eq!(cache.find(&k3), Some(3));
    }

    #[test]
    fn test_lsh_lru_cache_overwrite_behavior() {
        let mut cache = LshLruCache::new(NUM_HASH, DIM, 2, Some(303));

        let k = TestVecF32(vec![1.0; DIM]);
        cache.insert(k.clone(), 100, TOL);
        assert_eq!(cache.find(&k), Some(100));
        cache.insert(k.clone(), 200, TOL);

        cache.insert(k.clone(), 999, TOL); // LRU should discard 100
        let val = cache.find(&k);
        assert!(val == Some(999) || val == Some(200));
    }

    #[test]
    fn test_lsh_lru_cache_capacity_one() {
        let mut cache = LshLruCache::new(NUM_HASH, DIM, 1, Some(404));

        let k1 = TestVecF32(vec![2.0; DIM]);
        let k2 = TestVecF32(vec![1.0; DIM]);

        cache.insert(k1.clone(), 1, TOL);
        assert_eq!(cache.find(&k1), Some(1));

        cache.insert(k2.clone(), 2, TOL);
        let hits = vec![cache.find(&k1), cache.find(&k2)];

        assert_eq!(
            hits,
            vec![None, Some(2)],
            "Only one key should remain in cache due to capacity 1"
        );
    }
}
