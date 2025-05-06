use crate::caching::approximate_cache::ApproximateCache;
use crate::caching::approximate_cache::Tolerance;
use crate::caching::FifoCache;
use crate::caching::LruCache;

use crate::caching::lsh::hasher::SimHashHasher;
use crate::numerics::ApproxComparable;
use crate::numerics::F32Vector;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

/// A key-value store that uses cosine LSH to direct queries into fixed-size cache buckets.
pub struct LshCache<K: ApproxComparable, V, C>
where
    C: ApproximateCache<K, V>,
{
    hasher: SimHashHasher,
    buckets: HashMap<Vec<bool>, C>,
    bucket_capacity: usize,
    phantomas: PhantomData<(K, V)>,
}

impl<K: ApproxComparable + AsRef<[f32]>, V, C: ApproximateCache<K, V>> LshCache<K, V, C>
where
    V: Clone,
{
    /// Create a new LSH-based FIFO cache.
    ///
    /// `num_hash` and `dim` configure the random hyperplane hasher,
    /// `bucket_capacity` is the max size of each FIFO bucket,
    /// `tolerance` is used for approximate key matching in each bucket.
    pub fn new(num_hash: usize, dim: usize, bucket_capacity: usize, seed: Option<u64>) -> Self {
        let hasher = if let Some(seed) = seed {
            SimHashHasher::new_seeded(num_hash, dim, seed)
        } else {
            SimHashHasher::new(num_hash, dim)
        };

        Self {
            hasher,
            buckets: HashMap::new(),
            bucket_capacity,
            phantomas: PhantomData,
        }
    }

    /// Compute the LSH signature for a key (after normalization).
    fn signature(&self, key: &[f32]) -> Vec<bool> {
        self.hasher.hash(F32Vector::from(key).normalized().as_ref())
    }

    fn find(&mut self, target: &K) -> Option<V> {
        let sig = self.signature(target.as_ref());
        let bucket = self.buckets.get_mut(&sig)?;
        bucket.find(target)
    }

    fn len(&self) -> usize {
        self.buckets.values().map(|b| b.len()).sum()
    }
}

impl<K: ApproxComparable + AsRef<[f32]>, V> ApproximateCache<K, V>
    for LshCache<K, V, FifoCache<K, V>>
where
    V: Clone,
{
    /// Find a value by key, mutably accessing the bucket for potential reordering.
    fn find(&mut self, target: &K) -> Option<V> {
        self.find(target)
    }

    /// Insert a key-value pair, normalizing the key before hashing and storing.
    fn insert(&mut self, key: K, value: V, tol: f32) {
        let sig = self.signature(key.as_ref());
        self.buckets
            .entry(sig)
            .or_insert_with(|| FifoCache::new(self.bucket_capacity))
            .insert(key, value, tol);
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<K, V> ApproximateCache<K, V> for LshCache<K, V, LruCache<K, V>>
where
    K: ApproxComparable + AsRef<[f32]> + Eq + Hash + Clone,
    V: Clone,
{
    /// Find a value by key, mutably accessing the bucket for potential reordering.
    fn find(&mut self, target: &K) -> Option<V> {
        self.find(target)
    }

    /// Insert a key-value pair, normalizing the key before hashing and storing.
    fn insert(&mut self, key: K, value: V, tol: f32) {
        let sig = self.signature(key.as_ref());
        self.buckets
            .entry(sig)
            .or_insert_with(|| LruCache::new(self.bucket_capacity))
            .insert(key, value, tol);
    }

    fn len(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 8;
    const NUM_HASH: usize = 8;
    const BUCKET_CAP: usize = 2;
    const TOL: f32 = 1e-6;

    #[test]
    fn test_lsh_fifo_cache_basic() {
        let mut cache = LshCache::new(NUM_HASH, DIM, BUCKET_CAP, Some(42));

        let k1: Vec<f32> = vec![0.1; DIM];
        let k2: Vec<f32> = vec![-0.2; DIM];
        let k3: Vec<f32> = vec![0.1; DIM]; // same direction as k1

        cache.insert(k1.clone(), 1, TOL);
        cache.insert(k2, 2, TOL);
        assert_eq!(cache.find(&k1), Some(1));
        cache.insert(k3.clone(), 3, TOL);
        // Bucket for k1 now has entries [k1=1, k3=3], matches equally
        assert!(cache.find(&k3) == Some(3) || cache.find(&k3) == Some(1));
    }

    #[test]
    fn test_lsh_fifo_cache_eviction_order() {
        let mut cache = LshCache::new(NUM_HASH, DIM, 2, Some(123));

        let k2 = vec![1.0; DIM];
        let k3 = vec![2.0; DIM];
        let k4 = vec![3.0; DIM];

        cache.insert(k2.clone(), 20, TOL);
        cache.insert(k3.clone(), 30, TOL);
        cache.insert(k4.clone(), 40, TOL);

        // One of the earlier keys must have been evicted (depending on bucket)
        let hits = vec![cache.find(&k2), cache.find(&k3), cache.find(&k4)];

        assert_eq!(hits, vec![None, Some(30), Some(40)]);
    }

    #[test]
    fn test_lsh_fifo_cache_overwrite_behavior() {
        let mut cache = LshCache::new(NUM_HASH, DIM, 2, Some(77));

        let k = vec![1.0; DIM];
        cache.insert(k.clone(), 111, TOL);
        assert_eq!(cache.find(&k), Some(111));

        cache.insert(k.clone(), 999, TOL);
        let val = cache.find(&k).unwrap();
        assert!(
            val == 111,
            "FIFO-LSH will match both with a preference for oldest"
        );

        cache.insert(vec![2.0; DIM], 222, TOL); // Will evict one version of k, vectors that point in the same direction have the same hash
        let maybe = cache.find(&k);
        assert!(maybe == Some(999), "LSH will find the closest match");
    }

    #[test]
    fn test_lsh_fifo_cache_capacity_one() {
        let mut cache = LshCache::new(NUM_HASH, DIM, 1, Some(321));

        let k1 = vec![2.0; DIM];
        let k2 = vec![1.0; DIM];
        cache.insert(k1.clone(), 1, TOL);
        assert_eq!(cache.find(&k1), Some(1));

        cache.insert(k2.clone(), 2, TOL);

        let f1 = cache.find(&k1);
        let f2 = cache.find(&k2);
        let hits = vec![f1, f2].into_iter().filter(|x| x.is_some()).count();
        assert_eq!(hits, 1, "Only one key should be in cache due to capacity 1");
    }
}
