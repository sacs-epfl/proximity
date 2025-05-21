use std::collections::VecDeque;

use crate::caching::approximate_cache::ApproximateCache;
use crate::caching::approximate_cache::Tolerance;
use crate::numerics::ApproxComparable;

#[derive(Clone)]
struct CacheLine<K, V> {
    key: K,
    tol: Tolerance,
    value: V,
}

pub struct FifoCache<K, V> {
    max_capacity: usize,
    items: VecDeque<CacheLine<K, V>>,
}

impl<K, V> ApproximateCache<K, V> for FifoCache<K, V>
where
    K: ApproxComparable,
    V: Clone,
{
    fn find(&mut self, target: &K) -> Option<V> {
        let candidate = self
            .items
            .iter()
            .filter(|&entry| entry.key.roughly_matches(target, entry.tol))
            .min_by(|&x, &y| {
                target
                    .fuzziness(&x.key)
                    .partial_cmp(&target.fuzziness(&y.key))
                    .unwrap() // finding NaNs here should crash the program
            })?;
        Some(candidate.value.clone())
    }

    fn insert(&mut self, key: K, value: V, tolerance: f32) {
        let new_entry = CacheLine {
            key,
            tol: tolerance,
            value,
        };
        self.items.push_back(new_entry);
        if self.items.len() > self.max_capacity {
            self.items.pop_front();
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

impl<K, V> FifoCache<K, V> {
    pub fn new(max_capacity: usize) -> Self {
        assert!(max_capacity > 0);
        Self {
            max_capacity,
            items: VecDeque::with_capacity(max_capacity),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOLERANCE: f32 = 1e-8;
    #[test]
    fn test_fifo_cache_basic_operations() {
        let mut cache = FifoCache::new(2);
        cache.insert(1, 1, TEST_TOLERANCE); // Cache is {1=1}
        cache.insert(2, 2, TEST_TOLERANCE); // Cache is {1=1, 2=2}
        assert_eq!(cache.find(&1), Some(1)); // Returns 1, Cache is {1=1, 2=2}
        cache.insert(3, 3, TEST_TOLERANCE); // Evicts key 1, Cache is {2=2, 3=3}
        assert_eq!(cache.find(&1), None); // Key 1 not found
        assert_eq!(cache.find(&2), Some(2)); // Returns 2
        assert_eq!(cache.find(&3), Some(3)); // Returns 3
    }

    #[test]
    fn test_fifo_cache_eviction_order() {
        let mut cache = FifoCache::new(3);
        cache.insert(1, 1, TEST_TOLERANCE); // Cache is {1=1}
        cache.insert(2, 2, TEST_TOLERANCE); // Cache is {1=1, 2=2}
        cache.insert(3, 3, TEST_TOLERANCE); // Cache is {1=1, 2=2, 3=3}
        cache.insert(4, 4, TEST_TOLERANCE); // Evicts key 1, Cache is {2=2, 3=3, 4=4}
        assert_eq!(cache.find(&1), None); // Key 1 not found
        assert_eq!(cache.find(&2), Some(2)); // Returns 2
        assert_eq!(cache.find(&3), Some(3)); // Returns 3
        assert_eq!(cache.find(&4), Some(4)); // Returns 4
    }

    #[test]
    fn test_fifo_cache_overwrite() {
        let mut cache = FifoCache::new(2);
        cache.insert(1, 1, TEST_TOLERANCE); // Cache is {1=1}
        cache.insert(2, 2, TEST_TOLERANCE); // Cache is {1=1, 2=2}
        cache.insert(1, 10, TEST_TOLERANCE); // Overwrites key 1, Cache is {2=2, 1=10}
        assert_eq!(cache.find(&1), Some(10)); // Returns 10
        cache.insert(3, 3, TEST_TOLERANCE); // Evicts key 2, Cache is {1=10, 3=3}
        assert_eq!(cache.find(&2), None); // Key 2 not found
        assert_eq!(cache.find(&3), Some(3)); // Returns 3
    }

    #[test]
    fn test_fifo_cache_capacity_one() {
        let mut cache = FifoCache::new(1);
        cache.insert(1, 1, TEST_TOLERANCE); // Cache is {1=1}
        assert_eq!(cache.find(&1), Some(1)); // Returns 1
        cache.insert(2, 2, TEST_TOLERANCE); // Evicts key 1, Cache is {2=2}
        assert_eq!(cache.find(&1), None); // Key 1 not found
        assert_eq!(cache.find(&2), Some(2)); // Returns 2
    }

    #[test]
    #[should_panic]
    fn test_fifo_cache_empty() {
        let _cache: FifoCache<i16, i16> = FifoCache::new(0);
    }
}
