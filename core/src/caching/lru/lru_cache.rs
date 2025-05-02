use std::collections::HashMap;
use std::hash::Hash;

use crate::numerics::ApproxComparable;

use crate::caching::approximate_cache::ApproximateCache;

use super::linked_list::DoublyLinkedList;
use super::list_node::{Node, SharedNode};
use super::map_entry::MapEntry;

/// `LRUCache` is a bounded cache with approximate key matching support and LRU eviction.
///
/// # Approximate Key Matching
/// Keys must implement the `ApproxComparable` trait, which allows approximate equality comparisons based on the provided `tolerance`.
/// This enables the cache to retrieve values even when the queried key is not an exact match but is "close enough."
///
/// # Example Usage
/// ```
/// use proximity::caching::LRUCache;
/// use proximity::caching::ApproximateCache;
///
/// let mut cache = LRUCache::new(3);
/// const TEST_TOL: f32 = 2.0;
///
/// cache.insert(10 as i16, "Value 1", TEST_TOL);
/// cache.insert(20, "Value 2", TEST_TOL);
/// cache.insert(30, "Value 3", TEST_TOL);
///
/// assert_eq!(cache.find(&11), Some("Value 1"));
/// assert_eq!(cache.len(), 3);
///
/// cache.insert(40, "Value 4", TEST_TOL); // Evicts the least recently used (Key(20))
/// assert!(cache.find(&20).is_none());
/// ```
///
/// # Type Parameters
/// - `K`: The type of the keys, which must implement `ApproxComparable`, `Eq`, `Hash`, and `Clone`.
/// - `V`: The type of the values, which must implement `Clone`.
///
/// # Methods
/// - `new(max_capacity: usize, tolerance: f32) -> Self`: Creates a new `BoundedLinearCache` with the specified maximum capacity and tolerance.
/// - `find(&mut self, key: &K) -> Option<V>`: Attempts to find a value matching the given key approximately. Promotes the found key to the head of the list.
/// - `insert(&mut self, key: K, value: V)`: Inserts a key-value pair into the cache. Evicts the least recently used item if the cache is full.
/// - `len(&self) -> usize`: Returns the current size of the cache.
pub struct LRUCache<K, V> {
    max_capacity: usize,
    map: HashMap<MapEntry<K>, SharedNode<MapEntry<K>, V>>,
    list: DoublyLinkedList<MapEntry<K>, V>,
}

impl<K, V> ApproximateCache<K, V> for LRUCache<K, V>
where
    K: ApproxComparable + Eq + Hash + Clone,
    V: Clone,
{
    fn find(&mut self, target: &K) -> Option<V> {
        let candidate = self
            .map
            .keys()
            .filter(|&entry| entry.key.roughly_matches(target, entry.tolerance))
            .min_by(|&xentry, &yentry| {
                target
                    .fuzziness(&xentry.key)
                    .partial_cmp(&target.fuzziness(&yentry.key))
                    .unwrap()
            })?;

        let node: SharedNode<MapEntry<K>, V> = self.map.get(candidate).cloned()?;
        self.list.remove(node.clone());
        self.list.add_to_head(node.clone());
        return Some(node.borrow().value.clone());
    }

    fn insert(&mut self, key: K, value: V, tolerance: f32) {
        if self.len() >= self.max_capacity {
            if let Some(tail) = self.list.remove_tail() {
                self.map.remove(&tail.borrow().key);
            }
        }
        let map_entry = MapEntry {
            key: key.clone(),
            tolerance,
        };
        let new_node = Node::new(map_entry.clone(), value);
        self.list.add_to_head(new_node.clone());
        self.map.insert(map_entry, new_node);
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

impl<K, V> LRUCache<K, V> {
    pub fn new(max_capacity: usize) -> Self {
        assert!(max_capacity > 0);
        Self {
            max_capacity,
            map: HashMap::with_capacity(max_capacity),
            list: DoublyLinkedList::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOLERANCE: f32 = 1e-8;
    #[test]
    fn test_lru_cache_basic_operations() {
        let mut cache = LRUCache::new(2);
        cache.insert(1, 1, TEST_TOLERANCE); // Cache is {1=1}
        cache.insert(2, 2, TEST_TOLERANCE); // Cache is {1=1, 2=2}
        assert_eq!(cache.find(&1), Some(1)); // Returns 1, Cache is {2=2, 1=1}
        cache.insert(3, 3, TEST_TOLERANCE); // Evicts key 2, Cache is {1=1, 3=3}
        assert_eq!(cache.find(&2), None); // Key 2 not found
        cache.insert(4, 4, TEST_TOLERANCE); // Evicts key 1, Cache is {3=3, 4=4}
        assert_eq!(cache.find(&1), None); // Key 1 not found
        assert_eq!(cache.find(&3), Some(3)); // Returns 3
        assert_eq!(cache.find(&4), Some(4)); // Returns 4
    }

    #[test]
    fn test_lru_cache_eviction_order() {
        let mut cache = LRUCache::new(3);
        cache.insert(1, 1, TEST_TOLERANCE); // Cache is {1=1}
        cache.insert(2, 2, TEST_TOLERANCE); // Cache is {1=1, 2=2}
        cache.insert(3, 3, TEST_TOLERANCE); // Cache is {1=1, 2=2, 3=3}
        cache.find(&1); // Access key 1, Cache is {2=2, 3=3, 1=1}
        cache.insert(4, 4, TEST_TOLERANCE); // Evicts key 2, Cache is {3=3, 1=1, 4=4}
        assert_eq!(cache.find(&2), None); // Key 2 not found
        assert_eq!(cache.find(&3), Some(3)); // Returns 3
        assert_eq!(cache.find(&4), Some(4)); // Returns 4
        assert_eq!(cache.find(&1), Some(1)); // Returns 1
    }

    #[test]
    fn test_lru_cache_overwrite() {
        let mut cache = LRUCache::new(2);
        cache.insert(1, 1, TEST_TOLERANCE); // Cache is {1=1}
        cache.insert(2, 2, TEST_TOLERANCE); // Cache is {1=1, 2=2}
        cache.insert(1, 10, TEST_TOLERANCE); // Overwrites key 1, Cache is {2=2, 1=10}
        assert_eq!(cache.find(&1), Some(10)); // Returns 10
        cache.insert(3, 3, TEST_TOLERANCE); // Evicts key 2, Cache is {1=10, 3=3}
        assert_eq!(cache.find(&2), None); // Key 2 not found
        assert_eq!(cache.find(&3), Some(3)); // Returns 3
    }

    #[test]
    fn test_lru_cache_capacity_one() {
        let mut cache = LRUCache::new(1);
        cache.insert(1, 1, TEST_TOLERANCE); // Cache is {1=1}
        assert_eq!(cache.find(&1), Some(1)); // Returns 1
        cache.insert(2, 2, TEST_TOLERANCE); // Evicts key 1, Cache is {2=2}
        assert_eq!(cache.find(&1), None); // Key 1 not found
        assert_eq!(cache.find(&2), Some(2)); // Returns 2
    }

    #[test]
    #[should_panic]
    fn test_lru_cache_empty() {
        let _cache: LRUCache<i16, i16> = LRUCache::new(0);
    }
}
