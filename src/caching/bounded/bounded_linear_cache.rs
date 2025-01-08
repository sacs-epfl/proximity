use std::collections::HashMap;
use std::hash::Hash;

use pyo3::{pyclass, pymethods};

use crate::numerics::comp::ApproxComparable;

use crate::caching::approximate_cache::ApproximateCache;

use super::linked_list::DoublyLinkedList;
use super::list_node::{Node, SharedNode};

/// `BoundedLinearCache` is a bounded cache with approximate key matching support.
///
/// The cache enforces a maximum capacity, and when the capacity is exceeded, the least recently used (LRU) element is evicted.
///
/// # Approximate Key Matching
/// Keys must implement the `ApproxComparable` trait, which allows approximate equality comparisons based on the provided `tolerance`.
/// This enables the cache to retrieve values even when the queried key is not an exact match but is "close enough."
///
/// # Example Usage
/// ```
/// use proximitylib::caching::bounded::bounded_linear_cache::BoundedLinearCache;
/// use proximitylib::caching::approximate_cache::ApproximateCache;
///
/// let mut cache = BoundedLinearCache::new(3, 2.0);
///
/// cache.insert(10 as i16, "Value 1");
/// cache.insert(20, "Value 2");
/// cache.insert(30, "Value 3");
///
/// assert_eq!(cache.find(&11), Some("Value 1"));
/// assert_eq!(cache.len(), 3);
///
/// cache.insert(40, "Value 4"); // Evicts the least recently used (Key(20))
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
pub struct BoundedLinearCache<K, V> {
    max_capacity: usize,
    map: HashMap<K, SharedNode<K, V>>,
    list: DoublyLinkedList<K, V>,
    tolerance: f32,
}

impl<K, V> ApproximateCache<K, V> for BoundedLinearCache<K, V>
where
    K: ApproxComparable + Eq + Hash + Clone,
    V: Clone,
{
    fn find(&mut self, key: &K) -> Option<V> {
        let matching = self
            .map
            .keys()
            .find(|&k| key.roughly_matches(k, self.tolerance))?;
        let node: SharedNode<K, V> = self.map.get(matching).cloned()?;
        self.list.remove(node.clone());
        self.list.add_to_head(node.clone());
        return Some(node.borrow().value.clone());
    }

    fn insert(&mut self, key: K, value: V) {
        if self.len() == self.max_capacity {
            if let Some(tail) = self.list.remove_tail() {
                self.map.remove(&tail.borrow().key);
            }
        }
        let new_node = Node::new(key.clone(), value);
        self.list.add_to_head(new_node.clone());
        self.map.insert(key, new_node);
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

impl<K, V> BoundedLinearCache<K, V> {
    pub fn new(max_capacity: usize, tolerance: f32) -> Self {
        assert!(max_capacity > 0);
        assert!(tolerance > 0.0);
        Self {
            max_capacity,
            map: HashMap::new(),
            list: DoublyLinkedList::new(),
            tolerance,
        }
    }
}

macro_rules! create_pythonized_interface {
    ($name: ident, $type: ident) => {
        // unsendable == should hard-crash if Python tries to access it from
        // two different Python threads.
        //
        // The implementation is very much thread-unsafe anyways (lots of mutations),
        // so this is an OK behavior, we will detect it with a nice backtrace
        // and without UB.
        //
        // Even in the case where we want the cache to be multithreaded, this would
        // happen on the Rust side and will not be visible to the Python ML pipeline.
        #[pyclass(unsendable)]
        pub struct $name {
            inner: BoundedLinearCache<$type, $type>,
        }

        #[pymethods]
        impl $name {
            #[new]
            pub fn new(max_capacity: usize, tolerance: f32) -> Self {
                Self {
                    inner: BoundedLinearCache::new(max_capacity, tolerance),
                }
            }

            fn find(&mut self, k: $type) -> Option<$type> {
                self.inner.find(&k)
            }

            fn insert(&mut self, key: $type, value: $type) {
                self.inner.insert(key, value)
            }

            fn len(&self) -> usize {
                self.inner.len()
            }
        }
    };
}

create_pythonized_interface!(I16Cache, i16);

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOLERANCE: f32 = 1e-8;
    #[test]
    fn test_lru_cache_basic_operations() {
        let mut cache: BoundedLinearCache<i16, i16> = BoundedLinearCache::new(2, TEST_TOLERANCE);
        cache.insert(1, 1); // Cache is {1=1}
        cache.insert(2, 2); // Cache is {1=1, 2=2}
        assert_eq!(cache.find(&1), Some(1)); // Returns 1, Cache is {2=2, 1=1}
        cache.insert(3, 3); // Evicts key 2, Cache is {1=1, 3=3}
        assert_eq!(cache.find(&2), None); // Key 2 not found
        cache.insert(4, 4); // Evicts key 1, Cache is {3=3, 4=4}
        assert_eq!(cache.find(&1), None); // Key 1 not found
        assert_eq!(cache.find(&3), Some(3)); // Returns 3
        assert_eq!(cache.find(&4), Some(4)); // Returns 4
    }

    #[test]
    fn test_lru_cache_eviction_order() {
        let mut cache: BoundedLinearCache<i16, i16> = BoundedLinearCache::new(3, TEST_TOLERANCE);
        cache.insert(1, 1); // Cache is {1=1}
        cache.insert(2, 2); // Cache is {1=1, 2=2}
        cache.insert(3, 3); // Cache is {1=1, 2=2, 3=3}
        cache.find(&1); // Access key 1, Cache is {2=2, 3=3, 1=1}
        cache.insert(4, 4); // Evicts key 2, Cache is {3=3, 1=1, 4=4}
        assert_eq!(cache.find(&2), None); // Key 2 not found
        assert_eq!(cache.find(&3), Some(3)); // Returns 3
        assert_eq!(cache.find(&4), Some(4)); // Returns 4
        assert_eq!(cache.find(&1), Some(1)); // Returns 1
    }

    #[test]
    fn test_lru_cache_overwrite() {
        let mut cache: BoundedLinearCache<i16, i16> = BoundedLinearCache::new(2, TEST_TOLERANCE);
        cache.insert(1, 1); // Cache is {1=1}
        cache.insert(2, 2); // Cache is {1=1, 2=2}
        cache.insert(1, 10); // Overwrites key 1, Cache is {2=2, 1=10}
        assert_eq!(cache.find(&1), Some(10)); // Returns 10
        cache.insert(3, 3); // Evicts key 2, Cache is {1=10, 3=3}
        assert_eq!(cache.find(&2), None); // Key 2 not found
        assert_eq!(cache.find(&3), Some(3)); // Returns 3
    }

    #[test]
    fn test_lru_cache_capacity_one() {
        let mut cache: BoundedLinearCache<i16, i16> = BoundedLinearCache::new(1, TEST_TOLERANCE);
        cache.insert(1, 1); // Cache is {1=1}
        assert_eq!(cache.find(&1), Some(1)); // Returns 1
        cache.insert(2, 2); // Evicts key 1, Cache is {2=2}
        assert_eq!(cache.find(&1), None); // Key 1 not found
        assert_eq!(cache.find(&2), Some(2)); // Returns 2
    }

    #[test]
    #[should_panic]
    fn test_lru_cache_empty() {
        let _cache: BoundedLinearCache<i16, i16> = BoundedLinearCache::new(0, TEST_TOLERANCE);
    }
}
