use crate::cache::cache::*;
use crate::numerics::comp::*;

/// A cache implementation that checks all entries one-by-one, without eviction
/// # Generic Types
/// K and V for the cache keys and values
/// Comp : a type that implements a comparison method for keys
struct InfiniteLinearCache<K, V> {
    lines: Vec<(K, V)>,
    tolerance: f32,
}

impl<K, V> ApproximateCache<K, V> for InfiniteLinearCache<K, V>
where
    K: ApproxComparable,
    V: Clone,
{
    // to find a match in an infinite cache, iterate over all cache lines
    // and return early if you have something
    fn find(&self, key: &K) -> Option<V> {
        let x = self
            .lines
            .iter()
            .find(|&(k, _v)| key.roughly_matches(k, self.tolerance));
        match x {
            None => None,
            Some((_u, v)) => Some(v.clone()),
        }
    }

    // to insert a new value in a
    fn insert(&mut self, key: K, value: V) {
        self.lines.push((key, value));
    }

    fn len(&self) -> usize {
        self.lines.len()
    }
}
