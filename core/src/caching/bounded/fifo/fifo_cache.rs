use std::collections::VecDeque;

use crate::{caching::approximate_cache::ApproximateCache, numerics::comp::ApproxComparable};

pub struct FifoCache<K, V> {
    max_capacity: usize,
    items: VecDeque<(K, V)>,
    tolerance: f32,
}

impl<K, V> ApproximateCache<K, V> for FifoCache<K, V>
where
    K: ApproxComparable,
    V: Clone,
{
    fn find(&mut self, key: &K) -> Option<V> {
        let candidate = self
            .items
            .iter()
            .min_by(|&(x, _), &(y, _)| key.fuzziness(x).partial_cmp(&key.fuzziness(y)).unwrap())?;
        let (c_key, c_value) = candidate;
        if c_key.roughly_matches(key, self.tolerance) {
            Some(c_value.clone())
        } else {
            None
        }
    }

    fn insert(&mut self, key: K, value: V) {
        self.items.push_back((key, value));
        if self.items.len() > self.max_capacity {
            self.items.pop_front();
        }
    }

    fn len(&self) -> usize {
        self.items.iter().len()
    }
}
