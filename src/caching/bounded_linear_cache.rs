use crate::numerics::comp::ApproxComparable;

use super::approximate_cache::ApproximateCache;

struct BoundedLinearCache {}

impl<K, V> ApproximateCache<K, V> for BoundedLinearCache
where
    V: Clone,
    K: ApproxComparable,
{
    fn find(&self, _key: &K) -> Option<V> {
        todo!()
    }

    fn insert(&mut self, _key: K, _value: V) {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }
}
