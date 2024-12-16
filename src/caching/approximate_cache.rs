use crate::numerics::comp::ApproxComparable;

// size of caches in implementations where that should be known at comptime
pub const COMPTIME_CACHE_SIZE: usize = 1024;

pub trait ApproximateCache<K, V>
where
    K: ApproxComparable,
    V: Clone,
{
    fn find(&mut self, key: &K) -> Option<V>;
    fn insert(&mut self, key: K, value: V);
    fn len(&self) -> usize;
}
