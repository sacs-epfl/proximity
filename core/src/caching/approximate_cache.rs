use crate::numerics::ApproxComparable;

pub type Tolerance = f32;

pub trait ApproximateCache<K, V>
where
    K: ApproxComparable,
{
    fn find(&mut self, target: &K) -> Option<V>;
    fn insert(&mut self, key: K, value: V, tolerance: f32);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait DefaultApproximateCache<K, V> : ApproximateCache<K, V> where K : ApproxComparable {
    fn from_capacity(cap : usize) -> Self;
}