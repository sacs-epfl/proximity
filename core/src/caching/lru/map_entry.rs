use std::hash::Hash;

#[derive(Clone)]
pub struct MapEntry<K> {
    pub key: K,
    pub tolerance: f32,
}

impl<K: Eq> PartialEq for MapEntry<K> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.tolerance == other.tolerance
    }
}

impl<K: Eq> Eq for MapEntry<K> {}

impl<K: Hash> Hash for MapEntry<K> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
        self.tolerance.to_bits().hash(state);
    }
}
