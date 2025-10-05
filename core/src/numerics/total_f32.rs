use std::hash::Hash;

pub struct TotalF32 {
    pub inner: f32,
}

impl PartialEq for TotalF32 {
    fn eq(&self, other: &Self) -> bool {
        self.inner.to_bits() == other.inner.to_bits()
    }
}

impl Eq for TotalF32 {}

impl Hash for TotalF32 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.to_bits().hash(state);
    }
}

impl PartialOrd for TotalF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.inner.total_cmp(&other.inner))
    }
}

impl Ord for TotalF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.inner.total_cmp(&other.inner)
    }
}
