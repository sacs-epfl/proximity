use super::f32vector::F32Vector;

// rust Ord trait has some issues
pub trait ApproxComparable {
    fn roughly_matches(&self, instore: &Self, tolerance: f32) -> bool;
}

impl ApproxComparable for f32 {
    fn roughly_matches(&self, target: &f32, tolerance: f32) -> bool {
        (self - target).abs() < tolerance
    }
}

// f32 comparisons are implemented in raw aarch64, we can use them conditionally
impl<'a> ApproxComparable for F32Vector<'a> {
    #[cfg(target_arch = "aarch64")]
    fn roughly_matches(&self, target: &F32Vector<'a>, square_tolerance: f32) -> bool {
        self.l2_dist_aarch64(target) < square_tolerance
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn roughly_matches(&self, target: &F32Vector<'a>, square_tolerance: f32) -> bool {
        self.l2_dist_squared(target) < square_tolerance
    }
}
