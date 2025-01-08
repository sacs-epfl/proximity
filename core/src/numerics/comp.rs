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

impl<'a> ApproxComparable for F32Vector<'a> {
    fn roughly_matches(&self, target: &F32Vector<'a>, square_tolerance: f32) -> bool {
        self.l2_dist_squared(target) < square_tolerance
    }
}

impl ApproxComparable for i16 {
    fn roughly_matches(&self, instore: &Self, tolerance: f32) -> bool {
        let fself = f32::from(*self);
        let foth = f32::from(*instore);
        fself.roughly_matches(&foth, tolerance)
    }
}
