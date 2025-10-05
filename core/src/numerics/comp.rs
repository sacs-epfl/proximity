use crate::numerics::VectorLike;

pub trait ApproxComparable {
    #[inline]
    fn roughly_matches(&self, instore: &Self, tolerance: f32) -> bool {
        self.fuzziness(instore) < tolerance
    }
    fn fuzziness(&self, instore: &Self) -> f32;
}

impl ApproxComparable for f32 {
    fn fuzziness(&self, instore: &Self) -> f32 {
        (self - instore).abs()
    }
}

impl ApproxComparable for [f32] {
    #[inline]
    fn roughly_matches(&self, target: &[f32], tolerance: f32) -> bool {
        self.l2_dist_squared(target) < tolerance * tolerance
    }

    #[inline]
    fn fuzziness(&self, instore: &Self) -> f32 {
        self.l2_dist_squared(instore).sqrt()
    }
}

impl ApproxComparable for Vec<f32> {
    fn roughly_matches(&self, target: &Vec<f32>, tolerance: f32) -> bool {
        self.l2_dist_squared(target) < tolerance * tolerance
    }

    fn fuzziness(&self, instore: &Self) -> f32 {
        self.l2_dist_squared(instore).sqrt()
    }
}

impl ApproxComparable for i16 {
    fn fuzziness(&self, instore: &Self) -> f32 {
        let fself = f32::from(*self);
        let foth = f32::from(*instore);
        fself.fuzziness(&foth)
    }
}
