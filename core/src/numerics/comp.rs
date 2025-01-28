use super::f32vector::F32Vector;

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

impl<'a> ApproxComparable for F32Vector<'a> {
    fn roughly_matches(&self, target: &F32Vector<'a>, tolerance: f32) -> bool {
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
