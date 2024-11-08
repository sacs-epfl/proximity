pub struct Vector<'a> {
    _repr: &'a [f32],
}

impl<'a> Vector<'a> {
    pub fn len(&self) -> usize {
        self._repr.len()
    }

    /// Computes the **SQUARED** L2 distance between two vectors. This is cheaper to compute than the regular L2 distance. This is typically useful when comparing two distances : dist(u,v) < dist(w, x) <=> dist(u,v) ** 2 < dist(w,x) ** 2
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the two vectors have different lengths. In release mode, the longest vector will
    /// be silently truncated.
    #[inline]
    pub fn l2_dist_square(&self, other: &Vector<'a>) -> f32 {
        debug_assert!(self.len() == other.len());

        self._repr
            .iter()
            .zip(other._repr)
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
    }

    #[inline]
    pub fn l2_dist(&self, other: &Vector<'a>) -> f32 {
        self.l2_dist_square(other).sqrt()
    }
}

impl<'a> From<&'a [f32]> for Vector<'a> {
    fn from(value: &'a [f32]) -> Self {
        Vector { _repr: value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1e-8;

    fn close(actual: f32, target: f32) -> bool {
        (target - actual).abs() < TOLERANCE
    }

    fn check_l2(suspect: f32) -> bool {
        suspect.is_finite() && suspect >= 0.0
    }

    #[test]
    fn self_sim_is_zero() {
        let testvec = Vector::from([1.0, 2.0, 3.0, 1.5].as_slice());
        let selfsim = testvec.l2_dist(&testvec);
        assert!(check_l2(selfsim));
        assert!(close(selfsim, 0.0));
    }
}
