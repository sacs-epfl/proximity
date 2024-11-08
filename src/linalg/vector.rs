pub struct Vector<'a> {
    _repr: &'a [f32],
}

impl<'a> Vector<'a> {
    pub fn len(&self) -> usize {
        self._repr.len()
    }

    pub fn l2_sim(&self, other: &Vector<'a>) -> f32 {
        debug_assert!(self.len() == other.len());

        self._repr
            .iter()
            .zip(other._repr)
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
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

    fn unsuspicious_l2(suspect: f32) -> bool {
        suspect.is_finite() && suspect >= 0.0
    }

    #[test]
    fn self_sim_is_zero() {
        let testvec = Vector::from([1.0, 2.0, 3.0, 1.5].as_slice());
        let selfsim = testvec.l2_sim(&testvec);
        assert!(unsuspicious_l2(selfsim));
        assert!(close(selfsim, 0.0));
    }
}
