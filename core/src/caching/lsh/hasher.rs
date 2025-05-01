use rand::rngs::{StdRng, ThreadRng};
use rand::{rng, Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::numerics::f32vector::{F32Vector, SIMD_LANECOUNT};

pub struct SimHashHasher {
    pub stored_vectors_dim: usize,
    /// random hyperplane normals
    projections: Vec<Vec<f32>>,
}

impl SimHashHasher {
    /// Constructs a new hasher with `num_hash` hyperplanes in dimension `dim`,
    /// using a non‐seeded thread RNG.
    pub fn new(num_hash: usize, stored_vectors_dim: usize) -> Self {
        let mut rng: ThreadRng = rng();
        Self::with_rng(num_hash, stored_vectors_dim, &mut rng)
    }

    /// Constructs a new hasher with `num_hash` hyperplanes in dimension `dim`,
    /// seeded from the given `seed`. This is deterministic.
    pub fn new_seeded(num_hash: usize, stored_vectors_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::with_rng(num_hash, stored_vectors_dim, &mut rng)
    }

    /// Internal constructor from any Rng.
    fn with_rng<R: Rng>(num_hash: usize, dim: usize, rng: &mut R) -> Self {
        assert!(
            dim % SIMD_LANECOUNT == 0,
            "dim must be multiple of SIMD_LANECOUNT"
        );

        let mut gaussian_iter = rng.sample_iter(StandardNormal);
        let projections: Vec<Vec<f32>> = (0..num_hash)
            .map(|_| (0..dim).map(|_| gaussian_iter.next().unwrap()).collect())
            .collect();

        SimHashHasher {
            stored_vectors_dim: dim,
            projections,
        }
    }

    /// Hashes `vector` to a `k`-length binary signature.
    pub fn hash(&self, vector: &[f32]) -> Vec<bool> {
        debug_assert!(
            vector.len() == self.stored_vectors_dim,
            "input vector has wrong dimension"
        );
        self.projections
            .iter()
            .map(|proj| {
                let dot: f32 = F32Vector::from(vector).dot(&F32Vector::from(proj as &[f32]));
                dot >= 0.0
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_seeded_determinism() {
        let h1 = SimHashHasher::new_seeded(16, SIMD_LANECOUNT * 2, 12345);
        let h2 = SimHashHasher::new_seeded(16, SIMD_LANECOUNT * 2, 12345);
        // Same seed → identical projections
        assert_eq!(h1.projections, h2.projections);
    }

    #[test]
    fn test_new_and_new_seeded_difference() {
        // new() is unseeded, so overwhelmingly likely to differ from a seeded one
        let h_unseeded = SimHashHasher::new(16, SIMD_LANECOUNT * 2);
        let h_seeded = SimHashHasher::new_seeded(16, SIMD_LANECOUNT * 2, 12345);
        assert_ne!(h_unseeded.projections, h_seeded.projections);
    }

    #[test]
    fn test_hash_seeded_matches_with_rng() {
        // compare new_seeded against manual with_rng
        let mut rng1 = StdRng::seed_from_u64(42);
        let manual = SimHashHasher::with_rng(8, SIMD_LANECOUNT, &mut rng1);
        let seeded = SimHashHasher::new_seeded(8, SIMD_LANECOUNT, 42);
        assert_eq!(manual.projections, seeded.projections);
    }
}
