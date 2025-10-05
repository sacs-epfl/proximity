use rand::rngs::{StdRng, ThreadRng};
use rand::{rng, Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::numerics::{VectorLike, SIMD_LANECOUNT};

pub struct SimHashHasher {
    stored_vectors_dim: usize,
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

    fn with_rng<R: Rng>(num_hash: usize, stored_vectors_dim: usize, rng: &mut R) -> Self {
        assert!(
            stored_vectors_dim % SIMD_LANECOUNT == 0,
            "dim must be multiple of SIMD_LANECOUNT"
        );

        let mut gaussian_iter = rng.sample_iter(StandardNormal);
        let projections: Vec<Vec<f32>> = (0..num_hash)
            .map(|_| {
                (0..stored_vectors_dim)
                    .map(|_| gaussian_iter.next().unwrap())
                    .collect()
            })
            .collect();

        SimHashHasher {
            stored_vectors_dim,
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
                let dot: f32 = vector.dot(proj);
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

    #[test]
    fn test_hash_consistency_same_input() {
        let hasher = SimHashHasher::new_seeded(16, SIMD_LANECOUNT, 123);
        let vec = vec![1.0; SIMD_LANECOUNT];
        let hash1 = hasher.hash(&vec);
        let hash2 = hasher.hash(&vec);
        assert_eq!(hash1, hash2, "Hash must be consistent for same input");
    }

    #[test]
    fn test_hash_difference_on_orthogonal_vectors() {
        let dim = SIMD_LANECOUNT;
        let hasher = SimHashHasher::new_seeded(32, dim, 999);
        let mut v1 = vec![0.0; dim];
        let mut v2 = vec![0.0; dim];
        v1[0] = 1.0; // Unit vector along x
        v2[1] = 1.0; // Unit vector along y
        let h1 = hasher.hash(&v1);
        let h2 = hasher.hash(&v2);

        let hamming_distance: usize = h1.iter().zip(&h2).filter(|(a, b)| a != b).count();
        assert!(
            hamming_distance > 0,
            "Orthogonal vectors should be likely to hash differently"
        );
    }

    #[test]
    fn test_hash_dot_sign_behavior() {
        // Manually define a simple hasher with known projection
        let hasher = SimHashHasher {
            stored_vectors_dim: SIMD_LANECOUNT,
            projections: vec![
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ], // x-axis and y-axis projections
        };

        let input = vec![2.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = hasher.hash(&input);

        // Expect: dot([2,-3], [1,0]) = 2 → true
        //         dot([2,-3], [0,1]) = -3 → false
        assert_eq!(result, vec![true, false]);
    }
}
