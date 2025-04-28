use rand::distr::Uniform;
use rand::rng;
use rand::rngs::ThreadRng;
use rand::Rng;
use rand_distr::StandardNormal;

use crate::numerics::f32vector::{F32Vector, SIMD_LANECOUNT};

/// A 2-stable (Gaussian) LSH hasher using p-stable distributions for Euclidean distance.
///
/// # Examples
///
/// Basic usage with random projections:
///
/// ```
/// use proximity::caching::lsh::hasher::PStableHasher;
///
/// const DIM: usize = 8;
/// const NUM_HASH: usize = 4;
/// const BIN_SIZE: f32 = 1.0;
///
/// // Create a new hasher (dim must be a multiple of F32Vector::SIMD_LANECOUNT)
/// let hasher = PStableHasher::new(NUM_HASH, DIM, BIN_SIZE);
///
/// // Hash a sample vector of length DIM
/// let vec = vec![0.5_f32; DIM];
/// let signature = hasher.hash(&vec);
/// assert_eq!(signature.len(), NUM_HASH);
/// ```
///
pub struct PStableHasher {
    pub num_hash: usize,
    pub dim: usize,
    bin_size: f32,
    /// random projection vectors
    projections: Vec<Vec<f32>>,
    /// random offsets in [0, w).
    offsets: Vec<f32>,
}

impl PStableHasher {
    /// Constructs a new hasher with `num_hash` projections in dimension `dim`, using bin width `bin_size`.
    pub fn new(num_hash: usize, dim: usize, bin_size: f32) -> Self {
        let mut rng: ThreadRng = rng();
        Self::with_rng(num_hash, dim, bin_size, &mut rng)
    }

    /// Constructs a new hasher with a provided RNG (useful for deterministic testing).
    pub fn with_rng<R: Rng>(num_hash: usize, dim: usize, bin_size: f32, rng: &mut R) -> Self {
        assert!(dim % SIMD_LANECOUNT == 0); // we expect vectors of size accepted by F32Vector::dot
        assert!(bin_size > 0.0);
        // the projections are supposed to be N(0, Id) random gaussian vectors
        let mut gaussian_iter = rng.sample_iter(StandardNormal);
        let projections: Vec<Vec<f32>> = (0..num_hash)
            .map(|_| (0..dim).map(|_| gaussian_iter.next().unwrap()).collect())
            .collect();

        // Sample num_hash offsets uniformly in [0, bin_size)
        let uniform = Uniform::new(0.0_f32, bin_size).unwrap();
        let offsets: Vec<f32> = rng.sample_iter(uniform).take(num_hash).collect();

        PStableHasher {
            num_hash,
            dim,
            bin_size,
            projections,
            offsets,
        }
    }

    /// Hashes `vector` to a `k`-length integer signature.
    pub fn hash(&self, vector: &[f32]) -> Vec<i64> {
        debug_assert!(vector.len() == self.dim);
        self.projections
            .iter()
            .zip(&self.offsets)
            .map(|(proj, offset)| {
                let dot: f32 = F32Vector::from(vector).dot(&F32Vector::from(proj as &[f32]));
                ((dot + offset) / self.bin_size).floor() as i64
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    #[test]
    fn test_hash_known() {
        // Use dimension multiple of 8 (here dim=8), with zeros padding to meet SIMD requirements
        let projections = vec![
            vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let offsets = vec![0.1, 0.2];
        let hasher = PStableHasher {
            num_hash: 2,
            dim: 8,
            bin_size: 1.0,
            projections,
            offsets,
        };
        let vector = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = hasher.hash(&vector);
        // dot1 = 1*3 + 2*4 = 11; floor((11 + 0.1)/1.0) = 11
        // dot2 = 0.5*3 + (-1.0)*4 = -2.5; floor((-2.5 + 0.2)/1.0) = floor(-2.3) = -3
        assert_eq!(result, vec![11, -3]);
    }

    #[test]
    fn test_hash_length_zero_vec() {
        let hasher = PStableHasher::new(5, SIMD_LANECOUNT * 4, 2.0);
        let vec = vec![0.0; SIMD_LANECOUNT * 4];
        let result = hasher.hash(&vec);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_with_rng_determinism() {
        let mut seeded = StdRng::seed_from_u64(42);
        let h1 = PStableHasher::with_rng(3, SIMD_LANECOUNT, 0.5, &mut seeded);
        let mut seeded2 = StdRng::seed_from_u64(42);
        let h2 = PStableHasher::with_rng(3, SIMD_LANECOUNT, 0.5, &mut seeded2);
        assert_eq!(h1.projections, h2.projections);
        assert_eq!(h1.offsets, h2.offsets);
    }
}
