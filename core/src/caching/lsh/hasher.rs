use rand::distr::Uniform;
use rand::Rng;
use rand::rng;
use rand_distr::StandardNormal;

use crate::numerics::f32vector::F32Vector;

pub struct PStableHasher {
    pub num_hash: usize,
    bin_size: f32,
    /// random projection vectors
    projections: Vec<Vec<f32>>,
    /// random offsets in [0, w).
    offsets: Vec<f32>,
}

impl PStableHasher {
    /// Constructs a new hasher with `num_hash` projections in dimension `dim`, using bin `w`.
    pub fn new(num_hash: usize, dim: usize, bin_size: f32) -> Self {
        let mut rng: rand::prelude::ThreadRng = rng();

        let projections = (0..num_hash)
            .map(|_| (0..dim).map(|_| rng.sample::<f32, _>(StandardNormal) as f32).collect())
            .collect();

        let uniform = Uniform::new(0.0_f32, bin_size).unwrap();
        let offsets = (0..num_hash).map(|_| rng.sample(uniform)).collect();
        PStableHasher { num_hash, bin_size, projections, offsets }
    }

    /// Hashes `vector` to a `k`-length integer signature.
    pub fn hash(&self, vector: &[f32]) -> Vec<i64> {
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