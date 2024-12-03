#![allow(dead_code)]
#![feature(portable_simd, test, array_chunks)]

use std::path::Path;

use caching::approximate_cache::ApproximateCache;
use fs::file_manager;
use numerics::f32vector::F32Vector;
use rand::{seq::index, Rng};

extern crate npyz;
extern crate rand;
extern crate test;

mod caching;
mod fs;
mod numerics;

use crate::caching::unbounded_linear_cache::UnboundedLinearCache;

const PATH_TO_ROOT: &'static str = "/Users/matrix/Documents/proximity/";
fn main() {
    let vecs = file_manager::read_from_npy(Path::new(
        &(PATH_TO_ROOT.to_owned() + "res/sift/concatted.npy"),
    ));
    let vecs_f: Vec<f32> = vecs.into_iter().map(f32::from).collect();
    println!("{:?}", vecs_f.chunks_exact(128).next().unwrap());

    let mut ulc = UnboundedLinearCache::<F32Vector, ()>::new(0.001);
    let mut randd = rand::thread_rng();
    let mut count: u32 = 0;
    let mut scanned: usize = 0;

    let proportion = 0.5;
    for _ in 0..50_000 {
        let is_common = randd.gen_bool(proportion);

        let index = if is_common {
            randd.gen_range(0..1024)
        } else {
            randd.gen_range(1024..(vecs_f.len() / 128))
        };
        let f32v = F32Vector::from(&vecs_f[index..index + 128]);
        let found = ulc.find(&f32v).is_some();

        if found {
            count += 1;
        } else {
            scanned += ulc.len();
            ulc.insert(f32v, ());
        }
    }

    println!("count : {}, scanned lower bound : {}", count, scanned)
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;

    use rand::Rng as _;

    use crate::numerics::f32vector::F32Vector;

    const VEC_SIZE: usize = (u32::MAX / 128) as usize;

    #[bench]
    fn perftest(b: &mut test::Bencher) {
        let mut rng = rand::thread_rng();
        let v1: Vec<_> = (0..128)
            .map(|_| f32::from(rng.gen_range(-20 as i16..20)))
            .collect();
        let v2s: Vec<f32> = (0 as u64..(128 * 10_000))
            .map(|_| f32::from(rng.gen_range(-20 as i16..20)))
            .collect();

        assert!(v1.len() == 128);
        let v1_f32v = F32Vector::from(&v1[..]);

        b.iter(|| {
            for v2_i in v2s.chunks_exact(128) {
                let v2_f32v = F32Vector::from(v2_i);
                black_box(v1_f32v.l2_dist_squared(&v2_f32v));
            }
        })
    }
}
