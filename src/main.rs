#![allow(dead_code)]
#![feature(portable_simd, test, array_chunks)]

use std::path::Path;

use caching::approximate_cache::ApproximateCache;
use fs::file_manager;
use numerics::f32vector::F32Vector;

extern crate rand;
extern crate test;

mod caching;
mod numerics;
mod fs;

use crate::caching::unbounded_linear_cache::UnboundedLinearCache;

fn main() {
    let vecs = file_manager::read_from_file_f32(Path::new("/home/mathis/balblou"));
    let mut ulc = UnboundedLinearCache::<F32Vector, ()>::new(1.0);

    let mut count: u32 = 0;
    for vec in vecs.iter() {
        let f32v = F32Vector::from(&vec[..]);
        let found = ulc.find(&f32v).is_some();

        if found {
            count += 1;
        } else {
            ulc.insert(f32v, ());
        }
    }

    println!("{}", count)
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
