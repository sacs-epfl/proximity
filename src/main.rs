#![allow(dead_code)]
#![feature(portable_simd, test)]

extern crate rand;
extern crate test;

mod caching;
mod numerics;

fn main() {
    println!("Hello, world!");
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
        let v2s: Vec<f32> = (0 as u64..(128 * 20_000_000))
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
