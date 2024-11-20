#![allow(dead_code)]
#![feature(portable_simd)]

extern crate rand;

mod caching;
mod numerics;

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rand::Rng as _;

    use crate::numerics::f32vector::F32Vector;

    const VEC_SIZE: usize = (u32::MAX / 128) as usize;
    #[test]
    fn perftest() {
        let mut rng = rand::thread_rng();
        let v1: Vec<_> = (0..128)
            .map(|_| f32::from(rng.gen_range(-20 as i16..20)))
            .collect();
        let v2s: Vec<f32> = (0..(128 * 200))
            .map(|_| f32::from(rng.gen_range(-20 as i16..20)))
            .collect();

        assert!(v1.len() == 128);
        let v1_f32v = F32Vector::from(&v1[..]);
        let start_t = Instant::now();
        let mut sum = 0.0;
        for _ in 0..200_000 {
            for v2_i in (0..v2s.len()).step_by(128) {
                let v2 = &v2s[v2_i..v2_i + 128];
                let v2_f32v = F32Vector::from(v2);
                assert!(v2.len() == 128);

                sum += v1_f32v.l2_dist_squared(&v2_f32v);
            }
        }

        let elapsed = start_t.elapsed();
        println!("{:?} - {sum} - {}", elapsed, v2s.len() * 128 * 200_000)
    }
}
