#![allow(dead_code)]

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
        let v2: Vec<_> = (0..128)
            .map(|_| f32::from(rng.gen_range(-20 as i16..20)))
            .collect();

        let v1_f32v = F32Vector::from(&v1[..]);
        let v2_f32v = F32Vector::from(&v2[..]);
        let start_t = Instant::now();
        let mut sum = 0.0;
        for _ in 0..10_000_000 {
            sum += v1_f32v.l2_dist(&v2_f32v);
        }
        let elapsed = start_t.elapsed();
        println!("{:?} - {sum}", elapsed)
    }
}
