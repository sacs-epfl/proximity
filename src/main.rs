#![allow(dead_code)]
mod caching;
mod numerics;

fn main() {
    println!("Hello, world!");
}
#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::numerics::f32vector::F32Vector;

    const VEC_SIZE: usize = (u32::MAX / 128) as usize;
    #[test]
    fn perftest() {
        let data: Vec<_> = (0..(u32::MAX / 128)).map(|x| f32::from(x as u16)).collect();
        let vectorized = F32Vector::from(&data[..]);
        let start_t = Instant::now();
        let mut sum = 0.0;
        for _ in 0..200 {
            sum += vectorized.l2_dist(&vectorized);
        }
        let elapsed = start_t.elapsed();
        println!("{:?} - {sum}", elapsed)
    }
}
