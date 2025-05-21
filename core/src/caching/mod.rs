#![allow(unused_imports)]

mod approximate_cache;
mod fifo;
mod lru;
mod lsh;

pub use approximate_cache::ApproximateCache;
pub use fifo::FifoCache;
pub use lru::LruCache;
pub use lsh::LshCache;
