#![allow(unused_imports)]

mod approximate_cache;
mod fifo;
mod lru;
mod lsh;

pub use fifo::FifoCache;
pub use lru::LRUCache;
pub use lsh::LshFifoCache;
pub use approximate_cache::ApproximateCache;