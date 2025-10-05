use fifo::FifoCache;
use lru::LruCache;
use lsh_fifo::LshFifoCache;
use lsh_lru::LshLruCache;
use pyo3::prelude::*;

mod fifo;
mod lru;
mod lsh_fifo;
mod lsh_lru;
mod vecpy;

/// A Python module implemented in Rust.
#[pymodule]
fn proximipy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LruCache>()?;
    m.add_class::<FifoCache>()?;
    m.add_class::<LshFifoCache>()?;
    m.add_class::<LshLruCache>()?;
    Ok(())
}
