use fifo::FifoCache;
use lru::LruCache;
use lsh::LshFifoCache;
use pyo3::prelude::*;

mod api;
mod fifo;
mod lru;
mod lsh;

/// A Python module implemented in Rust.
#[pymodule]
fn proximipy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LruCache>()?;
    m.add_class::<FifoCache>()?;
    m.add_class::<LshFifoCache>()?;
    Ok(())
}
