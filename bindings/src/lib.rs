use api::{FifoCache, LRUCache};
use pyo3::prelude::*;

mod api;

/// A Python module implemented in Rust.
#[pymodule]
fn proximipy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LRUCache>()?;
    m.add_class::<FifoCache>()?;
    Ok(())
}
