use api::I16ToVectorCache;
use pyo3::prelude::*;

mod api;

/// A Python module implemented in Rust.
#[pymodule]
fn proximipy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<I16ToVectorCache>()?;
    Ok(())
}
