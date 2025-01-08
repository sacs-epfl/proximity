use api::I16Cache;
use pyo3::prelude::*;

mod api;

/// A Python module implemented in Rust.
#[pymodule]
fn proximipy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<I16Cache>()?;
    Ok(())
}
