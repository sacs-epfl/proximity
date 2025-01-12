use api::{FVecToU32VectorCache, FVecToUsizeVectorCache, I16ToF32VectorCache};
use pyo3::prelude::*;

mod api;

/// A Python module implemented in Rust.
#[pymodule]
fn proximipy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<I16ToF32VectorCache>()?;
    m.add_class::<FVecToU32VectorCache>()?;
    m.add_class::<FVecToUsizeVectorCache>()?;
    Ok(())
}
