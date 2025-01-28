use api::{FVecToUsizeVectorAny, FVecToUsizeVectorBest};
use pyo3::prelude::*;

mod api;

/// A Python module implemented in Rust.
#[pymodule]
fn proximipy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FVecToUsizeVectorBest>()?;
    m.add_class::<FVecToUsizeVectorAny>()?;
    Ok(())
}
