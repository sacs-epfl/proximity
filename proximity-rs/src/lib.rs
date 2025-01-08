#![feature(portable_simd, test, array_chunks)]

use caching::bounded::bounded_linear_cache::I16Cache;
use pyo3::prelude::*;

extern crate npyz;
extern crate rand;
extern crate test;

pub mod caching;
pub mod fs;
pub mod numerics;

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pymodule]
fn proximipy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<I16Cache>()?;
    Ok(())
}
