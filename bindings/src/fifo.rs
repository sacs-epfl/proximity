use proximity::caching::{
    approximate_cache::ApproximateCache, fifo::fifo_cache::FifoCache as FifoInternal,
};
use pyo3::{pyclass, pymethods, PyObject};

use crate::api::F32VecPy;

#[pyclass]
pub struct FifoCache {
    inner: FifoInternal<F32VecPy, PyObject>,
}

#[pymethods]
impl FifoCache {
    #[new]
    pub fn new(max_capacity: usize) -> Self {
        Self {
            inner: FifoInternal::new(max_capacity),
        }
    }

    fn find(&mut self, k: F32VecPy) -> Option<PyObject> {
        self.inner.find(&k)
    }

    fn batch_find(&mut self, ks: Vec<F32VecPy>) -> Vec<Option<PyObject>> {
        // more efficient than a python for loop
        ks.into_iter().map(|k| self.find(k)).collect()
    }

    fn insert(&mut self, key: F32VecPy, value: PyObject, tolerance: f32) {
        self.inner.insert(key, value, tolerance)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}
