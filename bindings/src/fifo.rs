use proximity::caching::{ApproximateCache, FifoCache as FifoInternal};
use pyo3::{pyclass, pymethods, PyObject};

use crate::vecpy::VecPy;

#[pyclass]
pub struct FifoCache {
    inner: FifoInternal<VecPy, PyObject>,
}

#[pymethods]
impl FifoCache {
    #[new]
    pub fn new(max_capacity: usize) -> Self {
        Self {
            inner: FifoInternal::new(max_capacity),
        }
    }

    fn find(&mut self, k: VecPy) -> Option<PyObject> {
        self.inner.find(&k)
    }

    fn batch_find(&mut self, ks: Vec<VecPy>) -> Vec<Option<PyObject>> {
        // more efficient than a python for loop
        ks.into_iter().map(|k| self.find(k)).collect()
    }

    fn insert(&mut self, key: VecPy, value: PyObject, tolerance: f32) {
        self.inner.insert(key, value, tolerance)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}
