use proximity::caching::{ApproximateCache, LshFifoCache as LshFifoInternal};
use pyo3::{pyclass, pymethods, PyObject};

use crate::api::F32VecPy;

#[pyclass]
pub struct LshFifoCache {
    inner: LshFifoInternal<F32VecPy, PyObject>,
}

#[pymethods]
impl LshFifoCache {
    #[new]
    #[pyo3(signature = (num_hash, dim, bucket_capacity, seed=None))]
    pub fn new(num_hash: usize, dim: usize, bucket_capacity: usize, seed: Option<u64>) -> Self {
        Self {
            inner: LshFifoInternal::new(num_hash, dim, bucket_capacity, seed),
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
