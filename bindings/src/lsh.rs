use proximity::caching::{
    approximate_cache::ApproximateCache, lsh::lsh_cache::LshFifoCache as LshInternal,
};
use pyo3::{pyclass, pymethods, PyObject};

use crate::api::F32VecPy;

// unsendable == should hard-crash if Python tries to access it from
// two different Python threads.
//
// The implementation is very much thread-unsafe anyways (lots of mutations),
// so this is an OK behavior, we will detect it with a nice backtrace
// and without UB.
//
// Even in the case where we want the cache to be multithreaded, this would
// happen on the Rust side and will not be visible to the Python ML pipeline.
#[pyclass(unsendable)]
pub struct LshFifoCache {
    inner: LshInternal<PyObject>,
}

#[pymethods]
impl LshFifoCache {
    #[new]
    #[pyo3(signature = (num_hash, dim, bucket_capacity, tolerance, seed=None))]
    pub fn new(
        num_hash: usize,
        dim: usize,
        bucket_capacity: usize,
        tolerance: f32,
        seed: Option<u64>,
    ) -> Self {
        Self {
            inner: LshInternal::new(num_hash, dim, bucket_capacity, tolerance, seed),
        }
    }

    fn find(&mut self, k: F32VecPy) -> Option<PyObject> {
        self.inner.find(&k.inner)
    }

    fn batch_find(&mut self, ks: Vec<F32VecPy>) -> Vec<Option<PyObject>> {
        // more efficient than a python for loop
        ks.into_iter().map(|k| self.find(k)).collect()
    }

    fn insert(&mut self, key: F32VecPy, value: PyObject, tolerance: f32) {
        self.inner.insert(key.inner, value, tolerance)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}
