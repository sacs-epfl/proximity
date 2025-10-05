use proximity::caching::{ApproximateCache, LruCache as LruInternal};
use pyo3::{pyclass, pymethods, PyObject};

use crate::vecpy::VecPy;

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
pub struct LruCache {
    inner: LruInternal<VecPy, PyObject>,
}

#[pymethods]
impl LruCache {
    #[new]
    pub fn new(max_capacity: usize) -> Self {
        Self {
            inner: LruInternal::new(max_capacity),
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
