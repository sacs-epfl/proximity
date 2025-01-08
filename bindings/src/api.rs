use proximipy::caching::approximate_cache::ApproximateCache;
use proximipy::caching::bounded::bounded_linear_cache::BoundedLinearCache;
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyList},
    Bound, FromPyObject, IntoPyObject, PyErr,
};

macro_rules! create_pythonized_interface {
    ($name: ident, $keytype: ident, $valuetype : ident) => {
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
        pub struct $name {
            inner: BoundedLinearCache<$keytype, $valuetype>,
        }

        #[pymethods]
        impl $name {
            #[new]
            pub fn new(max_capacity: usize, tolerance: f32) -> Self {
                Self {
                    inner: BoundedLinearCache::new(max_capacity, tolerance),
                }
            }

            fn find(&mut self, k: $keytype) -> Option<$valuetype> {
                self.inner.find(&k)
            }

            fn insert(&mut self, key: $keytype, value: $valuetype) {
                self.inner.insert(key, value)
            }

            fn len(&self) -> usize {
                self.inner.len()
            }

            fn __len__(&self) -> usize {
                self.len()
            }
        }
    };
}
struct F32VecPy {
    inner: Vec<f32>,
}

/// Explain to Rust how to parse some random python object into an actual Rust vector
/// This involves new allocations because Python cannot be trusted to keep this
/// reference alive.
///
/// This can fail if the random object in question is not a list of numbers,
/// in which case it is automatically reported by raising a TypeError exception
/// in the Python code
impl<'a> FromPyObject<'a> for F32VecPy {
    fn extract_bound(ob: &pyo3::Bound<'a, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let list: Vec<f32> = ob.downcast::<PyList>()?.extract()?;
        Ok(F32VecPy { inner: list })
    }
}

// Cast back the list of floats to a Python list
impl<'a> IntoPyObject<'a> for F32VecPy {
    type Target = PyList;
    type Output = Bound<'a, PyList>;
    type Error = PyErr;

    fn into_pyobject(self, py: pyo3::Python<'a>) -> Result<Self::Output, Self::Error> {
        let internal = self.inner;
        PyList::new(py, internal)
    }
}

impl Clone for F32VecPy {
    fn clone(&self) -> Self {
        F32VecPy {
            inner: self.inner.clone(),
        }
    }
}

create_pythonized_interface!(I16ToVectorCache, i16, F32VecPy);
