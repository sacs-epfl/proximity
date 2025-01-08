use proximipy::caching::approximate_cache::ApproximateCache;
use proximipy::caching::bounded::bounded_linear_cache::BoundedLinearCache;
use pyo3::{pyclass, pymethods};

macro_rules! create_pythonized_interface {
    ($name: ident, $type: ident) => {
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
            inner: BoundedLinearCache<$type, $type>,
        }

        #[pymethods]
        impl $name {
            #[new]
            pub fn new(max_capacity: usize, tolerance: f32) -> Self {
                Self {
                    inner: BoundedLinearCache::new(max_capacity, tolerance),
                }
            }

            fn find(&mut self, k: $type) -> Option<$type> {
                self.inner.find(&k)
            }

            fn insert(&mut self, key: $type, value: $type) {
                self.inner.insert(key, value)
            }

            fn len(&self) -> usize {
                self.inner.len()
            }
        }
    };
}

create_pythonized_interface!(I16Cache, i16);
