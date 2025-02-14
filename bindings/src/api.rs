use std::hash::{Hash, Hasher};

use proximipy::caching::bounded::fifo::fifo_cache::FifoCache as FifoInternal;
use proximipy::caching::bounded::lru::lru_cache::LRUCache as LruInternal;

use proximipy::numerics::f32vector::F32Vector;
use proximipy::{caching::approximate_cache::ApproximateCache, numerics::comp::ApproxComparable};
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyList},
    Bound, FromPyObject, IntoPyObject, PyErr,
};

macro_rules! create_pythonized_interface {
    ($internal : ident, $name: ident, $keytype: ident, $valuetype : ident) => {
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
            inner: $internal<$keytype, $valuetype>,
        }

        #[pymethods]
        impl $name {
            #[new]
            pub fn new(max_capacity: usize, tolerance: f32) -> Self {
                Self {
                    inner: $internal::new(max_capacity, tolerance),
                }
            }

            fn find(&mut self, k: $keytype) -> Option<$valuetype> {
                self.inner.find(&k)
            }

            fn batch_find(&mut self, ks : Vec<$keytype>) -> Vec<Option<$valuetype>> {
                ks.into_iter().map(|k| self.find(k)).collect()
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

struct VecPy<T> {
    inner: Vec<T>,
}

impl PartialEq for VecPy<f32> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.len() == other.inner.len()
            && self
                .inner
                .iter()
                .zip(&other.inner)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for VecPy<f32> {}

impl Hash for VecPy<f32> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &val in &self.inner {
            state.write_u32(val.to_bits());
        }
    }
}

/// Explain to Rust how to parse some random python object into an actual Rust vector
/// This involves new allocations because Python cannot be trusted to keep this
/// reference alive.
///
/// This can fail if the random object in question is not a list,
/// in which case it is automatically reported by raising a TypeError exception
/// in the Python code
impl<'a, T> FromPyObject<'a> for VecPy<T>
where
    T: FromPyObject<'a>,
{
    fn extract_bound(ob: &pyo3::Bound<'a, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let list: Vec<T> = ob.downcast::<PyList>()?.extract()?;
        Ok(VecPy::<T> { inner: list })
    }
}

// Cast back the list of T's to a Python list
impl<'a, T> IntoPyObject<'a> for VecPy<T>
where
    T: IntoPyObject<'a>,
{
    type Target = PyList;
    type Output = Bound<'a, PyList>;
    type Error = PyErr;

    fn into_pyobject(self, py: pyo3::Python<'a>) -> Result<Self::Output, Self::Error> {
        let internal = self.inner;
        PyList::new(py, internal)
    }
}

impl<T> Clone for VecPy<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        VecPy::<T> {
            inner: self.inner.clone(),
        }
    }
}

impl ApproxComparable for VecPy<f32> {
    fn roughly_matches(&self, instore: &Self, tolerance: f32) -> bool {
        F32Vector::from(&self.inner as &[f32])
            .roughly_matches(&F32Vector::from(&instore.inner as &[f32]), tolerance)
    }
    fn fuzziness(&self, instore: &Self) -> f32 {
        F32Vector::from(&self.inner as &[f32]).fuzziness(&F32Vector::from(&instore.inner as &[f32]))
    }
}

type F32VecPy = VecPy<f32>;
type UsizeVecPy = VecPy<usize>;
type UsizeWithRankingVecPy = (UsizeVecPy, F32VecPy);

create_pythonized_interface!(LruInternal, LRUCache, F32VecPy, UsizeWithRankingVecPy);
create_pythonized_interface!(FifoInternal, FifoCache, F32VecPy, UsizeWithRankingVecPy);
