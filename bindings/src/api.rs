use std::hash::{Hash, Hasher};

use proximity::numerics::ApproxComparable;
use proximity::numerics::F32Vector;

use pyo3::{
    types::{PyAnyMethods, PyList},
    Bound, FromPyObject, IntoPyObject, PyErr,
};

pub struct VecPy<T> {
    pub inner: Vec<T>,
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

impl AsRef<[f32]> for VecPy<f32> {
    fn as_ref(&self) -> &[f32] {
        self.inner.as_ref()
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

pub type F32VecPy = VecPy<f32>;
