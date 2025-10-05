use std::hash::{Hash, Hasher};

use proximity::numerics::ApproxComparable;

use pyo3::{
    types::{PyAnyMethods, PyList},
    Bound, FromPyObject, IntoPyObject, PyErr,
};

pub struct VecPy {
    pub inner: Vec<f32>,
}

impl PartialEq for VecPy {
    fn eq(&self, other: &Self) -> bool {
        self.inner.len() == other.inner.len()
            && self
                .inner
                .iter()
                .zip(&other.inner)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for VecPy {}

impl Hash for VecPy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &val in &self.inner {
            state.write_u32(val.to_bits());
        }
    }
}

impl AsRef<[f32]> for VecPy {
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
impl<'a> FromPyObject<'a> for VecPy {
    fn extract_bound(ob: &pyo3::Bound<'a, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let list: Vec<f32> = ob.downcast::<PyList>()?.extract()?;
        Ok(VecPy { inner: list })
    }
}

// Cast back the list of T's to a Python list
impl<'a> IntoPyObject<'a> for VecPy {
    type Target = PyList;
    type Output = Bound<'a, PyList>;
    type Error = PyErr;

    fn into_pyobject(self, py: pyo3::Python<'a>) -> Result<Self::Output, Self::Error> {
        let internal = self.inner;
        PyList::new(py, internal)
    }
}

impl Clone for VecPy {
    fn clone(&self) -> Self {
        VecPy {
            inner: self.inner.clone(),
        }
    }
}

impl ApproxComparable for VecPy {
    #[inline]
    fn roughly_matches(&self, instore: &Self, tolerance: f32) -> bool {
        (&self.inner as &[f32]).roughly_matches(&instore.inner, tolerance)
    }
    #[inline]
    fn fuzziness(&self, instore: &Self) -> f32 {
        (&self.inner as &[f32]).fuzziness(&instore.inner)
    }
}
