use pyo3::prelude::*;

// Core Rust functions (callable from Rust)
pub fn add(a: i64, b: i64) -> i64 {
    a + b
}

// Python-exposed functions
#[pyfunction]
fn py_add(a: i64, b: i64) -> i64 {
    add(a, b)
}

// Python module definition
#[pymodule]
fn russo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_add, m)?)?;
    Ok(())
}
