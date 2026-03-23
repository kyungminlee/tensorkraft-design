//! `tk-python` — Python bindings for the tensorkraft tensor network library.
//!
//! This crate is a thin translation layer. All physics logic, numerical
//! algorithms, and memory management reside in the underlying Rust crates.
//! What `tk-python` owns is the GIL boundary: where the GIL is released,
//! how cancellation signals cross that boundary safely, and how Rust memory
//! is presented to Python without copying.
//!
//! The golden rule is: Rayon workers must never touch the Python GIL,
//! and the monitor thread must never be alive when the main thread holds the GIL.

pub(crate) mod bath;
pub(crate) mod config;
pub(crate) mod dispatch;
pub(crate) mod dmft;
pub(crate) mod error;
pub(crate) mod monitor;
pub(crate) mod spectral;

use pyo3::prelude::*;

use crate::bath::PyBathParameters;
use crate::config::{PyDmftConfig, PyDmrgConfig, PyLinearPredictionConfig, PyTimeEvolutionConfig};
use crate::dmft::PyDmftLoop;
use crate::dmft::stats::PyDmftStats;
use crate::spectral::PySpectralFunction;

/// The top-level Python module `tensorkraft`.
///
/// # Python usage
///
/// ```python
/// import tensorkraft as tk
/// print(tk.__version__)   # e.g., "0.1.0"
/// help(tk.DMFTLoop)
/// ```
#[pymodule]
fn tensorkraft(py: Python<'_>, m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    // Register classes
    m.add_class::<PyDmftLoop>()?;
    m.add_class::<PySpectralFunction>()?;
    m.add_class::<PyBathParameters>()?;
    m.add_class::<PyDmftConfig>()?;
    m.add_class::<PyDmrgConfig>()?;
    m.add_class::<PyTimeEvolutionConfig>()?;
    m.add_class::<PyLinearPredictionConfig>()?;
    m.add_class::<PyDmftStats>()?;

    // Register exception hierarchy
    error::register_exceptions(py, m)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
