//! GIL-releasing solve implementation with cancellation monitor.

use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::prelude::*;

use tk_dmft::DmftError;

use crate::dispatch::DmftLoopVariant;
use crate::dispatch::macros::dispatch_variant_mut;
use crate::error::PythonError;
use crate::monitor::CancellationMonitor;
use crate::spectral::PySpectralFunction;

/// The canonical GIL release + cancellation pattern.
///
/// 1. Spawn a `CancellationMonitor` (AtomicBool + mpsc monitor thread).
/// 2. Release the GIL via `py.allow_threads()`.
/// 3. Inside: call `solve_with_cancel_flag(&cancel_flag)`.
/// 4. Inside: call `monitor.shutdown()` (CRITICAL: while GIL is still released).
/// 5. Convert the result to the appropriate Python exception.
pub(crate) fn solve_impl(
    py: Python<'_>,
    inner: &mut DmftLoopVariant,
) -> PyResult<PySpectralFunction> {
    let monitor = CancellationMonitor::spawn();
    let cancel_flag = monitor.flag().clone();

    let result = py.allow_threads(|| {
        let r = dispatch_variant_mut!(inner, solve_with_cancel_flag(&cancel_flag));

        // CRITICAL: shutdown while the GIL is still released.
        // If this line executed after allow_threads returns, the main
        // thread would hold the GIL while the monitor may be blocked on
        // Python::with_gil — AB/BA deadlock.
        monitor.shutdown();

        r
    });

    match result {
        Ok(spectral) => Ok(PySpectralFunction::from(spectral)),
        Err(DmftError::Cancelled) => Err(PyErr::new::<PyKeyboardInterrupt, _>(
            "DMFT cancelled by user",
        )),
        Err(e) => Err(PythonError::from(e).into()),
    }
}
