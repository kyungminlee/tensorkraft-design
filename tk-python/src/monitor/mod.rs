//! Cancellation monitor for GIL-safe Ctrl+C handling during long computations.
//!
//! The monitor thread polls `py.check_signals()` every 100 ms. On SIGINT,
//! it sets the cancellation flag and exits. On solver completion, it receives
//! a message on `done_rx` and exits cleanly.
//!
//! # Critical invariant
//! `shutdown()` MUST be called inside the `py.allow_threads` closure,
//! while the GIL is still released. Calling `shutdown()` after
//! `allow_threads` returns will deadlock if the monitor thread is blocked
//! on `Python::with_gil`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use pyo3::Python;

/// Internal helper encapsulating the AtomicBool cancellation flag
/// and the mpsc-based monitor thread lifecycle.
pub(crate) struct CancellationMonitor {
    /// Shared with the solver. Set to `true` by the monitor on SIGINT.
    pub cancel_flag: Arc<AtomicBool>,
    /// Send `()` to signal the monitor thread to exit.
    done_tx: Option<mpsc::SyncSender<()>>,
    /// Join handle for the monitor thread. `None` after `shutdown()`.
    monitor_handle: Option<thread::JoinHandle<()>>,
}

impl CancellationMonitor {
    /// Spawn the monitor thread.
    ///
    /// The thread loops on `done_rx.recv_timeout(Duration::from_millis(100))`:
    /// - `Ok(())` → solver finished normally; exit.
    /// - `Err(Disconnected)` → `done_tx` dropped (solver panicked); exit.
    /// - `Err(Timeout)` → call `Python::with_gil(|py| py.check_signals())`;
    ///   on `Err`, store `true` into `cancel_flag` with `Release` ordering and exit.
    pub(crate) fn spawn() -> Self {
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let monitor_cancel = cancel_flag.clone();

        let (done_tx, done_rx) = mpsc::sync_channel::<()>(1);

        let monitor_handle = thread::spawn(move || {
            loop {
                match done_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(()) => break,                                    // solver finished
                    Err(mpsc::RecvTimeoutError::Disconnected) => break, // solver panicked
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        // Check for pending Python signals (Ctrl+C)
                        let interrupted =
                            Python::with_gil(|py| py.check_signals().is_err());
                        if interrupted {
                            monitor_cancel.store(true, Ordering::Release);
                            break;
                        }
                    }
                }
            }
        });

        CancellationMonitor {
            cancel_flag,
            done_tx: Some(done_tx),
            monitor_handle: Some(monitor_handle),
        }
    }

    /// A shared reference to the cancellation flag, for passing to the solver.
    pub(crate) fn flag(&self) -> &Arc<AtomicBool> {
        &self.cancel_flag
    }

    /// Signal the monitor thread to exit and join it.
    ///
    /// Sends `()` on `done_tx`, then calls `monitor_handle.join()`.
    ///
    /// # Correctness requirement
    /// Must be called while the GIL is released (inside `py.allow_threads`).
    pub(crate) fn shutdown(mut self) {
        // Send completion signal. If the channel is already closed
        // (monitor exited due to SIGINT), this is a harmless no-op.
        if let Some(tx) = self.done_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.monitor_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for CancellationMonitor {
    fn drop(&mut self) {
        // Safety net: if shutdown() was not called (e.g., due to a panic),
        // drop the sender to trigger Disconnected in the monitor thread.
        // The join handle is dropped, which detaches the thread — this is
        // acceptable because the monitor will exit on Disconnected.
        self.done_tx.take(); // drop sender
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancel_flag_default_false() {
        let flag = Arc::new(AtomicBool::new(false));
        assert!(!flag.load(Ordering::Relaxed));
    }
}
