//! Storage device abstraction for tensor memory placement.
//!
//! The `StorageDevice` trait generalizes tensor storage to support host (CPU),
//! GPU, and distributed device memory. In Phases 1–3, only `HostDevice` is
//! available. `CudaDevice` and `MpiDevice` will gain full implementations
//! in Phase 5.
//!
//! # Migration Plan
//!
//! Currently `TensorStorage<'a, T>` is a simple `Owned(Vec<T>)` / `Borrowed(&'a [T])`
//! enum. In Phase 5, it will gain a default device type parameter:
//!
//! ```text
//! pub struct TensorStorage<T: Scalar, D: StorageDevice = HostDevice> { ... }
//! ```
//!
//! The default `HostDevice` parameter preserves backward compatibility for all
//! existing code that writes `TensorStorage<'a, T>`.

/// Trait abstracting the device on which tensor memory resides.
///
/// Each device type provides metadata (name, synchronization requirements)
/// that the tensor storage layer uses for dispatch. Allocation and
/// deallocation are device-specific and implemented via associated methods
/// on the concrete device types.
///
/// # Implementing a new device
///
/// ```ignore
/// pub struct MyDevice { /* ... */ }
///
/// impl StorageDevice for MyDevice {
///     fn name() -> &'static str { "my-device" }
///     fn requires_sync() -> bool { true }
///     fn synchronize(&self) { /* ... */ }
/// }
/// ```
pub trait StorageDevice: Send + Sync + 'static {
    /// Human-readable device name for diagnostics and logging.
    fn name() -> &'static str;

    /// Whether operations on this device require explicit synchronization
    /// before host-side reads. `false` for `HostDevice`, `true` for GPU devices.
    fn requires_sync() -> bool;

    /// Block until all pending operations on this device are complete.
    ///
    /// No-op for `HostDevice`. For GPU devices, this synchronizes the
    /// device stream to ensure that all preceding kernel launches and
    /// memory transfers have completed.
    fn synchronize(&self);
}

/// Host (CPU) memory device — the default for all tensor storage.
///
/// Uses standard heap allocation (`Vec<T>`) for owned data and
/// slice borrows (`&[T]`) for arena-backed temporary tensors.
/// No synchronization is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HostDevice;

impl StorageDevice for HostDevice {
    #[inline(always)]
    fn name() -> &'static str {
        "host"
    }

    #[inline(always)]
    fn requires_sync() -> bool {
        false
    }

    #[inline(always)]
    fn synchronize(&self) {
        // No-op: host operations are synchronous.
    }
}

/// CUDA GPU device identified by ordinal (device index).
///
/// When the `backend-cuda` feature is active, `CudaDevice` represents
/// a specific GPU. Memory allocated on this device resides in GPU
/// global memory and requires explicit synchronization before host reads.
///
/// Full allocation and transfer methods will be implemented in Phase 5
/// when CUDA runtime bindings are integrated.
#[cfg(feature = "backend-cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaDevice {
    /// CUDA device ordinal (0-based index from `cudaGetDeviceCount`).
    pub ordinal: usize,
}

#[cfg(feature = "backend-cuda")]
impl StorageDevice for CudaDevice {
    fn name() -> &'static str {
        "cuda"
    }

    fn requires_sync() -> bool {
        true
    }

    fn synchronize(&self) {
        // Phase 5: call cudaDeviceSynchronize() via FFI.
        // For now, this is a no-op placeholder. When CUDA bindings are
        // integrated, this will block until all GPU operations complete.
        log::debug!(
            "CudaDevice::synchronize() called for ordinal {} (placeholder)",
            self.ordinal
        );
    }
}

#[cfg(feature = "backend-cuda")]
impl CudaDevice {
    /// Create a new `CudaDevice` for the given ordinal.
    pub fn new(ordinal: usize) -> Self {
        CudaDevice { ordinal }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_device_properties() {
        assert_eq!(HostDevice::name(), "host");
        assert!(!HostDevice::requires_sync());
        // synchronize is a no-op; just verify it doesn't panic
        HostDevice.synchronize();
    }

    #[test]
    fn host_device_is_send_sync() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<HostDevice>();
    }

    #[test]
    fn host_device_implements_storage_device() {
        fn assert_storage_device<T: StorageDevice>() {}
        assert_storage_device::<HostDevice>();
    }
}
