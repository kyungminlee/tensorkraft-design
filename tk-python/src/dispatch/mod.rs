//! Type-erased dispatch for `DMFTLoop<T, Q, B>` → Python.
//!
//! PyO3's `#[pyclass]` cannot be applied to generic structs. This module
//! bridges Rust's monomorphization to Python's dynamic dispatch via
//! `DmftLoopVariant`, a non-generic enum wrapping concrete instantiations.

#[macro_use]
pub(crate) mod macros;

use tk_dmft::DMFTLoop;
use tk_linalg::DefaultDevice;
use tk_symmetry::{U1, Z2};

/// The default device for Python-exposed computations.
///
/// Resolves to `DeviceAPI<DeviceFaer, DeviceFaer>` when `backend-faer` is active.
/// Changing this requires rebuilding from source with the corresponding feature.
pub(crate) type PyDefaultDevice = DefaultDevice;

/// Type-erased wrapper over the supported concrete `DMFTLoop<T, Q, B>` instantiations.
///
/// | Variant        | Scalar | Symmetry | Use case                              |
/// |:---------------|:-------|:---------|:--------------------------------------|
/// | `RealU1`       | `f64`  | `U1`     | Standard single-orbital DMFT          |
/// | `RealZ2`       | `f64`  | `Z2`     | Particle-hole symmetric models        |
///
/// # Note
/// `ComplexU1` is not available in the draft because `DeviceFaer` does not
/// implement `LinAlgBackend<Complex<f64>>`. It will be added when the complex
/// backend is implemented in tk-linalg.
///
/// # Extending
/// Add a new combination by appending one enum variant here and adding
/// a corresponding arm in the `dispatch_variant!{}` macro in `macros.rs`.
pub(crate) enum DmftLoopVariant {
    RealU1(DMFTLoop<f64, U1, PyDefaultDevice>),
    RealZ2(DMFTLoop<f64, Z2, PyDefaultDevice>),
}

impl DmftLoopVariant {
    /// Human-readable variant name for `__repr__`.
    pub(crate) fn variant_name(&self) -> &'static str {
        match self {
            DmftLoopVariant::RealU1(_) => "RealU1",
            DmftLoopVariant::RealZ2(_) => "RealZ2",
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_variant_names() {
        assert_eq!("RealU1", "RealU1");
        assert_eq!("RealZ2", "RealZ2");
    }
}
