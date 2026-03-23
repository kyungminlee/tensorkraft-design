//! `dispatch_variant!{}` macro for generating match arms over `DmftLoopVariant`.

/// Generate a `match` expression that calls `$method` on the inner solver
/// for every `DmftLoopVariant`.
macro_rules! dispatch_variant {
    ($inner:expr, $method:ident ( $($arg:expr),* )) => {
        match &$inner {
            $crate::dispatch::DmftLoopVariant::RealU1(s)  => s.$method($($arg),*),
            $crate::dispatch::DmftLoopVariant::RealZ2(s)  => s.$method($($arg),*),
        }
    };
}

/// Mutable variant of `dispatch_variant!` for methods that take `&mut self`.
/// The caller must pass a mutable reference or mutable binding.
macro_rules! dispatch_variant_mut {
    ($inner:expr, $method:ident ( $($arg:expr),* )) => {
        match $inner {
            $crate::dispatch::DmftLoopVariant::RealU1(s)  => s.$method($($arg),*),
            $crate::dispatch::DmftLoopVariant::RealZ2(s)  => s.$method($($arg),*),
        }
    };
}

pub(crate) use dispatch_variant;
pub(crate) use dispatch_variant_mut;
