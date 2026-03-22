fn main() {
    // FFI BLAS backends both expose global symbols (dgemm_, dsyev_, etc.).
    // Enabling both causes linker collisions that produce cryptic symbol errors.
    // A compile-time check is vastly more ergonomic than a link-time error.
    #[cfg(all(feature = "backend-mkl", feature = "backend-openblas"))]
    compile_error!(
        "Features `backend-mkl` and `backend-openblas` are mutually exclusive. \
         Both expose global BLAS/LAPACK symbols and will cause linker collisions \
         (e.g., duplicate symbol `dgemm_`). Enable only one FFI-based backend. \
         If you need both on the same system, use separate workspaces."
    );
}
