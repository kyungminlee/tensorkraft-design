// tk-python build script: feature conflict detection and advisory warnings.

fn main() {
    // 1. FFI backend mutual exclusivity (defense-in-depth).
    // The canonical enforcement is in tk-linalg/build.rs; this check
    // surfaces the error closer to the build root.
    #[cfg(all(feature = "backend-mkl", feature = "backend-openblas"))]
    compile_error!(
        "Features `backend-mkl` and `backend-openblas` are mutually exclusive in tk-python. \
         Both expose global BLAS symbols and cause linker collisions."
    );

    // 2. TRIQS detection (informational).
    // When the `triqs` feature is enabled, probe for a TRIQS installation.
    // This is informational only — the crate compiles without TRIQS.
    #[cfg(feature = "triqs")]
    {
        let output = std::process::Command::new("python3")
            .args(["-c", "import triqs"])
            .output();
        match output {
            Ok(out) if !out.status.success() => {
                println!(
                    "cargo:warning=tk-python: `triqs` feature is enabled but \
                     `import triqs` failed. TRIQS interop will raise ImportError at runtime."
                );
            }
            Err(_) => {
                println!(
                    "cargo:warning=tk-python: could not run `python3 -c 'import triqs'`. \
                     TRIQS detection skipped."
                );
            }
            _ => {}
        }
    }
}
