//! Build script for tk-dmft.
//!
//! Performs two checks:
//! 1. MPI library detection (when backend-mpi is enabled)
//! 2. Feature conflict check for mutually exclusive BLAS backends

fn main() {
    // Feature conflict: backend-mkl and backend-openblas are mutually exclusive.
    // This mirrors the workspace-level enforcement in tk-linalg.
    #[cfg(all(feature = "backend-mkl", feature = "backend-openblas"))]
    compile_error!(
        "Features `backend-mkl` and `backend-openblas` are mutually exclusive. \
         Enable only one BLAS backend."
    );

    // MPI detection: when backend-mpi is enabled, check for mpicc.
    #[cfg(feature = "backend-mpi")]
    {
        use std::process::Command;
        let mpicc = Command::new("mpicc").arg("--version").output();
        if mpicc.is_err() {
            panic!(
                "Feature `backend-mpi` requires an MPI installation. \
                 `mpicc` was not found in PATH. Install OpenMPI or MPICH \
                 and ensure `mpicc` is available."
            );
        }
    }
}
