//! Adaptive threading regime selection for block-sparse operations.

use tk_core::Scalar;
use tk_symmetry::{BitPackable, BlockSparseTensor};

/// Adaptive threading strategy for block-sparse operations.
///
/// Mixing Rayon's work-stealing scheduler with multithreaded BLAS backends
/// creates thread oversubscription. Two disjoint regimes are used:
///
/// - **FatSectors**: Few large sectors (bond dimension D > ~500, few sectors).
///   Use the full machine thread pool per BLAS call. Rayon disabled.
///
/// - **FragmentedSectors**: Many small sectors (many symmetry sectors, small D).
///   Force BLAS to single-threaded mode. Rayon distributes independent sector
///   GEMMs in parallel. LPT pre-sorting ensures balanced load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadingRegime {
    /// Use multithreaded BLAS for each operation; do not use Rayon over sectors.
    FatSectors {
        /// Number of threads to grant to the BLAS backend.
        blas_threads: usize,
    },
    /// Use single-threaded BLAS per sector; parallelize over sectors with Rayon.
    FragmentedSectors {
        /// Number of Rayon worker threads.
        rayon_threads: usize,
    },
}

/// Default threshold for max sector dimension above which FatSectors is chosen.
/// Based on empirical BLAS crossover measurements. Should be calibrated per deployment.
const FAT_SECTOR_DIM_THRESHOLD: usize = 500;

impl ThreadingRegime {
    /// Select the appropriate regime for a given block-sparse tensor and core count.
    ///
    /// **Heuristic:** If the maximum sector dimension exceeds 500 and the number of
    /// sectors is less than `n_cores`, use FatSectors (BLAS can fill all cores with
    /// one sector). Otherwise, use FragmentedSectors.
    ///
    /// This threshold (500) is a conservative default. Profiling at D=1000 on target
    /// hardware should calibrate it per deployment.
    pub fn select<T: Scalar, Q: BitPackable>(
        tensor: &BlockSparseTensor<T, Q>,
        n_cores: usize,
    ) -> Self {
        let max_dim = max_sector_dim_any_leg(tensor);
        if max_dim > FAT_SECTOR_DIM_THRESHOLD && tensor.n_sectors() < n_cores {
            ThreadingRegime::FatSectors {
                blas_threads: n_cores,
            }
        } else {
            ThreadingRegime::FragmentedSectors {
                rayon_threads: n_cores,
            }
        }
    }
}

/// Compute the maximum sector dimension across all legs of a block-sparse tensor.
fn max_sector_dim_any_leg<T: Scalar, Q: BitPackable>(
    tensor: &BlockSparseTensor<T, Q>,
) -> usize {
    let rank = tensor.rank();
    (0..rank)
        .map(|leg| tensor.max_sector_dim_on_leg(leg))
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threading_regime_debug_display() {
        let fat = ThreadingRegime::FatSectors { blas_threads: 8 };
        assert_eq!(
            format!("{:?}", fat),
            "FatSectors { blas_threads: 8 }"
        );
        let frag = ThreadingRegime::FragmentedSectors { rayon_threads: 8 };
        assert_eq!(
            format!("{:?}", frag),
            "FragmentedSectors { rayon_threads: 8 }"
        );
    }

    #[test]
    fn threading_regime_equality() {
        let a = ThreadingRegime::FatSectors { blas_threads: 4 };
        let b = ThreadingRegime::FatSectors { blas_threads: 4 };
        let c = ThreadingRegime::FragmentedSectors { rayon_threads: 4 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
