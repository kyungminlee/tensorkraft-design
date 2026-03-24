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

/// FLOP threshold for partitioned dispatch. Tasks with FLOPs above this value
/// are classified as "heavy" and dispatched with multithreaded BLAS. Tasks below
/// this threshold are "light" and batched for Rayon parallel dispatch with
/// single-threaded BLAS.
const BLAS_FLOP_THRESHOLD: usize = 1_000_000; // ~100x100x100

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

    /// Partition tasks into heavy and light groups based on FLOP threshold.
    ///
    /// Returns `(heavy, light)` where:
    /// - `heavy`: tasks with FLOPs >= `BLAS_FLOP_THRESHOLD`, dispatched with
    ///   multithreaded BLAS (one at a time, all cores).
    /// - `light`: tasks with FLOPs < `BLAS_FLOP_THRESHOLD`, batched for Rayon
    ///   parallel dispatch with single-threaded BLAS per task.
    ///
    /// Both lists are LPT-sorted (descending FLOP cost) for optimal load balancing.
    pub(crate) fn partition_tasks<T: Scalar>(
        tasks: &[crate::tasks::SectorGemmTask<'_, T>],
    ) -> (Vec<usize>, Vec<usize>) {
        let mut heavy = Vec::new();
        let mut light = Vec::new();
        for (i, task) in tasks.iter().enumerate() {
            if task.flops >= BLAS_FLOP_THRESHOLD {
                heavy.push(i);
            } else {
                light.push(i);
            }
        }
        // Both are already in LPT order if `tasks` was LPT-sorted beforehand.
        (heavy, light)
    }

    /// Get the FLOP threshold used for partitioned dispatch.
    pub fn blas_flop_threshold() -> usize {
        BLAS_FLOP_THRESHOLD
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

    #[test]
    fn partition_tasks_splits_by_flop_threshold() {
        use crate::tasks::SectorGemmTask;
        use tk_core::TensorShape;
        use tk_symmetry::PackedSectorKey;

        let block = tk_core::DenseTensor::<f64>::zeros(TensorShape::row_major(&[2, 2]));

        let tasks = vec![
            SectorGemmTask {
                out_key: PackedSectorKey(0),
                block_a: &block,
                block_b: &block,
                flops: 2_000_000, // heavy
            },
            SectorGemmTask {
                out_key: PackedSectorKey(1),
                block_a: &block,
                block_b: &block,
                flops: 100, // light
            },
            SectorGemmTask {
                out_key: PackedSectorKey(2),
                block_a: &block,
                block_b: &block,
                flops: 1_000_000, // heavy (exactly at threshold)
            },
            SectorGemmTask {
                out_key: PackedSectorKey(3),
                block_a: &block,
                block_b: &block,
                flops: 999_999, // light (just below)
            },
        ];

        let (heavy, light) = ThreadingRegime::partition_tasks(&tasks);
        assert_eq!(heavy.len(), 2);
        assert_eq!(light.len(), 2);
        assert_eq!(tasks[heavy[0]].flops, 2_000_000);
        assert_eq!(tasks[heavy[1]].flops, 1_000_000);
        assert_eq!(tasks[light[0]].flops, 100);
        assert_eq!(tasks[light[1]].flops, 999_999);
    }

    #[test]
    fn blas_flop_threshold_is_reasonable() {
        let t = ThreadingRegime::blas_flop_threshold();
        assert!(t >= 100_000, "threshold too low: {t}");
        assert!(t <= 100_000_000, "threshold too high: {t}");
    }
}
