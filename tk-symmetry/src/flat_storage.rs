//! `FlatBlockStorage` — contiguous, compute-side read-only block storage.

use tk_core::{DenseTensor, Scalar, SweepArena, TensorShape};

use crate::block_sparse::BlockSparseTensor;
use crate::quantum_number::BitPackable;
use crate::sector_key::PackedSectorKey;

/// Compute-side read-only block storage: all sector data in one contiguous allocation.
/// Enables single-DMA GPU transfer of the entire tensor.
///
/// NOT used during structural mutations (subspace expansion); see mutation layout
/// in `BlockSparseTensor`.
pub struct FlatBlockStorage<'a, T: Scalar> {
    /// Single contiguous buffer containing all sector blocks back-to-back.
    /// Allocated from `SweepArena` (pinned memory when backend-cuda is active),
    /// NOT from the pageable heap. This guarantees DMA-capable memory for
    /// GPU transfers without the NVIDIA driver's hidden pin-copy-unpin dance.
    data: &'a mut [T],
    /// Start index of each sector block within `data`.
    /// `offsets[i]` is the start index of `sector_keys[i]`'s block data.
    offsets: Vec<usize>,
    /// Dimensions (rows, cols) of each sector block.
    shapes: Vec<(usize, usize)>,
}

impl<'a, T: Scalar> FlatBlockStorage<'a, T> {
    /// Read-only access to the contiguous buffer.
    pub fn data(&self) -> &[T] {
        self.data
    }

    /// Mutable access to the contiguous buffer.
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    /// Start offsets for each sector block.
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Shapes (rows, cols) for each sector block.
    pub fn shapes(&self) -> &[(usize, usize)] {
        &self.shapes
    }

    /// Number of sector blocks.
    pub fn n_sectors(&self) -> usize {
        self.offsets.len()
    }

    /// Total number of elements across all blocks.
    pub fn total_elements(&self) -> usize {
        self.data.len()
    }

    /// Get a slice for the i-th sector block.
    pub fn block_slice(&self, i: usize) -> &[T] {
        let start = self.offsets[i];
        let (rows, cols) = self.shapes[i];
        &self.data[start..start + rows * cols]
    }

    /// Get a mutable slice for the i-th sector block.
    pub fn block_slice_mut(&mut self, i: usize) -> &mut [T] {
        let start = self.offsets[i];
        let (rows, cols) = self.shapes[i];
        &mut self.data[start..start + rows * cols]
    }
}

impl<T: Scalar, Q: BitPackable> BlockSparseTensor<T, Q> {
    /// Pack fragmented blocks into a contiguous flat buffer for GPU/GEMM.
    /// Called after structural mutations are complete, before dispatch.
    ///
    /// CRITICAL: The flat buffer is allocated from the `SweepArena`, NOT from
    /// fresh pageable heap memory. When `backend-cuda` is active, the arena
    /// uses pinned memory, so the resulting buffer is directly DMA-capable —
    /// no hidden staging copy by the NVIDIA driver.
    ///
    /// Cost: O(D_total²) — a single memcpy pass, negligible relative to
    /// the O(D³) GEMM it feeds.
    pub fn flatten<'a>(&self, arena: &'a SweepArena) -> FlatBlockStorage<'a, T> {
        let total_elems: usize = self.sector_blocks().iter().map(|b| b.numel()).sum();

        // SAFETY: We fully initialize the buffer in the loop below before
        // any read can occur.
        let buf = unsafe { arena.alloc_slice_uninit::<T>(total_elems) };

        let mut offset = 0;
        let mut offsets = Vec::with_capacity(self.n_sectors());
        let mut shapes = Vec::with_capacity(self.n_sectors());

        for block in self.sector_blocks() {
            offsets.push(offset);
            let dims = block.shape().dims();
            // For rank-2 blocks: (rows, cols). For higher rank, flatten to (numel, 1).
            let shape = if dims.len() == 2 {
                (dims[0], dims[1])
            } else {
                (block.numel(), 1)
            };
            shapes.push(shape);

            let n = block.numel();
            buf[offset..offset + n].copy_from_slice(&block.as_slice()[..n]);
            offset += n;
        }

        FlatBlockStorage {
            data: buf,
            offsets,
            shapes,
        }
    }

    /// Restore fragmented layout from flat buffer (e.g., after GPU computation).
    pub fn unflatten(
        flat: &FlatBlockStorage<T>,
        keys: &[PackedSectorKey],
        indices: Vec<QIndex<Q>>,
        flux: Q,
        leg_directions: Vec<LegDirection>,
    ) -> Self
    where
        Q: BitPackable,
    {
        debug_assert_eq!(flat.n_sectors(), keys.len());

        let mut sector_keys = Vec::with_capacity(keys.len());
        let mut sector_blocks = Vec::with_capacity(keys.len());

        for (i, &key) in keys.iter().enumerate() {
            let slice = flat.block_slice(i);
            let (rows, cols) = flat.shapes[i];
            let shape = TensorShape::row_major(&[rows, cols]);
            let block = DenseTensor::from_vec(shape, slice.to_vec());
            sector_keys.push(key);
            sector_blocks.push(block);
        }

        BlockSparseTensor::from_raw_parts(indices, sector_keys, sector_blocks, flux, leg_directions)
    }
}

use crate::quantum_number::LegDirection;
use crate::sector_key::QIndex;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::U1;
    use crate::quantum_number::QuantumNumber;

    fn make_test_tensor() -> BlockSparseTensor<f64, U1> {
        let idx = QIndex::new(vec![(U1(0), 2), (U1(1), 3)]);
        let indices = vec![idx.clone(), idx.clone()];
        let dirs = vec![LegDirection::Incoming, LegDirection::Outgoing];

        let mut t = BlockSparseTensor::<f64, U1>::zeros(indices, U1::identity(), dirs);

        // Fill blocks with recognizable data
        if let Some(block) = t.get_block_mut(&[U1(0), U1(0)]) {
            for (i, v) in block.as_mut_slice().iter_mut().enumerate() {
                *v = i as f64;
            }
        }
        if let Some(block) = t.get_block_mut(&[U1(1), U1(1)]) {
            for (i, v) in block.as_mut_slice().iter_mut().enumerate() {
                *v = (100 + i) as f64;
            }
        }
        t
    }

    #[test]
    fn flatten_contiguous_data() {
        let t = make_test_tensor();
        let arena = SweepArena::new(64 * 1024);
        let flat = t.flatten(&arena);

        // Verify contiguous buffer matches element-by-element iteration
        let mut expected = Vec::new();
        for block in t.sector_blocks() {
            expected.extend_from_slice(&block.as_slice()[..block.numel()]);
        }
        assert_eq!(flat.data(), &expected[..]);
    }

    #[test]
    fn flatten_offsets_correct() {
        let t = make_test_tensor();
        let arena = SweepArena::new(64 * 1024);
        let flat = t.flatten(&arena);

        let mut cumulative = 0;
        for (i, &offset) in flat.offsets().iter().enumerate() {
            assert_eq!(offset, cumulative);
            let (rows, cols) = flat.shapes()[i];
            cumulative += rows * cols;
        }
        assert_eq!(cumulative, flat.total_elements());
    }

    #[test]
    fn unflatten_round_trip() {
        let t = make_test_tensor();
        let arena = SweepArena::new(64 * 1024);
        let flat = t.flatten(&arena);

        let reconstructed = BlockSparseTensor::<f64, U1>::unflatten(
            &flat,
            t.sector_keys(),
            t.indices().to_vec(),
            t.flux().clone(),
            t.leg_directions().to_vec(),
        );

        // Verify all blocks match
        assert_eq!(reconstructed.n_sectors(), t.n_sectors());
        for (orig_block, recon_block) in t.sector_blocks().iter().zip(reconstructed.sector_blocks())
        {
            assert_eq!(
                &orig_block.as_slice()[..orig_block.numel()],
                &recon_block.as_slice()[..recon_block.numel()],
            );
        }
    }
}
