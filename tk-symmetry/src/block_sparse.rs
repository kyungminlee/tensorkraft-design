//! `BlockSparseTensor<T, Q>` — Abelian block-sparse tensor.

use std::ops::Range;

use hashbrown::HashMap;
use smallvec::SmallVec;
use tk_core::{DenseTensor, Scalar, TensorShape};

use crate::flux::{check_flux_rule, enumerate_valid_sectors};
use crate::quantum_number::{BitPackable, LegDirection};
use crate::sector_key::{PackedSectorKey, QIndex};

/// Block-sparse tensor for systems with Abelian symmetry Q.
///
/// Data is partitioned into dense sub-blocks, one per symmetry sector.
/// Only blocks satisfying the flux rule are stored; all others are zero.
///
/// INVARIANT: `sector_keys` is sorted in ascending order at all times.
/// Any operation that modifies `sector_keys` must restore this invariant.
pub struct BlockSparseTensor<T: Scalar, Q: BitPackable> {
    /// `QIndex` for each tensor leg. `len() == rank`.
    indices: Vec<QIndex<Q>>,
    /// Sorted sector keys (packed multi-leg quantum-number tuples).
    /// Parallel to `sector_blocks`.
    sector_keys: Vec<PackedSectorKey>,
    /// Dense sub-blocks, one per sector.
    /// `sector_blocks[i]` corresponds to `sector_keys[i]`.
    sector_blocks: Vec<DenseTensor<'static, T>>,
    /// Total charge of the tensor. Non-zero for e.g. creation operators.
    flux: Q,
    /// Leg directions for flux rule validation.
    leg_directions: Vec<LegDirection>,
}

impl<T: Scalar, Q: BitPackable> BlockSparseTensor<T, Q> {
    /// Construct a zero tensor with the given leg bases and flux.
    /// Automatically enumerates all sectors satisfying the flux rule
    /// and allocates zero-filled `DenseTensor` blocks for each.
    pub fn zeros(indices: Vec<QIndex<Q>>, flux: Q, leg_directions: Vec<LegDirection>) -> Self {
        debug_assert_eq!(
            indices.len(),
            leg_directions.len(),
            "indices and leg_directions must have the same length"
        );

        let valid_sectors = enumerate_valid_sectors(&indices, &flux, &leg_directions);
        let mut sector_keys = Vec::with_capacity(valid_sectors.len());
        let mut sector_blocks = Vec::with_capacity(valid_sectors.len());

        for sector_qns in &valid_sectors {
            let key = PackedSectorKey::pack(sector_qns);
            let dims: Vec<usize> = sector_qns
                .iter()
                .zip(indices.iter())
                .map(|(q, idx)| idx.dim_of(q).expect("quantum number not found in QIndex"))
                .collect();
            let shape = TensorShape::row_major(&dims);
            sector_keys.push(key);
            sector_blocks.push(DenseTensor::zeros(shape));
        }

        // Sort by key, keeping blocks in sync
        let mut paired: Vec<_> = sector_keys
            .into_iter()
            .zip(sector_blocks.into_iter())
            .collect();
        paired.sort_by_key(|(k, _)| *k);

        let (sector_keys, sector_blocks): (Vec<_>, Vec<_>) = paired.into_iter().unzip();

        let tensor = BlockSparseTensor {
            indices,
            sector_keys,
            sector_blocks,
            flux,
            leg_directions,
        };

        #[cfg(debug_assertions)]
        tensor.assert_invariants();

        tensor
    }

    /// Construct from an explicit list of (sector_key, block) pairs.
    /// Panics in debug mode if any block violates the flux rule,
    /// or if sector_keys are not unique.
    pub fn from_blocks(
        indices: Vec<QIndex<Q>>,
        flux: Q,
        leg_directions: Vec<LegDirection>,
        blocks: Vec<(Vec<Q>, DenseTensor<'static, T>)>,
    ) -> Self {
        let mut sector_keys = Vec::with_capacity(blocks.len());
        let mut sector_blocks = Vec::with_capacity(blocks.len());

        for (qns, block) in blocks {
            #[cfg(debug_assertions)]
            {
                assert!(
                    check_flux_rule(&qns, &flux, &leg_directions),
                    "flux rule violated for sector {:?}",
                    qns,
                );
                // Verify block dimensions match QIndex sectors
                for (i, q) in qns.iter().enumerate() {
                    let expected = indices[i]
                        .dim_of(q)
                        .expect("quantum number not found in QIndex");
                    assert_eq!(
                        block.shape().dims()[i],
                        expected,
                        "block dim mismatch on leg {}: expected {}, got {}",
                        i,
                        expected,
                        block.shape().dims()[i],
                    );
                }
            }
            let key = PackedSectorKey::pack(&qns);
            sector_keys.push(key);
            sector_blocks.push(block);
        }

        // Sort by key
        let mut paired: Vec<_> = sector_keys
            .into_iter()
            .zip(sector_blocks.into_iter())
            .collect();
        paired.sort_by_key(|(k, _)| *k);

        let (sector_keys, sector_blocks): (Vec<_>, Vec<_>) = paired.into_iter().unzip();

        #[cfg(debug_assertions)]
        {
            // Check no duplicate keys
            for i in 1..sector_keys.len() {
                assert_ne!(
                    sector_keys[i - 1],
                    sector_keys[i],
                    "duplicate sector key"
                );
            }
        }

        let tensor = BlockSparseTensor {
            indices,
            sector_keys,
            sector_blocks,
            flux,
            leg_directions,
        };

        #[cfg(debug_assertions)]
        tensor.assert_invariants();

        tensor
    }

    /// O(log N) immutable block lookup. Returns `None` if the sector is absent
    /// (which means all elements in that sector are zero).
    #[inline(always)]
    pub fn get_block(&self, sector_qns: &[Q]) -> Option<&DenseTensor<'static, T>> {
        let key = PackedSectorKey::pack(sector_qns);
        self.sector_keys
            .binary_search(&key)
            .ok()
            .map(|idx| &self.sector_blocks[idx])
    }

    /// O(log N) mutable block lookup.
    #[inline(always)]
    pub fn get_block_mut(&mut self, sector_qns: &[Q]) -> Option<&mut DenseTensor<'static, T>> {
        let key = PackedSectorKey::pack(sector_qns);
        self.sector_keys
            .binary_search(&key)
            .ok()
            .map(|idx| &mut self.sector_blocks[idx])
    }

    /// Insert or overwrite a block. Maintains the sorted key invariant.
    pub fn insert_block(&mut self, sector_qns: Vec<Q>, block: DenseTensor<'static, T>) {
        #[cfg(debug_assertions)]
        {
            assert!(
                check_flux_rule(&sector_qns, &self.flux, &self.leg_directions),
                "flux rule violated for sector {:?}",
                sector_qns,
            );
        }

        let key = PackedSectorKey::pack(&sector_qns);
        match self.sector_keys.binary_search(&key) {
            Ok(idx) => {
                // Overwrite existing block
                self.sector_blocks[idx] = block;
            }
            Err(idx) => {
                // Insert at sorted position
                self.sector_keys.insert(idx, key);
                self.sector_blocks.insert(idx, block);
            }
        }

        #[cfg(debug_assertions)]
        self.assert_invariants();
    }

    /// Iterator over all non-zero (sector_qns, block) pairs.
    pub fn iter_blocks(
        &self,
    ) -> impl Iterator<Item = (SmallVec<[Q; 8]>, &DenseTensor<'static, T>)> {
        let rank = self.rank();
        self.sector_keys
            .iter()
            .zip(self.sector_blocks.iter())
            .map(move |(key, block)| (key.unpack::<Q>(rank), block))
    }

    /// Iterator over all non-zero (key, block) pairs by packed key.
    pub fn iter_keyed_blocks(
        &self,
    ) -> impl Iterator<Item = (PackedSectorKey, &DenseTensor<'static, T>)> {
        self.sector_keys.iter().copied().zip(self.sector_blocks.iter())
    }

    /// Number of tensor legs.
    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    /// Number of non-zero sectors.
    pub fn n_sectors(&self) -> usize {
        self.sector_blocks.len()
    }

    /// The tensor's flux (total charge).
    pub fn flux(&self) -> &Q {
        &self.flux
    }

    /// Read-only access to leg indices.
    pub fn indices(&self) -> &[QIndex<Q>] {
        &self.indices
    }

    /// Read-only access to leg directions.
    pub fn leg_directions(&self) -> &[LegDirection] {
        &self.leg_directions
    }

    /// Read-only access to the sorted sector keys.
    pub fn sector_keys(&self) -> &[PackedSectorKey] {
        &self.sector_keys
    }

    /// Read-only access to the sector blocks.
    pub fn sector_blocks(&self) -> &[DenseTensor<'static, T>] {
        &self.sector_blocks
    }

    /// Total stored element count (sum of all block sizes).
    pub fn nnz(&self) -> usize {
        self.sector_blocks.iter().map(|b| b.numel()).sum()
    }

    /// Maximum dimension across all sectors on one leg.
    pub fn max_sector_dim_on_leg(&self, leg: usize) -> usize {
        self.indices[leg]
            .iter_sectors()
            .map(|(_, _, dim)| dim)
            .max()
            .unwrap_or(0)
    }

    /// Permute tensor legs. Returns a new tensor with rearranged `QIndices`
    /// and re-packed sector keys. Block data is permuted via `DenseTensor::permute`
    /// (zero-copy stride permutation, materialized via `into_owned`).
    pub fn permute(&self, perm: &[usize]) -> Self {
        debug_assert_eq!(perm.len(), self.rank());

        let new_indices: Vec<_> = perm.iter().map(|&i| self.indices[i].clone()).collect();
        let new_directions: Vec<_> = perm.iter().map(|&i| self.leg_directions[i]).collect();
        let rank = self.rank();

        let mut new_keys = Vec::with_capacity(self.n_sectors());
        let mut new_blocks = Vec::with_capacity(self.n_sectors());

        for (key, block) in self.sector_keys.iter().zip(self.sector_blocks.iter()) {
            let old_qns: SmallVec<[Q; 8]> = key.unpack(rank);
            let new_qns: SmallVec<[Q; 8]> = perm.iter().map(|&i| old_qns[i].clone()).collect();
            let new_key = PackedSectorKey::pack(&new_qns);
            let new_block = block.permute(perm).into_owned();
            new_keys.push(new_key);
            new_blocks.push(new_block);
        }

        // Re-sort by key
        let mut paired: Vec<_> = new_keys.into_iter().zip(new_blocks.into_iter()).collect();
        paired.sort_by_key(|(k, _)| *k);
        let (sector_keys, sector_blocks): (Vec<_>, Vec<_>) = paired.into_iter().unzip();

        BlockSparseTensor {
            indices: new_indices,
            sector_keys,
            sector_blocks,
            flux: self.flux.clone(),
            leg_directions: new_directions,
        }
    }

    /// Fuse (combine) a contiguous range of legs into one combined leg.
    ///
    /// The combined QIndex has sectors given by all valid fused quantum numbers.
    /// Multiple original sectors that fuse to the same quantum number are placed
    /// at different offsets within the fused dimension.
    ///
    /// The fused leg direction is `Incoming`. The fused quantum number for a
    /// combination `(q_i, q_{i+1}, ...)` is computed respecting the original
    /// leg directions: `Incoming` contributes `q`, `Outgoing` contributes `q.dual()`.
    ///
    /// Used to reshape MPS tensors before GEMM.
    pub fn fuse_legs(&self, legs: Range<usize>) -> Self {
        let start = legs.start;
        let end = legs.end;
        assert!(
            start < end && end <= self.rank(),
            "fuse_legs: invalid range {}..{} for rank-{} tensor",
            start,
            end,
            self.rank(),
        );

        let n_fuse = end - start;
        if n_fuse == 1 {
            // Nothing to fuse — reconstruct with same data.
            return Self::from_raw_parts(
                self.indices.clone(),
                self.sector_keys.clone(),
                self.sector_blocks.iter().map(|b| {
                    DenseTensor::from_vec(b.shape().clone(), b.as_slice()[..b.numel()].to_vec())
                }).collect(),
                self.flux.clone(),
                self.leg_directions.clone(),
            );
        }

        // ---------------------------------------------------------------
        // 1. Build the fused QIndex via Cartesian product of sub-leg sectors.
        // ---------------------------------------------------------------

        // Iterative Cartesian product: (sub_qns, fused_q, product_dim)
        let mut combos: Vec<(SmallVec<[Q; 4]>, Q, usize)> =
            vec![(SmallVec::new(), Q::identity(), 1)];

        for leg_idx in start..end {
            let idx = &self.indices[leg_idx];
            let dir = self.leg_directions[leg_idx];
            let mut new_combos = Vec::with_capacity(combos.len() * idx.n_sectors());

            for (qns, partial_fused, partial_dim) in &combos {
                for &(ref q, dim) in idx.sectors() {
                    let mut new_qns = qns.clone();
                    new_qns.push(q.clone());
                    let effective_q = match dir {
                        LegDirection::Incoming => q.clone(),
                        LegDirection::Outgoing => q.dual(),
                    };
                    let new_fused = partial_fused.fuse(&effective_q);
                    new_combos.push((new_qns, new_fused, partial_dim * dim));
                }
            }
            combos = new_combos;
        }

        // Group by fused quantum number. Use BTreeMap for deterministic sector order.
        let mut fused_groups: std::collections::BTreeMap<Q, Vec<(SmallVec<[Q; 4]>, usize)>> =
            std::collections::BTreeMap::new();
        for (qns, fused_q, dim) in &combos {
            fused_groups
                .entry(fused_q.clone())
                .or_default()
                .push((qns.clone(), *dim));
        }

        // Build fused QIndex and an offset map: sub_qns -> (fused_q, offset_in_fused)
        let mut fused_qindex_sectors: Vec<(Q, usize)> = Vec::new();
        let mut offset_map: HashMap<SmallVec<[Q; 4]>, (Q, usize)> = HashMap::new();

        for (fused_q, sub_sectors) in &fused_groups {
            let mut offset = 0;
            for (qns, dim) in sub_sectors {
                offset_map.insert(qns.clone(), (fused_q.clone(), offset));
                offset += dim;
            }
            fused_qindex_sectors.push((fused_q.clone(), offset)); // offset is now total_dim
        }

        let fused_qindex = QIndex::new(fused_qindex_sectors);

        // ---------------------------------------------------------------
        // 2. Build new indices and directions.
        // ---------------------------------------------------------------

        let mut new_indices: Vec<QIndex<Q>> = Vec::with_capacity(self.rank() - n_fuse + 1);
        let mut new_directions: Vec<LegDirection> = Vec::with_capacity(self.rank() - n_fuse + 1);

        for i in 0..start {
            new_indices.push(self.indices[i].clone());
            new_directions.push(self.leg_directions[i]);
        }
        new_indices.push(fused_qindex.clone());
        new_directions.push(LegDirection::Incoming);
        for i in end..self.rank() {
            new_indices.push(self.indices[i].clone());
            new_directions.push(self.leg_directions[i]);
        }

        // ---------------------------------------------------------------
        // 3. Scatter original blocks into fused blocks.
        // ---------------------------------------------------------------

        // Key: new sector quantum numbers -> block
        let mut new_block_map: HashMap<SmallVec<[Q; 8]>, DenseTensor<'static, T>> = HashMap::new();

        for (key, block) in self.sector_keys.iter().zip(self.sector_blocks.iter()) {
            let old_qns: SmallVec<[Q; 8]> = key.unpack(self.rank());
            let old_dims = block.shape().dims();

            // Quantum numbers on the fused legs
            let fuse_qns: SmallVec<[Q; 4]> = old_qns[start..end].iter().cloned().collect();
            let (fused_q, fused_offset) = offset_map
                .get(&fuse_qns)
                .expect("fuse_legs: combination not found in offset_map");

            // New sector quantum numbers
            let mut new_qns: SmallVec<[Q; 8]> = SmallVec::new();
            for i in 0..start {
                new_qns.push(old_qns[i].clone());
            }
            new_qns.push(fused_q.clone());
            for i in end..self.rank() {
                new_qns.push(old_qns[i].clone());
            }

            // Dimensions
            let fuse_dim: usize = old_dims[start..end].iter().product();
            let fused_total_dim = fused_qindex.dim_of(fused_q).unwrap();

            let new_dims: SmallVec<[usize; 8]> = {
                let mut d = SmallVec::new();
                for i in 0..start {
                    d.push(old_dims[i]);
                }
                d.push(fused_total_dim);
                for i in end..self.rank() {
                    d.push(old_dims[i]);
                }
                d
            };

            let new_block = new_block_map.entry(new_qns).or_insert_with(|| {
                DenseTensor::zeros(TensorShape::row_major(&new_dims))
            });

            // Copy data: for each "outer" index, copy the fused chunk.
            let outer_size: usize = old_dims[..start].iter().product::<usize>().max(1);
            let trail_size: usize = old_dims[end..].iter().product::<usize>().max(1);

            let src = block.as_slice();
            let dst = new_block.as_mut_slice();

            for o in 0..outer_size {
                let src_start = o * fuse_dim * trail_size;
                let dst_start =
                    o * fused_total_dim * trail_size + fused_offset * trail_size;
                let chunk_size = fuse_dim * trail_size;
                dst[dst_start..dst_start + chunk_size]
                    .copy_from_slice(&src[src_start..src_start + chunk_size]);
            }
        }

        // Convert map to blocks list for from_blocks
        let blocks: Vec<(Vec<Q>, DenseTensor<'static, T>)> = new_block_map
            .into_iter()
            .map(|(qns, block)| (qns.to_vec(), block))
            .collect();

        Self::from_blocks(new_indices, self.flux.clone(), new_directions, blocks)
    }

    /// Split one fused leg back into its component legs.
    ///
    /// Inverse of `fuse_legs`. Requires the original QIndex and direction
    /// information for each sub-leg.
    ///
    /// # Arguments
    /// - `leg`: index of the fused leg to split
    /// - `original_indices`: `QIndex` for each sub-leg (in order)
    /// - `original_directions`: direction for each sub-leg
    pub fn split_leg(
        &self,
        leg: usize,
        original_indices: Vec<QIndex<Q>>,
        original_directions: Vec<LegDirection>,
    ) -> Self {
        assert!(leg < self.rank(), "split_leg: leg {} out of range", leg);
        assert_eq!(
            original_indices.len(),
            original_directions.len(),
            "split_leg: indices and directions length mismatch",
        );

        let n_split = original_indices.len();
        if n_split <= 1 {
            return Self::from_raw_parts(
                self.indices.clone(),
                self.sector_keys.clone(),
                self.sector_blocks.iter().map(|b| {
                    DenseTensor::from_vec(b.shape().clone(), b.as_slice()[..b.numel()].to_vec())
                }).collect(),
                self.flux.clone(),
                self.leg_directions.clone(),
            );
        }

        // ---------------------------------------------------------------
        // 1. Reconstruct the fuse map (same logic as fuse_legs).
        // ---------------------------------------------------------------

        let mut combos: Vec<(SmallVec<[Q; 4]>, Q, usize)> =
            vec![(SmallVec::new(), Q::identity(), 1)];

        for (i, idx) in original_indices.iter().enumerate() {
            let dir = original_directions[i];
            let mut new_combos = Vec::with_capacity(combos.len() * idx.n_sectors());

            for (qns, partial_fused, partial_dim) in &combos {
                for &(ref q, dim) in idx.sectors() {
                    let mut new_qns = qns.clone();
                    new_qns.push(q.clone());
                    let effective_q = match dir {
                        LegDirection::Incoming => q.clone(),
                        LegDirection::Outgoing => q.dual(),
                    };
                    new_combos.push((new_qns, partial_fused.fuse(&effective_q), partial_dim * dim));
                }
            }
            combos = new_combos;
        }

        // offset_map: sub_qns -> (fused_q, offset_in_fused)
        let mut fused_groups: std::collections::BTreeMap<Q, Vec<(SmallVec<[Q; 4]>, usize)>> =
            std::collections::BTreeMap::new();
        for (qns, fused_q, dim) in &combos {
            fused_groups
                .entry(fused_q.clone())
                .or_default()
                .push((qns.clone(), *dim));
        }

        let mut offset_map: HashMap<SmallVec<[Q; 4]>, (Q, usize)> = HashMap::new();
        for (fused_q, sub_sectors) in &fused_groups {
            let mut offset = 0;
            for (qns, dim) in sub_sectors {
                offset_map.insert(qns.clone(), (fused_q.clone(), offset));
                offset += dim;
            }
        }

        // Reverse map: (fused_q) -> [(sub_qns, offset, product_dim)]
        let mut reverse_map: HashMap<Q, Vec<(SmallVec<[Q; 4]>, usize, usize)>> = HashMap::new();
        for (qns, (fused_q, offset)) in &offset_map {
            let dim: usize = qns
                .iter()
                .zip(original_indices.iter())
                .map(|(q, idx)| idx.dim_of(q).unwrap())
                .product();
            reverse_map
                .entry(fused_q.clone())
                .or_default()
                .push((qns.clone(), *offset, dim));
        }

        // ---------------------------------------------------------------
        // 2. Build new indices and directions.
        // ---------------------------------------------------------------

        let new_rank = self.rank() - 1 + n_split;
        let mut new_indices: Vec<QIndex<Q>> = Vec::with_capacity(new_rank);
        let mut new_directions: Vec<LegDirection> = Vec::with_capacity(new_rank);

        for i in 0..leg {
            new_indices.push(self.indices[i].clone());
            new_directions.push(self.leg_directions[i]);
        }
        for (i, idx) in original_indices.iter().enumerate() {
            new_indices.push(idx.clone());
            new_directions.push(original_directions[i]);
        }
        for i in (leg + 1)..self.rank() {
            new_indices.push(self.indices[i].clone());
            new_directions.push(self.leg_directions[i]);
        }

        // ---------------------------------------------------------------
        // 3. Gather: extract sub-blocks from each fused block.
        // ---------------------------------------------------------------

        let mut all_blocks: Vec<(Vec<Q>, DenseTensor<'static, T>)> = Vec::new();

        for (key, block) in self.sector_keys.iter().zip(self.sector_blocks.iter()) {
            let old_qns: SmallVec<[Q; 8]> = key.unpack(self.rank());
            let old_dims = block.shape().dims();
            let fused_q = &old_qns[leg];
            let fused_total_dim = old_dims[leg];

            let sub_entries = match reverse_map.get(fused_q) {
                Some(entries) => entries,
                None => continue,
            };

            let outer_size: usize = old_dims[..leg].iter().product::<usize>().max(1);
            let trail_size: usize = old_dims[leg + 1..].iter().product::<usize>().max(1);

            for (sub_qns, fused_offset, product_dim) in sub_entries {
                // Build new sector quantum numbers
                let mut new_qns: Vec<Q> = Vec::with_capacity(new_rank);
                for i in 0..leg {
                    new_qns.push(old_qns[i].clone());
                }
                for q in sub_qns.iter() {
                    new_qns.push(q.clone());
                }
                for i in (leg + 1)..self.rank() {
                    new_qns.push(old_qns[i].clone());
                }

                // Build new block shape
                let mut new_dims: Vec<usize> = Vec::with_capacity(new_rank);
                for i in 0..leg {
                    new_dims.push(old_dims[i]);
                }
                for (q, idx) in sub_qns.iter().zip(original_indices.iter()) {
                    new_dims.push(idx.dim_of(q).unwrap());
                }
                for i in (leg + 1)..self.rank() {
                    new_dims.push(old_dims[i]);
                }

                let mut new_block =
                    DenseTensor::zeros(TensorShape::row_major(&new_dims));

                // Copy data: inverse of fuse_legs scatter
                let src = block.as_slice();
                let dst = new_block.as_mut_slice();

                for o in 0..outer_size {
                    let src_start =
                        o * fused_total_dim * trail_size + fused_offset * trail_size;
                    let dst_start = o * product_dim * trail_size;
                    let chunk_size = product_dim * trail_size;
                    dst[dst_start..dst_start + chunk_size]
                        .copy_from_slice(&src[src_start..src_start + chunk_size]);
                }

                all_blocks.push((new_qns, new_block));
            }
        }

        Self::from_blocks(new_indices, self.flux.clone(), new_directions, all_blocks)
    }

    /// Verify internal invariants (debug/test builds only).
    #[cfg(debug_assertions)]
    pub fn assert_invariants(&self) {
        // 1. sector_keys is strictly sorted
        for i in 1..self.sector_keys.len() {
            assert!(
                self.sector_keys[i - 1] < self.sector_keys[i],
                "sector_keys not strictly sorted at index {}",
                i,
            );
        }

        // 2. Each block's shape matches the QIndex sectors
        let rank = self.rank();
        for (key, block) in self.sector_keys.iter().zip(self.sector_blocks.iter()) {
            let qns: SmallVec<[Q; 8]> = key.unpack(rank);
            for (leg, q) in qns.iter().enumerate() {
                let expected_dim = self.indices[leg]
                    .dim_of(q)
                    .unwrap_or_else(|| panic!("sector {:?} not found in leg {}", q, leg));
                assert_eq!(
                    block.shape().dims()[leg],
                    expected_dim,
                    "block dim mismatch on leg {} for sector {:?}",
                    leg,
                    q,
                );
            }
        }

        // 3. Each sector satisfies the flux rule
        for key in &self.sector_keys {
            let qns: SmallVec<[Q; 8]> = key.unpack(rank);
            assert!(
                check_flux_rule(&qns, &self.flux, &self.leg_directions),
                "flux rule violated for sector {:?}",
                qns,
            );
        }
    }

    /// Construct from pre-sorted raw parts. Used by `unflatten`.
    ///
    /// # Safety contract (debug-checked)
    /// - `sector_keys` must be sorted and have no duplicates.
    /// - Each block must satisfy the flux rule.
    pub(crate) fn from_raw_parts(
        indices: Vec<QIndex<Q>>,
        sector_keys: Vec<PackedSectorKey>,
        sector_blocks: Vec<DenseTensor<'static, T>>,
        flux: Q,
        leg_directions: Vec<LegDirection>,
    ) -> Self {
        let tensor = BlockSparseTensor {
            indices,
            sector_keys,
            sector_blocks,
            flux,
            leg_directions,
        };

        #[cfg(debug_assertions)]
        tensor.assert_invariants();

        tensor
    }
}

impl<T: Scalar, Q: BitPackable> std::fmt::Debug for BlockSparseTensor<T, Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockSparseTensor")
            .field("rank", &self.rank())
            .field("n_sectors", &self.n_sectors())
            .field("nnz", &self.nnz())
            .field("flux", &self.flux)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::U1;
    use crate::quantum_number::QuantumNumber;

    fn make_test_indices() -> (Vec<QIndex<U1>>, Vec<LegDirection>) {
        let idx = QIndex::new(vec![(U1(-1), 2), (U1(0), 3), (U1(1), 2)]);
        let indices = vec![idx.clone(), idx.clone(), idx.clone()];
        let dirs = vec![
            LegDirection::Incoming,
            LegDirection::Incoming,
            LegDirection::Outgoing,
        ];
        (indices, dirs)
    }

    #[test]
    fn block_sparse_zeros_valid_sectors() {
        let (indices, dirs) = make_test_indices();
        let t = BlockSparseTensor::<f64, U1>::zeros(indices, U1::identity(), dirs);
        // All sectors should satisfy the flux rule
        assert!(t.n_sectors() > 0);
        assert_eq!(t.nnz(), t.sector_blocks.iter().map(|b| b.numel()).sum::<usize>());
    }

    #[test]
    fn block_sparse_get_block_present() {
        let (indices, dirs) = make_test_indices();
        let t = BlockSparseTensor::<f64, U1>::zeros(indices, U1::identity(), dirs);
        // U1(0) + U1(0) - U1(0) = 0 should be a valid sector
        let block = t.get_block(&[U1(0), U1(0), U1(0)]);
        assert!(block.is_some());
        let block = block.unwrap();
        assert_eq!(block.shape().dims(), &[3, 3, 3]);
    }

    #[test]
    fn block_sparse_get_block_absent() {
        let (indices, dirs) = make_test_indices();
        let t = BlockSparseTensor::<f64, U1>::zeros(indices, U1::identity(), dirs);
        // U1(0) + U1(0) - U1(1) = -1 ≠ 0, so this sector should be absent
        assert!(t.get_block(&[U1(0), U1(0), U1(1)]).is_none());
    }

    #[test]
    fn block_sparse_sector_key_sorted() {
        let (indices, dirs) = make_test_indices();
        let mut t = BlockSparseTensor::<f64, U1>::zeros(indices, U1::identity(), dirs);
        // Insert a block — keys should remain sorted
        let block = DenseTensor::<f64>::zeros(TensorShape::row_major(&[2, 3, 2]));
        t.insert_block(vec![U1(-1), U1(0), U1(-1)], block);
        for i in 1..t.sector_keys.len() {
            assert!(t.sector_keys[i - 1] < t.sector_keys[i]);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "flux rule violated")]
    fn block_sparse_flux_rule_enforced() {
        let (indices, dirs) = make_test_indices();
        let mut t = BlockSparseTensor::<f64, U1>::zeros(indices, U1::identity(), dirs);
        // U1(0) + U1(0) - U1(1) = -1 ≠ 0 — should panic
        let block = DenseTensor::<f64>::zeros(TensorShape::row_major(&[3, 3, 2]));
        t.insert_block(vec![U1(0), U1(0), U1(1)], block);
    }

    #[test]
    fn block_sparse_permute_numel() {
        let (indices, dirs) = make_test_indices();
        let t = BlockSparseTensor::<f64, U1>::zeros(indices, U1::identity(), dirs);
        let original_nnz = t.nnz();
        let permuted = t.permute(&[2, 0, 1]);
        assert_eq!(permuted.nnz(), original_nnz);
    }

    // --- fuse_legs / split_leg tests ---

    /// Helper: build a rank-3 tensor with recognizable data in each block.
    fn make_filled_rank3() -> BlockSparseTensor<f64, U1> {
        let idx = QIndex::new(vec![(U1(-1), 2), (U1(0), 3), (U1(1), 2)]);
        let indices = vec![idx.clone(), idx.clone(), idx.clone()];
        let dirs = vec![
            LegDirection::Incoming,
            LegDirection::Incoming,
            LegDirection::Outgoing,
        ];
        let mut t = BlockSparseTensor::<f64, U1>::zeros(indices, U1::identity(), dirs);

        // Fill each block with a distinct value based on sector qns
        for (key, block) in t.sector_keys.clone().iter().zip(t.sector_blocks.iter_mut()) {
            let qns: SmallVec<[U1; 8]> = key.unpack(3);
            let val = (qns[0].0 * 100 + qns[1].0 * 10 + qns[2].0) as f64;
            for v in block.as_mut_slice().iter_mut() {
                *v = val;
            }
        }
        t
    }

    #[test]
    fn fuse_legs_rank_reduction() {
        let t = make_filled_rank3();
        let original_nnz = t.nnz();

        // Fuse legs 1..3 (legs 1 and 2) → rank-2 tensor
        let fused = t.fuse_legs(1..3);
        assert_eq!(fused.rank(), 2);
        assert_eq!(fused.nnz(), original_nnz);
    }

    #[test]
    fn fuse_legs_preserves_nnz() {
        let t = make_filled_rank3();
        let original_nnz = t.nnz();

        // Fuse legs 0..2 (legs 0 and 1) → rank-2 tensor
        let fused = t.fuse_legs(0..2);
        assert_eq!(fused.nnz(), original_nnz);
    }

    #[test]
    fn fuse_all_legs() {
        let t = make_filled_rank3();
        let original_nnz = t.nnz();

        // Fuse all 3 legs → rank-1 tensor
        let fused = t.fuse_legs(0..3);
        assert_eq!(fused.rank(), 1);
        assert_eq!(fused.nnz(), original_nnz);
    }

    #[test]
    fn fuse_single_leg_is_identity() {
        let t = make_filled_rank3();

        // Fusing a single leg should be a no-op
        let fused = t.fuse_legs(1..2);
        assert_eq!(fused.rank(), t.rank());
        assert_eq!(fused.n_sectors(), t.n_sectors());
        assert_eq!(fused.nnz(), t.nnz());
    }

    #[test]
    fn fuse_then_split_round_trip() {
        let t = make_filled_rank3();

        // Save original leg info
        let orig_indices_12: Vec<QIndex<U1>> = vec![
            t.indices()[1].clone(),
            t.indices()[2].clone(),
        ];
        let orig_dirs_12: Vec<LegDirection> = vec![
            t.leg_directions()[1],
            t.leg_directions()[2],
        ];

        // Fuse legs 1..3 → rank-2
        let fused = t.fuse_legs(1..3);
        assert_eq!(fused.rank(), 2);

        // Split leg 1 back → rank-3
        let split = fused.split_leg(1, orig_indices_12, orig_dirs_12);
        assert_eq!(split.rank(), 3);
        assert_eq!(split.n_sectors(), t.n_sectors());
        assert_eq!(split.nnz(), t.nnz());

        // Verify data matches the original
        for (key, block) in t.sector_keys.iter().zip(t.sector_blocks.iter()) {
            let qns: SmallVec<[U1; 8]> = key.unpack(3);
            let recovered = split.get_block(&qns).expect("sector should exist");
            assert_eq!(
                &block.as_slice()[..block.numel()],
                &recovered.as_slice()[..recovered.numel()],
                "data mismatch for sector {:?}",
                qns,
            );
        }
    }

    #[test]
    fn fuse_then_split_round_trip_first_legs() {
        let t = make_filled_rank3();

        let orig_indices_01: Vec<QIndex<U1>> = vec![
            t.indices()[0].clone(),
            t.indices()[1].clone(),
        ];
        let orig_dirs_01: Vec<LegDirection> = vec![
            t.leg_directions()[0],
            t.leg_directions()[1],
        ];

        // Fuse legs 0..2 → rank-2
        let fused = t.fuse_legs(0..2);
        assert_eq!(fused.rank(), 2);

        // Split leg 0 back → rank-3
        let split = fused.split_leg(0, orig_indices_01, orig_dirs_01);
        assert_eq!(split.rank(), 3);
        assert_eq!(split.nnz(), t.nnz());

        // Verify data
        for (key, block) in t.sector_keys.iter().zip(t.sector_blocks.iter()) {
            let qns: SmallVec<[U1; 8]> = key.unpack(3);
            let recovered = split.get_block(&qns).expect("sector should exist");
            assert_eq!(
                &block.as_slice()[..block.numel()],
                &recovered.as_slice()[..recovered.numel()],
                "data mismatch for sector {:?}",
                qns,
            );
        }
    }

    #[test]
    fn fuse_with_mixed_directions() {
        // Rank-2 matrix: one incoming, one outgoing
        let idx = QIndex::new(vec![(U1(0), 2), (U1(1), 3)]);
        let indices = vec![idx.clone(), idx.clone()];
        let dirs = vec![LegDirection::Incoming, LegDirection::Outgoing];
        let t = BlockSparseTensor::<f64, U1>::zeros(indices, U1::identity(), dirs);
        let original_nnz = t.nnz();

        // Fuse both legs → rank-1
        let fused = t.fuse_legs(0..2);
        assert_eq!(fused.rank(), 1);
        assert_eq!(fused.nnz(), original_nnz);
    }
}
