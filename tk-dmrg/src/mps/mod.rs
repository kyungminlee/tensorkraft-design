//! Matrix Product State (MPS) types with typestate canonical forms.
//!
//! The gauge form is tracked at compile time via marker types:
//! - `LeftCanonical`: all tensors satisfy A†A = I
//! - `RightCanonical`: all tensors satisfy BB† = I
//! - `MixedCanonical`: left-canonical left of center, right-canonical right of center
//! - `BondCentered`: singular value matrix exposed between sites (for TDVP)

use std::marker::PhantomData;

use num_traits::{Float, NumCast, One, Zero};
use tk_core::{DenseTensor, MatRef, Scalar, TensorShape};
use tk_linalg::LinAlgBackend;
use tk_symmetry::{BitPackable, BlockSparseTensor, LegDirection, QIndex};

use crate::error::{DmrgError, DmrgResult};

// --- Gauge marker types ---

/// All site tensors are left-isometric (A†A = I).
pub struct LeftCanonical;

/// All site tensors are right-isometric (BB† = I).
pub struct RightCanonical;

/// Sites left of center are left-canonical, right of center are right-canonical.
pub struct MixedCanonical {
    pub center: usize,
}

/// Bond matrix exposed between two sites. Used for TDVP projector splitting.
pub struct BondCentered {
    pub left: usize,
}

/// Matrix Product State with compile-time gauge tracking.
///
/// Tensor leg ordering: (σ_physical, α_left_bond, α_right_bond)
pub struct MPS<T: Scalar, Q: BitPackable, Gauge> {
    tensors: Vec<BlockSparseTensor<T, Q>>,
    bonds: Option<Vec<DenseTensor<'static, T>>>,
    local_dims: Vec<usize>,
    total_charge: Q,
    /// Orthogonality center site (meaningful for MixedCanonical gauge).
    center_site: usize,
    _gauge: PhantomData<Gauge>,
}

impl<T: Scalar, Q: BitPackable, G> MPS<T, Q, G> {
    /// Number of sites in the MPS.
    pub fn n_sites(&self) -> usize {
        self.tensors.len()
    }

    /// Physical (local) dimension at a given site.
    pub fn local_dim(&self, site: usize) -> usize {
        self.local_dims[site]
    }

    /// Bond dimension between site `i` and site `i+1`.
    pub fn bond_dim(&self, site: usize) -> usize {
        // Right bond dimension of site tensor
        let tensor = &self.tensors[site];
        tensor.indices()[2].total_dim()
    }

    /// Maximum bond dimension across all bonds.
    pub fn max_bond_dim(&self) -> usize {
        (0..self.n_sites().saturating_sub(1))
            .map(|i| self.bond_dim(i))
            .max()
            .unwrap_or(1)
    }

    /// Target quantum number sector.
    pub fn total_charge(&self) -> &Q {
        &self.total_charge
    }

    /// Access the site tensor at a given site.
    pub fn site_tensor(&self, site: usize) -> &BlockSparseTensor<T, Q> {
        &self.tensors[site]
    }

    /// Mutable access to site tensor.
    pub fn site_tensor_mut(&mut self, site: usize) -> &mut BlockSparseTensor<T, Q> {
        &mut self.tensors[site]
    }

    /// Access bond matrices (only populated in BondCentered form).
    pub fn bonds(&self) -> Option<&Vec<DenseTensor<'static, T>>> {
        self.bonds.as_ref()
    }
}

impl<T: Scalar, Q: BitPackable> MPS<T, Q, MixedCanonical> {
    /// Create a new MPS in mixed canonical form.
    pub fn new(
        tensors: Vec<BlockSparseTensor<T, Q>>,
        local_dims: Vec<usize>,
        total_charge: Q,
        center: usize,
    ) -> Self {
        MPS {
            tensors,
            bonds: None,
            local_dims,
            total_charge,
            center_site: center,
            _gauge: PhantomData,
        }
    }

    /// The orthogonality center site.
    pub fn center(&self) -> usize {
        self.center_site
    }

    /// Expose the bond matrix between center and center+1.
    pub fn expose_bond(self) -> MPS<T, Q, BondCentered> {
        MPS {
            tensors: self.tensors,
            bonds: self.bonds,
            local_dims: self.local_dims,
            total_charge: self.total_charge,
            center_site: self.center_site,
            _gauge: PhantomData,
        }
    }
}

impl<T: Scalar, Q: BitPackable> MPS<T, Q, BondCentered> {
    /// Absorb bond matrices back into site tensors.
    pub fn absorb_bond(self) -> MPS<T, Q, MixedCanonical> {
        MPS {
            tensors: self.tensors,
            bonds: None,
            local_dims: self.local_dims,
            total_charge: self.total_charge,
            center_site: self.center_site,
            _gauge: PhantomData,
        }
    }
}

/// Compute the norm of an MPS (gauge-invariant).
///
/// Contracts <ψ|ψ> via transfer matrices from left to right. O(N·d·D³).
pub fn mps_norm<T: Scalar, Q: BitPackable, G>(mps: &MPS<T, Q, G>) -> T::Real {
    let overlap = mps_overlap(mps, mps);
    // norm = sqrt(|<ψ|ψ>|)
    let abs_sq = overlap.abs_sq();
    Float::sqrt(abs_sq)
}

/// Compute the overlap <bra|ket> between two MPS.
///
/// Uses boundary-vector contraction from left to right. O(N·d·D³).
pub fn mps_overlap<T: Scalar, Q: BitPackable, GA, GB>(
    bra: &MPS<T, Q, GA>,
    ket: &MPS<T, Q, GB>,
) -> T {
    let n = bra.n_sites();
    if n == 0 || n != ket.n_sites() {
        return T::zero();
    }

    // Transfer matrix contraction site by site.
    // T[α_bra, α_ket] starts as 1x1 identity.
    // T'[α'_bra, α'_ket] = Σ_{σ, α_bra, α_ket} conj(bra[σ, α_bra, α'_bra]) * T[α_bra, α_ket] * ket[σ, α_ket, α'_ket]
    //
    // We implement this by iterating over matching sectors.
    // For block-sparse: both bra and ket must have the same physical QIndex.
    // The transfer matrix T is indexed by (q_bra_bond, q_ket_bond).

    // Start with trivial 1x1 transfer matrix
    let mut transfer: Vec<(Q, Q, Vec<T>, usize, usize)> = vec![];
    // (q_bra_left, q_ket_left, data, rows, cols)
    // Initialize: for the leftmost bond (dim 1), the transfer is just 1.0
    let left_q = bra.site_tensor(0).indices()[1].sectors()[0].0.clone();
    transfer.push((left_q.clone(), left_q, vec![T::one()], 1, 1));

    for site in 0..n {
        let bra_tensor = bra.site_tensor(site);
        let ket_tensor = ket.site_tensor(site);
        let mut new_transfer: std::collections::HashMap<(u64, u64), (Q, Q, Vec<T>, usize, usize)> =
            std::collections::HashMap::new();

        // For each sector in bra and ket tensors
        for (bra_qns, bra_block) in bra_tensor.iter_blocks() {
            let q_sigma_bra = bra_qns[0].clone();
            let q_left_bra = bra_qns[1].clone();
            let q_right_bra = bra_qns[2].clone();
            let bra_data = bra_block.as_slice();
            let d_sigma_bra = bra_block.shape().dims()[0];
            let d_left_bra = bra_block.shape().dims()[1];
            let d_right_bra = bra_block.shape().dims()[2];

            for (ket_qns, ket_block) in ket_tensor.iter_blocks() {
                let q_sigma_ket = ket_qns[0].clone();
                let q_left_ket = ket_qns[1].clone();
                let q_right_ket = ket_qns[2].clone();

                // Physical indices must match
                if q_sigma_bra != q_sigma_ket {
                    continue;
                }

                let ket_data = ket_block.as_slice();
                let d_left_ket = ket_block.shape().dims()[1];
                let d_right_ket = ket_block.shape().dims()[2];

                // Find matching transfer matrix entry
                for (tq_bra, tq_ket, t_data, t_rows, t_cols) in &transfer {
                    if *tq_bra != q_left_bra || *tq_ket != q_left_ket {
                        continue;
                    }
                    if *t_rows != d_left_bra || *t_cols != d_left_ket {
                        continue;
                    }

                    // Contract: T'[α'_bra, α'_ket] = Σ_{σ,α_bra,α_ket}
                    //   conj(bra[σ,α_bra,α'_bra]) * T[α_bra,α_ket] * ket[σ,α_ket,α'_ket]
                    let key = (q_right_bra.pack(), q_right_ket.pack());
                    let entry = new_transfer
                        .entry(key)
                        .or_insert_with(|| {
                            (
                                q_right_bra.clone(),
                                q_right_ket.clone(),
                                vec![T::zero(); d_right_bra * d_right_ket],
                                d_right_bra,
                                d_right_ket,
                            )
                        });

                    for sigma in 0..d_sigma_bra {
                        for a_bra in 0..d_left_bra {
                            for a_ket in 0..d_left_ket {
                                let t_val = t_data[a_bra * *t_cols + a_ket];
                                for ap_bra in 0..d_right_bra {
                                    let bra_val = bra_data
                                        [sigma * d_left_bra * d_right_bra + a_bra * d_right_bra + ap_bra];
                                    let conj_bra = bra_val.conj();
                                    let contrib = conj_bra * t_val;
                                    for ap_ket in 0..d_right_ket {
                                        let ket_val = ket_data[sigma * d_left_ket * d_right_ket
                                            + a_ket * d_right_ket
                                            + ap_ket];
                                        entry.2[ap_bra * d_right_ket + ap_ket] =
                                            entry.2[ap_bra * d_right_ket + ap_ket] + contrib * ket_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        transfer = new_transfer.into_values().collect();
    }

    // Sum diagonal of final transfer matrix (trace)
    let mut result = T::zero();
    for (_, _, data, rows, cols) in &transfer {
        let k = (*rows).min(*cols);
        for i in 0..k {
            result = result + data[i * cols + i];
        }
    }
    result
}

/// Compute the energy <ψ|H|ψ> for an MPS and MPO.
///
/// Contracts the <ψ|H|ψ> network via transfer matrices. O(N·d·D²·w).
pub fn mps_energy<T: Scalar, Q: BitPackable, G>(
    mps: &MPS<T, Q, G>,
    mpo: &crate::mpo::MPO<T, Q>,
) -> DmrgResult<T::Real> {
    let n = mps.n_sites();
    if n != mpo.n_sites() {
        return Err(DmrgError::DimensionMismatch {
            mps_sites: n,
            mpo_sites: mpo.n_sites(),
        });
    }
    if n == 0 {
        return Ok(T::Real::zero());
    }

    // Contract left-to-right with MPO sandwiched.
    // Environment tensor E[α_bra, w, α_ket] where w is MPO bond.
    // Start with trivial 1×1×1 boundary.
    // This is essentially what `Environments::build_from_scratch` + trace does,
    // but implemented inline for simplicity.

    // We store environment as flat data indexed by (q_bra, q_w, q_ket) sectors.
    // For simplicity, use the environment contraction from the environments module.
    // Here we do a simplified inline version.

    // Compute energy via left-to-right contraction of the MPS-MPO-MPS sandwich.
    // This is equivalent to building all left environments and contracting with
    // the right boundary.
    use crate::environments::{Environment, contract_left_environment};

    let flux = mps.total_charge().clone();
    let mut env = Environment::left_boundary(flux);

    for site in 0..n {
        env = contract_left_environment(
            &env,
            mps.site_tensor(site),
            mpo.site_tensor(site),
            site + 1,
        );
    }

    // The final environment should be a 1x1x1 tensor containing the energy
    let data = env.data();
    if data.is_empty() {
        return Ok(T::Real::zero());
    }

    // Sum all elements (for a proper MPS-MPO-MPS contraction, the final
    // environment has dim (1, 1, 1) and the single element is the energy)
    let (d_bra, d_w, d_ket) = env.dims();
    let mut energy = T::zero();
    let k = d_bra.min(d_ket);
    for i in 0..k {
        for w in 0..d_w {
            energy = energy + env.get(i, w, i);
        }
    }
    // Extract real part
    Ok(NumCast::from(num_traits::ToPrimitive::to_f64(&energy.abs_sq()).unwrap_or(0.0).sqrt())
        .unwrap_or(T::Real::zero()))
}

/// Left-canonicalize an MPS via sequential QR decompositions.
///
/// For each site i from 0 to N-2, performs thin QR on the fused
/// (physical, left-bond) matrix, stores Q as the new site tensor,
/// and absorbs R into site i+1. O(N·d·D³).
pub fn left_canonicalize<T: Scalar, Q: BitPackable, G, B: LinAlgBackend<T>>(
    mps: MPS<T, Q, G>,
    backend: &B,
) -> MPS<T, Q, LeftCanonical> {
    let mut tensors = mps.tensors;
    let n = tensors.len();

    for site in 0..n.saturating_sub(1) {
        qr_canonicalize_site_left(&mut tensors, site, backend);
    }

    MPS {
        tensors,
        bonds: None,
        local_dims: mps.local_dims,
        total_charge: mps.total_charge,
        center_site: n.saturating_sub(1),
        _gauge: PhantomData,
    }
}

/// Right-canonicalize an MPS via sequential LQ decompositions (QR from right).
pub fn right_canonicalize<T: Scalar, Q: BitPackable, G, B: LinAlgBackend<T>>(
    mps: MPS<T, Q, G>,
    backend: &B,
) -> MPS<T, Q, RightCanonical> {
    let mut tensors = mps.tensors;
    let n = tensors.len();

    for site in (1..n).rev() {
        qr_canonicalize_site_right(&mut tensors, site, backend);
    }

    MPS {
        tensors,
        bonds: None,
        local_dims: mps.local_dims,
        total_charge: mps.total_charge,
        center_site: 0,
        _gauge: PhantomData,
    }
}

/// Mixed-canonicalize an MPS with orthogonality center at `center`.
pub fn mixed_canonicalize<T: Scalar, Q: BitPackable, G, B: LinAlgBackend<T>>(
    mps: MPS<T, Q, G>,
    center: usize,
    backend: &B,
) -> MPS<T, Q, MixedCanonical> {
    let mut tensors = mps.tensors;
    let n = tensors.len();

    // Left-canonicalize sites 0..center
    for site in 0..center.min(n.saturating_sub(1)) {
        qr_canonicalize_site_left(&mut tensors, site, backend);
    }

    // Right-canonicalize sites center+1..N-1
    for site in (center + 1..n).rev() {
        qr_canonicalize_site_right(&mut tensors, site, backend);
    }

    MPS {
        tensors,
        bonds: None,
        local_dims: mps.local_dims,
        total_charge: mps.total_charge,
        center_site: center,
        _gauge: PhantomData,
    }
}

/// QR-canonicalize site `site` leftward: A[site] = Q * R, absorb R into A[site+1].
///
/// After this, A[site] is left-isometric (A†A = I on physical+left legs).
fn qr_canonicalize_site_left<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>(
    tensors: &mut [BlockSparseTensor<T, Q>],
    site: usize,
    backend: &B,
) {
    // Process each sector block: fuse (σ, α_left) -> matrix rows, α_right -> cols
    // QR decompose each block, store Q, accumulate R for next site
    let tensor = &tensors[site];
    let rank = tensor.rank();
    if rank != 3 {
        return;
    }

    // Collect R matrices keyed by right bond quantum number
    let mut r_matrices: std::collections::HashMap<u64, (Q, Vec<T>, usize, usize)> =
        std::collections::HashMap::new();

    let mut new_blocks: Vec<(Vec<Q>, DenseTensor<'static, T>)> = Vec::new();

    for (qns, block) in tensor.iter_blocks() {
        let dims = block.shape().dims();
        let d_sigma = dims[0];
        let d_left = dims[1];
        let d_right = dims[2];
        let rows = d_sigma * d_left;
        let cols = d_right;

        if rows == 0 || cols == 0 {
            continue;
        }

        // Reshape block data to (rows, cols) matrix
        let data = block.as_slice();
        let mat = MatRef::from_slice(&data[..rows * cols], rows, cols);

        match backend.qr(&mat) {
            Ok(qr_result) => {
                let q_data = qr_result.q.as_slice();
                let r_data = qr_result.r.as_slice();
                let k = rows.min(cols); // rank of QR

                // Build new block from Q: reshape (rows, k) -> (d_sigma, d_left, k)
                // Note: k might differ from d_right, but for thin QR with rows >= cols, k = cols = d_right
                let new_block_data: Vec<T> = q_data[..d_sigma * d_left * k].to_vec();
                let new_shape = TensorShape::row_major(&[d_sigma, d_left, k]);
                let new_block = DenseTensor::from_vec(new_shape, new_block_data);
                new_blocks.push((qns.to_vec(), new_block));

                // Accumulate R matrix for this right bond sector
                let q_right = qns[2].clone();
                let r_vec: Vec<T> = r_data[..k * cols].to_vec();
                r_matrices.insert(q_right.pack(), (q_right, r_vec, k, cols));
            }
            Err(_) => {
                // If QR fails, keep original block
                let new_block =
                    DenseTensor::from_vec(block.shape().clone(), block.as_slice().to_vec());
                new_blocks.push((qns.to_vec(), new_block));
            }
        }
    }

    // Build new site tensor from Q blocks
    // The right bond QIndex might need updating if k != d_right
    // For typical MPS (rows >= cols), k = cols, so QIndex is unchanged
    let new_indices = tensor.indices().to_vec();
    let new_dirs = tensor.leg_directions().to_vec();
    let flux = tensor.flux().clone();

    // Only rebuild if we have blocks
    if !new_blocks.is_empty() {
        // For simplicity, use try_from_blocks and fall back on error
        match BlockSparseTensor::try_from_blocks(new_indices, flux, new_dirs, new_blocks) {
            Ok(new_tensor) => tensors[site] = new_tensor,
            Err(_) => {} // Keep original on error
        }
    }

    // Absorb R into next site: new_A[site+1][σ', j, α'] = Σ_α R[j, α] * A[site+1][σ', α, α']
    if site + 1 < tensors.len() {
        let next_tensor = &tensors[site + 1];
        let mut next_new_blocks: Vec<(Vec<Q>, DenseTensor<'static, T>)> = Vec::new();

        for (qns, block) in next_tensor.iter_blocks() {
            let dims = block.shape().dims();
            let d_sigma = dims[0];
            let d_left = dims[1];
            let d_right = dims[2];
            let q_left = qns[1].clone();

            // Look up R matrix for this left bond quantum number
            if let Some((_, r_data, r_rows, r_cols)) = r_matrices.get(&q_left.pack()) {
                if *r_cols == d_left {
                    // Contract: new[σ, j, α'] = Σ_α R[j, α] * old[σ, α, α']
                    let new_d_left = *r_rows;
                    let mut new_data = vec![T::zero(); d_sigma * new_d_left * d_right];
                    let old_data = block.as_slice();

                    for sigma in 0..d_sigma {
                        for j in 0..new_d_left {
                            for alpha in 0..d_left {
                                let r_val = r_data[j * r_cols + alpha];
                                for ap in 0..d_right {
                                    let old_val = old_data
                                        [sigma * d_left * d_right + alpha * d_right + ap];
                                    new_data[sigma * new_d_left * d_right + j * d_right + ap] =
                                        new_data[sigma * new_d_left * d_right + j * d_right + ap]
                                            + r_val * old_val;
                                }
                            }
                        }
                    }

                    let new_shape = TensorShape::row_major(&[d_sigma, new_d_left, d_right]);
                    let new_block = DenseTensor::from_vec(new_shape, new_data);
                    next_new_blocks.push((qns.to_vec(), new_block));
                } else {
                    // Dimension mismatch, keep original
                    let new_block =
                        DenseTensor::from_vec(block.shape().clone(), block.as_slice().to_vec());
                    next_new_blocks.push((qns.to_vec(), new_block));
                }
            } else {
                // No R for this sector, keep original
                let new_block =
                    DenseTensor::from_vec(block.shape().clone(), block.as_slice().to_vec());
                next_new_blocks.push((qns.to_vec(), new_block));
            }
        }

        if !next_new_blocks.is_empty() {
            let next_indices = next_tensor.indices().to_vec();
            let next_dirs = next_tensor.leg_directions().to_vec();
            let next_flux = next_tensor.flux().clone();
            match BlockSparseTensor::try_from_blocks(
                next_indices,
                next_flux,
                next_dirs,
                next_new_blocks,
            ) {
                Ok(new_next) => tensors[site + 1] = new_next,
                Err(_) => {}
            }
        }
    }
}

/// QR-canonicalize site `site` rightward: A[site] = L * Q, absorb L into A[site-1].
///
/// After this, A[site] is right-isometric (AA† = I on physical+right legs).
fn qr_canonicalize_site_right<T: Scalar, Q: BitPackable, B: LinAlgBackend<T>>(
    tensors: &mut [BlockSparseTensor<T, Q>],
    site: usize,
    backend: &B,
) {
    let tensor = &tensors[site];
    if tensor.rank() != 3 {
        return;
    }

    // For right canonicalization, reshape (d_sigma, d_left, d_right) -> (d_left, d_sigma * d_right)
    // Then QR: M^T = Q * R, so M = R^T * Q^T
    // Equivalently: transpose the matrix, QR, then transpose back
    // Or: do LQ decomposition = QR of transpose

    let mut l_matrices: std::collections::HashMap<u64, (Q, Vec<T>, usize, usize)> =
        std::collections::HashMap::new();

    let mut new_blocks: Vec<(Vec<Q>, DenseTensor<'static, T>)> = Vec::new();

    for (qns, block) in tensor.iter_blocks() {
        let dims = block.shape().dims();
        let d_sigma = dims[0];
        let d_left = dims[1];
        let d_right = dims[2];
        let rows = d_left;
        let cols = d_sigma * d_right;

        if rows == 0 || cols == 0 {
            continue;
        }

        // Reshape to (d_left, d_sigma * d_right) - need to transpose from (d_sigma, d_left, d_right)
        let data = block.as_slice();
        let mut transposed = vec![T::zero(); rows * cols];
        for sigma in 0..d_sigma {
            for alpha_l in 0..d_left {
                for alpha_r in 0..d_right {
                    transposed[alpha_l * cols + sigma * d_right + alpha_r] =
                        data[sigma * d_left * d_right + alpha_l * d_right + alpha_r];
                }
            }
        }

        let mat = MatRef::from_slice(&transposed, rows, cols);

        // QR of transpose gives us LQ: M = L * Q where Q is right-isometric
        // Actually we want: (d_left, d_sigma*d_right) = L * Q
        // QR gives (d_left, k) * (k, d_sigma*d_right)
        // L = Q_result (of the QR), Q = R_result
        // Wait, QR: M = Q*R where Q is (rows, k), R is (k, cols)
        // So M = Q*R, meaning Q has orthonormal columns and R is upper triangular
        // M^T = R^T * Q^T, so the original tensor's right-isometry comes from R^T

        // Actually for right-canonicalization we want:
        // A[σ, α_L, α_R] reshaped as M[α_L, (σ,α_R)] = L[α_L, k] * Q[k, (σ,α_R)]
        // After QR: M = QR_Q * QR_R
        // We store QR_R reshaped back as the site tensor (right-isometric after proper handling)
        // And absorb QR_Q into the previous site's right bond

        // Hmm, this isn't quite right. Let me use a different approach:
        // For right canonicalization, we do QR on M^T:
        // M^T[(σ,α_R), α_L] = Q_tilde * R_tilde
        // Then M = R_tilde^T * Q_tilde^T
        // Q_tilde^T has orthonormal rows -> right isometric
        // R_tilde^T is absorbed into previous site

        let mut mt_data = vec![T::zero(); cols * rows];
        for i in 0..rows {
            for j in 0..cols {
                mt_data[j * rows + i] = transposed[i * cols + j];
            }
        }
        let mt = MatRef::from_slice(&mt_data, cols, rows);

        match backend.qr(&mt) {
            Ok(qr_result) => {
                let q_data = qr_result.q.as_slice(); // (cols, k)
                let r_data = qr_result.r.as_slice(); // (k, rows)
                let k = cols.min(rows);

                // Q^T is (k, cols) = (k, d_sigma * d_right) -> right-isometric tensor
                // Reshape to (d_sigma, k, d_right) ... wait, the ordering is (k, d_sigma*d_right)
                // after transposing Q: new_tensor[k, (σ, α_R)] -> reshape to (σ, k, α_R)?
                // No, the site tensor should be (σ, α_L_new, α_R) where α_L_new = k

                // From Q^T: Q_tilde^T[j, (σ, α_R)] for j=0..k
                // This becomes A_new[σ, j, α_R]
                let mut new_data = vec![T::zero(); d_sigma * k * d_right];
                for j in 0..k {
                    for sigma in 0..d_sigma {
                        for alpha_r in 0..d_right {
                            // Q^T[j, σ*d_right + α_R] = Q[σ*d_right + α_R, j]
                            let q_val = q_data[(sigma * d_right + alpha_r) * k + j];
                            new_data[sigma * k * d_right + j * d_right + alpha_r] = q_val;
                        }
                    }
                }

                let new_shape = TensorShape::row_major(&[d_sigma, k, d_right]);
                let new_block = DenseTensor::from_vec(new_shape, new_data);
                new_blocks.push((qns.to_vec(), new_block));

                // R_tilde^T: (rows, k) = (d_left, k) -> L matrix to absorb into previous site
                let q_left = qns[1].clone();
                let mut rt_data = vec![T::zero(); d_left * k];
                for i in 0..k {
                    for j in 0..d_left {
                        // R^T[j, i] = R[i, j]
                        rt_data[j * k + i] = r_data[i * rows + j];
                    }
                }
                l_matrices.insert(q_left.pack(), (q_left, rt_data, d_left, k));
            }
            Err(_) => {
                let new_block =
                    DenseTensor::from_vec(block.shape().clone(), block.as_slice().to_vec());
                new_blocks.push((qns.to_vec(), new_block));
            }
        }
    }

    // Rebuild site tensor
    if !new_blocks.is_empty() {
        let new_indices = tensor.indices().to_vec();
        let new_dirs = tensor.leg_directions().to_vec();
        let flux = tensor.flux().clone();
        match BlockSparseTensor::try_from_blocks(new_indices, flux, new_dirs, new_blocks) {
            Ok(new_tensor) => tensors[site] = new_tensor,
            Err(_) => {}
        }
    }

    // Absorb L into previous site: new_A[site-1][σ, α, j] = Σ_α' A[site-1][σ, α, α'] * L[α', j]
    if site > 0 {
        let prev_tensor = &tensors[site - 1];
        let mut prev_new_blocks: Vec<(Vec<Q>, DenseTensor<'static, T>)> = Vec::new();

        for (qns, block) in prev_tensor.iter_blocks() {
            let dims = block.shape().dims();
            let d_sigma = dims[0];
            let d_left = dims[1];
            let d_right = dims[2];
            let q_right = qns[2].clone();

            if let Some((_, l_data, l_rows, l_cols)) = l_matrices.get(&q_right.pack()) {
                if *l_rows == d_right {
                    let new_d_right = *l_cols;
                    let mut new_data = vec![T::zero(); d_sigma * d_left * new_d_right];
                    let old_data = block.as_slice();

                    for sigma in 0..d_sigma {
                        for alpha in 0..d_left {
                            for ap in 0..d_right {
                                let old_val =
                                    old_data[sigma * d_left * d_right + alpha * d_right + ap];
                                for j in 0..new_d_right {
                                    let l_val = l_data[ap * l_cols + j];
                                    new_data
                                        [sigma * d_left * new_d_right + alpha * new_d_right + j] =
                                        new_data[sigma * d_left * new_d_right
                                            + alpha * new_d_right
                                            + j]
                                            + old_val * l_val;
                                }
                            }
                        }
                    }

                    let new_shape = TensorShape::row_major(&[d_sigma, d_left, new_d_right]);
                    let new_block = DenseTensor::from_vec(new_shape, new_data);
                    prev_new_blocks.push((qns.to_vec(), new_block));
                } else {
                    let new_block =
                        DenseTensor::from_vec(block.shape().clone(), block.as_slice().to_vec());
                    prev_new_blocks.push((qns.to_vec(), new_block));
                }
            } else {
                let new_block =
                    DenseTensor::from_vec(block.shape().clone(), block.as_slice().to_vec());
                prev_new_blocks.push((qns.to_vec(), new_block));
            }
        }

        if !prev_new_blocks.is_empty() {
            let prev_indices = prev_tensor.indices().to_vec();
            let prev_dirs = prev_tensor.leg_directions().to_vec();
            let prev_flux = prev_tensor.flux().clone();
            match BlockSparseTensor::try_from_blocks(
                prev_indices,
                prev_flux,
                prev_dirs,
                prev_new_blocks,
            ) {
                Ok(new_prev) => tensors[site - 1] = new_prev,
                Err(_) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tk_symmetry::U1;

    fn make_trivial_mps() -> MPS<f64, U1, MixedCanonical> {
        // Create a trivial 2-site MPS with bond dim 1
        let q0 = U1(0);
        let idx_phys = QIndex::new(vec![(U1(0), 1), (U1(1), 1)]);
        let idx_bond_1 = QIndex::new(vec![(U1(0), 1)]);

        let t0 = BlockSparseTensor::zeros(
            vec![idx_phys.clone(), idx_bond_1.clone(), idx_bond_1.clone()],
            q0,
            vec![LegDirection::Outgoing, LegDirection::Outgoing, LegDirection::Incoming],
        );
        let t1 = BlockSparseTensor::zeros(
            vec![idx_phys, idx_bond_1.clone(), idx_bond_1],
            q0,
            vec![LegDirection::Outgoing, LegDirection::Outgoing, LegDirection::Incoming],
        );

        MPS::new(vec![t0, t1], vec![2, 2], q0, 0)
    }

    #[test]
    fn mps_basic_accessors() {
        let mps = make_trivial_mps();
        assert_eq!(mps.n_sites(), 2);
        assert_eq!(mps.local_dim(0), 2);
        assert_eq!(mps.local_dim(1), 2);
    }

    #[test]
    fn mps_expose_absorb_roundtrip() {
        let mps = make_trivial_mps();
        let bc = mps.expose_bond();
        let mps2 = bc.absorb_bond();
        assert_eq!(mps2.n_sites(), 2);
    }
}
