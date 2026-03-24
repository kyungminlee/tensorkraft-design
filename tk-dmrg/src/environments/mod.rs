//! Environment block caching and incremental updates.
//!
//! During DMRG sweeps, left and right environment blocks are cached
//! and updated incrementally as the orthogonality center moves.

use num_traits::{One, Zero};
use tk_core::Scalar;
use tk_linalg::LinAlgBackend;
use tk_symmetry::{BitPackable, BlockSparseTensor, QIndex};

use crate::error::{DmrgError, DmrgResult};
use crate::mpo::MPO;
use crate::mps::MPS;

/// A single environment block (left or right).
///
/// Stored as a dense rank-3 tensor indexed by total bond dimensions:
/// (D_bra, w_mpo, D_ket).
///
/// For block-sparse MPS/MPO, the total dimensions are the sum across all
/// symmetry sectors, and the dense storage includes zero blocks for
/// sector combinations that violate the flux rule. This is correct but
/// not optimal; a production implementation would use block-sparse storage.
pub struct Environment<T: Scalar, Q: BitPackable> {
    /// Dense data of shape (d_bra, d_w, d_ket) in row-major order.
    data: Vec<T>,
    /// Dimensions: (bra_bond, mpo_bond, ket_bond).
    dims: (usize, usize, usize),
    /// QIndex for bra bond leg.
    bra_idx: QIndex<Q>,
    /// QIndex for MPO bond leg.
    mpo_idx: QIndex<Q>,
    /// QIndex for ket bond leg.
    ket_idx: QIndex<Q>,
    /// Site index this environment is built up to (exclusive for left, inclusive from right).
    up_to: usize,
}

impl<T: Scalar, Q: BitPackable> Environment<T, Q> {
    /// Create a left boundary environment (trivial 1×1×1 tensor with value 1).
    pub fn left_boundary(flux: Q) -> Self {
        let idx = QIndex::new(vec![(flux, 1)]);
        Environment {
            data: vec![T::one()],
            dims: (1, 1, 1),
            bra_idx: idx.clone(),
            mpo_idx: idx.clone(),
            ket_idx: idx,
            up_to: 0,
        }
    }

    /// Create a right boundary environment (trivial 1×1×1 tensor with value 1).
    pub fn right_boundary(n_sites: usize, flux: Q) -> Self {
        let idx = QIndex::new(vec![(flux, 1)]);
        Environment {
            data: vec![T::one()],
            dims: (1, 1, 1),
            bra_idx: idx.clone(),
            mpo_idx: idx.clone(),
            ket_idx: idx,
            up_to: n_sites,
        }
    }

    /// Site index this environment is built up to (exclusive).
    pub fn up_to(&self) -> usize {
        self.up_to
    }

    /// Access the environment tensor (for backward compatibility).
    /// Returns a reference to a dummy BlockSparseTensor.
    /// Use `data()` and `dims()` for the actual dense data.
    pub fn tensor(&self) -> &BlockSparseTensor<T, Q> {
        // This method exists for API compatibility but shouldn't be used
        // in the new implementation. Environment data is stored densely.
        unimplemented!("Use data() and dims() instead of tensor()")
    }

    /// Access the dense environment data.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Environment dimensions (bra_bond, mpo_bond, ket_bond).
    pub fn dims(&self) -> (usize, usize, usize) {
        self.dims
    }

    /// Element access: env[a_bra, w, a_ket].
    #[inline]
    pub fn get(&self, a_bra: usize, w: usize, a_ket: usize) -> T {
        let (_, dw, dk) = self.dims;
        self.data[a_bra * dw * dk + w * dk + a_ket]
    }

    /// Mutable element access.
    #[inline]
    pub fn get_mut(&mut self, a_bra: usize, w: usize, a_ket: usize) -> &mut T {
        let (_, dw, dk) = self.dims;
        &mut self.data[a_bra * dw * dk + w * dk + a_ket]
    }
}

/// Cached left and right environment blocks for a DMRG sweep.
pub struct Environments<T: Scalar, Q: BitPackable> {
    left_envs: Vec<Environment<T, Q>>,
    right_envs: Vec<Environment<T, Q>>,
    n_sites: usize,
}

impl<T: Scalar, Q: BitPackable> Environments<T, Q> {
    /// Build all environment blocks from scratch.
    ///
    /// Builds left boundary + all right environments by sweeping right-to-left.
    /// O(N·d·D²·w) total.
    pub fn build_from_scratch<G, B: LinAlgBackend<T>>(
        mps: &MPS<T, Q, G>,
        mpo: &MPO<T, Q>,
        _backend: &B,
    ) -> DmrgResult<Self> {
        let n = mps.n_sites();
        if n != mpo.n_sites() {
            return Err(DmrgError::DimensionMismatch {
                mps_sites: n,
                mpo_sites: mpo.n_sites(),
            });
        }

        let flux = mps.total_charge().clone();

        // Start with boundary environments
        let left_envs = vec![Environment::left_boundary(flux.clone())];
        let mut right_envs = vec![Environment::right_boundary(n, flux)];

        // Build right environments from right to left: sites N-1 down to 1
        for site in (1..n).rev() {
            let new_env = contract_right_environment(
                right_envs.last().unwrap(),
                mps.site_tensor(site),
                mpo.site_tensor(site),
                site,
            );
            right_envs.push(new_env);
        }

        // Reverse so right_envs[i] covers sites i+1..N
        right_envs.reverse();

        Ok(Environments {
            left_envs,
            right_envs,
            n_sites: n,
        })
    }

    /// Access the left environment for a given site.
    /// Returns the environment covering sites 0..site.
    pub fn left(&self, site: usize) -> &Environment<T, Q> {
        if site < self.left_envs.len() {
            &self.left_envs[site]
        } else {
            self.left_envs.last().unwrap()
        }
    }

    /// Access the right environment for a given site.
    /// Returns the environment covering sites site+1..N.
    pub fn right(&self, site: usize) -> &Environment<T, Q> {
        let idx = (site + 1).min(self.right_envs.len().saturating_sub(1));
        &self.right_envs[idx]
    }

    /// Grow the left environment by one site.
    ///
    /// Contracts L[α_bra, w, α_ket] with A[σ, α_ket, α'_ket],
    /// W[σ_in, σ_out, w, w'], and conj(A[σ_out, α_bra, α'_bra]).
    pub fn grow_left<G, B: LinAlgBackend<T>>(
        &mut self,
        site: usize,
        mps: &MPS<T, Q, G>,
        mpo: &MPO<T, Q>,
        _backend: &B,
    ) -> DmrgResult<()> {
        let env_l = self.left_envs.last().unwrap();
        let new_env = contract_left_environment(
            env_l,
            mps.site_tensor(site),
            mpo.site_tensor(site),
            site + 1,
        );
        self.left_envs.push(new_env);
        Ok(())
    }

    /// Grow the right environment by one site.
    pub fn grow_right<G, B: LinAlgBackend<T>>(
        &mut self,
        site: usize,
        mps: &MPS<T, Q, G>,
        mpo: &MPO<T, Q>,
        _backend: &B,
    ) -> DmrgResult<()> {
        let env_r = self.right_envs.last().unwrap();
        let new_env = contract_right_environment(
            env_r,
            mps.site_tensor(site),
            mpo.site_tensor(site),
            site,
        );
        self.right_envs.push(new_env);
        Ok(())
    }

    /// Number of sites.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }
}

/// Contract a left environment with a site's MPS and MPO tensors.
///
/// L'[α'_bra, w', α'_ket] = Σ_{α_bra, w, α_ket, σ_in, σ_out}
///   conj(A[σ_out, α_bra, α'_bra]) * L[α_bra, w, α_ket] * W[σ_in, σ_out, w, w'] * A[σ_in, α_ket, α'_ket]
pub fn contract_left_environment<T: Scalar, Q: BitPackable>(
    env: &Environment<T, Q>,
    mps_tensor: &BlockSparseTensor<T, Q>,
    mpo_tensor: &BlockSparseTensor<T, Q>,
    new_up_to: usize,
) -> Environment<T, Q> {
    let (d_bra, d_w, d_ket) = env.dims;

    // Determine new dimensions from MPS/MPO tensor right bond indices
    // MPS: (σ, α_left, α_right) -> new dims from α_right
    // MPO: (σ_in, σ_out, w_left, w_right) -> new dims from w_right
    let new_d_bra = mps_tensor.indices()[2].total_dim();
    let new_d_w = mpo_tensor.indices()[3].total_dim();
    let new_d_ket = mps_tensor.indices()[2].total_dim();

    let mut new_data = vec![T::zero(); new_d_bra * new_d_w * new_d_ket];

    // Iterate over all non-zero MPS blocks (ket)
    for (ket_qns, ket_block) in mps_tensor.iter_blocks() {
        let ket_data = ket_block.as_slice();
        let d_sigma_ket = ket_block.shape().dims()[0];
        let d_left_ket = ket_block.shape().dims()[1];
        let d_right_ket = ket_block.shape().dims()[2];
        let q_sigma_ket = &ket_qns[0];
        let q_left_ket = &ket_qns[1];
        let q_right_ket = &ket_qns[2];

        let ket_left_offset = mps_tensor.indices()[1].offset_of(q_left_ket).unwrap_or(0);
        let ket_right_offset = mps_tensor.indices()[2].offset_of(q_right_ket).unwrap_or(0);

        // Iterate over all non-zero MPS blocks (bra = conjugate)
        for (bra_qns, bra_block) in mps_tensor.iter_blocks() {
            let bra_data = bra_block.as_slice();
            let d_sigma_bra = bra_block.shape().dims()[0];
            let d_left_bra = bra_block.shape().dims()[1];
            let d_right_bra = bra_block.shape().dims()[2];
            let q_sigma_bra = &bra_qns[0];
            let q_left_bra = &bra_qns[1];
            let q_right_bra = &bra_qns[2];

            let bra_left_offset = mps_tensor.indices()[1].offset_of(q_left_bra).unwrap_or(0);
            let bra_right_offset = mps_tensor.indices()[2].offset_of(q_right_bra).unwrap_or(0);

            // Iterate over MPO blocks
            for (mpo_qns, mpo_block) in mpo_tensor.iter_blocks() {
                let mpo_data = mpo_block.as_slice();
                let d_sigma_in = mpo_block.shape().dims()[0];
                let d_sigma_out = mpo_block.shape().dims()[1];
                let d_w_left = mpo_block.shape().dims()[2];
                let d_w_right = mpo_block.shape().dims()[3];
                let q_sigma_in = &mpo_qns[0];
                let q_sigma_out = &mpo_qns[1];
                let q_w_left = &mpo_qns[2];
                let q_w_right = &mpo_qns[3];

                // Physical indices must match: σ_in matches ket σ, σ_out matches bra σ
                if q_sigma_in != q_sigma_ket || q_sigma_out != q_sigma_bra {
                    continue;
                }
                if d_sigma_in != d_sigma_ket || d_sigma_out != d_sigma_bra {
                    continue;
                }

                let w_left_offset = mpo_tensor.indices()[2].offset_of(q_w_left).unwrap_or(0);
                let w_right_offset = mpo_tensor.indices()[3].offset_of(q_w_right).unwrap_or(0);

                // Contract: L'[α'_bra, w', α'_ket] +=
                //   Σ_{σ_in, σ_out, α_bra, w, α_ket}
                //   conj(A_bra[σ_out, α_bra, α'_bra]) * L[α_bra, w, α_ket]
                //   * W[σ_in, σ_out, w, w'] * A_ket[σ_in, α_ket, α'_ket]
                for s_in in 0..d_sigma_in {
                    for s_out in 0..d_sigma_out {
                        for a_bra in 0..d_left_bra {
                            for w in 0..d_w_left {
                                for a_ket in 0..d_left_ket {
                                    let env_val = env.get(
                                        bra_left_offset + a_bra,
                                        w_left_offset + w,
                                        ket_left_offset + a_ket,
                                    );
                                    // Skip exact zeros for performance (optional)
                                    let mpo_val = mpo_data[s_in * d_sigma_out * d_w_left * d_w_right
                                        + s_out * d_w_left * d_w_right
                                        + w * d_w_right..][..d_w_right]
                                        .to_vec();

                                    for wp in 0..d_w_right {
                                        let w_val = mpo_val[wp];
                                        // Skip exact zeros for performance (optional)
                                        let contrib = env_val * w_val;

                                        for ap_ket in 0..d_right_ket {
                                            let ket_val = ket_data[s_in * d_left_ket * d_right_ket
                                                + a_ket * d_right_ket
                                                + ap_ket];
                                            // Skip exact zeros for performance (optional)
                                            let ket_contrib = contrib * ket_val;

                                            for ap_bra in 0..d_right_bra {
                                                let bra_val = bra_data[s_out * d_left_bra * d_right_bra
                                                    + a_bra * d_right_bra
                                                    + ap_bra];
                                                let conj_bra = bra_val.conj();

                                                let idx = (bra_right_offset + ap_bra)
                                                    * new_d_w
                                                    * new_d_ket
                                                    + (w_right_offset + wp) * new_d_ket
                                                    + (ket_right_offset + ap_ket);
                                                new_data[idx] = new_data[idx] + conj_bra * ket_contrib;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Environment {
        data: new_data,
        dims: (new_d_bra, new_d_w, new_d_ket),
        bra_idx: mps_tensor.indices()[2].clone(),
        mpo_idx: mpo_tensor.indices()[3].clone(),
        ket_idx: mps_tensor.indices()[2].clone(),
        up_to: new_up_to,
    }
}

/// Contract a right environment with a site's MPS and MPO tensors.
///
/// R'[α_bra, w, α_ket] = Σ_{α'_bra, w', α'_ket, σ_in, σ_out}
///   conj(A[σ_out, α_bra, α'_bra]) * R[α'_bra, w', α'_ket] * W[σ_in, σ_out, w, w'] * A[σ_in, α_ket, α'_ket]
pub fn contract_right_environment<T: Scalar, Q: BitPackable>(
    env: &Environment<T, Q>,
    mps_tensor: &BlockSparseTensor<T, Q>,
    mpo_tensor: &BlockSparseTensor<T, Q>,
    new_up_to: usize,
) -> Environment<T, Q> {
    let (d_bra, d_w, d_ket) = env.dims;

    // New dimensions from left bond indices
    let new_d_bra = mps_tensor.indices()[1].total_dim();
    let new_d_w = mpo_tensor.indices()[2].total_dim();
    let new_d_ket = mps_tensor.indices()[1].total_dim();

    let mut new_data = vec![T::zero(); new_d_bra * new_d_w * new_d_ket];

    for (ket_qns, ket_block) in mps_tensor.iter_blocks() {
        let ket_data = ket_block.as_slice();
        let d_sigma_ket = ket_block.shape().dims()[0];
        let d_left_ket = ket_block.shape().dims()[1];
        let d_right_ket = ket_block.shape().dims()[2];
        let q_sigma_ket = &ket_qns[0];
        let q_left_ket = &ket_qns[1];
        let q_right_ket = &ket_qns[2];

        let ket_left_offset = mps_tensor.indices()[1].offset_of(q_left_ket).unwrap_or(0);
        let ket_right_offset = mps_tensor.indices()[2].offset_of(q_right_ket).unwrap_or(0);

        for (bra_qns, bra_block) in mps_tensor.iter_blocks() {
            let bra_data = bra_block.as_slice();
            let d_sigma_bra = bra_block.shape().dims()[0];
            let d_left_bra = bra_block.shape().dims()[1];
            let d_right_bra = bra_block.shape().dims()[2];
            let q_sigma_bra = &bra_qns[0];
            let q_left_bra = &bra_qns[1];
            let q_right_bra = &bra_qns[2];

            let bra_left_offset = mps_tensor.indices()[1].offset_of(q_left_bra).unwrap_or(0);
            let bra_right_offset = mps_tensor.indices()[2].offset_of(q_right_bra).unwrap_or(0);

            for (mpo_qns, mpo_block) in mpo_tensor.iter_blocks() {
                let mpo_data = mpo_block.as_slice();
                let d_sigma_in = mpo_block.shape().dims()[0];
                let d_sigma_out = mpo_block.shape().dims()[1];
                let d_w_left = mpo_block.shape().dims()[2];
                let d_w_right = mpo_block.shape().dims()[3];
                let q_sigma_in = &mpo_qns[0];
                let q_sigma_out = &mpo_qns[1];
                let q_w_left = &mpo_qns[2];
                let q_w_right = &mpo_qns[3];

                if q_sigma_in != q_sigma_ket || q_sigma_out != q_sigma_bra {
                    continue;
                }
                if d_sigma_in != d_sigma_ket || d_sigma_out != d_sigma_bra {
                    continue;
                }

                let w_left_offset = mpo_tensor.indices()[2].offset_of(q_w_left).unwrap_or(0);
                let w_right_offset = mpo_tensor.indices()[3].offset_of(q_w_right).unwrap_or(0);

                // R'[α_bra, w, α_ket] +=
                //   Σ conj(A_bra[σ_out, α_bra, α'_bra]) * R[α'_bra, w', α'_ket]
                //   * W[σ_in, σ_out, w, w'] * A_ket[σ_in, α_ket, α'_ket]
                for s_in in 0..d_sigma_in {
                    for s_out in 0..d_sigma_out {
                        for ap_bra in 0..d_right_bra {
                            for wp in 0..d_w_right {
                                for ap_ket in 0..d_right_ket {
                                    let env_val = env.get(
                                        bra_right_offset + ap_bra,
                                        w_right_offset + wp,
                                        ket_right_offset + ap_ket,
                                    );
                                    // Skip exact zeros for performance (optional)

                                    for w in 0..d_w_left {
                                        let w_val = mpo_data[s_in * d_sigma_out * d_w_left * d_w_right
                                            + s_out * d_w_left * d_w_right
                                            + w * d_w_right
                                            + wp];
                                        // Skip exact zeros for performance (optional)
                                        let contrib = env_val * w_val;

                                        for a_ket in 0..d_left_ket {
                                            let ket_val = ket_data[s_in * d_left_ket * d_right_ket
                                                + a_ket * d_right_ket
                                                + ap_ket];
                                            // Skip exact zeros for performance (optional)
                                            let ket_contrib = contrib * ket_val;

                                            for a_bra in 0..d_left_bra {
                                                let bra_val = bra_data[s_out * d_left_bra * d_right_bra
                                                    + a_bra * d_right_bra
                                                    + ap_bra];
                                                let conj_bra = bra_val.conj();

                                                let idx = (bra_left_offset + a_bra)
                                                    * new_d_w
                                                    * new_d_ket
                                                    + (w_left_offset + w) * new_d_ket
                                                    + (ket_left_offset + a_ket);
                                                new_data[idx] = new_data[idx] + conj_bra * ket_contrib;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Environment {
        data: new_data,
        dims: (new_d_bra, new_d_w, new_d_ket),
        bra_idx: mps_tensor.indices()[1].clone(),
        mpo_idx: mpo_tensor.indices()[2].clone(),
        ket_idx: mps_tensor.indices()[1].clone(),
        up_to: new_up_to,
    }
}

/// Build the effective Hamiltonian matvec closure for a two-site DMRG update.
///
/// Returns `(matvec_closure, hilbert_dim)` where hilbert_dim = d_i · d_{i+1} · D_L · D_R.
///
/// The closure contracts L · W_i · W_{i+1} · R with the two-site vector.
/// Vector layout: row-major (σ_i, σ_{i+1}, α_L, α_R).
pub fn build_heff_two_site<T: Scalar, Q: BitPackable>(
    env_l: &Environment<T, Q>,
    env_r: &Environment<T, Q>,
    mpo: &MPO<T, Q>,
    sites: (usize, usize),
) -> DmrgResult<(Box<dyn Fn(&[T], &mut [T])>, usize)> {
    let (site_i, site_j) = sites;
    let mpo_i = mpo.site_tensor(site_i);
    let mpo_j = mpo.site_tensor(site_j);

    let d_i = mpo.local_dim(site_i);
    let d_j = mpo.local_dim(site_j);
    let (_, _, d_l) = env_l.dims;
    let (_, _, d_r) = env_r.dims;
    let dim = d_i * d_j * d_l * d_r;

    // Build the full H_eff matrix
    let mut heff = vec![T::zero(); dim * dim];

    for (mpo_i_qns, mpo_i_block) in mpo_i.iter_blocks() {
        let mi_data = mpo_i_block.as_slice();
        let mi_dims = mpo_i_block.shape().dims();
        let d_sin_i = mi_dims[0];
        let d_sout_i = mi_dims[1];
        let d_wl_i = mi_dims[2];
        let d_wm_i = mi_dims[3];

        let sin_i_off = mpo_i.indices()[0].offset_of(&mpo_i_qns[0]).unwrap_or(0);
        let sout_i_off = mpo_i.indices()[1].offset_of(&mpo_i_qns[1]).unwrap_or(0);
        let wl_off = mpo_i.indices()[2].offset_of(&mpo_i_qns[2]).unwrap_or(0);
        let wm_i_off = mpo_i.indices()[3].offset_of(&mpo_i_qns[3]).unwrap_or(0);

        for (mpo_j_qns, mpo_j_block) in mpo_j.iter_blocks() {
            let mj_data = mpo_j_block.as_slice();
            let mj_dims = mpo_j_block.shape().dims();
            let d_sin_j = mj_dims[0];
            let d_sout_j = mj_dims[1];
            let d_wm_j = mj_dims[2];
            let d_wr_j = mj_dims[3];

            // w_mid quantum numbers must match
            if mpo_i_qns[3] != mpo_j_qns[2] {
                continue;
            }

            let sin_j_off = mpo_j.indices()[0].offset_of(&mpo_j_qns[0]).unwrap_or(0);
            let sout_j_off = mpo_j.indices()[1].offset_of(&mpo_j_qns[1]).unwrap_or(0);
            let wr_off = mpo_j.indices()[3].offset_of(&mpo_j_qns[3]).unwrap_or(0);

            for s_in_i in 0..d_sin_i {
                for s_out_i in 0..d_sout_i {
                    for s_in_j in 0..d_sin_j {
                        for s_out_j in 0..d_sout_j {
                            for wl in 0..d_wl_i {
                                for wm in 0..d_wm_i.min(d_wm_j) {
                                    let wi_val = mi_data[s_in_i * d_sout_i * d_wl_i * d_wm_i
                                        + s_out_i * d_wl_i * d_wm_i
                                        + wl * d_wm_i
                                        + wm];

                                    for wr in 0..d_wr_j {
                                        let wj_val = mj_data[s_in_j * d_sout_j * d_wm_j * d_wr_j
                                            + s_out_j * d_wm_j * d_wr_j
                                            + wm * d_wr_j
                                            + wr];
                                        let w_contrib = wi_val * wj_val;

                                        for a_l in 0..d_l {
                                            for ap_l in 0..d_l {
                                                let l_val = env_l.get(ap_l, wl_off + wl, a_l);
                                                let lw = l_val * w_contrib;

                                                for a_r in 0..d_r {
                                                    for ap_r in 0..d_r {
                                                        let r_val = env_r.get(ap_r, wr_off + wr, a_r);

                                                        let in_idx = (sin_i_off + s_in_i) * d_j * d_l * d_r
                                                            + (sin_j_off + s_in_j) * d_l * d_r
                                                            + a_l * d_r + a_r;
                                                        let out_idx = (sout_i_off + s_out_i) * d_j * d_l * d_r
                                                            + (sout_j_off + s_out_j) * d_l * d_r
                                                            + ap_l * d_r + ap_r;

                                                        if in_idx < dim && out_idx < dim {
                                                            heff[out_idx * dim + in_idx] =
                                                                heff[out_idx * dim + in_idx] + lw * r_val;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let matvec = Box::new(move |x: &[T], y: &mut [T]| {
        for i in 0..dim {
            y[i] = T::zero();
            for j in 0..dim {
                y[i] = y[i] + heff[i * dim + j] * x[j];
            }
        }
    });

    Ok((matvec, dim))
}

/// Build the effective Hamiltonian matvec closure for a single-site DMRG update.
///
/// Vector layout: row-major (σ, α_L, α_R). dim = d · D_L · D_R.
pub fn build_heff_single_site<T: Scalar, Q: BitPackable>(
    env_l: &Environment<T, Q>,
    env_r: &Environment<T, Q>,
    mpo: &MPO<T, Q>,
    site: usize,
) -> DmrgResult<(Box<dyn Fn(&[T], &mut [T])>, usize)> {
    let mpo_s = mpo.site_tensor(site);
    let d = mpo.local_dim(site);
    let (_, _, d_l) = env_l.dims;
    let (_, _, d_r) = env_r.dims;
    let dim = d * d_l * d_r;

    let mut heff = vec![T::zero(); dim * dim];

    for (mpo_qns, mpo_block) in mpo_s.iter_blocks() {
        let m_data = mpo_block.as_slice();
        let m_dims = mpo_block.shape().dims();
        let d_sin = m_dims[0];
        let d_sout = m_dims[1];
        let d_wl = m_dims[2];
        let d_wr = m_dims[3];

        let sin_off = mpo_s.indices()[0].offset_of(&mpo_qns[0]).unwrap_or(0);
        let sout_off = mpo_s.indices()[1].offset_of(&mpo_qns[1]).unwrap_or(0);
        let wl_off = mpo_s.indices()[2].offset_of(&mpo_qns[2]).unwrap_or(0);
        let wr_off = mpo_s.indices()[3].offset_of(&mpo_qns[3]).unwrap_or(0);

        for s_in in 0..d_sin {
            for s_out in 0..d_sout {
                for wl in 0..d_wl {
                    for wr in 0..d_wr {
                        let w_val = m_data[s_in * d_sout * d_wl * d_wr
                            + s_out * d_wl * d_wr
                            + wl * d_wr + wr];

                        for a_l in 0..d_l {
                            for ap_l in 0..d_l {
                                let l_val = env_l.get(ap_l, wl_off + wl, a_l);
                                let lw = l_val * w_val;

                                for a_r in 0..d_r {
                                    for ap_r in 0..d_r {
                                        let r_val = env_r.get(ap_r, wr_off + wr, a_r);

                                        let in_idx = (sin_off + s_in) * d_l * d_r + a_l * d_r + a_r;
                                        let out_idx = (sout_off + s_out) * d_l * d_r + ap_l * d_r + ap_r;

                                        if in_idx < dim && out_idx < dim {
                                            heff[out_idx * dim + in_idx] =
                                                heff[out_idx * dim + in_idx] + lw * r_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let matvec = Box::new(move |x: &[T], y: &mut [T]| {
        for i in 0..dim {
            y[i] = T::zero();
            for j in 0..dim {
                y[i] = y[i] + heff[i * dim + j] * x[j];
            }
        }
    });

    Ok((matvec, dim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tk_symmetry::U1;

    #[test]
    fn environment_boundary_creation() {
        let env = Environment::<f64, U1>::left_boundary(U1(0));
        assert_eq!(env.up_to(), 0);
        assert_eq!(env.dims(), (1, 1, 1));
        assert_eq!(env.data(), &[1.0]);
    }

    #[test]
    fn environment_right_boundary() {
        let env = Environment::<f64, U1>::right_boundary(10, U1(0));
        assert_eq!(env.up_to(), 10);
        assert_eq!(env.dims(), (1, 1, 1));
        assert_eq!(env.data(), &[1.0]);
    }
}
