//! Zero-copy NumPy array construction utilities.
//!
//! The `into_pyarray` / `from_vec_bound` pattern clones the Vec<f64> and
//! transfers buffer ownership to CPython's reference-counting memory manager.
//! This is acceptable for spectral function arrays (typically 2000–10000 points,
//! i.e., 16–80 KB per clone).

// This module exists as a placeholder for future optimization (pinned-view
// alternatives) if the clone-on-getter strategy becomes a bottleneck.
// See spec §12.3 for the trade-off analysis.
