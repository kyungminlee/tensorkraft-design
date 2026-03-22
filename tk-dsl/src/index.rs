//! Intelligent index system with unique IDs, prime levels, and direction.
//!
//! An `Index` wraps a `tk-contract::IndexId` with human-readable tags,
//! prime levels for bra/ket distinction, and optional direction for
//! symmetry-aware contractions.

use smallstr::SmallString;
use tk_contract::IndexId;

use crate::error::{DslError, DslResult};

/// Direction of quantum-number flow on an index leg.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexDirection {
    /// Quantum numbers flow in (ket).
    Incoming,
    /// Quantum numbers flow out (bra).
    Outgoing,
    /// Dense / no-symmetry path.
    None,
}

/// A named tensor index with unique identity, dimension, prime level, and direction.
///
/// Equality and hashing are based on `id` and `prime_level` only.
/// The `tag` is for human readability and does not participate in comparisons.
#[derive(Clone, Debug)]
pub struct Index {
    id: IndexId,
    tag: SmallString<[u8; 32]>,
    dim: usize,
    prime_level: u32,
    direction: IndexDirection,
}

impl Index {
    /// Create a new index with a unique ID.
    pub fn new(tag: impl Into<SmallString<[u8; 32]>>, dim: usize, direction: IndexDirection) -> Self {
        Index {
            id: IndexId::new_unique(),
            tag: tag.into(),
            dim,
            prime_level: 0,
            direction,
        }
    }

    /// The globally unique identifier.
    pub fn id(&self) -> IndexId {
        self.id
    }

    /// Human-readable label.
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// Number of basis states this index spans.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Prime level: 0 = ket, 1 = bra, 2+ = multi-time.
    pub fn prime_level(&self) -> u32 {
        self.prime_level
    }

    /// Quantum-number flow direction.
    pub fn direction(&self) -> IndexDirection {
        self.direction
    }

    /// Return a copy with prime level incremented by 1.
    pub fn prime(&self) -> Self {
        let mut copy = self.clone();
        copy.prime_level += 1;
        copy
    }

    /// Return a copy with prime level incremented by `n`.
    pub fn prime_n(&self, n: u32) -> Self {
        let mut copy = self.clone();
        copy.prime_level += n;
        copy
    }

    /// Return a copy with prime level reset to 0.
    pub fn unprime(&self) -> Self {
        let mut copy = self.clone();
        copy.prime_level = 0;
        copy
    }

    /// Check if two indices share the same underlying IndexId.
    pub fn same_id(&self, other: &Index) -> bool {
        self.id == other.id
    }

    /// Check if this index contracts with `other`:
    /// same IndexId and |prime_level difference| == 1.
    pub fn contracts_with(&self, other: &Index) -> bool {
        self.id == other.id
            && (self.prime_level as i64 - other.prime_level as i64).unsigned_abs() == 1
    }
}

impl PartialEq for Index {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.prime_level == other.prime_level
    }
}

impl Eq for Index {}

impl std::hash::Hash for Index {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.prime_level.hash(state);
    }
}

/// Registry for creating indices with unique tags.
pub struct IndexRegistry {
    entries: Vec<(SmallString<[u8; 32]>, Index)>,
}

impl IndexRegistry {
    pub fn new() -> Self {
        IndexRegistry {
            entries: Vec::new(),
        }
    }

    /// Register a new index tag. Returns an error if the tag is already registered.
    pub fn register(
        &mut self,
        tag: impl Into<SmallString<[u8; 32]>>,
        dim: usize,
        direction: IndexDirection,
    ) -> DslResult<Index> {
        let tag = tag.into();
        if self.entries.iter().any(|(t, _)| *t == tag) {
            return Err(DslError::DuplicateIndexTag {
                tag: tag.to_string(),
            });
        }
        let index = Index::new(tag.clone(), dim, direction);
        self.entries.push((tag, index.clone()));
        Ok(index)
    }

    /// Look up a previously registered index by tag.
    pub fn get(&self, tag: &str) -> Option<&Index> {
        self.entries
            .iter()
            .find(|(t, _)| t.as_str() == tag)
            .map(|(_, idx)| idx)
    }
}

impl Default for IndexRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_prime_increments() {
        let i = Index::new("s", 2, IndexDirection::None);
        assert_eq!(i.prime_level(), 0);
        let ip = i.prime();
        assert_eq!(ip.prime_level(), 1);
        let ipp = ip.prime();
        assert_eq!(ipp.prime_level(), 2);
    }

    #[test]
    fn index_contracts_with_prime() {
        let i = Index::new("s", 2, IndexDirection::None);
        let ip = i.prime();
        assert!(i.contracts_with(&ip));
        assert!(ip.contracts_with(&i));
        // Same prime level does not contract
        assert!(!i.contracts_with(&i));
        // Prime diff of 2 does not contract
        let ipp = ip.prime();
        assert!(!i.contracts_with(&ipp));
    }

    #[test]
    fn index_unprime_resets() {
        let i = Index::new("s", 2, IndexDirection::None);
        let ip = i.prime().prime();
        let unp = ip.unprime();
        assert_eq!(unp.prime_level(), 0);
        assert!(unp.same_id(&i));
    }

    #[test]
    fn index_registry_unique_ids() {
        let mut reg = IndexRegistry::new();
        let a = reg
            .register("site_a", 2, IndexDirection::None)
            .unwrap();
        let b = reg
            .register("site_b", 3, IndexDirection::None)
            .unwrap();
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn index_registry_duplicate_tag_error() {
        let mut reg = IndexRegistry::new();
        reg.register("s", 2, IndexDirection::None).unwrap();
        let result = reg.register("s", 3, IndexDirection::None);
        assert!(result.is_err());
    }

    #[test]
    fn index_equality_ignores_tag() {
        // Two indices with same id and prime level are equal regardless of tag.
        let a = Index::new("a", 2, IndexDirection::None);
        let mut b = a.clone();
        b.tag = SmallString::from("different_tag");
        assert_eq!(a, b);
    }
}
