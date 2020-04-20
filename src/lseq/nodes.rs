use crate::vclock::{Actor, VClock};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fmt::{self, Display},
};

/// Variable-size identifier
#[derive(Debug, Eq, Clone, Serialize, Deserialize)]
pub struct Identifier(Vec<u64>);

impl Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Ord for Identifier {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Identifier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Identifier {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Identifier {
    pub fn new(id: &[u64]) -> Self {
        Self(id.to_vec())
    }

    pub fn push(&mut self, i: u64) -> &Self {
        self.0.push(i);
        self
    }

    pub fn remove(&mut self, index: usize) -> u64 {
        self.0.remove(index)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn at(&self, index: usize) -> u64 {
        self.0[index]
    }
}

/// Each node in the tree can be a leaf or contain children
/// It optionally contains a value, or None if it was deleted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Atom<V: Ord + Clone, A: Actor> {
    Node((Option<V>, SiblingsNodes<V, A>)),
    Leaf(Option<V>),
}

impl<V: Ord + Clone + Display, A: Actor + Display> Display for Atom<V, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            Atom::Node((Some(v), _)) => write!(f, "Node('{}')", v),
            Atom::Node((None, _)) => write!(f, "Node()"),
            Atom::Leaf(Some(v)) => write!(f, "Leaf('{}')", v),
            Atom::Leaf(None) => write!(f, "Leaf()"),
        }
    }
}

pub type SiblingsNodes<V, A> = BTreeMap<u64, (VClock<A>, Atom<V, A>)>;
