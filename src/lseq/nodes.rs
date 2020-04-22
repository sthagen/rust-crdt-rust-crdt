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

/// It contains a single value with its associated vector clock.
/// The value is optional, and it's None when an operation deleted it,
/// keeping the vector clock information provided in the deletion operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomValue<V: Ord + Clone, A: Actor> {
    pub clock: VClock<A>,
    pub value: Option<V>,
}

impl<V: Ord + Clone + Display, A: Actor + Display> Display for AtomValue<V, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.value {
            Some(val) => write!(f, "{}", val),
            None => write!(f, "*Removed*"),
        }
    }
}

// TODO: we need to use something like the Actor (not u64) to order mini-nodes deterministically
// but at the moment we don't have info of which actor/site sent this request
pub type MiniNodes<V, A> = BTreeMap<u64, AtomValue<V, A>>;

/// Each node in the tree can optionall contain a value, or None if the value was deleted.
/// Each node can also contain (optionally) children.
/// A node could alternativelly container mini-nodes if there were
/// concurrent insertions with the same Identifier. Refer to the
/// TreeDoc paper for details about the need and use of mini-nodes in such scenarios
/// TODO: support mini-nodes to contain trees for insertions between their atoms,
/// current impl means concurrent inserts can occur, but no inserts will be possible
/// between them afterwards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Atom<V: Ord + Clone, A: Actor> {
    Node((AtomValue<V, A>, SiblingsNodes<V, A>)),
    MiniNodes(MiniNodes<V, A>),
}

impl<V: Ord + Clone + Display, A: Actor + Display> Display for Atom<V, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            Atom::Node((v, _)) => write!(f, "Node('{}')", v),
            Atom::MiniNodes(mini_nodes) => {
                let mut output = String::from("MiniNodes[ ");
                mini_nodes.iter().for_each(|(_, v)| {
                    output.push_str(&format!("'{}' ", v));
                });
                write!(f, "{}]", output)
            }
        }
    }
}

pub type SiblingsNodes<V, A> = BTreeMap<u64, Atom<V, A>>;
