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
    Node((Option<V>, Siblings<V, A>)),
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

type IdNodeMap<V, A> = BTreeMap<u64, (VClock<A>, Atom<V, A>)>;

/// Set of siblings nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Siblings<V: Ord + Clone, A: Actor>(IdNodeMap<V, A>);

impl<V: Ord + Clone + Display, A: Actor + Display> Siblings<V, A> {
    /// Create a new and empty set of siblings nodes
    pub fn new() -> Self {
        Self(BTreeMap::default())
    }

    pub fn inner(&self) -> &IdNodeMap<V, A> {
        &self.0
    }

    pub fn inner_mut(&mut self) -> &mut IdNodeMap<V, A> {
        &mut self.0
    }

    /// Find the atom in the tree following the path of the given identifier and delete its value
    pub fn delete_id(&mut self, mut id: Identifier) {
        if id.len() > 1 {
            let cur_number = id.remove(0);
            match self.0.get_mut(&cur_number) {
                Some(&mut (_, Atom::Node((_, ref mut siblings)))) => {
                    // good, keep traversing the tree
                    siblings.delete_id(id);
                }
                None | Some(&mut (_, Atom::Leaf(_))) => {
                    // found a leaf already, then the id is not found in tree
                }
            }
        } else if !id.is_empty() {
            match self.0.get(&id.at(0)) {
                Some(&(ref c, Atom::Node((_, ref siblings)))) => {
                    // found it as a node, we need to clear the value from it
                    let new_atom = Atom::Node((None, siblings.clone()));
                    self.0.insert(id.at(0), (c.clone(), new_atom));
                }
                Some(&(_, Atom::Leaf(_))) => {
                    // found it as leaf, we remove the leaf from the tree
                    // TODO: we may need to keep it so we maintain the VClock for subsequent ops
                    self.0.remove(&id.at(0));
                }
                None => { /* not found */ }
            }
        }
    }
}
