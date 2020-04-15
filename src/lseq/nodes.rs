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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Atom<V: Ord + Clone, A: Actor> {
    Node((V, Siblings<V, A>)),
    Leaf(V),
}

impl<V: Ord + Clone + Display, A: Actor + Display> Display for Atom<V, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            Atom::Node((v, _)) => write!(f, "Node('{}')", v),
            Atom::Leaf(v) => write!(f, "Leaf('{}')", v),
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

    /// Find an atom in the tree using its identifier
    #[allow(dead_code)]
    pub fn find_atom(&self, id: &Identifier) -> Option<(VClock<A>, Atom<V, A>)> {
        let mut cur_atom = None;
        let mut cur_depth_nodes = &self.0;
        for i in 0..id.len() {
            match cur_depth_nodes.get(&id.at(i)) {
                Some(&(ref c, Atom::Node((ref v, ref siblings)))) => {
                    if i < id.len() - 1 {
                        // we found the intermediate node, keep going
                        cur_depth_nodes = &siblings.0;
                    } else {
                        // found it as a node
                        cur_atom = Some((c.clone(), Atom::Node((v.clone(), siblings.clone()))));
                    }
                }
                Some(&(ref c, Atom::Leaf(ref v))) if i == id.len() - 1 => {
                    // found it as a leaf
                    cur_atom = Some((c.clone(), Atom::Leaf(v.clone())));
                }
                None | Some(&(_, Atom::Leaf(_))) => {
                    // not found
                    cur_atom = None;
                    break;
                }
            }
        }
        cur_atom
    }
}
