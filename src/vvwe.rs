// Causality barrier
// Keeps for each known peer, keeps track of the latest clock seen
// And a set of messages that are from the future
// and outputs the full-in-order sequence of messages
//
//

#![allow(missing_docs)]

use std::collections::*;
use serde::{Deserialize, self, Serialize};

/// Version Vector with Exceptions
#[derive(Debug, Serialize, Deserialize)]
pub struct CausalityBarrier<A: Actor, T: CausalOp<A>> {
    peers: HashMap<A, VectorEntry>,
    local_id: A,
    pub buffer: HashMap<Dot<A>, T>,
}

type LogTime = u64;

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct VectorEntry {
    // The version of the next message we'd like to see
    max_version: LogTime,
    exceptions: HashSet<LogTime>,
}


impl VectorEntry {
    pub fn new() -> Self {
        VectorEntry::default()
    }

    pub fn increment(&mut self, clk: LogTime) {
        // We've just found an exception
        if clk < self.max_version {
            self.exceptions.take(&clk);
        } else if clk == self.max_version {
            self.max_version = self.max_version + 1;
        } else {
            let mut x = self.max_version + 1;
            while x < clk {
                self.exceptions.insert(x);
                x = x + 1;
            }
        }
    }

    pub fn is_ready(&self, clk: &LogTime) -> bool {
        *clk < self.max_version && !self.exceptions.contains(clk)
    }

    /// Calculate the difference between a remote VectorEntry and ours.
    /// Specifically, we want the set of operations we've seen that the remote hasn't
    pub fn diff_from(&self, other: &Self) -> HashSet<LogTime> {
        // 1. Find (new) operations that we've seen locally that the remote hasn't
        let local_ops = (other.max_version.into()..self.max_version.into()).into_iter().filter(|ix : &u64| {
            !self.exceptions.contains(&(*ix).into())
        }).map(LogTime::from);

        // 2. Find exceptions that we've seen.
        let mut local_exceptions  = other.exceptions.difference(&self.exceptions).map(|ix| ix.to_owned());

        local_ops.chain(&mut local_exceptions).collect()
    }
}

use crate::Actor;
use crate::dot::Dot;

pub trait CausalOp<A> {

    /// If the result is Some(dot) then this operation cannot occur until the operation that
    /// occured at dot has.
    fn happens_after(&self) -> Option<Dot<A>>;

    /// The time that the current operation occured at
    fn dot(&self) -> Dot<A>;
}

impl<A: Actor, T: CausalOp<A>> CausalityBarrier<A, T> {
    pub fn new(site_id: A) -> Self {
        CausalityBarrier { peers: HashMap::new(), buffer: HashMap::new(), local_id: site_id }
    }

    pub fn ingest(&mut self, op: T) -> Option<T> {
        let v = self.peers.entry(op.dot().actor).or_default();
        // Have we already seen this op?
        if v.is_ready(&op.dot().counter) {
            return None
        }

        v.increment(op.dot().counter);

        // Ok so it's an exception but maybe we can still integrate it if it's not constrained
        // by a happens-before relation.
        // For example: we can always insert into most CRDTs but we can only delete if the
        // corresponding insert happened before!
        match op.happens_after() {
            // Dang! we have a happens after relation!
            Some(dot) => {
                // Let's buffer this operation then.
                if !self.saw_site_do(&dot.actor, &dot.counter) {
                    self.buffer.insert(dot, op);
                    // and do nothing
                    None
                } else {
                    Some(op)
                }
            }
            None => {
                // Ok so we're not causally constrained, but maybe we already saw an associated
                // causal operation? If so let's just delete the pair
                match self.buffer.remove(&op.dot()) {
                    Some(_) => None,
                    None => Some(op),
                }
            }
        }
    }

    fn saw_site_do(&self, site: &A, t: &LogTime) -> bool {
        match self.peers.get(site) {
            Some(ent) => ent.is_ready(t),
            None => { false }
        }
    }

    pub fn expel(&mut self, op: T) -> T {
        let v = self.peers.entry(op.dot().actor).or_insert_with(VectorEntry::new);
        v.increment(op.dot().counter);
        op
    }

    pub fn diff_from(&self, other: &HashMap<A, VectorEntry>) -> HashMap<A, HashSet<LogTime>> {
        let mut ret = HashMap::new();
        for (site_id, entry) in self.peers.iter() {
            let e_diff = match other.get(site_id) {
                Some(remote_entry) => entry.diff_from(remote_entry),
                None => (0..entry.max_version).collect(),
            };
            ret.insert(site_id.clone(), e_diff);
        }
        ret
    }

    pub fn vvwe(&self) -> HashMap<A, VectorEntry> {
        self.peers.clone()
    }

}

#[cfg(test)]
mod test {
    use super::*;
    use derive_more::{From};

    #[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, PartialOrd, Ord, From, Deserialize, Serialize)]
    pub struct SiteId(pub u32);

    #[derive(PartialEq, Debug, Hash, Clone)]
    enum Op {
        Insert(u64),
        Delete(SiteId, LogTime),
    }

    #[derive(PartialEq, Debug, Hash, Clone)]
    pub struct CausalMessage {
        time: LogTime,
        local_id: SiteId,
        op: Op,
    }

    impl CausalOp<SiteId> for CausalMessage {
        fn happens_after(&self) -> Option<Dot<SiteId>> {
            match self.op {
                Op::Insert(_) => None,
                Op::Delete(s, l) => Some(Dot::new(s, l)),
            }
        }

        fn dot(&self) -> Dot<SiteId> {
            Dot::new(self.local_id, self.time)
        }
    }

    #[test]
    fn delete_before_insert() {
        let mut barrier = CausalityBarrier::new(0.into());

        let del = CausalMessage { time: 0, local_id: 1.into(), op: Op::Delete(1.into(), 1) };
        let ins = CausalMessage { time: 1, local_id: 1.into(), op: Op::Insert(0) };
        assert_eq!(barrier.ingest(del), None);
        assert_eq!(barrier.ingest(ins), None);
    }

    #[test]
    fn insert() {
        let mut barrier = CausalityBarrier::new(0.into());

        let ins = CausalMessage { time: 1, local_id: 1.into(), op: Op::Insert(0) };
        assert_eq!(barrier.ingest(ins.clone()), Some(ins.clone()));
    }

    #[test]
    fn insert_then_delete () {
        let mut barrier = CausalityBarrier::new(0.into());

        let ins = CausalMessage { time: 0, local_id: 1.into(), op: Op::Insert(0) };
        let del = CausalMessage { time: 1, local_id: 1.into(), op: Op::Delete(1.into(), 1) };
        assert_eq!(barrier.ingest(ins.clone()), Some(ins));
        assert_eq!(barrier.ingest(del.clone()), Some(del));
    }

    #[test]
    fn delete_before_insert_multiple_sites() {
        let mut barrier = CausalityBarrier::new(0.into());

        let del = CausalMessage { time: 0, local_id: 2.into(), op: Op::Delete(1.into(), 5) };
        let ins = CausalMessage { time: 5, local_id: 1.into(), op: Op::Insert(0) };
        assert_eq!(barrier.ingest(del), None);
        assert_eq!(barrier.ingest(ins), None);
    }

    #[test]
    fn entry_diff_new_entries() {
        let a = VectorEntry::new();
        let b = VectorEntry { max_version: 10, exceptions: HashSet::new() };

        let c : HashSet<LogTime> = (0..10).into_iter().collect();
        assert_eq!(b.diff_from(&a), c);
    }


    #[test]
    fn entry_diff_found_exceptions() {
        let a = VectorEntry { max_version: 10, exceptions: [1,2,3,4].iter().cloned().collect() };
        let b = VectorEntry { max_version: 5, exceptions: HashSet::new() };

        let c : HashSet<LogTime> = [1,2,3,4].iter().cloned().collect();
        assert_eq!(b.diff_from(&a), c);
    }

    #[test]
    fn entry_diff_complex() {
        // a has seen 0, 5
        let a = VectorEntry { max_version: 6, exceptions: [1,2,3,4].iter().cloned().collect() };
        // b has seen 0, 1, 5,6,7,8
        let b = VectorEntry { max_version: 9, exceptions:  [2, 3, 4].iter().cloned().collect() };

        // c should be 1,6,7,8
        let c : HashSet<LogTime> = [1,6,7,8].iter().cloned().collect();
        assert_eq!(b.diff_from(&a), c);
    }
}
