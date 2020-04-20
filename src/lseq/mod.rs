mod lseq;
mod nodes;

use nodes::Atom;
use crate::traits::{Causal, CmRDT};
use crate::vclock::{Actor, VClock};
pub use lseq::{LSeq, Op};
use std::fmt::Display;

impl<V: Ord + Clone + PartialEq + Display, A: Actor + Display> PartialEq for LSeq<V, A> {
    fn eq(&self, other: &Self) -> bool {
        for (_, (dot, _)) in self.tree.inner() {
            let num_found = other
                .tree
                .inner()
                .iter()
                .filter(|(_, (d, _))| d == dot)
                .count();

            if num_found == 0 {
                return false;
            }
            // sanity check
            assert_eq!(num_found, 1);
        }
        for (_, (dot, _)) in other.tree.inner() {
            let num_found = self
                .tree
                .inner()
                .iter()
                .filter(|(_, (d, _))| d == dot)
                .count();

            if num_found == 0 {
                return false;
            }
            // sanity check
            assert_eq!(num_found, 1);
        }
        true
    }
}

impl<V: Ord + Clone + Eq + Display, A: Actor + Display> Eq for LSeq<V, A> {}

impl<V: Ord + Clone + Clone + Display, A: Actor + Display> Causal<A> for LSeq<V, A> {
    fn forget(&mut self, clock: &VClock<A>) {
        for (_, (val_clock, atom)) in self.tree.inner_mut().iter_mut() {
            val_clock.forget(&clock);
            if let Atom::Node((_, ref mut siblings)) = atom {
                // good, keep traversing the tree
                siblings.forget(clock);
            }
        }
        /*                if val_clock.is_empty() {
            None // remove this value from the register
        } else {
            Some((id, (val_clock, val)))
        }*/
    }
}

impl<V: Ord + Clone + Display, A: Actor + Display> CmRDT for LSeq<V, A> {
    type Op = Op<V, A>;

    fn apply(&mut self, op: Self::Op) {
        match op {
            Op::Insert { clock, value, p, q } => {
                if clock.is_empty() {
                    return;
                }
                // first filter out all values that are dominated by the Op clock
                /*self.tree.siblings
                    .retain(|(_, (val_clock, _))| match val_clock.partial_cmp(&clock) {
                        None | Some(Ordering::Greater) => true,
                        _ => false,
                    });

                // TAI: in the case were the Op has a context that already was present,
                //      the above line would remove that value, the next lines would
                //      keep the val from the Op, so.. a malformed Op could break
                //      commutativity.

                // now check if we've already seen this op
                let mut should_add = true;
                let mut id = 0;
                for (i, (existing_clock, _)) in self.tree.siblings.iter() {
                    if existing_clock > &clock {
                        // we've found an entry that dominates this op
                        should_add = false;
                    }
                    id = i + 1;
                }

                if should_add {
                    self.tree.siblings.insert(id, (clock, Atom::Leaf(value)));
                }*/

                println!("\n\nINSERTING {} between {:?} and {:?}", value, p, q);

                // Allocate a new identifier between on p and q
                self.alloc_id(p, q, clock, value);
            }
            Op::Delete { id, .. } => {
                println!("\n\nDELETING {}", id);
                // Delete value from the atom which corresponds to the given identifier
                self.tree.delete_id(id);
            }
        }
    }
}
