use super::nodes::{Atom, Identifier, Siblings};
use crate::ctx::{AddCtx, ReadCtx, RmCtx};
use crate::traits::{CmRDT, CvRDT};
use crate::vclock::{Actor, VClock};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::{
    cmp,
    fmt::{self, Display},
};

const DEFAULT_STRATEGY_BOUNDARY: u8 = 10;
const DEFAULT_ROOT_BASE: u8 = 32; // This needs to be greater than boundary, and probably needs to be a power of 2
const BEGIN_ID: u64 = 0;
const END_ID: u64 = std::u64::MAX;

/// An LSeq, a variable-size identifiers class of sequence CRDT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSeq<V: Ord + Clone + Display + Default, A: Actor + Display> {
    /// Boundary for choosing a new number when allocating an identifier
    boundary: u8,
    /// Arity of the root tree node. The arity is doubled at each depth
    root_arity: u8,
    /// When inserting, we have a randomly chosen strategy for
    /// generating the id of the atom at each depth
    strategies: Vec<bool>, // true = boundary+, false = boundary-
    /// Depth 1 siblings nodes
    pub(crate) tree: Siblings<V, A>,
}

impl<V: Ord + Clone + Display + Default, A: Actor + Display> Default for LSeq<V, A> {
    fn default() -> Self {
        Self::new(DEFAULT_STRATEGY_BOUNDARY, DEFAULT_ROOT_BASE)
    }
}

/// Defines the set of operations supported by LSeq
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op<V: Ord + Clone, A: Actor> {
    /// Insert a value in the sequence
    Insert {
        /// context of the operation
        clock: VClock<A>,
        /// the value to insert
        value: V,
        /// preceding value identifier (None == BEGIN)
        p: Option<Identifier>,
        /// succeeding value identifier (None == END, which can be used to append values)
        q: Option<Identifier>,
    },

    /// Delete a value from the sequence
    Delete {
        /// context of the operation
        clock: VClock<A>,
        /// the identifier of the value to delete
        id: Identifier,
    },
}

impl<V: Ord + Clone + Display + Default, A: Actor + Display> Display for LSeq<V, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "|")?;
        for (i, (ctx, val)) in self.tree.inner().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{} {}@{:?}", val.1, i, ctx)?;
        }
        write!(f, "|")
    }
}

/// Implementation of the core LSeq functionality
impl<V: Ord + Clone + Display + Default, A: Actor + Display> LSeq<V, A> {
    /// Construct a new empty LSeq with given boundary and root arity settings
    pub fn new(boundary: u8, root_arity: u8) -> Self {
        Self {
            boundary,
            root_arity,
            strategies: vec![true], // boundary+ strategy for depth 0
            tree: Siblings::new(),
        }
    }

    /// Generate operation for inserting a value between identifiers p and q
    pub fn insert(
        &self,
        value: V,
        p: Option<Identifier>,
        q: Option<Identifier>,
        ctx: AddCtx<A>,
    ) -> Op<V, A> {
        Op::Insert {
            clock: ctx.clock,
            value,
            p,
            q,
        }
    }

    /// Generate operation to deleting a value given its identifier
    pub fn delete(&self, id: Identifier, ctx: RmCtx<A>) -> Op<V, A> {
        Op::Delete {
            clock: ctx.clock,
            id,
        }
    }

    /// Generates a read operation to obtain current state of the sequence
    pub fn read(&self) -> ReadCtx<Vec<(Identifier, V)>, A>
    where
        V: Clone,
    {
        let clock = self.clock();
        let sequence = self.flatten();
        ReadCtx {
            add_clock: clock.clone(),
            rm_clock: clock,
            val: sequence,
        }
    }

    /// Retrieve the current read context
    pub fn read_ctx(&self) -> ReadCtx<(), A> {
        let clock = self.clock();
        ReadCtx {
            add_clock: clock.clone(),
            rm_clock: clock,
            val: (),
        }
    }

    // Private helpers functions

    /// Flatten tree into an ordered sequence of (Identifier, Value)
    fn flatten(&self) -> Vec<(Identifier, V)> {
        let mut seq = vec![];
        self.flatten_tree(&self.tree, Identifier::new(&[]), &mut seq);
        seq
    }

    /// Returns a clock with latest versions of all actors operating on this LSeq
    fn clock(&self) -> VClock<A> {
        self.tree
            .inner()
            .iter()
            .fold(VClock::new(), |mut accum_clock, (_, (c, _))| {
                accum_clock.merge(c.clone());
                accum_clock
            })
    }

    /// This method chooses randomly a stratey for each depth
    /// It's not clear if this would work for CRDT when applying operations to different replicas???
    /// We may need to allow user to provide a seed for the random function which can suffice its use case
    #[allow(dead_code)]
    fn get_random_strategy(&mut self, depth: usize) -> bool {
        if depth >= self.strategies.len() {
            // we need to add a new strategy to our cache
            let new_strategy = thread_rng().gen_bool(0.5);
            println!("NEW strategy: {}", new_strategy);
            self.strategies.push(new_strategy);
            new_strategy
        } else {
            self.strategies[depth]
        }
    }

    /// This method deterministically chooses an stratey for each depth,
    /// a boundary+ is chosen if depth is even, and boundary- otherwise
    fn get_deterministic_strategy(&self, depth: usize) -> bool {
        if depth % 2 == 0 {
            true
        } else {
            false
        }
    }

    /// Returns the arity of the tree at a given depth
    fn arity_at(&self, depth: usize) -> u64 {
        let mut arity = self.root_arity as u64;
        for _ in 0..depth {
            arity = arity * 2;
        }
        arity
    }

    /// Allocates a new identifier between given p and q identifiers
    pub(crate) fn alloc_id(
        &mut self,
        p: Option<Identifier>,
        q: Option<Identifier>,
        clock: VClock<A>,
        value: V,
    ) {
        let p = p.unwrap_or_else(|| Identifier::new(&[BEGIN_ID]));
        let q = q.unwrap_or_else(|| Identifier::new(&[END_ID]));

        // Let's get the interval between p and q, and also the depth at which
        // we should generate the new identifier
        let (new_id_depth, interval) = self.find_new_id_depth(&p, &q);
        println!("INTERVAL FOUND: {}", interval);

        // Let's make sure we allocate the new number within the preset boundary and interval obtained
        let step = cmp::min(interval, self.boundary as u64);

        // Define if we should apply a boundary+ or boundary- stratey for the
        // new number, based on the depth where it's being added
        let depth_strategy = self.get_deterministic_strategy(new_id_depth);

        // Depening on the strategy to apply, let's figure which is the new number
        let new_number = self.gen_new_number(new_id_depth, depth_strategy, step, &p, &q);

        // Let's now attempt to insert the new identifier in the tree at new_id_depth
        let mut cur_depth_nodes = self.tree.inner_mut();
        for d in 0..new_id_depth {
            // This is not yet the depth where to add the new number,
            // therefore we just check which child is the path of p/q at current's depth
            let cur_number = if depth_strategy { p.at(d) } else { q.at(d) };

            // If there is a 'Leaf' at this depth, or if there is not even an atom,
            // we make sure we have a 'Node' so we can allocate the children afterwards
            match cur_depth_nodes.get(&cur_number) {
                Some(&(ref c, Atom::Leaf(ref v))) => {
                    // convert it into a 'Node' maintaining its value and clock info
                    let children = Siblings::new();
                    let new_atom = Atom::Node((v.clone(), children));
                    cur_depth_nodes.insert(cur_number, (c.clone(), new_atom));
                }
                Some(&(_, Atom::Node(_))) => { /* there is a Node already so we are good */ }
                None => {
                    // TODO: handle it properly and discover if it's a valid case
                    panic!("Do we need to create not only 1 new level but more???");
                }
            }

            // Now we can just step into the next depth of siblings to keep traversing the tree
            if let Some(&mut (_, Atom::Node((_, ref mut siblings)))) =
                cur_depth_nodes.get_mut(&cur_number)
            {
                cur_depth_nodes = siblings.inner_mut();
            } else {
                // TODO: handle it properly, this should never happen
                panic!("unexpected!!!");
            }
        }

        // 'cur_depth_nodes' should now be referencing the siblings
        // where we need to insert the new number
        println!("New number {} for depth {}", new_number, new_id_depth);
        if !cur_depth_nodes.contains_key(&new_number) {
            // It seems the slot picked is available, thus we'll use that one
            println!("It's free!!!");
            let new_atom = Atom::Leaf(value.clone());
            cur_depth_nodes.insert(new_number, (clock.clone(), new_atom));
        } else {
            // TODO: we should retry find a new number, or merge clocks?
            panic!("number was already taken!");
        }

        println!(
            "New number {} allocated at depth {}",
            new_number, new_id_depth
        );
    }

    // Finds out what's the interval between p and q (reagrdless of their length/height),
    // and figure out which depth (either on p or q path) the new identifier should be generated at
    fn find_new_id_depth(&self, p: &Identifier, q: &Identifier) -> (usize, u64) {
        let mut interval: u64;
        let mut p_position = 0;
        let mut q_position = 0;
        let mut new_id_depth = 0;
        loop {
            // tree arity at current depth
            let arity = self.arity_at(new_id_depth);

            println!(
                "Checking interval at depth {} between {} and {}, arity {}...",
                new_id_depth, p, q, arity
            );

            // Calculate what would be the position in the sequence of p at current depth
            let shift = (arity as f64).log2() as u32;
            if new_id_depth < p.len() {
                let i = p.at(new_id_depth);
                p_position = (p_position << shift) + i;
            } else {
                p_position = p_position << shift;
            }

            // Calculate what would be the position in the sequence of q at current depth
            if new_id_depth < q.len() {
                let i = q.at(new_id_depth);
                q_position = (q_position << shift) + i;
            } else {
                q_position = (q_position << shift) + arity - 1;
            }

            // What's the interval between p and q identifiers at current depth?
            interval = if p_position > q_position {
                // TODO: return error? the trait doesn't support that type of Result currently
                panic!("p cannot be greater than q");
            } else if q_position > p_position {
                q_position - p_position - 1
            } else {
                // p and q positions are equal
                0
            };

            // Did we reach a depth where there is room for a new id?
            if interval > 0 {
                break;
            } else {
                // ...nope...let's keep going
                new_id_depth = new_id_depth + 1;
            }
        }

        (new_id_depth, interval)
    }

    /// Get a new number to insert in either p or q path at a
    /// given depth, and based on the depth's strategy
    fn gen_new_number(
        &self,
        depth: usize,
        strategy: bool,
        step: u64,
        p: &Identifier,
        q: &Identifier,
    ) -> u64 {
        // Depening on the strategy to apply, let's figure which is the reference number
        // we'll be adding to, or substracting from, to obtain the new number
        if strategy {
            // We then apply boundary+ strategy from p
            let reference_num = if depth < p.len() {
                p.at(depth)
            } else {
                BEGIN_ID
            };

            // TODO: we may need a seed provided by the user to we get a deterministic result
            let n = thread_rng().gen_range(reference_num + 1, reference_num + step + 1);
            //let n = reference_num + (step / 2) + 1;
            println!("STEP boundary+ (step {}): {}", step, n);
            n
        } else {
            // ...ok, then apply boundary- strategy from q
            let reference_num = if depth < q.len() {
                q.at(depth)
            } else {
                self.arity_at(depth) - 1 // == END at new id's depth
            };

            // TODO: we may need a seed provided by the user to we get a deterministic result
            let n = thread_rng().gen_range(reference_num - step, reference_num);
            //let n = reference_num - (step / 2) - 1;
            println!("STEP boundary- (step {}): {}", step, n);
            n
        }
    }

    /// Recursivelly flattens the tree formed by the given siblings nodes
    /// The prefix is used for generating each Identifier in the sequence
    fn flatten_tree(
        &self,
        siblings: &Siblings<V, A>,
        prefix: Identifier,
        seq: &mut Vec<(Identifier, V)>,
    ) {
        for (id, (_, atom)) in siblings.inner() {
            let mut new_prefix = prefix.clone();
            // We first push current node's number to the prefix
            new_prefix.push(*id);
            match atom {
                Atom::Leaf(value) => seq.push((new_prefix.clone(), value.clone())),
                Atom::Node((value, s)) => {
                    // Add current item to the sequence before/after processing chldren,
                    // depending on the current level's strategy
                    let chidren_depth = prefix.len() + 1;
                    if self.get_deterministic_strategy(chidren_depth) {
                        seq.push((new_prefix.clone(), value.clone()));
                        self.flatten_tree(&s, new_prefix, seq);
                    } else {
                        self.flatten_tree(&s, new_prefix.clone(), seq);
                        seq.push((new_prefix, value.clone()));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // Helper to populate an LSeq with some elements
    fn populate_seq<V: Ord + Clone + Display + Default, A: Actor + Display>(
        elems: &[V],
        seq: &mut LSeq<V, A>,
        actor: A,
    ) {
        for e in elems {
            // Insert e between BEGIN and END
            let add_ctx = seq.read_ctx().derive_add_ctx(actor.clone());
            seq.apply(seq.insert(e.clone(), None, None, add_ctx.clone()));
        }
    }

    #[test]
    fn test_default() {
        let seq = LSeq::<u64, u64>::default();
        assert_eq!(
            seq,
            LSeq {
                boundary: DEFAULT_STRATEGY_BOUNDARY,
                root_arity: DEFAULT_ROOT_BASE,
                strategies: vec![true],
                tree: Siblings::new()
            }
        );
    }

    #[test]
    fn test_insert() {
        let mut seq = LSeq::<char, u64>::default();
        let actor = 100;

        // Insert A to [] (between BEGIN and END)
        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        seq.apply(seq.insert('A', None, None, add_ctx.clone()));

        let current_seq = seq.read().val;
        println!("SEQ [A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 1);
        assert_eq!(current_seq[0].1, 'A');

        // Insert B to [A] (between A and END)
        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let (id_of_a, _) = &current_seq[0];
        seq.apply(seq.insert('B', Some(id_of_a.clone()), None, add_ctx.clone()));

        let current_seq = seq.read().val;
        println!("SEQ [A, B]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 2);
        assert_eq!(current_seq[0].1, 'A');
        assert_eq!(current_seq[1].1, 'B');
    }

    #[test]
    fn test_delete() {
        let mut seq = LSeq::<char, u64>::default();
        let actor = 100;

        // Insert A to [] (between BEGIN and END)
        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        seq.apply(seq.insert('A', None, None, add_ctx.clone()));

        let current_seq = seq.read().val;
        println!("SEQ [A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 1);
        assert_eq!(current_seq[0].1, 'A');

        // Insert B to [A] (between A and END)
        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let (id_of_a, _) = &current_seq[0];
        seq.apply(seq.insert('B', Some(id_of_a.clone()), None, add_ctx.clone()));

        let current_seq = seq.read().val;
        println!("SEQ [A, B]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 2);
        assert_eq!(current_seq[0].1, 'A');
        assert_eq!(current_seq[1].1, 'B');

        // Delete B from [A, B]
        let rm_ctx = seq.read_ctx().derive_rm_ctx();
        let (id_of_b, _) = &current_seq[1];
        seq.apply(seq.delete(id_of_b.clone(), rm_ctx.clone()));

        let current_seq = seq.read().val;
        println!("SEQ [A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 1);
        assert_eq!(current_seq[0].1, 'A');
    }

    #[test]
    #[ignore]
    fn test_insert_new_depth() {
        let mut seq = LSeq::<char, u64>::default();
        let actor = 100;
        populate_seq(&['A', 'B'], &mut seq, actor);

        let current_seq = seq.read().val;
        println!("SEQ: {:?}", current_seq);
    }

    #[test]
    fn test_several_insertions() {
        let mut seq = LSeq::<char, u64>::default();
        let actor = 100;

        // Insert A to [] (between BEGIN and END)
        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('A', None, None, add_ctx.clone());
        assert_eq!(
            op,
            Op::Insert {
                clock: add_ctx.clock,
                value: 'A',
                p: None,
                q: None
            }
        );
        seq.apply(op);

        // Insert B to [A] (between BEGIN and A)
        let current_seq = seq.read().val;
        println!("SEQ [A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 1);
        let (id_of_a, _) = &current_seq[0];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('B', None, Some(id_of_a.clone()), add_ctx.clone());
        seq.apply(op);

        // Insert C to [B, A] (between B and A)
        let current_seq = seq.read().val;
        println!("SEQ [B, A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 2);
        let (id_of_b, _) = &current_seq[0];
        let (id_of_a, _) = &current_seq[1];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert(
            'C',
            Some(id_of_b.clone()),
            Some(id_of_a.clone()),
            add_ctx.clone(),
        );
        seq.apply(op);

        // Insert D to [B, C, A] (between C and A)
        let current_seq = seq.read().val;
        println!("SEQ [B, C, A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 3);
        let (id_of_c, _) = &current_seq[1];
        let (id_of_a, _) = &current_seq[2];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert(
            'D',
            Some(id_of_c.clone()),
            Some(id_of_a.clone()),
            add_ctx.clone(),
        );
        seq.apply(op);

        // Insert E to [B, C, D, A] (between B and C)
        let current_seq = seq.read().val;
        println!("SEQ [B, C, D, A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 4);
        let (id_of_b, _) = &current_seq[0];
        let (id_of_c, _) = &current_seq[1];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert(
            'E',
            Some(id_of_b.clone()),
            Some(id_of_c.clone()),
            add_ctx.clone(),
        );
        seq.apply(op);

        // Insert F to [B, E, C, D, A] (between D and A)
        let current_seq = seq.read().val;
        println!("SEQ [B, E, C, D, A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 5);
        let (id_of_d, _) = &current_seq[3];
        let (id_of_a, _) = &current_seq[4];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert(
            'F',
            Some(id_of_d.clone()),
            Some(id_of_a.clone()),
            add_ctx.clone(),
        );
        seq.apply(op);

        // Test final length
        let current_seq = seq.read().val;
        println!("FINAL SEQ: {:?}", current_seq);
        assert_eq!(current_seq.len(), 6);
    }

    #[test]
    fn test_append() {
        let mut seq = LSeq::<char, u64>::default();
        let actor = 100;

        // Append A to [] (between BEGIN and END)
        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('A', None, None, add_ctx.clone());
        seq.apply(op);

        // Append B to [A] (between A and END)
        let current_seq = seq.read().val;
        println!("SEQ [A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 1);
        let (id_of_a, _) = &current_seq[0];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('B', Some(id_of_a.clone()), None, add_ctx.clone());
        seq.apply(op);

        // Append C to [A, B] (between B and END)
        let current_seq = seq.read().val;
        println!("SEQ [A, B]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 2);
        let (id_of_b, _) = &current_seq[1];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('C', Some(id_of_b.clone()), None, add_ctx.clone());
        seq.apply(op);

        // Append D to [A, B, C] (between C and END)
        let current_seq = seq.read().val;
        println!("SEQ [A, B, C]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 3);
        let (id_of_c, _) = &current_seq[2];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('D', Some(id_of_c.clone()), None, add_ctx.clone());
        seq.apply(op);

        // Test final length
        let current_seq = seq.read().val;
        println!("FINAL SEQ: {:?}", current_seq);
        assert_eq!(current_seq.len(), 4);
    }

    #[test]
    fn test_insert_at_begining() {
        let mut seq = LSeq::<char, u64>::default();
        let actor = 100;

        // Insert A to [] (between BEGIN and END)
        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('A', None, None, add_ctx.clone());
        seq.apply(op);

        // Insert B to [A] (between BEGIN and A)
        let current_seq = seq.read().val;
        println!("SEQ [A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 1);
        let (id_of_a, _) = &current_seq[0];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('B', None, Some(id_of_a.clone()), add_ctx.clone());
        seq.apply(op);

        // Insert C to [B, A] (between BEGIN and B)
        let current_seq = seq.read().val;
        println!("SEQ [B, A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 2);
        let (id_of_b, _) = &current_seq[0];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('C', None, Some(id_of_b.clone()), add_ctx.clone());
        seq.apply(op);

        // Insert D to [C, B, A] (between BEGIN and C)
        let current_seq = seq.read().val;
        println!("SEQ [C, B, A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 3);
        let (id_of_c, _) = &current_seq[0];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('D', None, Some(id_of_c.clone()), add_ctx.clone());
        seq.apply(op);

        // Test final length
        let current_seq = seq.read().val;
        println!("FINAL SEQ: {:?}", current_seq);
        assert_eq!(current_seq.len(), 4);
    }

    #[test]
    #[should_panic]
    fn test_insert_p_greater_than_q() {
        let mut seq = LSeq::<char, u64>::new(2, 2);
        let actor = 100;

        // Insert A to [] (between BEGIN and END)
        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('A', None, None, add_ctx.clone());
        seq.apply(op);

        // Insert B to [A] (between A and END)
        let current_seq = seq.read().val;
        println!("SEQ [A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 1);
        let (id_of_a, _) = &current_seq[0];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('B', Some(id_of_a.clone()), None, add_ctx.clone());
        seq.apply(op);

        // Insert C to [A, B] (between B and A == wrong order)
        let current_seq = seq.read().val;
        println!("SEQ [A, B]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 3);
        let (id_of_a, _) = &current_seq[0];
        let (id_of_b, _) = &current_seq[1];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert(
            'C',
            Some(id_of_b.clone()),
            Some(id_of_a.clone()),
            add_ctx.clone(),
        );

        seq.apply(op); // should fail
    }

    #[test]
    #[ignore]
    fn test_insert_nonexisting_id() {
        let mut seq = LSeq::<char, u64>::default();
        let actor = 100;

        // Insert A to [] (between BEGIN and END)
        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('A', None, None, add_ctx.clone());
        seq.apply(op);

        // Insert B to [A] (between BEGIN and <invalid id>)
        let current_seq = seq.read().val;
        println!("SEQ [A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 1);

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('B', None, Some(Identifier::new(&[11])), add_ctx.clone());
        // should fail? will VClock help us here to know it's just an id we are not aware of yet??
        seq.apply(op);
    }

    #[test]
    #[ignore]
    fn test_insert_somewhere_strange() {
        let mut seq = LSeq::<char, u64>::default();
        let actor = 100;

        // Insert A to [] (between BEGIN and END)
        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('A', None, None, add_ctx.clone());
        seq.apply(op);

        // Insert B to [A] (between BEGIN and A)
        let current_seq = seq.read().val;
        println!("SEQ [A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 1);
        let (id_of_a, _) = &current_seq[0];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('B', None, Some(id_of_a.clone()), add_ctx.clone());
        seq.apply(op);

        // Insert C to [B, A] (between B and A)
        let current_seq = seq.read().val;
        println!("SEQ [B, A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 2);
        let (id_of_b, _) = &current_seq[0];
        let (id_of_a, _) = &current_seq[1];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert(
            'C',
            Some(id_of_b.clone()),
            Some(id_of_a.clone()),
            add_ctx.clone(),
        );
        seq.apply(op);

        // Insert D to [B, C, A] (between C and A)
        let current_seq = seq.read().val;
        println!("SEQ [B, C, A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 3);
        let (id_of_c, _) = &current_seq[1];
        let (id_of_a, _) = &current_seq[2];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert(
            'D',
            Some(id_of_c.clone()),
            Some(id_of_a.clone()),
            add_ctx.clone(),
        );
        seq.apply(op);

        // Insert E to [B, C, D, A] (between None and D)
        let current_seq = seq.read().val;
        println!("SEQ [B, C, D, A]: {:?}", current_seq);
        assert_eq!(current_seq.len(), 4);
        let (id_of_d, _) = &current_seq[2];

        let add_ctx = seq.read_ctx().derive_add_ctx(actor);
        let op = seq.insert('E', None, Some(id_of_d.clone()), add_ctx.clone());
        seq.apply(op);

        // Test final length
        let current_seq = seq.read().val;
        println!("FINAL SEQ: {:?}", current_seq);
        assert_eq!(current_seq.len(), 5);
    }
}
