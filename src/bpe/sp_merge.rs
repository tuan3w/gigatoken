//! Ranked BPE merge for SentencePiece units over a flat pair table.
//!
//! Same result as [`crate::bpe::bpe_merge_symbols_ranked`] (repeatedly merge
//! the lowest-rank adjacent pair, leftmost first on a tie) without its
//! hashbrown probes (two cache lines and a SIMD group match per lookup) and
//! without per-call allocation on long units.

use crate::bpe::ranked_merge_key;
use crate::token::TokenId;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone, Copy)]
#[repr(C)]
struct PairSlot {
    /// `ranked_merge_key(a, b)`; `EMPTY` for a free slot (no real key is
    /// all-ones: that would be two `u32::MAX` token IDs).
    key: u64,
    merged: u32,
    rank: u32,
}

const EMPTY: u64 = u64::MAX;
const HASH_MUL: u64 = 0x9E37_79B9_7F4A_7C15;

/// All merges in one open-addressed table at <= 1/2 load: a probe is one
/// multiply, one 16-byte slot load (key, merged ID and rank share a line),
/// and a miss usually ends on the first empty slot.
pub(crate) struct SpPairTable {
    slots: Box<[PairSlot]>,
    mask: usize,
    shift: u32,
}

impl SpPairTable {
    pub(crate) fn empty() -> Self {
        Self::build(&HashMap::<u64, (TokenId, u32), std::hash::RandomState>::new())
    }

    pub(crate) fn build<S: std::hash::BuildHasher>(
        merges: &HashMap<u64, (TokenId, u32), S>,
    ) -> Self {
        let n_slots = (merges.len().max(1) * 2).next_power_of_two().max(64);
        let shift = 64 - n_slots.trailing_zeros();
        let mask = n_slots - 1;
        let mut slots = vec![
            PairSlot {
                key: EMPTY,
                merged: 0,
                rank: u32::MAX,
            };
            n_slots
        ]
        .into_boxed_slice();
        for (&key, &(merged, rank)) in merges {
            let mut idx = (key.wrapping_mul(HASH_MUL) >> shift) as usize;
            while slots[idx].key != EMPTY {
                idx = (idx + 1) & mask;
            }
            slots[idx] = PairSlot {
                key,
                merged: merged.0,
                rank,
            };
        }
        SpPairTable { slots, mask, shift }
    }

    /// `(merged ID, rank)` of the pair, rank `u32::MAX` when it does not merge.
    #[inline(always)]
    fn get(&self, a: u32, b: u32) -> (u32, u32) {
        let key = ((a as u64) << 32) | b as u64;
        let mut idx = (key.wrapping_mul(HASH_MUL) >> self.shift) as usize;
        loop {
            // SAFETY: idx < slots.len(): the hash keeps log2(len) bits and
            // the probe step is masked.
            let slot = unsafe { *self.slots.get_unchecked(idx) };
            if slot.key == key {
                return (slot.merged, slot.rank);
            }
            if slot.key == EMPTY {
                return (0, u32::MAX);
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// Request the first line `get(a, b)` would load.
    #[inline(always)]
    fn prefetch(&self, a: u32, b: u32) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: prefetch has no memory effects; the address is in-bounds.
        unsafe {
            let key = ((a as u64) << 32) | b as u64;
            let idx = (key.wrapping_mul(HASH_MUL) >> self.shift) as usize;
            core::arch::x86_64::_mm_prefetch(
                self.slots.as_ptr().add(idx) as *const i8,
                core::arch::x86_64::_MM_HINT_T0,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (a, b);
    }
}

/// Longest sequence handled by the stack-array merge.
const SMALL_MAX: usize = 32;

/// `(rank << 8) | position` of the pair starting at `pos`; ranks of
/// non-merging pairs are `u32::MAX`, which [`has_merge`] rejects.
#[inline(always)]
fn pack(rank: u32, pos: usize) -> u64 {
    ((rank as u64) << 8) | pos as u64
}

#[inline(always)]
fn has_merge(best: u64) -> bool {
    (best >> 8) as u32 != u32::MAX
}

/// Merge `symbols` (2..=SMALL_MAX of them) in place on stack arrays; returns
/// the surviving count.
fn merge_small(table: &SpPairTable, symbols: &mut [TokenId]) -> usize {
    let n = symbols.len();
    debug_assert!((2..=SMALL_MAX).contains(&n));
    // Doubly-linked list over positions: positions never move.
    let mut next = [0u8; SMALL_MAX];
    let mut prev = [0u8; SMALL_MAX];
    for i in 0..n {
        next[i] = (i + 1) as u8;
        prev[i] = (i as u8).wrapping_sub(1);
    }
    // pr[i]: packed priority of the pair starting at live position i. The
    // plain u64 minimum picks the lowest rank, then the leftmost position.
    let mut pr = [u64::MAX; SMALL_MAX];
    let mut merged = [0u32; SMALL_MAX];
    for i in 0..n - 1 {
        let (m, r) = table.get(symbols[i].0, symbols[i + 1].0);
        merged[i] = m;
        pr[i] = pack(r, i);
    }
    loop {
        let mut best = u64::MAX;
        for &p in &pr[..n - 1] {
            best = best.min(p);
        }
        if !has_merge(best) {
            break;
        }
        let i = (best & 0xFF) as usize;
        let new_sym = merged[i];
        let dead = next[i] as usize;
        let new_right = next[dead] as usize;
        let left = prev[i] as usize;
        // Both refresh pairs are fixed now; start their loads before the
        // list surgery.
        if new_right < n {
            table.prefetch(new_sym, symbols[new_right].0);
        }
        if left < n {
            table.prefetch(symbols[left].0, new_sym);
        }
        symbols[i] = TokenId(new_sym);
        next[i] = new_right as u8;
        pr[dead] = u64::MAX;
        if new_right < n {
            prev[new_right] = i as u8;
            let (m, r) = table.get(new_sym, symbols[new_right].0);
            merged[i] = m;
            pr[i] = pack(r, i);
        } else {
            pr[i] = u64::MAX;
        }
        if left < n {
            let (m, r) = table.get(symbols[left].0, new_sym);
            merged[left] = m;
            pr[left] = pack(r, left);
        }
    }
    let mut write = 0;
    let mut i = 0;
    while i < n {
        symbols[write] = symbols[i];
        write += 1;
        i = next[i] as usize;
    }
    write
}

/// Total capacity above which [`merge_ranked`] drops its long-sequence
/// buffers after use: a single giant unit (100 MB of CJK or base64 without
/// a mark) must not leave hundreds of MB resident in the state.
pub(crate) const SCRATCH_RETAIN_BYTES: usize = 1 << 20;

/// Reusable buffers for [`merge_ranked`]'s long-sequence path.
#[derive(Default)]
pub(crate) struct SpMergeScratch {
    next: Vec<u32>,
    prev: Vec<u32>,
    rank: Vec<u32>,
    merged: Vec<u32>,
    heap: BinaryHeap<Reverse<u64>>,
}

impl SpMergeScratch {
    /// Drop every buffer if together they hold more than
    /// [`SCRATCH_RETAIN_BYTES`].
    fn release_if_oversized(&mut self) {
        let bytes = (self.next.capacity()
            + self.prev.capacity()
            + self.rank.capacity()
            + self.merged.capacity())
            * std::mem::size_of::<u32>()
            + self.heap.capacity() * std::mem::size_of::<Reverse<u64>>();
        if bytes > SCRATCH_RETAIN_BYTES {
            *self = Self::default();
        }
    }
}

/// Heap-driven merge for sequences longer than [`SMALL_MAX`]: O(n log n),
/// no allocation once the scratch has grown.
fn merge_large(table: &SpPairTable, symbols: &mut Vec<TokenId>, s: &mut SpMergeScratch) {
    const NONE: u32 = u32::MAX;
    let n = symbols.len();
    s.next.clear();
    s.next.extend(1..=n as u32);
    s.prev.clear();
    s.prev.push(NONE);
    s.prev.extend(0..n as u32 - 1);
    s.rank.clear();
    s.rank.resize(n, u32::MAX);
    s.merged.clear();
    s.merged.resize(n, 0);
    s.heap.clear();
    for i in 0..n - 1 {
        let (m, r) = table.get(symbols[i].0, symbols[i + 1].0);
        s.merged[i] = m;
        s.rank[i] = r;
        if r != u32::MAX {
            s.heap.push(Reverse(((r as u64) << 32) | i as u64));
        }
    }
    while let Some(Reverse(entry)) = s.heap.pop() {
        let pos = (entry & 0xFFFF_FFFF) as usize;
        // Stale: merged away, or its pair changed since the push (a rank
        // names exactly one pair, so equal rank means the same pair).
        if s.rank[pos] != (entry >> 32) as u32 {
            continue;
        }
        let new_sym = s.merged[pos];
        symbols[pos] = TokenId(new_sym);
        let dead = s.next[pos] as usize;
        let new_right = s.next[dead] as usize;
        s.next[pos] = new_right as u32;
        s.rank[dead] = u32::MAX;
        if new_right < n {
            s.prev[new_right] = pos as u32;
            let (m, r) = table.get(new_sym, symbols[new_right].0);
            s.merged[pos] = m;
            s.rank[pos] = r;
            if r != u32::MAX {
                s.heap.push(Reverse(((r as u64) << 32) | pos as u64));
            }
        } else {
            s.rank[pos] = u32::MAX;
        }
        let left = s.prev[pos];
        if left != NONE {
            let left = left as usize;
            let (m, r) = table.get(symbols[left].0, new_sym);
            s.merged[left] = m;
            s.rank[left] = r;
            if r != u32::MAX {
                s.heap.push(Reverse(((r as u64) << 32) | left as u64));
            }
        }
    }
    let mut write = 0;
    let mut i = 0;
    while i < n {
        symbols[write] = symbols[i];
        write += 1;
        i = s.next[i] as usize;
    }
    symbols.truncate(write);
    s.release_if_oversized();
}

/// Apply the ranked merges to `symbols` in place.
pub(crate) fn merge_ranked(
    table: &SpPairTable,
    symbols: &mut Vec<TokenId>,
    scratch: &mut SpMergeScratch,
) {
    let n = symbols.len();
    if n < 2 {
        return;
    }
    if n <= SMALL_MAX {
        let len = merge_small(table, &mut symbols[..]);
        symbols.truncate(len);
    } else {
        merge_large(table, symbols, scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::bpe_merge_symbols_ranked;
    use rustc_hash::FxBuildHasher;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// The flat-table merges must agree with the hashbrown heap merge on
    /// random merge tables and sequences, across the small/large boundary,
    /// including sequences where one pair (one rank) recurs at many
    /// adjacent and overlapping positions.
    #[test]
    fn flat_merge_matches_heap_merge() {
        let mut rng = Rng(0x0DDB_1A5E_5BAD_5EED);
        let mut scratch = SpMergeScratch::default();
        for trial in 0..3000u64 {
            let id_range = 8 + (trial % 5) * 40;
            let mut merges: HashMap<u64, (TokenId, u32), FxBuildHasher> =
                HashMap::with_hasher(FxBuildHasher);
            let n_merges = 20 + (trial % 11) * 60;
            let mut rank = 0u32;
            for _ in 0..n_merges {
                let a = TokenId(rng.below(id_range) as u32);
                let b = TokenId(rng.below(id_range) as u32);
                merges
                    .entry(ranked_merge_key(a, b))
                    .or_insert((TokenId(rng.below(id_range + n_merges) as u32), rank));
                rank += 1;
            }
            // Every third trial: a repeating motif (period 1-3) with sparse
            // noise, plus merges for the motif's own pairs. The same pair,
            // hence the same rank, then recurs at many positions (also
            // overlapping ones, "a a a"), so the leftmost-first tie rule
            // decides which merge happens first.
            let n = 2 + rng.below(90) as usize;
            let motif: Option<Vec<TokenId>> = (trial % 3 == 0).then(|| {
                let period = 1 + rng.below(3) as usize;
                (0..period)
                    .map(|_| TokenId(rng.below(id_range) as u32))
                    .collect()
            });
            if let Some(motif) = &motif {
                for i in 0..motif.len() {
                    let (a, b) = (motif[i], motif[(i + 1) % motif.len()]);
                    merges
                        .entry(ranked_merge_key(a, b))
                        .or_insert((TokenId(rng.below(id_range + n_merges) as u32), rank));
                    rank += 1;
                }
            }
            let table = SpPairTable::build(&merges);
            let init: Vec<TokenId> = (0..n)
                .map(|i| {
                    let noise = rng.below(8) == 0;
                    match &motif {
                        Some(m) if !noise => m[i % m.len()],
                        _ => TokenId(rng.below(id_range) as u32),
                    }
                })
                .collect();
            let mut reference = init.clone();
            bpe_merge_symbols_ranked(&merges, &mut reference);
            let mut got = init.clone();
            merge_ranked(&table, &mut got, &mut scratch);
            assert_eq!(got, reference, "diverged on {init:?}");
        }
    }
}

#[cfg(test)]
mod scratch_tests {
    use super::*;
    use rustc_hash::FxBuildHasher;

    /// A giant sequence must not leave its merge buffers behind, while a
    /// modest long one keeps them for reuse.
    #[test]
    fn oversized_scratch_is_released() {
        let mut merges: HashMap<u64, (TokenId, u32), FxBuildHasher> =
            HashMap::with_hasher(FxBuildHasher);
        merges.insert(ranked_merge_key(TokenId(1), TokenId(1)), (TokenId(1), 0));
        let table = SpPairTable::build(&merges);
        let mut scratch = SpMergeScratch::default();

        let mut small = vec![TokenId(1); 100];
        merge_ranked(&table, &mut small, &mut scratch);
        assert_eq!(small, [TokenId(1)]);
        assert!(scratch.next.capacity() >= 100, "modest scratch is kept");

        let mut giant = vec![TokenId(1); 400_000];
        merge_ranked(&table, &mut giant, &mut scratch);
        assert_eq!(giant, [TokenId(1)]);
        assert_eq!(scratch.next.capacity(), 0, "giant scratch is dropped");
        assert_eq!(scratch.heap.capacity(), 0);
    }
}
