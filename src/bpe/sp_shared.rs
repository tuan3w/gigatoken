//! Cross-thread table of unit encodings for the SentencePiece encoder.
//!
//! Every worker thread owns a private front cache, so without sharing each
//! distinct unit would be ranked-merged once per thread that meets it. This
//! table holds the encodings of short units (key <= 15 bytes, <= 12
//! tokens: nearly every distinct unit) for all the threads of one encode
//! call, so a unit is merged once per call, not once per thread. It lives
//! only as long as that call and is sized by the call's input (see
//! `batch::SharedUnitsParams`: 32-128 MiB, outside the per-worker cache
//! budget).
//!
//! Entries are write-once: a slot goes empty -> claimed -> published and is
//! never modified afterwards, so readers need no retry loop. A full probe
//! window just means the unit stays thread-local.

use crate::token::TokenId;
use std::sync::atomic::{AtomicU64, Ordering};

/// One 64-byte entry: a full cache line.
#[repr(C, align(64))]
struct Slot {
    /// Key bytes 0..8 (valid once `hi` holds the real key's high half).
    lo: AtomicU64,
    /// Key bytes 8..16 — nonzero for any real key (the length byte lives
    /// there). 0 = empty, `CLAIMED` = a writer is filling the slot.
    hi: AtomicU64,
    /// Tokens, two per word (low half first); `UNUSED` in unused lanes.
    toks: [AtomicU64; TOKEN_WORDS],
}

/// Never equals a real `hi`: the key's top byte is the length, <= 15.
const CLAIMED: u64 = u64::MAX;
const UNUSED: u32 = u32::MAX;
const TOKEN_WORDS: usize = 6;
/// Slots (cache lines) probed per lookup.
const PROBE: usize = 2;
/// Most tokens a shared entry can hold.
pub(crate) const MAX_SHARED_TOKENS: usize = 2 * TOKEN_WORDS;

pub(crate) struct SharedUnits {
    /// Zeroed backing store (calloc: pages map lazily), over-allocated so
    /// `base` can sit on a cache-line boundary.
    _storage: Vec<u64>,
    base: *const Slot,
    mask: usize,
    /// `64 - log2(slots)`: the table index is the top bits of the key hash.
    shift: u32,
}

// SAFETY: every field is read-only after construction except the slots,
// which are accessed only through atomics.
unsafe impl Send for SharedUnits {}
unsafe impl Sync for SharedUnits {}

/// 64-bit hash of a packed unit key, shared by the front cache index (top
/// bits) and this table (its own top bits).
#[inline(always)]
pub(crate) fn key_hash(key: u128) -> u64 {
    let folded = (key as u64) ^ ((key >> 64) as u64);
    folded.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

impl SharedUnits {
    /// A table of `2^log2_slots` slots (64 B each), zero pages mapped lazily.
    pub(crate) fn new(log2_slots: u32) -> Self {
        let n = 1usize << log2_slots;
        let bytes = n * std::mem::size_of::<Slot>();
        let storage = vec![0u64; (bytes + 64) / 8];
        let addr = storage.as_ptr() as usize;
        let aligned = (addr + 63) & !63;
        // SAFETY: `aligned` stays within the `bytes + 64` allocation with
        // `bytes` to spare after it.
        let base = aligned as *const Slot;
        super::madvise_hugepage(aligned as *mut u8, bytes);
        SharedUnits {
            _storage: storage,
            base,
            mask: n - 1,
            shift: 64 - log2_slots,
        }
    }

    /// Slot `i` (masked into range).
    #[inline(always)]
    fn slot(&self, i: usize) -> &Slot {
        // SAFETY: the index is masked below the slot count, `base` is
        // 64-aligned (Slot's alignment) and the zeroed memory is a valid
        // Slot (all-zero atomics).
        unsafe { &*self.base.add(i & self.mask) }
    }

    #[inline(always)]
    fn base(&self, key: u128) -> usize {
        (key_hash(key) >> self.shift) as usize
    }

    /// Address of the first line a lookup of `key` loads.
    #[inline(always)]
    pub(crate) fn slot_ptr(&self, key: u128) -> *const u8 {
        self.base.wrapping_add(self.base(key) & self.mask) as *const u8
    }

    /// Tokens of `key` into `out`; returns their count, 0 when absent.
    #[inline]
    pub(crate) fn get(&self, key: u128, out: &mut [u32; MAX_SHARED_TOKENS]) -> usize {
        let (klo, khi) = (key as u64, (key >> 64) as u64);
        if khi == 0 {
            return 0;
        }
        let base = self.base(key);
        for k in 0..PROBE {
            let slot = self.slot(base + k);
            let hi = slot.hi.load(Ordering::Acquire);
            if hi == khi {
                if slot.lo.load(Ordering::Relaxed) == klo {
                    // Words 0 and 1 hold the first four tokens (nearly every
                    // unit); the rest are read only for a full prefix.
                    let w0 = slot.toks[0].load(Ordering::Relaxed);
                    let w1 = slot.toks[1].load(Ordering::Relaxed);
                    out[0] = w0 as u32;
                    out[1] = (w0 >> 32) as u32;
                    out[2] = w1 as u32;
                    out[3] = (w1 >> 32) as u32;
                    if out[3] == UNUSED {
                        let mut n = 3;
                        while n > 1 && out[n - 1] == UNUSED {
                            n -= 1;
                        }
                        return n;
                    }
                    let mut n = 4;
                    'words: for w in 2..TOKEN_WORDS {
                        let v = slot.toks[w].load(Ordering::Relaxed);
                        for lane in [v as u32, (v >> 32) as u32] {
                            if lane == UNUSED {
                                break 'words;
                            }
                            out[n] = lane;
                            n += 1;
                        }
                    }
                    return n;
                }
            } else if hi == 0 {
                return 0;
            }
        }
        0
    }

    /// Publish `tokens` (1..=MAX_SHARED_TOKENS) for `key`. Returns false when
    /// the probe window is full (the caller keeps the unit thread-local); a unit
    /// already published by another thread counts as success.
    pub(crate) fn insert(&self, key: u128, tokens: &[TokenId]) -> bool {
        debug_assert!((1..=MAX_SHARED_TOKENS).contains(&tokens.len()));
        let (klo, khi) = (key as u64, (key >> 64) as u64);
        if khi == 0 {
            return false;
        }
        let mut t = [UNUSED; MAX_SHARED_TOKENS];
        for (dst, src) in t.iter_mut().zip(tokens) {
            *dst = src.0;
        }
        let base = self.base(key);
        for k in 0..PROBE {
            let slot = self.slot(base + k);
            let mut hi = slot.hi.load(Ordering::Acquire);
            if hi == 0 {
                match slot
                    .hi
                    .compare_exchange(0, CLAIMED, Ordering::Acquire, Ordering::Acquire)
                {
                    Ok(_) => {
                        slot.lo.store(klo, Ordering::Relaxed);
                        for (w, word) in slot.toks.iter().enumerate() {
                            word.store(
                                t[2 * w] as u64 | (t[2 * w + 1] as u64) << 32,
                                Ordering::Relaxed,
                            );
                        }
                        slot.hi.store(khi, Ordering::Release);
                        return true;
                    }
                    Err(now) => hi = now,
                }
            }
            if hi == khi && slot.lo.load(Ordering::Relaxed) == klo {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(bytes: &[u8]) -> u128 {
        crate::pretokenize::pack_pretoken_key(bytes).unwrap()
    }

    /// Published entries read back exactly, absent keys read as absent, a
    /// duplicate publish is a success, and a full window refuses.
    #[test]
    fn insert_get_roundtrip() {
        const _: () = assert!(std::mem::size_of::<Slot>() == 64);
        let table = SharedUnits::new(10);
        let mut out = [0u32; MAX_SHARED_TOKENS];
        assert_eq!(table.get(key(b"hello"), &mut out), 0);

        for len in 1..=MAX_SHARED_TOKENS {
            let k = key(format!("unit{len}").as_bytes());
            let tokens: Vec<TokenId> = (0..len as u32).map(|i| TokenId(1000 * len as u32 + i)).collect();
            assert!(table.insert(k, &tokens));
            assert!(table.insert(k, &tokens));
            let n = table.get(k, &mut out);
            assert_eq!(n, len);
            for (got, want) in out[..n].iter().zip(&tokens) {
                assert_eq!(*got, want.0);
            }
        }
        assert_eq!(table.get(key(b"unit"), &mut out), 0);
        // Key 0 is never stored.
        assert!(!table.insert(0, &[TokenId(1)]));
        assert_eq!(table.get(0, &mut out), 0);

        // A tiny table fills its probe window and then refuses.
        let tiny = SharedUnits::new(1);
        let mut refused = 0;
        for i in 0..32u32 {
            if !tiny.insert(key(format!("w{i}").as_bytes()), &[TokenId(i)]) {
                refused += 1;
            }
        }
        assert!(refused > 0);
    }
}
