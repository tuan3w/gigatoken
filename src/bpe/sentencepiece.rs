use crate::bpe::sp_merge::{SCRATCH_RETAIN_BYTES, SpMergeScratch, SpPairTable, merge_ranked};
use crate::bpe::sp_shared::{MAX_SHARED_TOKENS, SharedUnits, key_hash};
use crate::pretokenize::pack_pretoken_key;
use crate::token::TokenId;
use rustc_hash::FxBuildHasher;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

/// SentencePiece uses U+2581 (▁) as a space marker.
const SENTENCEPIECE_SPACE: char = '\u{2581}';
const SENTENCEPIECE_SPACE_STR: &str = "\u{2581}";
const SP_MARK: [u8; 3] = [0xE2, 0x96, 0x81]; // UTF-8 bytes of ▁

/// How text divides into independently-encodable (and cacheable) word units.
/// Computed once at load from the pre-tokenizer config and the vocab.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WordSplit {
    /// A unit starts at each ▁ that follows a non-▁ char: units look like
    /// `▁▁▁word`. Valid when no vocab piece contains a ▁ after a non-▁ char,
    /// so no merge can cross a unit boundary and per-unit BPE equals the
    /// global merge.
    SpaceRuns,
    /// Metaspace `split=true`: a unit starts at every ▁ (HF splits there, so
    /// this is exact regardless of the vocab).
    EveryMark,
    /// The vocab has boundary-crossing pieces; merge whole chunks, uncached.
    None,
}

/// How the raw fast path prepends the dummy-prefix ▁ to a chunk.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RawPrepend {
    Never,
    /// `Prepend` normalizer: unconditional (Llama 2).
    Unguarded,
    /// Metaspace `always`: skipped when the chunk already starts with a mark.
    GuardedAlways,
    /// Metaspace `first`: like `GuardedAlways`, first chunk of the input only.
    GuardedFirst,
}

/// One step of a tokenizer.json `normalizer` sequence, applied in order to
/// each text chunk between added tokens (HF normalizes those independently).
pub enum NormOp {
    /// `Prepend`: unconditional prefix, e.g. Llama 2's "▁". HF's Prepend
    /// leaves empty chunks empty.
    Prepend(String),
    /// `Replace` with a literal `String` pattern.
    Replace { pattern: String, content: String },
    /// `Replace` with the `" {2,}"` regex (transformers' SpmConverter emits it
    /// for `remove_extra_whitespaces`): each run of 2+ ASCII spaces becomes
    /// `content`.
    CollapseSpaces { content: String },
    /// `Strip` Unicode whitespace.
    Strip { left: bool, right: bool },
    /// `Precompiled` charsmap (sentencepiece's nmt_nfkc and friends).
    Precompiled(PrecompiledCharsmap),
}

/// A precompiled charsmap plus lookup tables that let ASCII-dominant text
/// skip the per-grapheme trie walk (which runs at ~50 MB/s).
pub struct PrecompiledCharsmap {
    pre: spm_precompiled::Precompiled,
    /// `transform` of each single ASCII char; `None` = pass through.
    ascii_map: [Option<Box<str>>; 128],
    /// The "\r\n" grapheme's mapping (`None` = pass through).
    crlf: Option<Box<str>>,
    /// True when no printable ASCII char is remapped, so the SIMD clean-run
    /// scan only has to stop on control bytes and non-ASCII.
    fast_scan: bool,
}

impl PrecompiledCharsmap {
    pub(crate) fn new(pre: spm_precompiled::Precompiled) -> Self {
        let mut ascii_map: [Option<Box<str>>; 128] = std::array::from_fn(|_| None);
        let mut fast_scan = true;
        let mut buf = [0u8; 4];
        for b in 0..128u8 {
            let ch = b as char;
            let s = ch.encode_utf8(&mut buf);
            if let Some(norm) = pre.transform(s)
                && norm != s
            {
                if (0x20..0x7F).contains(&b) {
                    // A remapped printable would make clean runs unsound.
                    fast_scan = false;
                }
                ascii_map[b as usize] = Some(norm.into());
            }
        }
        let crlf = pre.transform("\r\n").map(Into::into);
        PrecompiledCharsmap {
            pre,
            ascii_map,
            crlf,
            fast_scan,
        }
    }

    /// Exactly `spm_precompiled::Precompiled::normalize_string`, but ASCII
    /// runs bypass the grapheme walk: printable ASCII chars are standalone
    /// grapheme clusters (only CR×LF joins, and only non-ASCII extends a
    /// cluster), and control chars break unconditionally on both sides, so
    /// the walk is only needed for spans around non-ASCII bytes — including
    /// one ASCII margin byte on each side, which non-ASCII prepend/combining
    /// characters can absorb into their cluster.
    pub(crate) fn normalize_into(&self, input: &str, out: &mut String) {
        use std::simd::prelude::*;

        if !self.fast_scan {
            out.push_str(&self.pre.normalize_string(input));
            return;
        }
        let bytes = input.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            // SIMD hop to the next attention byte (non-ASCII, control, DEL).
            let mut j = i;
            'scan: {
                while j + 16 <= bytes.len() {
                    let v = u8x16::from_slice(&bytes[j..]);
                    let attention = v.simd_ge(u8x16::splat(0x7F)) | v.simd_lt(u8x16::splat(0x20));
                    let m = attention.to_bitmask();
                    if m != 0 {
                        j += m.trailing_zeros() as usize;
                        break 'scan;
                    }
                    j += 16;
                }
                while j < bytes.len() && (0x20..0x7F).contains(&bytes[j]) {
                    j += 1;
                }
            }

            if j >= bytes.len() {
                out.push_str(&input[i..]);
                return;
            }
            let b = bytes[j];
            if b < 0x80 {
                // Control or DEL: a standalone grapheme except CR before LF.
                out.push_str(&input[i..j]);
                if b == b'\r' && bytes.get(j + 1) == Some(&b'\n') {
                    match &self.crlf {
                        Some(norm) => out.push_str(norm),
                        None => out.push_str("\r\n"),
                    }
                    i = j + 2;
                } else {
                    match &self.ascii_map[b as usize] {
                        Some(norm) => out.push_str(norm),
                        None => out.push(b as char),
                    }
                    i = j + 1;
                }
                continue;
            }
            // Non-ASCII span: pull in one preceding printable-ASCII byte (a
            // combining mark would extend its cluster), then extend until a
            // printable-ASCII byte whose successor is also ASCII (or a
            // control, which always breaks).
            let span_start = if j > i { j - 1 } else { j };
            out.push_str(&input[i..span_start]);
            let mut k = j;
            let span_end = loop {
                if k >= bytes.len() {
                    break bytes.len();
                }
                let c = bytes[k];
                if c < 0x20 || c == 0x7F {
                    break k; // control: hard break before it
                }
                if c < 0x80 && bytes.get(k + 1).is_none_or(|&n| n < 0x80) {
                    break k + 1; // ASCII with ASCII successor: safe cut after
                }
                k += 1;
            };
            out.push_str(&self.pre.normalize_string(&input[span_start..span_end]));
            i = span_end;
        }
    }
}

/// Position of a section (the text between added-token matches) within its
/// document, which decides whether the raw fast path ▁-prefixes the
/// section's first unit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SectionPos {
    /// The very start of the document (Metaspace `first` prepends only here).
    First,
    /// After an added-token match; per-section prepends still apply.
    Middle,
    /// Continues a section begun in an earlier fragment of a split document
    /// (see [`SentencePieceBPE::safe_fragment_ranges`]): never prefixed —
    /// the section's prefix, if any, was emitted with its first unit.
    Continuation,
}

/// When the Metaspace pre-tokenizer prepends ▁ to a chunk.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrependScheme {
    Never,
    Always,
    /// Only the chunk at the very start of the input; chunks after an added
    /// token don't count.
    First,
}

/// tokenizer.json `Metaspace` pre-tokenizer: replaces spaces with ▁, then
/// prepends ▁ per `prepend` (unless the chunk already starts with ▁), and —
/// with `split` — keeps BPE merges from crossing ▁ word boundaries.
pub struct Metaspace {
    pub prepend: PrependScheme,
    pub split: bool,
}

/// An added token, matched atomically in text before encoding.
pub struct AddedTokenSpec {
    /// What to find in the text. For `normalized` tokens this is the content
    /// after running it through the normalizer ops (HF matches those against
    /// normalized text).
    pub content: String,
    pub id: TokenId,
    /// Consume whitespace immediately before the match.
    pub lstrip: bool,
    /// Consume whitespace immediately after the match.
    pub rstrip: bool,
}

/// Leftmost-longest matcher over the added tokens' contents (pattern index
/// == `added_tokens` index) with a first-byte prefilter.
///
/// A plain Aho-Corasick `find_iter` visits every input byte (a DFA lookup
/// per byte), which measured as ~half of SentencePiece encode time even for
/// a three-token vocabulary: the scan costs more than the BPE it guards.
/// Every match must begin with one of the contents' first bytes, so a SIMD
/// scan for those bytes (plus a check of the second byte against the
/// contents' second bytes) finds the only positions where a match can start,
/// and the automaton runs *anchored* at just those. Anchored leftmost-longest
/// at the leftmost viable start equals the unanchored leftmost-longest match
/// (an Aho-Corasick match has no lookbehind), so the match stream is
/// identical to `find_iter`'s.
pub(crate) struct AddedMatcher {
    ac: aho_corasick::AhoCorasick,
    /// Distinct first bytes of all contents; empty = no prefilter (a
    /// content is empty, or there are more than [`Self::MAX_FIRST`] first
    /// bytes, so candidates would be too dense to help).
    first: Vec<u8>,
    /// `second[b]`: some content of length >= 2 has `b` as its second byte,
    /// or some content has length 1 (then any follower may complete a
    /// match, so every entry is true).
    second: [bool; 256],
}

impl AddedMatcher {
    const MAX_FIRST: usize = 8;

    fn new<'a>(contents: impl Iterator<Item = &'a [u8]> + Clone) -> Self {
        let ac = aho_corasick::AhoCorasick::builder()
            .match_kind(aho_corasick::MatchKind::LeftmostLongest)
            // Anchored searches are issued at prefilter candidates.
            .start_kind(aho_corasick::StartKind::Both)
            // A DFA's one lookup per byte beats the contiguous NFA when
            // the added vocabulary is large (gemma-3 carries 6.4k added
            // tokens). Prefix sharing (<unusedNNNN>…) keeps the table small.
            .kind(Some(aho_corasick::AhoCorasickKind::DFA))
            .build(contents.clone())
            .expect("added-token automaton");
        let mut first: Vec<u8> = Vec::new();
        let mut second = [false; 256];
        let mut usable = true;
        for content in contents {
            match content {
                [] => usable = false,
                [b] => {
                    if !first.contains(b) {
                        first.push(*b);
                    }
                    second = [true; 256];
                }
                [b, c, ..] => {
                    if !first.contains(b) {
                        first.push(*b);
                    }
                    second[*c as usize] = true;
                }
            }
        }
        if !usable || first.len() > Self::MAX_FIRST {
            first.clear();
        }
        AddedMatcher { ac, first, second }
    }

    /// The leftmost-longest match starting at or after `from`, as
    /// `(start, end, pattern)`.
    fn find_from(&self, hay: &[u8], from: usize) -> Option<(usize, usize, usize)> {
        use aho_corasick::{Anchored, Input};
        if self.first.is_empty() {
            let m = self.ac.find(Input::new(hay).span(from..hay.len()))?;
            return Some((m.start(), m.end(), m.pattern().as_usize()));
        }
        let mut pos = from;
        loop {
            let p = pos + self.next_first(&hay[pos..])?;
            // A content of length 1 is the only way a match ends at the
            // haystack's last byte without a second byte.
            let second_ok = hay.get(p + 1).is_none_or(|&c| self.second[c as usize]);
            if second_ok {
                let input = Input::new(hay).span(p..hay.len()).anchored(Anchored::Yes);
                if let Some(m) = self.ac.find(input) {
                    return Some((m.start(), m.end(), m.pattern().as_usize()));
                }
            }
            pos = p + 1;
        }
    }

    /// Offset of the first byte of `hay` that can start a match.
    #[inline]
    fn next_first(&self, hay: &[u8]) -> Option<usize> {
        match self.first.as_slice() {
            [a] => memchr::memchr(*a, hay),
            [a, b] => memchr::memchr2(*a, *b, hay),
            [a, b, c] => memchr::memchr3(*a, *b, *c, hay),
            set => {
                use std::simd::prelude::*;
                let mut i = 0;
                while i + 16 <= hay.len() {
                    let v = u8x16::from_slice(&hay[i..]);
                    let mut m = v.simd_eq(u8x16::splat(set[0]));
                    for &b in &set[1..] {
                        m |= v.simd_eq(u8x16::splat(b));
                    }
                    let bits = m.to_bitmask();
                    if bits != 0 {
                        return Some(i + bits.trailing_zeros() as usize);
                    }
                    i += 16;
                }
                hay[i..].iter().position(|b| set.contains(b)).map(|j| i + j)
            }
        }
    }
}

/// A tokenizer that mirrors SentencePiece BPE with `byte_fallback`.
///
/// This struct holds the immutable model data (vocab, merges, added tokens,
/// normalizer configuration). For encoding, create an [`Encoder`] which
/// processes text through the normalize → character init → BPE merge pipeline.
pub struct SentencePieceBPE {
    /// Merges with explicit rank, keyed by [`crate::bpe::ranked_merge_key`]:
    /// `key(a, b) → (merged, rank)`.
    pub(crate) merges: HashMap<u64, (TokenId, u32), FxBuildHasher>,
    /// Flat copy of `merges` for the encode path, built by
    /// `finalize_speed_paths`: a second copy of the merge table (16 B
    /// slots at <= 1/2 load), resident for the model's lifetime and not
    /// part of any cache budget.
    pub(crate) pair_table: SpPairTable,
    pub(crate) vocab: Vec<Arc<[u8]>>,
    /// Maps byte sequences → token IDs (for character lookup).
    pub(crate) vocab_inv: HashMap<Arc<[u8]>, TokenId, FxBuildHasher>,
    /// Token ID for each byte value (0x00–0xFF) via `<0xHH>` fallback tokens.
    /// `None` when the vocab lacks that byte's token (e.g. Gemma has literal
    /// `\t` pieces instead of `<0x09>`); such bytes can then only appear in
    /// text as chars the vocab covers, so the fallback is never consulted.
    pub(crate) byte_fallback_ids: [Option<TokenId>; 256],
    /// Added tokens with `normalized: false`, matched in the raw input.
    pub(crate) added_tokens: Vec<AddedTokenSpec>,
    /// Added tokens with `normalized: true`, matched (with pre-normalized
    /// content) against normalizer output, before the Metaspace step.
    pub(crate) norm_added_tokens: Vec<AddedTokenSpec>,
    /// The tokenizer.json normalizer sequence, applied per chunk in order.
    pub(crate) norm_ops: Vec<NormOp>,
    /// The Metaspace pre-tokenizer, if the tokenizer.json has one. `None`
    /// (e.g. Llama 2) means spaces are already handled by `norm_ops` and
    /// merges may cross word boundaries.
    pub(crate) metaspace: Option<Metaspace>,
    /// Unit-splitting mode; see [`WordSplit`]. Set by `finalize_speed_paths`.
    pub(crate) word_split: WordSplit,
    /// `Some` when the whole normalizer pipeline reduces to
    /// optional-▁-prepend + space→▁, so encoding can split raw text directly
    /// without materializing a normalized string. Set by
    /// `finalize_speed_paths`.
    pub(crate) raw_prepend: Option<RawPrepend>,
    /// Initial symbol(s) for the ▁ marker (its vocab piece, or its UTF-8
    /// bytes through byte fallback). Set by `finalize_speed_paths`.
    pub(crate) space_init: Vec<TokenId>,
    /// Initial symbol for each ASCII char: its single-char vocab piece or its
    /// byte-fallback token, skipping the per-char `vocab_inv` probe on the
    /// merge path. `None` = the byte has no token at all. Set by
    /// `finalize_speed_paths`.
    pub(crate) ascii_init: [Option<TokenId>; 128],
    /// Leftmost-longest automaton over `added_tokens` contents (pattern index
    /// == `added_tokens` index), like the byte-level path's added matcher —
    /// repeated `str::find` per token costs ~8% of encode on long inputs.
    /// Set by `finalize_speed_paths`.
    pub(crate) added_matcher: Option<AddedMatcher>,
    /// Frequent ASCII punctuation to split units *before* (0 = unused slot):
    /// units like "▁word," otherwise explode the distinct-unit space (and
    /// the pretoken cache) with word×punctuation combinations. Set by
    /// `finalize_speed_paths`.
    pub(crate) split_bytes: [u8; NUM_SPLIT_BYTES],
    /// Indexed by the split byte, a bitset over the previous byte for when
    /// the split is safe — exactly the predecessors that never appear
    /// immediately before that byte inside any vocab piece, so no merge can
    /// span the boundary. Empty (all zeros) for non-split bytes.
    pub(crate) split_safe: Vec<[u64; 4]>,
    /// Decompositions of the vocab pieces that can cross a `▁▁▁word` unit
    /// boundary, as `(pre, post)` around one interior ▁ — e.g. gemma's
    /// `>▁</` yields `(">", "</")`. Under `SpaceRuns` the scanner skips a
    /// split exactly where such a piece occurs spanning the candidate
    /// boundary (a merge result is always a contiguous vocab piece, so
    /// every other boundary is provably safe). Set by
    /// `finalize_speed_paths`; empty under `EveryMark`/`None`.
    pub(crate) cross_pieces: Vec<(Box<[u8]>, Box<[u8]>)>,
    /// Bitset over the byte just before a candidate boundary (the last byte
    /// of some `cross_pieces` pre) for a one-load rejection in the scanner.
    pub(crate) cross_prev: [u64; 4],
    /// Cache budget snapshot by each [`Self::encoder`]; `None` = unbounded.
    pub(crate) max_cache_bytes: Option<usize>,
}

/// How many distinct punctuation bytes the unit splitter checks for.
pub(crate) const NUM_SPLIT_BYTES: usize = 8;

impl std::fmt::Debug for SentencePieceBPE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SentencePieceBPE {{ vocab_size: {}, merges_count: {} }}",
            self.vocab.len(),
            self.merges.len(),
        )
    }
}

impl SentencePieceBPE {
    /// Apply the normalizer ops and Metaspace replacement/prepend to one text
    /// chunk. `first_chunk` is true only for the chunk at the very start of
    /// the input (the Metaspace "first" prepend scheme needs it).
    pub fn normalize<'a>(&self, input: &'a str, first_chunk: bool) -> Cow<'a, str> {
        self.apply_metaspace(self.apply_norm_ops(input), first_chunk)
    }

    /// The tokenizer.json normalizer sequence only (no Metaspace step).
    pub(crate) fn apply_norm_ops<'a>(&self, input: &'a str) -> Cow<'a, str> {
        let mut s: Cow<'a, str> = Cow::Borrowed(input);
        for op in &self.norm_ops {
            match op {
                NormOp::Prepend(prefix) => {
                    if !s.is_empty() {
                        let mut out = String::with_capacity(prefix.len() + s.len());
                        out.push_str(prefix);
                        out.push_str(&s);
                        s = Cow::Owned(out);
                    }
                }
                NormOp::Replace { pattern, content } => {
                    if s.contains(pattern.as_str()) {
                        let replaced = s.replace(pattern.as_str(), content);
                        s = Cow::Owned(replaced);
                    }
                }
                NormOp::CollapseSpaces { content } => {
                    if s.contains("  ") {
                        s = Cow::Owned(collapse_space_runs(&s, content));
                    }
                }
                NormOp::Strip { left, right } => {
                    let trimmed = match (left, right) {
                        (true, true) => s.trim(),
                        (true, false) => s.trim_start(),
                        (false, true) => s.trim_end(),
                        (false, false) => &s,
                    };
                    if trimmed.len() != s.len() {
                        s = Cow::Owned(trimmed.to_string());
                    }
                }
                NormOp::Precompiled(charsmap) => {
                    let mut out = String::with_capacity(s.len() + 16);
                    charsmap.normalize_into(&s, &mut out);
                    s = Cow::Owned(out);
                }
            }
        }
        s
    }

    /// The Metaspace pre-tokenizer's space replacement and ▁ prepend.
    pub(crate) fn apply_metaspace<'a>(
        &self,
        input: Cow<'a, str>,
        first_chunk: bool,
    ) -> Cow<'a, str> {
        let mut s = input;
        if let Some(ms) = &self.metaspace {
            if s.contains(' ') {
                let replaced: String = s
                    .chars()
                    .map(|c| if c == ' ' { SENTENCEPIECE_SPACE } else { c })
                    .collect();
                s = Cow::Owned(replaced);
            }
            let prepend = match ms.prepend {
                PrependScheme::Always => true,
                PrependScheme::First => first_chunk,
                PrependScheme::Never => false,
            };
            if prepend && !s.is_empty() && !s.starts_with(SENTENCEPIECE_SPACE) {
                let mut out = String::with_capacity(SENTENCEPIECE_SPACE_STR.len() + s.len());
                out.push(SENTENCEPIECE_SPACE);
                out.push_str(&s);
                s = Cow::Owned(out);
            }
        }
        s
    }

    /// Whether encoding puts a ▁ in front of ordinary text — decode then
    /// strips the resulting leading space, like HF's decoder does.
    fn prepends_space(&self) -> bool {
        self.metaspace
            .as_ref()
            .is_some_and(|ms| ms.prepend != PrependScheme::Never)
            || self
                .norm_ops
                .iter()
                .any(|op| matches!(op, NormOp::Prepend(_)))
    }

    /// Compute the encode fast-path configuration (`word_split`,
    /// `raw_prepend`, `space_init`) from the assembled model. Must be called
    /// once by the loader after all other fields are final.
    pub(crate) fn finalize_speed_paths(&mut self) {
        self.pair_table = SpPairTable::build(&self.merges);
        self.space_init = match self.vocab_inv.get(SP_MARK.as_slice()) {
            Some(&id) => vec![id],
            None => SP_MARK
                .iter()
                .filter_map(|&b| self.byte_fallback_ids[b as usize])
                .collect(),
        };

        for b in 0u8..128 {
            self.ascii_init[b as usize] = self
                .vocab_inv
                .get([b].as_slice())
                .copied()
                .or(self.byte_fallback_ids[b as usize]);
        }

        self.added_matcher = (!self.added_tokens.is_empty()).then(|| {
            AddedMatcher::new(self.added_tokens.iter().map(|t| t.content.as_bytes()))
        });

        self.cross_pieces = Vec::new();
        self.cross_prev = [0u64; 4];
        self.word_split = if self.metaspace.as_ref().is_some_and(|ms| ms.split) {
            WordSplit::EveryMark
        } else if let Some(cross) = self.unit_crossing_pieces() {
            for (pre, _) in &cross {
                let b = *pre.last().expect("interior ▁ always has a predecessor");
                self.cross_prev[(b >> 6) as usize] |= 1 << (b & 63);
            }
            self.cross_pieces = cross;
            WordSplit::SpaceRuns
        } else {
            WordSplit::None
        };

        self.raw_prepend = self.compute_raw_prepend();

        // Interior byte adjacency across all vocab pieces: splitting a unit
        // between bytes (x, y) is safe exactly when no piece contains x
        // immediately before y (a merge's result is always a vocab piece, so
        // no merge can then span the boundary; initial symbols are single
        // chars and cannot either).
        self.split_safe = vec![[0u64; 4]; 256];
        if self.word_split != WordSplit::None {
            let mut adjacent = vec![[0u64; 4]; 256]; // adjacent[x] bitset over y
            for piece in &self.vocab {
                for pair in piece.windows(2) {
                    adjacent[pair[0] as usize][(pair[1] >> 6) as usize] |= 1 << (pair[1] & 63);
                }
            }
            // The most frequent English punctuation, best value per SIMD
            // compare in the scanner.
            for (slot, &b) in [b'.', b',', b'"', b')', b';', b':', b'!', b'?']
                .iter()
                .enumerate()
            {
                let mut safe = [u64::MAX; 4];
                for x in 0..256usize {
                    if adjacent[x][(b >> 6) as usize] & (1 << (b & 63)) != 0 {
                        safe[x >> 6] &= !(1 << (x & 63));
                    }
                }
                // Only worth a compare if splitting is ever allowed. A byte
                // whose split is never safe keeps slot 0 (never matches).
                if safe != [0u64; 4] {
                    self.split_bytes[slot] = b;
                    self.split_safe[b as usize] = safe;
                }
            }
        }
    }

    /// The vocab pieces that can cross a `▁▁▁word`-shaped unit boundary — a
    /// unit starts at each ▁ following a non-▁ char, so a piece crosses
    /// exactly when it contains such an interior ▁ — decomposed as
    /// `(pre, post)` around that ▁. `Some(vec![])` means units are
    /// unconditionally safe (per-unit BPE equals global BPE); a non-empty
    /// list means safe with the scanner's occurrence guard; `None` means a
    /// crossing piece is too complex for the guard's byte-compare (a ▁ or a
    /// raw space inside `pre`/`post`, whose raw-mode text form varies) or
    /// there are too many, so whole-chunk merging must stay.
    fn unit_crossing_pieces(&self) -> Option<Vec<(Box<[u8]>, Box<[u8]>)>> {
        // Beyond this the per-boundary guard stops being "a handful of
        // memcmps on a rare prev byte" and whole-chunk merging is safer.
        const MAX_CROSS_PIECES: usize = 32;
        let mut out: Vec<(Box<[u8]>, Box<[u8]>)> = Vec::new();
        for piece in &self.vocab {
            let Ok(s) = std::str::from_utf8(piece) else {
                // Byte-fallback pieces are single bytes; they can't hold a ▁.
                continue;
            };
            let mut prev_is_mark = true; // leading ▁s are fine
            for (pos, c) in s.char_indices() {
                if c == SENTENCEPIECE_SPACE {
                    if !prev_is_mark {
                        // Interior ▁: the piece spans from `pre`'s unit into
                        // the one starting at this mark.
                        let (pre, rest) = s.split_at(pos);
                        let post = &rest[SP_MARK.len()..];
                        if pre.contains(SENTENCEPIECE_SPACE)
                            || post.contains(SENTENCEPIECE_SPACE)
                            || pre.contains(' ')
                            || post.contains(' ')
                            || out.len() == MAX_CROSS_PIECES
                        {
                            return None;
                        }
                        out.push((pre.as_bytes().into(), post.as_bytes().into()));
                    }
                    prev_is_mark = true;
                } else {
                    prev_is_mark = false;
                }
            }
        }
        Some(out)
    }

    /// Does some crossing vocab piece occur spanning the candidate unit
    /// boundary at `mark_pos` (mark of `width` bytes)? Only such an
    /// occurrence can let a merge cross the boundary; everywhere else the
    /// split is exact.
    #[inline(always)]
    fn piece_spans_boundary(&self, bytes: &[u8], mark_pos: usize, width: usize) -> bool {
        let prev = bytes[mark_pos - 1];
        if self.cross_prev[(prev >> 6) as usize] & (1 << (prev & 63)) == 0 {
            return false;
        }
        let after = &bytes[mark_pos + width..];
        self.cross_pieces
            .iter()
            .any(|(pre, post)| bytes[..mark_pos].ends_with(pre) && after.starts_with(post))
    }

    /// Whether an oversized document can be split into fragments that encode
    /// independently with token-identical output (see
    /// [`Self::safe_fragment_ranges`]). Requires the raw fast path: its
    /// per-unit encoding is what makes a provable unit boundary a safe cut.
    /// The materialized-normalizer path applies whole-section ops (Strip
    /// trims section edges, charsmaps span graphemes, Prepend re-fires per
    /// section), so no interior cut has a local safety argument there — and
    /// `WordSplit::None` merges whole chunks, so nothing can be split at all.
    pub(crate) fn supports_fragment_split(&self) -> bool {
        self.raw_prepend.is_some()
    }

    /// Split raw text into ranges of roughly `target` bytes at cuts the raw
    /// scanner provably treats as unit boundaries, so encoding the first
    /// range with `encode_raw_cb` and each later one with
    /// `encode_raw_fragment_cb` and concatenating the token streams is
    /// identical to encoding `text` in one call. Only valid when
    /// [`Self::supports_fragment_split`] holds.
    ///
    /// A cut at `p` is safe when the serial scan is guaranteed to close a
    /// unit exactly at `p` and every decision it makes is local to one side:
    ///
    /// - `p` starts a mark — a raw space, or a complete ▁ (0xE2 always
    ///   leads a 3-byte char in valid UTF-8, so neither test can misread a
    ///   continuation byte) — and, under `SpaceRuns`, does not extend a
    ///   mark run (the char before it is not a mark) and no crossing vocab
    ///   piece occurrence spans it (`piece_spans_boundary`). The scanner's
    ///   remaining decisions cannot reach across `p`: a crossing piece's
    ///   `pre`/`post` contain no space or ▁ (see `unit_crossing_pieces`),
    ///   so no occurrence of one can span the mark at `p`; mark-run state
    ///   resets at every non-mark char; and the punctuation split test
    ///   reads one preceding byte.
    /// - No added-token occurrence blocks `p` (see
    ///   `added_token_cut_blocks`): leftmost-longest matching carries no
    ///   state across an occurrence-free point, so restarting it at `p`
    ///   reproduces the single-pass matches, and lstrip/rstrip whitespace
    ///   trimming stays within one fragment.
    ///
    /// Where no safe cut exists the scan keeps probing forward, so a range
    /// can exceed `target` (in the worst case it is the rest of the
    /// document — output stays exact, only parallelism degrades).
    pub(crate) fn safe_fragment_ranges(
        &self,
        text: &str,
        target: usize,
    ) -> Vec<std::ops::Range<usize>> {
        debug_assert!(self.supports_fragment_split());
        let bytes = text.as_bytes();
        let len = bytes.len();
        if len == 0 {
            // Zero ranges would drop the document's output row.
            return vec![0..0];
        }
        let blocks = self.added_token_cut_blocks(text);
        let mut bi = 0usize; // cursor into `blocks`; probes are monotone
        let every_mark = self.word_split == WordSplit::EveryMark;
        let target = target.max(1);
        let mut out = Vec::new();
        let mut start = 0usize;
        'chunks: while start < len {
            let mut probe = start + target;
            while probe < len {
                let Some(width) = mark_width(bytes, probe, true) else {
                    probe += 1;
                    continue;
                };
                let extends_run = !every_mark
                    && (bytes[probe - 1] == b' '
                        || (probe >= 3 && bytes[probe - 3..probe] == SP_MARK));
                if extends_run || self.piece_spans_boundary(bytes, probe, width) {
                    probe += 1;
                    continue;
                }
                while bi < blocks.len() && blocks[bi].end <= probe {
                    bi += 1;
                }
                if bi < blocks.len() && blocks[bi].start <= probe {
                    // Inside a blocked interval: hop past it (the merged
                    // intervals are disjoint and sorted).
                    probe = blocks[bi].end;
                    continue;
                }
                out.push(start..probe);
                start = probe;
                continue 'chunks;
            }
            out.push(start..len);
            break;
        }
        out
    }

    /// Sorted, disjoint byte intervals in which no fragment cut may land,
    /// derived from added-token occurrences. Every occurrence is collected,
    /// overlapping ones included — a superset of the matches an encode
    /// selects, which is what makes the blocking conservative and exact:
    ///
    /// - inside an occurrence `[s, e)`: the cut halves would BPE-encode as
    ///   plain text (and truncation could change leftmost-longest choices);
    /// - at `e` itself, under `Unguarded` prepend only: the fragment's
    ///   continuation section would go un-prefixed where the serial encode
    ///   ▁-prefixes the post-match section it opens at `e`. (Guarded
    ///   schemes skip the prefix on the cut's own mark either way, and both
    ///   sides then scan the identical section from `e`.);
    /// - `lstrip`: back through the whitespace run before `s`, whose
    ///   trimming would otherwise reach into the previous fragment;
    /// - `rstrip`: forward through the whitespace run after `e`, which the
    ///   serial match consumes.
    ///
    /// A cut only ever lands on a mark start (a space or ▁'s 0xE2 lead), so
    /// an occurrence's interior can hold one only when its content carries
    /// such a byte. A token without one, without lstrip/rstrip, on a
    /// non-`Unguarded` model therefore cannot block anything and is skipped
    /// without a scan — gemma-3 carries 6.4k added tokens, and one pass per
    /// token over a multi-hundred-MB document is seconds, not milliseconds.
    ///
    /// The surviving tokens' scans are independent and run on the rayon
    /// pool: gemma keeps 30 ▁-run tokens, and their ~9 GB of memmem over a
    /// 290 MB document was ~35% of the whole parallel encode when run
    /// serially. Only the parallel encode entry points fragment documents,
    /// so touching rayon here is safe (the fork-safe `_serial` paths never
    /// reach this).
    fn added_token_cut_blocks(&self, text: &str) -> Vec<std::ops::Range<usize>> {
        use rayon::prelude::*;
        let bytes = text.as_bytes();
        let unguarded = self.raw_prepend == Some(RawPrepend::Unguarded);
        let mut blocks: Vec<std::ops::Range<usize>> = self
            .added_tokens
            .par_iter()
            .filter(|spec| {
                let content = spec.content.as_bytes();
                let interior = content.contains(&b' ') || content.contains(&0xE2);
                !content.is_empty() && (interior || unguarded || spec.lstrip || spec.rstrip)
            })
            .flat_map_iter(|spec| {
                let content = spec.content.as_bytes();
                let finder = memchr::memmem::Finder::new(content);
                let mut out = Vec::new();
                let mut from = 0usize;
                while let Some(off) = finder.find(&bytes[from..]) {
                    let s = from + off;
                    let e = s + content.len();
                    // A valid-UTF-8 needle only matches at char boundaries
                    // (its first byte is a lead byte, and a full match ends
                    // a complete char), so slicing `text` at `s`/`e` cannot
                    // panic.
                    let lo = if spec.lstrip {
                        text[..s].trim_end().len()
                    } else {
                        s + 1
                    };
                    let hi = if spec.rstrip {
                        e + (text[e..].len() - text[e..].trim_start().len())
                    } else if unguarded {
                        e
                    } else {
                        e - 1 // only the interior (s, e) blocks; `e` is safe
                    };
                    if lo <= hi {
                        out.push(lo..hi + 1);
                    }
                    from = s + 1; // step one byte: overlapping occurrences too
                }
                out
            })
            .collect();
        blocks.sort_unstable_by_key(|r| r.start);
        blocks.dedup_by(|next, prev| {
            if next.start <= prev.end {
                prev.end = prev.end.max(next.end);
                true
            } else {
                false
            }
        });
        blocks
    }

    /// Raw-fast-path eligibility: the normalizer sequence must reduce to
    /// optional ▁ prepend + literal space→▁, with no normalized added tokens
    /// (those are matched against materialized normalizer output).
    fn compute_raw_prepend(&self) -> Option<RawPrepend> {
        if self.word_split == WordSplit::None || !self.norm_added_tokens.is_empty() {
            return None;
        }
        let is_space_replace = |op: &NormOp| {
            matches!(op, NormOp::Replace { pattern, content }
                if pattern == " " && content == SENTENCEPIECE_SPACE_STR)
        };
        let from_metaspace = match self.metaspace.as_ref().map(|ms| ms.prepend) {
            None | Some(PrependScheme::Never) => RawPrepend::Never,
            Some(PrependScheme::Always) => RawPrepend::GuardedAlways,
            Some(PrependScheme::First) => RawPrepend::GuardedFirst,
        };
        match self.norm_ops.as_slice() {
            [NormOp::Prepend(p), op] if p == SENTENCEPIECE_SPACE_STR && is_space_replace(op) => {
                // Under EveryMark the unguarded prepend would fuse a "▁" that
                // HF splits off the first unit; take the materialized path.
                (self.word_split != WordSplit::EveryMark).then_some(RawPrepend::Unguarded)
            }
            [op] if is_space_replace(op) => Some(from_metaspace),
            [] if self.metaspace.is_some() => Some(from_metaspace),
            _ => None,
        }
    }

    /// Create an encoder for this model.
    pub fn encoder(&self) -> Encoder<'_> {
        Encoder {
            model: self,
            state: EncodeState::with_budget(self.max_cache_bytes),
        }
    }

    /// SentencePiece analog of [`super::Tokenizer::set_max_cache_bytes`],
    /// same default and same per-worker meaning: bounds the growing caches
    /// of every [`EncodeState`] created afterward (a parallel call holds
    /// one per worker), which wipe back to empty (there is no vocab seed
    /// here). A state's fixed footprint (`FIXED_STATE_BYTES`, 24 MiB: front
    /// cache and scratch) counts against the budget, so smaller budgets are
    /// floored to it. Outside the budget: the shared unit table of a call
    /// over a large input (32-128 MiB, one per call) and the model's flat
    /// `pair_table`.
    pub fn set_max_cache_bytes(&mut self, budget: Option<usize>) {
        self.max_cache_bytes = budget;
    }

    /// The configured cache budget in bytes; `None` when unbounded.
    pub fn max_cache_bytes(&self) -> Option<usize> {
        self.max_cache_bytes
    }

    /// Size of the vocabulary: one greater than the largest token ID,
    /// including added tokens (IDs with no assigned content count too).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Vocabulary entries as `(id, bytes)` pairs in ID order, including
    /// added tokens and skipping IDs with no assigned content.
    pub fn vocab_entries(&self) -> impl Iterator<Item = (u32, &[u8])> {
        super::vocab_entries(&self.vocab)
    }

    /// Merge rules as `(left, right)` byte pairs in rank order.
    pub fn merge_entries(&self) -> Vec<(&[u8], &[u8])> {
        let mut ranked: Vec<_> = self.merges.iter().collect();
        ranked.sort_unstable_by_key(|&(_, &(_, rank))| rank);
        ranked
            .into_iter()
            .map(|(&key, _)| {
                (
                    self.vocab[(key >> 32) as usize].as_ref(),
                    self.vocab[(key & u32::MAX as u64) as usize].as_ref(),
                )
            })
            .collect()
    }

    /// Convenience: encode a single text (creates a temporary encoder).
    pub fn encode_raw(&self, input: &str) -> Vec<TokenId> {
        self.encoder().encode_raw(input)
    }

    /// Decode token IDs back to a UTF-8 string.
    pub fn decode(&self, tokens: &[TokenId]) -> Vec<u8> {
        let mut raw = Vec::new();
        for &t in tokens {
            let idx: usize = t.into();
            if idx < self.vocab.len() {
                raw.extend_from_slice(&self.vocab[idx]);
            }
        }
        let text = String::from_utf8_lossy(&raw);
        let mut out: Vec<u8> = text.replace(SENTENCEPIECE_SPACE, " ").into_bytes();
        if self.prepends_space() && out.first() == Some(&b' ') {
            out.remove(0);
        }
        out
    }
}

/// Replace each run of 2 or more ASCII spaces with `content`, like HF's
/// `Replace(Regex(" {2,}"), content)`.
fn collapse_space_runs(input: &str, content: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b' ' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] == b' ' {
                j += 1;
            }
            if j - i >= 2 {
                out.push_str(content);
            } else {
                out.push(' ');
            }
            i = j;
        } else {
            let start = i;
            while i < bytes.len() && bytes[i] != b' ' {
                i += 1;
            }
            out.push_str(&input[start..i]);
        }
    }
    out
}

/// log2 of the direct-mapped front-cache size in entries: 2^19 entries of
/// 32 B is 16 MiB per state. Large enough to keep the Zipf-hot working set
/// resident despite eviction churn from tail units; 2^20 measured slightly
/// faster but costs another 16 MiB in every worker.
const FRONT_BITS: u32 = 19;

/// Part of a state's cache budget reserved for its fixed footprint: the
/// 16 MiB front cache plus headroom for the scratch buffers. The growing
/// caches (short/long maps and arena) get what the budget leaves, so a
/// budget `B` bounds them at `B - FIXED_STATE_BYTES`; budgets below
/// `FIXED_STATE_BYTES + 1 MiB` are floored to it.
const FIXED_STATE_BYTES: usize = 24 << 20;
const _: () = assert!(FIXED_STATE_BYTES > (1 << FRONT_BITS) * std::mem::size_of::<FrontEntry>());

/// Estimated bytes per map entry, payload + SwissTable slack (`long` adds
/// its key bytes on top). Length-based, because the wipe keeps allocations.
const SHORT_ENTRY_BYTES: usize = 48;
const LONG_ENTRY_BYTES: usize = 48;

/// Index of `key` in the front cache: multiplicative hash, top bits.
#[inline(always)]
fn front_index(key: u128) -> usize {
    (key_hash(key) >> (64 - FRONT_BITS)) as usize
}

/// One front-cache entry: the key and the unit's tokens side by side, so a
/// hit touches one cache line — no second array, no arena. Key 0 = empty.
///
/// `p` holds up to four tokens inline (unused trailing lanes `u32::MAX`);
/// encodings of 0 or >4 tokens are recorded as an arena reference
/// `[offset, len, 0, ARENA_TAG]` instead (see [`front_payload`]).
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct FrontEntry {
    key: u128,
    p: [u32; 4],
}

/// Last payload word of an arena-referencing front entry. Never a token ID
/// (vocabularies are far smaller) and never `u32::MAX` (the unused-lane fill).
const ARENA_TAG: u32 = u32::MAX - 1;

/// Front-entry payload for the `len` tokens at `arena[off..]`.
#[inline(always)]
fn front_payload(arena: &[TokenId], off: u32, len: u32) -> [u32; 4] {
    if (1..=4).contains(&len) {
        let mut p = [u32::MAX; 4];
        for (dst, src) in p.iter_mut().zip(&arena[off as usize..(off + len) as usize]) {
            *dst = src.0;
        }
        p
    } else {
        [off, len, 0, ARENA_TAG]
    }
}

/// Emit the tokens a front payload stands for.
#[inline(always)]
fn emit_payload<S: TokenSink>(sink: &mut S, p: &[u32; 4], arena: &[TokenId]) {
    if p[3] != ARENA_TAG {
        let count = 1
            + (p[1] != u32::MAX) as usize
            + (p[2] != u32::MAX) as usize
            + (p[3] != u32::MAX) as usize;
        // SAFETY: `p` is 16 readable bytes (four TokenIds, a transparent
        // u32), the first `count` of which are the run.
        unsafe { sink.emit_short(p.as_ptr() as *const TokenId, count) }
    } else {
        emit_run(sink, arena, p[0] as usize, p[1] as usize);
    }
}

/// Direct-mapped table of `2^FRONT_BITS` [`FrontEntry`]s on a 64-byte
/// boundary (two entries per line), zeroed lazily by the allocator.
struct FrontTable {
    _storage: Vec<u64>,
    base: *mut FrontEntry,
}

// SAFETY: the table owns its storage; the raw pointer only ever aliases it.
unsafe impl Send for FrontTable {}
// SAFETY: reads go through `&self` only; mutation needs `&mut self`.
unsafe impl Sync for FrontTable {}

impl FrontTable {
    fn new() -> Self {
        let bytes = std::mem::size_of::<FrontEntry>() << FRONT_BITS;
        let storage = vec![0u64; (bytes + 64) / 8];
        let aligned = (storage.as_ptr() as usize + 63) & !63;
        // The front cache is probed at random for every unit: with 4 KiB
        // pages that is a dTLB miss + page walk per probe. Ask for 2 MiB
        // pages before the first touch (calloc'd, still untouched).
        super::madvise_hugepage(aligned as *mut u8, bytes);
        FrontTable {
            _storage: storage,
            base: aligned as *mut FrontEntry,
        }
    }

    /// Entry `idx`, which must be `< 2^FRONT_BITS` (every `front_index`).
    #[inline(always)]
    fn at(&self, idx: usize) -> &FrontEntry {
        debug_assert!(idx < 1 << FRONT_BITS);
        // SAFETY: in bounds per the contract; the zeroed memory is a valid
        // FrontEntry and `base` is 64-aligned.
        unsafe { &*self.base.add(idx) }
    }

    #[inline(always)]
    fn set(&mut self, idx: usize, key: u128, p: [u32; 4]) {
        debug_assert!(idx < 1 << FRONT_BITS);
        // SAFETY: as in `at`, with exclusive access through `&mut self`.
        unsafe {
            *self.base.add(idx) = FrontEntry { key, p };
        }
    }

    #[inline(always)]
    fn ptr(&self, idx: usize) -> *const u8 {
        self.base.wrapping_add(idx) as *const u8
    }

    /// Forget every entry (their offsets name a truncated arena).
    fn clear(&mut self) {
        // SAFETY: zeroing the whole table; all-zero is the empty entry.
        unsafe { std::ptr::write_bytes(self.base, 0, 1 << FRONT_BITS) };
    }
}

/// Per-thread mutable encoding context: the pretoken cache plus scratch
/// buffers, mirroring the byte-level BPE path's memoization. Units repeat
/// heavily in natural text, so the ranked merge only runs on cache misses.
pub struct EncodeState {
    /// Append-only arena of encoded token IDs; cache entries are
    /// `(offset, len)` slices into it.
    arena: Vec<TokenId>,
    /// Direct-mapped front cache in front of `short`: each 32-byte entry
    /// holds a unit's key and tokens, so a probe touches one cache line.
    /// Zipf-hot units resolve here with one compare instead of a SwissTable
    /// probe into a couple-hundred-MB map. Key 0 = empty (packed keys
    /// always carry a nonzero length tag).
    front: FrontTable,
    /// Cache for units of ≤ 15 key bytes (the overwhelming majority), keyed
    /// by the same packed `u128` scheme as the byte-level path.
    short: HashMap<u128, (u32, u32), FxBuildHasher>,
    /// Fallback cache for longer units.
    long: HashMap<Box<[u8]>, (u32, u32), FxBuildHasher>,
    /// Scratch for the merge loop.
    symbols: Vec<TokenId>,
    /// Long-sequence merge buffers.
    merge_scratch: SpMergeScratch,
    /// Scratch for composing keys with a virtual leading space.
    key_buf: Vec<u8>,
    /// Byte budget for the growing caches ([`Self::with_budget`]);
    /// `usize::MAX` = unbounded.
    budget: usize,
    /// Estimated `long` bytes: key bytes + [`LONG_ENTRY_BYTES`] per entry.
    long_bytes_used: usize,
    /// Cross-thread table of short-unit encodings (set by the worker pool);
    /// `None` for a standalone state, which keeps everything in `short`.
    shared: Option<Arc<SharedUnits>>,
    /// Largest single encoding, as wipe-trigger slack: one recurring giant
    /// unit must not force a wipe per occurrence. Kept across wipes.
    max_encoding: usize,
    /// Bumped by every [`Self::wipe`]: arena offsets resolved before a wipe
    /// (a batched unit lookup, see [`UnitBatch`]) dangle after it.
    epoch: u32,
}

impl EncodeState {
    pub fn new() -> Self {
        Self::with_budget(Some(super::Tokenizer::DEFAULT_MAX_CACHE_BYTES))
    }

    /// A state whose growing caches (short/long maps + arena) wipe back to
    /// empty when their estimated footprint exceeds `max_bytes` minus
    /// `FIXED_STATE_BYTES` (24 MiB: front cache and scratch), floored at
    /// 1 MiB so tiny budgets stay functional. `None` = unbounded.
    pub fn with_budget(max_bytes: Option<usize>) -> Self {
        EncodeState {
            arena: Vec::new(),
            front: FrontTable::new(),
            short: HashMap::with_hasher(FxBuildHasher),
            long: HashMap::with_hasher(FxBuildHasher),
            symbols: Vec::new(),
            merge_scratch: SpMergeScratch::default(),
            key_buf: Vec::new(),
            budget: max_bytes.map_or(usize::MAX, |t| {
                t.saturating_sub(FIXED_STATE_BYTES).max(1 << 20)
            }),
            long_bytes_used: 0,
            shared: None,
            max_encoding: 0,
            epoch: 0,
        }
    }

    /// Number of cached units (for diagnostics).
    pub fn cache_size(&self) -> usize {
        self.short.len() + self.long.len()
    }

    /// Number of cache wipes so far.
    #[cfg(test)]
    pub(crate) fn wipes(&self) -> u32 {
        self.epoch
    }

    /// Drop `symbols` and `key_buf` when one giant unit (100 MB of CJK or
    /// base64 without a mark) grew them past [`SCRATCH_RETAIN_BYTES`], so
    /// that capacity does not stay resident for the state's lifetime.
    #[inline]
    fn release_oversized_scratch(&mut self) {
        if self.symbols.capacity() * std::mem::size_of::<TokenId>() > SCRATCH_RETAIN_BYTES {
            self.symbols = Vec::new();
        }
        if self.key_buf.capacity() > SCRATCH_RETAIN_BYTES {
            self.key_buf = Vec::new();
        }
    }

    #[inline]
    fn over_budget(&self) -> bool {
        self.short.len() * SHORT_ENTRY_BYTES + self.long_bytes_used + self.arena.len() * 4
            > self.budget.saturating_add(self.max_encoding * 4)
    }

    /// Wipe every cache back to empty, keeping allocations. The front keys
    /// must go too: their values are slices of the truncated arena.
    #[cold]
    #[inline(never)]
    fn wipe(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.arena.clear();
        self.front.clear();
        self.short.clear();
        self.long.clear();
        self.long_bytes_used = 0;
    }
}

impl Default for EncodeState {
    fn default() -> Self {
        Self::new()
    }
}

/// An encoder that holds a reference to the model plus its own cache and
/// scratch. Create one per thread for parallel encoding.
pub struct Encoder<'a> {
    model: &'a SentencePieceBPE,
    state: EncodeState,
}

/// Leftmost match of any spec's content in `text`; ties go to the longest
/// content, like HF's aho-corasick LeftmostLongest added-token matching.
fn find_added_token<'s>(
    specs: &'s [AddedTokenSpec],
    text: &str,
) -> Option<(usize, &'s AddedTokenSpec)> {
    let mut best: Option<(usize, &AddedTokenSpec)> = None;
    for spec in specs {
        if let Some(pos) = text.find(spec.content.as_str()) {
            let better = match best {
                None => true,
                Some((best_pos, best_spec)) => {
                    pos < best_pos
                        || (pos == best_pos && spec.content.len() > best_spec.content.len())
                }
            };
            if better {
                best = Some((pos, spec));
            }
        }
    }
    best
}

impl<'a> Encoder<'a> {
    /// An encoder over `model` that resumes `state`, a worker's cache kept
    /// from earlier chunks of the same call (see `batch::SpCallPool`).
    pub(crate) fn from_state(model: &'a SentencePieceBPE, state: EncodeState) -> Self {
        Encoder { model, state }
    }

    /// Back this encoder's short-unit cache with the call's shared table
    /// (a state keeps the table it was given).
    pub(crate) fn ensure_shared(&mut self, shared: impl FnOnce() -> Arc<SharedUnits>) {
        if self.state.shared.is_none() {
            self.state.shared = Some(shared());
        }
    }

    /// Give the cache state back to the call's pool.
    pub(crate) fn into_state(self) -> EncodeState {
        self.state
    }

    /// Encode raw (un-normalized) text with added-token splitting.
    pub fn encode_raw(&mut self, input: &str) -> Vec<TokenId> {
        let mut result = Vec::new();
        self.model
            .encode_raw_with(&mut self.state, input, &mut result);
        result
    }

    /// Like [`Self::encode_raw`], emitting token runs through `f`.
    pub fn encode_raw_cb<F: FnMut(&[TokenId])>(&mut self, input: &str, f: &mut F) {
        self.model.encode_raw_cb(&mut self.state, input, f);
    }

    /// [`Self::encode_raw_cb`] for a [`TokenSink`] (fast-path emission).
    pub(crate) fn encode_raw_sink<S: TokenSink>(&mut self, input: &str, sink: &mut S) {
        self.model
            .encode_raw_from(&mut self.state, input, SectionPos::First, sink);
    }

    /// [`Self::encode_raw_fragment_cb`] for a [`TokenSink`].
    pub(crate) fn encode_raw_fragment_sink<S: TokenSink>(
        &mut self,
        input: &str,
        first: bool,
        sink: &mut S,
    ) {
        let pos = if first {
            SectionPos::First
        } else {
            SectionPos::Continuation
        };
        self.model.encode_raw_from(&mut self.state, input, pos, sink);
    }

    /// Like [`Self::encode_raw_cb`] for one fragment of a document split at
    /// scanner-safe boundaries (see
    /// [`SentencePieceBPE::safe_fragment_ranges`]). A non-first fragment's
    /// leading section continues one begun in an earlier fragment, so it is
    /// never ▁-prefixed.
    pub fn encode_raw_fragment_cb<F: FnMut(&[TokenId])>(
        &mut self,
        input: &str,
        first: bool,
        f: &mut F,
    ) {
        let pos = if first {
            SectionPos::First
        } else {
            SectionPos::Continuation
        };
        self.model.encode_raw_from(&mut self.state, input, pos, f);
    }
}

/// Is the byte at `pos` the start of a unit mark? In raw mode both a space
/// and a literal ▁ count; in normalized mode only ▁ does. Returns the mark's
/// byte width.
#[inline(always)]
fn mark_width(bytes: &[u8], pos: usize, raw: bool) -> Option<usize> {
    match bytes[pos] {
        b' ' if raw => Some(1),
        0xE2 if bytes.len() - pos >= 3 && bytes[pos + 1] == 0x96 && bytes[pos + 2] == 0x81 => {
            Some(3)
        }
        _ => None,
    }
}

/// Destination of encoded token runs. Closures `FnMut(&[TokenId])` are
/// sinks; [`VecSink`] additionally takes short runs branch-free.
pub trait TokenSink {
    fn emit(&mut self, tokens: &[TokenId]);

    /// Emit the `count` (<= 4) tokens at `tokens`.
    ///
    /// # Safety
    /// `tokens` must point at 4 readable, initialized `TokenId`s, of which
    /// the first `count` are the run.
    #[inline(always)]
    unsafe fn emit_short(&mut self, tokens: *const TokenId, count: usize) {
        // SAFETY: the caller guarantees `count` readable tokens.
        self.emit(unsafe { std::slice::from_raw_parts(tokens, count) });
    }
}

impl<F: FnMut(&[TokenId])> TokenSink for F {
    #[inline(always)]
    fn emit(&mut self, tokens: &[TokenId]) {
        self(tokens)
    }
}

/// Appends token ids to a `Vec<u32>`.
pub struct VecSink<'a>(pub &'a mut Vec<u32>);

impl TokenSink for VecSink<'_> {
    #[inline(always)]
    fn emit(&mut self, tokens: &[TokenId]) {
        self.0.extend(tokens.iter().map(|&t| u32::from(t)));
    }

    /// One fixed-size 16-byte copy plus a length bump: no length-dependent
    /// branch or loop, which mispredicted on the 1-4 token runs of nearly
    /// every unit.
    #[inline(always)]
    unsafe fn emit_short(&mut self, tokens: *const TokenId, count: usize) {
        let v = &mut *self.0;
        if v.capacity() - v.len() < 4 {
            v.reserve(4);
        }
        let len = v.len();
        // SAFETY: 4 tokens are readable at `tokens` (caller), 4 u32 slots of
        // spare capacity are writable at `len` (reserved above), and the
        // regions cannot overlap (a token source is never the output Vec).
        unsafe {
            std::ptr::copy_nonoverlapping(tokens as *const u8, v.as_mut_ptr().add(len) as *mut u8, 16);
            v.set_len(len + count);
        }
    }
}

/// Tokens appended after every arena encoding, so a run of <= 4 tokens can
/// always be read as one 16-byte block ([`emit_run`]).
const ARENA_PAD: [TokenId; 3] = [TokenId(0); 3];

/// Emit `arena[start..start + len]`; every arena run is followed by
/// [`ARENA_PAD`], so short runs may read 4 tokens.
#[inline(always)]
fn emit_run<S: TokenSink>(sink: &mut S, arena: &[TokenId], start: usize, len: usize) {
    debug_assert!(start + len <= arena.len());
    if (1..=4).contains(&len) {
        // SAFETY: the run plus its 3-token pad lies in the arena, so with
        // len >= 1 the 4 tokens from `start` are initialized and in bounds.
        unsafe { sink.emit_short(arena.as_ptr().add(start), len) }
    } else {
        // SAFETY: caller's recorded (start, len) is within the arena.
        sink.emit(unsafe { arena.get_unchecked(start..start + len) });
    }
}

/// Units queued per [`SentencePieceBPE::flush_units`] call: enough
/// independent cache lookups in flight to cover a DRAM round trip (a
/// core has ~10-16 line-fill buffers), few enough that the queue stays in
/// registers/L1.
const UNIT_BATCH: usize = 32;

/// Pending word units of one `encode_units_impl` walk: byte range, packed
/// cache key (0 = no short key; unit keys are never 0 because units are
/// non-empty) and front-cache index.
struct UnitBatch {
    start: [usize; UNIT_BATCH],
    end: [usize; UNIT_BATCH],
    key: [u128; UNIT_BATCH],
    idx: [usize; UNIT_BATCH],
    n: usize,
}

impl UnitBatch {
    #[inline(always)]
    fn new() -> Self {
        UnitBatch {
            start: [0; UNIT_BATCH],
            end: [0; UNIT_BATCH],
            key: [0; UNIT_BATCH],
            idx: [0; UNIT_BATCH],
            n: 0,
        }
    }

    /// Queue `bytes[start..end]` and prefetch its front-cache entry.
    #[inline(always)]
    fn push(&mut self, state: &EncodeState, bytes: &[u8], start: usize, end: usize) {
        let n = self.n;
        // SAFETY: the walker passes start <= end <= bytes.len().
        let (key, idx) = match pack_pretoken_key(unsafe { bytes.get_unchecked(start..end) }) {
            Some(key) if key != 0 => {
                let idx = front_index(key);
                prefetch_read(state.front.ptr(idx));
                (key, idx)
            }
            _ => (0, 0),
        };
        self.start[n] = start;
        self.end[n] = end;
        self.key[n] = key;
        self.idx[n] = idx;
        self.n = n + 1;
    }
}

/// Request the cache line holding `p` into L1; no memory effects, any
/// address is allowed. No-op on arches without a prefetch hint.
#[inline(always)]
pub(crate) fn prefetch_read<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetch has no memory effects; any address is allowed.
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: prefetch has no memory effects; any address is allowed.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{p}]",
            p = in(reg) p,
            options(nostack, preserves_flags, readonly)
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = p;
}

impl SentencePieceBPE {
    /// Encode raw (un-normalized) text into a token buffer. See
    /// [`Self::encode_raw_cb`].
    pub fn encode_raw_with(&self, state: &mut EncodeState, input: &str, out: &mut Vec<TokenId>) {
        self.encode_raw_cb(state, input, &mut |tokens: &[TokenId]| {
            out.extend_from_slice(tokens)
        });
    }

    /// Encode raw (un-normalized) text with added-token splitting: first the
    /// raw-matched (`normalized: false`) tokens, then — per remaining section —
    /// the normalizer ops, the normalized-matched tokens, and Metaspace + BPE.
    /// Emits token runs through `f`, like the byte-level path's
    /// `memoized_encode`.
    pub fn encode_raw_cb<F: FnMut(&[TokenId])>(
        &self,
        state: &mut EncodeState,
        input: &str,
        f: &mut F,
    ) {
        self.encode_raw_from(state, input, SectionPos::First, f);
    }

    /// [`Self::encode_raw_cb`] with the position of the input's leading
    /// section given explicitly: `First` for a whole document,
    /// `Continuation` for a non-first fragment of one (see
    /// [`Self::safe_fragment_ranges`]). Sections after an added-token match
    /// are always `Middle`.
    fn encode_raw_from<F: TokenSink>(
        &self,
        state: &mut EncodeState,
        input: &str,
        start: SectionPos,
        f: &mut F,
    ) {
        let Some(matcher) = &self.added_matcher else {
            self.encode_section_cb(state, input, start, f);
            return;
        };

        // Iterate leftmost-longest added-token matches; the text between
        // consecutive matches forms the chunks. lstrip/rstrip consume the
        // whitespace adjacent to a match.
        let mut chunk_start = 0usize;
        let mut pos = start;
        let hay = input.as_bytes();
        let mut scan_from = 0usize;
        while let Some((m_start, m_end, pattern)) = matcher.find_from(hay, scan_from) {
            // Like `find_iter`, resume after the match itself — not after
            // any rstrip-consumed whitespace.
            scan_from = m_end;
            let spec = &self.added_tokens[pattern];
            if m_start < chunk_start {
                // Swallowed by the previous match's rstrip.
                continue;
            }
            let mut chunk = &input[chunk_start..m_start];
            if spec.lstrip {
                chunk = chunk.trim_end();
            }
            if !chunk.is_empty() {
                self.encode_section_cb(state, chunk, pos, f);
            }
            f.emit(&[spec.id]);
            chunk_start = m_end;
            if spec.rstrip {
                chunk_start += input[chunk_start..].len() - input[chunk_start..].trim_start().len();
            }
            pos = SectionPos::Middle;
        }
        let chunk = &input[chunk_start..];
        if !chunk.is_empty() {
            self.encode_section_cb(state, chunk, pos, f);
        }
    }

    /// Encode one raw section: the raw fast path when eligible, otherwise
    /// normalizer ops → normalized added-token splitting → Metaspace → BPE.
    /// HF does not re-normalize the parts around a normalized added-token
    /// match.
    fn encode_section_cb<F: TokenSink>(
        &self,
        state: &mut EncodeState,
        text: &str,
        pos: SectionPos,
        f: &mut F,
    ) {
        if let Some(prepend) = self.raw_prepend {
            self.encode_chunk_raw(state, text, pos, prepend, f);
            return;
        }

        // Fragment splitting is gated on the raw fast path
        // (`supports_fragment_split`), so a continuation never reaches the
        // materialized-normalizer path below — its whole-section ops
        // (Strip, per-section Prepend, ...) have no continuation form.
        debug_assert!(pos != SectionPos::Continuation);
        let first_chunk = pos == SectionPos::First;
        let normed = self.apply_norm_ops(text);

        if self.norm_added_tokens.is_empty() {
            let final_text = self.apply_metaspace(normed, first_chunk);
            self.encode_normalized_sink(state, &final_text, f);
            return;
        }

        let mut remaining: &str = &normed;
        let mut first = first_chunk;
        while !remaining.is_empty() {
            match find_added_token(&self.norm_added_tokens, remaining) {
                Some((pos, spec)) => {
                    let part = if spec.lstrip {
                        remaining[..pos].trim_end()
                    } else {
                        &remaining[..pos]
                    };
                    if !part.is_empty() {
                        let final_text = self.apply_metaspace(Cow::Borrowed(part), first);
                        self.encode_normalized_sink(state, &final_text, f);
                    }
                    f.emit(&[spec.id]);
                    let mut rest = &remaining[pos + spec.content.len()..];
                    if spec.rstrip {
                        rest = rest.trim_start();
                    }
                    remaining = rest;
                    first = false;
                }
                None => {
                    let final_text = self.apply_metaspace(Cow::Borrowed(remaining), first);
                    self.encode_normalized_sink(state, &final_text, f);
                    break;
                }
            }
        }
    }

    /// Raw fast path: split un-normalized text into units directly, mapping
    /// spaces to ▁ on the fly. The dummy-prefix ▁ becomes a virtual leading
    /// space on the first unit, which keys and encodes identically.
    fn encode_chunk_raw<F: TokenSink>(
        &self,
        state: &mut EncodeState,
        chunk: &str,
        pos: SectionPos,
        prepend: RawPrepend,
        f: &mut F,
    ) {
        if chunk.is_empty() {
            return;
        }
        let bytes = chunk.as_bytes();
        let starts_with_mark = mark_width(bytes, 0, true).is_some();
        let virtual_prefix = match prepend {
            // A continuation resumes a section mid-way: its prefix, if any,
            // was emitted with the section's first unit in an earlier
            // fragment.
            _ if pos == SectionPos::Continuation => false,
            RawPrepend::Unguarded => true,
            RawPrepend::GuardedAlways => !starts_with_mark,
            RawPrepend::GuardedFirst => pos == SectionPos::First && !starts_with_mark,
            RawPrepend::Never => false,
        };
        self.encode_units(state, bytes, virtual_prefix, true, f);
    }

    /// Encode already-normalized text: unit split (per `word_split`) with the
    /// pretoken cache, or a whole-chunk merge when units aren't safe.
    pub fn encode_normalized_cb<F: FnMut(&[TokenId])>(
        &self,
        state: &mut EncodeState,
        input: &str,
        f: &mut F,
    ) {
        self.encode_normalized_sink(state, input, f);
    }

    /// [`Self::encode_normalized_cb`] for any [`TokenSink`].
    fn encode_normalized_sink<F: TokenSink>(
        &self,
        state: &mut EncodeState,
        input: &str,
        f: &mut F,
    ) {
        match self.word_split {
            WordSplit::None => self.bpe_chunk(state, input, f),
            _ => self.encode_units(state, input.as_bytes(), false, false, f),
        }
    }

    /// Split a chunk into word units and encode each through the cache.
    ///
    /// `raw` selects raw-mode marks (space or ▁) vs normalized-mode marks
    /// (▁ only); `virtual_prefix` logically prepends one space to the first
    /// unit (the raw fast path's dummy prefix).
    fn encode_units<F: TokenSink>(
        &self,
        state: &mut EncodeState,
        bytes: &[u8],
        virtual_prefix: bool,
        raw: bool,
        f: &mut F,
    ) {
        #[cfg(target_arch = "x86_64")]
        if crate::pretokenize::fast::mask::avx2_scanner_available() {
            // SAFETY: the AVX2/BMI tier was just detected.
            unsafe {
                if raw {
                    self.encode_units_avx2::<true, F>(state, bytes, virtual_prefix, f);
                } else {
                    self.encode_units_avx2::<false, F>(state, bytes, virtual_prefix, f);
                }
            }
            return;
        }
        if raw {
            self.encode_units_plain::<true, F>(state, bytes, virtual_prefix, f);
        } else {
            self.encode_units_plain::<false, F>(state, bytes, virtual_prefix, f);
        }
    }

    /// [`Self::encode_units_impl`] compiled for the baseline target.
    #[inline(never)]
    fn encode_units_plain<const RAW: bool, F: TokenSink>(
        &self,
        state: &mut EncodeState,
        bytes: &[u8],
        virtual_prefix: bool,
        f: &mut F,
    ) {
        self.encode_units_impl::<RAW, false, F>(state, bytes, virtual_prefix, f);
    }

    /// [`Self::encode_units_impl`] compiled with AVX2 + the BMI bit
    /// features (tzcnt/blsr/shlx/shrx in the unit walk and key packing).
    ///
    /// SAFETY: the caller verified `avx2_scanner_available`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
    #[inline(never)]
    unsafe fn encode_units_avx2<const RAW: bool, F: TokenSink>(
        &self,
        state: &mut EncodeState,
        bytes: &[u8],
        virtual_prefix: bool,
        f: &mut F,
    ) {
        self.encode_units_impl::<RAW, true, F>(state, bytes, virtual_prefix, f);
    }

    /// Walk 32-byte SIMD blocks whose mark-candidate bitmask is drained bit
    /// by bit, closing a unit at each qualifying mark. Candidates are dense
    /// in natural text (~1 per 5 bytes), so the block mask is kept in a
    /// register and the per-candidate hot path is a couple of bit ops.
    #[inline(always)]
    fn encode_units_impl<const RAW: bool, const AVX: bool, F: TokenSink>(
        &self,
        state: &mut EncodeState,
        bytes: &[u8],
        virtual_prefix: bool,
        f: &mut F,
    ) {
        use std::simd::prelude::*;

        let every_mark = self.word_split == WordSplit::EveryMark;
        let mut unit_start = 0usize;
        let mut last_mark_end = usize::MAX;
        let mut first_unit = true;

        let splats = self.split_bytes.map(u8x16::splat);
        let mut batch = UnitBatch::new();
        // Emit `bytes[s..e]` as one unit: the first unit of a virtually
        // prefixed chunk goes straight through (its key is built in
        // `key_buf`); every other unit queues for a batched cache lookup.
        macro_rules! emit_unit {
            ($s:expr, $e:expr) => {{
                let (s, e) = ($s, $e);
                if virtual_prefix && first_unit {
                    self.encode_unit(state, &bytes[s..e], true, RAW, f);
                } else {
                    batch.push(state, bytes, s, e);
                    if batch.n == UNIT_BATCH {
                        self.flush_units_for::<AVX, F>(state, bytes, &mut batch, RAW, f);
                    }
                }
                first_unit = false;
            }};
        }

        let mut block = 0usize;
        while block < bytes.len() {
            let mut mask: u32;
            if block + 32 <= bytes.len() {
                let lo = u8x16::from_slice(&bytes[block..]);
                let hi = u8x16::from_slice(&bytes[block + 16..]);
                let mut m_lo = lo.simd_eq(u8x16::splat(0xE2));
                let mut m_hi = hi.simd_eq(u8x16::splat(0xE2));
                if RAW {
                    m_lo |= lo.simd_eq(u8x16::splat(b' '));
                    m_hi |= hi.simd_eq(u8x16::splat(b' '));
                }
                // Split-punct candidates (unused slots are 0x00 splats; NUL
                // bytes then take the punct path and split_safe[0] is empty).
                for splat in splats {
                    m_lo |= lo.simd_eq(splat);
                    m_hi |= hi.simd_eq(splat);
                }
                mask = (m_lo.to_bitmask() as u32) | ((m_hi.to_bitmask() as u32) << 16);
            } else {
                mask = 0;
                for (i, &b) in bytes[block..].iter().enumerate() {
                    if b == 0xE2 || (RAW && b == b' ') || self.split_bytes.contains(&b) {
                        mask |= 1 << i;
                    }
                }
            }

            while mask != 0 {
                let mark_pos = block + mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let byte = bytes[mark_pos];
                if (RAW && byte == b' ') || byte == 0xE2 {
                    // A candidate is only a mark if it's a space (raw mode)
                    // or a full ▁ (0xE2 also starts other three-byte chars;
                    // its continuation bytes are never candidates).
                    let Some(width) = mark_width(bytes, mark_pos, RAW) else {
                        continue;
                    };
                    // SpaceRuns: only a mark that doesn't extend a run
                    // starts a unit — and not where a crossing vocab piece
                    // spans the boundary (`cross_prev` is all-zero when
                    // there are none, so the guard is one load + test).
                    let boundary = every_mark || mark_pos != last_mark_end;
                    if boundary
                        && mark_pos != unit_start
                        && !self.piece_spans_boundary(bytes, mark_pos, width)
                    {
                        emit_unit!(unit_start, mark_pos);
                        unit_start = mark_pos;
                    }
                    last_mark_end = mark_pos + width;
                } else if mark_pos != unit_start {
                    // Split punctuation: a unit boundary only after a
                    // vocab-verified safe predecessor. A raw space acts as ▁,
                    // so check its final UTF-8 byte.
                    let mut prev = bytes[mark_pos - 1];
                    if RAW && prev == b' ' {
                        prev = SP_MARK[2];
                    }
                    let safe = &self.split_safe[byte as usize];
                    if safe[(prev >> 6) as usize] & (1 << (prev & 63)) != 0 {
                        emit_unit!(unit_start, mark_pos);
                        unit_start = mark_pos;
                    }
                }
            }
            block += 32;
        }
        // `unit_start` only ever advances to a mark position < len, so a
        // non-empty chunk always has a final unit.
        if unit_start < bytes.len() {
            emit_unit!(unit_start, bytes.len());
        }
        if batch.n != 0 {
            self.flush_units_for::<AVX, F>(state, bytes, &mut batch, RAW, f);
        }
    }

    /// Resolve and emit every queued unit, in order. Phase 1 touches only
    /// the front-cache slots (prefetched when the units were queued) and, for
    /// hits, requests the cached tokens' arena lines; phase 2 emits. The
    /// units' cache lookups are then in flight together instead of one
    /// dependent DRAM round trip per unit — the front table (16 MiB per
    /// thread) is far beyond L2 and the lookups are the unit loop's
    /// critical path. A unit that missed the front cache (or whose arena
    /// offsets a budget wipe invalidated mid-batch) takes the full lookup
    /// in phase 2, so output order and content are exactly the unbatched
    /// loop's.
    #[inline(always)]
    fn flush_units_for<const AVX: bool, F: TokenSink>(
        &self,
        state: &mut EncodeState,
        bytes: &[u8],
        batch: &mut UnitBatch,
        raw: bool,
        f: &mut F,
    ) {
        #[cfg(target_arch = "x86_64")]
        if AVX {
            // SAFETY: `AVX` is only instantiated by `encode_units_avx2`,
            // whose caller verified the AVX2/BMI tier.
            unsafe { self.flush_units_avx2(state, bytes, batch, raw, f) };
            return;
        }
        self.flush_units(state, bytes, batch, raw, f);
    }

    /// [`Self::flush_units_body`] for the baseline target.
    #[inline(never)]
    fn flush_units<F: TokenSink>(
        &self,
        state: &mut EncodeState,
        bytes: &[u8],
        batch: &mut UnitBatch,
        raw: bool,
        f: &mut F,
    ) {
        self.flush_units_body(state, bytes, batch, raw, f);
    }

    /// [`Self::flush_units_body`] with BMI/AVX2 codegen.
    ///
    /// SAFETY: the caller verified `avx2_scanner_available`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
    #[inline(never)]
    unsafe fn flush_units_avx2<F: TokenSink>(
        &self,
        state: &mut EncodeState,
        bytes: &[u8],
        batch: &mut UnitBatch,
        raw: bool,
        f: &mut F,
    ) {
        self.flush_units_body(state, bytes, batch, raw, f);
    }

    #[inline(always)]
    fn flush_units_body<F: TokenSink>(
        &self,
        state: &mut EncodeState,
        bytes: &[u8],
        batch: &mut UnitBatch,
        raw: bool,
        f: &mut F,
    ) {
        let n = batch.n;
        batch.n = 0;
        let epoch = state.epoch;
        let mut hits = 0u32;
        let mut pay = [[0u32; 4]; UNIT_BATCH];
        // Branch-free resolution of the front probes: hit/miss is data-random
        // (~20% misses), so a branch here mispredicts; the loads are
        // unconditional and the prefetch targets are selected, not branched.
        for i in 0..n {
            let key = batch.key[i];
            let idx = batch.idx[i];
            // SAFETY: `idx` came from `front_index`, which returns
            // FRONT_BITS bits (0 for key 0), and the front table has
            // 2^FRONT_BITS entries.
            let entry = state.front.at(idx);
            let (stored, p) = (entry.key, entry.p);
            let hit = key != 0 && stored == key;
            pay[i] = p;
            hits |= (hit as u32) << i;
            // Only two cases still need a line: a hit that refers to the
            // arena, and a miss (its shared-table slot). Selected, not
            // branched; an inline hit re-requests its own resident line.
            let target = if !hit {
                match &state.shared {
                    Some(shared) => shared.slot_ptr(key),
                    None => state.front.ptr(idx),
                }
            } else if p[3] == ARENA_TAG {
                state.arena.as_ptr().wrapping_add(p[0] as usize) as *const u8
            } else {
                state.front.ptr(idx)
            };
            prefetch_read(target);
        }
        for i in 0..n {
            if hits >> i & 1 != 0 && state.epoch == epoch {
                // An arena payload names tokens appended before the probe and
                // no wipe has run since (the epoch matches); an inline one is
                // self-contained.
                emit_payload(f, &pay[i], &state.arena);
            } else {
                let key = batch.key[i];
                self.encode_unit_keyed(
                    state,
                    &bytes[batch.start[i]..batch.end[i]],
                    (key != 0).then_some(key),
                    batch.idx[i],
                    false,
                    raw,
                    f,
                );
            }
        }
    }

    /// Encode one unit through the pretoken cache; run the ranked merge only
    /// on a miss. The cache key is the unit's bytes (raw or normalized —
    /// byte-equal keys always encode identically), with a `b' '` prefix for
    /// the virtual leading space.
    #[inline]
    fn encode_unit<F: TokenSink>(
        &self,
        state: &mut EncodeState,
        unit: &[u8],
        virtual_prefix: bool,
        raw: bool,
        f: &mut F,
    ) {
        let packed = if virtual_prefix {
            state.key_buf.clear();
            state.key_buf.push(b' ');
            state.key_buf.extend_from_slice(unit);
            pack_pretoken_key(&state.key_buf)
        } else {
            pack_pretoken_key(unit)
        };
        let front_idx = packed.map_or(0, front_index);
        self.encode_unit_keyed(state, unit, packed, front_idx, virtual_prefix, raw, f);
    }

    /// [`Self::encode_unit`] with the unit's packed key and front index
    /// already computed (`front_idx` is meaningful only for `Some` keys).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn encode_unit_keyed<F: TokenSink>(
        &self,
        state: &mut EncodeState,
        unit: &[u8],
        packed: Option<u128>,
        front_idx: usize,
        virtual_prefix: bool,
        raw: bool,
        f: &mut F,
    ) {
        // Front cache first: one L1/L2 compare resolves Zipf-hot units.
        if let Some(key) = packed {
            // SAFETY: the front table has 2^FRONT_BITS entries and
            // `front_index` returns FRONT_BITS bits.
            let entry = state.front.at(front_idx);
            if entry.key == key {
                let p = entry.p;
                // Arena references were recorded right after appending
                // their tokens; the arena only clears in a budget wipe,
                // which also zeroes every front key.
                emit_payload(f, &p, &state.arena);
                return;
            }
        }
        if let Some(key) = packed {
            let mut toks = [0u32; MAX_SHARED_TOKENS];
            let n = match &state.shared {
                Some(shared) => shared.get(key, &mut toks),
                None => 0,
            };
            if n != 0 {
                if state.over_budget() {
                    state.wipe();
                }
                let offset = state.arena.len();
                state.arena.extend(toks[..n].iter().map(|&t| TokenId(t)));
                state.arena.extend_from_slice(&ARENA_PAD);
                let p = front_payload(&state.arena, offset as u32, n as u32);
                state.front.set(front_idx, key, p);
                f.emit(&state.arena[offset..offset + n]);
                return;
            }
        }
        let cached = match packed {
            Some(key) => state.short.get(&key).copied(),
            None if virtual_prefix => state.long.get(state.key_buf.as_slice()).copied(),
            None => state.long.get(unit).copied(),
        };
        if let Some((offset, len)) = cached {
            if let Some(key) = packed {
                let p = front_payload(&state.arena, offset, len);
                state.front.set(front_idx, key, p);
            }
            let start = offset as usize;
            // SAFETY: every cached (offset, len) was recorded right after
            // appending those `len` tokens at `offset`; the arena only
            // clears in a budget wipe, which also clears both maps.
            emit_run(f, &state.arena, start, len as usize);
            return;
        }

        // Miss: budget check first (the lookups above already missed, and
        // stay misses against the freshly wiped caches), then character
        // init → ranked merge, then record in the arena.
        if state.over_budget() {
            state.wipe();
        }
        state.symbols.clear();
        if virtual_prefix {
            state.symbols.extend_from_slice(&self.space_init);
        }
        // SAFETY: units start at mark starts and end at mark starts or the
        // chunk end, all of which are char boundaries of the original &str.
        let unit_str = unsafe { std::str::from_utf8_unchecked(unit) };
        self.init_symbols(unit_str, raw, &mut state.symbols);
        merge_ranked(&self.pair_table, &mut state.symbols, &mut state.merge_scratch);

        let offset = state.arena.len() as u32;
        let len = state.symbols.len() as u32;
        state.arena.extend_from_slice(&state.symbols);
        state.arena.extend_from_slice(&ARENA_PAD);
        state.max_encoding = state.max_encoding.max(len as usize);
        match packed {
            Some(key) => {
                let published = (1..=MAX_SHARED_TOKENS).contains(&(len as usize))
                    && state
                        .shared
                        .as_ref()
                        .is_some_and(|shared| shared.insert(key, &state.symbols));
                if !published {
                    state.short.insert(key, (offset, len));
                }
                let p = front_payload(&state.arena, offset, len);
                state.front.set(front_idx, key, p);
            }
            None => {
                let key: Box<[u8]> = if virtual_prefix {
                    state.key_buf.as_slice().into()
                } else {
                    unit.into()
                };
                state.long_bytes_used += key.len() + LONG_ENTRY_BYTES;
                state.long.insert(key, (offset, len));
            }
        }
        f.emit(&state.symbols);
        state.release_oversized_scratch();
    }

    /// Whole-chunk merge without caching (vocabs with boundary-crossing
    /// pieces).
    fn bpe_chunk<F: TokenSink>(&self, state: &mut EncodeState, chunk: &str, f: &mut F) {
        state.symbols.clear();
        self.init_symbols(chunk, false, &mut state.symbols);
        merge_ranked(&self.pair_table, &mut state.symbols, &mut state.merge_scratch);
        f.emit(&state.symbols);
        state.release_oversized_scratch();
    }

    /// Character init: each char's vocab piece, or its UTF-8 bytes through
    /// byte fallback. In raw mode spaces initialize as the ▁ marker.
    #[inline]
    fn init_symbols(&self, text: &str, raw: bool, symbols: &mut Vec<TokenId>) {
        for ch in text.chars() {
            if raw && ch == ' ' {
                symbols.extend_from_slice(&self.space_init);
                continue;
            }
            if (ch as u32) < 128 {
                if let Some(id) = self.ascii_init[ch as usize] {
                    symbols.push(id);
                }
                continue;
            }
            let mut buf = [0u8; 4];
            let ch_bytes = ch.encode_utf8(&mut buf).as_bytes();
            if let Some(&id) = self.vocab_inv.get(ch_bytes) {
                symbols.push(id);
            } else {
                for &b in ch_bytes {
                    if let Some(id) = self.byte_fallback_ids[b as usize] {
                        symbols.push(id);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The prefiltered matcher must yield exactly `find_iter`'s match
    /// stream (leftmost-longest, non-overlapping), for vocabularies with
    /// length-1 contents, shared prefixes, a shared first byte, and many
    /// distinct first bytes (which disables the prefilter), over random
    /// haystacks dense in the tokens' alphabet.
    #[test]
    fn added_matcher_matches_find_iter() {
        let vocabs: Vec<Vec<&str>> = vec![
            vec!["<unk>", "<s>", "</s>"],
            vec!["<s>", "<", "ab"],
            vec!["\n", "\n\n", "▁▁", "<td>", "<t>"],
            vec!["a", "ab", "abc", "bcd", "cd", "x"],
            vec!["ab", "ba", "cd", "dc", "ef", "fe", "gh", "hg", "ij", "ji", "<>"],
            vec!["<a", "<b", "<c", "<d", "<e", "<f", "<g", "<h", "<i"],
        ];
        let alphabet: Vec<&str> = vec![
            "<", ">", "/", "s", "u", "n", "k", "a", "b", "c", "d", "x", "\n", "▁", " ", "t", "e",
            "f", "g", "h", "i", "j", "<td>", "<unk>", "</s>", "▁▁",
        ];
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for vocab in &vocabs {
            let contents = || vocab.iter().map(|t| t.as_bytes());
            let reference = aho_corasick::AhoCorasick::builder()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(contents())
                .unwrap();
            let matcher = AddedMatcher::new(contents());
            for round in 0..3000 {
                let len = (next() % if round % 10 == 0 { 400 } else { 24 }) as usize;
                let mut hay = String::new();
                for _ in 0..len {
                    hay.push_str(alphabet[(next() % alphabet.len() as u64) as usize]);
                }
                let expected: Vec<(usize, usize, usize)> = reference
                    .find_iter(hay.as_bytes())
                    .map(|m| (m.start(), m.end(), m.pattern().as_usize()))
                    .collect();
                let mut got = Vec::new();
                let mut from = 0;
                while let Some(m) = matcher.find_from(hay.as_bytes(), from) {
                    from = m.1;
                    got.push(m);
                }
                assert_eq!(got, expected, "vocab {vocab:?} haystack {hay:?}");
            }
        }
    }

    #[test]
    fn test_precompiled_fast_scan_matches_grapheme_walk() {
        // The SIMD clean-run scan must reproduce normalize_string exactly,
        // including CRLF pairs, control chars, combining marks that extend an
        // ASCII cluster, and non-ASCII spans with ASCII margins.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/fineweb_4096_bpe_tokenizer.json"
        );
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: {path} not found");
            return;
        }
        let tok = crate::load_tokenizer::hf::load_hf_sentencepiece(path).unwrap();
        let charsmap = tok
            .norm_ops
            .iter()
            .find_map(|op| match op {
                NormOp::Precompiled(c) => Some(c),
                _ => None,
            })
            .expect("fineweb model has a precompiled charsmap");
        let cases = [
            "plain ascii text",
            "tabs\tand\r\nnewlines\rmixed\x07controls\x00",
            "combining: a\u{301} e\u{308} at end x\u{300}",
            "ﬁligature ½ ㎒ Ⅷ ｆｕｌｌｗｉｄｔｈ",
            "mixed日本語and ascii, ünïcode wörds",
            "\u{200b}zero width start",
            "trailing non-ascii é",
            "é",
            "",
            "\r",
            "\r\n",
            "a\r\nb",
        ];
        for case in cases {
            let mut fast = String::new();
            charsmap.normalize_into(case, &mut fast);
            assert_eq!(
                fast,
                charsmap.pre.normalize_string(case),
                "fast path diverged for {case:?}"
            );
        }
    }

    #[test]
    fn test_collapse_space_runs() {
        assert_eq!(collapse_space_runs("a  b   c d  ", "▁"), "a▁b▁c d▁");
        assert_eq!(collapse_space_runs("a \t  b", "▁"), "a \t▁b");
        assert_eq!(collapse_space_runs("  ", "▁"), "▁");
        assert_eq!(collapse_space_runs("no runs", "▁"), "no runs");
    }

    /// PARITY + BOUNDS: a budgeted EncodeState that wipes mid-corpus must
    /// produce the exact token stream of an unbounded one, while the
    /// estimated footprint of the growing caches stays within the budget
    /// (plus the documented giant-encoding slack).
    #[test]
    fn budgeted_wipe_matches_unbounded() {
        let Some(path) = crate::test_hub::hf_tokenizer_json("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        else {
            eprintln!("Skipping: TinyLlama tokenizer.json not in the HF cache");
            return;
        };
        let model = crate::load_tokenizer::hf::load_hf_sentencepiece(&path).unwrap();
        assert_eq!(
            model.max_cache_bytes(),
            Some(super::super::Tokenizer::DEFAULT_MAX_CACHE_BYTES)
        );

        // ~100k distinct short words plus interleaved > 15-byte words (the
        // long-map path) and repeats of a common head.
        let mut text = String::new();
        let mut x = 0x1234_5678_9ABC_DEF0u64;
        let mut rand = |m: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % m
        };
        for i in 0..100_000u32 {
            let len = if i % 17 == 0 { 16 + rand(24) } else { 3 + rand(8) };
            for _ in 0..len {
                text.push((b'a' + rand(26) as u8) as char);
            }
            text.push(' ');
            if i % 5 == 0 {
                text.push_str("the quick brown fox ");
            }
        }

        let mut unbounded = EncodeState::with_budget(None);
        let mut expected = Vec::new();
        model.encode_raw_with(&mut unbounded, &text, &mut expected);

        // The fixed footprint plus 1 MiB: several wipes over ~100k distinct units.
        let mut budgeted = EncodeState::with_budget(Some(FIXED_STATE_BYTES + (1 << 20)));
        let mut actual = Vec::new();
        model.encode_raw_with(&mut budgeted, &text, &mut actual);
        assert_eq!(actual, expected, "budgeted output diverged");
        assert!(budgeted.long_bytes_used > 0, "corpus never hit the long map");
        // Both saw ~100k distinct units, so a small survivor count proves
        // wipes happened, and the estimate must end up within budget.
        assert!(
            budgeted.cache_size() < unbounded.cache_size() / 3,
            "survivors {} vs unbounded {}",
            budgeted.cache_size(),
            unbounded.cache_size()
        );
        let used = budgeted.short.len() * SHORT_ENTRY_BYTES
            + budgeted.long_bytes_used
            + budgeted.arena.len() * 4;
        let limit = budgeted.budget + budgeted.max_encoding * 4;
        assert!(used <= limit + 4096, "estimate {used} exceeds budget {limit}");
    }
}
