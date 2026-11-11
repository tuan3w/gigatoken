//! Parallel chunked batch encoding: the engine behind encode_batch and
//! encode_files. Documents are grouped into coarse chunks (an oversized
//! document is split internally — at pretoken-safe boundaries for BPE,
//! scanner-safe unit boundaries for SentencePiece), encoded by pooled
//! workers, and reassembled into one flat id buffer plus per-document row
//! lengths. BPE workers persist across calls so their pretoken caches stay
//! warm; SentencePiece encoder states live for the duration of one call
//! and are freed when it returns.

use crate::Tokenizer;
use crate::bpe;
use crate::bpe::madvise_hugepage;
use crate::input::DocumentIter;
use crate::input::file_source::{DocFormat, chunk_ranges};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, TryLockError};

/// Parallel chunks must hold at least this many bytes: a chunk this size
/// encodes for tens of milliseconds, so worker acquisition and rayon
/// scheduling/work-stealing overhead is noise. An input that does not fill
/// more than one chunk is encoded serially — for small inputs the thread
/// fan-out costs more than it saves.
const MIN_CHUNK_BYTES: usize = 1 << 20;

/// Spent id buffers totalling at least this many bytes are freed by a
/// background task after the gather returns (see [`drop_buffers`]);
/// smaller totals drop inline.
const DEFERRED_DROP_MIN_BYTES: usize = 32 << 20;

/// Spent chunk id buffers at least this large are kept for reuse by the
/// chunks still to start in the same call (see [`IdPool`]); smaller ones
/// are cheap to allocate and free.
const ID_POOL_MIN_BYTES: usize = 1 << 20;

/// Upper bound on the bytes of id buffers one call keeps parked in its
/// [`IdPool`].
const ID_POOL_MAX_BYTES: usize = 128 << 20;

/// Slack, in ids, on the 2x waste bound of [`id_buffer_fits`], so tiny
/// requests can still reuse a slightly larger buffer.
const ID_FIT_SLACK: usize = 4096;

/// Whether a pooled buffer of `capacity` ids can serve a request for `cap`:
/// big enough, without wasting more than 2x (plus [`ID_FIT_SLACK`]).
fn id_buffer_fits(capacity: usize, cap: usize) -> bool {
    capacity >= cap && capacity <= 2 * cap + ID_FIT_SLACK
}

/// Chunk id buffers recycled within ONE encode call. Every chunk allocates
/// a ~MB id buffer; once the gather has copied a chunk's ids out, the
/// buffer comes back here and the chunks still to start reuse its
/// already-faulted, cache-warm pages instead of paying fresh `mmap`
/// first-touch faults (and THP zeroing/compaction stalls under
/// `defrag=madvise`) on the full token output. The pool is a local of the
/// call, so nothing outlives it: [`run_gathered`] frees what is left once
/// the gather is done.
#[derive(Default)]
struct IdPool {
    parked: Mutex<ParkedIds>,
}

/// The buffers an [`IdPool`] holds and their total capacity in bytes.
#[derive(Default)]
struct ParkedIds {
    bufs: Vec<Vec<u32>>,
    bytes: usize,
}

impl IdPool {
    fn lock(&self) -> MutexGuard<'_, ParkedIds> {
        self.parked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// An empty id buffer with capacity for at least `cap` ids: the
    /// smallest pooled buffer that fits (see [`id_buffer_fits`]), else a
    /// fresh allocation hinted for huge pages.
    fn take(&self, cap: usize) -> Vec<u32> {
        if cap * std::mem::size_of::<u32>() >= ID_POOL_MIN_BYTES {
            let mut parked = self.lock();
            let best = parked
                .bufs
                .iter()
                .enumerate()
                .filter(|(_, v)| id_buffer_fits(v.capacity(), cap))
                .min_by_key(|(_, v)| v.capacity())
                .map(|(i, _)| i);
            if let Some(i) = best {
                let v = parked.bufs.swap_remove(i);
                parked.bytes -= v.capacity() * std::mem::size_of::<u32>();
                return v;
            }
        }
        let mut ids = Vec::with_capacity(cap);
        madvise_hugepage(ids.as_mut_ptr() as *mut u8, ids.capacity() * 4);
        ids
    }

    /// Park a spent id buffer; hands it back when it is too small to be
    /// worth keeping or the pool is full.
    fn recycle(&self, mut ids: Vec<u32>) -> Option<Vec<u32>> {
        let bytes = ids.capacity() * std::mem::size_of::<u32>();
        if bytes < ID_POOL_MIN_BYTES {
            return Some(ids);
        }
        let mut parked = self.lock();
        if parked.bytes + bytes > ID_POOL_MAX_BYTES {
            return Some(ids);
        }
        ids.clear();
        parked.bytes += bytes;
        parked.bufs.push(ids);
        None
    }

    /// Every buffer still parked, for [`drop_buffers`].
    fn into_buffers(self) -> Vec<Vec<u32>> {
        self.parked
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .bufs
    }
}

/// Target bytes per parallel chunk: ~16 chunks per thread for work-stealing
/// load balancing, floored at MIN_CHUNK_BYTES so chunks stay coarse.
pub(crate) fn chunk_target_bytes(total_bytes: usize) -> usize {
    (total_bytes / (16 * rayon::current_num_threads())).max(MIN_CHUNK_BYTES)
}

/// Append one document's token ids to `ids` and its row length to `lens`.
pub(crate) fn encode_into(tokenizer: &mut Tokenizer, doc: &[u8], ids: &mut Vec<u32>, lens: &mut Vec<i64>) {
    let before = ids.len();
    tokenizer.encode_with_added_tokens_flat(doc, ids);
    lens.push((ids.len() - before) as i64);
}

/// SentencePiece analog of `encode_into`, using `encoder`'s pretoken cache.
pub(crate) fn sp_encode_into(
    encoder: &mut bpe::sentencepiece::Encoder<'_>,
    text: &str,
    ids: &mut Vec<u32>,
    lens: &mut Vec<i64>,
) {
    let before = ids.len();
    encoder.encode_raw_sink(text, &mut bpe::sentencepiece::VecSink(ids));
    lens.push((ids.len() - before) as i64);
}

/// `sp_encode_into` for one fragment of a split document (see
/// `SpChunk::Fragment`): a non-first fragment continues a section begun in
/// an earlier fragment, so its leading section is never ▁-prefixed.
fn sp_encode_fragment_into(
    encoder: &mut bpe::sentencepiece::Encoder<'_>,
    text: &str,
    first: bool,
    ids: &mut Vec<u32>,
    lens: &mut Vec<i64>,
) {
    let before = ids.len();
    encoder.encode_raw_fragment_sink(text, first, &mut bpe::sentencepiece::VecSink(ids));
    lens.push((ids.len() - before) as i64);
}

/// Iterate the documents in a byte region per `format`: JSONL lines,
/// separator-delimited text, or the whole region as one document.
pub(crate) fn for_each_doc(bytes: &[u8], format: &DocFormat, mut f: impl FnMut(&[u8])) {
    use crate::input::jsonl::JsonLinesSlice;
    match format {
        DocFormat::Jsonl { field } => {
            for doc in JsonLinesSlice::new(bytes, field) {
                f(doc.as_ref());
            }
        }
        DocFormat::Text {
            separator: Some(sep),
        } if !sep.is_empty() => {
            for doc in DocumentIter::new(bytes, sep) {
                f(doc);
            }
        }
        DocFormat::Text { .. } => f(bytes),
        // Parquet rows are materialized into whole documents before encoding
        // (encode_files_ragged rewrites the format to Text { separator: None }),
        // so parquet bytes must never arrive here.
        DocFormat::Parquet { .. } => {
            unreachable!("parquet files are materialized into documents before encoding")
        }
    }
}

/// Work unit for parallel encoding.
pub(crate) enum EncodeChunk<'a> {
    /// A run of whole documents, one output row each.
    Docs(Vec<&'a [u8]>),
    /// A byte region holding many documents, split during encoding
    /// (JSONL lines or separator-delimited text).
    Region {
        bytes: &'a [u8],
        format: &'a DocFormat,
    },
    /// A pretoken-safe fragment of one oversized document (see
    /// `pretokenize::safe_split_ranges`). Fragments of a document are
    /// consecutive chunks; `first` marks the document's first fragment.
    Fragment { bytes: &'a [u8], first: bool },
}

/// Token output of one chunk: a flat id buffer plus one length per document
/// row. `continues` means the first length extends the previous chunk's
/// last row (a non-first fragment of a split document).
pub(crate) struct ChunkTokens {
    pub(crate) ids: Vec<u32>,
    pub(crate) lens: Vec<i64>,
    pub(crate) continues: bool,
}

/// Id capacity requested for a chunk's output: input bytes over the
/// bytes-per-token estimate documented in `encode_chunk`.
fn chunk_id_cap(chunk: &EncodeChunk) -> usize {
    let byte_len = match chunk {
        EncodeChunk::Docs(docs) => docs.iter().map(|d| d.len()).sum::<usize>(),
        EncodeChunk::Region { bytes, .. } | EncodeChunk::Fragment { bytes, .. } => bytes.len(),
    };
    byte_len / 4 + 16
}

fn encode_chunk(tokenizer: &mut Tokenizer, chunk: &EncodeChunk, pool: &IdPool) -> ChunkTokens {
    // Reserve the output once, from a bytes-per-token estimate on the low
    // side of natural language (~4.4 on OWT/GPT-2). Growing from empty
    // instead re-copies roughly the final size in doublings — per chunk,
    // on every chunk of a first pass.
    let mut ids = pool.take(chunk_id_cap(chunk));
    let mut lens = Vec::new();
    let mut continues = false;
    match chunk {
        EncodeChunk::Docs(docs) => {
            for doc in docs {
                encode_into(tokenizer, doc, &mut ids, &mut lens);
            }
        }
        EncodeChunk::Region { bytes, format } => {
            for_each_doc(bytes, format, |doc| {
                encode_into(tokenizer, doc, &mut ids, &mut lens)
            })
        }
        EncodeChunk::Fragment { bytes, first } => {
            encode_into(tokenizer, bytes, &mut ids, &mut lens);
            continues = !*first;
        }
    }
    ChunkTokens {
        ids,
        lens,
        continues,
    }
}

/// Whether LPT chunk sizing is enabled: killed by setting `GIGATOK_NO_LPT`
/// in the environment (to any value, empty included). Read once per encode
/// call — never in per-chunk or per-pretoken loops. Both shapes are token-
/// and order-identical; the switch changes chunk sizing only, and exists
/// so future measurement can flip it without a rebuild.
fn lpt_from_env() -> bool {
    std::env::var_os("GIGATOK_NO_LPT").is_none()
}

/// Group documents into parallel chunks of descending (LPT-scheduled)
/// sizes: ~2x-target chunks over the first ~80% of bytes, quarter-target
/// chunks over the last ~20%. Rayon hands out chunks in index order, so
/// whichever core draws the last chunk (on asymmetric parts, often an
/// E-core) strands the others behind a short tail instead of a full-size
/// one, while the big early chunks amortize per-chunk overhead. A document
/// larger than the target is split into consecutive Fragment chunks at
/// pretoken-safe boundaries that no added-token occurrence straddles, so
/// even a single huge document is encoded across all cores with
/// token-identical output.
///
/// With LPT disabled (GIGATOK_NO_LPT) this restores uniform sizing:
/// head_bytes = 0 makes every Docs group aim for `target`, and
/// frag_head = usize::MAX makes every fragment take the primary split
/// size `target` (the sub-split branch is never entered), which is
/// exactly the old safe_split_ranges(doc, target) loop. The oversize
/// threshold (`doc.len() > 2 * target`) is identical in both shapes.
fn build_doc_chunks<'a>(
    docs: &[&'a [u8]],
    total: usize,
    target: usize,
    added_tokens: &[(&[u8], bool)],
    lpt: bool,
) -> Vec<EncodeChunk<'a>> {
    let (head_bytes, group_big, frag_big, tail_target) = if lpt {
        (
            total - total / 5,
            2 * target,
            2 * target,
            (target / 4).max(MIN_CHUNK_BYTES),
        )
    } else {
        (0, target, target, target)
    };
    let mut chunks = Vec::new();
    let mut group: Vec<&[u8]> = Vec::new();
    // Bytes already assigned to chunks: positions below `head_bytes` take
    // the big target, the rest the small one.
    let mut emitted = 0usize;
    let mut acc = 0usize;
    for &doc in docs {
        if doc.len() > 2 * target {
            if !group.is_empty() {
                chunks.push(EncodeChunk::Docs(std::mem::take(&mut group)));
                emitted += acc;
                acc = 0;
            }
            push_fragment_chunks(
                &mut chunks,
                doc,
                if lpt {
                    head_bytes.saturating_sub(emitted)
                } else {
                    usize::MAX
                },
                frag_big,
                tail_target,
                added_tokens,
            );
            emitted += doc.len();
            continue;
        }
        group.push(doc);
        acc += doc.len();
        let group_target = if emitted < head_bytes {
            group_big
        } else {
            tail_target
        };
        if acc >= group_target {
            chunks.push(EncodeChunk::Docs(std::mem::take(&mut group)));
            emitted += acc;
            acc = 0;
        }
    }
    if !group.is_empty() {
        chunks.push(EncodeChunk::Docs(group));
    }
    chunks
}

/// Split one oversized document into consecutive Fragment chunks with the
/// descending sizes of `build_doc_chunks`: `big`-sized fragments over the
/// first `head_len` bytes, `tail_target`-sized fragments after.
/// Sub-splitting a tail fragment preserves boundary safety: the pretoken
/// cut check is purely local (3 bytes around the cut), and an added-token
/// occurrence is orders of magnitude shorter than the >= MIN_CHUNK_BYTES
/// distance of any sub-cut from its fragment's (already safe) edges.
fn push_fragment_chunks<'a>(
    chunks: &mut Vec<EncodeChunk<'a>>,
    doc: &'a [u8],
    head_len: usize,
    big: usize,
    tail_target: usize,
    added_tokens: &[(&[u8], bool)],
) {
    let mut first = true;
    for r in crate::pretokenize::safe_split_ranges(doc, big, added_tokens) {
        if r.start < head_len || r.len() <= tail_target {
            chunks.push(EncodeChunk::Fragment {
                bytes: &doc[r],
                first: std::mem::take(&mut first),
            });
        } else {
            for sub in
                crate::pretokenize::safe_split_ranges(&doc[r.clone()], tail_target, added_tokens)
            {
                chunks.push(EncodeChunk::Fragment {
                    bytes: &doc[r.start + sub.start..r.start + sub.end],
                    first: std::mem::take(&mut first),
                });
            }
        }
    }
}

/// Cut every file into document-aligned byte regions for the separator /
/// JSONL encode paths, with the same descending (LPT) sizes as
/// `build_doc_chunks`: ~2x-target regions over the first ~80% of the total
/// bytes (measured across files in order), quarter-target regions over the
/// last ~20%. Regions are handed out in order, so the small tail bounds how
/// long the whole pool waits on whichever core — often an E-core — draws
/// the last region. With `lpt` off every region aims for `target`. Chunking
/// only moves region boundaries between whole documents, so tokens and
/// row order are identical either way.
fn region_chunks<'a>(
    files: &[&'a [u8]],
    format: &DocFormat,
    total: usize,
    lpt: bool,
) -> Vec<&'a [u8]> {
    let target = chunk_target_bytes(total);
    let head_bytes = total - total / 5;
    let tail_target = (target / 4).max(MIN_CHUNK_BYTES);
    let mut regions = Vec::new();
    let mut base = 0usize;
    for &bytes in files {
        let target_at = |start: usize| {
            if !lpt {
                target
            } else if base + start < head_bytes {
                2 * target
            } else {
                tail_target
            }
        };
        regions.extend(
            chunk_ranges(bytes, format, target_at)
                .into_iter()
                .map(|r| &bytes[r]),
        );
        base += bytes.len();
    }
    regions
}

/// Concatenate per-chunk row lengths into per-document row counts, merging
/// `continues` fragments into the previous document's row.
fn row_counts(chunks: &[ChunkTokens]) -> Vec<i64> {
    let mut counts: Vec<i64> = Vec::new();
    for chunk in chunks {
        let mut lens = chunk.lens.iter().copied();
        if chunk.continues
            && let Some(l) = lens.next()
        {
            *counts
                .last_mut()
                .expect("continuation fragment before any document") += l;
        }
        counts.extend(lens);
    }
    counts
}

/// The id buffers `chunks` still hold (buffers already committed to the
/// flat result were recycled and are empty).
fn spent_buffers(chunks: &mut [ChunkTokens]) -> Vec<Vec<u32>> {
    chunks
        .iter_mut()
        .map(|c| std::mem::take(&mut c.ids))
        .filter(|v| v.capacity() > 0)
        .collect()
}

/// Free spent id buffers off the caller's critical path. They total as
/// much memory as the gathered result, and tearing them down is munmap
/// page teardown under the address-space write lock: inline frees convoy
/// the gather copy's first-touch faults (read lock) behind each munmap
/// (write lock), and a serial free after the copy keeps the munmaps inside
/// the timed window. A detached background task pays the same teardown CPU
/// after the caller has returned, overlapped with whatever runs next, and
/// occupies at most one pool thread. Totals under `DEFERRED_DROP_MIN_BYTES`
/// drop inline: their teardown is microseconds.
fn drop_buffers(bufs: Vec<Vec<u32>>) {
    let bytes = bufs.iter().map(|v| v.capacity()).sum::<usize>() * std::mem::size_of::<u32>();
    if bytes >= DEFERRED_DROP_MIN_BYTES {
        rayon::spawn(move || drop(bufs));
    }
}

/// The in-flight state of the overlapped gather: a flat id buffer reserved
/// at an upper bound BEFORE chunk sizes are known, plus a cursor over the
/// longest fully-encoded prefix of the chunk sequence.
///
/// Run as a separate phase after the encode, the final gather's first-touch
/// faults + memcpy are a serial tail, most of whose CPU is kernel
/// fault-path contention from every thread faulting the flat buffer at
/// once. Chunk completion is near-sequential (strict in-order handout,
/// descending LPT sizes), so a worker that finishes a chunk can commit the
/// ready prefix — offsets are exact, they are sums of *completed* chunk
/// sizes — while the tail still encodes. The copy work then hides inside
/// the encode phase at ~1 thread at a time (no fault convoy), leaving only
/// a small residual drain after the last chunk.
///
/// The upper bound: a token consumes at least one input byte, so
/// `total_bytes` tokens bounds the output; untouched reserved pages cost
/// address space only. Two escapes fall back to the collect-then-gather
/// path (`gather_flat`): the reservation itself failing (e.g. Linux
/// heuristic overcommit refusing a 4x-input VA block for a huge batch),
/// and the cursor overflowing the bound, which is impossible for plain
/// byte input and reachable only when NFC normalization expands bytes
/// (composition-exclusion pathologies) — `advance` stops committing and
/// `finish` assembles the result into a fresh exact-size buffer rather than
/// write past the reservation.
struct Committer {
    /// Owns the reservation; the Vec struct itself is read or written only
    /// in `finish`. `UnsafeCell` so each `advance` derives the destination
    /// pointer fresh under the cursor lock: no pointer is captured across
    /// the struct's construction-time moves (a move retags the Vec's
    /// unique pointer under strict aliasing models).
    flat: UnsafeCell<Vec<u32>>,
    /// Reserved capacity in tokens; commits never write at or past it.
    cap: usize,
    cursor: Mutex<CommitCursor>,
}

struct CommitCursor {
    /// Index of the first uncommitted chunk.
    next: usize,
    /// Tokens committed so far == sum of ids.len() over chunks[..next].
    offset: usize,
    /// A chunk did not fit under `cap`; committing has stopped for good.
    overflowed: bool,
}

// SAFETY: the heap buffer behind `flat` is never reallocated while shared
// (nothing pushes to the Vec; it is resized only in `finish`, after all
// shared use has ended). During the shared phase the cell is used solely
// to derive the buffer pointer under the `cursor` lock, and every write
// through it lands in a disjoint, in-bounds range.
unsafe impl Send for Committer {}
unsafe impl Sync for Committer {}

impl Committer {
    /// Chunks committed per `advance` call. Bounds how long one worker is
    /// away from encoding (a backlog can pile up behind a long-held lock);
    /// anything left over is drained by later completions or `finish`.
    const MAX_DRAIN: usize = 8;

    /// Reserve `cap` tokens up front, or None (→ classic gather) if the
    /// allocator refuses.
    fn try_new(cap: usize) -> Option<Self> {
        let mut flat: Vec<u32> = Vec::new();
        if cap == 0 || flat.try_reserve_exact(cap).is_err() {
            return None;
        }
        // The commits fault this reservation in while the encode runs.
        madvise_hugepage(flat.as_mut_ptr() as *mut u8, cap * std::mem::size_of::<u32>());
        Some(Self {
            flat: UnsafeCell::new(flat),
            cap,
            cursor: Mutex::new(CommitCursor {
                next: 0,
                offset: 0,
                overflowed: false,
            }),
        })
    }

    /// Copy any freshly completed prefix chunks into the flat buffer.
    /// Non-blocking: if another worker is mid-commit, return to encoding —
    /// the current holder (or a later completion, or `finish`) picks the
    /// chunk up. A completion that lands between the holder's last check
    /// and its unlock is likewise deferred, never lost.
    fn advance(&self, outs: &[OnceLock<Mutex<ChunkTokens>>], pool: &IdPool) {
        let Ok(mut cur) = self.cursor.try_lock() else {
            return;
        };
        // SAFETY: the cursor lock is held; the Vec struct is not mutated
        // during the shared phase (see the `flat` field doc), so deriving
        // the buffer pointer only reads it.
        let base = unsafe { (*self.flat.get()).as_mut_ptr() };
        for _ in 0..Self::MAX_DRAIN {
            if cur.overflowed {
                return;
            }
            let Some(slot) = outs.get(cur.next).and_then(OnceLock::get) else {
                return;
            };
            let mut chunk = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let len = chunk.ids.len();
            if self.cap - cur.offset < len {
                cur.overflowed = true;
                return;
            }
            // SAFETY: holding `cursor`, writing [offset, offset+len), which
            // is within the reservation (checked above) and disjoint from
            // every earlier commit (offset is monotone).
            unsafe {
                std::ptr::copy_nonoverlapping(chunk.ids.as_ptr(), base.add(cur.offset), len);
            }
            cur.offset += len;
            cur.next += 1;
            // The tokens now live in `flat`: hand the chunk buffer back to
            // the call's pool right away so the chunks still to start reuse
            // its (cache-warm, already faulted) pages instead of faulting in
            // fresh memory, and so the live chunk memory stays at the
            // in-flight set instead of the whole output.
            let spent = std::mem::take(&mut chunk.ids);
            drop(chunk);
            drop(pool.recycle(spent));
        }
    }

    /// After all chunks are encoded (and the scope joined): copy the
    /// uncommitted suffix in parallel, size the buffer to the total and
    /// trim the reservation. If the bound was overrun (see type docs) the
    /// result is assembled into a fresh exact-size buffer instead. Either
    /// way the spent chunk buffers are returned for the caller to free.
    fn finish(self, mut chunks: Vec<ChunkTokens>) -> (Vec<u32>, Vec<Vec<u32>>) {
        use rayon::prelude::*;
        /// Raw destination pointer, shareable across the copy tasks. (The
        /// accessor keeps closure capture at the wrapper, not the field.)
        struct SyncPtr(*mut u32);
        // SAFETY: only used for the disjoint in-bounds writes below.
        unsafe impl Send for SyncPtr {}
        unsafe impl Sync for SyncPtr {}
        impl SyncPtr {
            /// SAFETY: `off` must be within the reservation.
            unsafe fn at(&self, off: usize) -> *mut u32 {
                unsafe { self.0.add(off) }
            }
        }

        let Committer { flat, cap, cursor } = self;
        let mut flat = flat.into_inner();
        let cur = cursor
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Chunks before `cur.next` were recycled after their commit: their
        // tokens now live only in `flat[..cur.offset]`.
        let rest_total: usize = chunks[cur.next..].iter().map(|c| c.ids.len()).sum();
        let total = cur.offset + rest_total;
        if cur.overflowed || total > cap {
            // Bound overrun (NFC expansion only): assemble into a fresh
            // exact-size buffer from the committed prefix plus the intact
            // uncommitted chunks.
            let mut out: Vec<u32> = Vec::with_capacity(total);
            // SAFETY: flat[..cur.offset] was fully written by `advance`.
            out.extend_from_slice(unsafe {
                std::slice::from_raw_parts(flat.as_ptr(), cur.offset)
            });
            for chunk in &chunks[cur.next..] {
                out.extend_from_slice(&chunk.ids);
            }
            drop(flat);
            return (out, spent_buffers(&mut chunks));
        }
        let rest = &chunks[cur.next..];
        let mut offsets = Vec::with_capacity(rest.len());
        let mut offset = cur.offset;
        for chunk in rest {
            offsets.push(offset);
            offset += chunk.ids.len();
        }
        debug_assert_eq!(offset, total);
        // Derived AFTER the Vec moved out of the cell: a move retags the
        // Vec's unique pointer under strict aliasing models, so the suffix
        // writes and the `set_len` below go through a post-move pointer.
        let base = SyncPtr(flat.as_mut_ptr());
        // `with_max_len(1)` keeps the multi-MB copies stealable one by one.
        rest.par_iter()
            .zip(offsets)
            .with_max_len(1)
            .for_each(|(chunk, off)| {
                // SAFETY: exclusive access (workers are joined); suffix
                // ranges are disjoint from each other and from the
                // committed prefix, and end at total <= cap.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        chunk.ids.as_ptr(),
                        base.at(off),
                        chunk.ids.len(),
                    );
                }
            });
        // SAFETY: capacity >= cap >= total, and [0, total) was fully
        // initialized by the prefix commits plus the suffix copies above.
        unsafe {
            flat.set_len(total);
        }
        // Return the unused reservation. Large allocations trim in place on
        // the mainstream allocators (macOS libmalloc large entries, glibc
        // mmap'd chunks via mremap): pointer-stable, no copy, and the
        // untouched tail pages were never faulted so there is nothing to
        // tear down. An allocator that copies instead only costs one
        // memcpy; correctness is unaffected.
        flat.shrink_to_fit();
        // The suffix is copied out; its buffers are spent.
        (flat, spent_buffers(&mut chunks))
    }
}

/// Encode all chunks with pooled workers and gather them into one flat id
/// buffer plus per-document row counts — in parallel when there is more
/// than one chunk, serially otherwise. Each worker's caches are pre-sized
/// for its share of `total_bytes` (capacity hints only — see
/// `Tokenizer::fork_sized`; workers already forked on an earlier call keep
/// their warm caches).
///
/// Chunks are handed out in strict index order through an atomic counter
/// (one pulling task per rayon thread), not `par_iter`: recursive range
/// splitting lets a thread steal a subrange of early big chunks and still
/// be *starting* a 2×-target chunk after everyone else has reached the
/// small tail, stranding the rest of the pool behind that one straggler.
/// In-order handout makes the LPT descending-size order of
/// `build_doc_chunks` a guarantee, bounding the tail at roughly one small
/// chunk — and makes chunk completion near-sequential, which is what lets
/// the gather copy overlap the encode (see `Committer`).
pub(crate) fn encode_chunks_gathered(
    workers: &WorkerPool,
    proto: &Tokenizer,
    chunks: &[EncodeChunk],
    total_bytes: usize,
) -> (Vec<u32>, Vec<i64>) {
    // A token consumes >= 1 input byte, so total_bytes tokens is the
    // reservation bound (NFC expansion is caught by the overflow escape).
    encode_chunks_gathered_with_cap(workers, proto, chunks, total_bytes, total_bytes)
}

/// `encode_chunks_gathered` with the committer's reservation bound passed
/// explicitly, so tests can force the overflow fallback in-process.
fn encode_chunks_gathered_with_cap(
    workers: &WorkerPool,
    proto: &Tokenizer,
    chunks: &[EncodeChunk],
    total_bytes: usize,
    cap_tokens: usize,
) -> (Vec<u32>, Vec<i64>) {
    let share = total_bytes / rayon::current_num_threads().max(1);
    run_gathered(chunks.len(), cap_tokens, |i, pool| {
        workers.with_worker(proto, share, |tok| encode_chunk(tok, &chunks[i], pool))
    })
}

/// The backend-independent core of the gathered encode: run `encode(i, pool)`
/// for every chunk index `0..n_chunks` on the rayon pool, handing indices
/// out in strict order, and gather the results (see
/// `encode_chunks_gathered` for the handout and `Committer` for the
/// overlapped copy). `encode` must be callable concurrently, takes its
/// output buffer from the call's [`IdPool`], and is passed each index
/// exactly once. Every buffer the call used is freed before it returns
/// (large totals on a background task, see [`drop_buffers`]).
fn run_gathered(
    n_chunks: usize,
    cap_tokens: usize,
    encode: impl Fn(usize, &IdPool) -> ChunkTokens + Sync,
) -> (Vec<u32>, Vec<i64>) {
    let pool = IdPool::default();
    if n_chunks <= 1 {
        // Small inputs skip the thread fan-out — and a lone chunk's id
        // buffer IS the flat result, no gather copy at all.
        return if n_chunks == 1 {
            let out = encode(0, &pool);
            let counts = row_counts(std::slice::from_ref(&out));
            (out.ids, counts)
        } else {
            (Vec::new(), Vec::new())
        };
    }
    let next = AtomicUsize::new(0);
    let outs: Vec<OnceLock<Mutex<ChunkTokens>>> =
        (0..n_chunks).map(|_| OnceLock::new()).collect();
    let committer = Committer::try_new(cap_tokens);
    let tasks = rayon::current_num_threads().min(n_chunks);
    rayon::scope(|s| {
        for _ in 0..tasks {
            s.spawn(|_| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n_chunks {
                        // One last opportunistic drain on the way out: this
                        // worker's final chunk may have been skipped while
                        // another held the commit lock.
                        if let Some(c) = &committer {
                            c.advance(&outs, &pool);
                        }
                        break;
                    }
                    // Each index is claimed exactly once, so `set` cannot
                    // already be filled.
                    let _ = outs[i].set(Mutex::new(encode(i, &pool)));
                    if let Some(c) = &committer {
                        c.advance(&outs, &pool);
                    }
                }
            });
        }
    });
    let outs: Vec<ChunkTokens> = outs
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .expect("every claimed chunk was encoded")
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        })
        .collect();
    let counts = row_counts(&outs);
    let (flat, mut spent) = match committer {
        Some(c) => c.finish(outs),
        None => gather_flat(outs),
    };
    spent.extend(pool.into_buffers());
    drop_buffers(spent);
    (flat, counts)
}

/// Copy all chunk id buffers into one freshly allocated exact-size flat
/// buffer, in parallel, and return it with the spent chunk buffers for the
/// caller to free. The fallback of the overlapped gather (see
/// `Committer`), used only when its up-front reservation is refused or
/// overrun.
fn gather_flat(mut chunks: Vec<ChunkTokens>) -> (Vec<u32>, Vec<Vec<u32>>) {
    use rayon::prelude::*;
    let total: usize = chunks.iter().map(|c| c.ids.len()).sum();
    let mut flat = vec![0u32; total];
    // The parallel copy below faults in the whole buffer.
    madvise_hugepage(flat.as_mut_ptr() as *mut u8, total * std::mem::size_of::<u32>());
    let mut rest: &mut [u32] = &mut flat;
    let mut slices = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let (head, tail) = rest.split_at_mut(chunk.ids.len());
        slices.push(head);
        rest = tail;
    }
    // `with_max_len(1)` keeps the multi-MB copies stealable one by one.
    slices
        .into_par_iter()
        .zip(chunks.par_iter())
        .with_max_len(1)
        .for_each(|(dst, chunk)| dst.copy_from_slice(&chunk.ids));
    (flat, spent_buffers(&mut chunks))
}

/// Persistent pool of forked tokenizer workers used by encode_batch and
/// encode_files. One slot per rayon thread, forked lazily on first use and
/// retained for the tokenizer's lifetime, so each worker's pretoken cache
/// stays warm when encoding is invoked repeatedly (e.g. in a loop).
///
/// Invariant: the prototype tokenizer must not be mutated between encodes
/// that share a pool. Workers are forked lazily (first use of each slot)
/// and never refreshed, so mutating the prototype (`add_special_token`,
/// `set_added_tokens`, `set_pretokenizer_type`, ...) after some slots have
/// forked leaves those workers on the OLD state while slots forked later
/// (or rebuilt after a worker panic) capture the new one — a chunk's
/// tokens would then depend on which slot the handout gave it. Finish all
/// model mutation before the pool's first encode (the loaders do), or use
/// a fresh pool after mutating. The Python bindings uphold this by
/// construction: no mutator is exposed after the pyclass is built.
pub struct WorkerPool {
    slots: OnceLock<Vec<Mutex<Option<Tokenizer>>>>,
    /// Worker for the sequential (`parallel=false`) encode paths, kept
    /// separate from `slots` so a sequential call never sizes or touches
    /// the rayon pool (even `rayon::current_num_threads()` would build it).
    serial: Mutex<Option<Tokenizer>>,
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerPool {
    pub fn new() -> Self {
        Self {
            slots: OnceLock::new(),
            serial: Mutex::new(None),
        }
    }

    /// Run `f` with exclusive access to a pooled worker, forking one sized
    /// for `expected_bytes` of input if the slot is empty. Rayon never runs
    /// more tasks concurrently than it has threads, and there is one slot
    /// per thread, so a free slot always exists; the yield loop only spins
    /// when non-rayon threads encode at the same time.
    ///
    /// `proto` must be the same, unmutated prototype on every call for a
    /// given pool: forks are cached per slot and never compared against
    /// `proto` again, so a mutated (or different) prototype yields stale
    /// workers for already-filled slots (see the type-level invariant).
    fn with_worker<R>(
        &self,
        proto: &Tokenizer,
        expected_bytes: usize,
        f: impl FnOnce(&mut Tokenizer) -> R,
    ) -> R {
        let slots = self.slots.get_or_init(|| {
            (0..rayon::current_num_threads())
                .map(|_| Mutex::new(None))
                .collect()
        });
        loop {
            for slot in slots {
                match slot.try_lock() {
                    Ok(mut guard) => {
                        return f(guard.get_or_insert_with(|| proto.fork_sized(expected_bytes)));
                    }
                    Err(TryLockError::Poisoned(poisoned)) => {
                        // A worker panicked mid-encode; its cache may be
                        // inconsistent, so rebuild it from the prototype.
                        // The slot is un-poisoned so the rebuilt worker is
                        // kept (and stays warm) instead of being rebuilt on
                        // every later call.
                        let mut guard = poisoned.into_inner();
                        *guard = None;
                        slot.clear_poison();
                        return f(guard.get_or_insert_with(|| proto.fork_sized(expected_bytes)));
                    }
                    Err(TryLockError::WouldBlock) => {}
                }
            }
            std::thread::yield_now();
        }
    }

    /// `with_worker` for the sequential paths: run `f` with the dedicated
    /// serial worker (forked lazily, retained so its pretoken cache stays
    /// warm across calls), without initializing the rayon-sized slots. The
    /// same unmutated-prototype invariant applies. Blocks if another thread
    /// is in a sequential encode on the same pool.
    fn with_serial_worker<R>(
        &self,
        proto: &Tokenizer,
        expected_bytes: usize,
        f: impl FnOnce(&mut Tokenizer) -> R,
    ) -> R {
        let mut guard = self.serial.lock().unwrap_or_else(|poisoned| {
            // A worker panicked mid-encode; its cache may be inconsistent,
            // so rebuild it from the prototype (un-poisoned, so the rebuilt
            // worker is kept).
            let mut guard = poisoned.into_inner();
            *guard = None;
            self.serial.clear_poison();
            guard
        });
        f(guard.get_or_insert_with(|| proto.fork_sized(expected_bytes)))
    }
}

/// Sizing of the table of short-unit encodings shared by the workers of one
/// SentencePiece encode call (see `sp_shared`). Callers use
/// [`Self::DEFAULT`]; tests pass tiny values to reach the shared-table
/// paths without a 16 MiB input.
#[derive(Clone, Copy)]
struct SharedUnitsParams {
    /// Inputs below this many bytes keep private caches only: the shared
    /// table's first-touch page faults would outweigh the merges it saves.
    min_input: usize,
    /// Bounds of the table's log2 slot count (64 B per slot).
    min_log2: u32,
    max_log2: u32,
}

/// Input bytes per shared-table slot. A short unit spans several bytes and
/// most occurrences repeat an earlier one, so distinct units are far fewer
/// than one per 16 input bytes; the table stays sparse enough for its
/// two-line probe window to rarely refuse a unit.
const SHARED_UNITS_BYTES_PER_SLOT: usize = 16;

impl SharedUnitsParams {
    /// Inputs of 16 MiB and up share a table of 2^19..2^21 slots (32 MiB to
    /// 128 MiB, mapped lazily as zero pages). The 2^21 cap is reached from
    /// a 32 MiB input on; at the 16 MiB threshold the size rule already asks
    /// for 2^20 slots, so the 2^19 floor only matters for a lowered
    /// threshold. The table is not part of any per-worker cache budget:
    /// one per call, freed when the call returns.
    const DEFAULT: Self = Self {
        min_input: 16 << 20,
        min_log2: 19,
        max_log2: 21,
    };

    /// log2 of the slot count for a call over `size_hint` input bytes.
    fn log2_slots(&self, size_hint: usize) -> u32 {
        (size_hint / SHARED_UNITS_BYTES_PER_SLOT)
            .next_power_of_two()
            .trailing_zeros()
            .clamp(self.min_log2, self.max_log2)
    }
}

/// The encoder states of ONE SentencePiece encode call: one slot per rayon
/// thread, filled lazily by the first chunk a thread encodes, so every
/// later chunk of the call reuses a warm pretoken cache instead of paying a
/// cold 16 MiB front cache and empty unit maps per chunk. Cache contents
/// never affect the produced ids (a hit returns what the merge would
/// compute).
///
/// Memory: each state is bounded by the model's per-worker cache budget
/// (`SentencePieceBPE::max_cache_bytes`), so the call holds up to threads x
/// budget plus the shared unit table (see [`SharedUnitsParams`]). Nothing
/// outlives the call: [`Self::release`] frees every state and the table.
struct SpCallPool<'a> {
    model: &'a bpe::SentencePieceBPE,
    slots: Vec<Mutex<Option<bpe::sentencepiece::EncodeState>>>,
    /// Short-unit encodings shared by every worker (see `sp_shared`);
    /// `None` when the call's input is below `SharedUnitsParams::min_input`.
    shared: Option<std::sync::Arc<bpe::sp_shared::SharedUnits>>,
}

impl<'a> SpCallPool<'a> {
    /// A pool for a call over `size_hint` input bytes, which sizes the
    /// shared table.
    fn new(model: &'a bpe::SentencePieceBPE, size_hint: usize, params: SharedUnitsParams) -> Self {
        Self {
            model,
            slots: (0..rayon::current_num_threads())
                .map(|_| Mutex::new(None))
                .collect(),
            shared: (size_hint >= params.min_input).then(|| {
                std::sync::Arc::new(bpe::sp_shared::SharedUnits::new(
                    params.log2_slots(size_hint),
                ))
            }),
        }
    }

    /// Run `f` with a pooled encoder. The slot's state is moved into the
    /// encoder and put back afterwards (a panic inside `f` drops it, so the
    /// slot rebuilds a fresh state next time). Rayon never runs more tasks
    /// concurrently than it has threads and there is one slot per thread,
    /// so a free slot always exists; the yield loop only spins when
    /// non-rayon threads encode at the same time.
    fn with_encoder<R>(&self, f: impl FnOnce(&mut bpe::sentencepiece::Encoder<'_>) -> R) -> R {
        loop {
            for slot in &self.slots {
                let mut guard = match slot.try_lock() {
                    Ok(guard) => guard,
                    Err(TryLockError::Poisoned(poisoned)) => {
                        // The state was dropped by the panic; the slot just
                        // starts empty again.
                        slot.clear_poison();
                        let mut guard = poisoned.into_inner();
                        *guard = None;
                        guard
                    }
                    Err(TryLockError::WouldBlock) => continue,
                };
                let mut encoder = match guard.take() {
                    Some(state) => bpe::sentencepiece::Encoder::from_state(self.model, state),
                    None => self.model.encoder(),
                };
                if let Some(shared) = &self.shared {
                    encoder.ensure_shared(|| std::sync::Arc::clone(shared));
                }
                let result = f(&mut encoder);
                *guard = Some(encoder.into_state());
                return result;
            }
            std::thread::yield_now();
        }
    }

    /// Free every state and the shared table, in parallel: each state's
    /// front cache and the table are tens of MiB to unmap.
    fn release(self) {
        use rayon::prelude::*;
        let Self { slots, shared, .. } = self;
        let states: Vec<bpe::sentencepiece::EncodeState> = slots
            .into_iter()
            .filter_map(|slot| {
                slot.into_inner()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
            })
            .collect();
        rayon::join(|| states.into_par_iter().for_each(drop), || drop(shared));
    }
}

/// Shared core of encode_batch / encode_files for pre-resolved document
/// slices: chunk (splitting oversized documents at pretoken-safe
/// boundaries), encode with pooled workers, and assemble the ragged result
/// (one flat id buffer plus per-document row lengths). Public so Rust
/// benches exercise the identical parallel path as the Python bindings.
///
/// Environment: setting `GIGATOK_NO_LPT` (to any value, empty included)
/// disables LPT chunk sizing in favor of uniform chunks — token- and
/// order-identical output, chunk shaping only; see `lpt_from_env`. The
/// variable is read once per call, never in per-chunk loops.
pub fn encode_docs_ragged(
    workers: &WorkerPool,
    proto: &Tokenizer,
    docs: &[&[u8]],
) -> (Vec<u32>, Vec<i64>) {
    encode_docs_ragged_with(workers, proto, docs, lpt_from_env())
}

/// `encode_docs_ragged` with the LPT switch passed explicitly instead of
/// read from the environment, so tests can cover both shapes in-process
/// without mutating process env.
pub(crate) fn encode_docs_ragged_with(
    workers: &WorkerPool,
    proto: &Tokenizer,
    docs: &[&[u8]],
    lpt: bool,
) -> (Vec<u32>, Vec<i64>) {
    let total: usize = docs.iter().map(|d| d.len()).sum();
    let added = proto.added_token_split_blockers();
    let chunks = build_doc_chunks(docs, total, chunk_target_bytes(total), &added, lpt);
    encode_chunks_gathered(workers, proto, &chunks, total)
}

/// Sequential `encode_docs_ragged`: encode every document in order on the
/// calling thread with the pool's dedicated serial worker. Never touches
/// rayon — required when the caller is a forked child of a process whose
/// global rayon pool was already built (the pool's threads do not survive
/// the fork, so injecting work into it would wait forever), and what the
/// Python bindings' `parallel=false` promises. Token- and order-identical
/// to the parallel path (which `parallel_ragged_matches_serial` checks
/// against exactly this shape of serial loop).
pub fn encode_docs_ragged_serial(
    workers: &WorkerPool,
    proto: &Tokenizer,
    docs: &[&[u8]],
) -> (Vec<u32>, Vec<i64>) {
    let total: usize = docs.iter().map(|d| d.len()).sum();
    workers.with_serial_worker(proto, total, |tok| {
        let mut ids: Vec<u32> = Vec::with_capacity(total / 4 + 16);
        // Huge pages before the encode's stores fault the buffer in (as in
        // encode_chunk); serially the one buffer is the whole result.
        madvise_hugepage(ids.as_mut_ptr() as *mut u8, ids.capacity() * 4);
        let mut lens = Vec::with_capacity(docs.len());
        for doc in docs {
            encode_into(tok, doc, &mut ids, &mut lens);
        }
        (ids, lens)
    })
}

/// Work unit for parallel SentencePiece encoding, mirroring `EncodeChunk`.
enum SpChunk<'a> {
    /// A run of whole documents, one output row each.
    Docs(Vec<&'a str>),
    /// A scanner-safe fragment of one oversized document (see
    /// `SentencePieceBPE::safe_fragment_ranges`). Fragments of a document
    /// are consecutive chunks; `first` marks the document's first fragment.
    Fragment { text: &'a str, first: bool },
    /// A byte region holding many documents, split during encoding
    /// (JSONL lines or separator-delimited text); contents are trusted
    /// valid UTF-8.
    Region {
        bytes: &'a [u8],
        format: &'a DocFormat,
    },
}

/// Group documents into parallel chunks of roughly `target` bytes. A
/// document larger than `2 * target` (the BPE path's oversize threshold) is
/// split into consecutive Fragment chunks at unit boundaries the scanner
/// proves safe, so even a single huge document is encoded across all cores
/// with token-identical output — except on models without the raw fast path
/// (`supports_fragment_split`), where no interior cut is provably safe and
/// the document stays one chunk.
fn sp_build_chunks<'a>(
    tokenizer: &bpe::SentencePieceBPE,
    texts: &[&'a str],
    target: usize,
) -> Vec<SpChunk<'a>> {
    let can_split = tokenizer.supports_fragment_split();
    let mut chunks = Vec::new();
    let mut group: Vec<&str> = Vec::new();
    let mut acc = 0usize;
    for &text in texts {
        if can_split && text.len() > 2 * target {
            if !group.is_empty() {
                chunks.push(SpChunk::Docs(std::mem::take(&mut group)));
                acc = 0;
            }
            let mut first = true;
            for r in tokenizer.safe_fragment_ranges(text, target) {
                chunks.push(SpChunk::Fragment {
                    text: &text[r],
                    first: std::mem::take(&mut first),
                });
            }
            continue;
        }
        group.push(text);
        acc += text.len();
        if acc >= target {
            chunks.push(SpChunk::Docs(std::mem::take(&mut group)));
            acc = 0;
        }
    }
    if !group.is_empty() {
        chunks.push(SpChunk::Docs(group));
    }
    chunks
}

/// Id capacity requested for a SentencePiece chunk's output (see
/// `chunk_id_cap`).
fn sp_chunk_id_cap(chunk: &SpChunk) -> usize {
    let byte_len = match chunk {
        SpChunk::Docs(group) => group.iter().map(|t| t.len()).sum::<usize>(),
        SpChunk::Fragment { text, .. } => text.len(),
        SpChunk::Region { bytes, .. } => bytes.len(),
    };
    // Llama-family vocabularies emit ~1 token per 3.5-4 bytes of English; a
    // /3 reservation keeps a chunk's output within its buffer. Untouched
    // reserved pages cost address space only.
    byte_len / 3 + 16
}

/// Encode one SentencePiece chunk with `encoder`, reserving the output once
/// from a bytes-per-token estimate on the low side of natural language
/// (pooled, or huge-page hinted before the encode's stores fault it in — as
/// in `encode_chunk`).
fn sp_encode_chunk(
    encoder: &mut bpe::sentencepiece::Encoder<'_>,
    chunk: &SpChunk,
    id_pool: &IdPool,
) -> ChunkTokens {
    let mut ids: Vec<u32> = id_pool.take(sp_chunk_id_cap(chunk));
    let mut lens: Vec<i64> = Vec::new();
    let mut continues = false;
    match chunk {
        SpChunk::Docs(group) => {
            for text in group {
                sp_encode_into(encoder, text, &mut ids, &mut lens);
            }
        }
        SpChunk::Fragment { text, first } => {
            sp_encode_fragment_into(encoder, text, *first, &mut ids, &mut lens);
            continues = !*first;
        }
        SpChunk::Region { bytes, format } => {
            for_each_doc(bytes, format, |doc| {
                // SAFETY: documents are trusted valid UTF-8 (see
                // `sp_encode_files_docs`).
                let text = unsafe { std::str::from_utf8_unchecked(doc) };
                sp_encode_into(encoder, text, &mut ids, &mut lens);
            });
        }
    }
    ChunkTokens {
        ids,
        lens,
        continues,
    }
}

/// Encode SentencePiece chunks on `pool`'s warm encoder states and gather
/// them into one flat id buffer plus per-document row counts (`row_counts`
/// merges a continuation fragment's first row into the previous document's
/// row), overlapping the gather copy with the encode exactly as the BPE
/// path does (see `run_gathered`).
fn sp_encode_chunks_in(
    pool: &SpCallPool,
    chunks: &[SpChunk],
    total_bytes: usize,
) -> (Vec<u32>, Vec<i64>) {
    // A token consumes >= 1 input byte, so total_bytes tokens is the
    // reservation bound (normalization expansion is caught by the
    // committer's overflow escape).
    run_gathered(chunks.len(), total_bytes, |i, id_pool| {
        pool.with_encoder(|encoder| sp_encode_chunk(encoder, &chunks[i], id_pool))
    })
}

/// [`sp_encode_chunks_in`] on a pool made for this one call: the encoder
/// states and the shared unit table are freed before it returns.
fn sp_encode_chunks(
    tokenizer: &bpe::SentencePieceBPE,
    chunks: &[SpChunk],
    total_bytes: usize,
    shared: SharedUnitsParams,
) -> (Vec<u32>, Vec<i64>) {
    let pool = SpCallPool::new(tokenizer, total_bytes, shared);
    let result = sp_encode_chunks_in(&pool, chunks, total_bytes);
    pool.release();
    result
}

/// SentencePiece analog of `encode_docs_ragged`: group whole documents into
/// parallel chunks — splitting an oversized document into scanner-safe
/// fragments (see `sp_build_chunks`) — and encode the chunks with the
/// call's pooled encoder states.
pub fn sp_encode_docs_ragged(
    tokenizer: &bpe::SentencePieceBPE,
    texts: &[&str],
) -> (Vec<u32>, Vec<i64>) {
    let total: usize = texts.iter().map(|t| t.len()).sum();
    let chunks = sp_build_chunks(tokenizer, texts, chunk_target_bytes(total));
    sp_encode_chunks(tokenizer, &chunks, total, SharedUnitsParams::DEFAULT)
}

/// Sequential `sp_encode_docs_ragged`: one Encoder (so one pretoken cache)
/// over all documents, on the calling thread, never touching rayon. Token-
/// and order-identical to the parallel path, which encodes the same
/// documents in the same order with per-worker encoder states.
pub(crate) fn sp_encode_docs_ragged_serial(
    tokenizer: &bpe::SentencePieceBPE,
    texts: &[&str],
) -> (Vec<u32>, Vec<i64>) {
    let mut encoder = tokenizer.encoder();
    let mut ids: Vec<u32> = Vec::new();
    let mut lens: Vec<i64> = Vec::with_capacity(texts.len());
    for &text in texts {
        sp_encode_into(&mut encoder, text, &mut ids, &mut lens);
    }
    (ids, lens)
}

/// encode_files core for the BPE backend. With no separator each file is one
/// document (small files are grouped, huge ones split at pretoken-safe
/// boundaries); otherwise each file is cut into byte regions at document
/// boundaries and documents are extracted while encoding.
pub(crate) fn encode_files_docs(
    workers: &WorkerPool,
    proto: &Tokenizer,
    files: &[&[u8]],
    format: &DocFormat,
) -> (Vec<u32>, Vec<i64>) {
    if matches!(format, DocFormat::Text { separator: None }) {
        return encode_docs_ragged(workers, proto, files);
    }
    let total: usize = files.iter().map(|f| f.len()).sum();
    let chunks: Vec<EncodeChunk> = region_chunks(files, format, total, lpt_from_env())
        .into_iter()
        .map(|bytes| EncodeChunk::Region { bytes, format })
        .collect();
    encode_chunks_gathered(workers, proto, &chunks, total)
}

/// Sequential `encode_files_docs`: extract and encode every document in
/// file order on the calling thread, never touching rayon. Document
/// iteration matches the parallel path's chunk regions (`for_each_doc`
/// over the same format), so the output is token- and order-identical.
pub(crate) fn encode_files_docs_serial(
    workers: &WorkerPool,
    proto: &Tokenizer,
    files: &[&[u8]],
    format: &DocFormat,
) -> (Vec<u32>, Vec<i64>) {
    let total: usize = files.iter().map(|f| f.len()).sum();
    workers.with_serial_worker(proto, total, |tok| {
        let mut ids: Vec<u32> = Vec::with_capacity(total / 4 + 16);
        // Huge pages before the encode's stores fault the buffer in (as in
        // encode_chunk); serially the one buffer is the whole result.
        madvise_hugepage(ids.as_mut_ptr() as *mut u8, ids.capacity() * 4);
        let mut lens = Vec::new();
        for &bytes in files {
            for_each_doc(bytes, format, |doc| encode_into(tok, doc, &mut ids, &mut lens));
        }
        (ids, lens)
    })
}

/// encode_files core for the SentencePiece backend. With no separator each
/// file is one document (small files are grouped, huge ones split at
/// scanner-safe boundaries); otherwise each file is cut into byte regions at
/// document boundaries and each region's documents are encoded with a
/// pooled Encoder. Documents are assumed to be valid UTF-8.
pub(crate) fn sp_encode_files_docs(
    tokenizer: &bpe::SentencePieceBPE,
    files: &[&[u8]],
    format: &DocFormat,
) -> (Vec<u32>, Vec<i64>) {
    if matches!(format, DocFormat::Text { separator: None }) {
        // Whole-file documents: same grouping and oversized-document
        // fragmenting as the batch path (mirrors `encode_files_docs`).
        let texts: Vec<&str> = files
            .iter()
            // SAFETY: file contents are trusted valid UTF-8 (encode_files'
            // documented contract, like the unchecked conversion in
            // `sp_encode_chunk`).
            .map(|&bytes| unsafe { std::str::from_utf8_unchecked(bytes) })
            .collect();
        return sp_encode_docs_ragged(tokenizer, &texts);
    }
    let total: usize = files.iter().map(|f| f.len()).sum();
    let chunks: Vec<SpChunk> = region_chunks(files, format, total, lpt_from_env())
        .into_iter()
        .map(|bytes| SpChunk::Region { bytes, format })
        .collect();
    sp_encode_chunks(tokenizer, &chunks, total, SharedUnitsParams::DEFAULT)
}

/// Sequential `sp_encode_files_docs`: one Encoder over every file's
/// documents in order, on the calling thread, never touching rayon.
pub(crate) fn sp_encode_files_docs_serial(
    tokenizer: &bpe::SentencePieceBPE,
    files: &[&[u8]],
    format: &DocFormat,
) -> (Vec<u32>, Vec<i64>) {
    let mut encoder = tokenizer.encoder();
    let mut ids: Vec<u32> = Vec::new();
    let mut lens: Vec<i64> = Vec::new();
    for &bytes in files {
        for_each_doc(bytes, format, |doc| {
            let text = unsafe { std::str::from_utf8_unchecked(doc) };
            sp_encode_into(&mut encoder, text, &mut ids, &mut lens);
        });
    }
    (ids, lens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The parallel chunked path — LPT descending chunk sizes, pooled
    /// pre-sized workers, overlapped Committer gather (with
    /// collect-then-gather as the fallback) — must be
    /// token-identical, in the same order, to a serial per-document
    /// encode, with LPT both on and off (GIGATOK_NO_LPT), passed
    /// explicitly so no process env is mutated. A byte-level vocab makes
    /// any misordered or dropped chunk visible in the flat buffer, not
    /// just in the counts.
    #[test]
    fn parallel_ragged_matches_serial() {
        let merges = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        let vocab = (0..=u8::MAX).map(|b| vec![b]).collect();
        let proto = Tokenizer::new(merges, vocab, None);

        // Deterministic pseudo-text with plenty of alnum-space-alpha cut
        // points for safe_split_ranges.
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut text = |len: usize| -> Vec<u8> {
            (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let r = (state >> 33) as usize;
                    b"abcdefghijklmnopqrstuvwxyz0123456789    "[r % 40]
                })
                .collect()
        };
        // Mid-size docs that group into Docs chunks, one oversized doc
        // that splits into Fragment chunks (spanning the head/tail
        // boundary, so both fragment sizes appear), then small docs so
        // continuation rows land mid-output.
        let mut owned: Vec<Vec<u8>> = Vec::new();
        for _ in 0..30 {
            owned.push(text(300 << 10));
        }
        owned.push(text(12 << 20));
        for _ in 0..30 {
            owned.push(text(100 << 10));
        }
        let docs: Vec<&[u8]> = owned.iter().map(|d| d.as_slice()).collect();

        let mut ids_ref: Vec<u32> = Vec::new();
        let mut lens_ref: Vec<i64> = Vec::new();
        let mut serial = proto.fork();
        for doc in &docs {
            encode_into(&mut serial, doc, &mut ids_ref, &mut lens_ref);
        }

        for lpt in [true, false] {
            // A fresh pool per shape so each run exercises the pre-sized
            // fork (slots fork lazily on first use).
            let workers = WorkerPool::new();
            let (flat, lens) = encode_docs_ragged_with(&workers, &proto, &docs, lpt);
            assert_eq!(lens, lens_ref, "lens mismatch (lpt={lpt})");
            assert_eq!(flat, ids_ref, "ids mismatch (lpt={lpt})");
        }

        // The sequential entry point (bindings' parallel=false) must match
        // too — it runs the reference loop through the pool's serial worker.
        let workers = WorkerPool::new();
        let (flat, lens) = encode_docs_ragged_serial(&workers, &proto, &docs);
        assert_eq!(lens, lens_ref, "lens mismatch (serial)");
        assert_eq!(flat, ids_ref, "ids mismatch (serial)");
    }

    /// Assert token identity with a readable failure: the position of the
    /// first divergence and a short window of both streams (a plain
    /// assert_eq would print millions of ids).
    #[track_caller]
    fn assert_ids_match(tag: &str, ids: &[u32], ids_ref: &[u32]) {
        if ids == ids_ref {
            return;
        }
        let i = ids_ref
            .iter()
            .zip(ids)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| ids_ref.len().min(ids.len()));
        panic!(
            "{tag}: ids mismatch at token {i}: serial[{i}..] = {:?}, fragmented[{i}..] = {:?}",
            &ids_ref[i..(i + 8).min(ids_ref.len())],
            &ids[i..(i + 8).min(ids.len())],
        );
    }

    /// SentencePiece parallel chunked encode — grouped documents plus
    /// oversized documents split into continuation fragments at
    /// scanner-safe unit boundaries — must be token- and order-identical
    /// to the serial one-encoder path. The cached HF models cover the raw
    /// fast-path shapes: TinyLlama (unguarded ▁ prepend), gemma-2b
    /// (SpaceRuns with `>▁</`-style crossing pieces), gemma-3 (guarded
    /// prepend, 6.4k added tokens). The input is wall-to-wall
    /// boundary-hostile content — whitespace runs, literal ▁, multi-byte
    /// UTF-8 with combining marks, split punctuation, crossing-piece text,
    /// added tokens mid-text and whitespace-adjacent — and a small explicit
    /// target forces fragment boundaries all through it.
    #[test]
    fn sp_parallel_fragmented_matches_serial() {
        let models = [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "unsloth/gemma-2b",
            "google/gemma-3-4b-it",
        ];
        let block = concat!(
            "The  quick   brown fox \u{2014} jumps; over, the: lazy. dog!  \n\n",
            "\t\tindent()\twide  spacing      here \r\nCRLF\r\n",
            "<s>raw<s> tokens </s> spaced <unk> out <bos> and <eos>\n",
            "<start_of_turn>user hello<end_of_turn> <pad>\n",
            "html-ish <b> </b> crossing > </ pieces >\u{2581}</ literal mark \u{2581}word\n",
            "日本語のテキストと émojis 🎉🚀, combining a\u{301}e\u{308}o\u{302}, nbsp\u{a0}here ½ ﬁ ㎒\n",
            "   leading and trailing   \n",
        );
        let mut big = String::new();
        while big.len() < (6 << 20) {
            big.push_str(block);
        }
        let texts: Vec<&str> = vec![
            "short doc <s> one",
            block,
            &big,
            "",
            "tail doc </s>",
            block,
        ];
        for repo in models {
            let Some(path) = crate::test_hub::hf_tokenizer_json(repo) else {
                eprintln!("Skipping {repo}: tokenizer.json not in the HF cache");
                continue;
            };
            let tok = crate::load_tokenizer::hf::load_hf_sentencepiece(&path).unwrap();
            assert!(
                tok.supports_fragment_split(),
                "{repo} should take the raw fast path (fragment splitting)"
            );
            let (ids_ref, lens_ref) = sp_encode_docs_ragged_serial(&tok, &texts);

            // The public parallel path (default target sizing).
            let (ids, lens) = sp_encode_docs_ragged(&tok, &texts);
            assert_eq!(lens, lens_ref, "{repo}: lens mismatch (default target)");
            assert_ids_match(&format!("{repo} (default target)"), &ids, &ids_ref);

            // A small target forces ~a hundred fragment boundaries inside
            // the hostile content.
            let chunks = sp_build_chunks(&tok, &texts, 64 << 10);
            let fragments = chunks
                .iter()
                .filter(|c| matches!(c, SpChunk::Fragment { .. }))
                .count();
            assert!(
                fragments > 50,
                "{repo}: expected many fragments, got {fragments}"
            );
            let (ids, lens) = sp_encode_chunks(
                &tok,
                &chunks,
                texts.iter().map(|t| t.len()).sum(),
                SharedUnitsParams::DEFAULT,
            );
            assert_eq!(lens, lens_ref, "{repo}: lens mismatch (small target)");
            assert_ids_match(&format!("{repo} (small target)"), &ids, &ids_ref);
        }
    }

    /// Fragment cuts vs added-token edge cases on synthetic models covering
    /// every raw-prepend shape (`Unguarded`, where a mis-placed cut would
    /// inject a spurious ▁; `GuardedAlways` + `EveryMark`; `GuardedFirst`),
    /// with lstrip- and rstrip-flagged tokens whose whitespace trimming must
    /// never straddle a cut, a space-carrying token that can straddle one,
    /// and a self-overlapping token. A tiny target tries a cut every few
    /// bytes, so every blocked interval edge in the text gets exercised.
    #[test]
    fn sp_fragment_cuts_respect_added_tokens() {
        use crate::load_tokenizer::hf::load_hf_slice;
        // Char vocab + byte fallback, no merges (unit and section boundary
        // divergence is fully visible in char-level ids), Llama-2-style
        // normalizer, and added tokens with every strip shape.
        let mut vocab_entries: Vec<String> = (0u16..=255)
            .map(|b| format!("\"<0x{b:02X}>\": {b}"))
            .collect();
        let mut next_id = 256u32;
        vocab_entries.push(format!("\"\u{2581}\": {next_id}"));
        next_id += 1;
        for c in "abcdefghijklmnopqrstuvwxyz.,!?".chars() {
            vocab_entries.push(format!("\"{c}\": {next_id}"));
            next_id += 1;
        }
        let added = [
            ("<p>", false, false),
            ("<l>", true, false),
            ("<r>", false, true),
            ("w w", false, false), // can straddle a space cut
            ("aa", false, false),  // self-overlapping occurrences
        ];
        let added_json: Vec<String> = added
            .iter()
            .map(|(content, lstrip, rstrip)| {
                let id = next_id;
                next_id += 1;
                format!(
                    "{{\"id\": {id}, \"content\": \"{content}\", \"lstrip\": {lstrip}, \
                     \"rstrip\": {rstrip}, \"normalized\": false, \"special\": true}}"
                )
            })
            .collect();
        // Every raw-prepend shape a cut interacts with: Llama-2's Prepend
        // normalizer (`Unguarded`, where the added-token `e`-blocking is
        // load-bearing), Metaspace always+split (`GuardedAlways` +
        // `EveryMark`: cuts inside mark runs, `e` cuts allowed), and
        // Metaspace first without split (`GuardedFirst` + `SpaceRuns`).
        use crate::bpe::sentencepiece::{RawPrepend, WordSplit};
        let pipelines = [
            (
                "{\"type\": \"Sequence\", \"normalizers\": [\
                   {\"type\": \"Prepend\", \"prepend\": \"\u{2581}\"}, \
                   {\"type\": \"Replace\", \"pattern\": {\"String\": \" \"}, \"content\": \"\u{2581}\"}]}",
                "null",
                RawPrepend::Unguarded,
                WordSplit::SpaceRuns,
            ),
            (
                "null",
                "{\"type\": \"Metaspace\", \"replacement\": \"\u{2581}\", \
                  \"prepend_scheme\": \"always\", \"split\": true}",
                RawPrepend::GuardedAlways,
                WordSplit::EveryMark,
            ),
            (
                "null",
                "{\"type\": \"Metaspace\", \"replacement\": \"\u{2581}\", \
                  \"prepend_scheme\": \"first\", \"split\": false}",
                RawPrepend::GuardedFirst,
                WordSplit::SpaceRuns,
            ),
        ];

        // Tokens adjacent to and inside whitespace runs, back to back, and
        // overlapping ("aaa" holds two "aa" occurrences); words with the
        // split punctuation; a "w w" occurrence wherever the text has one.
        let block = concat!(
            "plain words here <p> and <p><p> doubled\n",
            "lstrip near ws   <l> and far<l>tight\n",
            "rstrip eats ws <r>   after and<r>tight\n",
            "aaa aaaa a aa overlapping aa\n",
            "w w w w w straddling spaces\n",
            "punct, split. here! ok? more,words.now\n",
            "   runs\t\tof   whitespace \n\n",
            "<l>   <r>   <l><r> adjacent tokens\n",
        );
        let mut text = String::new();
        while text.len() < 100_000 {
            text.push_str(block);
        }
        let texts = [text.as_str()];

        for (normalizer, pre_tokenizer, prepend, word_split) in pipelines {
            let json = format!(
                "{{\"added_tokens\": [{}], \
                  \"normalizer\": {normalizer}, \
                  \"pre_tokenizer\": {pre_tokenizer}, \
                  \"model\": {{\"type\": \"BPE\", \"byte_fallback\": true, \
                    \"vocab\": {{{}}}, \"merges\": []}}}}",
                added_json.join(", "),
                vocab_entries.join(", "),
            );
            let tok = match load_hf_slice(json.as_bytes()).unwrap() {
                crate::load_tokenizer::hf::HfTokenizer::SentencePiece(tok) => tok,
                _ => panic!("synthetic model should load as SentencePiece"),
            };
            // Pin the shape under test — a loader change that silently lands
            // on another fast path would hollow the test out.
            assert_eq!(tok.raw_prepend, Some(prepend), "unexpected raw prepend");
            assert_eq!(tok.word_split, word_split, "unexpected word split");

            let (ids_ref, lens_ref) = sp_encode_docs_ragged_serial(&tok, &texts);
            // A cut attempt every ~48 bytes: thousands of boundaries,
            // hitting every edge of every blocked interval shape in the
            // block.
            let chunks = sp_build_chunks(&tok, &texts, 48);
            assert!(
                chunks.len() > 1000,
                "expected a cut attempt every few bytes, got {} chunks",
                chunks.len()
            );
            let (ids, lens) = sp_encode_chunks(
                &tok,
                &chunks,
                texts.iter().map(|t| t.len()).sum(),
                SharedUnitsParams::DEFAULT,
            );
            assert_eq!(lens, lens_ref, "{prepend:?}: lens mismatch");
            assert_ids_match(&format!("{prepend:?}"), &ids, &ids_ref);
        }
    }

    /// The shared unit table path — normally reached only by inputs of
    /// 16 MiB and up — forced onto a 2-16 slot table, with a tiny cache
    /// budget (every state wipes repeatedly) and several encodes on one
    /// pool whose states and table stay warm in between, must be token-
    /// and order-identical to the serial one-encoder encode. The words make
    /// units of 2-4 tokens (inline front payload), 6-11 (shared-table
    /// entries past the first four tokens) and 15 (never shared).
    #[test]
    fn sp_shared_table_pool_matches_serial() {
        use crate::load_tokenizer::hf::{HfTokenizer, load_hf_slice};

        // ▁, 16 letters and four merged pairs on top of the byte-fallback
        // tokens; Llama-2-style normalizer (raw fast path).
        let mut vocab_entries: Vec<String> = (0u16..=255)
            .map(|b| format!("\"<0x{b:02X}>\": {b}"))
            .collect();
        let mut next_id = 256u32;
        vocab_entries.push(format!("\"\u{2581}\": {next_id}"));
        let pieces = "abcdefghijklmnop"
            .chars()
            .map(String::from)
            .chain(["ab", "cd", "ef", "gh"].map(String::from));
        for piece in pieces {
            next_id += 1;
            vocab_entries.push(format!("\"{piece}\": {next_id}"));
        }
        let json = format!(
            "{{\"added_tokens\": [], \
              \"normalizer\": {{\"type\": \"Sequence\", \"normalizers\": [\
                {{\"type\": \"Prepend\", \"prepend\": \"\u{2581}\"}}, \
                {{\"type\": \"Replace\", \"pattern\": {{\"String\": \" \"}}, \
                  \"content\": \"\u{2581}\"}}]}}, \
              \"pre_tokenizer\": null, \
              \"model\": {{\"type\": \"BPE\", \"byte_fallback\": true, \
                \"vocab\": {{{}}}, \
                \"merges\": [[\"a\", \"b\"], [\"c\", \"d\"], [\"e\", \"f\"], [\"g\", \"h\"]]}}}}",
            vocab_entries.join(", ")
        );
        let load = || match load_hf_slice(json.as_bytes()).unwrap() {
            HfTokenizer::SentencePiece(tok) => tok,
            _ => panic!("synthetic model should load as SentencePiece"),
        };
        let reference = load();
        let mut tok = load();
        // Floored to the smallest budget: 1 MiB for the growing caches.
        tok.set_max_cache_bytes(Some(0));

        let mut state = 0x853C_49E6_748F_EA9Bu64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        let mut words = |n: usize| -> String {
            let mut text = String::new();
            for i in 0..n {
                if i > 0 {
                    text.push(' ');
                }
                let len = match next() % 3 {
                    0 => 1 + next() % 3,
                    1 => 5 + next() % 6,
                    _ => 14,
                };
                for _ in 0..len {
                    text.push((b'a' + (next() % 16) as u8) as char);
                }
            }
            text
        };
        let owned_a = [
            words(12_000),
            words(1),
            String::new(),
            words(24_000),
            words(200),
        ];
        let owned_b = [words(18_000), words(300), words(15_000)];
        let texts_a: Vec<&str> = owned_a.iter().map(String::as_str).collect();
        let texts_b: Vec<&str> = owned_b.iter().map(String::as_str).collect();
        let inputs = [&texts_a, &texts_b];
        let refs = inputs.map(|texts| sp_encode_docs_ragged_serial(&reference, texts));

        // Two worker threads, so each state sees enough distinct units to
        // wipe whatever the machine's core count.
        let rayon_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        rayon_pool.install(|| {
            for log2 in [1u32, 2, 4] {
                let params = SharedUnitsParams {
                    min_input: 0,
                    min_log2: log2,
                    max_log2: log2,
                };
                let pool = SpCallPool::new(&tok, 0, params);
                for (round, which) in [0usize, 1, 0].into_iter().enumerate() {
                    let texts = inputs[which];
                    let total: usize = texts.iter().map(|t| t.len()).sum();
                    let chunks = sp_build_chunks(&tok, texts, 16 << 10);
                    assert!(
                        chunks.len() >= 8,
                        "expected many chunks, got {}",
                        chunks.len()
                    );
                    let (ids, lens) = sp_encode_chunks_in(&pool, &chunks, total);
                    let (ids_ref, lens_ref) = &refs[which];
                    let tag = format!("log2={log2} round={round}");
                    assert_eq!(&lens, lens_ref, "{tag}: lens mismatch");
                    assert_ids_match(&tag, &ids, ids_ref);
                }
                assert!(pool.shared.is_some(), "log2={log2}: no shared table");
                let wipes: u32 = pool
                    .slots
                    .iter()
                    .map(|slot| slot.lock().unwrap().as_ref().map_or(0, |s| s.wipes()))
                    .sum();
                assert!(wipes > 0, "log2={log2}: the tiny budget never wiped");
                pool.release();
            }
        });
    }

    /// SentencePiece parallel-vs-serial on REAL text: ~290 MB of OWT
    /// (owt_valid) as one huge document plus a few multi-MB slices, for
    /// each cached SP model. Token AND order identity against the serial
    /// one-encoder encode.
    /// `cargo test --release verify_sp_parallel_matches_serial_owt -- --ignored --nocapture`
    #[test]
    #[ignore = "reads ~290 MB of OWT; run explicitly in release mode"]
    fn verify_sp_parallel_matches_serial_owt() {
        let path = std::env::home_dir().unwrap().join("data/owt_valid.txt");
        let input = std::fs::read(&path).expect("read ~/data/owt_valid.txt");
        let text = match std::str::from_utf8(&input) {
            Ok(t) => t,
            Err(e) => std::str::from_utf8(&input[..e.valid_up_to()]).unwrap(),
        };
        // One huge doc (the whole file) plus a few multi-MB slices cut at
        // char boundaries, so grouped-doc and fragment chunks both appear.
        let mut texts: Vec<&str> = vec![text];
        let mut off = 0usize;
        for mb in [3, 7, 12] {
            let mut end = (off + (mb << 20)).min(text.len());
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            texts.push(&text[off..end]);
            off = end;
        }
        for repo in [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "unsloth/gemma-2b",
            "google/gemma-3-4b-it",
        ] {
            let Some(path) = crate::test_hub::hf_tokenizer_json(repo) else {
                eprintln!("Skipping {repo}: tokenizer.json not in the HF cache");
                continue;
            };
            let tok = crate::load_tokenizer::hf::load_hf_sentencepiece(&path).unwrap();
            let t0 = std::time::Instant::now();
            let (ids_ref, lens_ref) = sp_encode_docs_ragged_serial(&tok, &texts);
            let t_serial = t0.elapsed();
            let t0 = std::time::Instant::now();
            let (ids, lens) = sp_encode_docs_ragged(&tok, &texts);
            let t_par = t0.elapsed();
            eprintln!(
                "{repo}: {} tokens; serial {:.2?}, parallel {:.2?}",
                ids_ref.len(),
                t_serial,
                t_par
            );
            assert_eq!(lens, lens_ref, "{repo}: lens mismatch");
            assert_ids_match(repo, &ids, &ids_ref);
        }
    }

    /// Parallel-vs-serial at scale on a REAL tokenizer: ~1 GB of OWT as a
    /// multi-doc group (small grouped docs, mid docs, oversized docs that
    /// fragment at pretoken-safe boundaries) with `<|endoftext|>` injected
    /// mid-doc and doc-final, LPT on and off. Token AND order identity
    /// against a serial per-document encode. A few seconds in release mode
    /// (both sides use the cached encode).
    /// `cargo test --release verify_parallel_ragged_matches_serial_owt_gpt2_1g -- --ignored --nocapture`
    #[test]
    #[ignore = "reads 1 GB of OWT; run explicitly in release mode"]
    fn verify_parallel_ragged_matches_serial_owt_gpt2_1g() {
        use crate::load_tokenizer::hf::load_hf_bpe;
        use std::io::Read;
        let tokenizer_path = crate::test_hub::gpt2_tokenizer_json();
        let proto = load_hf_bpe(&tokenizer_path).expect("load GPT-2 tokenizer");
        let added = proto.added_token_split_blockers();
        let sep: Vec<u8> = added.first().expect("GPT-2 has an added token").0.to_vec();

        let path = std::env::home_dir().unwrap().join("data/owt_train.txt");
        let f = std::fs::File::open(&path).expect("open ~/data/owt_train.txt");
        let mut input = Vec::new();
        f.take(1_000_000_000).read_to_end(&mut input).unwrap();
        while !input.is_empty() && std::str::from_utf8(&input).is_err() {
            input.pop();
        }
        assert!(input.len() > 900_000_000, "corpus too small: {}", input.len());

        // Doc size pattern: small (group), mid, large; every 20th doc is
        // oversized (24 MB) so it splits into Fragment chunks.
        let sizes = [64 << 10, 300 << 10, 1 << 20, 100 << 10, 3 << 20];
        let mut owned: Vec<Vec<u8>> = Vec::new(); // docs with injected added tokens
        let mut ranges: Vec<(usize, usize, bool)> = Vec::new(); // (start, end, inject)
        let mut pos = 0usize;
        let mut i = 0usize;
        while pos < input.len() {
            let want = if i % 20 == 19 { 24 << 20 } else { sizes[i % sizes.len()] };
            let end = (pos + want).min(input.len());
            // Inject the added token into every 7th doc (mid + tail).
            ranges.push((pos, end, i % 7 == 3));
            pos = end;
            i += 1;
        }
        for &(s, e, inject) in &ranges {
            if inject {
                let piece = &input[s..e];
                let mid = piece.len() / 2;
                let mut doc = Vec::with_capacity(piece.len() + 2 * sep.len());
                doc.extend_from_slice(&piece[..mid]);
                doc.extend_from_slice(&sep);
                doc.extend_from_slice(&piece[mid..]);
                doc.extend_from_slice(&sep);
                owned.push(doc);
            }
        }
        let mut docs: Vec<&[u8]> = Vec::with_capacity(ranges.len());
        let mut oi = 0usize;
        for &(s, e, inject) in &ranges {
            if inject {
                docs.push(&owned[oi]);
                oi += 1;
            } else {
                docs.push(&input[s..e]);
            }
        }
        eprintln!(
            "{} docs ({} with injected {:?}), {} bytes total",
            docs.len(),
            owned.len(),
            String::from_utf8_lossy(&sep),
            docs.iter().map(|d| d.len()).sum::<usize>()
        );

        let mut ids_ref: Vec<u32> = Vec::new();
        let mut lens_ref: Vec<i64> = Vec::new();
        let mut serial = proto.fork();
        for doc in &docs {
            encode_into(&mut serial, doc, &mut ids_ref, &mut lens_ref);
        }
        drop(serial);
        eprintln!("serial reference: {} tokens", ids_ref.len());

        for lpt in [true, false] {
            let workers = WorkerPool::new();
            let (flat, lens) = encode_docs_ragged_with(&workers, &proto, &docs, lpt);
            assert_eq!(lens, lens_ref, "lens mismatch (lpt={lpt})");
            if flat != ids_ref {
                let i = ids_ref
                    .iter()
                    .zip(&flat)
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| ids_ref.len().min(flat.len()));
                panic!(
                    "ids mismatch (lpt={lpt}) at token {i}: serial[{i}..] = {:?}, parallel[{i}..] = {:?}",
                    &ids_ref[i..(i + 8).min(ids_ref.len())],
                    &flat[i..(i + 8).min(flat.len())],
                );
            }
            eprintln!("lpt={lpt}: {} tokens identical", flat.len());
        }
    }

    /// The overlapped gather's escape hatches must be output-identical to
    /// the committed path: cap 0 stands in for a refused up-front
    /// reservation (no committer at all), cap 1 overflows on the first
    /// commit, and a mid-range cap overflows mid-flight after a real
    /// prefix has been committed — the fallback must discard that prefix
    /// and re-gather from the (intact) chunk buffers.
    #[test]
    fn gather_fallbacks_match() {
        let merges = HashMap::with_hasher(rustc_hash::FxBuildHasher {});
        let vocab = (0..=u8::MAX).map(|b| vec![b]).collect();
        let proto = Tokenizer::new(merges, vocab, None);

        let mut state = 0xD1B54A32D192ED03u64;
        let mut text = |len: usize| -> Vec<u8> {
            (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let r = (state >> 33) as usize;
                    b"abcdefghijklmnopqrstuvwxyz0123456789    "[r % 40]
                })
                .collect()
        };
        let owned: Vec<Vec<u8>> = (0..20).map(|_| text(1 << 20)).collect();
        let docs: Vec<&[u8]> = owned.iter().map(|d| d.as_slice()).collect();
        let total: usize = docs.iter().map(|d| d.len()).sum();
        let added = proto.added_token_split_blockers();
        let chunks = build_doc_chunks(&docs, total, chunk_target_bytes(total), &added, true);
        assert!(chunks.len() > 1, "test must exercise the parallel path");

        let workers = WorkerPool::new();
        let (flat_ref, lens_ref) = encode_chunks_gathered(&workers, &proto, &chunks, total);
        // Byte-level vocab: one token per byte, so any cap below `total`
        // overflows; total / 3 overflows mid-flight with a committed
        // prefix behind it.
        for cap in [0, 1, total / 3] {
            let workers = WorkerPool::new();
            let (flat, lens) =
                encode_chunks_gathered_with_cap(&workers, &proto, &chunks, total, cap);
            assert_eq!(lens, lens_ref, "lens mismatch (cap={cap})");
            assert_eq!(flat, flat_ref, "ids mismatch (cap={cap})");
        }
    }
}
