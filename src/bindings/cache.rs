//! The process-global cache-budget knob, read once per tokenizer
//! construction. Semantics in the .pyi stubs, mechanism in
//! `Tokenizer::set_max_cache_bytes`.
//!
//! The budget is per encode worker. SentencePiece models additionally hold,
//! outside it, a flat merge table per model and — for the duration of one
//! parallel encode call over a large input — a shared unit table of
//! 32-128 MiB (see `SentencePieceBPE::set_max_cache_bytes`).

use crate::bpe::{SentencePieceBPE, Tokenizer};
use pyo3::prelude::*;
use std::sync::Mutex;

/// The budget applied to tokenizers constructed after the last
/// `set_max_cache_bytes` call; `None` = unbounded.
static MAX_CACHE_BYTES: Mutex<Option<usize>> = Mutex::new(Some(Tokenizer::DEFAULT_MAX_CACHE_BYTES));

/// Apply the global setting to a freshly constructed tokenizer (which
/// already carries the built-in default, hence the `!=` skip).
pub(crate) fn apply_max_cache_bytes(mut tokenizer: Tokenizer) -> Tokenizer {
    let configured = *MAX_CACHE_BYTES.lock().unwrap();
    if configured != tokenizer.max_cache_bytes() {
        tokenizer.set_max_cache_bytes(configured);
    }
    tokenizer
}

/// SentencePiece analog of [`apply_max_cache_bytes`].
pub(crate) fn apply_max_cache_bytes_sp(mut model: SentencePieceBPE) -> SentencePieceBPE {
    model.set_max_cache_bytes(*MAX_CACHE_BYTES.lock().unwrap());
    model
}

#[pyfunction]
#[pyo3(signature = (max_bytes))]
pub(crate) fn set_max_cache_bytes(max_bytes: Option<usize>) {
    *MAX_CACHE_BYTES.lock().unwrap() = max_bytes;
}

#[pyfunction]
pub(crate) fn get_max_cache_bytes() -> Option<usize> {
    *MAX_CACHE_BYTES.lock().unwrap()
}
