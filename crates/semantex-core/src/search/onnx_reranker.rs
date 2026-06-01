//! Generic ONNX cross-encoder reranker.
//!
//! fastembed 5.9's `RerankerModel` enum cannot express newer checkpoints
//! (e.g. Qwen3-Reranker-0.6B), so this module loads any HuggingFace
//! cross-encoder's ONNX model + `tokenizer.json` directly through `ort` +
//! `tokenizers`, mirroring how `embedding/colbert.rs` pins the CPU execution
//! provider. It supports two ways to turn model output into a relevance score:
//!
//! - [`ScoreStrategy::ClassifierLogit`] — sequence-classification head
//!   (bge-style): the model emits a single relevance logit; the score is that
//!   logit.
//! - [`ScoreStrategy::YesNoLogit`] — generative reranker (Qwen3-Reranker-style):
//!   the model emits vocabulary logits; the score is the softmax-normalized
//!   probability mass on the "yes" token vs the "no" token at the final
//!   position, per the values carried by S8's `ModelRegistry` reranker spec
//!   (recorded in `docs/superpowers/plans/2026-05-31-research-notes.md`).
//!
//! All concrete model ids, prompt strings, and token ids come from the
//! reranker spec (`crate::model::RerankerSpec` via `RerankerChoice::from_spec`)
//! — nothing repo-specific is hardcoded here.

use anyhow::{Context, Result};

/// Prompt wrapper for a generative (yes/no) reranker. Strings come from the
/// model's reranker spec (S8 registry / recorded spike), never hardcoded per
/// corpus.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    /// Text emitted before the query (e.g. the instruction header).
    pub prefix: String,
    /// Text emitted between the query and the document.
    pub middle: String,
    /// Text emitted after the document (terminates the prompt so the next-token
    /// logit is the relevance judgment).
    pub suffix: String,
}

impl PromptTemplate {
    /// Render the full prompt string for a (query, document) pair.
    #[must_use]
    pub fn render(&self, query: &str, document: &str) -> String {
        format!(
            "{}{}{}{}{}",
            self.prefix, query, self.middle, document, self.suffix
        )
    }
}

/// How to convert model output logits into a single relevance score.
#[derive(Debug, Clone)]
pub enum ScoreStrategy {
    /// Sequence-classification head: a single relevance logit per pair.
    ClassifierLogit,
    /// Generative reranker: compare the "yes" vs "no" token logits at the
    /// final position. `yes_id`/`no_id` come from the reranker spec.
    YesNoLogit {
        yes_id: usize,
        no_id: usize,
        prompt: PromptTemplate,
    },
}

/// Score for a classifier-head model: the relevance logit is the last element
/// of the flattened output (shape `[batch=1, num_labels]`, num_labels==1 for
/// bge rerankers). Returns the raw logit (monotonic in relevance; the caller
/// only needs ordering).
pub(crate) fn classifier_score_from_logits(logits: &[f32]) -> Result<f32> {
    logits
        .last()
        .copied()
        .context("classifier reranker produced an empty logits tensor")
}

/// Score for a yes/no generative reranker: take the logits at the final
/// sequence position (`final_pos_logits`, length == vocab size), then return
/// the softmax probability of "yes" over {yes, no}:
/// `exp(l_yes) / (exp(l_yes) + exp(l_no))`. Computed in a numerically stable
/// way by subtracting the max of the two logits.
pub(crate) fn yes_no_score_from_logits(
    final_pos_logits: &[f32],
    yes_id: usize,
    no_id: usize,
) -> Result<f32> {
    let l_yes = *final_pos_logits.get(yes_id).with_context(|| {
        format!("yes_id {yes_id} out of range (vocab {})", final_pos_logits.len())
    })?;
    let l_no = *final_pos_logits.get(no_id).with_context(|| {
        format!("no_id {no_id} out of range (vocab {})", final_pos_logits.len())
    })?;
    let m = l_yes.max(l_no);
    let e_yes = (l_yes - m).exp();
    let e_no = (l_no - m).exp();
    Ok(e_yes / (e_yes + e_no))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_score_reads_last_logit() {
        assert_eq!(classifier_score_from_logits(&[0.42]).unwrap(), 0.42);
        // Multi-label safety: still reads the last element.
        assert_eq!(classifier_score_from_logits(&[-1.0, 3.5]).unwrap(), 3.5);
    }

    #[test]
    fn classifier_score_errors_on_empty() {
        assert!(classifier_score_from_logits(&[]).is_err());
    }

    #[test]
    fn yes_no_score_is_monotonic_and_bounded() {
        // Vocab of 4; yes_id=2, no_id=3.
        let logits = [0.0, 0.0, 2.0, 1.0];
        let s = yes_no_score_from_logits(&logits, 2, 3).unwrap();
        // softmax(2,1) over {yes,no} = e^1/(e^1+e^0) = 0.7310586
        assert!((s - 0.731_058_6).abs() < 1e-5, "got {s}");
        // Strong yes -> near 1.0; strong no -> near 0.0.
        assert!(yes_no_score_from_logits(&[0.0, 0.0, 10.0, 0.0], 2, 3).unwrap() > 0.99);
        assert!(yes_no_score_from_logits(&[0.0, 0.0, 0.0, 10.0], 2, 3).unwrap() < 0.01);
    }

    #[test]
    fn yes_no_score_errors_on_oob_token() {
        assert!(yes_no_score_from_logits(&[0.1, 0.2], 5, 0).is_err());
    }

    #[test]
    fn prompt_template_renders_in_order() {
        let t = PromptTemplate {
            prefix: "<I>".into(),
            middle: "<M>".into(),
            suffix: "<S>".into(),
        };
        assert_eq!(t.render("Q", "D"), "<I>Q<M>D<S>");
    }
}
