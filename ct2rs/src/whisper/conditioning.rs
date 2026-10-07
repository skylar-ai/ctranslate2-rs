//! `initial_prompt`/`hotwords` conditioning for [`super::Whisper::generate_segments_conditioned`].
//!
//! Builds the `<|startofprev|>`-prefixed prompt tokens for `initial_prompt`/`hotwords`, mirroring
//! faster-whisper's `get_prompt` token order but applied identically to every chunk (no rolling
//! previous-text conditioning), and capped so the conditioned decode keeps the same output budget
//! as the plain one (see [`conditioning_token_budget`]). When conditioning is requested, each
//! chunk is decoded twice — once with the plain prompt, once with the conditioned one — and the
//! two transcripts are reconciled word by word via
//! [`reconciliation::merge_word_hypotheses`][super::reconciliation::merge_word_hypotheses], so
//! conditioning can bias wording but never silently drop content.

use anyhow::{anyhow, Result};

use super::reconciliation::{join_words, merge_word_hypotheses};
use super::{group_tokens_into_words, is_special_token, process_word_timings};
use super::{sys, Segment, Tokenizer, Whisper, WhisperOptions, Word};

impl Whisper {
    /// Decodes and aligns a single prompt for every chunk, with no conditioning involved. This is
    /// the path used whenever `initial_prompt`/`hotwords` are both `None`.
    pub(super) fn generate_segments_plain(
        &self,
        encoder_output: &sys::StorageView,
        prompt: &[String],
        num_chunks: usize,
        options: &WhisperOptions,
    ) -> Result<Vec<Segment>> {
        let gen_results =
            self.whisper
                .generate(encoder_output, &vec![prompt.to_vec(); num_chunks], options)?;
        let alignment_results =
            self.align_chunks(encoder_output, prompt, &gen_results, num_chunks)?;

        let mut segments = Vec::with_capacity(num_chunks);
        for (chunk_idx, (res, align_res)) in
            gen_results.iter().zip(alignment_results.iter()).enumerate()
        {
            let chunk_offset =
                (chunk_idx * self.config.n_samples) as f32 / self.config.sampling_rate as f32;
            let tokens = &res.sequences[0];

            // `tokens` (`res.sequences[0]`) never echoes back the forced prompt beyond its
            // single trailing task token (`<|transcribe|>`/`<|notimestamps|>`). CTranslate2
            // feeds everything before that purely through the decoder's KV cache and only
            // returns the newly generated continuation (see `WhisperReplica::generate` in
            // `CTranslate2/src/models/whisper.cc`). That leading task token is filtered out
            // below like any other special token.
            let words = self.words_from_tokens(tokens, align_res, chunk_offset)?;

            let clean_tokens: Vec<String> = tokens
                .iter()
                .filter(|t| !is_special_token(t))
                .cloned()
                .collect();
            let chunk_text = self.tokenizer.decode(clean_tokens)?.trim().to_string();

            let seg_start = words.first().map(|w| w.start).unwrap_or(chunk_offset);
            let seg_end = words.last().map(|w| w.end).unwrap_or(chunk_offset);

            segments.push(Segment {
                id: chunk_idx,
                text: chunk_text,
                start: seg_start,
                end: seg_end,
                words: Some(words),
            });
        }

        Ok(segments)
    }

    /// Decodes every chunk with both the conditioned and the plain prompt, then reconciles the
    /// two transcripts word by word instead of picking one whole chunk as the winner.
    ///
    /// Forced-prompt conditioning biases the whole decode trajectory, not just the target word,
    /// so on real audio it can occasionally derail unrelated content elsewhere in the same chunk.
    /// A whole-chunk "keep whichever scored better" comparison would still accept that collateral
    /// damage as long as the rest of the chunk improved enough to raise its average score.
    /// Reconciling word by word (see
    /// [`reconciliation::merge_word_hypotheses`][super::reconciliation::merge_word_hypotheses])
    /// avoids that: matching words are kept as-is, a word conditioning changed is kept only if
    /// its local confidence improved, and content conditioning dropped entirely is never accepted
    /// silently: the baseline's words are used for that span instead.
    ///
    /// Cost: roughly 2x decode and 2x alignment work for chunks that use conditioning (nothing
    /// extra when `initial_prompt`/`hotwords` aren't set, since this path isn't taken then).
    pub(super) fn generate_segments_with_conditioning(
        &self,
        encoder_output: &sys::StorageView,
        baseline_prompt: &[String],
        conditioned_prompt: &[String],
        num_chunks: usize,
        options: &WhisperOptions,
    ) -> Result<Vec<Segment>> {
        // These must be two separate `generate()` calls, not one batch of both prompt kinds:
        // CTranslate2 requires every prompt within a single batch to have
        // `<|startoftranscript|>` at the same index, which the conditioned prompt (longer,
        // prefixed with the conditioning tokens) and the plain prompt (unprefixed) don't share.
        let conditioned_results = self.whisper.generate(
            encoder_output,
            &vec![conditioned_prompt.to_vec(); num_chunks],
            options,
        )?;
        let baseline_results = self.whisper.generate(
            encoder_output,
            &vec![baseline_prompt.to_vec(); num_chunks],
            options,
        )?;

        // Each candidate is aligned with its own matching prompt, so word-level confidence
        // (used by `merge_word_hypotheses` below) is measured fairly for both, not skewed by
        // forcing one candidate's words through the other's decoder context.
        let conditioned_alignments = self.align_chunks(
            encoder_output,
            conditioned_prompt,
            &conditioned_results,
            num_chunks,
        )?;
        let baseline_alignments = self.align_chunks(
            encoder_output,
            baseline_prompt,
            &baseline_results,
            num_chunks,
        )?;

        let mut segments = Vec::with_capacity(num_chunks);
        for chunk_idx in 0..num_chunks {
            let chunk_offset =
                (chunk_idx * self.config.n_samples) as f32 / self.config.sampling_rate as f32;

            let cond_words = self.words_from_tokens(
                &conditioned_results[chunk_idx].sequences[0],
                &conditioned_alignments[chunk_idx],
                chunk_offset,
            )?;
            let base_words = self.words_from_tokens(
                &baseline_results[chunk_idx].sequences[0],
                &baseline_alignments[chunk_idx],
                chunk_offset,
            )?;

            let words = merge_word_hypotheses(&base_words, &cond_words);
            let seg_start = words.first().map(|w| w.start).unwrap_or(chunk_offset);
            let seg_end = words.last().map(|w| w.end).unwrap_or(chunk_offset);
            let text = join_words(&words);

            segments.push(Segment {
                id: chunk_idx,
                text,
                start: seg_start,
                end: seg_end,
                words: Some(words),
            });
        }

        Ok(segments)
    }

    /// Extracts word-level text, timing and per-word confidence for one chunk from its decoded
    /// tokens and matching DTW alignment.
    fn words_from_tokens(
        &self,
        tokens: &[String],
        align_res: &sys::WhisperAlignmentResult,
        chunk_offset: f32,
    ) -> Result<Vec<Word>> {
        let word_token_ranges = group_tokens_into_words(tokens);
        let chunk_words = process_word_timings(
            &word_token_ranges,
            &align_res.alignments,
            &align_res.text_token_probs,
            tokens.len(),
        );

        let mut final_words = Vec::new();
        for (range, mut word) in word_token_ranges.into_iter().zip(chunk_words) {
            let word_text = self.tokenizer.decode(tokens[range.clone()].to_vec())?;
            let clean_word_text = word_text.trim().to_string();
            if clean_word_text.is_empty() {
                continue;
            }
            word.word = clean_word_text;
            word.start += chunk_offset;
            word.end += chunk_offset;
            final_words.push(word);
        }
        Ok(final_words)
    }

    /// Builds the `<|startofprev|>`-prefixed conditioning tokens for `hotwords`/`initial_prompt`,
    /// mirroring faster-whisper's `get_prompt` token order, but applied identically to every chunk
    /// (no rolling previous-text conditioning). At most `budget` content tokens are kept in total
    /// (see [`conditioning_token_budget`] and [`fit_to_budget`]). Blank (empty/whitespace-only)
    /// text counts as not provided. Returns an empty vector if neither is provided, in which case
    /// callers see no behavior change at all.
    pub(super) fn build_conditioning_prefix(
        &self,
        initial_prompt: Option<&str>,
        hotwords: Option<&str>,
        budget: usize,
    ) -> Result<Vec<String>> {
        let initial_prompt = non_blank(initial_prompt);
        let hotwords = non_blank(hotwords);
        if initial_prompt.is_none() && hotwords.is_none() {
            return Ok(Vec::new());
        }

        let hotwords_tokens = hotwords.map(|hw| self.encode_raw(hw)).transpose()?;
        let prompt_tokens = initial_prompt
            .map(|prompt| self.encode_raw(prompt))
            .transpose()?;
        let (hotwords_tokens, prompt_tokens) = fit_to_budget(
            hotwords_tokens.unwrap_or_default(),
            prompt_tokens.unwrap_or_default(),
            budget,
        );

        let mut prefix = vec!["<|startofprev|>".to_string()];
        prefix.extend(hotwords_tokens);
        prefix.extend(prompt_tokens);
        Ok(prefix)
    }

    /// Encodes free text into raw BPE token piece strings, bypassing the wrapping
    /// `hf::Tokenizer`'s special-token template (we only want the plain content pieces, not the
    /// model's own start/end-of-sequence tokens injected around them). Uses faster-whisper's
    /// leading-space convention (`" " + text.trim()`) for consistent piece boundaries.
    fn encode_raw(&self, text: &str) -> Result<Vec<String>> {
        let with_leading_space = format!(" {}", text.trim());
        (*self.tokenizer)
            .encode(with_leading_space.as_str(), false)
            .map(|encoding| encoding.get_tokens().to_vec())
            .map_err(|err| anyhow!("failed to encode conditioning text: {err}"))
    }

    /// Runs DTW alignment for a batch of chunks that were all decoded from the same forced
    /// `prompt` (`align()` applies one shared `start_sequence` to its whole batch, so chunks
    /// decoded with different prompts must go through separate calls).
    fn align_chunks(
        &self,
        encoder_output: &sys::StorageView,
        prompt: &[String],
        gen_results: &[sys::WhisperGenerationResult],
        num_chunks: usize,
    ) -> Result<Vec<sys::WhisperAlignmentResult>> {
        let start_seq: Vec<usize> = prompt
            .iter()
            .map(|t| {
                self.tokenizer
                    .token_to_id(t)
                    .map(|id| id as usize)
                    .unwrap_or(0)
            })
            .collect();
        let num_frames = vec![self.config.nb_max_frames; num_chunks];
        let text_tokens: Vec<Vec<usize>> = gen_results
            .iter()
            .map(|res| res.sequences_ids[0].clone())
            .collect();

        self.whisper
            .align(encoder_output, &start_seq, &text_tokens, &num_frames, 7)
    }
}

/// Treats empty/whitespace-only text as absent, so e.g. a blank form field or an empty proto3
/// `string` can't trigger conditioning with no actual content.
fn non_blank(text: Option<&str>) -> Option<&str> {
    text.filter(|t| !t.trim().is_empty())
}

/// Maximum number of `hotwords` + `initial_prompt` content tokens for a decode whose
/// start-of-transcript sequence is `sot_len` tokens long.
///
/// CTranslate2 lets a Whisper decode generate `min(max_length / 2, max_length - prompt_len)` new
/// tokens, and alignment then re-feeds `prompt + <|notimestamps|> + text + <|eot|>` through the
/// decoder, which only has `max_length` positions. Keeping the whole conditioned prompt
/// (`<|startofprev|>` + content + sot sequence) at most `max_length / 2 - 2` tokens therefore
/// leaves the conditioned decode the same `max_length / 2` output budget as the plain one, and
/// guarantees its alignment fits too.
pub(super) fn conditioning_token_budget(max_length: usize, sot_len: usize) -> usize {
    (max_length / 2).saturating_sub(sot_len + 3)
}

/// Fits `hotwords` and `initial_prompt` tokens into `budget` tokens in total. When both don't fit,
/// each is guaranteed at least half of the budget, and either may use whatever the other leaves
/// unused. Hotwords keep their head (as in faster-whisper's `hotwords_tokens[:n]`) and the initial
/// prompt keeps its tail (as in faster-whisper's `previous_tokens[-n:]`), the end closest to the
/// audio being transcribed.
fn fit_to_budget(
    mut hotwords: Vec<String>,
    prompt: Vec<String>,
    budget: usize,
) -> (Vec<String>, Vec<String>) {
    if hotwords.len() + prompt.len() <= budget {
        return (hotwords, prompt);
    }
    let hotwords_keep = hotwords
        .len()
        .min((budget / 2).max(budget.saturating_sub(prompt.len())));
    let prompt_keep = prompt.len().min(budget - hotwords_keep);

    hotwords.truncate(hotwords_keep);
    let prompt = prompt[prompt.len() - prompt_keep..].to_vec();
    (hotwords, prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_blank_treats_empty_and_whitespace_as_absent() {
        assert_eq!(non_blank(None), None);
        assert_eq!(non_blank(Some("")), None);
        assert_eq!(non_blank(Some(" \t\n")), None);
        assert_eq!(non_blank(Some(" Kubernetes ")), Some(" Kubernetes "));
    }

    fn tokens(prefix: &str, n: usize) -> Vec<String> {
        (0..n).map(|i| format!("{prefix}{i}")).collect()
    }

    #[test]
    fn test_conditioning_token_budget_leaves_room_for_output_and_alignment() {
        // 448 / 2 = 224 positions for the prompt, minus the 3-token sot sequence,
        // `<|startofprev|>`, and the `<|notimestamps|>`/`<|eot|>` alignment adds.
        assert_eq!(conditioning_token_budget(448, 3), 218);
        assert_eq!(conditioning_token_budget(4, 3), 0);
    }

    #[test]
    fn test_fit_to_budget_leaves_short_inputs_untouched() {
        let (hw, prompt) = fit_to_budget(tokens("h", 5), tokens("p", 5), 10);
        assert_eq!(hw, tokens("h", 5));
        assert_eq!(prompt, tokens("p", 5));
    }

    #[test]
    fn test_fit_to_budget_splits_evenly_when_both_are_long() {
        let (hw, prompt) = fit_to_budget(tokens("h", 20), tokens("p", 20), 10);
        assert_eq!(hw, tokens("h", 5), "hotwords keep their head");
        assert_eq!(
            prompt,
            tokens("p", 20)[15..],
            "initial prompt keeps its tail"
        );
    }

    #[test]
    fn test_fit_to_budget_gives_unused_share_to_the_other_input() {
        let (hw, prompt) = fit_to_budget(tokens("h", 20), tokens("p", 2), 10);
        assert_eq!((hw.len(), prompt.len()), (8, 2));

        let (hw, prompt) = fit_to_budget(tokens("h", 2), tokens("p", 20), 10);
        assert_eq!((hw.len(), prompt.len()), (2, 8));
        assert_eq!(prompt, tokens("p", 20)[12..]);
    }

    #[test]
    fn test_fit_to_budget_with_single_input() {
        let (hw, prompt) = fit_to_budget(tokens("h", 20), Vec::new(), 10);
        assert_eq!((hw, prompt.len()), (tokens("h", 10), 0));

        let (hw, prompt) = fit_to_budget(Vec::new(), tokens("p", 20), 10);
        assert_eq!((hw.len(), prompt), (0, tokens("p", 20)[10..].to_vec()));
    }
}
