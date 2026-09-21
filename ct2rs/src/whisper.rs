// whisper.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! This module provides a speach transcriber.

use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{anyhow, Result};
use mel_spec::mel::{log_mel_spectrogram, mel, norm_mel};
use mel_spec::stft::Spectrogram;
use ndarray::{s, stack, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

use super::tokenizers::hf;
use super::{sys, Config, Tokenizer};

/// Represents a transcribed word with detailed timing and probability.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Word {
    /// The transcribed word text.
    pub word: String,
    /// Start time in seconds relative to the audio start.
    pub start: f32,
    /// End time in seconds relative to the audio start.
    pub end: f32,
    /// Confidence probability score bounded between 0.0 and 1.0.
    pub probability: f32,
}

/// Represents a transcribed audio segment with start/end timestamps and word-level information.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Segment {
    /// Segment ID.
    pub id: usize,
    /// Text content of the segment.
    pub text: String,
    /// Start time in seconds relative to the audio start.
    pub start: f32,
    /// End time in seconds relative to the audio start.
    pub end: f32,
    /// Word-level alignment information.
    pub words: Option<Vec<Word>>,
}

/// Options for whisper generation, plus optional content-conditioning knobs.
///
/// This mirrors [`sys::WhisperOptions`] field-for-field — those fields are used verbatim by the
/// underlying decode in [`Whisper::generate`] and [`Whisper::generate_segments`] — and adds
/// `initial_prompt`/`hotwords`, which only [`Whisper::generate_segments`] consumes to bias
/// transcription. Both default to `None`, so existing callers relying on `..Default::default()`
/// see no behavior change at all.
///
/// # Examples
///
/// Example of creating a default `WhisperOptions`:
///
/// ```
/// use ct2rs::WhisperOptions;
///
/// let options = WhisperOptions::default();
/// ```
#[derive(Clone, Debug)]
pub struct WhisperOptions {
    /// Beam size to use for beam search (set 1 to run greedy search). (default: 5)
    pub beam_size: usize,
    /// Beam search patience factor, as described in <https://arxiv.org/abs/2204.05424>.
    /// The decoding will continue until beam_size*patience hypotheses are finished.
    /// (default: 1.0)
    pub patience: f32,
    /// Exponential penalty applied to the length during beam search. (default: 1.0)
    pub length_penalty: f32,
    /// Penalty applied to the score of previously generated tokens, as described in
    /// <https://arxiv.org/abs/1909.05858> (set > 1 to penalize). (default: 1.0)
    pub repetition_penalty: f32,
    /// Prevent repetitions of ngrams with this size (set 0 to disable). (default: 0)
    pub no_repeat_ngram_size: usize,
    /// Maximum generation length. (default: 448)
    pub max_length: usize,
    /// Randomly sample from the top K candidates (set 0 to sample from the full distribution).
    /// (default: 1)
    pub sampling_topk: usize,
    /// High temperatures increase randomness. (default: 1.0)
    pub sampling_temperature: f32,
    /// Number of hypotheses to include in the result. (default: 1)
    pub num_hypotheses: usize,
    /// Include scores in the result. (default: false)
    pub return_scores: bool,
    /// Include log probs of each token in the result. (default: false)
    pub return_logits_vocab: bool,
    /// Include the probability of the no speech token in the result. (default: false)
    pub return_no_speech_prob: bool,
    /// Maximum index of the first predicted timestamp. (default: 50)
    pub max_initial_timestamp_index: usize,
    /// Suppress blank outputs at the beginning of the sampling. (default: true)
    pub suppress_blank: bool,
    /// List of token IDs to suppress.
    /// -1 will suppress a default set of symbols as defined in the model config.json file.
    /// (default: `[-1]`)
    pub suppress_tokens: Vec<i32>,
    /// Optional free-form context text (e.g. topic, prior conversation). Consumed only by
    /// [`Whisper::generate_segments`], which tokenizes it once and prepends it (via
    /// `<|startofprev|>`) to the decoder prompt used for **every** ~30s chunk. Unlike
    /// faster-whisper's `condition_on_previous_text`, this is a static prefix reused identically
    /// across chunks — it is not updated with each chunk's own transcribed output. `None` (the
    /// default) leaves generation completely unchanged.
    ///
    /// Conditioning is a soft bias on decoding: `generate_segments` decodes each chunk both with
    /// and without it and reconciles the two transcripts word by word, keeping a conditioned
    /// word only where it's an improvement and never accepting content conditioning dropped
    /// outright (see [`Whisper::generate_segments`] and `merge_word_hypotheses`).
    pub initial_prompt: Option<String>,
    /// Optional hint words/phrases biasing decoding toward specific vocabulary or spellings.
    /// Consumed only by [`Whisper::generate_segments`], which tokenizes it once and prepends it
    /// (via `<|startofprev|>`, before `initial_prompt`'s tokens) to the decoder prompt used for
    /// every chunk. `None` (the default) leaves generation completely unchanged.
    ///
    /// Same per-chunk word-level reconciliation as `initial_prompt` applies here too.
    pub hotwords: Option<String>,
}

impl Default for WhisperOptions {
    fn default() -> Self {
        let sys::WhisperOptions {
            beam_size,
            patience,
            length_penalty,
            repetition_penalty,
            no_repeat_ngram_size,
            max_length,
            sampling_topk,
            sampling_temperature,
            num_hypotheses,
            return_scores,
            return_logits_vocab,
            return_no_speech_prob,
            max_initial_timestamp_index,
            suppress_blank,
            suppress_tokens,
        } = sys::WhisperOptions::default();
        Self {
            beam_size,
            patience,
            length_penalty,
            repetition_penalty,
            no_repeat_ngram_size,
            max_length,
            sampling_topk,
            sampling_temperature,
            num_hypotheses,
            return_scores,
            return_logits_vocab,
            return_no_speech_prob,
            max_initial_timestamp_index,
            suppress_blank,
            suppress_tokens,
            initial_prompt: None,
            hotwords: None,
        }
    }
}

/// Converts to the FFI options consumed by the underlying decode. `initial_prompt`/`hotwords`
/// never cross this boundary — they are applied purely on the Rust side, as extra prompt tokens,
/// before generation is invoked.
impl From<&WhisperOptions> for sys::WhisperOptions {
    fn from(o: &WhisperOptions) -> Self {
        Self {
            beam_size: o.beam_size,
            patience: o.patience,
            length_penalty: o.length_penalty,
            repetition_penalty: o.repetition_penalty,
            no_repeat_ngram_size: o.no_repeat_ngram_size,
            max_length: o.max_length,
            sampling_topk: o.sampling_topk,
            sampling_temperature: o.sampling_temperature,
            num_hypotheses: o.num_hypotheses,
            return_scores: o.return_scores,
            return_logits_vocab: o.return_logits_vocab,
            return_no_speech_prob: o.return_no_speech_prob,
            max_initial_timestamp_index: o.max_initial_timestamp_index,
            suppress_blank: o.suppress_blank,
            suppress_tokens: o.suppress_tokens.clone(),
        }
    }
}

const PREPROCESSOR_CONFIG_FILE: &str = "preprocessor_config.json";

/// A speach transcriber using the Whisper speech recognition model published by OpenAI.
///
/// # Example
/// ```no_run
/// use ct2rs::Whisper;
///
/// # fn main() -> anyhow::Result<()>{
/// let whisper = Whisper::new("/path/to/model", Default::default())?;
///
/// let sampling_rate = whisper.sampling_rate();
/// // Sample the source audio at the sampling rates shown above.
/// // Each sample must be normalized to the range [-1, 1].
/// let samples = vec![];
///
/// let res = whisper.generate(&samples, None, false, &Default::default())?;
/// for r in res {
///     println!("{}", r);
/// }
/// # Ok(())
/// # }
/// ```
pub struct Whisper {
    whisper: sys::Whisper,
    tokenizer: hf::Tokenizer,
    config: PreprocessorConfig,
}

impl Whisper {
    /// Initializes the transcriber.
    ///
    /// # Arguments
    /// * `path` - A path to the directory containing the language model to be loaded.
    /// * `config` - A [`Config`] structure that specifies various settings
    ///   and configurations for the `Whisper`.
    ///
    /// # Returns
    /// Returns a `Result` that, if successful, contains the initialized `Whisper`. If an error
    /// occurs during initialization, the function will return an error wrapped in the `Result`.
    pub fn new<T: AsRef<Path>>(model_path: T, config: Config) -> Result<Self> {
        Ok(Self {
            whisper: sys::Whisper::new(&model_path, config)?,
            tokenizer: hf::Tokenizer::new(&model_path)?,
            config: PreprocessorConfig::read(model_path.as_ref().join(PREPROCESSOR_CONFIG_FILE))?,
        })
    }

    /// Transcribe the given samples.
    ///
    /// # Arguments
    /// * `samples` - Samples of the source audio. They must be sampled at the sampling rate
    ///   returned by [`sampling_rate`][Whisper::sampling_rate] method and normalized to the range
    ///   `[-1, 1]`. If the samples are longer than the maximum number of samples returned by
    ///   [`n_samples`][Whisper::n_samples] method, they will be processed in segments.
    /// * `language` - An optional language setting. It transcribes assuming the specified language.
    ///   If `None`, it uses Whisper's language detection.
    /// * `timestamp` - If `true`, the output will include timestamps.
    /// * `options` - Settings.
    ///
    /// # Returns
    /// Returns a `Result` containing a vector of transcribed strings if successful,
    /// or an error if the translation fails.
    pub fn generate(
        &self,
        samples: &[f32],
        language: Option<&str>,
        timestamp: bool,
        options: &WhisperOptions,
    ) -> Result<Vec<String>> {
        let (mut mel_spectrogram, num_chunks) = self.generate_mel_spectrogram(samples)?;
        if num_chunks == 0 {
            return Ok(Vec::new());
        }

        let shape = mel_spectrogram.shape().to_vec();
        let storage_view = sys::StorageView::new(
            &shape,
            mel_spectrogram.as_slice_mut().unwrap(),
            Default::default(),
        )?;

        let lang_token = self.detect_language_token(&storage_view, language)?;

        let prompt = self.generate_prompt(&lang_token, timestamp);

        self.whisper
            .generate(
                &storage_view,
                &vec![prompt; num_chunks],
                &sys::WhisperOptions::from(options),
            )?
            .into_iter()
            .map(|res| {
                let r = res
                    .sequences
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("failed to transcribe samples"))?;
                self.tokenizer.decode(r)
            })
            .collect()
    }

    /// Generate transcription segments for the given samples with detailed word-level and segment-level timelines.
    ///
    /// # Arguments
    /// * `samples` - Samples of the source audio. They must be sampled at the sampling rate
    ///   returned by [`sampling_rate`][Whisper::sampling_rate] method and normalized to the range
    ///   `[-1, 1]`. If the samples are longer than the maximum number of samples returned by
    ///   [`n_samples`][Whisper::n_samples] method, they will be processed in segments.
    /// * `language` - An optional language setting. It generates segments assuming the specified language.
    ///   If `None`, it uses Whisper's language detection.
    /// * `options` - Settings, including the optional `initial_prompt`/`hotwords` conditioning
    ///   knobs (see [`WhisperOptions`]). When either is set, each chunk is decoded twice (with
    ///   and without conditioning) and the two transcripts are reconciled word by word: a
    ///   conditioned word is kept only where it's more confident than the baseline, and content
    ///   conditioning drops entirely is never accepted, at roughly 2x the decode and alignment
    ///   cost for those chunks.
    ///
    /// # Returns
    /// Returns a `Result` containing a vector of transcribed `Segment`s if successful,
    /// or an error if the segment generation fails.
    pub fn generate_segments(
        &self,
        samples: &[f32],
        language: Option<&str>,
        options: &WhisperOptions,
    ) -> Result<Vec<Segment>> {
        let (mut mel_spectrogram, num_chunks) = self.generate_mel_spectrogram(samples)?;
        if num_chunks == 0 {
            return Ok(Vec::new());
        }

        let shape = mel_spectrogram.shape().to_vec();
        let storage_view = sys::StorageView::new(
            &shape,
            mel_spectrogram.as_slice_mut().unwrap(),
            Default::default(),
        )?;

        let lang_token = self.detect_language_token(&storage_view, language)?;

        // Pass features through the encoder network to get encoder outputs
        let encoder_output = self.whisper.encode(&storage_view, false)?;

        let conditioning_prefix = self.build_conditioning_prefix(
            options.initial_prompt.as_deref(),
            options.hotwords.as_deref(),
            options.max_length,
        )?;

        let sot_sequence = self.generate_prompt(&lang_token, true);
        let baseline_prompt: Vec<String> = sot_sequence.iter().map(|t| t.to_string()).collect();

        if conditioning_prefix.is_empty() {
            return self.generate_segments_plain(
                &encoder_output,
                &baseline_prompt,
                num_chunks,
                options,
            );
        }

        let conditioned_prompt: Vec<String> = conditioning_prefix
            .into_iter()
            .chain(baseline_prompt.iter().cloned())
            .collect();

        self.generate_segments_with_conditioning(
            &encoder_output,
            &baseline_prompt,
            &conditioned_prompt,
            num_chunks,
            options,
        )
    }

    /// Decodes and aligns a single prompt for every chunk, with no conditioning involved. This is
    /// the path used whenever `initial_prompt`/`hotwords` are both `None`.
    fn generate_segments_plain(
        &self,
        encoder_output: &sys::StorageView,
        prompt: &[String],
        num_chunks: usize,
        options: &WhisperOptions,
    ) -> Result<Vec<Segment>> {
        let gen_results = self.whisper.generate(
            encoder_output,
            &vec![prompt.to_vec(); num_chunks],
            &sys::WhisperOptions::from(options),
        )?;
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
    /// Reconciling word by word (see [`merge_word_hypotheses`]) avoids that: matching words are
    /// kept as-is, a word conditioning changed is kept only if its local confidence improved, and
    /// content conditioning dropped entirely is never accepted silently: the baseline's words
    /// are used for that span instead.
    ///
    /// Cost: roughly 2x decode and 2x alignment work for chunks that use conditioning (nothing
    /// extra when `initial_prompt`/`hotwords` aren't set, since this path isn't taken then).
    fn generate_segments_with_conditioning(
        &self,
        encoder_output: &sys::StorageView,
        baseline_prompt: &[String],
        conditioned_prompt: &[String],
        num_chunks: usize,
        options: &WhisperOptions,
    ) -> Result<Vec<Segment>> {
        let ffi_options = sys::WhisperOptions::from(options);

        // These must be two separate `generate()` calls, not one batch of both prompt kinds:
        // CTranslate2 requires every prompt within a single batch to have
        // `<|startoftranscript|>` at the same index, which the conditioned prompt (longer,
        // prefixed with the conditioning tokens) and the plain prompt (unprefixed) don't share.
        let conditioned_results = self.whisper.generate(
            encoder_output,
            &vec![conditioned_prompt.to_vec(); num_chunks],
            &ffi_options,
        )?;
        let baseline_results = self.whisper.generate(
            encoder_output,
            &vec![baseline_prompt.to_vec(); num_chunks],
            &ffi_options,
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
        for (range, mut word) in word_token_ranges.into_iter().zip(chunk_words.into_iter()) {
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

    /// Returns the expected sampling rate.
    pub fn sampling_rate(&self) -> usize {
        self.config.sampling_rate
    }

    /// Max number of samples per batch.
    pub fn n_samples(&self) -> usize {
        self.config.n_samples
    }

    /// Returns `true` if this model is multilingual.
    #[inline]
    pub fn is_multilingual(&self) -> bool {
        self.whisper.is_multilingual()
    }

    /// Returns the number of languages supported.
    #[inline]
    pub fn num_languages(&self) -> usize {
        self.whisper.num_languages()
    }

    /// Number of batches in the work queue.
    #[inline]
    pub fn num_queued_batches(&self) -> usize {
        self.whisper.num_queued_batches()
    }

    /// Number of batches in the work queue or currently processed by a worker.
    #[inline]
    pub fn num_active_batches(&self) -> usize {
        self.whisper.num_active_batches()
    }

    /// Number of parallel replicas.
    #[inline]
    pub fn num_replicas(&self) -> usize {
        self.whisper.num_replicas()
    }

    /// Generates a log-mel spectrogram for the given audio samples.
    ///
    /// It partitions the samples into chunks and extracts log-mel features
    /// for each chunk.
    ///
    /// # Returns
    /// A tuple containing:
    /// - An `Array3<f32>` with the stacked log-mel spectrogram.
    /// - The number of chunks processed.
    fn generate_mel_spectrogram(&self, samples: &[f32]) -> Result<(Array3<f32>, usize)> {
        let mut stft = Spectrogram::new(self.config.n_fft, self.config.hop_length);

        let mut mel_spectrogram_vec = vec![];
        for chunk in samples.chunks(self.config.n_samples) {
            let mut mel_spectrogram_per_chunk =
                Array2::zeros((self.config.feature_size, self.config.nb_max_frames));
            for (i, flame) in chunk.chunks(self.config.hop_length).enumerate() {
                if let Some(fft_frame) = stft.add(flame) {
                    let mel = norm_mel(&log_mel_spectrogram(&fft_frame, &self.config.mel_filters))
                        .mapv(|v| v as f32);
                    mel_spectrogram_per_chunk
                        .slice_mut(s![.., i])
                        .assign(&mel.slice(s![.., 0]));
                }
            }
            mel_spectrogram_vec.push(mel_spectrogram_per_chunk);
        }

        let num_chunks = mel_spectrogram_vec.len();
        if num_chunks == 0 {
            return Ok((Array3::zeros((0, 0, 0)), 0));
        }

        let mut mel_spectrogram = stack(
            Axis(0),
            &mel_spectrogram_vec
                .iter()
                .map(|a| a.view())
                .collect::<Vec<_>>(),
        )?;
        if !mel_spectrogram.is_standard_layout() {
            mel_spectrogram = mel_spectrogram.as_standard_layout().into_owned();
        }

        Ok((mel_spectrogram, num_chunks))
    }

    /// Detects or formats the language token.
    ///
    /// If `language` is specified, it returns the formatted token (e.g. `<|en|>`).
    /// Otherwise, it runs the language detector on the given storage view.
    fn detect_language_token(
        &self,
        storage_view: &sys::StorageView,
        language: Option<&str>,
    ) -> Result<String> {
        let lang_token = match language {
            Some(lang) => {
                format!("<|{}|>", lang)
            }
            None => {
                let detection_result = self.whisper.detect_language(storage_view)?;
                detection_result
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("failed to detect language"))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("failed to detect language"))?
                    .language
            }
        };
        Ok(lang_token)
    }

    /// Generates the transcript prompt tokens.
    ///
    /// # Arguments
    /// * `lang_token` - The language token (e.g. `<|en|>`).
    /// * `timestamp` - If `true`, timestamps will be generated. Otherwise, adds `"<|notimestamps|>"`.
    fn generate_prompt<'a>(&self, lang_token: &'a str, timestamp: bool) -> Vec<&'a str> {
        let mut prompt = vec!["<|startoftranscript|>", lang_token, "<|transcribe|>"];
        if !timestamp {
            prompt.push("<|notimestamps|>");
        }
        prompt
    }

    /// Builds the `<|startofprev|>`-prefixed conditioning tokens for `hotwords`/`initial_prompt`,
    /// mirroring faster-whisper's `get_prompt` token order and length-capping semantics, but
    /// applied identically to every chunk (no rolling previous-text conditioning). Returns an
    /// empty vector if neither is provided, in which case callers see no behavior change at all.
    fn build_conditioning_prefix(
        &self,
        initial_prompt: Option<&str>,
        hotwords: Option<&str>,
        max_length: usize,
    ) -> Result<Vec<String>> {
        if initial_prompt.is_none() && hotwords.is_none() {
            return Ok(Vec::new());
        }

        let mut prefix = vec!["<|startofprev|>".to_string()];

        if let Some(hw) = hotwords {
            prefix.extend(cap_head(self.encode_raw(hw)?, max_length));
        }
        if let Some(prompt) = initial_prompt {
            prefix.extend(cap_tail(self.encode_raw(prompt)?, max_length));
        }

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

/// Hotwords are head-truncated when too long: keep only the first `max_length/2 - 1` tokens,
/// matching faster-whisper's `hotwords_tokens[: max_length // 2 - 1]`.
fn cap_head(mut tokens: Vec<String>, max_length: usize) -> Vec<String> {
    let threshold = max_length / 2;
    if tokens.len() >= threshold {
        tokens.truncate(threshold.saturating_sub(1));
    }
    tokens
}

/// `initial_prompt` tokens are tail-kept when too long: keep only the last `max_length/2 - 1`
/// tokens, matching faster-whisper's `previous_tokens[-(max_length // 2 - 1):]`.
fn cap_tail(tokens: Vec<String>, max_length: usize) -> Vec<String> {
    let keep = (max_length / 2).saturating_sub(1);
    if tokens.len() > keep {
        tokens[tokens.len() - keep..].to_vec()
    } else {
        tokens
    }
}

fn starts_new_word(token: &str) -> bool {
    // If the token starts with 'Ġ' (GPT-2/Whisper space representation)
    if token.starts_with('Ġ') {
        return true;
    }
    // If the token starts with ' ' (SentencePiece space representation)
    if token.starts_with(' ') {
        return true;
    }
    // If the token starts with a regular space
    if token.starts_with(' ') {
        return true;
    }
    // If the token is a punctuation/special character (excluding letters and digits)
    if let Some(first_char) = token.chars().next() {
        if first_char.is_ascii_punctuation() {
            return true;
        }
    }
    false
}

fn is_special_token(token: &str) -> bool {
    token.starts_with("<|") && token.ends_with("|>")
}

fn group_tokens_into_words(tokens: &[String]) -> Vec<std::ops::Range<usize>> {
    let mut word_ranges = Vec::new();
    let mut current_word_start = None;

    for (i, token) in tokens.iter().enumerate() {
        if is_special_token(token) {
            if let Some(start) = current_word_start {
                word_ranges.push(start..i);
                current_word_start = None;
            }
            continue;
        }

        if current_word_start.is_none() {
            current_word_start = Some(i);
        } else if starts_new_word(token) {
            if let Some(start) = current_word_start {
                word_ranges.push(start..i);
            }
            current_word_start = Some(i);
        }
    }

    if let Some(start) = current_word_start {
        word_ranges.push(start..tokens.len());
    }

    word_ranges
}

fn process_word_timings(
    word_token_ranges: &[std::ops::Range<usize>],
    alignments: &[sys::WhisperTokenAlignment],
    text_token_probs: &[f32],
    num_tokens: usize,
) -> Vec<Word> {
    if num_tokens == 0 || word_token_ranges.is_empty() {
        return Vec::new();
    }

    let mut token_start_frames = vec![-1i64; num_tokens];
    let mut token_end_frames = vec![-1i64; num_tokens];

    // First pass: extract from alignments
    for m in 0..num_tokens {
        let aligned_frames: Vec<i64> = alignments
            .iter()
            .filter(|a| a.token_x == m as i64)
            .map(|a| a.frame_x)
            .collect();
        if !aligned_frames.is_empty() {
            token_start_frames[m] = *aligned_frames.iter().min().unwrap();
            token_end_frames[m] = *aligned_frames.iter().max().unwrap() + 1;
        }
    }

    // Second pass: fill in missing/empty and enforce monotonicity
    let mut last_end = 0;
    for m in 0..num_tokens {
        if token_start_frames[m] == -1 {
            token_start_frames[m] = last_end;
            token_end_frames[m] = last_end;
        } else {
            if token_start_frames[m] < last_end {
                token_start_frames[m] = last_end;
            }
            if token_end_frames[m] < token_start_frames[m] {
                token_end_frames[m] = token_start_frames[m];
            }
        }
        last_end = token_end_frames[m];
    }

    // Convert token frames to words
    let mut words = Vec::new();
    for range in word_token_ranges {
        let u = range.start;
        let v = range.end - 1;

        let word_start_frame = token_start_frames[u];
        let word_end_frame = token_end_frames[v];

        // 50.0 is the downsampled temporal resolution constant of Whisper encoder output (1 frame = 20ms)
        let word_start_sec = word_start_frame as f32 / 50.0;
        let word_end_sec = word_end_frame as f32 / 50.0;

        // Confidence probability score as the arithmetic mean of token probabilities
        let sum_prob: f32 = text_token_probs[u..=v].iter().sum();
        let word_prob = sum_prob / (v - u + 1) as f32;

        words.push(Word {
            word: "".to_string(), // Text will be filled later by the caller
            start: word_start_sec,
            end: word_end_sec,
            probability: word_prob,
        });
    }

    // Apply Heuristic 2: Median-Based Duration Capping
    let mut durations: Vec<f32> = words
        .iter()
        .map(|w| w.end - w.start)
        .filter(|&d| d > 0.0)
        .collect();
    if !durations.is_empty() {
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_idx = durations.len() / 2;
        let mut d_median = durations[median_idx];
        if d_median > 0.7 {
            d_median = 0.7;
        }
        let d_max = 2.0 * d_median;
        for w in &mut words {
            let duration = w.end - w.start;
            if duration > d_max {
                w.end = w.start + d_max;
            }
        }
    }

    words
}

/// One elementary edit-script operation aligning a `base` word sequence to a `cond` one, with
/// the index (or indices) of the word(s) it consumes from each side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WordEditOp {
    /// Same word (case-insensitively) on both sides.
    Match(usize, usize),
    /// Different word at this position on both sides.
    Substitute(usize, usize),
    /// A word present in `base` with nothing corresponding in `cond`.
    Delete(usize),
    /// A word present in `cond` with nothing corresponding in `base`.
    Insert(usize),
}

/// Computes a minimal word-level edit script aligning `base` to `cond` (Levenshtein alignment,
/// comparing word text case-insensitively). `O(base.len() * cond.len())`, which is trivial at
/// the scale of one ~30s chunk's word count.
fn word_edit_script(base: &[Word], cond: &[Word]) -> Vec<WordEditOp> {
    let n = base.len();
    let m = cond.len();

    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i as u32;
    }
    for j in 0..=m {
        dp[0][j] = j as u32;
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub_cost = if base[i - 1].word.eq_ignore_ascii_case(&cond[j - 1].word) {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j - 1] + sub_cost)
                .min(dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1);
        }
    }

    let mut ops = Vec::with_capacity(n.max(m));
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let sub_cost = if base[i - 1].word.eq_ignore_ascii_case(&cond[j - 1].word) {
                0
            } else {
                1
            };
            if dp[i][j] == dp[i - 1][j - 1] + sub_cost {
                ops.push(if sub_cost == 0 {
                    WordEditOp::Match(i - 1, j - 1)
                } else {
                    WordEditOp::Substitute(i - 1, j - 1)
                });
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            ops.push(WordEditOp::Delete(i - 1));
            i -= 1;
            continue;
        }
        ops.push(WordEditOp::Insert(j - 1));
        j -= 1;
    }
    ops.reverse();
    ops
}

/// Reconciles two word-level transcripts of the same audio chunk, one decoded without
/// `initial_prompt`/`hotwords` conditioning (`base`) and one with it (`cond`), word by word
/// rather than picking one whole transcript as the winner.
///
/// Policy per aligned block between two matching anchor words (see [`word_edit_script`]):
/// - Both sides agree: kept as-is.
/// - `cond` added words `base` didn't have (pure insertion): kept, since conditioning
///   contributing extra content it's confident about is not a risk worth guarding against.
/// - `base` has words `cond` doesn't (pure deletion): **`base`'s words are always kept**.
///   Conditioning silently dropping content is the one failure mode this function exists to
///   prevent, and there is no "confidence of an absence" to weigh it against, so it is never
///   accepted.
/// - Both sides have different words at the same position (replace): whichever side's local
///   average word probability is higher is kept, so a conditioned word only overrides the
///   baseline when the model is actually more confident in it, not merely because the prompt
///   nudged it there.
fn merge_word_hypotheses(base: &[Word], cond: &[Word]) -> Vec<Word> {
    let ops = word_edit_script(base, cond);
    let mut merged = Vec::with_capacity(ops.len());

    let mut idx = 0;
    while idx < ops.len() {
        if let WordEditOp::Match(_, cond_idx) = ops[idx] {
            merged.push(cond[cond_idx].clone());
            idx += 1;
            continue;
        }

        let start = idx;
        while idx < ops.len() && !matches!(ops[idx], WordEditOp::Match(..)) {
            idx += 1;
        }

        let mut base_range: Option<(usize, usize)> = None;
        let mut cond_range: Option<(usize, usize)> = None;
        for op in &ops[start..idx] {
            match *op {
                WordEditOp::Substitute(bi, ci) => {
                    base_range = Some(extend_range(base_range, bi));
                    cond_range = Some(extend_range(cond_range, ci));
                }
                WordEditOp::Delete(bi) => base_range = Some(extend_range(base_range, bi)),
                WordEditOp::Insert(ci) => cond_range = Some(extend_range(cond_range, ci)),
                WordEditOp::Match(..) => unreachable!("matches are handled above"),
            }
        }

        match (base_range, cond_range) {
            (Some((lo, hi)), None) => merged.extend(base[lo..=hi].iter().cloned()),
            (None, Some((lo, hi))) => merged.extend(cond[lo..=hi].iter().cloned()),
            (Some((base_lo, base_hi)), Some((cond_lo, cond_hi))) => {
                let base_slice = &base[base_lo..=base_hi];
                let cond_slice = &cond[cond_lo..=cond_hi];
                if average_probability(cond_slice) >= average_probability(base_slice) {
                    merged.extend(cond_slice.iter().cloned());
                } else {
                    merged.extend(base_slice.iter().cloned());
                }
            }
            (None, None) => unreachable!("a non-match run must consume at least one side"),
        }
    }

    merged
}

fn extend_range(range: Option<(usize, usize)>, idx: usize) -> (usize, usize) {
    match range {
        Some((lo, hi)) => (lo.min(idx), hi.max(idx)),
        None => (idx, idx),
    }
}

fn average_probability(words: &[Word]) -> f32 {
    if words.is_empty() {
        return 0.0;
    }
    words.iter().map(|w| w.probability).sum::<f32>() / words.len() as f32
}

/// Joins word-level output back into display text, matching common detokenization practice: no
/// space is inserted before a "word" that is pure punctuation (e.g. `,`, `.`, `!`).
fn join_words(words: &[Word]) -> String {
    let mut text = String::new();
    for w in words {
        if !text.is_empty() && starts_with_alphanumeric(&w.word) {
            text.push(' ');
        }
        text.push_str(&w.word);
    }
    text
}

/// Whether a word should get a leading space when joined after another. Words starting with
/// punctuation (`,`, `!`, or a split-off contraction piece like `'t` in "don't") attach directly
/// to what precedes them instead.
fn starts_with_alphanumeric(word: &str) -> bool {
    word.chars().next().is_some_and(|c| c.is_alphanumeric())
}

#[cfg(test)]
mod tests_grouping {
    use super::*;

    #[test]
    fn test_starts_new_word() {
        assert!(starts_new_word("ĠHello"));
        assert!(starts_new_word(" world"));
        assert!(starts_new_word(" "));
        assert!(starts_new_word("!"));
        assert!(!starts_new_word("llo"));
    }

    #[test]
    fn test_group_tokens_into_words() {
        let tokens = vec![
            "<|startoftranscript|>".to_string(),
            "<|en|>".to_string(),
            "<|transcribe|>".to_string(),
            "ĠHello".to_string(),
            "llo".to_string(),
            "Ġworld".to_string(),
            "!".to_string(),
        ];
        let ranges = group_tokens_into_words(&tokens);
        assert_eq!(ranges, vec![3..5, 5..6, 6..7]);
    }

    #[test]
    fn test_cap_head_truncates_hotwords_tokens() {
        let tokens: Vec<String> = (0..20).map(|i| format!("t{i}")).collect();
        let capped = cap_head(tokens, 20); // threshold = 10, keep = 9
        assert_eq!(capped.len(), 9);
        assert_eq!(capped[0], "t0");
    }

    #[test]
    fn test_cap_head_leaves_short_hotwords_untouched() {
        let tokens: Vec<String> = (0..5).map(|i| format!("t{i}")).collect();
        assert_eq!(cap_head(tokens.clone(), 20), tokens);
    }

    #[test]
    fn test_cap_tail_keeps_last_tokens_of_initial_prompt() {
        let tokens: Vec<String> = (0..20).map(|i| format!("t{i}")).collect();
        let capped = cap_tail(tokens, 20); // keep = 9
        assert_eq!(capped.len(), 9);
        assert_eq!(capped[0], "t11");
    }

    #[test]
    fn test_cap_tail_leaves_short_prompt_untouched() {
        let tokens: Vec<String> = (0..5).map(|i| format!("t{i}")).collect();
        assert_eq!(cap_tail(tokens.clone(), 20), tokens);
    }

    #[test]
    fn test_default_whisper_options_has_no_conditioning() {
        let options = super::WhisperOptions::default();
        assert!(options.initial_prompt.is_none());
        assert!(options.hotwords.is_none());
        assert_eq!(options.beam_size, 5);
        assert_eq!(options.max_length, 448);
    }

    fn word(text: &str, probability: f32) -> Word {
        Word {
            word: text.to_string(),
            start: 0.0,
            end: 0.0,
            probability,
        }
    }

    fn words(pairs: &[(&str, f32)]) -> Vec<Word> {
        pairs.iter().map(|&(w, p)| word(w, p)).collect()
    }

    #[test]
    fn test_merge_word_hypotheses_keeps_matching_words() {
        let base = words(&[("hello", 0.5), ("world", 0.5)]);
        let cond = words(&[("hello", 0.9), ("world", 0.9)]);
        let merged = merge_word_hypotheses(&base, &cond);
        let text: Vec<&str> = merged.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(text, vec!["hello", "world"]);
    }

    #[test]
    fn test_merge_word_hypotheses_never_accepts_a_dropped_clause() {
        // `cond` drops "wonderful strange" entirely relative to `base`. This must never be
        // accepted silently, regardless of confidence, since there is nothing on the `cond`
        // side to compare it against.
        let base = words(&[
            ("this", 0.9),
            ("is", 0.9),
            ("a", 0.9),
            ("wonderful", 0.9),
            ("strange", 0.9),
            ("day", 0.9),
        ]);
        let cond = words(&[("this", 0.9), ("is", 0.9), ("a", 0.9), ("day", 0.9)]);
        let merged = merge_word_hypotheses(&base, &cond);
        let text: Vec<&str> = merged.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(text, vec!["this", "is", "a", "wonderful", "strange", "day"]);
    }

    #[test]
    fn test_merge_word_hypotheses_keeps_higher_confidence_replacement() {
        // `cond` replaces "world" with a lower-confidence "word": keep `base`'s word.
        let base = words(&[("hello", 0.9), ("world", 0.85)]);
        let cond = words(&[("hello", 0.9), ("word", 0.2)]);
        let merged = merge_word_hypotheses(&base, &cond);
        let text: Vec<&str> = merged.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(text, vec!["hello", "world"]);
    }

    #[test]
    fn test_merge_word_hypotheses_prefers_higher_confidence_conditioned_word() {
        // `cond` replaces a garbled word with a correctly-spelled, higher-confidence one: keep
        // `cond`'s word, i.e. the hotwords use case.
        let base = words(&[("hello", 0.9), ("wrld", 0.3)]);
        let cond = words(&[("hello", 0.9), ("world", 0.95)]);
        let merged = merge_word_hypotheses(&base, &cond);
        let text: Vec<&str> = merged.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(text, vec!["hello", "world"]);
    }

    #[test]
    fn test_merge_word_hypotheses_keeps_pure_insertion() {
        let base = words(&[("hello", 0.9), ("world", 0.9)]);
        let cond = words(&[("hello", 0.9), ("there", 0.9), ("world", 0.9)]);
        let merged = merge_word_hypotheses(&base, &cond);
        let text: Vec<&str> = merged.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(text, vec!["hello", "there", "world"]);
    }

    #[test]
    fn test_merge_word_hypotheses_empty_base_keeps_all_conditioned_words() {
        let base: Vec<Word> = vec![];
        let cond = words(&[("hello", 0.9), ("world", 0.9)]);
        let merged = merge_word_hypotheses(&base, &cond);
        let text: Vec<&str> = merged.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(text, vec!["hello", "world"]);
    }

    #[test]
    fn test_merge_word_hypotheses_empty_conditioned_keeps_all_base_words() {
        // Degenerate case of "conditioning dropped everything": must still recover `base`.
        let base = words(&[("hello", 0.9), ("world", 0.9)]);
        let cond: Vec<Word> = vec![];
        let merged = merge_word_hypotheses(&base, &cond);
        let text: Vec<&str> = merged.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(text, vec!["hello", "world"]);
    }

    #[test]
    fn test_join_words_skips_space_before_punctuation() {
        let ws = words(&[("Hello", 0.9), (",", 0.9), ("world", 0.9), ("!", 0.9)]);
        assert_eq!(join_words(&ws), "Hello, world!");
    }

    #[test]
    fn test_join_words_skips_space_before_split_contraction_piece() {
        // "didn't" tokenized as two words, "didn" + "'t", must not get a space in between.
        let ws = words(&[("she", 0.9), ("didn", 0.9), ("'t", 0.9), ("know", 0.9)]);
        assert_eq!(join_words(&ws), "she didn't know");
    }

    #[test]
    fn test_join_words_skips_space_before_hyphen_continuation() {
        let ws = words(&[("M", 0.9), ("-R", 0.9), ("-C", 0.9), ("-S", 0.9)]);
        assert_eq!(join_words(&ws), "M-R-C-S");
    }

    #[test]
    fn test_process_word_timings() {
        use crate::sys::WhisperTokenAlignment;

        let word_token_ranges = vec![0..2, 2..3]; // Two words: token 0..2, token 2..3
                                                  // Token 0 aligned to frames 10..15, Token 1 has no alignments, Token 2 aligned to frame 20..22
        let alignments = vec![
            WhisperTokenAlignment {
                token_x: 0,
                frame_x: 10,
            },
            WhisperTokenAlignment {
                token_x: 0,
                frame_x: 14,
            },
            WhisperTokenAlignment {
                token_x: 2,
                frame_x: 20,
            },
            WhisperTokenAlignment {
                token_x: 2,
                frame_x: 21,
            },
        ];
        let text_token_probs = vec![0.9, 0.8, 0.95];

        let words = process_word_timings(&word_token_ranges, &alignments, &text_token_probs, 3);
        assert_eq!(words.len(), 2);

        // Word 0 (tokens 0..1):
        // Token 0: starts at 10, ends at 15
        // Token 1: starts at last_end (15), ends at last_end (15)
        // Word 0: starts at 10 (0.2s), ends at 15 (0.3s)
        assert_eq!(words[0].start, 0.2);
        assert_eq!(words[0].end, 0.3);
        // Average probability: (0.9 + 0.8) / 2 = 0.85
        assert_eq!(words[0].probability, 0.85);

        // Word 1 (token 2):
        // Token 2: starts at 20, ends at 22
        // Word 1: starts at 20 (0.4s), ends at 22 (0.44s)
        assert_eq!(words[1].start, 0.4);
        assert_eq!(words[1].end, 0.44);
        assert_eq!(words[1].probability, 0.95);
    }
}

impl Debug for Whisper {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.whisper)
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct PreprocessorConfig {
    chunk_length: usize,
    feature_extractor_type: String,
    feature_size: usize,
    hop_length: usize,
    n_fft: usize,
    n_samples: usize,
    nb_max_frames: usize,
    padding_side: String,
    padding_value: f32,
    processor_class: String,
    return_attention_mask: bool,
    sampling_rate: usize,
    mel_filters: Array2<f64>,
}

impl PreprocessorConfig {
    fn read<T: AsRef<Path>>(path: T) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        #[derive(Deserialize)]
        struct PreprocessorConfigAux {
            chunk_length: usize,
            feature_extractor_type: String,
            feature_size: usize,
            hop_length: usize,
            n_fft: usize,
            n_samples: usize,
            nb_max_frames: usize,
            padding_side: String,
            padding_value: f32,
            processor_class: String,
            return_attention_mask: bool,
            sampling_rate: usize,
            mel_filters: Option<Vec<Vec<f64>>>,
        }
        let aux: PreprocessorConfigAux = serde_json::from_reader(reader)?;

        let mel_filters = if let Some(mel_filters) = aux.mel_filters {
            let rows = mel_filters.len();
            let cols = mel_filters.first().map(|row| row.len()).unwrap_or_default();
            Array2::from_shape_vec((rows, cols), mel_filters.into_iter().flatten().collect())?
        } else {
            mel(
                aux.sampling_rate as f64,
                aux.n_fft,
                aux.feature_size,
                None,
                None,
                false,
                true,
            )
        };

        Ok(Self {
            chunk_length: aux.chunk_length,
            feature_extractor_type: aux.feature_extractor_type,
            feature_size: aux.feature_size,
            hop_length: aux.hop_length,
            n_fft: aux.n_fft,
            n_samples: aux.n_samples,
            nb_max_frames: aux.nb_max_frames,
            padding_side: aux.padding_side,
            padding_value: aux.padding_value,
            processor_class: aux.processor_class,
            return_attention_mask: aux.return_attention_mask,
            sampling_rate: aux.sampling_rate,
            mel_filters,
        })
    }
}

#[cfg(test)]
#[cfg(feature = "hub")]
mod tests {
    use crate::{download_model, Config, Device, Whisper};
    use std::path::Path;

    const MODEL_ID: &str = "jkawamoto/whisper-tiny-ct2";

    fn read_audio<T: AsRef<Path>>(path: T, sample_rate: usize) -> anyhow::Result<Vec<f32>> {
        use hound::WavReader;

        fn resample(samples: Vec<f32>, src_rate: usize, target_rate: usize) -> Vec<f32> {
            if src_rate == target_rate {
                return samples;
            }
            if src_rate > target_rate {
                let step = src_rate / target_rate;
                samples.into_iter().step_by(step).collect()
            } else {
                let factor = target_rate as f32 / src_rate as f32;
                let new_len = (samples.len() as f32 * factor) as usize;
                let mut resampled = Vec::with_capacity(new_len);
                for i in 0..new_len {
                    let src_idx = i as f32 / factor;
                    let idx_low = src_idx.floor() as usize;
                    let idx_high = (idx_low + 1).min(samples.len() - 1);
                    let weight = src_idx - idx_low as f32;
                    let val = samples[idx_low] * (1.0 - weight) + samples[idx_high] * weight;
                    resampled.push(val);
                }
                resampled
            }
        }

        let mut reader = WavReader::open(path)?;
        let spec = reader.spec();

        let max = 2_i32.pow((spec.bits_per_sample - 1) as u32) as f32;
        let samples = reader
            .samples::<i32>()
            .map(|s| s.unwrap() as f32 / max)
            .collect::<Vec<f32>>();

        if spec.channels == 1 {
            return Ok(resample(samples, spec.sample_rate as usize, sample_rate));
        }

        let mut mono = vec![];
        for chunk in samples.chunks(2) {
            if chunk.len() == 2 {
                mono.push((chunk[0] + chunk[1]) / 2.);
            }
        }

        Ok(resample(mono, spec.sample_rate as usize, sample_rate))
    }

    #[test]
    fn test_whisper_debug() {
        let model_path = download_model(MODEL_ID).unwrap();
        let w = Whisper::new(
            &model_path,
            Config {
                device: if cfg!(feature = "cuda") {
                    Device::CUDA
                } else {
                    Device::CPU
                },
                ..Default::default()
            },
        )
        .unwrap();

        assert!(format!("{:?}", w).contains(model_path.file_name().unwrap().to_str().unwrap()));
    }

    #[test]
    fn test_whisper_generate_segments() {
        let model_path = download_model(MODEL_ID).unwrap();
        let w = Whisper::new(
            &model_path,
            Config {
                device: if cfg!(feature = "cuda") {
                    Device::CUDA
                } else {
                    Device::CPU
                },
                ..Default::default()
            },
        )
        .unwrap();

        let wav_path = std::path::Path::new("tests/assets/test.wav");
        if !wav_path.exists() {
            if let Some(parent) = wav_path.parent() {
                std::fs::create_dir_all(parent).expect("failed to create directory for wav file");
            }
            let url =
                "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav";
            let response = ureq::get(url).call().expect("failed to download wav file");
            let mut out = std::fs::File::create(wav_path).expect("failed to create wav file");
            std::io::copy(&mut response.into_reader(), &mut out).expect("failed to write wav file");
        }

        let samples = read_audio(wav_path, w.sampling_rate()).unwrap();

        let segments = w
            .generate_segments(&samples, Some("en"), &Default::default())
            .unwrap();
        assert!(
            !segments.is_empty(),
            "Generated segments should not be empty"
        );

        for segment in &segments {
            println!(
                "Segment {}: [{:.2} - {:.2}]: {}",
                segment.id, segment.start, segment.end, segment.text
            );
            if let Some(words) = &segment.words {
                for word in words {
                    println!(
                        "  Word: '{}' [{:.2} - {:.2}] prob={:.3}",
                        word.word, word.start, word.end, word.probability
                    );
                    assert!(
                        word.start <= word.end,
                        "Word start time must be less than or equal to end time"
                    );
                    assert!(
                        word.probability >= 0.0 && word.probability <= 1.0,
                        "Word probability must be between 0.0 and 1.0"
                    );
                }
            }
        }
    }

    /// Transcribes the same audio with and without `initial_prompt`/`hotwords` conditioning and
    /// prints both transcripts so the difference in wording/spelling can be inspected by hand.
    /// Not a golden-output assertion: conditioning is a soft bias, so exact wording shifts are
    /// not guaranteed on arbitrary audio, but the injected conditioning tokens must never leak
    /// into the transcribed text/words either way.
    #[test]
    fn test_whisper_generate_segments_prompt_conditioning_comparison() {
        let model_path = download_model(MODEL_ID).unwrap();
        let w = Whisper::new(
            &model_path,
            Config {
                device: if cfg!(feature = "cuda") {
                    Device::CUDA
                } else {
                    Device::CPU
                },
                ..Default::default()
            },
        )
        .unwrap();

        let wav_path = std::path::Path::new("tests/assets/test.wav");
        if !wav_path.exists() {
            if let Some(parent) = wav_path.parent() {
                std::fs::create_dir_all(parent).expect("failed to create directory for wav file");
            }
            let url =
                "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav";
            let response = ureq::get(url).call().expect("failed to download wav file");
            let mut out = std::fs::File::create(wav_path).expect("failed to create wav file");
            std::io::copy(&mut response.into_reader(), &mut out).expect("failed to write wav file");
        }

        let samples = read_audio(wav_path, w.sampling_rate()).unwrap();

        let baseline = w
            .generate_segments(&samples, Some("en"), &Default::default())
            .unwrap();

        let conditioned_options = super::WhisperOptions {
            initial_prompt: Some(
                "A recording used for telephone audio quality testing.".to_string(),
            ),
            hotwords: Some("OSR".to_string()),
            ..Default::default()
        };
        let conditioned = w
            .generate_segments(&samples, Some("en"), &conditioned_options)
            .unwrap();

        assert!(
            !conditioned.is_empty(),
            "Generated segments should not be empty"
        );

        let baseline_text: String = baseline.iter().map(|s| s.text.as_str()).collect();
        let conditioned_text: String = conditioned.iter().map(|s| s.text.as_str()).collect();

        println!("--- WITHOUT initial_prompt/hotwords ---\n{baseline_text}");
        println!("--- WITH initial_prompt/hotwords ------\n{conditioned_text}");

        // The injected conditioning tokens must never leak into the transcribed text/words,
        // regardless of whether the conditioning otherwise nudges the transcribed wording.
        for segment in &conditioned {
            if let Some(words) = &segment.words {
                for word in words {
                    assert!(word.start <= word.end);
                    assert!((0.0..=1.0).contains(&word.probability));
                }
            }
        }
    }
}
