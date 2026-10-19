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

mod conditioning;
mod reconciliation;

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

pub use super::sys::WhisperOptions;

/// Optional content-conditioning inputs for [`Whisper::generate_segments_conditioned`] and
/// [`Whisper::generate_conditioned`].
///
/// Kept separate from [`WhisperOptions`], whose decoder settings cross into CTranslate2:
/// conditioning describes one recording and is only applied when a caller explicitly opts in.
/// Every field defaults to unset, and leaving `initial_prompt` and `hotwords` unset makes the
/// conditioned methods behave exactly like [`Whisper::generate_segments`] and
/// [`Whisper::generate`]. `None`, an empty string and a whitespace-only string all mean unset.
///
/// Together, `hotwords` and `initial_prompt` may use at most about half of the decoder's
/// `max_length` context (218 tokens with the default `max_length` of 448), so the conditioned
/// decode keeps the same output budget as the plain one. Longer text is truncated rather than
/// rejected: hotwords keep their beginning and the initial prompt its end, each keeping at least
/// half of that budget when both are long.
///
/// # Examples
///
/// ```
/// use ct2rs::WhisperConditioning;
///
/// let conditioning = WhisperConditioning {
///     hotwords: Some("Kubernetes"),
///     ..Default::default()
/// };
/// assert!(conditioning.initial_prompt.is_none());
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct WhisperConditioning<'a> {
    /// Optional free-form context text (e.g. topic, prior conversation). Tokenized once and
    /// prepended (via `<|startofprev|>`) to the decoder prompt used for **every** ~30s chunk.
    /// Unlike faster-whisper's `condition_on_previous_text`, this is a static prefix reused
    /// identically across chunks — it is not updated with each chunk's own transcribed output.
    ///
    /// By default each conditioned chunk is decoded once, as faster-whisper's batched pipeline
    /// does, and that transcript is returned as-is. Conditioning biases the whole decode, not
    /// just the target words, so the transcript may omit, copy from the prompt, or translate
    /// content the unconditioned transcript had. Set
    /// [`reconcile_conditioning`](WhisperConditioning::reconcile_conditioning) to rule that out
    /// in [`Whisper::generate_segments_conditioned`] at about twice the cost.
    pub initial_prompt: Option<&'a str>,
    /// Optional hint words/phrases biasing decoding toward specific vocabulary or spellings.
    /// Tokenized once and prepended (via `<|startofprev|>`, before `initial_prompt`'s tokens) to
    /// the decoder prompt used for every chunk.
    ///
    /// Decoded like `initial_prompt`, with the same shared token budget; see
    /// [`reconcile_conditioning`](WhisperConditioning::reconcile_conditioning).
    pub hotwords: Option<&'a str>,
    /// Reconciles conditioned transcripts against unconditioned ones. (default: `false`)
    ///
    /// Only [`Whisper::generate_segments_conditioned`] reads it, and only when `initial_prompt`
    /// or `hotwords` is set. When `true`, each chunk is decoded and aligned twice, with and
    /// without conditioning, and the two transcripts are merged word by word: a conditioned word
    /// is kept only where it's more confident, so the result keeps every word of the
    /// unconditioned transcript, at about 2x decode and alignment cost. A chunk whose conditioned
    /// decode copies the prompt, or whose merged word timings go backwards, keeps the
    /// unconditioned transcript. When `false`, each conditioned chunk is decoded once and that
    /// transcript is used as-is.
    ///
    /// [`Whisper::generate_conditioned`] has no word alignment to reconcile, so it
    /// ignores `reconcile_conditioning` and decodes each chunk once.
    pub reconcile_conditioning: bool,
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
        self.generate_conditioned(samples, language, timestamp, options, &Default::default())
    }

    /// Like [`generate`][Whisper::generate], but optionally biases transcription with
    /// `initial_prompt`/`hotwords` conditioning.
    ///
    /// # Arguments
    /// * `samples` - Samples of the source audio, as for [`generate`][Whisper::generate].
    /// * `language` - An optional language setting. It transcribes assuming the specified language.
    ///   If `None`, it uses Whisper's language detection.
    /// * `timestamp` - If `true`, the output will include timestamps.
    /// * `options` - Settings.
    /// * `conditioning` - Optional `initial_prompt`/`hotwords` (see [`WhisperConditioning`]). When
    ///   both are unset, this is identical to [`generate`][Whisper::generate]. When either is
    ///   set, each chunk is decoded once with `<|startofprev|>`, the hotwords and the initial
    ///   prompt placed before the prompt [`generate`][Whisper::generate] would use for the same
    ///   language and `timestamp`; the transcript may omit, copy from the prompt, or translate
    ///   content. It has no word alignment to reconcile, unlike
    ///   [`generate_segments_conditioned`][Whisper::generate_segments_conditioned] (see
    ///   [`WhisperConditioning::reconcile_conditioning`]), so it
    ///   ignores `reconcile_conditioning` and decodes each chunk once.
    ///
    /// # Returns
    /// Returns a `Result` containing a vector of transcribed strings if successful,
    /// or an error if the translation fails.
    pub fn generate_conditioned(
        &self,
        samples: &[f32],
        language: Option<&str>,
        timestamp: bool,
        options: &WhisperOptions,
        conditioning: &WhisperConditioning,
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

        let plain_prompt: Vec<String> = self
            .generate_prompt(&lang_token, timestamp)
            .into_iter()
            .map(String::from)
            .collect();
        let mut prompt =
            self.build_conditioning_prefix(conditioning, options.max_length, plain_prompt.len())?;
        prompt.extend(plain_prompt);

        self.whisper
            .generate(&storage_view, &vec![prompt; num_chunks], options)?
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
    /// * `options` - Settings.
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
        self.generate_segments_conditioned(samples, language, options, &Default::default())
    }

    /// Like [`generate_segments`][Whisper::generate_segments], but optionally biases
    /// transcription with `initial_prompt`/`hotwords` conditioning.
    ///
    /// # Arguments
    /// * `samples` - Samples of the source audio, as for
    ///   [`generate_segments`][Whisper::generate_segments].
    /// * `language` - An optional language setting. It generates segments assuming the specified language.
    ///   If `None`, it uses Whisper's language detection.
    /// * `options` - Settings.
    /// * `conditioning` - Optional `initial_prompt`/`hotwords` (see [`WhisperConditioning`]). When
    ///   both are unset, this is identical to [`generate_segments`][Whisper::generate_segments].
    ///   When either is set, each chunk is decoded once with the conditioned prompt and its words
    ///   are aligned against the plain start-of-transcript sequence, as faster-whisper's batched
    ///   pipeline does; the transcript may omit, copy from the prompt, or translate content. With
    ///   [`reconcile_conditioning`][WhisperConditioning::reconcile_conditioning] set, each
    ///   conditioned chunk is instead decoded twice (with and without conditioning) and the two
    ///   transcripts are reconciled word by word, keeping every word of the unconditioned
    ///   transcript at roughly 2x the decode and alignment cost.
    ///
    /// # Returns
    /// Returns a `Result` containing a vector of transcribed `Segment`s if successful,
    /// or an error if the segment generation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ct2rs::{Whisper, WhisperConditioning, WhisperOptions};
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let whisper = Whisper::new("/path/to/model", Default::default())?;
    /// let samples: Vec<f32> = vec![];
    /// let options = WhisperOptions::default();
    ///
    /// let conditioning = WhisperConditioning {
    ///     hotwords: Some("Kubernetes"),
    ///     ..Default::default()
    /// };
    /// let segments =
    ///     whisper.generate_segments_conditioned(&samples, Some("en"), &options, &conditioning)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate_segments_conditioned(
        &self,
        samples: &[f32],
        language: Option<&str>,
        options: &WhisperOptions,
        conditioning: &WhisperConditioning,
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

        // For alignment and timing, we do want timestamps
        let sot_sequence: Vec<String> = self
            .generate_prompt(&lang_token, true)
            .into_iter()
            .map(String::from)
            .collect();

        let conditioning_prefix =
            self.build_conditioning_prefix(conditioning, options.max_length, sot_sequence.len())?;
        if conditioning_prefix.is_empty() {
            return self.generate_segments_once(
                &encoder_output,
                &sot_sequence,
                &sot_sequence,
                num_chunks,
                options,
            );
        }

        let conditioned_prompt: Vec<String> = conditioning_prefix
            .into_iter()
            .chain(sot_sequence.iter().cloned())
            .collect();

        if conditioning.reconcile_conditioning {
            self.generate_segments_with_conditioning(
                &encoder_output,
                &sot_sequence,
                &conditioned_prompt,
                num_chunks,
                options,
            )
        } else {
            self.generate_segments_once(
                &encoder_output,
                &conditioned_prompt,
                &sot_sequence,
                num_chunks,
                options,
            )
        }
    }

    /// Decodes every chunk once with `prompt` and aligns the result against `alignment_prompt`.
    ///
    /// Words are aligned against the plain start-of-transcript sequence even when `prompt` carries
    /// a conditioning prefix, as faster-whisper's `find_alignment` does: the decoded tokens are
    /// the same continuation of the same task token either way, and a long prefix would otherwise
    /// eat into the decoder positions the alignment needs.
    fn generate_segments_once(
        &self,
        encoder_output: &sys::StorageView,
        prompt: &[String],
        alignment_prompt: &[String],
        num_chunks: usize,
        options: &WhisperOptions,
    ) -> Result<Vec<Segment>> {
        let gen_results =
            self.whisper
                .generate(encoder_output, &vec![prompt.to_vec(); num_chunks], options)?;
        let alignment_results =
            self.align_chunks(encoder_output, alignment_prompt, &gen_results, num_chunks)?;

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
            let (words, _) = self.words_from_tokens(tokens, align_res, chunk_offset)?;

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

/// Whether the token carries the GPT-2/Whisper (`Ġ`) or a plain leading-space marker.
fn has_leading_space(token: &str) -> bool {
    token.starts_with('Ġ') || token.starts_with(' ')
}

/// Characters that, standing alone after a space, open the word that follows them instead of
/// closing the one before (openai-whisper's default `prepend_punctuations`).
const OPENING_PUNCTUATION: &[char] = &['"', '\'', '“', '¿', '(', '[', '{', '-'];

/// A word of a decoded chunk: its token range, its display text, and whether a space precedes it.
#[derive(Clone, Debug, PartialEq)]
struct WordTokens {
    range: std::ops::Range<usize>,
    /// Decoded, trimmed text. Absorbed punctuation is attached without the space it may have
    /// been decoded with, so a word never contains whitespace it didn't already have.
    text: String,
    /// Whether the word's first token carries a leading-space marker.
    space_before: bool,
}

/// How a piece produced by [`split_tokens_into_pieces`] takes part in punctuation merging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PieceKind {
    /// Contains at least one alphanumeric character.
    Word,
    /// Punctuation-only after a space, made only of [`OPENING_PUNCTUATION`]: joins the next word.
    Opening,
    /// Any other punctuation-only (or whitespace-only) piece: joins the previous word.
    Trailing,
}

impl PieceKind {
    /// Classifies a piece by its decoded `text`, not its raw token strings: Whisper's byte-level
    /// BPE spells non-ASCII characters as other characters (`，` is `ï¼Į`), which would read as
    /// letters.
    fn of(text: &str, first_token: &str) -> Self {
        if text.chars().any(char::is_alphanumeric) {
            PieceKind::Word
        } else if !text.is_empty()
            && has_leading_space(first_token)
            && text.chars().all(|c| OPENING_PUNCTUATION.contains(&c))
        {
            PieceKind::Opening
        } else {
            PieceKind::Trailing
        }
    }
}

/// Groups decoded tokens into words, attaching punctuation to a neighbouring word as
/// openai-whisper's `merge_punctuations` does: opening punctuation joins the word after it, any
/// other punctuation-only piece joins the word before it. Merging stays within a run of text
/// tokens between special tokens; a run with no word in it is returned as split.
fn group_tokens_into_words(
    tokens: &[String],
    mut decode: impl FnMut(&[String]) -> Result<String>,
) -> Result<Vec<WordTokens>> {
    let pieces = split_tokens_into_pieces(tokens);
    let mut words = Vec::with_capacity(pieces.len());
    for run in pieces.chunk_by(|a, b| a.end == b.start) {
        let mut classified = Vec::with_capacity(run.len());
        for range in run {
            let text = decode(&tokens[range.clone()])?.trim().to_string();
            let kind = PieceKind::of(&text, &tokens[range.start]);
            classified.push((range.clone(), text, kind));
        }
        merge_punctuation_run(tokens, classified, &mut words);
    }
    Ok(words)
}

/// Appends one run's pieces to `words`, merging its punctuation-only pieces into words (see
/// [`group_tokens_into_words`]). Punctuation with no word on its usual side joins the other one.
fn merge_punctuation_run(
    tokens: &[String],
    pieces: Vec<(std::ops::Range<usize>, String, PieceKind)>,
    words: &mut Vec<WordTokens>,
) {
    let word_tokens = |range: std::ops::Range<usize>, text: String| WordTokens {
        space_before: has_leading_space(&tokens[range.start]),
        range,
        text,
    };

    if pieces.iter().all(|(_, _, kind)| *kind != PieceKind::Word) {
        words.extend(
            pieces
                .into_iter()
                .map(|(range, text, _)| word_tokens(range, text)),
        );
        return;
    }

    let run_start = words.len();
    let mut run_end = 0;
    // Punctuation waiting for the next word of this run: its first token index and its text.
    let mut pending: Option<(usize, String)> = None;
    for (range, text, kind) in pieces {
        run_end = range.end;
        match kind {
            PieceKind::Word => {
                let (start, prefix) = pending.take().unwrap_or((range.start, String::new()));
                words.push(word_tokens(start..range.end, prefix + &text));
            }
            PieceKind::Trailing if words.len() > run_start => {
                let last = words.last_mut().expect("checked above");
                if let Some((_, waiting)) = pending.take() {
                    last.text.push_str(&waiting);
                }
                last.range.end = range.end;
                last.text.push_str(&text);
            }
            PieceKind::Trailing | PieceKind::Opening => {
                pending
                    .get_or_insert_with(|| (range.start, String::new()))
                    .1
                    .push_str(&text);
            }
        }
    }

    if let Some((_, waiting)) = pending {
        let last = words.last_mut().expect("the run contains a word");
        last.range.end = run_end;
        last.text.push_str(&waiting);
    }
}

/// Splits decoded tokens into pieces at leading-space markers, punctuation and special tokens,
/// before any punctuation is merged back (see [`group_tokens_into_words`]).
fn split_tokens_into_pieces(tokens: &[String]) -> Vec<std::ops::Range<usize>> {
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
        let words = group_tokens_into_words(&tokens, decode_test_tokens).unwrap();
        let ranges: Vec<_> = words.iter().map(|w| w.range.clone()).collect();
        assert_eq!(ranges, vec![3..5, 5..7]);
        assert_eq!(texts(&words), vec!["Hellollo", "world!"]);
    }

    /// Stands in for the model tokenizer: `Ġ` decodes to a space and the byte-level spelling of
    /// `，` to `，`; everything else decodes to itself.
    fn decode_test_tokens(tokens: &[String]) -> Result<String> {
        Ok(tokens.concat().replace('Ġ', " ").replace("ï¼Į", "，"))
    }

    fn group(tokens: &[&str]) -> Vec<WordTokens> {
        let tokens: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
        group_tokens_into_words(&tokens, decode_test_tokens).unwrap()
    }

    fn texts(words: &[WordTokens]) -> Vec<&str> {
        words.iter().map(|w| w.text.as_str()).collect()
    }

    #[test]
    fn test_trailing_punctuation_runs_join_previous_word() {
        assert_eq!(texts(&group(&["Ġworld", "?", "!"])), vec!["world?!"]);
        assert_eq!(texts(&group(&["Ġworld", ".\""])), vec!["world.\""]);
    }

    #[test]
    fn test_any_punctuation_only_word_joins_previous_word() {
        for (punctuation, expected) in [
            (";", "wait;"),
            ("...", "wait..."),
            // Space-marked, so the splitter makes it a piece of its own and the merge must act.
            ("Ġ…", "wait…"),
            ("Ġ—", "wait—"),
        ] {
            assert_eq!(texts(&group(&["Ġwait", punctuation])), vec![expected]);
        }
        assert_eq!(texts(&group(&["Ġ50", "%"])), vec!["50%"]);
    }

    #[test]
    fn test_punctuation_is_classified_on_decoded_text() {
        // `Ġï¼Į` decodes to ` ，`: punctuation-only, though its raw characters read as letters.
        assert_eq!(texts(&group(&["Ġwait", "Ġï¼Į"])), vec!["wait，"]);
    }

    #[test]
    fn test_merged_word_timing_and_probability() {
        use crate::sys::WhisperTokenAlignment;

        let words = group(&["Ġworld", "!"]);
        let ranges: Vec<_> = words.iter().map(|w| w.range.clone()).collect();
        assert_eq!(ranges, vec![0..2]);

        let alignments = [(0, 10), (0, 14), (1, 15), (1, 16)]
            .map(|(token_x, frame_x)| WhisperTokenAlignment { token_x, frame_x });
        let timed = process_word_timings(&ranges, &alignments, &[0.9, 0.5], 2);
        assert_eq!(timed.len(), 1);
        assert_eq!(timed[0].start, 10.0 / 50.0, "start of the `world` token");
        assert_eq!(timed[0].end, 17.0 / 50.0, "end of the `!` token");
        assert!((timed[0].probability - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_spaced_opening_punctuation_joins_next_word() {
        let said = group(&["Ġsaid", "Ġ\"", "Hello"]);
        assert_eq!(texts(&said), vec!["said", "\"Hello"]);
        assert!(said[1].space_before);
        assert_eq!(
            texts(&group(&["Ġsaid", "Ġ\"", "ĠHello"])),
            vec!["said", "\"Hello"]
        );
        assert_eq!(texts(&group(&["Ġ(", "see"])), vec!["(see"]);
        assert_eq!(texts(&group(&["Ġ¿", "Qué"])), vec!["¿Qué"]);
        assert_eq!(
            texts(&group(&["Ġsee", "Ġ(", "\"", "Hello"])),
            vec!["see", "(\"Hello"]
        );
    }

    #[test]
    fn test_unspaced_quote_joins_previous_word() {
        assert_eq!(
            texts(&group(&["ĠHello", "\"", "Ġthen"])),
            vec!["Hello\"", "then"]
        );
    }

    #[test]
    fn test_run_edge_punctuation_joins_other_neighbour() {
        let then = group(&[".", "Ġthen"]);
        assert_eq!(texts(&then), vec![".then"]);
        assert!(!then[0].space_before);
        assert_eq!(texts(&group(&["ĠHello", "Ġ("])), vec!["Hello("]);
    }

    #[test]
    fn test_all_punctuation_run_is_unchanged() {
        assert_eq!(texts(&group(&["..."])), vec!["..."]);
        assert_eq!(texts(&group(&[".", "!"])), vec![".", "!"]);
    }

    #[test]
    fn test_merge_never_crosses_a_special_token() {
        let words = group(&["Ġworld", "<|5.00|>", ".", "ĠNext"]);
        let ranges: Vec<_> = words.iter().map(|w| w.range.clone()).collect();
        assert_eq!(ranges, vec![0..1, 2..4]);
        assert_eq!(texts(&words), vec!["world", ".Next"]);
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
    use crate::{download_model, Config, Device, Tokenizer, Whisper, WhisperConditioning};
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

        let conditioning = WhisperConditioning {
            initial_prompt: Some("A recording used for telephone audio quality testing."),
            hotwords: Some("OSR"),
            ..Default::default()
        };
        let conditioned = w
            .generate_segments_conditioned(&samples, Some("en"), &Default::default(), &conditioning)
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

    fn load_test_audio(w: &Whisper) -> Vec<f32> {
        let wav_path = Path::new("tests/assets/test.wav");
        assert!(
            wav_path.exists(),
            "run test_whisper_generate_segments_prompt_conditioning_comparison first to fetch it"
        );
        read_audio(wav_path, w.sampling_rate()).unwrap()
    }

    fn load_test_model() -> Whisper {
        let model_path = download_model(MODEL_ID).unwrap();
        Whisper::new(
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
        .unwrap()
    }

    /// `generate_segments` must behave exactly like `generate_segments_conditioned` with the
    /// default conditioning: byte-for-byte the same segments, word timings and probabilities.
    ///
    /// That comparison is only meaningful on a backend that reproduces its own output for
    /// identical calls. Some don't (e.g. multithreaded OpenBLAS on Windows), so the check is
    /// skipped there.
    #[test]
    fn test_whisper_generate_segments_is_conditioned_with_default() {
        let w = load_test_model();
        let samples = load_test_audio(&w);
        let options = super::WhisperOptions::default();
        if !is_deterministic(&w, &samples, &WhisperConditioning::default()) {
            return;
        }

        let plain = w.generate_segments(&samples, Some("en"), &options).unwrap();
        assert!(!plain.is_empty());
        let conditioned = w
            .generate_segments_conditioned(
                &samples,
                Some("en"),
                &options,
                &WhisperConditioning::default(),
            )
            .unwrap();
        assert_eq!(conditioned, plain);
    }

    /// Every way of leaving the texts unset (`None`, empty, whitespace-only).
    const UNSET_TEXTS: [Option<&str>; 3] = [None, Some(""), Some(" \n")];

    /// Unset `initial_prompt`/`hotwords` must be a true no-op: byte-for-byte the same segments,
    /// word timings and probabilities as `generate_segments`, whatever `reconcile_conditioning`
    /// says. Skipped on a backend that doesn't reproduce its own output (see
    /// `test_whisper_generate_segments_is_conditioned_with_default`);
    /// `test_whisper_conditioning_prefix_contains_hotwords_then_initial_prompt` still covers
    /// blank conditioning producing no prefix.
    #[test]
    fn test_whisper_unset_conditioning_is_identical_to_plain() {
        let w = load_test_model();
        let samples = load_test_audio(&w);
        let options = super::WhisperOptions::default();
        if !is_deterministic(&w, &samples, &WhisperConditioning::default()) {
            return;
        }

        let plain = w.generate_segments(&samples, Some("en"), &options).unwrap();
        assert!(!plain.is_empty());
        for initial_prompt in UNSET_TEXTS {
            for hotwords in UNSET_TEXTS {
                for reconcile_conditioning in [false, true] {
                    let conditioning = WhisperConditioning {
                        initial_prompt,
                        hotwords,
                        reconcile_conditioning,
                    };
                    let conditioned = w
                        .generate_segments_conditioned(
                            &samples,
                            Some("en"),
                            &options,
                            &conditioning,
                        )
                        .unwrap();
                    assert_eq!(conditioned, plain, "{conditioning:?} changed the output");
                }
            }
        }
    }

    /// Set conditioning must actually reach the decoder prompt: `<|startofprev|>`, then the
    /// hotwords, then the initial prompt, ahead of the regular start-of-transcript sequence.
    #[test]
    fn test_whisper_conditioning_prefix_contains_hotwords_then_initial_prompt() {
        let w = load_test_model();
        let prefix_of = |initial_prompt: Option<&str>, hotwords: Option<&str>| {
            let conditioning = WhisperConditioning {
                initial_prompt,
                hotwords,
                ..Default::default()
            };
            w.build_conditioning_prefix(&conditioning, 448, 3).unwrap()
        };

        let prefix = prefix_of(Some("telephone audio"), Some("OSR"));
        assert_eq!(prefix[0], "<|startofprev|>");
        let decoded = w.tokenizer.decode(prefix[1..].to_vec()).unwrap();
        assert_eq!(decoded.trim(), "OSR telephone audio");

        let only_hotwords = prefix_of(None, Some("OSR"));
        assert_eq!(only_hotwords[0], "<|startofprev|>");
        assert_eq!(
            w.tokenizer
                .decode(only_hotwords[1..].to_vec())
                .unwrap()
                .trim(),
            "OSR"
        );

        assert!(prefix_of(None, None).is_empty());
        assert!(prefix_of(Some("  "), Some("")).is_empty());
        assert!(prefix_of(Some(""), Some(" \n")).is_empty());
    }

    /// Conditioning far longer than the decoder context must be truncated to fit, not fail with a
    /// position-encoding error in `generate` or `align`. Silence makes the decode run to its
    /// length limit, which is when an oversized prefix overflows the 448 positions.
    #[test]
    fn test_whisper_oversized_conditioning_fits_decoder_context() {
        let w = load_test_model();
        let long_prompt = LONG_PROMPT.repeat(200);
        let long_hotwords = LONG_HOTWORDS.repeat(200);
        assert!(bpe_pieces(&w, &long_prompt).len() > 218);
        assert!(bpe_pieces(&w, &long_hotwords).len() > 218);

        let options = super::WhisperOptions::default();
        assert_eq!(options.max_length, 448);
        for conditioning in oversized_conditionings(&long_prompt, &long_hotwords) {
            let prefix = w
                .build_conditioning_prefix(&conditioning, options.max_length, 3)
                .unwrap();
            assert!(prefix.len() + 3 <= options.max_length / 2 - 2);

            for samples in [vec![0.0f32; w.n_samples()], load_test_audio(&w)] {
                let segments = w
                    .generate_segments_conditioned(&samples, Some("en"), &options, &conditioning)
                    .unwrap_or_else(|e| panic!("oversized conditioning failed: {e:#}"));
                assert!(!segments.is_empty());
            }
        }
    }

    const LONG_PROMPT: &str = "A recording used for telephone audio quality testing. ";
    const LONG_HOTWORDS: &str = "OSR, telephone, harvard sentences, ";

    /// The prompt alone, the hotwords alone, then both.
    fn oversized_conditionings<'a>(
        long_prompt: &'a str,
        long_hotwords: &'a str,
    ) -> [WhisperConditioning<'a>; 3] {
        [
            (Some(long_prompt), None),
            (None, Some(long_hotwords)),
            (Some(long_prompt), Some(long_hotwords)),
        ]
        .map(|(initial_prompt, hotwords)| WhisperConditioning {
            initial_prompt,
            hotwords,
            ..Default::default()
        })
    }

    /// Asserts no segment returns a punctuation-only word next to real words, and that the
    /// audio's sentence punctuation did reach some word (so the check isn't vacuous).
    fn assert_punctuation_attached(segments: &[super::Segment]) {
        let is_punctuation_only = |w: &str| !w.chars().any(char::is_alphanumeric);
        let mut any_ends_in_punctuation = false;
        for segment in segments {
            let words = segment.words.as_deref().unwrap_or_default();
            if words.iter().all(|w| is_punctuation_only(&w.word)) {
                continue;
            }
            for word in words {
                assert!(
                    !is_punctuation_only(&word.word),
                    "segment {} returned punctuation-only word {:?}",
                    segment.id,
                    word.word
                );
                any_ends_in_punctuation |= word.word.ends_with(|c: char| c.is_ascii_punctuation());
            }
        }
        assert!(any_ends_in_punctuation, "no word carries punctuation");
    }

    #[test]
    fn test_whisper_generate_segments_attaches_punctuation() {
        let w = load_test_model();
        let samples = load_test_audio(&w);
        let segments = w
            .generate_segments(&samples, Some("en"), &Default::default())
            .unwrap();
        assert_punctuation_attached(&segments);
    }

    #[test]
    fn test_whisper_conditioned_segments_attach_punctuation() {
        let w = load_test_model();
        let samples = load_test_audio(&w);
        let segments = w
            .generate_segments_conditioned(
                &samples,
                Some("en"),
                &Default::default(),
                &conditioned(false),
            )
            .unwrap();
        assert_punctuation_attached(&segments);
    }

    const CONDITIONED_PROMPT: &str = "A recording used for telephone audio quality testing.";
    const CONDITIONED_HOTWORDS: &str = "OSR";

    fn conditioned(reconcile_conditioning: bool) -> WhisperConditioning<'static> {
        WhisperConditioning {
            initial_prompt: Some(CONDITIONED_PROMPT),
            hotwords: Some(CONDITIONED_HOTWORDS),
            reconcile_conditioning,
        }
    }

    /// Runs the encoder over `samples`, returning its output and the number of chunks.
    fn encode_samples(w: &Whisper, samples: &[f32]) -> (crate::sys::StorageView<'static>, usize) {
        let (mut mel, num_chunks) = w.generate_mel_spectrogram(samples).unwrap();
        let shape = mel.shape().to_vec();
        let features =
            crate::sys::StorageView::new(&shape, mel.as_slice_mut().unwrap(), Default::default())
                .unwrap();
        (w.whisper.encode(&features, false).unwrap(), num_chunks)
    }

    /// Raw BPE pieces of `" " + text`, the way faster-whisper encodes prompt text.
    fn bpe_pieces(w: &Whisper, text: &str) -> Vec<String> {
        (*w.tokenizer)
            .encode(format!(" {text}").as_str(), false)
            .unwrap()
            .get_tokens()
            .to_vec()
    }

    fn plain_sot() -> Vec<String> {
        ["<|startoftranscript|>", "<|en|>", "<|transcribe|>"]
            .map(String::from)
            .to_vec()
    }

    /// `<|startofprev|>`, hotword tokens, initial-prompt tokens, then the plain sot sequence.
    fn conditioned_prompt(w: &Whisper) -> Vec<String> {
        let mut prompt = vec!["<|startofprev|>".to_string()];
        prompt.extend(bpe_pieces(w, CONDITIONED_HOTWORDS));
        prompt.extend(bpe_pieces(w, CONDITIONED_PROMPT));
        prompt.extend(plain_sot());
        prompt
    }

    /// Whether this backend reproduces its own `generate_segments_conditioned` output for
    /// identical calls.
    fn is_deterministic(w: &Whisper, samples: &[f32], conditioning: &WhisperConditioning) -> bool {
        let segments = || {
            w.generate_segments_conditioned(samples, Some("en"), &Default::default(), conditioning)
                .unwrap()
        };
        if segments() != segments() {
            eprintln!("skipping: this backend's output differs between identical calls");
            return false;
        }
        true
    }

    /// faster-whisper's `test_hotwords`, on its own `hotwords.mp3` converted to 16 kHz mono WAV.
    #[test]
    fn test_whisper_hotwords_single_pass_finds_comfyui() {
        let w = load_test_model();
        let samples = read_audio("tests/assets/hotwords.wav", w.sampling_rate()).unwrap();
        let conditioning = WhisperConditioning {
            hotwords: Some("ComfyUI"),
            reconcile_conditioning: false,
            ..Default::default()
        };
        let segments = w
            .generate_segments_conditioned(&samples, None, &Default::default(), &conditioning)
            .unwrap();
        assert!(
            segments[0].text.contains("ComfyUI"),
            "first segment: {:?}",
            segments[0].text
        );
    }

    /// Without `reconcile_conditioning`, a conditioned chunk's text is exactly what a single
    /// decode with the conditioned prompt produces.
    #[test]
    fn test_whisper_single_pass_conditioning_is_one_decode() {
        let w = load_test_model();
        let samples = load_test_audio(&w);
        let options = super::WhisperOptions::default();
        let conditioning = conditioned(false);
        if !is_deterministic(&w, &samples, &conditioning) {
            return;
        }

        let segments = w
            .generate_segments_conditioned(&samples, Some("en"), &options, &conditioning)
            .unwrap();

        let (encoder_output, num_chunks) = encode_samples(&w, &samples);
        let decoded = w
            .whisper
            .generate(
                &encoder_output,
                &vec![conditioned_prompt(&w); num_chunks],
                &options,
            )
            .unwrap();
        let expected: Vec<String> = decoded
            .into_iter()
            .map(|res| {
                let text_tokens: Vec<String> = res.sequences[0]
                    .iter()
                    .filter(|t| !super::is_special_token(t))
                    .cloned()
                    .collect();
                w.tokenizer.decode(text_tokens).unwrap().trim().to_string()
            })
            .collect();

        let texts: Vec<&str> = segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, expected);
    }

    /// Without `reconcile_conditioning`, word timings and probabilities come from aligning the
    /// conditioned decode against the plain start-of-transcript sequence, as faster-whisper does.
    #[test]
    fn test_whisper_single_pass_conditioning_aligns_against_plain_sot() {
        let w = load_test_model();
        let samples = load_test_audio(&w);
        let options = super::WhisperOptions::default();
        let conditioning = conditioned(false);
        if !is_deterministic(&w, &samples, &conditioning) {
            return;
        }

        let segments = w
            .generate_segments_conditioned(&samples, Some("en"), &options, &conditioning)
            .unwrap();

        let (encoder_output, num_chunks) = encode_samples(&w, &samples);
        let prompt = conditioned_prompt(&w);
        let decoded = w
            .whisper
            .generate(&encoder_output, &vec![prompt.clone(); num_chunks], &options)
            .unwrap();
        let words_aligned_against = |start_sequence: &[String]| -> Vec<Vec<super::Word>> {
            let alignments = w
                .align_chunks(&encoder_output, start_sequence, &decoded, num_chunks)
                .unwrap();
            decoded
                .iter()
                .zip(&alignments)
                .enumerate()
                .map(|(chunk_idx, (res, align_res))| {
                    let offset = (chunk_idx * w.n_samples()) as f32 / w.sampling_rate() as f32;
                    w.words_from_tokens(&res.sequences[0], align_res, offset)
                        .unwrap()
                        .0
                })
                .collect()
        };

        let words: Vec<Vec<super::Word>> = segments
            .iter()
            .map(|s| s.words.clone().unwrap_or_default())
            .collect();
        assert_eq!(words, words_aligned_against(&plain_sot()));
        assert_ne!(
            words,
            words_aligned_against(&prompt),
            "aligning against the conditioned prompt must give other timings, or this test can't \
             tell the two apart"
        );
    }

    /// With `reconcile_conditioning`, conditioned segments are ADR 001's two-pass reconciliation.
    /// The expected text was recorded from that implementation before single-pass conditioning
    /// existed, on the CPU `ruy` backend; other backends may legitimately transcribe differently,
    /// so they skip.
    #[test]
    fn test_whisper_reconciled_conditioning_matches_adr_001() {
        if !cfg!(feature = "ruy") || cfg!(feature = "cuda") {
            eprintln!("skipping: expected text was recorded on the CPU ruy backend");
            return;
        }
        let w = load_test_model();
        let samples = load_test_audio(&w);
        let segments = w
            .generate_segments_conditioned(
                &samples,
                Some("en"),
                &Default::default(),
                &conditioned(true),
            )
            .unwrap();
        let texts: Vec<&str> = segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "The birch can use lid on the smooth planks. Glue the sheet to the dark blue \
                 background. It is easy to tell the depth of a well. These days, the chicken leg \
                 is a rare dish. Rice is also served in round bowls. The juice of lemon makes fine \
                 punch. The box was thrown beside the pork chuck. The hogs are fed chopped corn \
                 and garbage.",
                "A large size and stockings is hard to sell.",
            ]
        );
        let word_counts: Vec<usize> = segments
            .iter()
            .map(|s| s.words.as_ref().map_or(0, Vec::len))
            .collect();
        assert_eq!(word_counts, vec![66, 9]);
    }

    /// Attaching punctuation to words must not change the unconditioned `Segment.text`, which is
    /// decoded from the tokens. The expected text was recorded before the change, on the CPU
    /// `ruy` backend; other backends may legitimately transcribe differently, so they skip.
    #[test]
    fn test_whisper_plain_segment_text_matches_base() {
        if !cfg!(feature = "ruy") || cfg!(feature = "cuda") {
            eprintln!("skipping: expected text was recorded on the CPU ruy backend");
            return;
        }
        let w = load_test_model();
        let samples = load_test_audio(&w);
        let segments = w
            .generate_segments(&samples, Some("en"), &Default::default())
            .unwrap();
        let texts: Vec<&str> = segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "The birch can use lid on the smooth planks. Glue the sheet to the dark blue \
                 background. It is easy to tell the depth of a well. These days, the chicken leg \
                 is a rare dish. Rice is also served in round bowls. The juice of lemon makes fine \
                 punch. The box was thrown beside the pork truck. The hogs were fed chopped corn \
                 and garbage.",
                "A large size and stockings is hard to sell.",
            ]
        );
    }

    /// Whether this backend reproduces its own `generate_conditioned` output for identical calls.
    fn generate_is_deterministic(
        w: &Whisper,
        samples: &[f32],
        timestamp: bool,
        conditioning: &WhisperConditioning,
    ) -> bool {
        let strings = || {
            w.generate_conditioned(
                samples,
                Some("en"),
                timestamp,
                &Default::default(),
                conditioning,
            )
            .unwrap()
        };
        if strings() != strings() {
            eprintln!("skipping: this backend's output differs between identical calls");
            return false;
        }
        true
    }

    /// `generate` must behave exactly like `generate_conditioned` with the default conditioning.
    #[test]
    fn test_whisper_generate_is_conditioned_with_default() {
        let w = load_test_model();
        let samples = load_test_audio(&w);
        let options = super::WhisperOptions::default();
        for timestamp in [true, false] {
            if !generate_is_deterministic(&w, &samples, timestamp, &Default::default()) {
                return;
            }
            let plain = w
                .generate(&samples, Some("en"), timestamp, &options)
                .unwrap();
            assert!(!plain.is_empty());
            let conditioned = w
                .generate_conditioned(
                    &samples,
                    Some("en"),
                    timestamp,
                    &options,
                    &WhisperConditioning::default(),
                )
                .unwrap();
            assert_eq!(conditioned, plain, "timestamp: {timestamp}");
        }
    }

    /// Unset `initial_prompt`/`hotwords` must leave `generate_conditioned` identical to
    /// `generate`, even with `reconcile_conditioning` set.
    #[test]
    fn test_whisper_generate_unset_conditioning_is_identical_to_plain() {
        let w = load_test_model();
        let samples = load_test_audio(&w);
        let options = super::WhisperOptions::default();
        for timestamp in [true, false] {
            if !generate_is_deterministic(&w, &samples, timestamp, &Default::default()) {
                return;
            }
            let plain = w
                .generate(&samples, Some("en"), timestamp, &options)
                .unwrap();
            for initial_prompt in UNSET_TEXTS {
                for hotwords in UNSET_TEXTS {
                    let conditioning = WhisperConditioning {
                        initial_prompt,
                        hotwords,
                        reconcile_conditioning: true,
                    };
                    let conditioned = w
                        .generate_conditioned(
                            &samples,
                            Some("en"),
                            timestamp,
                            &options,
                            &conditioning,
                        )
                        .unwrap();
                    assert_eq!(
                        conditioned, plain,
                        "{conditioning:?} with timestamp {timestamp} changed the output"
                    );
                }
            }
        }
    }

    /// `generate_conditioned` decodes every chunk once with `<|startofprev|>`, the hotwords and
    /// the initial prompt, followed by exactly the prompt `generate` uses: `<|notimestamps|>`
    /// last when `timestamp` is `false`, absent when it is `true`.
    #[test]
    fn test_whisper_generate_conditioned_prompt_is_prefix_then_plain_prompt() {
        let w = load_test_model();
        let samples = load_test_audio(&w);
        let options = super::WhisperOptions::default();
        let conditioning = conditioned(false);

        let (mut mel, num_chunks) = w.generate_mel_spectrogram(&samples).unwrap();
        let shape = mel.shape().to_vec();
        let features =
            crate::sys::StorageView::new(&shape, mel.as_slice_mut().unwrap(), Default::default())
                .unwrap();
        let decode_with = |prompt: Vec<String>| -> Vec<String> {
            w.whisper
                .generate(&features, &vec![prompt; num_chunks], &options)
                .unwrap()
                .into_iter()
                .map(|res| w.tokenizer.decode(res.sequences[0].clone()).unwrap())
                .collect()
        };

        for timestamp in [true, false] {
            if !generate_is_deterministic(&w, &samples, timestamp, &conditioning) {
                return;
            }
            let mut prompt = conditioned_prompt(&w);
            if !timestamp {
                prompt.push("<|notimestamps|>".to_string());
            }
            let strings = w
                .generate_conditioned(&samples, Some("en"), timestamp, &options, &conditioning)
                .unwrap();
            assert_eq!(strings, decode_with(prompt), "timestamp: {timestamp}");
        }
    }

    /// faster-whisper's `test_hotwords` through `generate_conditioned`, with timestamps.
    #[test]
    fn test_whisper_generate_conditioned_hotwords_finds_comfyui() {
        let w = load_test_model();
        let samples = read_audio("tests/assets/hotwords.wav", w.sampling_rate()).unwrap();
        let conditioning = WhisperConditioning {
            hotwords: Some("ComfyUI"),
            ..Default::default()
        };
        let strings = w
            .generate_conditioned(&samples, None, true, &Default::default(), &conditioning)
            .unwrap();
        assert!(
            strings[0].contains("ComfyUI"),
            "first string: {:?}",
            strings[0]
        );
    }

    /// `generate_conditioned` is always single-pass: `reconcile_conditioning` changes nothing.
    #[test]
    fn test_whisper_generate_conditioned_ignores_reconcile_conditioning() {
        let w = load_test_model();
        let samples = load_test_audio(&w);
        let options = super::WhisperOptions::default();
        for timestamp in [true, false] {
            if !generate_is_deterministic(&w, &samples, timestamp, &conditioned(false)) {
                return;
            }
            let strings = |conditioning: &WhisperConditioning| {
                w.generate_conditioned(&samples, Some("en"), timestamp, &options, conditioning)
                    .unwrap()
            };
            assert_eq!(
                strings(&conditioned(true)),
                strings(&conditioned(false)),
                "timestamp: {timestamp}"
            );
        }
    }

    /// Oversized conditioning must fit the budget of the start-of-transcript sequence each
    /// `timestamp` value uses (3 tokens, or 4 with `<|notimestamps|>`) and never fail.
    #[test]
    fn test_whisper_generate_conditioned_oversized_conditioning_fits_decoder_context() {
        let w = load_test_model();
        let long_prompt = LONG_PROMPT.repeat(200);
        let long_hotwords = LONG_HOTWORDS.repeat(200);
        let options = super::WhisperOptions::default();
        assert_eq!(options.max_length, 448);

        for (timestamp, sot_len) in [(true, 3), (false, 4)] {
            for conditioning in oversized_conditionings(&long_prompt, &long_hotwords) {
                let prefix = w
                    .build_conditioning_prefix(&conditioning, options.max_length, sot_len)
                    .unwrap();
                assert!(prefix.len() + sot_len <= options.max_length / 2 - 2);

                for samples in [vec![0.0f32; w.n_samples()], load_test_audio(&w)] {
                    let strings = w
                        .generate_conditioned(
                            &samples,
                            Some("en"),
                            timestamp,
                            &options,
                            &conditioning,
                        )
                        .unwrap_or_else(|e| panic!("oversized conditioning failed: {e:#}"));
                    assert!(!strings.is_empty());
                }
            }
        }
    }
}
