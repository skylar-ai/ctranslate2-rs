# How faster-whisper differs from `generate_segments`

This page compares `ct2rs::Whisper::generate_segments` with the transcription
pipelines in [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Use it
when you port code from faster-whisper, compare transcripts between the two, or
decide which faster-whisper behavior to bring into this crate. Conditioning
(`initial_prompt`, `hotwords`, `reconcile_conditioning`) is passed as a
`WhisperConditioning` to `generate_segments_conditioned`; with none set, it
behaves exactly like `generate_segments`.

The comparison uses faster-whisper's
[`faster_whisper/transcribe.py` at commit `9e472ae`](https://github.com/SYSTRAN/faster-whisper/blob/9e472ae9de8bb374552f531f752654554b76b2d6/faster_whisper/transcribe.py),
reviewed on October 19, 2026.

## Two faster-whisper pipelines

faster-whisper ships two pipelines, and only one of them resembles
`generate_segments`:

- **`WhisperModel.transcribe`** (sequential) decodes one 30-second window at a
  time. Each window can depend on the previous window's output, so windows run
  one after another.
- **`BatchedInferencePipeline.transcribe`** (batched) cuts the audio into
  speech chunks with voice activity detection (VAD) and decodes them as
  independent items in a batch.

`generate_segments` is closest to the batched pipeline: it decodes independent
chunks in one batch and gives every chunk the same prompt.

## Summary

| Area | `generate_segments` | faster-whisper sequential | faster-whisper batched |
| --- | --- | --- | --- |
| Audio windowing | Fixed 30-second chunks, encoded and decoded as one batch | A moving window that resumes at the last complete timestamp, or at the last word's end, so words aren't cut at window edges | Speech chunks found by Silero VAD (on by default), decoded in batches of 8 |
| Segments returned | One `Segment` per 30-second chunk | Split at timestamp tokens into sentence-sized segments | Split at timestamp tokens, like sequential |
| Prompt | Static `<\|startofprev\|>` + hotwords + initial prompt on every chunk | Rolling previous text (`condition_on_previous_text=True`), seeded by `initial_prompt`; hotwords only when there is no `prefix`; the prompt resets when the decode temperature exceeds 0.5 | Static prompt on every chunk, in the same order as `generate_segments`; rolling text is always off |
| Decoding | One decode per chunk, with `WhisperOptions` as given | Temperature fallback from 0.0 to 1.0 in steps of 0.2 on repetitive or low-confidence output; skips windows that contain no speech | One decode at the first temperature |
| Conditioning safety | By default none: one conditioned decode, used as-is ([ADR 003](adr/003-single-pass-whisper-conditioning-by-default.md)). With `reconcile_conditioning = true`, two decodes merged word by word, so no unconditioned word is lost ([ADR 001](adr/001-word-level-reconciliation-for-whisper-conditioning.md)) | None; the single conditioned decode is used as-is | None |
| Alignment input | By default, plain start-of-transcript tokens; with `reconcile_conditioning = true`, each decode's own prompt, so the conditioned decode includes its conditioning prefix. Either way, every generated token ID, timestamp tokens included | Plain start-of-transcript tokens plus text tokens only (`token < eot`) | Same as sequential |
| Word timing fixes | Punctuation attached to neighboring words ([ADR 002](adr/002-attach-punctuation-to-whisper-words-in-token-grouping.md)) | Punctuation merge, plus trimming overly long words at pauses and sentence ends, and snapping words to segment timestamps | Same as sequential |
| Extras | None | `clip_timestamps`, `prefix`, `max_new_tokens`, `hallucination_silence_threshold`, and per-window language detection | VAD and per-chunk language detection |

The rest of this page explains each row.

## Audio windowing

`generate_segments` splits the samples into fixed 30-second chunks, computes a
mel spectrogram per chunk, encodes all chunks in one call, and decodes them in
one batched call. A word that crosses a 30-second boundary is split between two
chunks.

The sequential pipeline computes features for the whole recording and moves a
30-second window through it. After decoding a window, it moves the window to
the last complete timestamp token. When word timestamps are on and the window
doesn't end on a single timestamp, it moves to the last word's end instead.
Speech at the edge of a window is decoded again in the next window rather than
cut.

The batched pipeline runs Silero VAD first and decodes the detected speech
chunks in batches of 8 by default.

## Segments

`generate_segments` returns one `Segment` per 30-second chunk, with `id`,
`text`, `start`, `end`, and `words`. It always computes word timings.

Both faster-whisper pipelines split each decoded window at consecutive
timestamp tokens, which usually produces one segment per sentence. They skip
segments whose start equals their end or whose text is empty. Each faster-whisper
`Segment` also carries `seek`, `tokens`, `temperature`, `avg_logprob`,
`compression_ratio`, and `no_speech_prob`. Word timings are optional
(`word_timestamps=False` by default).

## Prompt construction

All three use the same prompt layout when conditioning is set:

1. `<|startofprev|>`
2. Hotword tokens
3. Initial prompt or previous-text tokens
4. The start-of-transcript sequence (`<|startoftranscript|>`, the language
   token, and `<|transcribe|>`)

They differ in what fills step 3 and how long it can be:

- **`generate_segments`** treats blank or whitespace-only text as unset, and
  encodes text as `" " + text.trim()`. Hotwords and the initial prompt share one
  budget of `max_length / 2` minus the start-of-transcript length minus 3
  tokens (218 tokens with the default `max_length` of 448). Hotwords keep their
  beginning and the initial prompt keeps its end.
- **faster-whisper sequential** seeds the previous-text history with
  `" " + initial_prompt.strip()`, then appends each window's output. Hotwords
  and previous text are each capped at `max_length // 2 - 1` tokens (223), so
  together they can fill most of the 448-token context. Hotwords are ignored
  when `prefix` is set. The history is cleared when
  `condition_on_previous_text=False` or when a window needed a temperature
  above `prompt_reset_on_temperature` (0.5).
- **faster-whisper batched** encodes `initial_prompt` without adding a leading
  space and uses the same per-field caps as sequential. It never appends
  previous output.

## Decoding and fallback

`generate_segments` decodes each chunk once with the `WhisperOptions` you pass.
It doesn't request scores or the no-speech probability.

The sequential pipeline retries each window at increasing temperatures
(`[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` by default):

- At temperature 0, it uses beam search (`beam_size=5`). Above 0, it samples
  with `best_of=5` hypotheses.
- It retries when the compression ratio of the text is above 2.4 (repetitive
  output) or the average log probability is below -1.0.
- It doesn't retry when the no-speech probability is above 0.6 and the average
  log probability is below -1.0; it treats the window as silence instead.
- If every temperature fails, it keeps the result with the highest average log
  probability, preferring results under the compression-ratio threshold.

Independently, it skips a window when the no-speech probability is above 0.6,
unless the average log probability is above -1.0.

The batched pipeline decodes once at the first temperature and computes the
same scores without retrying.

## Conditioning safety

A conditioned decode can drop a clause or a whole sentence, copy the prompt
into the transcript, or switch language. faster-whisper uses the conditioned
decode as-is in both pipelines.

`generate_segments_conditioned` has two modes, selected by
`WhisperConditioning::reconcile_conditioning`:

- **Single-pass** (`false`, the default) matches faster-whisper: each
  conditioned chunk is decoded once and that transcript is used as-is, so it
  can lose, copy, or translate content. See
  [ADR 003](adr/003-single-pass-whisper-conditioning-by-default.md).
- **Reconciled** (`true`) is described below.

With `reconcile_conditioning = true`, `generate_segments_conditioned` decodes each
conditioned chunk twice, with and without conditioning, and merges the two
transcripts word by word, as described in
[ADR 001](adr/001-word-level-reconciliation-for-whisper-conditioning.md):

- Words only the unconditioned transcript has are always kept.
- Words that differ go to whichever side is locally more confident.
- A chunk keeps the unconditioned transcript when more than 25% of the
  conditioned words look copied from the prompt, or when the merged word
  timings go back by more than 1 second.

This costs about twice the decode and alignment work for conditioned chunks.
It doesn't protect against a hotword that replaces a real word with higher
confidence, or against a short insertion.

`generate_conditioned`, the conditioned `generate`, has no word alignment to
reconcile, so it ignores `reconcile_conditioning` and is always single-pass.

## Word alignment

Both implementations run CTranslate2's dynamic time warping (DTW) alignment on
cross-attention weights with a median filter width of 7. They differ in what
they pass to it:

- **Start sequence:** by default, `generate_segments_conditioned` passes the plain
  start-of-transcript sequence, as faster-whisper always does, so a long
  conditioning prefix never reduces the room left for alignment. With
  `reconcile_conditioning = true`, it aligns each of the two decodes against
  the prompt it was decoded with, so the conditioned decode's start sequence
  includes its conditioning prefix and both sides' word confidences are
  measured in their own context.
- **Text tokens:** `generate_segments` passes every generated token ID,
  including timestamp tokens. faster-whisper passes only text tokens.

After alignment, faster-whisper adjusts word timings in ways
`generate_segments` doesn't:

- It caps the median word duration at 0.7 seconds and limits words at sentence
  boundaries to twice that median.
- It shortens the first words after a pause when they are too long.
- It snaps the first and last words of a segment to the segment's timestamps
  when a word extends more than 0.5 seconds past them.

Both attach punctuation to neighboring words. faster-whisper uses fixed
prepend and append character lists; `generate_segments` classifies decoded
pieces, as described in
[ADR 002](adr/002-attach-punctuation-to-whisper-words-in-token-grouping.md).

## Language detection

When you don't pass a language, `generate_segments` detects it once, from the
first 30-second chunk, and uses it for every chunk.

faster-whisper also detects the language from the first window by default
(`language_detection_segments=1`, `language_detection_threshold=0.5`). With
`multilingual=True`, it detects the language again for every window or chunk.

## faster-whisper defaults

The two faster-whisper pipelines use different defaults for the same
arguments:

| Argument | Sequential | Batched |
| --- | --- | --- |
| `beam_size` | 5 | 5 |
| `temperature` | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` | Same list; only the first value is used |
| `condition_on_previous_text` | `True` | Always `False` |
| `without_timestamps` | `False` | `True` |
| `word_timestamps` | `False` | `False` |
| `vad_filter` | `False` | `True` |
| `compression_ratio_threshold` | 2.4 | Not used for retries |
| `log_prob_threshold` | -1.0 | Not used for retries |
| `no_speech_threshold` | 0.6 | Not used for skipping |
| `hallucination_silence_threshold` | `None` (off) | Not supported |

## Porting faster-whisper behavior to Rust

Everything faster-whisper asks of CTranslate2 is already available in
`ct2rs::sys::Whisper`:

- `encode`, `generate`, `detect_language`, and `align`.
- The `WhisperOptions` fields the sequential pipeline sets: `beam_size`,
  `patience`, `length_penalty`, `repetition_penalty`, `no_repeat_ngram_size`,
  `max_length`, `sampling_topk`, `sampling_temperature`, `num_hypotheses`,
  `return_scores`, `return_no_speech_prob`, `max_initial_timestamp_index`,
  `suppress_blank`, and `suppress_tokens`.
- The `scores` and `no_speech_prob` fields of `WhisperGenerationResult`.

The pieces that aren't available yet:

- **Compression ratio:** the fallback rule needs a zlib implementation, such
  as the `flate2` crate.
- **Whole-recording features:** a moving window needs a mel spectrogram of the
  whole recording that it can slice at any frame. The current spectrogram code
  pads each 30-second piece separately.
- **VAD:** the batched pipeline's Silero model runs on ONNX Runtime, which
  would be a large new dependency.

The main constraint is a trade-off, not feasibility. Rolling previous-text
conditioning makes each window depend on the one before it, so decoding becomes
sequential and loses the batching across chunks that `generate_segments` relies
on. faster-whisper makes the same trade: its batched pipeline turns rolling
conditioning off.

## Related documents

- [ADR 001: Reconcile Whisper conditioning output word by word](adr/001-word-level-reconciliation-for-whisper-conditioning.md)
- [ADR 002: Attach punctuation to Whisper words in token grouping](adr/002-attach-punctuation-to-whisper-words-in-token-grouping.md)
- [ADR 003: Decode conditioned Whisper chunks once by default](adr/003-single-pass-whisper-conditioning-by-default.md)
- [faster-whisper `transcribe.py` at commit `9e472ae`](https://github.com/SYSTRAN/faster-whisper/blob/9e472ae9de8bb374552f531f752654554b76b2d6/faster_whisper/transcribe.py)
