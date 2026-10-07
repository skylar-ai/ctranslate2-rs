# ADR-001: Reconcile Whisper Conditioning Output Word-by-Word Against the Baseline

- **Date**: 2026-09-21
- **Status**: Accepted
- **Deciders**: Lucas Guerreiro
- **Tags**: whisper, accuracy, decoding

## Context and Problem Statement

`WhisperConditioning::initial_prompt`/`hotwords` bias `Whisper::generate_segments_conditioned` toward known
vocabulary (proper nouns, domain terms) by prepending a static `<|startofprev|>` token prefix to
the decoder prompt for every chunk, mirroring faster-whisper's `get_prompt`. That prefix biases
the whole chunk's decode trajectory, not just the target word's position: on real audio, it can
make the decoder drop a clause, or occasionally a whole sentence, elsewhere in the same chunk
instead of just correcting the target term. We needed a way to keep the accuracy benefit of
conditioning while guaranteeing it never makes a transcript worse than not using the feature.

## Decision Drivers

- Conditioning must never lose content relative to the unconditioned baseline.
- Must still reliably fix target vocabulary (invented words, foreign proper nouns) when it helps.
- Extra decode cost is only acceptable for chunks that actually request conditioning.

## Considered Options

- Decode once with the conditioned prompt and trust it outright (no reconciliation).
- Decode both the conditioned and baseline prompt, keep whichever scores higher as a whole chunk
  (via `WhisperOptions::return_scores`, length-normalized as in CTranslate2's
  `finalize_hypothesis_score`).
- Decode both, align each independently, and reconcile the two transcripts word by word.

## Decision Outcome

Chosen option: **decode both prompts and reconcile word by word**, because whole-chunk scoring
was tested and rejected: a chunk that fixed ten words and silently dropped one sentence could
still win on average score, so content loss was still possible.

Each chunk is decoded twice (baseline prompt, conditioned prompt), each aligned separately so
per-word confidence is measured fairly for both, then merged via a Levenshtein alignment over
word text (`word_edit_script`/`merge_word_hypotheses`):

- Words both sides agree on: kept.
- Words only the conditioned side added: kept.
- Words only the baseline has (conditioning dropped them): baseline's words are **always** kept,
  unconditionally. There is no confidence score for something that isn't there, so a drop is
  never accepted regardless of how well the rest of the chunk scored.
- Words that differ at the same position: whichever side has the higher local average word
  probability wins.

The unconditional-keep rule on deletions is what makes "never worse than baseline" a guarantee
rather than an observation from testing.

### Positive Consequences

- Content loss is structurally impossible, not just empirically rare.
- Reliably fixes target vocabulary with no other valid reading (invented words, foreign proper
  nouns in fluent native speech).

### Negative Consequences

- Roughly 2x decode and 2x alignment cost per chunk when conditioning is used (nothing extra
  otherwise, since this path isn't taken unless `initial_prompt`/`hotwords` is set).
- Does not fix a wrong word that is equally or more confident than the correct one.
- Can occasionally flicker a single word back to the baseline spelling if its local confidence
  alone doesn't win, even though whole-chunk scoring had picked the conditioned version. Never
  worse than baseline, just not always the best available answer.

## Validation

Tested on `whisper-tiny` (smallest model, most likely to need help and most likely to still get
things wrong) via the `whisper_hotwords` example, across 5 languages on both synthetic speech
(invented name + "Kubernetes") and real public-domain audio (LibriVox), targeting proper nouns
the baseline had already mis-transcribed.

| Test set                                                  | Without conditioning                   | With conditioning                                                              |
| --------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------ |
| Synthetic, invented name + "Kubernetes" (5 languages)     | 0/5 correct                            | 4/5 exact, 1/5 partial (surname only)                                          |
| English, "Jabberwocky" (LibriVox)                         | garbled invented vocabulary throughout | nearly all terms corrected                                                     |
| Spanish/French/German/Portuguese (LibriVox, proper nouns) | mixed garbling                         | most target nouns corrected; previously-dropped clauses restored in every case |

No case in testing lost content relative to the baseline.

## Links

- [`whisper_hotwords` example](../../examples/whisper_hotwords.rs)
- [README: initial_prompt/hotwords conditioning](../../../README.md)
