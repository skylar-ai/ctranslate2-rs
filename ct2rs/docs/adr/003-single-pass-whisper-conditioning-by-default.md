# ADR-003: Decode Conditioned Whisper Chunks Once by Default, Reconcile on Opt-In

- **Date**: 2026-10-19
- **Status**: Accepted
- **Deciders**: Lucas Silveira
- **Tags**: whisper, decoding, performance, api

> **2026-10-19:** `initial_prompt`, `hotwords` and `reconcile_conditioning` live on
> `WhisperConditioning`, passed per call to `Whisper::generate_segments_conditioned` and
> `Whisper::generate_conditioned`, and `WhisperOptions` keeps its `0.10.1-skylar-3` fields. Read
> `WhisperOptions::reconcile_conditioning` below as `WhisperConditioning::reconcile_conditioning`,
> read only by `Whisper::generate_segments_conditioned`; `Whisper::generate_conditioned` ignores it
> and decodes each chunk once. The decision below is otherwise unchanged.

## Context and Problem Statement

`WhisperOptions::initial_prompt` and `hotwords` had one implementation, the one
[ADR-001](001-word-level-reconciliation-for-whisper-conditioning.md) decided: every conditioned
chunk is decoded and aligned twice, with and without conditioning, and the two transcripts are
merged word by word. That costs about twice the decode and alignment work, and its cost and
complexity were why conditioning had been switched off, leaving both fields ignored.

Callers coming from faster-whisper expect its batched pipeline's semantics instead: one static
prompt per chunk (`<|startofprev|>`, hotwords, initial prompt, start-of-transcript sequence),
decoded once, with words aligned against the plain start-of-transcript sequence. No consumer sets
either field yet, so the default can still be chosen freely; the field that selects the mode is
public API of a published crate, so its name and default are not.

## Decision Drivers

- A conditioned chunk should cost one decode and one alignment unless the caller asks for more.
- Behaviour should match faster-whisper's batched pipeline, the reference callers port from.
- Callers that cannot accept lost words must still be able to get ADR-001's guarantee.

## Considered Options

- Single-pass by default, with a `bool` field `reconcile_conditioning` that opts into ADR-001's
  reconciliation.
- Reconciliation by default, with a "fast" opt-out.
- An enum mode field instead of a `bool`.

## Decision Outcome

Chosen option: **single-pass by default, `reconcile_conditioning` as the opt-in**.

- `WhisperOptions::reconcile_conditioning: bool`, default `false`, read only by
  `Whisper::generate_segments`. CTranslate2 ignores it, like `initial_prompt` and `hotwords`.
- With it `false` and conditioning set, each chunk is decoded once with the conditioned prompt and
  that transcript is used as-is. Words are aligned against the plain start-of-transcript sequence,
  as faster-whisper's `find_alignment` does.
- With it `true` and conditioning set, behaviour is exactly ADR-001's, including the fallback to
  the unconditioned transcript when more than 25% of conditioned words are copied from the prompt
  or merged word starts go back by more than 1 second.
- With conditioning blank, the field changes nothing.
- Both modes use the same prompt layout and the same shared token budget for hotwords and the
  initial prompt.

Reconciliation by default was rejected because every conditioned call would pay 2x unless the
caller opted out, which is the cost this change exists to remove. An enum was rejected because no
third mode is planned, and a `bool` can become an enum in a later breaking release.

### Positive Consequences

- A conditioned chunk costs one decode and one alignment by default, down from two of each.
- Transcripts match faster-whisper's batched pipeline: faster-whisper's own hotwords check
  (`hotwords = "ComfyUI"` on its `hotwords.mp3`, whisper-tiny) passes here.
- Aligning against the plain start-of-transcript sequence keeps a long conditioning prefix from
  eating into the positions the alignment needs.

### Negative Consequences

- **The default accepts content loss.** A single conditioned decode can omit a clause or a whole
  sentence, copy the prompt into the transcript, or switch language; ADR-001's validation on
  whisper-tiny saw all three. How often this happens on larger models has not been measured.
  Only `reconcile_conditioning = true` guarantees that no word of the unconditioned transcript is
  lost.
- Adding the field breaks callers that build `WhisperOptions` without `..Default::default()`; it
  ships in the same breaking release as `initial_prompt` and `hotwords`.
- `Segment` doesn't say which mode ran or whether a reconciled chunk fell back.

## Links

- [ADR-001: word-level reconciliation for Whisper conditioning](001-word-level-reconciliation-for-whisper-conditioning.md)
- [How faster-whisper differs from `generate_segments`](../faster-whisper-comparison.md)
- faster-whisper `transcribe.py` at commit `9e472ae`:
  https://github.com/SYSTRAN/faster-whisper/blob/9e472ae9de8bb374552f531f752654554b76b2d6/faster_whisper/transcribe.py
