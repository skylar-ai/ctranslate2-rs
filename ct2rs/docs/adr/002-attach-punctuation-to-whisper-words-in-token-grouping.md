# ADR-002: Attach Punctuation to Neighbouring Words During Whisper Token Grouping

- **Date**: 2026-10-19
- **Status**: Accepted
- **Deciders**: Lucas Silveira
- **Tags**: whisper, word-timestamps, api-output

## Context and Problem Statement

`Whisper::generate_segments` returned `Word`s that were only punctuation (`,`, `.`, `!`), each
with its own timing and probability. Whisper's tokenizer makes punctuation its own token, and the
word splitter started a new word at every token beginning with ASCII punctuation; nothing merged
it back. On the OSR test audio, 10 of 85 words were punctuation-only, and a sentence-final `.`
often stretched into the silence after the sentence. openai-whisper and faster-whisper split the
same way but then run `merge_punctuations`; our port had taken the split and not the merge.

A `Word` is part of the crate's public output, so what one contains is a contract callers bind to.
Changing it also interacts with ADR-001, whose reconciliation matches words by their text.

## Decision Drivers

- `Segment.words` must not contain punctuation-only entries while a real word is there to take them.
- Stay close to openai-whisper's behaviour, the reference everyone compares word output against.
- `Segment.text` on the unconditioned path must not change.

## Considered Options

- **Where to merge**: inside token grouping (before reconciliation), or on the final word list
  after reconciliation.
- **What to merge**: openai-whisper's exact punctuation sets, or any punctuation-only word.

## Decision Outcome

Chosen: **merge inside token grouping, and merge any punctuation-only word**.

- A piece is punctuation-only when its **decoded** text has no alphanumeric character. Raw token
  strings can't be used: byte-level BPE spells `，` as `ï¼Į`, which reads as letters.
- Opening punctuation (`"` `'` `“` `¿` `(` `[` `{` `-`) preceded by a space joins the next word;
  every other punctuation-only piece joins the previous word.
- Merging stays within a run of text tokens between special tokens. Punctuation with no word on
  its usual side joins the other neighbour (`.then`); a run with no word at all is left as split.
- Merged words take their timing and probability from all of their tokens, through the existing
  `process_word_timings`.
- The conditioned path carries each word's leading-space flag through reconciliation, so the
  joined text keeps `said "Hello` while `didn't` and `M-R-C-S` still join without a space.

Merging in grouping matches openai-whisper's placement and gives both decodes of the conditioned
path the same word shape. openai-whisper's exact sets were rejected because `;`, `...`, `…` and `—`
would still come out as standalone words.

### Positive Consequences

- No punctuation-only words on the test audio (10 before, 0 after); `Segment.text` unchanged.
- Words before a pause no longer inherit the pause's duration through a trailing `.`: segment
  end times move earlier (26.44s → 25.90s on segment 0 of the test audio).
- Non-ASCII punctuation is handled the same way as ASCII.

### Negative Consequences

- **Reconciliation (ADR-001) matches fewer words.** `world.` against `world` is now a
  substitution decided by local confidence, taking the winning side's timing, instead of a match
  on baseline timing. Prompts often change punctuation style, so conditioned chunks may fail the
  monotonicity check and fall back to the baseline more often. Comparing words without
  punctuation in `word_edit_script` would restore the matches; it was left out of scope.
- `Segment.words` is shorter, which breaks callers that index into it or count it.
- Edge results like `.then` are unusual; openai-whisper leaves such punctuation standalone.
- Contraction pieces (`didn` + `'t`) are still separate words, since `'t` isn't punctuation-only.

## Links

- [ADR-001: word-level reconciliation for Whisper conditioning](001-word-level-reconciliation-for-whisper-conditioning.md)
- openai-whisper `merge_punctuations`: https://github.com/openai/whisper/blob/main/whisper/timing.py
- Commits `9de8861` (behaviour) and `e9cb7da` (test fixtures)
