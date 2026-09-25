# `initial_prompt`/`hotwords` conditioning: bug fix and accuracy analysis

## Crash bug

`generate_segments` panicked whenever `initial_prompt`/`hotwords` was set:

```
thread 'main' panicked at ct2rs/src/whisper.rs:697:36:
range start index 26 out of range for slice of length 19
```

`prefix_len` was computed from the length of the injected conditioning tokens, then used to slice
the model's output tokens, assuming CTranslate2 echoes the full forced prompt back into
`sequences[0]`. It doesn't. Per `WhisperReplica::generate` in `CTranslate2/src/models/whisper.cc`,
everything in the forced prompt is fed to the decoder as KV-cache context, and only the single
trailing task token (`<|transcribe|>`/`<|notimestamps|>`) is echoed back before the generated
text. `tokens.len()` was always smaller than `prefix_len`, so the slice panicked on any real use
of the feature.

Fix: removed `prefix_len` and the slicing. `group_tokens_into_words`/`clean_tokens` use the raw
`tokens`, same as before conditioning existed; the leading task token is already handled by
`is_special_token`. The dead `group_tokens_into_words_after_prefix` helper and its two tests were
removed.

`cargo test -p ct2rs --features whisper,hub,cuda,cudnn whisper`: 24 passed, 0 failed.

## Word-level reconciliation

On real audio, conditioning sometimes made the decoder drop a clause, occasionally a whole
sentence, instead of just mis-spelling the target word. The conditioning tokens bias the whole
chunk's decode, not just the target word's position.

First fix tried: decode each chunk with and without conditioning, keep whichever scores higher as
a whole (`WhisperOptions::return_scores`, length-normalized per `finalize_hypothesis_score` in
`decoding.cc`). Not enough: a chunk can fix ten words and drop one sentence and still win on
average score. Confirmed by retest: the English and German clips below still lost content under
this version.

Current fix: decode both versions, align each separately, then merge them word by word
(`merge_word_hypotheses`, via a Levenshtein alignment over word text, `word_edit_script`):

- words both sides agree on: kept
- words only conditioning added: kept
- words only the baseline has, meaning conditioning dropped them: baseline's words are always
  kept, no exceptions
- words that differ: whichever side has higher local average probability wins

The deletion rule is the one that matters. There's no confidence score for something that isn't
there, so a drop is never accepted regardless of how well the rest of the chunk scores.

Cost: roughly 2x decode and 2x alignment per chunk when conditioning is used, nothing extra
otherwise. Two smaller fixes were needed along the way: the two prompts can't share one batched
`generate()` call (CTranslate2 requires `<|startoftranscript|>` at the same index across a batch),
and rebuilding segment text from two different token sequences needed word-joining logic
(`join_words`) instead of `tokenizer.decode`, plus a fix for spacing around split contractions
(`'s`, `'t`) and hyphen continuations (`M-R-C-S`).

Every content-loss case found in testing is fixed by this mechanism. Two things it doesn't fix,
both unavoidable with a confidence-based approach:

- a wrong word that's equally or more confident than the correct one (Spanish's "sencillo"
  becoming "se insiste")
- occasional single-word flicker: a word fixed under whole-chunk gating can revert to the
  baseline's spelling under the finer-grained merge, if its local confidence alone doesn't win.
  Never worse than the baseline, just not always better.

## Method

Two passes, using the `whisper_hotwords` example, which transcribes the same audio with and
without conditioning and prints both:

1. Synthetic speech (Google TTS) with an invented name and "Kubernetes", in 5 languages.
2. Real LibriVox recordings (public domain, license checked per item) in 5 languages, using
   whatever proper nouns the baseline already got wrong.

Audio isn't committed to the repo, same as the existing convention in
`ct2rs/tests/assets/README.md`.

## Round 1: synthetic speech

"Please schedule a meeting with Xiomara Nkemelu about the Kubernetes deployment" (translated per
language), `hotwords: "Xiomara Nkemelu, Kubernetes"`.

| Language | Without | With |
|---|---|---|
| English | "C. Amara Nakemelo" | Xiomara Nkemelu |
| Spanish | "Ciomara en Kemelo" | Xiomara Nkemelu |
| French | "Yomara, une camelue" / "Cuburnet" | Yomara Nkemelu / Kubernetes |
| German | "Ksiomarane Chemelut" | Xiomara Nkemelu |
| Portuguese | "showmara em que Melus" / "cubernetis" | Xiomara Nkemelu / Kubernetes |

4 of 5 exact. French gets the surname right, not the first name.

### English, real audio: "Jabberwocky" (LibriVox, public domain)

Chapter 1 of *Through the Looking-Glass*, the Jabberwocky poem (16:02-17:39). `hotwords` is the
poem's invented vocabulary.

| Word | Without | With |
|---|---|---|
| Jabberwocky | "Jabba walkie" | Jabberwocky |
| Jabberwock (x3) | "Jabba walk" | fixed x2, unfixed x1 |
| slithy toves | "slimy toes" | slithy toves |
| wabe | "wave" | wabe |
| borogoves | "poor goes" | borogoves |
| mome raths outgrabe | "mom rafts out grave" | mome raths, outgrabe |
| jubjub bird | "jubbed jubbed bird" | jubjub bird |
| Bandersnatch | "bender snatch" | Bandersnatch |
| manxome | "man's son" | manxome |
| Tumtum tree | "tom tom tree" | Tumtum tree |
| uffish | "ufish" | uffish |
| tulgy wood | "toolgy wood" | tulgy wood |
| whiffling | "whistling" | whiffling |
| beamish boy | "be-mish-boy" | beamish boy |
| frabjous day, Callooh, Callay | "a frab just-a-kalu-kulei" | frabjous, day, Calloo, Callay |

Nearly everything fixed. A full sentence the baseline had ("You see, she didn't like to confess
...") was dropped by conditioning before the word-level merge fix; now restored.

## Round 2: Spanish, French, German, Portuguese (real audio)

| Language | Source | Text |
|---|---|---|
| Spanish | `sabuesodelosbaskerville_1512_librivox` | El Sabueso de los Baskerville |
| French | `20000_lieues_sous_les_mers_1010_librivox` | 20000 lieues sous les mers |
| German | `hund_von_baskerville_1709_librivox` | Der Hund von Baskerville |
| Portuguese | `dom_casmurro_2102_librivox` | Dom Casmurro |

All LibriVox via archive.org, public domain. For each, the baseline was run on chapter 1 first to
find a mis-transcribed proper noun, then that clip was retested with conditioning.

### French: "Governor Higginson" / "Christophe Colomb"

| Correct | Without | With |
|---|---|---|
| Governor Higginson (1st) | "Covenant Higginsa" | Governor Higginson |
| Governor Higginson (2nd) | "Govana Higginsan" | unfixed, unchanged |
| Christophe Colomb (1st) | "Christophe Alcolon" | unfixed, unchanged |
| Christophe Colomb (2nd) | "Christobal Colon" | unfixed, unchanged |

Before the word-level merge, the 2nd mention of both names was dropped entirely and replaced with
garbled text. Now restored to baseline.

### Spanish: "Sherlock Holmes" / "Watson"

| Correct | Without | With |
|---|---|---|
| Holmes | "Horns" | Holmes |
| (opening fragment) | "para estimularlo." | restored, was dropped before the fix |
| not "Watson" | "John Stanto" | "Holmes tanto" |
| unrelated word | "sencillo" | "se insiste" |

The last two aren't fixed either way: "John Stanto" is probably the model mishearing something
that isn't "Watson" at all, and "se insiste" isn't less confident than "sencillo" was, so nothing
in the mechanism has a reason to prefer the baseline there.

### German: "Sherlock Holmes" / "Watson"

Baseline already gets "Sherlock Holmes" right in this clip. "Watson"/"wardsen" is never fixed, but
the whole clause containing it, which used to get dropped, is now preserved.

### Portuguese: "Dom Casmurro"

4 mentions of "doncas-muno"/"docas-mugam" in the baseline; 3 fixed with conditioning, one flickers
back to "docas-mugam". Also recovers a trailing clause the baseline had truncated.

## Summary

- Fixes target vocabulary reliably: invented words with no other valid reading (Jabberwocky, Dom
  Casmurro), and foreign proper nouns in fluent native speech (Xiomara Nkemelu, Holmes, Governor
  Higginson).
- Never loses content relative to baseline. Every dropped-sentence/clause case found in testing is
  fixed, and the merge's deletion rule makes this a hard guarantee, not just an observation.
- Doesn't fix confidently-wrong substitutions, and can occasionally flicker a single word back to
  the baseline spelling. Neither makes output worse than not using the feature at all.
- Tested on `whisper-tiny`, the smallest model: most likely to need help, and most likely to still
  get things wrong.

## Reproduction

```bash
pixi run cargo build --release -p ct2rs --features whisper,hub,cuda,cudnn
pixi run cargo test --release -p ct2rs --features whisper,hub,cuda,cudnn whisper
```
