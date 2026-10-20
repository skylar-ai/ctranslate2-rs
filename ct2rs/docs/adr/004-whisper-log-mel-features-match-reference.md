# ADR-004: Compute Whisper Log-Mel Features Like openai-whisper, in One Pass

- **Date**: 2026-10-19
- **Status**: Accepted
- **Deciders**: Antonio Moreira
- **Tags**: whisper, features, performance

## Context and Problem Statement

`Whisper::generate_mel_spectrogram` fed each `hop_length` slice of the samples to `mel_spec`'s
streaming `Spectrogram` and turned every FFT frame into features with `log_mel_spectrogram` and
`norm_mel`. After the upgrade to `mel_spec` 0.5, two problems showed up.

It was slow. In 0.5, `log_mel_spectrogram` builds a `SparseMelFilterbank` from the dense
`mel_filters` matrix on every call, and it was called once per frame, 3000 times per 30 s chunk,
with about six allocations per frame on top. Extracting features for 600 s of audio took
3165 ms (release build, one thread).

Its features were not Whisper's. Compared with openai-whisper's `log_mel_spectrogram` on a 69 s
clip with silent gaps, they differed by 0.19 on average and up to 0.8, on features that range
over about [-1.5, 1.5]:

1. `norm_mel` clamps each frame to its own maximum minus 8. Whisper clamps to the maximum of the
   whole input minus 8, so quiet and silent frames came out unlike anything the model was
   trained on.
2. Frames after the end of the audio were left at `0.0`, which reads as moderate noise. Whisper
   gives them the silence floor (about -0.70 on the test clip). Clips shorter than 30 s and the
   last chunk of longer audio are mostly padding.
3. The streaming STFT is not centred: every frame was 40 samples (2.5 ms) late, and the first two
   frames of the input were left at `0.0`.

## Decision Drivers

- The model must see the features it was trained on.
- Feature extraction should cost little next to encoding and decoding.
- Prefer `mel_spec` over owning DSP code, where it fits.

## Considered Options

- Keep the streaming path and only build the `SparseMelFilterbank` once. Output is unchanged
  bit for bit and as fast as the chosen option, but all three divergences stay.
- Use one of `mel_spec` 0.5's batch helpers. `MelSpectrogram` and
  `Spectrogram::compute_mel_spectrogram_cpu` also normalize per frame and are not centred;
  `BatchLogMelSpectrogram` follows NeMo's recipe instead (natural log, symmetric window, zero
  centre padding).
- Run our own centred STFT with `realfft` (already a dependency of `mel_spec`) and project with
  `mel_spec`'s `SparseMelFilterbank`, built once from `mel_filters`.

## Decision Outcome

Chosen option: **our own centred STFT with `mel_spec`'s `SparseMelFilterbank`**, in
`PreprocessorConfig::log_mel_spectrogram`.

- The samples are zero-padded to whole `n_samples` chunks and reflect-padded by `n_fft / 2` on
  each side, then framed every `hop_length` samples with a periodic Hann window, as
  `torch.stft(center=True, pad_mode="reflect")` does. All `n_fft / 2 + 1` power bins go through
  the filterbank, then `log10(max(x, 1e-10))`.
- Values are clamped to the maximum over **the whole input** minus 8 and mapped to
  `(x + 4) / 4`, as openai-whisper's `transcribe()` does. This is not configurable. Normalizing
  each 30 s chunk on its own, as Hugging Face's `WhisperFeatureExtractor` and faster-whisper's
  batched pipeline do, would only move where the maximum is taken; one fixed behaviour is
  enough, and the whole-input form lets frames at a chunk cut see the audio on the other side.
- `PreprocessorConfig` keeps the filterbank in sparse form and rejects, when the model is loaded,
  a `mel_filters` shape that doesn't match `feature_size` and `n_fft`, and an `nb_max_frames` that
  doesn't cover `n_samples`, instead of panicking during inference.

### Positive Consequences

- Features match an independent implementation of openai-whisper's within 1e-5 (a unit test
  checks this); the scratch benchmark measured 1.2e-7.
- 600 s of audio takes 265 ms instead of 3165 ms, about 12× faster.
- Feature extraction is unit-tested without downloading a model.

### Negative Consequences

- Transcripts and word timings can change for the same audio, compared with `0.10.1-skylar-3`.
  On the test recording (Harvard sentences, `OSR_us_000_0010_8k.wav`, whisper-tiny, CPU `ruy`)
  the plain transcript fixed "Rice is often served" but newly got "pork chuck" and "hogs are fed"
  wrong; the reconciled transcript fixed "often" and "large size in stockings". The two
  golden-transcript tests were re-recorded. One clip on the smallest model does not show whether
  accuracy improves overall; that has not been measured on a test set.
- A chunk's features depend on the loudest part of the whole input, so the same 30 s of audio
  can produce slightly different features inside a longer recording. openai-whisper makes the
  same trade-off.
- `realfft` becomes a direct dependency of the `whisper` feature, and this crate now owns the
  STFT framing.

## Links

- openai-whisper `audio.py` (`log_mel_spectrogram`):
  https://github.com/openai/whisper/blob/main/whisper/audio.py
- mel_spec: https://github.com/wavey-ai/mel-spec
