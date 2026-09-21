// whisper_hotwords.rs
//
// Copyright (c) 2023-2024 Junpei Kawamoto
//
// This software is released under the MIT License.
//
// http://opensource.org/licenses/mit-license.php

//! Compare Whisper transcription with and without `initial_prompt`/`hotwords` conditioning.
//!
//! `WhisperOptions::initial_prompt` and `WhisperOptions::hotwords` bias
//! `Whisper::generate_segments` toward known vocabulary, such as proper nouns or
//! domain-specific terms the model would otherwise misspell. This example transcribes the
//! same audio twice, once with default options and once with conditioning applied, and prints
//! both so the difference can be compared directly.
//!
//! See the `whisper` example for how to convert and obtain a model.
//!
//! ```bash
//! cargo run -F whisper --example whisper_hotwords -- \
//!     ./whisper-tiny-ct2 audio.wav --hotwords "Kubernetes, Xiomara Nkemelu"
//! ```

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;
use hound::WavReader;

use ct2rs::{Whisper, WhisperOptions};

#[cfg(not(feature = "whisper"))]
compile_error!("This example requires 'whisper' feature.");

/// Compare transcription with and without `initial_prompt`/`hotwords` conditioning.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the directory that contains model.bin.
    model_dir: PathBuf,
    /// Path to the WAVE file.
    audio_file: PathBuf,
    /// Language to transcribe (e.g. "en"). Detected automatically if omitted.
    #[arg(long)]
    language: Option<String>,
    /// Free-form context text biasing decoding, e.g. a topic description.
    #[arg(long)]
    initial_prompt: Option<String>,
    /// Comma-separated hint words/phrases biasing decoding toward specific spellings.
    #[arg(long)]
    hotwords: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let whisper = Whisper::new(args.model_dir, Default::default())?;
    let samples = read_audio(args.audio_file, whisper.sampling_rate())?;
    let language = args.language.as_deref();

    let baseline = whisper.generate_segments(&samples, language, &Default::default())?;
    let baseline_text: String = baseline.iter().map(|s| s.text.as_str()).collect();

    let conditioned_options = WhisperOptions {
        initial_prompt: args.initial_prompt,
        hotwords: args.hotwords,
        ..Default::default()
    };
    let conditioned = whisper.generate_segments(&samples, language, &conditioned_options)?;
    let conditioned_text: String = conditioned.iter().map(|s| s.text.as_str()).collect();

    println!("--- WITHOUT initial_prompt/hotwords ---\n{baseline_text}\n");
    println!("--- WITH initial_prompt/hotwords -------\n{conditioned_text}");

    Ok(())
}

fn read_audio<T: AsRef<Path>>(path: T, sample_rate: usize) -> Result<Vec<f32>> {
    // Should use a better resampling algorithm.
    fn resample(samples: Vec<f32>, src_rate: usize, target_rate: usize) -> Vec<f32> {
        if src_rate == target_rate {
            return samples;
        }
        if src_rate > target_rate {
            let step = src_rate / target_rate;
            return samples.into_iter().step_by(step).collect();
        }

        let factor = target_rate as f32 / src_rate as f32;
        let new_len = (samples.len() as f32 * factor) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_idx = i as f32 / factor;
            let idx_low = src_idx.floor() as usize;
            let idx_high = (idx_low + 1).min(samples.len() - 1);
            let weight = src_idx - idx_low as f32;
            resampled.push(samples[idx_low] * (1. - weight) + samples[idx_high] * weight);
        }
        resampled
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
