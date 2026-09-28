//! Word-level reconciliation between a baseline transcript and one decoded with
//! `initial_prompt`/`hotwords` conditioning, used by
//! [`conditioning::generate_segments_with_conditioning`][super::conditioning::generate_segments_with_conditioning]
//! so that conditioning can bias wording but never silently drop content.

use super::Word;

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
pub(super) fn merge_word_hypotheses(base: &[Word], cond: &[Word]) -> Vec<Word> {
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
pub(super) fn join_words(words: &[Word]) -> String {
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
mod tests {
    use super::*;

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
}
