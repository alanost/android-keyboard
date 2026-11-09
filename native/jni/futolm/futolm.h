// futolm: the keyboard's transformer language model on a current llama.cpp.
//
// Lives in its own shared library (libfutolm.so) so that the llama.cpp/ggml inside it can be
// newer than the ggml that voice input and on-device training still use in libjni_latinime.so.
// Only the functions below are exported; everything else is hidden.
//
// Two model protocols are supported (selected by keyboardlm.features in the GGUF):
//   * FUTO's original one: external SentencePiece tokenizer whose pieces carry the space as a
//     suffix ("word▁"); a word ends with a "▁" piece or <XEC>.
//   * "xec_word_v1": the model's own tokenizer (any llama.cpp vocab, e.g. Qwen3); the predicted
//     word follows <XBC> as " word" and always ends with <XEC>.
// Both use <XBU> + one mixed embedding per tap (<CHAR_A>..<CHAR_Z>) + <XBC> for corrections.

#ifndef FUTOLM_H
#define FUTOLM_H

#ifdef __cplusplus
extern "C" {
#endif

#define FUTOLM_API __attribute__((visibility("default")))

#define FUTOLM_MIX_KEYS 4
#define FUTOLM_MAX_WORD 96

typedef struct futolm futolm;

// One tap: the keys under the finger with their share of the touch area.
typedef struct futolm_mix {
    float x, y;                         // tap position as a share of keyboard width/height, -1 if unknown
                                        // (used by models with the coordinate encoder)
    float weight[FUTOLM_MIX_KEYS];      // sums to 1 over the letters
    int codepoint[FUTOLM_MIX_KEYS];     // 'a'..'z' or 'A'..'Z'; anything else is ignored
} futolm_mix;

typedef struct futolm_result {
    char word[FUTOLM_MAX_WORD];         // UTF-8, without surrounding spaces
    float probability;
} futolm_result;

enum futolm_mode {
    FUTOLM_MODE_AUTOCORRECT = 0,        // the top result is clearly better than the rest
    FUTOLM_MODE_UNCERTAIN = 1,
    FUTOLM_MODE_CLUELESS = 2,
};

// Loads a model. Returns NULL if the file can't be used (the reason is logged).
FUTOLM_API futolm *futolm_open(const char *path, int n_threads);
FUTOLM_API void futolm_close(futolm *lm);

// Short description for logs: architecture, protocol, vocabulary size.
FUTOLM_API const char *futolm_describe(const futolm *lm);

// Suggestions for the word being typed (partial_word not empty) or the next word (empty).
// mixes: one entry per typed letter, computed by the caller from the tap positions exactly as FUTO
// did (see futolm_taps.h). input_mode 1 = swipe. Writes up to max_results results sorted by
// probability and the confidence mode in *mode, returns the number of results, or -1 when there
// must be no suggestions at all (a word was typed but none of it maps to letters, e.g. a number).
FUTOLM_API int futolm_suggest(futolm *lm,
                              const char *context,
                              const char *partial_word,
                              const futolm_mix *mixes, int n_mixes,
                              int input_mode,
                              float autocorrect_threshold,
                              const char *const *banned_words, int n_banned_words,
                              futolm_result *results, int max_results,
                              int *mode);

// Multiplies dictionary scores (0..1) by how likely the model finds each word after context.
FUTOLM_API void futolm_rescore(futolm *lm, const char *context,
                               const char *const *words, const float *scores, int n_words,
                               float *out_scores);

// Forward passes (llama_decode calls) and tokens computed since opening, for benchmarks.
FUTOLM_API void futolm_counters(const futolm *lm, long *decode_calls, long *decode_tokens);

#ifdef __cplusplus
}
#endif

#endif // FUTOLM_H
