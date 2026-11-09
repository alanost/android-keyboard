// futolm.cpp: FUTO's keyboard language model logic (originally in
// org_futo_inputmethod_latin_xlm_LanguageModel.cpp and src/ggml/LanguageModel.cpp) ported to a
// current llama.cpp. The prediction behaviour for existing FUTO models is unchanged; see futolm.h
// for the second protocol ("xec_word_v1") used by models with their own tokenizer such as Qwen3.

#include "futolm.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "sentencepiece/sentencepiece_processor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__aarch64__) && defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#ifdef __linux__
#include <sched.h>
#endif

#ifdef __ANDROID__
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "futolm", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "futolm", __VA_ARGS__)
#else
#define LOGI(...) do { if (getenv("FUTOLM_VERBOSE")) { fprintf(stderr, "futolm: " __VA_ARGS__); fputc('\n', stderr); } } while (0)
#define LOGE(...) do { fprintf(stderr, "futolm error: " __VA_ARGS__); fputc('\n', stderr); } while (0)
#endif

namespace {

constexpr int NUM_RESULTS = 3;
constexpr int CONTEXT_SIZE = 512;        // FUTO cuts the context to about 70 characters, this is plenty
constexpr int MAX_PROMPT = CONTEXT_SIZE - 64;
// Tokens per forward pass. Keystrokes compute at most 3 at once and prompts are short, while
// llama.cpp sizes its work buffer for this many tokens times the vocabulary (Qwen: 152k).
constexpr int MICRO_BATCH = 64;
constexpr float EPS = 0.0001f;

constexpr const char *FEATURE_AUTOCORRECT = "xbu_char_autocorrect_v1";
constexpr const char *FEATURE_SWIPE_TYPING = "xc0_swipe_typing_v1";
constexpr const char *FEATURE_EMBED_MIXING = "char_embed_mixing_v1";
constexpr const char *FEATURE_ENCODER = "experiment_linear_208_209_210";
constexpr const char *FEATURE_XEC_WORD = "xec_word_v1";
constexpr int ENCODER_W_X_ID = 208, ENCODER_W_Y_ID = 209, ENCODER_B_ID = 210;

typedef std::vector<llama_token> token_sequence;

// ---------------------------------------------------------------- small helpers (as in FUTO)

std::string trim(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char) s[a])) a++;
    while (b > a && std::isspace((unsigned char) s[b - 1])) b--;
    return s.substr(a, b - a);
}

template<typename T>
bool sortProbabilityPairDescending(const std::pair<float, T> &a, const std::pair<float, T> &b) {
    return a.first > b.first;
}

template<typename T>
void sortProbabilityPairVectorDescending(std::vector<std::pair<float, T>> &vec) {
    std::sort(vec.begin(), vec.end(), sortProbabilityPairDescending<T>);
}

template<typename T>
void sortProbabilityPairVectorDescending(std::vector<std::pair<float, T>> &vec, size_t partial) {
    if (partial > vec.size()) partial = vec.size();
    std::partial_sort(vec.begin(), vec.begin() + partial, vec.end(), sortProbabilityPairDescending<T>);
}

struct potential_sequence_data {
    token_sequence tokens;
    llama_seq_id seq_id{};
};
typedef std::pair<float, potential_sequence_data> potential_sequence;   // P = P(t0) * P(t1) * ...

struct banned_sequence {
    token_sequence sequence;
    int hash;
};

int compute_sequence_hash(const token_sequence &seq) {
    int hash = 0;
    for (llama_token t : seq) hash = (hash + t) % 999999999;
    return hash;
}

int append_sequence_hash(int hash, llama_token t) { return (hash + t) % 999999999; }

void softmax(float *input, size_t input_len) {
    float m = -INFINITY;
    for (size_t i = 0; i < input_len; i++) m = std::max(m, input[i]);
    float sum = 0.0f;
    for (size_t i = 0; i < input_len; i++) sum += expf(input[i] - m);
    float offset = m + logf(sum);
    for (size_t i = 0; i < input_len; i++) input[i] = expf(input[i] - offset);
}

struct TokenMix {
    float x, y;
    struct {
        float weight;
        llama_token token;
    } mixes[FUTOLM_MIX_KEYS];
};

struct DecodeResult {
    int logits_head;
    int size;
    // FUTO allowed <XEC> as the end of a word whenever the logits came from the first token of a
    // batch: always for corrections, and for next-word predictions after a single new token
    bool allow_correction_token;
};

enum WordCapitalizeMode { IgnoredCapitals, FirstCapital, AllCapitals };

bool isLowercase(unsigned char c, bool strict) { return strict ? islower(c) : isupper(c) == 0; }

bool isFirstCharLowercase(const char *str, bool strict) {
    if (str == nullptr || str[0] == '\0') return false;
    return isLowercase(static_cast<unsigned char>(str[0]), strict);
}

bool hasLowercase(const char *str, bool strict) {
    if (str == nullptr) return false;
    for (; *str != '\0'; ++str) if (isLowercase(static_cast<unsigned char>(*str), strict)) return true;
    return false;
}

bool isExactMatch(const std::string &a, const std::string &b) {
    auto preprocess = [](const std::string &str) {
        std::string result;
        for (char c : str) if (c != '\'' && c != '-' && c != ' ') result += (char) tolower(c);
        return result;
    };
    return preprocess(a) == preprocess(b);
}

bool isTokenMixRoughlyEqual(const TokenMix &a, const TokenMix &b) {
    for (int j = 0; j < FUTOLM_MIX_KEYS; j++) {
        if (a.mixes[j].token != b.mixes[j].token || std::abs(a.mixes[j].weight - b.mixes[j].weight) >= EPS) return false;
    }
    return true;
}

// transformer_context_fastforward from src/ggml/context.cpp: how much of the cached sequence can stay.
std::pair<token_sequence, int> fastforward(const token_sequence &active, const token_sequence &next, bool allow_empty) {
    int npast = 0;
    int max_length = (int) std::min(active.size(), next.size());
    for (int i = 0; i < max_length; i++) {
        if (active[i] != next[i]) break;
        npast = i + 1;
    }
    // A shorter or equal input needs its last token recomputed to get up-to-date logits
    if (!allow_empty && npast == (int) next.size() && next.size() <= active.size()) npast -= 1;
    if (npast < 0) npast = 0;
    return {token_sequence(next.begin() + npast, next.end()), npast};
}

// ---------------------------------------------------------------- model file access

struct Metadata {
    bool ok = false;
    std::unordered_set<std::string> features;
    std::string ext_tokenizer_type;
    std::string ext_tokenizer_data;
    bool has(const char *f) const { return features.count(f) > 0; }
};

Metadata read_metadata(const std::string &path) {
    Metadata m;
    gguf_init_params params = {/*no_alloc*/ true, /*ctx*/ nullptr};
    gguf_context *g = gguf_init_from_file(path.c_str(), params);
    if (!g) return m;
    auto get_str = [&](const char *key) -> std::string {
        int64_t k = gguf_find_key(g, key);
        if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_STRING) return "";
        return gguf_get_val_str(g, k);
    };
    std::istringstream fs(get_str("keyboardlm.features"));
    std::string f;
    while (fs >> f) m.features.insert(f);
    m.ext_tokenizer_type = get_str("keyboardlm.ext_tokenizer_type");
    int64_t k = gguf_find_key(g, "keyboardlm.ext_tokenizer_data");
    if (k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_ARRAY) {
        m.ext_tokenizer_data.assign((const char *) gguf_get_arr_data(g, k), gguf_get_arr_n(g, k));
    }
    gguf_free(g);
    m.ok = true;
    return m;
}

// Input embeddings of a few tokens, read straight from the file and converted to float. FUTO used
// to convert the whole table (n_vocab x n_embd floats); only the key tokens are ever mixed.
bool read_embedding_rows(const std::string &path, int n_embd, const std::vector<llama_token> &tokens,
                         std::unordered_map<llama_token, std::vector<float>> &out) {
    gguf_init_params params = {/*no_alloc*/ true, /*ctx*/ nullptr};
    gguf_context *g = gguf_init_from_file(path.c_str(), params);
    if (!g) return false;
    int64_t t = gguf_find_tensor(g, "token_embd.weight");
    if (t < 0) {
        gguf_free(g);
        LOGE("no token_embd.weight in %s", path.c_str());
        return false;
    }
    const ggml_type type = gguf_get_tensor_type(g, t);
    const size_t row_size = ggml_row_size(type, n_embd);
    const size_t base = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, t);
    const size_t total = gguf_get_tensor_size(g, t);
    gguf_free(g);

    const ggml_type_traits *traits = ggml_get_type_traits(type);
    if (type != GGML_TYPE_F32 && traits->to_float == nullptr) {
        LOGE("can't convert embeddings of type %s", ggml_type_name(type));
        return false;
    }
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    std::vector<uint8_t> buf(row_size);
    bool ok = true;
    for (llama_token tok : tokens) {
        if (tok < 0 || (size_t) (tok + 1) * row_size > total) continue;
        if (pread(fd, buf.data(), row_size, (off_t) (base + (size_t) tok * row_size)) != (ssize_t) row_size) {
            ok = false;
            break;
        }
        std::vector<float> row(n_embd);
        if (type == GGML_TYPE_F32) memcpy(row.data(), buf.data(), n_embd * sizeof(float));
        else traits->to_float(buf.data(), row.data(), n_embd);
        out[tok] = std::move(row);
    }
    close(fd);
    return ok;
}

// ---------------------------------------------------------------- tokenizer

// FUTO models bring their own SentencePiece model (suffix spaces, "word▁"); others use llama.cpp's.
struct Tokenizer {
    bool spm_mode = false;
    sentencepiece::SentencePieceProcessor spm;
    const llama_vocab *vocab = nullptr;
    std::unordered_map<std::string, llama_token> special_text;

    token_sequence tokenize(const std::string &text) const {
        if (spm_mode) {
            std::vector<int> ids = spm.EncodeAsIds(text);
            return token_sequence(ids.begin(), ids.end());
        }
        int n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, false, false);
        token_sequence out(std::max(n, 0));
        if (n > 0) llama_tokenize(vocab, text.data(), (int32_t) text.size(), out.data(), n, false, false);
        return out;
    }

    // Id of one piece: SentencePiece piece, or for llama vocabularies a special token's text or a
    // string that is a single token. Returns -1 (llama) / unk (SentencePiece) if there is none.
    llama_token piece_to_id(const std::string &piece) const {
        if (spm_mode) return spm.PieceToId(piece);
        auto it = special_text.find(piece);
        if (it != special_text.end()) return it->second;
        token_sequence t = tokenize(piece);
        return t.size() == 1 ? t[0] : -1;
    }

    std::string decode(const token_sequence &tokens) const {
        if (spm_mode) return spm.DecodeIds(std::vector<int>(tokens.begin(), tokens.end()));
        std::string out;
        char buf[256];
        for (llama_token t : tokens) {
            int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
            if (n > 0) out.append(buf, n);
        }
        return out;
    }

    std::string piece(llama_token id) const {
        if (spm_mode) return id < spm.GetPieceSize() ? spm.IdToPiece(id) : std::string();
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        return n > 0 ? std::string(buf, n) : std::string();
    }
};

// ggml is built for the instructions named in GGML_CPU_ARM_ARCH; refuse CPUs without them instead
// of crashing on the first illegal instruction.
bool cpu_supported() {
#if defined(__aarch64__) && defined(__linux__)
    const unsigned long hwcap = getauxval(AT_HWCAP);
#ifdef FUTOLM_NEEDS_DOTPROD
    if (!(hwcap & HWCAP_ASIMDDP)) return false;
#endif
#ifdef FUTOLM_NEEDS_FP16
    if (!(hwcap & HWCAP_ASIMDHP)) return false;
#endif
    (void) hwcap;
#endif
    return true;
}

#ifdef __linux__
// The fastest cores (cpu_capacity, else maximum frequency): the n_threads best ones and every core
// that is as fast as the slowest of those. On big.LITTLE phones the little cores take 3-5 times
// longer, and the scheduler doesn't move a short burst of work to the big ones quickly enough.
bool fastest_cpus(int n_threads, cpu_set_t &out) {
    std::vector<std::pair<long, int>> speed;
    for (int cpu = 0; cpu < CPU_SETSIZE && cpu < 64; cpu++) {
        char path[96];
        long value = -1;
        for (const char *file : {"cpu_capacity", "cpufreq/cpuinfo_max_freq"}) {
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/%s", cpu, file);
            if (FILE *f = fopen(path, "r")) {
                if (fscanf(f, "%ld", &value) != 1) value = -1;
                fclose(f);
            }
            if (value > 0) break;
        }
        if (value > 0) speed.emplace_back(value, cpu);
    }
    if ((int) speed.size() <= n_threads) return false;   // nothing to choose
    std::sort(speed.begin(), speed.end(), [](const std::pair<long, int> &a, const std::pair<long, int> &b) { return a.first > b.first; });
    const long threshold = speed[n_threads - 1].first;
    if (speed.back().first >= threshold) return false;   // all cores are alike
    CPU_ZERO(&out);
    for (const auto &s : speed) if (s.first >= threshold) CPU_SET(s.second, &out);
    return true;
}
#endif

void log_callback(ggml_log_level level, const char *text, void *) {
    static const bool verbose = getenv("FUTOLM_VERBOSE") != nullptr;
    if (level == GGML_LOG_LEVEL_ERROR) LOGE("llama: %s", text);
    else if (level == GGML_LOG_LEVEL_WARN || verbose) LOGI("llama: %s", text);
}

} // namespace

// ---------------------------------------------------------------- the engine

struct futolm {
    std::mutex lock;
    std::string path;
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    const llama_vocab *vocab = nullptr;
    llama_batch batch{};
    int n_vocab = 0, n_embd = 0, n_batch = 0;
    bool xec = false;                      // xec_word_v1 protocol
    Metadata meta;
    Tokenizer tok;
    std::string description;
    long decode_calls = 0, decode_tokens = 0;
#ifdef __linux__
    bool pin = false;
    cpu_set_t fast_cpus{};
#endif

    std::unordered_map<llama_token, std::vector<float>> embd_rows;
    std::vector<float> encoder_weight, encoder_bias;
    std::vector<float> xbc_embedding;      // to compute <XBC> in the same pass as the taps

    token_sequence active_context;         // what sequence 0 in the model memory holds
    std::vector<TokenMix> past_mixes;

    struct {
        llama_token SPACE = 0;             // FUTO: "▁"; xec protocol: <XEC>, where a word ends
        llama_token XBU = -1, XBC = -1, XEC = -1, XC0_SWIPE_MODE = 0;
        llama_token DASH = 0, STAR = 0;
        llama_token LETTERS_TO_IDS[26] = {0};
        std::vector<llama_token> banned_start_of_word_tokens;
        std::vector<llama_token> banned_tokens_for_first_capital;
        std::vector<llama_token> banned_tokens_for_all_capitals;
        std::vector<llama_token> banned_tokens_word_separators;   // their probability goes to SPACE
        std::vector<llama_token> general_banned_tokens;
        // xec protocol only
        std::vector<llama_token> starts_with_space;     // begin a new word
        std::vector<llama_token> not_starting_with_space;
    } special;

    ~futolm() {
        if (batch.token || batch.embd) llama_batch_free(batch);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    int decode(const llama_batch &b) {
        decode_calls++;
        decode_tokens += b.n_tokens;
        return llama_decode(ctx, b);
    }

    bool init_special_tokens();
    bool transform_logits(float *logits, bool is_first_token, bool allow_correction_token, WordCapitalizeMode capitals, llama_token prev_token);
    int cached_mix_amount(const std::vector<TokenMix> &mixes);
    DecodeResult decode_prompt_and_mixes(const token_sequence &prompt, const std::vector<TokenMix> &mixes);
    bool matches_banned(const token_sequence &prior, int prior_hash, llama_token next, const std::vector<banned_sequence> &banned) const;
    bool is_word_end(llama_token t) const;
    std::vector<std::pair<float, token_sequence>> sample(DecodeResult r, int n_results, WordCapitalizeMode capitals, const std::vector<banned_sequence> &banned);
    std::vector<banned_sequence> banned_sequences(const std::vector<std::string> &words) const;
    token_sequence context_prompt(const std::string &context, bool for_correction) const;
    std::string word_text(const token_sequence &tokens) const;
    std::vector<std::pair<float, std::string>> predict_next_word(const std::string &context, const std::vector<std::string> &banned);
    std::vector<std::pair<float, std::string>> predict_correction(const std::string &context, const std::vector<TokenMix> &mixes, bool swipe, WordCapitalizeMode capitals, const std::vector<std::string> &banned);
};

bool futolm::init_special_tokens() {
    const int n_pieces = xec ? n_vocab : std::min(n_vocab, tok.spm.GetPieceSize());
    std::vector<std::string> pieces(n_vocab);
    for (int i = 0; i < n_pieces; i++) pieces[i] = tok.piece(i);

    if (meta.has(FEATURE_AUTOCORRECT) || xec) {
        special.XBU = tok.piece_to_id("<XBU>");
        special.XBC = tok.piece_to_id("<XBC>");
        special.XEC = tok.piece_to_id("<XEC>");
        bool letters_ok = true;
        for (int i = 0; i < 26; i++) {
            std::string name = std::string("<CHAR_") + (char) ('A' + i) + ">";
            // FUTO models have consecutive letter ids; look each one up anyway
            special.LETTERS_TO_IDS[i] = xec ? tok.piece_to_id(name) : tok.piece_to_id("<CHAR_A>") + i;
            if (special.LETTERS_TO_IDS[i] <= 0) letters_ok = false;
        }
        if (special.XBU <= 0 || special.XBC <= 0 || special.XEC <= 0 || !letters_ok) {
            LOGE("model lacks the key tokens <XBU> <XBC> <XEC> <CHAR_A>..<CHAR_Z>");
            return false;
        }
        if (meta.has(FEATURE_SWIPE_TYPING)) special.XC0_SWIPE_MODE = tok.piece_to_id("<XC0>");
    }
    special.DASH = tok.piece_to_id("-");
    special.STAR = tok.piece_to_id("*");

    const char *blacklist_symbols = ".!@#$%^&*()_=?/,\\][{};:\"><+`~|\r\n\t\x0b\x0c";
    auto has_symbol = [&](const std::string &s) {
        for (char c : s) if (strchr(blacklist_symbols, c) != nullptr) return true;
        return false;
    };

    if (!xec) {
        special.SPACE = tok.piece_to_id("▁");
        special.general_banned_tokens = {tok.piece_to_id("-▁")};
        for (int i = 0; i < n_pieces; i++) {
            if (has_symbol(pieces[i])) special.banned_tokens_word_separators.push_back(i);
        }
        for (int i = 0; i < n_vocab; i++) {
            const char *text = pieces[i].c_str();
            if (isFirstCharLowercase(text, true)) {
                special.banned_tokens_for_first_capital.push_back(i);
                special.banned_tokens_for_all_capitals.push_back(i);
            } else if (hasLowercase(text, true)) {
                special.banned_tokens_for_all_capitals.push_back(i);
            }
            if (text[0] == '\'' || text[0] == '-') special.banned_start_of_word_tokens.push_back(i);
        }
        return true;
    }

    // xec protocol: a word is " w" + "ord" ... + <XEC>; the first piece starts with the space
    special.SPACE = special.XEC;
    std::unordered_set<llama_token> ours = {special.XBU, special.XBC};
    if (special.XC0_SWIPE_MODE > 0) ours.insert(special.XC0_SWIPE_MODE);
    for (llama_token t : special.LETTERS_TO_IDS) ours.insert(t);
    for (int i = 0; i < n_vocab; i++) {
        if (i == special.XEC) continue;
        const std::string &p = pieces[i];
        const llama_token_attr attr = llama_vocab_get_attr(vocab, i);
        if (ours.count(i) || p.empty() || (attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNUSED | LLAMA_TOKEN_ATTR_UNKNOWN))) {
            special.general_banned_tokens.push_back(i);
            continue;
        }
        const bool space_first = p[0] == ' ';
        const std::string body = space_first ? p.substr(1) : p;
        bool odd_space = body.empty() || body.find_first_of(" \t\r\n") != std::string::npos;
        if (has_symbol(body) || odd_space) {
            special.banned_tokens_word_separators.push_back(i);
            continue;
        }
        (space_first ? special.starts_with_space : special.not_starting_with_space).push_back(i);
        if (space_first) {
            if (isFirstCharLowercase(body.c_str(), true)) {
                special.banned_tokens_for_first_capital.push_back(i);
                special.banned_tokens_for_all_capitals.push_back(i);
            } else if (hasLowercase(body.c_str(), true)) {
                special.banned_tokens_for_all_capitals.push_back(i);
            }
            if (body[0] == '\'' || body[0] == '-') special.banned_start_of_word_tokens.push_back(i);
        } else if (hasLowercase(body.c_str(), true)) {
            special.banned_tokens_for_all_capitals.push_back(i);
        }
    }
    return true;
}

bool futolm::transform_logits(float *logits, bool is_first_token, bool allow_correction_token, WordCapitalizeMode capitals, llama_token prev_token) {
    for (int i = 0; i < n_vocab; i++) if (std::isnan(logits[i])) return false;
    softmax(logits, n_vocab);

    for (llama_token x : special.banned_tokens_word_separators) {
        if (allow_correction_token && x == special.XEC) continue;
        logits[special.SPACE] += std::max(0.0f, logits[x]);
        logits[x] = -999.0f;
    }

    if (xec) {
        if (is_first_token) {
            for (llama_token i : special.not_starting_with_space) logits[i] = -999.0f;
        } else {
            // A piece starting with a space would begin the next word: this word has ended
            for (llama_token i : special.starts_with_space) {
                logits[special.SPACE] += std::max(0.0f, logits[i]);
                logits[i] = -999.0f;
            }
        }
    }

    if (is_first_token) {
        logits[special.SPACE] = -999.0f;
        for (llama_token i : special.banned_start_of_word_tokens) logits[i] = -999.0f;
    }
    for (llama_token i : special.general_banned_tokens) if (i >= 0) logits[i] = -999.0f;
    if (prev_token == special.DASH && special.DASH >= 0) logits[special.DASH] = -999.0f;

    if (capitals == FirstCapital && is_first_token) {
        for (llama_token i : special.banned_tokens_for_first_capital) logits[i] = -999.0f;
    } else if (capitals == AllCapitals) {
        for (llama_token i : special.banned_tokens_for_all_capitals) logits[i] = -999.0f;
    }
    return true;
}

// Runs the model's work on the fastest cores while in scope; llama.cpp's worker threads are created
// per forward pass and inherit the mask. The caller's own affinity is restored afterwards.
struct FastCores {
#ifdef __linux__
    cpu_set_t old{};
    bool restore = false;
    explicit FastCores(const futolm *lm) {
        if (lm->pin && sched_getaffinity(0, sizeof(old), &old) == 0) {
            restore = sched_setaffinity(0, sizeof(cpu_set_t), &lm->fast_cpus) == 0;
        }
    }
    ~FastCores() {
        if (restore) sched_setaffinity(0, sizeof(old), &old);
    }
#else
    explicit FastCores(const futolm *) {}
#endif
};

int futolm::cached_mix_amount(const std::vector<TokenMix> &mixes) {
    size_t i;
    for (i = 0; i < std::min(past_mixes.size(), mixes.size()); i++) {
        if (!isTokenMixRoughlyEqual(past_mixes[i], mixes[i])) break;
    }
    return (int) i;
}

DecodeResult futolm::decode_prompt_and_mixes(const token_sequence &prompt, const std::vector<TokenMix> &mixes) {
    llama_memory_t mem = llama_get_memory(ctx);
    if ((int) (prompt.size() + mixes.size()) + 16 > CONTEXT_SIZE) {   // room for <XBC> and the beams
        LOGE("input of %zu tokens and %zu taps is too long", prompt.size(), mixes.size());
        return {-1, 0, false};
    }
    auto ff = fastforward(active_context, prompt, !mixes.empty());

    int head = -1;
    if (!ff.first.empty()) {
        llama_memory_seq_rm(mem, 0, (llama_pos) ff.second, -1);
        for (size_t b = 0; b < (ff.first.size() + n_batch - 1) / n_batch; b++) {
            batch.n_tokens = std::min(n_batch, (int) (ff.first.size() - b * n_batch));
            for (int i = 0; i < batch.n_tokens; i++) {
                batch.token[i] = ff.first[n_batch * b + i];
                batch.pos[i] = (llama_pos) (ff.second + n_batch * b + i);
                batch.seq_id[i][0] = 0;
                batch.n_seq_id[i] = 1;
                batch.logits[i] = false;
            }
            batch.logits[batch.n_tokens - 1] = (int8_t) mixes.empty();
            if (mixes.empty()) head = batch.n_tokens - 1;
            if (decode(batch) != 0) {
                LOGE("llama_decode() failed");
                active_context.clear();
                past_mixes.clear();
                llama_memory_seq_rm(mem, -1, -1, -1);
                return {-1, 0, false};
            }
        }
    }
    active_context.resize(ff.second);
    active_context.insert(active_context.end(), ff.first.begin(), ff.first.end());

    size_t size = prompt.size();
    std::vector<float> embeds;
    const bool use_encoder = !encoder_weight.empty();
    for (const auto &mix : mixes) {
        int num_added = 0;
        std::vector<float> mix_f(n_embd, 0.0f);
        if (use_encoder && mix.x >= 0.0f && mix.y >= 0.0f) {
            num_added = 1;
            for (int i = 0; i < n_embd; i++) {
                mix_f[i] = encoder_bias[i] + encoder_weight[i * 2] * mix.x + encoder_weight[i * 2 + 1] * mix.y;
            }
        } else {
            for (const auto &t : mix.mixes) {
                if (t.weight < EPS) continue;
                auto row = embd_rows.find(t.token);
                if (row == embd_rows.end()) continue;
                for (int i = 0; i < n_embd; i++) mix_f[i] += row->second[i] * t.weight;
                num_added++;
            }
        }
        if (num_added == 0) {
            LOGE("a token mix had 0 weight for everything");
            return {-1, 0, false};
        }
        embeds.insert(embeds.end(), mix_f.begin(), mix_f.end());
        size++;
    }

    const int n_tokens = (int) mixes.size();
    int n_past = cached_mix_amount(mixes);
    past_mixes = mixes;
    if (!ff.first.empty()) n_past = 0;   // the prompt changed, so every embedding has to be recomputed
    llama_memory_seq_rm(mem, 0, (llama_pos) prompt.size() + n_past, -1);

    if (!embeds.empty()) {
        // The taps not yet in the cache, then always an <XBC>. FUTO computed one token per forward
        // pass; with the <XBC> embedding at hand all of them go into a single pass.
        const bool one_pass = !xbc_embedding.empty();
        std::vector<float> input(embeds.begin() + (size_t) n_past * n_embd, embeds.end());
        if (one_pass) input.insert(input.end(), xbc_embedding.begin(), xbc_embedding.end());
        const int n_input = (int) (input.size() / n_embd);
        const int per_pass = one_pass ? n_input : 1;
        for (int start = 0; start < n_input; start += per_pass) {
            llama_batch eb = {};
            eb.n_tokens = std::min(per_pass, n_input - start);
            eb.embd = input.data() + (size_t) start * n_embd;
            eb.pos = batch.pos;
            eb.n_seq_id = batch.n_seq_id;
            eb.seq_id = batch.seq_id;
            eb.logits = batch.logits;
            for (int i = 0; i < eb.n_tokens; i++) {
                batch.pos[i] = (llama_pos) (prompt.size() + n_past + start + i);
                batch.seq_id[i][0] = 0;
                batch.n_seq_id[i] = 1;
                batch.logits[i] = false;
            }
            if (one_pass) batch.logits[eb.n_tokens - 1] = true;
            if (decode(eb) != 0) {
                LOGE("llama_decode() with embeddings failed");
                past_mixes.clear();
                return {-1, 0, false};
            }
        }
        size += 1;
        head = n_input - 1;

        if (!one_pass) {
            batch.n_tokens = 1;
            batch.token[0] = special.XBC;
            batch.seq_id[0][0] = 0;
            batch.n_seq_id[0] = 1;
            batch.logits[0] = true;
            batch.pos[0] = (llama_pos) (prompt.size() + n_tokens);
            head = 0;
            if (decode(batch) != 0) {
                LOGE("llama_decode() for XBC failed");
                past_mixes.clear();
                return {-1, 0, false};
            }
        }
    }

    llama_memory_seq_rm(mem, 0, (llama_pos) size, -1);
    return {head, (int) size, !mixes.empty() || head == 0};
}

bool futolm::matches_banned(const token_sequence &prior, int prior_hash, llama_token next, const std::vector<banned_sequence> &banned) const {
    int new_hash = append_sequence_hash(prior_hash, next);
    for (const auto &b : banned) {
        if (b.sequence.empty()) continue;
        if (b.sequence.back() == special.STAR && prior.size() >= b.sequence.size() - 1) {
            bool matches = true;
            for (size_t i = 0; i < b.sequence.size() - 1; i++) {
                if (prior[i] != b.sequence[i]) { matches = false; break; }
            }
            if (matches) return true;
        } else if (b.sequence.size() == prior.size() + 1 && b.hash == new_hash && b.sequence.back() == next) {
            bool matches = true;
            for (size_t i = 0; i < prior.size(); i++) {
                if (prior[i] != b.sequence[i]) { matches = false; break; }
            }
            if (matches) return true;
        }
    }
    return false;
}

bool futolm::is_word_end(llama_token t) const {
    if (xec) return false;   // only <XEC>, checked separately
    const std::string p = tok.piece(t);
    return p.size() >= 3 && p.compare(p.size() - 3, 3, "\xe2\x96\x81") == 0;   // ends with "▁"
}

std::vector<std::pair<float, token_sequence>> futolm::sample(DecodeResult r, int n_results, WordCapitalizeMode capitals, const std::vector<banned_sequence> &banned) {
    llama_memory_t mem = llama_get_memory(ctx);
    std::vector<potential_sequence> sequences;
    const bool allow_correction_token = r.allow_correction_token;

    float *logits = llama_get_logits_ith(ctx, r.logits_head);
    bool is_bugged = !xec && n_vocab > 561 && logits[561] == 0.0f;
    if (!transform_logits(logits, true, allow_correction_token, capitals, 0)) {
        LOGE("logits have NaN");
        return {};
    }
    // FUTO's guard against a broken cache state, kept for its own models
    is_bugged = is_bugged && logits[561] < -990.0f && logits[561] > -1100.0f;
    if (is_bugged) {
        LOGE("detected FUTO bug 1, resetting the cache");
        llama_memory_seq_rm(mem, -1, -1, -1);
        active_context.clear();
        return {};
    }

    std::vector<std::pair<float, int>> index_value;
    index_value.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) index_value.emplace_back(logits[i], i);
    sortProbabilityPairVectorDescending(index_value, n_results * 2);
    const token_sequence blank = {};
    for (int i = 0; i < n_results * 2; i++) {
        if (matches_banned(blank, 0, index_value[i].second, banned)) index_value[i].first = 0.0f;
    }
    sortProbabilityPairVectorDescending(index_value, n_results);

    sequences.reserve(n_results);
    for (int i = 0; i < n_results; i++) {
        sequences.emplace_back(index_value[i].first, potential_sequence_data{{index_value[i].second}, i});
    }

    if (!xec) {
        is_bugged = true;
        for (const auto &seq : sequences) {
            if (seq.second.tokens.front() > 48 || seq.first != sequences[0].first) { is_bugged = false; break; }
        }
        if (is_bugged) {
            LOGE("detected FUTO bug 2, resetting the cache");
            llama_memory_seq_rm(mem, -1, -1, -1);
            active_context.clear();
            return {};
        }
    }

    for (auto &sequence : sequences) {
        if (sequence.second.seq_id == 0) continue;
        llama_memory_seq_cp(mem, 0, sequence.second.seq_id, 0, r.size);
    }

    std::vector<potential_sequence> next_sequences;
    std::vector<std::pair<float, token_sequence>> outputs;

    static const int max_steps = getenv("FUTOLM_MAX_STEPS") ? atoi(getenv("FUTOLM_MAX_STEPS")) : 10;   // for benchmarks
    for (int step = 0; step < max_steps; step++) {
        next_sequences.clear();
        for (auto sequence : std::move(sequences)) {
            llama_token next_token = sequence.second.tokens.back();
            if (next_token == special.XEC) {     // end of correction / of the word
                token_sequence result = std::move(sequence.second.tokens);
                result.resize(result.size() - 1);
                outputs.emplace_back(sequence.first, result);
                continue;
            }
            if (is_word_end(next_token)) {
                outputs.emplace_back(sequence.first, std::move(sequence.second.tokens));
                continue;
            }
            next_sequences.emplace_back(sequence);
        }
        sequences = next_sequences;
        next_sequences.clear();

        size_t remaining_count = n_results - outputs.size();
        batch.n_tokens = 0;
        for (auto &sequence : sequences) {
            batch.token[batch.n_tokens] = sequence.second.tokens.back();
            batch.pos[batch.n_tokens] = (llama_pos) (r.size + (sequence.second.tokens.size() - 1));
            batch.seq_id[batch.n_tokens][0] = sequence.second.seq_id;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.logits[batch.n_tokens] = true;
            batch.n_tokens += 1;
        }
        if (batch.n_tokens != (int) remaining_count) {
            LOGE("beam bookkeeping out of sync (%d vs %zu)", batch.n_tokens, remaining_count);
            break;
        }
        if (batch.n_tokens == 0) break;
        if (decode(batch) != 0) {
            LOGE("llama_decode() for a beam step failed");
            break;
        }

        for (int seq = 0; seq < (int) remaining_count; seq++) {
            const potential_sequence &parent_seq = sequences[seq];
            auto hash = compute_sequence_hash(parent_seq.second.tokens);
            llama_token prev_token = parent_seq.second.tokens.empty() ? 0 : parent_seq.second.tokens.back();

            logits = llama_get_logits_ith(ctx, seq);
            if (!transform_logits(logits, false, allow_correction_token, capitals, prev_token)) {
                LOGE("logits have NaN");
                return {};
            }
            index_value.clear();
            for (int i = 0; i < n_vocab; i++) index_value.emplace_back(logits[i], i);
            sortProbabilityPairVectorDescending(index_value, remaining_count * 2);
            for (size_t i = 0; i < remaining_count * 2; i++) {
                if (matches_banned(parent_seq.second.tokens, hash, index_value[i].second, banned)) index_value[i].first = 0.0f;
            }
            sortProbabilityPairVectorDescending(index_value, remaining_count);
            for (size_t i = 0; i < remaining_count; i++) {
                token_sequence new_sequence = parent_seq.second.tokens;
                new_sequence.push_back(index_value[i].second);
                next_sequences.emplace_back(index_value[i].first * parent_seq.first,
                                            potential_sequence_data{new_sequence, parent_seq.second.seq_id});
            }
        }

        sortProbabilityPairVectorDescending(next_sequences, remaining_count);
        next_sequences.resize(remaining_count);
        sequences.clear();

        // Several survivors may come from the same parent: give them their own sequence ids
        std::vector<int> seq_id_use_count(n_results, 0);
        for (auto &seq : next_sequences) seq_id_use_count[seq.second.seq_id] += 1;
        for (auto &seq : next_sequences) {
            if (seq_id_use_count[seq.second.seq_id] <= 1) continue;
            int old_seq_id = seq.second.seq_id;
            int new_seq_id = -1;
            for (int i = 0; i < n_results; i++) if (seq_id_use_count[i] == 0) { new_seq_id = i; break; }
            if (new_seq_id == -1) {
                LOGE("no free sequence id, this should never happen");
                return {};
            }
            seq_id_use_count[old_seq_id]--;
            seq_id_use_count[new_seq_id]++;
            llama_memory_seq_cp(mem, old_seq_id, new_seq_id, 0,
                                (llama_pos) (r.size + (seq.second.tokens.size() - 1)));
            seq.second.seq_id = new_seq_id;
        }
        sequences = next_sequences;
    }

    for (int i = 1; i < n_results; i++) llama_memory_seq_rm(mem, i, 0, -1);
    return outputs;
}

std::vector<banned_sequence> futolm::banned_sequences(const std::vector<std::string> &words) const {
    std::vector<banned_sequence> out;
    for (const std::string &bw : words) {
        if (xec) {
            auto t = tok.tokenize(" " + trim(bw));
            out.push_back({t, compute_sequence_hash(t)});
        } else {
            auto t1 = tok.tokenize(trim(bw) + " ");
            out.push_back({t1, compute_sequence_hash(t1)});
            auto t2 = tok.tokenize(trim(bw));
            out.push_back({t2, compute_sequence_hash(t2)});
        }
    }
    return out;
}

// The context as the model saw it in training. FUTO: BOS + "context " (trailing space as a
// suffix piece; FUTO skipped it for corrections without context). xec: [BOS if the vocabulary
// wants one] + "context" (no trailing space).
token_sequence futolm::context_prompt(const std::string &context, bool for_correction) const {
    token_sequence out;
    if (xec) {
        if (llama_vocab_get_add_bos(vocab)) out.push_back(llama_vocab_bos(vocab));
        if (!trim(context).empty()) {
            auto t = tok.tokenize(trim(context));
            out.insert(out.end(), t.begin(), t.end());
        }
    } else {
        out.push_back(1);   // BOS
        if (!for_correction || !context.empty()) {
            auto t = tok.tokenize(trim(context) + " ");
            out.insert(out.end(), t.begin(), t.end());
        }
    }
    if ((int) out.size() > MAX_PROMPT) {   // keep the beginning-of-text token and the most recent part
        LOGI("context of %zu tokens cut to %d", out.size(), MAX_PROMPT);
        token_sequence cut;
        cut.push_back(out[0]);
        cut.insert(cut.end(), out.end() - (MAX_PROMPT - 1), out.end());
        out = cut;
    }
    return out;
}

// FUTO models decode a finished word with its trailing space ("word "); FUTO's rules below (exact
// match, "way shorter") saw that string, so it is kept until the result is handed out.
std::string futolm::word_text(const token_sequence &tokens) const {
    return xec ? trim(tok.decode(tokens)) : tok.decode(tokens);
}

std::vector<std::pair<float, std::string>> futolm::predict_next_word(const std::string &context, const std::vector<std::string> &banned) {
    token_sequence prompt = context_prompt(context, false);
    if (xec) prompt.push_back(special.XBC);
    auto r = decode_prompt_and_mixes(prompt, {});
    if (r.logits_head < 0) return {};
    auto results = sample(r, NUM_RESULTS, IgnoredCapitals, banned_sequences(banned));
    std::vector<std::pair<float, std::string>> out;
    for (const auto &res : results) out.emplace_back(res.first, word_text(res.second));
    return out;
}

std::vector<std::pair<float, std::string>> futolm::predict_correction(const std::string &context, const std::vector<TokenMix> &mixes, bool swipe, WordCapitalizeMode capitals, const std::vector<std::string> &banned) {
    if (special.XBU <= 0) return {};
    token_sequence prompt = context_prompt(context, true);
    prompt.push_back(special.XBU);
    if (swipe && (!xec || special.XC0_SWIPE_MODE > 0)) prompt.push_back(special.XC0_SWIPE_MODE);
    auto r = decode_prompt_and_mixes(prompt, mixes);
    if (r.logits_head < 0) return {};
    auto results = sample(r, NUM_RESULTS, capitals, banned_sequences(banned));
    std::vector<std::pair<float, std::string>> out;
    for (const auto &res : results) out.emplace_back(res.first, word_text(res.second));
    return out;
}

// ---------------------------------------------------------------- C API

extern "C" {

futolm *futolm_open(const char *path, int n_threads) {
    if (!cpu_supported()) {
        LOGE("this CPU lacks the instructions libfutolm was built for");
        return nullptr;
    }
    static std::once_flag once;
    std::call_once(once, [] {
        llama_backend_init();
        llama_log_set(log_callback, nullptr);
    });

    auto lm = std::make_unique<futolm>();
    lm->path = path;
    lm->meta = read_metadata(path);
    if (!lm->meta.ok) {
        LOGE("can't read %s", path);
        return nullptr;
    }

    llama_model_params mp = llama_model_default_params();
    // Most weights get repacked into a faster layout in memory. Reading the file instead of
    // mapping it keeps the original bytes from staying resident next to that copy.
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    lm->model = llama_model_load_from_file(path, mp);
    if (!lm->model) {
        LOGE("llama.cpp can't load %s", path);
        return nullptr;
    }
    if (llama_model_is_recurrent(lm->model) || llama_model_is_hybrid(lm->model)) {
        // Their state can't be rolled back to an earlier letter yet (needs per-keystroke snapshots)
        LOGE("models with a recurrent state (e.g. Qwen3.5) are not supported yet");
        return nullptr;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = CONTEXT_SIZE;
    cp.n_batch = CONTEXT_SIZE;
    cp.n_ubatch = MICRO_BATCH;
    cp.n_seq_max = NUM_RESULTS;
    cp.n_threads = cp.n_threads_batch = std::max(1, n_threads);
    cp.kv_unified = true;     // the beams share the prompt's cache cells
    cp.no_perf = true;
    if (const char *fa = getenv("FUTOLM_FLASH_ATTN")) {   // for benchmarks: -1 auto, 0 off, 1 on
        cp.flash_attn_type = (llama_flash_attn_type) atoi(fa);
    }
    lm->ctx = llama_init_from_model(lm->model, cp);
    if (!lm->ctx) {
        LOGE("can't create a context for %s", path);
        return nullptr;
    }

    lm->vocab = llama_model_get_vocab(lm->model);
    lm->n_vocab = llama_vocab_n_tokens(lm->vocab);
    lm->n_embd = llama_model_n_embd_inp(lm->model);
    lm->n_batch = CONTEXT_SIZE;
    lm->batch = llama_batch_init(CONTEXT_SIZE, 0, NUM_RESULTS);
    lm->xec = lm->meta.has(FEATURE_XEC_WORD);

    if (lm->xec) {
        lm->tok.vocab = lm->vocab;
        for (int i = 0; i < lm->n_vocab; i++) {
            if (llama_vocab_get_attr(lm->vocab, i) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
                lm->tok.special_text[llama_vocab_get_text(lm->vocab, i)] = i;
            }
        }
    } else if (lm->meta.ext_tokenizer_type == "sentencepiece") {
        lm->tok.spm_mode = true;
        auto status = lm->tok.spm.LoadFromSerializedProto(lm->meta.ext_tokenizer_data);
        if (!status.ok()) {
            LOGE("SentencePiece model doesn't load: %s", status.ToString().c_str());
            return nullptr;
        }
    } else {
        LOGE("%s has neither a SentencePiece tokenizer nor the xec_word_v1 feature", path);
        return nullptr;
    }

    if (!lm->init_special_tokens()) return nullptr;

    if (lm->meta.has(FEATURE_EMBED_MIXING) || lm->xec) {
        std::vector<llama_token> rows(std::begin(lm->special.LETTERS_TO_IDS), std::end(lm->special.LETTERS_TO_IDS));
        if (lm->meta.has(FEATURE_ENCODER)) rows.insert(rows.end(), {ENCODER_W_X_ID, ENCODER_W_Y_ID, ENCODER_B_ID});
        if (lm->special.XBC > 0) rows.push_back(lm->special.XBC);
        if (!read_embedding_rows(path, lm->n_embd, rows, lm->embd_rows)) {
            LOGE("can't read the key embeddings");
            return nullptr;
        }
        if (lm->meta.has(FEATURE_ENCODER)) {
            lm->encoder_weight.resize(lm->n_embd * 2);
            lm->encoder_bias.resize(lm->n_embd);
            for (int i = 0; i < lm->n_embd; i++) {
                lm->encoder_weight[i * 2] = lm->embd_rows[ENCODER_W_X_ID][i];
                lm->encoder_weight[i * 2 + 1] = lm->embd_rows[ENCODER_W_Y_ID][i];
                lm->encoder_bias[i] = lm->embd_rows[ENCODER_B_ID][i];
            }
        }
        auto xbc = lm->embd_rows.find(lm->special.XBC);
        const char *one_pass_env = getenv("FUTOLM_ONE_PASS");   // for benchmarks: 0 = as FUTO did
        if (xbc != lm->embd_rows.end() && !(one_pass_env && atoi(one_pass_env) == 0)) lm->xbc_embedding = xbc->second;
    }

    std::string cores = "any cores";
#ifdef __linux__
    const char *pin_env = getenv("FUTOLM_PIN");   // for benchmarks: 0 = let the scheduler decide
    lm->pin = !(pin_env && atoi(pin_env) == 0) && fastest_cpus(std::max(1, n_threads), lm->fast_cpus);
    if (lm->pin) {
        cores = "cores";
        for (int c = 0; c < CPU_SETSIZE && c < 64; c++) if (CPU_ISSET(c, &lm->fast_cpus)) cores += " " + std::to_string(c);
    }
#endif

    char arch[64] = {0};
    llama_model_meta_val_str(lm->model, "general.architecture", arch, sizeof(arch));
    char desc[256];
    snprintf(desc, sizeof(desc), "%s, %d tokens, %s, %d threads on %s", arch, lm->n_vocab,
             lm->xec ? "xec_word_v1" : "FUTO SentencePiece", std::max(1, n_threads), cores.c_str());
    lm->description = desc;
    LOGI("loaded %s: %s", path, desc);
    return lm.release();
}

void futolm_close(futolm *lm) { delete lm; }

const char *futolm_describe(const futolm *lm) { return lm ? lm->description.c_str() : ""; }

int futolm_suggest(futolm *lm, const char *context, const char *partial_word,
                   const futolm_mix *in_mixes, int n_mixes, int input_mode, float autocorrect_threshold,
                   const char *const *banned_words, int n_banned_words,
                   futolm_result *results_out, int max_results, int *mode) {
    if (!lm) return -1;
    std::lock_guard<std::mutex> guard(lm->lock);
    FastCores fast(lm);
    const std::string contextString = context ? context : "";
    const std::string partialWordString = partial_word ? partial_word : "";

    WordCapitalizeMode capitals = IgnoredCapitals;
    if (!partialWordString.empty() && !isFirstCharLowercase(partialWordString.c_str(), false)) {
        capitals = (partialWordString.size() > 1 && !hasLowercase(partialWordString.c_str(), false)) ? AllCapitals : FirstCapital;
    }
    std::vector<std::string> bannedWords;
    for (int i = 0; i < n_banned_words; i++) bannedWords.emplace_back(banned_words[i]);

    std::vector<TokenMix> mixes;
    for (int i = 0; i < n_mixes; i++) {
        TokenMix m{};
        m.x = in_mixes[i].x;
        m.y = in_mixes[i].y;
        for (int j = 0; j < FUTOLM_MIX_KEYS; j++) {
            int c = in_mixes[i].codepoint[j];
            m.mixes[j].weight = in_mixes[i].weight[j];
            if (c >= 'a' && c <= 'z') m.mixes[j].token = lm->special.LETTERS_TO_IDS[c - 'a'];
            else if (c >= 'A' && c <= 'Z') m.mixes[j].token = lm->special.LETTERS_TO_IDS[c - 'A'];
            else m.mixes[j].weight = 0.0f;
        }
        mixes.push_back(m);
    }
    // Probably a number is being typed: no predictions at all
    if (mixes.empty() && !partialWordString.empty()) return -1;

    std::vector<std::pair<float, std::string>> results;
    if (partialWordString.empty()) {
        results = lm->predict_next_word(contextString, bannedWords);
    } else {
        results = lm->predict_correction(contextString, mixes, input_mode == 1, capitals, bannedWords);
        bool hasExactMatch = false;
        for (const auto &r : results) if (isExactMatch(r.second, partialWordString)) hasExactMatch = true;
        if (hasExactMatch) {
            for (auto &r : results) if (!isExactMatch(r.second, partialWordString)) r.first -= 1.0f;
        }
    }
    sortProbabilityPairVectorDescending(results);

    int m;
    if (results.size() < 2) m = FUTOLM_MODE_UNCERTAIN;
    else if (results[0].first > autocorrect_threshold * results[1].first) m = FUTOLM_MODE_AUTOCORRECT;
    else if (results[0].first > (autocorrect_threshold * 0.1f) * results[1].first) m = FUTOLM_MODE_UNCERTAIN;
    else m = FUTOLM_MODE_CLUELESS;
    // No way it's correct if it's way shorter (unless swiping)
    if (!results.empty() && !partialWordString.empty() && results[0].second.size() * 2 < partialWordString.size() && input_mode != 1) {
        m = FUTOLM_MODE_CLUELESS;
    }
    if (mode) *mode = m;

    int n = 0;
    for (const auto &r : results) {
        if (n >= max_results) break;
        snprintf(results_out[n].word, FUTOLM_MAX_WORD, "%s", trim(r.second).c_str());
        results_out[n].probability = r.first;
        n++;
    }
    return n;
}

void futolm_rescore(futolm *lm, const char *context, const char *const *words, const float *scores, int n_words, float *out_scores) {
    if (!lm) return;
    std::lock_guard<std::mutex> guard(lm->lock);
    FastCores fast(lm);
    token_sequence prompt = lm->context_prompt(context ? context : "", false);
    if (lm->xec) prompt.push_back(lm->special.XBC);
    auto r = lm->decode_prompt_and_mixes(prompt, {});
    if (r.logits_head < 0) {
        for (int i = 0; i < n_words; i++) out_scores[i] = scores[i];
        return;
    }
    float *logits = llama_get_logits_ith(lm->ctx, r.logits_head);
    softmax(logits, lm->n_vocab);
    for (int i = 0; i < n_words; i++) {
        auto t = lm->xec ? lm->tok.tokenize(" " + trim(words[i])) : lm->tok.tokenize(trim(words[i]) + " ");
        float pseudo = t.empty() ? 0.0f : logits[t[0]] / (float) t.size();
        out_scores[i] = scores[i] * pseudo * 1000.0f;
    }
}

void futolm_counters(const futolm *lm, long *decode_calls, long *decode_tokens) {
    if (decode_calls) *decode_calls = lm ? lm->decode_calls : 0;
    if (decode_tokens) *decode_tokens = lm ? lm->decode_tokens : 0;
}

} // extern "C"
