// lmbench: runs FUTO's transformer language model exactly like the keyboard does per keystroke,
// without Android UI or JNI calls. Used to measure latency on a phone (adb shell) and to compare
// prediction quality of models on recorded typing.
//
//   lmbench <model.gguf> <workload.tsv> [-t threads] [-n max_queries] [-q]
//
// Workload lines (tab separated):
//   K  <keyboard width> <keyboard height> <most common key width> <code,x,y,w,h;code,x,y,w,h;...>
//   Q  <id> <x,y;x,y;...> <partial word> <context>
// For every Q line one R line is printed:
//   R  <id> <milliseconds> <mode> <word>\x1f<probability> ...
// A summary (latency percentiles for next-word and in-word queries, peak memory) goes to stderr.
//
// The model code is FUTO's own (org_futo_inputmethod_latin_xlm_LanguageModel.cpp is included
// verbatim); get_suggestions() mirrors xlm_LanguageModel_getSuggestions with the JNI and
// ProximityInfo access replaced by plain data from the workload.

// Every llama_decode the model code makes goes through decode_counted(), so the summary can report
// how many forward passes (and tokens) a keystroke costs. Latency is roughly passes x cost per pass.
#include "../../native/jni/src/ggml/llama.h"
static long g_decodeCalls = 0, g_decodeTokens = 0;
static int decode_counted(llama_context *ctx, llama_batch batch) {
    g_decodeCalls++;
    g_decodeTokens += batch.n_tokens;
    return llama_decode(ctx, batch);
}
#define llama_decode decode_counted
#include "../../native/jni/org_futo_inputmethod_latin_xlm_LanguageModel.cpp"
#undef llama_decode

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

namespace latinime {
// Only referenced by register_LanguageModel(), which is never called here.
int registerNativeMethods(JNIEnv *, const char *const, const JNINativeMethod *, const int) { return 0; }
}

namespace {

struct Keyboard {
    int width = 0, height = 0, mostCommonKeyWidth = 0;
    std::vector<int> code, x, y, w, h;

    // ProximityInfo::decomposeTapPosition
    std::vector<float> decompose(int tapX, int tapY) const {
        std::vector<float> percentages(code.size(), 0.0f);
        float tapRadius = mostCommonKeyWidth * 0.292f;
        float totalArea = M_PI * ((float)(tapRadius * tapRadius));
        bool anySet = false;
        for (size_t key = 0; key < code.size(); key++) {
            const int left = x[key];
            const int top = y[key];
            const int right = left + w[key] + 1;
            const int bottom = top + h[key];
            percentages[key] = insmat::area(left, right, bottom, top, tapX, tapY, tapRadius) / totalArea;
            if (percentages[key] > 0.05f) anySet = true;
        }
        if (!anySet) {
            int closestKey = -1, minDistance = 1000000;
            for (size_t key = 0; key < code.size(); key++) {
                const int distance = (x[key] - tapX) * (x[key] - tapX) + (y[key] - tapY) * (y[key] - tapY);
                if (distance < minDistance) { minDistance = distance; closestKey = (int)key; }
            }
            if (closestKey != -1) percentages[closestKey] = 1.0f;
        }
        return percentages;
    }
};

struct Result {
    std::string mode;
    std::vector<std::pair<float, std::string>> words;
};

// Mirrors xlm_LanguageModel_getSuggestions (tap typing, no banned words, threshold from settings).
Result get_suggestions(LanguageModelState *state, const Keyboard &kb, const std::string &contextString,
                       const std::string &partialWordString, const std::vector<int> &xCoordinates,
                       const std::vector<int> &yCoordinates, float autocorrectThreshold) {
    Result out;
    size_t inputSize = xCoordinates.size();
    if (partialWordString.size() < inputSize) inputSize = partialWordString.size();

    WordCapitalizeMode capitals = WordCapitalizeMode::IgnoredCapitals;
    if (!partialWordString.empty() && !isFirstCharLowercase(partialWordString.c_str(), false)) {
        if (partialWordString.size() > 1 && !hasLowercase(partialWordString.c_str(), false)) {
            capitals = WordCapitalizeMode::AllCapitals;
        } else {
            capitals = WordCapitalizeMode::FirstCapital;
        }
    }
    std::vector<std::string> bannedWords;

    std::vector<TokenMix> mixes;
    int numSkippedDueToNoCoordinate = 0;
    for (size_t i = 0; i < inputSize; i++) {
        char wc = partialWordString[i];
        if (!(wc >= 'a' && wc <= 'z') && !(wc >= 'A' && wc <= 'Z') && !(wc >= '0' && wc <= '9')) continue;
        if (xCoordinates[i] == -1 || yCoordinates[i] == -1) { numSkippedDueToNoCoordinate++; continue; }

        int tapY = yCoordinates[i];
        if (wc >= '0' && wc <= '9') tapY += kb.mostCommonKeyWidth * 2 / 3;

        std::vector<float> proportions = kb.decompose(xCoordinates[i], tapY);
        for (float &f : proportions) if (f < 0.05f) f = 0.0f;

        std::vector<std::pair<float, int>> index_value;
        for (size_t k = 0; k < proportions.size(); k++) index_value.emplace_back(proportions[k], k);
        sortProbabilityPairVectorDescending(index_value, NUM_TOKEN_MIX);
        if (index_value.size() < NUM_TOKEN_MIX) break;

        bool needs_resorting = false;
        int num_symbols = 0;
        for (int s = 0; s < 4; s++) {
            num_symbols = 0;
            for (int j = 0; j < NUM_TOKEN_MIX; j++) {
                char c = (char)(kb.code[index_value[j].second]);
                if (c >= 'a' && c <= 'z') {
                } else if (c >= 'A' && c <= 'Z') {
                } else if (index_value[j].first > 0.0f) {
                    index_value[j].first = 0.0f;
                    needs_resorting = true;
                    num_symbols++;
                }
            }
            if (num_symbols == NUM_TOKEN_MIX) break;
            if (!needs_resorting) break;
            sortProbabilityPairVectorDescending(index_value, NUM_TOKEN_MIX);
        }
        if (num_symbols == NUM_TOKEN_MIX) continue;

        float total_sum = 0.0f;
        for (int j = 0; j < NUM_TOKEN_MIX; j++) total_sum += index_value[j].first;
        if (total_sum == 0.0f) { numSkippedDueToNoCoordinate++; continue; }
        for (int j = 0; j < NUM_TOKEN_MIX; j++) index_value[j].first /= total_sum;

        TokenMix results {};
        results.x = ((float)xCoordinates[i]) / ((float)kb.width);
        results.y = ((float)yCoordinates[i]) / ((float)kb.height);
        for (int j = 0; j < NUM_TOKEN_MIX; j++) {
            char c = (char)(kb.code[index_value[j].second]);
            float w = index_value[j].first;
            results.mixes[j].weight = w;
            if (c >= 'a' && c <= 'z') {
                results.mixes[j].token = (state->specialTokens.LETTERS_TO_IDS[c - 'a']);
            } else if (c >= 'A' && c <= 'Z') {
                results.mixes[j].token = (state->specialTokens.LETTERS_TO_IDS[c - 'A']);
            } else {
                results.mixes[j].weight = 0.0f;
            }
        }
        mixes.push_back(results);
    }

    if (mixes.empty() && numSkippedDueToNoCoordinate > 0) {
        for (size_t i = 0; i < inputSize; i++) {
            char wc = partialWordString[i];
            if (!(wc >= 'a' && wc <= 'z') && !(wc >= 'A' && wc <= 'Z')) continue;
            TokenMix results {};
            results.x = -1.0f;
            results.y = -1.0f;
            for (int j = 0; j < NUM_TOKEN_MIX; j++) {
                results.mixes[j].weight = 0.0f;
                if (wc >= 'a' && wc <= 'z') results.mixes[j].token = (state->specialTokens.LETTERS_TO_IDS[wc - 'a']);
                else if (wc >= 'A' && wc <= 'Z') results.mixes[j].token = (state->specialTokens.LETTERS_TO_IDS[wc - 'A']);
            }
            results.mixes[0].weight = 1.0f;
            mixes.push_back(results);
        }
    }
    if (mixes.empty() && !partialWordString.empty()) { out.mode = "none"; return out; }

    std::vector<std::pair<float, std::string>> results;
    if (partialWordString.empty()) {
        results = state->PredictNextWord(contextString, bannedWords);
    } else {
        results = state->PredictCorrection(contextString, mixes, false, capitals, bannedWords);
        bool hasExactMatch = false;
        for (const auto &result : results) if (isExactMatch(result.second, partialWordString)) hasExactMatch = true;
        if (hasExactMatch) {
            for (auto &result : results) if (!isExactMatch(result.second, partialWordString)) result.first -= 1.0f;
        }
    }

    sortProbabilityPairVectorDescending(results);
    const char *mode;
    if (results.size() < 2) mode = RETURNVAL_UNCERTAIN;
    else if (results[0].first > autocorrectThreshold * results[1].first) mode = RETURNVAL_AUTOCORRECT;
    else if (results[0].first > (autocorrectThreshold * 0.1f) * results[1].first) mode = RETURNVAL_UNCERTAIN;
    else mode = RETURNVAL_CLUELESS;
    if (!results.empty() && !partialWordString.empty() && (results[0].second.size() * 2 < partialWordString.size())) {
        mode = RETURNVAL_CLUELESS;
    }
    out.mode = mode;
    out.words = results;
    return out;
}

std::vector<std::string> split(const std::string &s, char sep) {
    std::vector<std::string> parts;
    std::string cur;
    for (char c : s) {
        if (c == sep) { parts.push_back(cur); cur.clear(); } else cur += c;
    }
    parts.push_back(cur);
    return parts;
}

long status_kb(const char *field) {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(field, 0) == 0) return std::atol(line.c_str() + strlen(field));
    }
    return -1;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5))];
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: lmbench <model.gguf> <workload.tsv> [-t threads] [-n max_queries] [-q]\n");
        return 2;
    }
    std::string modelPath = argv[1], workloadPath = argv[2];
    int threads = 1;
    long maxQueries = -1;
    bool quiet = false;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-t" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "-n" && i + 1 < argc) maxQueries = std::atol(argv[++i]);
        else if (a == "-q") quiet = true;
    }

    llama_backend_init(true);   // as in register_LanguageModel
    auto t0 = std::chrono::steady_clock::now();
    auto *state = new LanguageModelState();
    if (!state->Initialize(modelPath)) {
        fprintf(stderr, "could not load %s\n", modelPath.c_str());
        return 1;
    }
    llama_set_n_threads(state->model->context(), threads, threads);
    double loadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::ifstream in(workloadPath);
    std::string line;
    Keyboard kb;
    std::vector<double> nextWord, inWord;
    double passes[2] = {0, 0}, tokens[2] = {0, 0};   // [0] next word, [1] in word
    long n = 0;
    while (std::getline(in, line) && (maxQueries < 0 || n < maxQueries)) {
        if (line.empty()) continue;
        auto f = split(line, '\t');
        if (f[0] == "K" && f.size() >= 5) {
            kb = Keyboard();
            kb.width = std::atoi(f[1].c_str());
            kb.height = std::atoi(f[2].c_str());
            kb.mostCommonKeyWidth = std::atoi(f[3].c_str());
            for (const auto &k : split(f[4], ';')) {
                auto v = split(k, ',');
                if (v.size() != 5) continue;
                kb.code.push_back(std::atoi(v[0].c_str()));
                kb.x.push_back(std::atoi(v[1].c_str()));
                kb.y.push_back(std::atoi(v[2].c_str()));
                kb.w.push_back(std::atoi(v[3].c_str()));
                kb.h.push_back(std::atoi(v[4].c_str()));
            }
        } else if (f[0] == "Q" && f.size() >= 5) {
            std::vector<int> xs, ys;
            if (!f[2].empty()) {
                for (const auto &p : split(f[2], ';')) {
                    auto v = split(p, ',');
                    if (v.size() != 2) continue;
                    xs.push_back(std::atoi(v[0].c_str()));
                    ys.push_back(std::atoi(v[1].c_str()));
                }
            }
            long calls0 = g_decodeCalls, tokens0 = g_decodeTokens;
            auto s = std::chrono::steady_clock::now();
            Result r = get_suggestions(state, kb, f[4], f[3], xs, ys, 4.0f);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count();
            (f[3].empty() ? nextWord : inWord).push_back(ms);
            passes[f[3].empty() ? 0 : 1] += g_decodeCalls - calls0;
            tokens[f[3].empty() ? 0 : 1] += g_decodeTokens - tokens0;
            n++;
            if (!quiet) {
                std::ostringstream o;
                o << "R\t" << f[1] << "\t" << ms << "\t" << r.mode;
                for (size_t i = 0; i < r.words.size() && i < 8; i++) o << "\t" << r.words[i].second << '\x1f' << r.words[i].first;
                std::cout << o.str() << "\n";
            }
        }
    }
    std::cout.flush();
    fprintf(stderr, "model %s threads %d load %.0f ms, queries %ld\n", modelPath.c_str(), threads, loadMs, n);
    fprintf(stderr, "next word: n=%zu median %.1f ms p90 %.1f p99 %.1f max %.1f\n", nextWord.size(),
            percentile(nextWord, .5), percentile(nextWord, .9), percentile(nextWord, .99), percentile(nextWord, 1));
    fprintf(stderr, "in word:   n=%zu median %.1f ms p90 %.1f p99 %.1f max %.1f\n", inWord.size(),
            percentile(inWord, .5), percentile(inWord, .9), percentile(inWord, .99), percentile(inWord, 1));
    fprintf(stderr, "forward passes per query: next word %.2f (%.1f tokens), in word %.2f (%.1f tokens)\n",
            passes[0] / std::max<size_t>(1, nextWord.size()), tokens[0] / std::max<size_t>(1, nextWord.size()),
            passes[1] / std::max<size_t>(1, inWord.size()), tokens[1] / std::max<size_t>(1, inWord.size()));
    fprintf(stderr, "memory: peak %ld MB, now %ld MB\n", status_kb("VmHWM:") / 1024, status_kb("VmRSS:") / 1024);
    delete state;
    return 0;
}
