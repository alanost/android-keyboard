// lmbench2: lmbench for libfutolm, the keyboard's language model on a current llama.cpp.
// Same workload format and output as lmbench, so both engines can be compared query by query.
//
//   lmbench2 <model.gguf> <workload.tsv> [-t threads] [-n max_queries] [-q]
//
// Workload lines (tab separated):
//   K  <keyboard width> <keyboard height> <most common key width> <code,x,y,w,h;code,x,y,w,h;...>
//   Q  <id> <x,y;x,y;...> <partial word> <context>
// For every Q line one R line is printed:
//   R  <id> <milliseconds> <mode> <word>\x1f<probability> ...
// A summary (latency percentiles, forward passes, peak memory) goes to stderr.

#include "futolm.h"
#include "futolm_taps.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ProximityInfo's tap decomposition, verbatim (src/suggest/core/layout/proximity_info.h)
namespace insmat {
    inline float max(float a, float b) { return ((a) > (b)) ? (a) : (b); }
    inline float min(float a, float b) { return ((a) > (b)) ? (b) : (a); }
    inline float section(float h, float r = 1) { return (h < r) ? sqrt(r * r - h * h) : 0; }
    inline float g(float x, float h, float r = 1) {
        return .5f * (sqrt(1 - x * x / (r * r)) * x * r + r * r * asin(x / r) - 2 * h * x);
    }
    inline float area(float x0, float x1, float h, float r) {
        if (x0 > x1) std::swap(x0, x1);
        float s = section(h, r);
        return g(max(-s, min(s, x1)), h, r) - g(max(-s, min(s, x0)), h, r);
    }
    inline float area(float x0, float x1, float y0, float y1, float r) {
        if (y0 > y1) std::swap(y0, y1);
        if (y0 < 0) {
            if (y1 < 0) return area(x0, x1, -y0, -y1, r);
            else return area(x0, x1, 0, -y0, r) + area(x0, x1, 0, y1, r);
        } else {
            return area(x0, x1, y0, r) - area(x0, x1, y1, r);
        }
    }
    inline float area(float x0, float x1, float y0, float y1, float cx, float cy, float r) {
        x0 -= cx; x1 -= cx;
        y0 -= cy; y1 -= cy;
        return area(x0, x1, y0, y1, r);
    }
}

struct Keyboard {
    int width = 0, height = 0, mostCommonKeyWidth = 0;
    std::vector<int> code, x, y, w, h;

    std::vector<float> decomposeTapPosition(int tapX, int tapY) const {
        std::vector<float> percentages(code.size(), 0.0f);
        float tapRadius = mostCommonKeyWidth * 0.292f;
        float totalArea = M_PI * ((float) (tapRadius * tapRadius));
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
                if (distance < minDistance) { minDistance = distance; closestKey = (int) key; }
            }
            if (closestKey != -1) percentages[closestKey] = 1.0f;
        }
        return percentages;
    }
    int getKeyCodePoint(int key) const { return code[key]; }
    int getMostCommonKeyWidth() const { return mostCommonKeyWidth; }
    int getKeyboardWidth() const { return width; }
    int getKeyboardHeight() const { return height; }
};

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
    return v[std::min(v.size() - 1, (size_t) (p * (v.size() - 1) + 0.5))];
}

const char *mode_name(int n, int mode) {
    if (n < 0) return "none";
    switch (mode) {
        case FUTOLM_MODE_AUTOCORRECT: return "autocorrect";
        case FUTOLM_MODE_CLUELESS: return "clueless";
        default: return "uncertain";
    }
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: lmbench2 <model.gguf> <workload.tsv> [-t threads] [-n max_queries] [-q]\n");
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

    auto t0 = std::chrono::steady_clock::now();
    futolm *lm = futolm_open(modelPath.c_str(), threads);
    if (!lm) {
        fprintf(stderr, "could not load %s\n", modelPath.c_str());
        return 1;
    }
    double loadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::ifstream in(workloadPath);
    std::string line;
    Keyboard kb;
    std::vector<double> nextWord, inWord;
    double passes[2] = {0, 0}, tokens[2] = {0, 0};   // [0] next word, [1] in word
    long n = 0;
    futolm_result results[8];
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
            long calls0, tokens0, calls1, tokens1;
            futolm_counters(lm, &calls0, &tokens0);
            auto s = std::chrono::steady_clock::now();
            // What the keyboard does per keystroke: taps to mixes, then the model
            auto mixes = futolm_taps::mixes_from_taps(kb, f[3], xs.data(), ys.data(), xs.size());
            int mode = FUTOLM_MODE_UNCERTAIN;
            int nres = futolm_suggest(lm, f[4].c_str(), f[3].c_str(), mixes.data(), (int) mixes.size(), 0, 4.0f,
                                      nullptr, 0, results, 8, &mode);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count();
            futolm_counters(lm, &calls1, &tokens1);
            (f[3].empty() ? nextWord : inWord).push_back(ms);
            passes[f[3].empty() ? 0 : 1] += calls1 - calls0;
            tokens[f[3].empty() ? 0 : 1] += tokens1 - tokens0;
            n++;
            if (!quiet) {
                std::ostringstream o;
                o << "R\t" << f[1] << "\t" << ms << "\t" << mode_name(nres, mode);
                for (int i = 0; i < nres; i++) o << "\t" << results[i].word << '\x1f' << results[i].probability;
                std::cout << o.str() << "\n";
            }
        }
    }
    std::cout.flush();
    fprintf(stderr, "model %s (%s) threads %d load %.0f ms, queries %ld\n", modelPath.c_str(), futolm_describe(lm),
            threads, loadMs, n);
    fprintf(stderr, "next word: n=%zu median %.1f ms p90 %.1f p99 %.1f max %.1f\n", nextWord.size(),
            percentile(nextWord, .5), percentile(nextWord, .9), percentile(nextWord, .99), percentile(nextWord, 1));
    fprintf(stderr, "in word:   n=%zu median %.1f ms p90 %.1f p99 %.1f max %.1f\n", inWord.size(),
            percentile(inWord, .5), percentile(inWord, .9), percentile(inWord, .99), percentile(inWord, 1));
    fprintf(stderr, "forward passes per query: next word %.2f (%.1f tokens), in word %.2f (%.1f tokens)\n",
            passes[0] / std::max<size_t>(1, nextWord.size()), tokens[0] / std::max<size_t>(1, nextWord.size()),
            passes[1] / std::max<size_t>(1, inWord.size()), tokens[1] / std::max<size_t>(1, inWord.size()));
    fprintf(stderr, "memory: peak %ld MB, now %ld MB\n", status_kb("VmHWM:") / 1024, status_kb("VmRSS:") / 1024);
    futolm_close(lm);
    return 0;
}
