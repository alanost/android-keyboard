// futolm_taps.h: turns the taps of the word being typed into futolm mixes, exactly as FUTO's
// xlm_LanguageModel_getSuggestions did. Header-only so that the keyboard's JNI glue (with its
// ProximityInfo) and the benchmark (keyboard read from a workload file) share the same code.
//
// Keyboard must provide (ProximityInfo does):
//   std::vector<float> decomposeTapPosition(int x, int y) const   share of the touch area per key
//   int getKeyCodePoint(int key) const
//   int getMostCommonKeyWidth() const
//   int getKeyboardWidth() const
//   int getKeyboardHeight() const

#ifndef FUTOLM_TAPS_H
#define FUTOLM_TAPS_H

#include "futolm.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace futolm_taps {

inline void sort_descending(std::vector<std::pair<float, int>> &vec, size_t partial) {
    if (partial > vec.size()) partial = vec.size();
    std::partial_sort(vec.begin(), vec.begin() + partial, vec.end(),
                      [](const std::pair<float, int> &a, const std::pair<float, int> &b) { return a.first > b.first; });
}

inline bool is_letter(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

// xCoordinates/yCoordinates: one tap per character of partialWord, -1 where it is unknown.
template<typename Keyboard>
std::vector<futolm_mix> mixes_from_taps(const Keyboard &kb, const std::string &partialWord,
                                        const int *xCoordinates, const int *yCoordinates, size_t inputSize) {
    if (partialWord.size() < inputSize) inputSize = partialWord.size();

    std::vector<futolm_mix> mixes;
    int numSkippedDueToNoCoordinate = 0;
    for (size_t i = 0; i < inputSize; i++) {
        const char wc = partialWord[i];
        if (!is_letter(wc) && !(wc >= '0' && wc <= '9')) continue;
        if (xCoordinates[i] == -1 || yCoordinates[i] == -1) {
            numSkippedDueToNoCoordinate++;
            continue;
        }

        int tapY = yCoordinates[i];
        // A number key: move the tap a little down to find the key below when typing with the
        // number row active
        if (wc >= '0' && wc <= '9') tapY += kb.getMostCommonKeyWidth() * 2 / 3;

        std::vector<float> proportions = kb.decomposeTapPosition(xCoordinates[i], tapY);
        for (float &f : proportions) if (f < 0.05f) f = 0.0f;

        std::vector<std::pair<float, int>> index_value;
        for (size_t k = 0; k < proportions.size(); k++) index_value.emplace_back(proportions[k], (int) k);
        sort_descending(index_value, FUTOLM_MIX_KEYS);
        if (index_value.size() < FUTOLM_MIX_KEYS) break;

        bool needs_resorting = false;
        int num_symbols = 0;
        for (int s = 0; s < 4; s++) {
            num_symbols = 0;
            for (int j = 0; j < FUTOLM_MIX_KEYS; j++) {
                const char c = (char) kb.getKeyCodePoint(index_value[j].second);
                if (!is_letter(c) && index_value[j].first > 0.0f) {
                    index_value[j].first = 0.0f;
                    needs_resorting = true;
                    num_symbols++;
                }
            }
            if (num_symbols == FUTOLM_MIX_KEYS) break;
            if (!needs_resorting) break;
            sort_descending(index_value, FUTOLM_MIX_KEYS);
        }
        if (num_symbols == FUTOLM_MIX_KEYS) continue;   // a symbol key

        float total_sum = 0.0f;
        for (int j = 0; j < FUTOLM_MIX_KEYS; j++) total_sum += index_value[j].first;
        if (total_sum == 0.0f) {
            numSkippedDueToNoCoordinate++;
            continue;
        }
        for (int j = 0; j < FUTOLM_MIX_KEYS; j++) index_value[j].first /= total_sum;

        futolm_mix mix{};
        mix.x = ((float) xCoordinates[i]) / ((float) kb.getKeyboardWidth());
        mix.y = ((float) yCoordinates[i]) / ((float) kb.getKeyboardHeight());
        for (int j = 0; j < FUTOLM_MIX_KEYS; j++) {
            const char c = (char) kb.getKeyCodePoint(index_value[j].second);
            mix.codepoint[j] = is_letter(c) ? c : 0;
            mix.weight[j] = is_letter(c) ? index_value[j].first : 0.0f;
        }
        mixes.push_back(mix);
    }

    if (mixes.empty() && numSkippedDueToNoCoordinate > 0) {
        // No usable coordinates at all: fall back to the typed letters themselves
        for (size_t i = 0; i < inputSize; i++) {
            const char wc = partialWord[i];
            if (!is_letter(wc)) continue;
            futolm_mix mix{};
            mix.x = -1.0f;
            mix.y = -1.0f;
            for (int j = 0; j < FUTOLM_MIX_KEYS; j++) {
                mix.codepoint[j] = wc;
                mix.weight[j] = 0.0f;
            }
            mix.weight[0] = 1.0f;
            mixes.push_back(mix);
        }
    }
    return mixes;
}

} // namespace futolm_taps

#endif // FUTOLM_TAPS_H
