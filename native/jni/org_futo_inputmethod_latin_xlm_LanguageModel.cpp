#define LOG_TAG "LatinIME: jni: LanguageModel"

// JNI glue for the keyboard's transformer language model. The model itself runs in libfutolm
// (native/jni/futolm: FUTO's prediction logic on a current llama.cpp); here only the Java data is
// converted and the taps are turned into key mixes with the keyboard's ProximityInfo.

#include "org_futo_inputmethod_latin_xlm_LanguageModel.h"

#include <cmath>
#include <string>
#include <vector>

#include "jni.h"
#include "jni_common.h"
#include "defines.h"
#include "suggest/core/layout/proximity_info.h"
#include "jni_utils.h"
#include "ggml/llama.h"

#include "futolm.h"
#include "futolm_taps.h"

#define RETURNVAL_AUTOCORRECT "autocorrect"
#define RETURNVAL_UNCERTAIN "uncertain"
#define RETURNVAL_CLUELESS "clueless"

// The model runs on the two fastest cores (libfutolm picks them)
#define NUM_THREADS 2
#define MAX_RESULTS 8

namespace latinime {
    static jlong xlm_LanguageModel_open(JNIEnv *env, jclass clazz, jstring modelDir) {
        GGML_UNUSED(clazz);

        AKLOGI("open LM");
        const std::string path = jstring2string(env, modelDir);
        if (path.empty()) {
            AKLOGE("LanguageModel: no model path");
            return 0;
        }

        futolm *lm = futolm_open(path.c_str(), NUM_THREADS);
        if (lm == nullptr) {
            AKLOGE("LanguageModel: could not load %s", path.c_str());
            return 0;
        }
        AKLOGI("LanguageModel: %s", futolm_describe(lm));
        return reinterpret_cast<jlong>(lm);
    }

    static void xlm_LanguageModel_close(JNIEnv *env, jclass clazz, jlong statePtr) {
        GGML_UNUSED(env);
        GGML_UNUSED(clazz);

        AKLOGI("LanguageModel_close called!");
        futolm_close(reinterpret_cast<futolm *>(statePtr));
    }

    // (JLjava/lang/String;[Ljava/lang/String;[I[I)V
    static void xlm_LanguageModel_rescoreSuggestions(JNIEnv *env, jclass clazz,
        jlong dict,
        jstring context,
        jobjectArray inWords,
        jintArray inScores,

        jintArray outScores
    ) {
        GGML_UNUSED(clazz);
        auto *lm = reinterpret_cast<futolm *>(dict);

        const std::string contextString = jstring2string(env, context);

        const jsize inputSize = env->GetArrayLength(inScores);
        std::vector<jint> scores(inputSize);
        env->GetIntArrayRegion(inScores, 0, inputSize, scores.data());

        float maxScore = -INFINITY;
        float minScore = INFINITY;
        for (jint score : scores) {
            auto scoref = (float) score;
            if (scoref > maxScore) maxScore = scoref;
            if (scoref < minScore) minScore = scoref;
        }
        minScore -= (maxScore - minScore) * 0.33f;

        const jsize numWords = env->GetArrayLength(inWords);
        std::vector<std::string> words;
        std::vector<const char *> wordPtrs;
        std::vector<float> transformed;
        for (jsize i = 0; i < numWords && i < inputSize; i++) {
            auto jstr = (jstring) env->GetObjectArrayElement(inWords, i);
            words.push_back(jstring2string(env, jstr));
            transformed.push_back(((float) scores[i] - minScore) / (maxScore - minScore));
            env->DeleteLocalRef(jstr);
        }
        for (const auto &w : words) wordPtrs.push_back(w.c_str());

        std::vector<float> rescored(words.size());
        futolm_rescore(lm, contextString.c_str(), wordPtrs.data(), transformed.data(), (int) words.size(), rescored.data());

        jint *outArray = env->GetIntArrayElements(outScores, nullptr);
        for (size_t i = 0; i < words.size(); i++) {
            outArray[i] = (jint) (rescored[i] * (maxScore - minScore) + minScore);
        }
        env->ReleaseIntArrayElements(outScores, outArray, 0);
    }

    static void xlm_LanguageModel_getSuggestions(JNIEnv *env, jclass clazz,
         // inputs
         jlong dict,
         jlong proximityInfo,
         jstring context,
         jstring partialWord,
         jint inputMode,
         jintArray inComposeX,
         jintArray inComposeY,
         jfloat autocorrectThreshold,
         jobjectArray bannedWordsArray,

         // outputs
         jobjectArray outPredictions,
         jfloatArray outProbabilities
    ) {
        GGML_UNUSED(clazz);

        auto *lm = reinterpret_cast<futolm *>(dict);
        auto *pInfo = reinterpret_cast<ProximityInfo *>(proximityInfo);

        std::string contextString;
        if (context != nullptr) contextString = jstring2string(env, context);

        std::string partialWordString;
        if (partialWord != nullptr) partialWordString = jstring2string(env, partialWord);

        size_t inputSize = env->GetArrayLength(inComposeX);
        if (partialWordString.size() < inputSize) inputSize = partialWordString.size();

        std::vector<std::string> bannedWords;
        const size_t numBannedWords = env->GetArrayLength(bannedWordsArray);
        for (size_t i = 0; i < numBannedWords; i++) {
            auto bannedWord = (jstring) env->GetObjectArrayElement(bannedWordsArray, (jsize) i);
            bannedWords.push_back(jstring2string(env, bannedWord));
            env->DeleteLocalRef(bannedWord);
        }
        std::vector<const char *> bannedPtrs;
        for (const auto &w : bannedWords) bannedPtrs.push_back(w.c_str());

        std::vector<int> xCoordinates(inputSize), yCoordinates(inputSize);
        env->GetIntArrayRegion(inComposeX, 0, (jsize) inputSize, xCoordinates.data());
        env->GetIntArrayRegion(inComposeY, 0, (jsize) inputSize, yCoordinates.data());

        std::vector<futolm_mix> mixes = futolm_taps::mixes_from_taps(
                *pInfo, partialWordString, xCoordinates.data(), yCoordinates.data(), inputSize);

        futolm_result results[MAX_RESULTS];
        int mode = FUTOLM_MODE_UNCERTAIN;
        const int n = futolm_suggest(lm, contextString.c_str(), partialWordString.c_str(),
                                     mixes.data(), (int) mixes.size(), inputMode, autocorrectThreshold,
                                     bannedPtrs.data(), (int) bannedPtrs.size(), results, MAX_RESULTS, &mode);
        if (n < 0) {
            // Probably a number is being entered: no predictions at all
            return;
        }

        const char *result_probability_mode = RETURNVAL_UNCERTAIN;
        if (mode == FUTOLM_MODE_AUTOCORRECT) result_probability_mode = RETURNVAL_AUTOCORRECT;
        else if (mode == FUTOLM_MODE_CLUELESS) result_probability_mode = RETURNVAL_CLUELESS;

        const size_t size = env->GetArrayLength(outPredictions);
        jstring result_str = string2jstring(env, result_probability_mode);
        env->SetObjectArrayElement(outPredictions, (jsize) (size - 1), result_str);
        env->DeleteLocalRef(result_str);

        jfloat *probsArray = env->GetFloatArrayElements(outProbabilities, nullptr);
        for (int i = 0; i < n && i < (int) size - 1; i++) {
            jstring jstr = string2jstring(env, results[i].word);
            env->SetObjectArrayElement(outPredictions, i, jstr);
            probsArray[i] = results[i].probability;
            env->DeleteLocalRef(jstr);
        }
        env->ReleaseFloatArrayElements(outProbabilities, probsArray, 0);
    }

    static const JNINativeMethod sMethods[] = {
            {
                    const_cast<char *>("openNative"),
                    const_cast<char *>("(Ljava/lang/String;)J"),
                    reinterpret_cast<void *>(xlm_LanguageModel_open)
            },
            {
                    const_cast<char *>("closeNative"),
                    const_cast<char *>("(J)V"),
                    reinterpret_cast<void *>(xlm_LanguageModel_close)
            },
            {
                    const_cast<char *>("getSuggestionsNative"),
                    const_cast<char *>("(JJLjava/lang/String;Ljava/lang/String;I[I[IF[Ljava/lang/String;[Ljava/lang/String;[F)V"),
                    reinterpret_cast<void *>(xlm_LanguageModel_getSuggestions)
            },
            {
                    const_cast<char *>("rescoreSuggestionsNative"),
                    const_cast<char *>("(JLjava/lang/String;[Ljava/lang/String;[I[I)V"),
                    reinterpret_cast<void *>(xlm_LanguageModel_rescoreSuggestions)
            }
    };

    static void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
        GGML_UNUSED(user_data);

        switch(level) {
            case GGML_LOG_LEVEL_ERROR:
                AKLOGE("llama err:  %s", text);
                break;
            case GGML_LOG_LEVEL_WARN:
                AKLOGI("llama warn: %s", text);
                break;
            case GGML_LOG_LEVEL_INFO:
                AKLOGI("llama info: %s", text);
                break;
        }
    }

    int register_LanguageModel(JNIEnv *env) {
        // The older ggml in this library (voice input, training) still expects this set-up
        llama_backend_init(true /* numa??? */);
        llama_log_set(llama_log_callback, nullptr);

        const char *const kClassPathName = "org/futo/inputmethod/latin/xlm/LanguageModel";
        return registerNativeMethods(env, kClassPathName, sMethods, NELEMS(sMethods));
    }
} // namespace latinime
