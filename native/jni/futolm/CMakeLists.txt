# libfutolm: the keyboard's transformer language model on a current llama.cpp (see futolm.h).
#
# llama.cpp and ggml are linked in statically with hidden symbols, so they can't clash with the
# older ggml in libjni_latinime.so (voice input, training). Only the futolm_* functions are exported.
#
#   FUTOLM_SHARED=ON   libfutolm.so for the app (default)
#   FUTOLM_SHARED=OFF  static library, for tools such as tools/lmbench
cmake_minimum_required(VERSION 3.22.1)

option(FUTOLM_SHARED "Build libfutolm as a shared library" ON)

# The keyboard's warning options are meant for its own code, not for llama.cpp
set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "")

# The language model runs on every keystroke: always optimise, also in debug builds of the app
set(CMAKE_C_FLAGS_DEBUG "-O3 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -g -DNDEBUG")

set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BUILD_SHARED_LIBS OFF)
set(LLAMA_BUILD_COMMON OFF)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_TOOLS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_SERVER OFF)
set(LLAMA_BUILD_APP OFF)
set(LLAMA_OPENSSL OFF)
set(GGML_OPENMP OFF)
set(GGML_NATIVE OFF)
set(GGML_BACKEND_DL OFF)
if (ANDROID_ABI STREQUAL "arm64-v8a")
    # Dot product and fp16 instructions (Cortex-A55/A75 and newer, 2017+); futolm_open checks the
    # CPU before running anything
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "ggml: CPU architecture for ARM")
endif()

if (ANDROID_ABI STREQUAL "armeabi-v7a")
    # llamafile's matrix kernels use fp16 vector instructions that 32-bit ARM lacks
    set(GGML_LLAMAFILE OFF)
endif()

add_subdirectory(llama.cpp EXCLUDE_FROM_ALL)

# FUTO's SentencePiece (the tokenizer of FUTO's own models) and the protobuf-lite it needs
set(JNI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(SPM_SOURCES)
foreach(f arena arenastring bytestream coded_stream common extension_set generated_enum_util
        generated_message_table_driven_lite generated_message_util implicit_weak_message int128
        io_win32 message_lite parse_context repeated_field status statusor stringpiece stringprintf
        structurally_valid strutil time wire_format_lite zero_copy_stream zero_copy_stream_impl
        zero_copy_stream_impl_lite)
    list(APPEND SPM_SOURCES ${JNI_DIR}/src/third_party/protobuf-lite/${f}.cc)
endforeach()
foreach(f builtin_pb/sentencepiece_model.pb builtin_pb/sentencepiece.pb bpe_model char_model error
        filesystem model_factory model_interface normalizer sentencepiece_processor unigram_model util
        word_model)
    list(APPEND SPM_SOURCES ${JNI_DIR}/src/sentencepiece/${f}.cc)
endforeach()

if (FUTOLM_SHARED)
    add_library(futolm SHARED futolm.cpp ${SPM_SOURCES})
    target_link_options(futolm PRIVATE -Wl,--exclude-libs,ALL -Wl,--gc-sections)
else()
    add_library(futolm STATIC futolm.cpp ${SPM_SOURCES})
endif()

target_compile_features(futolm PRIVATE cxx_std_17)
target_include_directories(futolm
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE ${JNI_DIR}/src
                ${JNI_DIR}/src/sentencepiece/builtin_pb
                ${JNI_DIR}/src/third_party
                ${JNI_DIR}/src/third_party/protobuf-lite
                ${JNI_DIR}/src/third_party/esaxx
                ${JNI_DIR}/src/third_party/darts_clone
                ${JNI_DIR}/src/third_party/absl)
target_compile_options(futolm PRIVATE -fexceptions -ffunction-sections -fdata-sections)
target_compile_definitions(futolm PRIVATE HAVE_PTHREAD)
target_link_libraries(futolm PRIVATE llama ggml)
if (GGML_CPU_ARM_ARCH MATCHES "dotprod")
    target_compile_definitions(futolm PRIVATE FUTOLM_NEEDS_DOTPROD)
endif()
if (GGML_CPU_ARM_ARCH MATCHES "fp16")
    target_compile_definitions(futolm PRIVATE FUTOLM_NEEDS_FP16)
endif()
if (ANDROID)
    target_link_libraries(futolm PRIVATE log)
endif()
