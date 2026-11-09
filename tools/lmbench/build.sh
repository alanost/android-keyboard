#!/usr/bin/env bash
# build.sh <android|android-dotprod|host|futolm-android|futolm-host> <build-dir>
# Builds lmbench from FUTO's native sources (the same file list and flags the app uses).
#   android         NDK 28.2, arm64, flags as in native/jni/CMakeLists.txt (what the app ships)
#   android-dotprod same plus -march=armv8.2-a+dotprod+fp16 (what the phone's CPU could use)
#   host            this machine, for quality comparisons of models
# or lmbench2, the same benchmark on libfutolm (the new engine, current llama.cpp):
#   futolm-android  NDK 28.2, arm64, built like the app's libfutolm.so
#   futolm-host     this machine, only for quick functional checks
set -euo pipefail
variant="$1"; out="$2"
here="$(cd "$(dirname "$0")" && pwd)"
J="$(cd "$here/../../native/jni" && pwd)"
NDK="${ANDROID_NDK:-$HOME/tools/android-sdk/ndk/28.2.13676358}"
mkdir -p "$out"

case "$variant" in
    futolm-android)
        cmake -S "$here" -B "$out" -G Ninja -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
            -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30
        cmake --build "$out" --target lmbench2
        echo "built $out/lmbench2"; exit 0;;
    futolm-host)
        cmake -S "$here" -B "$out" -G Ninja -DCMAKE_BUILD_TYPE=Release
        cmake --build "$out" --target lmbench2
        echo "built $out/lmbench2"; exit 0;;
esac

# Core sources exactly as listed for the app's static library
sed -n '/set(LATIN_IME_CORE_SRC_FILES/,/)/p' "$J/NativeFileList.cmake" \
    | grep -o '\${CMAKE_CURRENT_SOURCE_DIR}/[^ )]*' | sed "s|\${CMAKE_CURRENT_SOURCE_DIR}|$J|" > "$out/sources.txt"

INC="-I$J -I$J/src -I$J/src/sentencepiece/builtin_pb -I$J/src/third_party -I$J/src/third_party/protobuf-lite \
-I$J/src/third_party/esaxx -I$J/src/third_party/darts_clone -I$J/src/third_party/absl"
FLAGS="-O3 -DNDEBUG -fexceptions -DHAVE_PTHREAD -w"; PRE=""
case "$variant" in
    android|android-dotprod)
        BIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
        CC="$BIN/aarch64-linux-android30-clang"; CXX="$BIN/aarch64-linux-android30-clang++"
        FLAGS="$FLAGS -fPIE"
        [ "$variant" = android-dotprod ] && FLAGS="$FLAGS -march=armv8.2-a+dotprod+fp16"
        LIBS="-static-libstdc++ -pie -llog -ldl -lm";;
    host)
        CC=gcc; CXX=g++
        JH="${JAVA_HOME:-$HOME/tools/jdk17}"
        INC="$INC -I$JH/include -I$JH/include/linux"
        FLAGS="$FLAGS -march=native"
        LIBS="-lpthread -ldl -lm"
        PRE="-include cstring -include math.h";;   # jni_utils.cpp relies on <cstring> arriving indirectly (true with the NDK, not with gcc)
    *) echo "unknown variant $variant"; exit 2;;
esac

{
    echo "OBJS :="
    while read -r src; do
        obj="$out/obj/$(echo "${src#$J/}" | tr '/' '_').o"
        echo "OBJS += $obj"
        case "$src" in
            *.c) echo "$obj: $src"; echo "	@mkdir -p $out/obj; $CC $FLAGS $INC -c $< -o \$@";;
            *)   echo "$obj: $src"; echo "	@mkdir -p $out/obj; $CXX -std=c++17 $FLAGS $INC -c $< -o \$@";;
        esac
    done < "$out/sources.txt"
    echo "$out/lmbench: \$(OBJS) $here/lmbench.cpp $J/src/jni_utils.cpp $J/org_futo_inputmethod_latin_xlm_LanguageModel.cpp"
    echo "	$CXX -std=c++17 $FLAGS $PRE $INC $here/lmbench.cpp $J/src/jni_utils.cpp \$(OBJS) -o \$@ $LIBS"
} > "$out/Makefile"
make -s -f "$out/Makefile" -j"$(nproc)" "$out/lmbench"
echo "built $out/lmbench"
