# lmbench2: benchmark for libfutolm (the keyboard's language model on a current llama.cpp).
# Built by build.sh futolm-android|futolm-host; lmbench (the old engine) is built by build.sh directly.
cmake_minimum_required(VERSION 3.22.1)
project(lmbench2 C CXX)

set(FUTOLM_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(../../native/jni/futolm futolm)

add_executable(lmbench2 lmbench2.cpp)
target_compile_features(lmbench2 PRIVATE cxx_std_17)
target_link_libraries(lmbench2 PRIVATE futolm)
if (ANDROID)
    target_link_options(lmbench2 PRIVATE -static-libstdc++)
endif()
