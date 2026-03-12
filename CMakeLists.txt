cmake_minimum_required(VERSION 3.25)
project(paircraft VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Library ──────────────────────────────────────────────────────────────────
add_library(paircraft
    src/bpe.cpp
)
target_include_directories(paircraft PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(WIN32)
    # No extra link deps on Windows (mmap uses Win32 API)
else()
    # POSIX mmap needs nothing extra; large-file support is default on 64-bit
endif()

# ── GTest (fetched at configure time) ────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Prevent GTest from overriding our compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ── CLI executable ───────────────────────────────────────────────────────────
add_executable(paircraft_cli
    tools/tokenize.cpp
)
target_link_libraries(paircraft_cli PRIVATE paircraft)

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
