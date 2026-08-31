cmake_minimum_required(VERSION 3.15...3.31)
project(hyper_symbolic_kan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
# HSKAN_ENABLE_AVX2 (audit C6): `-mavx2 -mfma` / `/arch:AVX2` used to be
# unconditional on x86-64, which means the module raised SIGILL on import on any
# CPU without AVX2. The default build is now the x86-64 baseline (SSE2); AVX2 is
# opt-in until runtime dispatch exists. Do NOT turn this on for a published
# wheel -- there is no `__builtin_cpu_supports` guard yet.
option(HSKAN_ENABLE_AVX2 "Compile the whole extension with AVX2/FMA (no runtime dispatch -- SIGILL on pre-AVX2 CPUs)" OFF)

# HSKAN_WERROR: warnings-as-errors on the extension target. On in CI; a distro
# or end-user build that hits a warning from a newer compiler can pass
# -DHSKAN_WERROR=OFF without patching this file.
option(HSKAN_WERROR "Treat compiler warnings on _cpp_kernels as errors" ON)

# Find Python & nanobind
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# nanobind detection
if(NOT TARGET nanobind)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_DIR
    RESULT_VARIABLE nanobind_RES
  )
  if(nanobind_RES EQUAL 0 AND EXISTS "${nanobind_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
  endif()
  find_package(nanobind CONFIG REQUIRED)
endif()

# Find OpenMP. Not required: AppleClang has no OpenMP runtime unless libomp is
# installed. Every OpenMP pragma in the kernels is emitted through the HS_OMP*
# macros (fast_kan_kernel.hpp), which expand to nothing when _OPENMP is not
# defined -- so a build without OpenMP is correct, single-threaded and
# warning-free under -Wunknown-pragmas.
find_package(OpenMP)

# Source files
set(CPP_KERNEL_SOURCES
    src/cpp_kernels/fast_kan_kernel.cpp
    src/cpp_kernels/fast_kan_bindings.cpp
)

# Build nanobind module
nanobind_add_module(_cpp_kernels
    ${CPP_KERNEL_SOURCES}
)

# Target include directories
target_include_directories(_cpp_kernels PRIVATE src/cpp_kernels)

# ---------------------------------------------------------------------------
# Warnings (audit C5)
# ---------------------------------------------------------------------------
# -Wunknown-pragmas is what makes the MSVC-only `#pragma loop(ivdep)` class of
# bug visible on GCC/Clang instead of being silently ignored.
if(MSVC)
    target_compile_options(_cpp_kernels PRIVATE /W4)
    if(HSKAN_WERROR)
        target_compile_options(_cpp_kernels PRIVATE /WX)
    endif()
else()
    target_compile_options(_cpp_kernels PRIVATE -Wall -Wextra -Wunknown-pragmas)
    if(HSKAN_WERROR)
        target_compile_options(_cpp_kernels PRIVATE -Werror)
    endif()
endif()

# ---------------------------------------------------------------------------
# Optimization / floating point / SIMD
# ---------------------------------------------------------------------------
if(MSVC)
    target_compile_options(_cpp_kernels PRIVATE
        /O2
        # IEEE-conformant FP semantics. /fp:fast was removed (audit C7): it
        # permits reassociation of the Chebyshev recurrence and assumes the
        # absence of NaN/Inf, which invalidates the `total_energy < 1e-14`
        # guards and NaN detection in the kernels.
        /fp:precise
        # FMA contraction only -- increases accuracy, keeps IEEE semantics.
        /fp:contract
    )
    if(HSKAN_ENABLE_AVX2)
        target_compile_options(_cpp_kernels PRIVATE /arch:AVX2)
    endif()
    if(OpenMP_CXX_FOUND)
        target_compile_options(_cpp_kernels PRIVATE /openmp)
    endif()
else()
    target_compile_options(_cpp_kernels PRIVATE
        -O3
        # -ffast-math was removed (audit C7): it permits reassociation of the
        # Chebyshev recurrence and assumes the absence of NaN/Inf, which
        # invalidates the `total_energy < 1e-14` guards and NaN detection in
        # the kernels. Only FMA contraction is kept -- it is IEEE-safe and
        # increases accuracy.
        -ffp-contract=fast
    )
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(amd64)|(i[3-6]86)")
        if(HSKAN_ENABLE_AVX2)
            target_compile_options(_cpp_kernels PRIVATE -mavx2 -mfma)
        else()
            # Explicit, so the baseline claim is provable from the compile line
            # in the CI log, not just from the absence of -mavx2.
            target_compile_options(_cpp_kernels PRIVATE -mno-avx2 -mno-fma)
        endif()
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(_cpp_kernels PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Install module into src/cpp_kernels
install(TARGETS _cpp_kernels DESTINATION src/cpp_kernels)
