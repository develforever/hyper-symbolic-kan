cmake_minimum_required(VERSION 3.15...3.31)
project(hyper_symbolic_kan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Find OpenMP
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

# SIMD and Optimization compiler flags
if(MSVC)
    target_compile_options(_cpp_kernels PRIVATE
        /O2
        /arch:AVX2
        /fp:fast
    )
    if(OpenMP_CXX_FOUND)
        target_compile_options(_cpp_kernels PRIVATE /openmp)
    endif()
else()
    target_compile_options(_cpp_kernels PRIVATE
        -O3
        -ffast-math
    )
    # AVX2 flags on x86/x64
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(amd64)")
        target_compile_options(_cpp_kernels PRIVATE -mavx2 -mfma)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(_cpp_kernels PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Install module into src/cpp_kernels
install(TARGETS _cpp_kernels DESTINATION src/cpp_kernels)
