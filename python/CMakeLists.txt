cmake_minimum_required(VERSION 3.18)

# ----------------------------
# Python / pybind11
# ----------------------------
# Expect pybind11_DIR passed from command line, but also allow normal find.
find_package(pybind11 REQUIRED)

# ----------------------------
# CUDA Toolkit (for cuRAND, cudart, etc.)
# ----------------------------
# This is the key fix for: undefined symbol: curandGenerateNormal
find_package(CUDAToolkit REQUIRED)

# If CMAKE_CUDA_ARCHITECTURES is empty, set a safe default for Jetson Orin (sm_87)
if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

# ----------------------------
# Build the extension
# ----------------------------
pybind11_add_module(rmppi_cuda MODULE
  ${CMAKE_CURRENT_LIST_DIR}/bindings/rmppi_cuda.cu
)

target_compile_features(rmppi_cuda PRIVATE cxx_std_17)

target_include_directories(rmppi_cuda PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

# Link required project deps + CUDA runtime libs explicitly
target_link_libraries(rmppi_cuda PRIVATE
  cnpy
  Eigen3::Eigen
  CUDA::cudart
  CUDA::curand
)

# Ensure CUDA compilation settings are applied
set_target_properties(rmppi_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# Place built module into build/lib so your earlier PYTHONPATH convention works
set_target_properties(rmppi_cuda PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
)

# ----------------------------
# Runtime loader stability (so import works from site-packages too)
# ----------------------------
# Add an RPATH/RUNPATH pointing at the Jetson CUDA lib directory so Python can resolve
# libcurand/libcudart at import time without relying on your shell env.
set_target_properties(rmppi_cuda PROPERTIES
  BUILD_RPATH "/usr/local/cuda/targets/aarch64-linux/lib"
  INSTALL_RPATH "/usr/local/cuda/targets/aarch64-linux/lib"
)

