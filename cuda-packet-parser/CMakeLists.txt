cmake_minimum_required(VERSION 3.18)
project(CudaPacketParser LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (support Maxwell to Ada)
set(CMAKE_CUDA_ARCHITECTURES 50 61 75 86 89)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/pcap_reader.cpp
    src/batch_manager.cpp
    src/gpu_parser.cu
)

# Executable
add_executable(cuda_packet_parser ${SOURCES})

# Link libraries
target_link_libraries(cuda_packet_parser
    CUDA::cudart
    pcap
)

# Compiler options
target_compile_options(cuda_packet_parser PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        --use_fast_math
    >
)

# Add debug symbols for CUDA debugging
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
