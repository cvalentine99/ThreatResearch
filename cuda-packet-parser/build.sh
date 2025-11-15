#!/bin/bash

set -e

echo "=== CUDA Packet Parser Build Script ==="

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA Toolkit."
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "CUDA Version:"
nvcc --version | grep "release"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found. Please install CMake 3.18+."
    exit 1
fi

echo "CMake Version:"
cmake --version | head -1

# Check for libpcap
if ! ldconfig -p | grep -q libpcap; then
    echo "Error: libpcap not found."
    echo "Install with: sudo apt-get install libpcap-dev"
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Configure
echo ""
echo "=== Configuring with CMake ==="
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "=== Building ==="
make -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Executable: build/cuda_packet_parser"
echo ""
echo "Usage: ./cuda_packet_parser <pcap_file> [batch_size]"
echo ""
