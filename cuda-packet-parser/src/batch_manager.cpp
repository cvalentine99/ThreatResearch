#include "batch_manager.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(err) \
            ); \
        } \
    } while(0)

BatchManager::BatchManager(uint32_t batch_size)
    : batch_size_(batch_size)
    , host_packet_data_(nullptr)
    , host_metadata_(nullptr)
    , host_parsed_output_(nullptr)
    , device_packet_data_(nullptr)
    , device_metadata_(nullptr)
    , device_parsed_output_(nullptr)
{
    // Calculate buffer sizes
    // Assume average packet size of 1500 bytes (Ethernet MTU)
    packet_buffer_size_ = batch_size_ * 1500;
    metadata_buffer_size_ = batch_size_ * sizeof(PacketMetadata);
    output_buffer_size_ = batch_size_ * sizeof(ParsedPacket);

    std::cout << "BatchManager initialized:" << std::endl;
    std::cout << "  Batch size: " << batch_size_ << " packets" << std::endl;
    std::cout << "  Packet buffer: " << packet_buffer_size_ / (1024*1024) << " MB" << std::endl;
    std::cout << "  Metadata buffer: " << metadata_buffer_size_ / 1024 << " KB" << std::endl;
    std::cout << "  Output buffer: " << output_buffer_size_ / 1024 << " KB" << std::endl;
}

BatchManager::~BatchManager() {
    free_buffers();
}

void BatchManager::allocate_buffers() {
    std::cout << "Allocating GPU buffers..." << std::endl;

    // Allocate pinned host memory for fast transfers
    CUDA_CHECK(cudaHostAlloc(
        (void**)&host_packet_data_,
        packet_buffer_size_,
        cudaHostAllocDefault
    ));

    CUDA_CHECK(cudaHostAlloc(
        (void**)&host_metadata_,
        metadata_buffer_size_,
        cudaHostAllocDefault
    ));

    CUDA_CHECK(cudaHostAlloc(
        (void**)&host_parsed_output_,
        output_buffer_size_,
        cudaHostAllocDefault
    ));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(
        (void**)&device_packet_data_,
        packet_buffer_size_
    ));

    CUDA_CHECK(cudaMalloc(
        (void**)&device_metadata_,
        metadata_buffer_size_
    ));

    CUDA_CHECK(cudaMalloc(
        (void**)&device_parsed_output_,
        output_buffer_size_
    ));

    std::cout << "GPU buffers allocated successfully" << std::endl;

    // Query GPU memory usage
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << (total_mem - free_mem) / (1024*1024)
              << " / " << total_mem / (1024*1024) << " MB used" << std::endl;
}

void BatchManager::free_buffers() {
    if (host_packet_data_) {
        cudaFreeHost(host_packet_data_);
        host_packet_data_ = nullptr;
    }
    if (host_metadata_) {
        cudaFreeHost(host_metadata_);
        host_metadata_ = nullptr;
    }
    if (host_parsed_output_) {
        cudaFreeHost(host_parsed_output_);
        host_parsed_output_ = nullptr;
    }
    if (device_packet_data_) {
        cudaFree(device_packet_data_);
        device_packet_data_ = nullptr;
    }
    if (device_metadata_) {
        cudaFree(device_metadata_);
        device_metadata_ = nullptr;
    }
    if (device_parsed_output_) {
        cudaFree(device_parsed_output_);
        device_parsed_output_ = nullptr;
    }
}
