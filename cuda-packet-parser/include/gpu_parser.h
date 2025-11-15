#pragma once

#include "common.h"
#include <cuda_runtime.h>

// GPU parser interface
class GpuParser {
public:
    GpuParser();
    ~GpuParser();

    // Parse a batch of packets on GPU
    void parse_batch(
        const uint8_t* device_packet_data,
        const PacketMetadata* device_metadata,
        ParsedPacket* device_output,
        uint32_t packet_count,
        DataLinkType datalink_type,
        cudaStream_t stream = 0
    );

    // Get last error
    cudaError_t get_last_error() const { return last_error_; }

private:
    cudaError_t last_error_;
};

// CUDA kernel declarations
extern "C" {
    void launch_parse_packets_kernel(
        const uint8_t* packet_data,
        const PacketMetadata* metadata,
        ParsedPacket* output,
        uint32_t packet_count,
        uint16_t datalink_type,
        cudaStream_t stream
    );
}
