#pragma once

#include "common.h"
#include <vector>

class BatchManager {
public:
    BatchManager(uint32_t batch_size = DEFAULT_BATCH_SIZE);
    ~BatchManager();

    // Allocate memory for batch processing
    void allocate_buffers();
    void free_buffers();

    // Accessors
    uint8_t* get_host_packet_buffer() { return host_packet_data_; }
    PacketMetadata* get_host_metadata() { return host_metadata_; }
    uint8_t* get_device_packet_buffer() { return device_packet_data_; }
    PacketMetadata* get_device_metadata() { return device_metadata_; }
    ParsedPacket* get_device_output() { return device_parsed_output_; }
    ParsedPacket* get_host_output() { return host_parsed_output_; }

    uint32_t get_batch_size() const { return batch_size_; }

private:
    uint32_t batch_size_;

    // Host buffers (pinned memory for fast transfer)
    uint8_t* host_packet_data_;
    PacketMetadata* host_metadata_;
    ParsedPacket* host_parsed_output_;

    // Device buffers
    uint8_t* device_packet_data_;
    PacketMetadata* device_metadata_;
    ParsedPacket* device_parsed_output_;

    // Buffer sizes
    size_t packet_buffer_size_;
    size_t metadata_buffer_size_;
    size_t output_buffer_size_;
};
