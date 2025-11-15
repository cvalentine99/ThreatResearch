#pragma once

#include "common.h"
#include <string>
#include <vector>
#include <memory>

class PcapReader {
public:
    PcapReader(const std::string& filename);
    ~PcapReader();

    // Read next batch of packets
    bool read_batch(std::vector<PacketMetadata>& metadata,
                    std::vector<uint8_t>& packet_data,
                    uint32_t max_packets);

    // Get PCAP file information
    DataLinkType get_datalink_type() const { return datalink_type_; }
    uint64_t get_total_packets() const { return total_packets_; }
    uint64_t get_total_bytes() const { return total_bytes_; }
    bool is_little_endian() const { return little_endian_; }
    bool is_nanosecond_precision() const { return nanosecond_precision_; }

private:
    void read_global_header();
    uint16_t to_uint16(const uint8_t* data) const;
    uint32_t to_uint32(const uint8_t* data) const;

    int fd_;                        // File descriptor
    uint8_t* mapped_data_;          // Memory-mapped file
    size_t file_size_;              // Total file size
    size_t current_offset_;         // Current read position

    bool little_endian_;
    bool nanosecond_precision_;
    DataLinkType datalink_type_;
    uint64_t total_packets_;
    uint64_t total_bytes_;
};
