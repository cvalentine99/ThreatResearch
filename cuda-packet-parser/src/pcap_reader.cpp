#include "pcap_reader.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdexcept>
#include <cstring>
#include <iostream>

PcapReader::PcapReader(const std::string& filename)
    : fd_(-1)
    , mapped_data_(nullptr)
    , file_size_(0)
    , current_offset_(0)
    , little_endian_(true)
    , nanosecond_precision_(false)
    , datalink_type_(DLT_ETHERNET)
    , total_packets_(0)
    , total_bytes_(0)
{
    // Open file
    fd_ = open(filename.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open PCAP file: " + filename);
    }

    // Get file size
    struct stat st;
    if (fstat(fd_, &st) < 0) {
        close(fd_);
        throw std::runtime_error("Failed to stat PCAP file");
    }
    file_size_ = st.st_size;

    if (file_size_ < PCAP_GLOBAL_HEADER_SIZE) {
        close(fd_);
        throw std::runtime_error("File too small to be valid PCAP");
    }

    // Memory map the file for fast random access
    mapped_data_ = static_cast<uint8_t*>(
        mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0)
    );

    if (mapped_data_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("Failed to mmap PCAP file");
    }

    // Advise kernel we'll read sequentially
    madvise(mapped_data_, file_size_, MADV_SEQUENTIAL);

    // Parse global header
    read_global_header();

    std::cout << "PCAP file opened successfully:" << std::endl;
    std::cout << "  File size: " << file_size_ << " bytes" << std::endl;
    std::cout << "  Endianness: " << (little_endian_ ? "Little" : "Big") << std::endl;
    std::cout << "  Data link type: " << static_cast<int>(datalink_type_) << std::endl;
    std::cout << "  Precision: " << (nanosecond_precision_ ? "nanosecond" : "microsecond") << std::endl;
}

PcapReader::~PcapReader() {
    if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
        munmap(mapped_data_, file_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

void PcapReader::read_global_header() {
    if (file_size_ < PCAP_GLOBAL_HEADER_SIZE) {
        throw std::runtime_error("File too small for PCAP header");
    }

    uint32_t magic = *reinterpret_cast<uint32_t*>(mapped_data_);

    // Determine endianness and precision
    if (magic == PCAP_MAGIC_LE) {
        little_endian_ = true;
        nanosecond_precision_ = false;
    } else if (magic == PCAP_MAGIC_BE) {
        little_endian_ = false;
        nanosecond_precision_ = false;
    } else if (magic == PCAP_NSEC_MAGIC_LE) {
        little_endian_ = true;
        nanosecond_precision_ = true;
    } else if (magic == PCAP_NSEC_MAGIC_BE) {
        little_endian_ = false;
        nanosecond_precision_ = true;
    } else {
        throw std::runtime_error("Invalid PCAP magic number");
    }

    // Read data link type (offset 20)
    uint32_t network = to_uint32(mapped_data_ + 20);
    datalink_type_ = static_cast<DataLinkType>(network);

    // Start reading packets after global header
    current_offset_ = PCAP_GLOBAL_HEADER_SIZE;
}

bool PcapReader::read_batch(
    std::vector<PacketMetadata>& metadata,
    std::vector<uint8_t>& packet_data,
    uint32_t max_packets
) {
    metadata.clear();
    packet_data.clear();

    uint32_t packets_read = 0;

    while (packets_read < max_packets && current_offset_ + PCAP_PACKET_HEADER_SIZE <= file_size_) {
        // Read packet header
        const uint8_t* header_ptr = mapped_data_ + current_offset_;

        uint32_t ts_sec = to_uint32(header_ptr);
        uint32_t ts_subsec = to_uint32(header_ptr + 4);
        uint32_t incl_len = to_uint32(header_ptr + 8);
        uint32_t orig_len = to_uint32(header_ptr + 12);

        // Validate packet length
        if (incl_len > MAX_PACKET_SIZE || incl_len == 0) {
            std::cerr << "Warning: Invalid packet length " << incl_len
                      << " at offset " << current_offset_ << std::endl;
            break;
        }

        // Check if we have enough data
        if (current_offset_ + PCAP_PACKET_HEADER_SIZE + incl_len > file_size_) {
            std::cerr << "Warning: Truncated packet at end of file" << std::endl;
            break;
        }

        // Create metadata
        PacketMetadata meta;
        meta.frame_number = total_packets_ + packets_read;
        meta.packet_offset = packet_data.size();  // Offset in batch buffer
        meta.packet_length = incl_len;
        meta.original_length = orig_len;

        // Convert timestamp to microseconds
        if (nanosecond_precision_) {
            meta.timestamp_us = static_cast<uint64_t>(ts_sec) * 1000000ULL + ts_subsec / 1000ULL;
        } else {
            meta.timestamp_us = static_cast<uint64_t>(ts_sec) * 1000000ULL + ts_subsec;
        }

        metadata.push_back(meta);

        // Copy packet data
        const uint8_t* packet_ptr = mapped_data_ + current_offset_ + PCAP_PACKET_HEADER_SIZE;
        packet_data.insert(packet_data.end(), packet_ptr, packet_ptr + incl_len);

        // Move to next packet
        current_offset_ += PCAP_PACKET_HEADER_SIZE + incl_len;
        packets_read++;
        total_bytes_ += incl_len;
    }

    total_packets_ += packets_read;

    return packets_read > 0;
}

uint16_t PcapReader::to_uint16(const uint8_t* data) const {
    if (little_endian_) {
        return data[0] | (data[1] << 8);
    } else {
        return (data[0] << 8) | data[1];
    }
}

uint32_t PcapReader::to_uint32(const uint8_t* data) const {
    if (little_endian_) {
        return data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
    } else {
        return (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
    }
}
