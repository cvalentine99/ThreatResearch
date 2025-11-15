#pragma once

#include <cstdint>
#include <cstddef>

// Configuration constants
constexpr uint32_t DEFAULT_BATCH_SIZE = 100000;  // 100K packets per batch
constexpr uint32_t MAX_PACKET_SIZE = 65535;      // Max IP packet size
constexpr size_t PCAP_GLOBAL_HEADER_SIZE = 24;
constexpr size_t PCAP_PACKET_HEADER_SIZE = 16;

// PCAP magic numbers
constexpr uint32_t PCAP_MAGIC_LE = 0xa1b2c3d4;
constexpr uint32_t PCAP_MAGIC_BE = 0xd4c3b2a1;
constexpr uint32_t PCAP_NSEC_MAGIC_LE = 0xa1b23c4d;
constexpr uint32_t PCAP_NSEC_MAGIC_BE = 0x4d3cb2a1;

// Data link types (from libpcap)
enum DataLinkType : uint16_t {
    DLT_NULL = 0,
    DLT_ETHERNET = 1,
    DLT_RAW = 101,
    DLT_LINUX_SLL = 113,
    DLT_IEEE802_11_RADIO = 127
};

// Packet metadata (CPU-side)
struct PacketMetadata {
    uint64_t timestamp_us;      // Timestamp in microseconds
    uint32_t packet_offset;     // Offset in batch buffer
    uint32_t packet_length;     // Captured length
    uint32_t original_length;   // Original length on wire
    uint32_t frame_number;      // Sequential frame number
} __attribute__((aligned(16)));

// Parsed packet structure (GPU output)
struct ParsedPacket {
    uint32_t src_ip;            // Source IPv4 address
    uint32_t dst_ip;            // Destination IPv4 address
    uint16_t src_port;          // Source port
    uint16_t dst_port;          // Destination port
    uint8_t protocol;           // IP protocol (6=TCP, 17=UDP)
    uint8_t tcp_flags;          // TCP flags byte
    uint16_t payload_offset;    // Offset to payload in packet
    uint32_t flow_hash;         // Flow hash for grouping
} __attribute__((aligned(16)));

// PCAP file header
struct PcapFileHeader {
    uint32_t magic_number;
    uint16_t version_major;
    uint16_t version_minor;
    int32_t thiszone;
    uint32_t sigfigs;
    uint32_t snaplen;
    uint32_t network;
} __attribute__((packed));

// PCAP packet header
struct PcapPacketHeader {
    uint32_t ts_sec;
    uint32_t ts_usec;
    uint32_t incl_len;
    uint32_t orig_len;
} __attribute__((packed));

// Statistics
struct ParserStats {
    uint64_t total_packets;
    uint64_t total_bytes;
    uint64_t packets_parsed;
    uint64_t packets_ipv4;
    uint64_t packets_tcp;
    uint64_t packets_udp;
    uint64_t parse_errors;
    double cpu_time_ms;
    double gpu_time_ms;
    double transfer_time_ms;
};
