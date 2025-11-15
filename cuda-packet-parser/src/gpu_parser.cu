#include "gpu_parser.h"
#include <cuda_runtime.h>
#include <cstdio>

// Device helper functions
__device__ __forceinline__ uint16_t read_uint16_be(const uint8_t* data) {
    return (data[0] << 8) | data[1];
}

__device__ __forceinline__ uint32_t read_uint32_be(const uint8_t* data) {
    return (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
}

// Jenkins hash for 5-tuple flow hashing
__device__ __forceinline__ uint32_t compute_flow_hash(
    uint32_t src_ip,
    uint32_t dst_ip,
    uint16_t src_port,
    uint16_t dst_port,
    uint8_t protocol
) {
    uint32_t hash = 0;
    hash += src_ip; hash += (hash << 10); hash ^= (hash >> 6);
    hash += dst_ip; hash += (hash << 10); hash ^= (hash >> 6);
    hash += ((uint32_t)src_port << 16) | dst_port;
    hash += (hash << 10); hash ^= (hash >> 6);
    hash += protocol; hash += (hash << 10); hash ^= (hash >> 6);
    hash += (hash << 3); hash ^= (hash >> 11); hash += (hash << 15);
    return hash;
}

// Main packet parsing kernel
__global__ void parse_packets_kernel(
    const uint8_t* packet_data,
    const PacketMetadata* metadata,
    ParsedPacket* output,
    uint32_t packet_count,
    uint16_t datalink_type
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= packet_count) return;

    // Get packet pointer and length
    const uint8_t* pkt = packet_data + metadata[tid].packet_offset;
    uint16_t pkt_len = metadata[tid].packet_length;

    // Initialize output
    ParsedPacket result = {0};
    uint16_t offset = 0;

    // === Layer 2: Ethernet or other data link ===
    if (datalink_type == 1) {  // DLT_ETHERNET
        if (pkt_len < 14) return;  // Too short

        uint16_t ethertype = read_uint16_be(pkt + 12);
        offset = 14;

        // Handle VLAN (802.1Q)
        if (ethertype == 0x8100) {
            if (pkt_len < 18) return;
            ethertype = read_uint16_be(pkt + 16);
            offset = 18;
        }

        // Only process IPv4 for now
        if (ethertype != 0x0800) {
            return;  // Skip non-IPv4 (IPv6, ARP, etc.)
        }
    }
    else if (datalink_type == 101) {  // DLT_RAW (starts with IP)
        offset = 0;
    }
    else if (datalink_type == 113) {  // DLT_LINUX_SLL (Linux cooked capture)
        if (pkt_len < 16) return;
        uint16_t ethertype = read_uint16_be(pkt + 14);
        offset = 16;
        if (ethertype != 0x0800) return;  // Not IPv4
    }
    else {
        return;  // Unsupported data link type
    }

    // === Layer 3: IPv4 ===
    if (pkt_len < offset + 20) return;  // Too short for IPv4 header

    const uint8_t* ip_hdr = pkt + offset;
    uint8_t version = (ip_hdr[0] >> 4) & 0x0F;

    if (version != 4) return;  // Not IPv4

    uint8_t ihl = (ip_hdr[0] & 0x0F) * 4;  // IP header length in bytes
    uint8_t protocol = ip_hdr[9];
    uint16_t total_length = read_uint16_be(ip_hdr + 2);

    // Extract IP addresses (already in network byte order)
    result.src_ip = read_uint32_be(ip_hdr + 12);
    result.dst_ip = read_uint32_be(ip_hdr + 16);
    result.protocol = protocol;

    offset += ihl;

    // === Layer 4: TCP/UDP ===
    if (protocol == 6) {  // TCP
        if (pkt_len < offset + 20) return;  // Too short for TCP header

        const uint8_t* tcp_hdr = pkt + offset;
        result.src_port = read_uint16_be(tcp_hdr);
        result.dst_port = read_uint16_be(tcp_hdr + 2);
        result.tcp_flags = tcp_hdr[13];  // Flags byte

        uint8_t data_offset = ((tcp_hdr[12] >> 4) & 0x0F) * 4;
        result.payload_offset = offset + data_offset;
    }
    else if (protocol == 17) {  // UDP
        if (pkt_len < offset + 8) return;  // Too short for UDP header

        const uint8_t* udp_hdr = pkt + offset;
        result.src_port = read_uint16_be(udp_hdr);
        result.dst_port = read_uint16_be(udp_hdr + 2);
        result.payload_offset = offset + 8;
    }
    else {
        // Other protocols (ICMP, etc.) - just record IP info
        result.src_port = 0;
        result.dst_port = 0;
        result.payload_offset = offset;
    }

    // Compute flow hash
    result.flow_hash = compute_flow_hash(
        result.src_ip,
        result.dst_ip,
        result.src_port,
        result.dst_port,
        result.protocol
    );

    // Write output
    output[tid] = result;
}

// Kernel launcher
extern "C" void launch_parse_packets_kernel(
    const uint8_t* packet_data,
    const PacketMetadata* metadata,
    ParsedPacket* output,
    uint32_t packet_count,
    uint16_t datalink_type,
    cudaStream_t stream
) {
    if (packet_count == 0) return;

    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int num_blocks = (packet_count + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    parse_packets_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        packet_data,
        metadata,
        output,
        packet_count,
        datalink_type
    );
}

// GpuParser class implementation
GpuParser::GpuParser() : last_error_(cudaSuccess) {
}

GpuParser::~GpuParser() {
}

void GpuParser::parse_batch(
    const uint8_t* device_packet_data,
    const PacketMetadata* device_metadata,
    ParsedPacket* device_output,
    uint32_t packet_count,
    DataLinkType datalink_type,
    cudaStream_t stream
) {
    launch_parse_packets_kernel(
        device_packet_data,
        device_metadata,
        device_output,
        packet_count,
        static_cast<uint16_t>(datalink_type),
        stream
    );

    last_error_ = cudaGetLastError();
}
