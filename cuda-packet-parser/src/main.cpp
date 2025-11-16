#include "pcap_reader.h"
#include "batch_manager.h"
#include "gpu_parser.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <arpa/inet.h>
#include <vector>

void print_stats(const ParserStats& stats) {
    std::cout << "\n========== Parsing Statistics ==========" << std::endl;
    std::cout << "Total packets:    " << stats.total_packets << std::endl;
    std::cout << "Total bytes:      " << stats.total_bytes << std::endl;
    std::cout << "Packets parsed:   " << stats.packets_parsed << std::endl;
    std::cout << "  IPv4 packets:   " << stats.packets_ipv4 << std::endl;
    std::cout << "  TCP packets:    " << stats.packets_tcp << std::endl;
    std::cout << "  UDP packets:    " << stats.packets_udp << std::endl;
    std::cout << "Parse errors:     " << stats.parse_errors << std::endl;
    std::cout << "\n========== Performance ==========" << std::endl;
    std::cout << "CPU time:         " << std::fixed << std::setprecision(2)
              << stats.cpu_time_ms << " ms" << std::endl;
    std::cout << "GPU time:         " << stats.gpu_time_ms << " ms" << std::endl;
    std::cout << "Transfer time:    " << stats.transfer_time_ms << " ms" << std::endl;
    std::cout << "Total GPU time:   " << (stats.gpu_time_ms + stats.transfer_time_ms) << " ms" << std::endl;

    if (stats.gpu_time_ms + stats.transfer_time_ms > 0) {
        double speedup = stats.cpu_time_ms / (stats.gpu_time_ms + stats.transfer_time_ms);
        std::cout << "Speedup:          " << std::setprecision(2) << speedup << "x" << std::endl;
    }

    double throughput_gbps = (stats.total_bytes * 8.0) / (stats.gpu_time_ms * 1000000.0);
    std::cout << "GPU throughput:   " << std::setprecision(2) << throughput_gbps << " Gbps" << std::endl;
    std::cout << "======================================\n" << std::endl;
}

std::string ip_to_string(uint32_t ip) {
    struct in_addr addr;
    addr.s_addr = htonl(ip);
    return inet_ntoa(addr);
}

const char* protocol_to_string(uint8_t proto) {
    switch(proto) {
        case 1: return "ICMP";
        case 6: return "TCP";
        case 17: return "UDP";
        default: return "Other";
    }
}

void print_sample_packets(const ParsedPacket* packets, uint32_t count, uint32_t max_samples = 10) {
    std::cout << "\n========== Sample Parsed Packets ==========" << std::endl;
    uint32_t samples = std::min(count, max_samples);

    for (uint32_t i = 0; i < samples; i++) {
        const auto& pkt = packets[i];
        if (pkt.src_ip == 0) continue;  // Skip unparsed packets

        std::cout << "Packet " << i << ": "
                  << ip_to_string(pkt.src_ip) << ":" << pkt.src_port
                  << " -> "
                  << ip_to_string(pkt.dst_ip) << ":" << pkt.dst_port
                  << " [" << protocol_to_string(pkt.protocol) << "]";

        if (pkt.protocol == 6) {
            std::cout << " Flags: 0x" << std::hex << static_cast<int>(pkt.tcp_flags) << std::dec;
        }

        std::cout << " Hash: 0x" << std::hex << pkt.flow_hash << std::dec << std::endl;
    }
    std::cout << "==========================================\n" << std::endl;
}

void write_packets_json(const std::string& output_file, const std::vector<ParsedPacket>& all_packets) {
    std::cout << "\nWriting " << all_packets.size() << " packets to JSON file: " << output_file << std::endl;

    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }

    out << "{\"packets\":[";

    bool first = true;
    for (const auto& pkt : all_packets) {
        if (pkt.src_ip == 0) continue;  // Skip unparsed packets

        if (!first) out << ",";
        first = false;

        out << "\n  {";
        out << "\"src_ip\":\"" << ip_to_string(pkt.src_ip) << "\",";
        out << "\"src_port\":" << pkt.src_port << ",";
        out << "\"dst_ip\":\"" << ip_to_string(pkt.dst_ip) << "\",";
        out << "\"dst_port\":" << pkt.dst_port << ",";
        out << "\"protocol\":\"" << protocol_to_string(pkt.protocol) << "\",";
        out << "\"protocol_num\":" << static_cast<int>(pkt.protocol) << ",";

        if (pkt.protocol == 6) {
            out << "\"tcp_flags\":\"0x" << std::hex << static_cast<int>(pkt.tcp_flags) << std::dec << "\",";
        } else {
            out << "\"tcp_flags\":null,";
        }

        out << "\"flow_hash\":\"0x" << std::hex << pkt.flow_hash << std::dec << "\",";
        out << "\"payload_offset\":" << pkt.payload_offset;
        out << "}";
    }

    out << "\n],\"total\":" << all_packets.size() << "}\n";
    out.close();

    std::cout << "JSON output complete!" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <pcap_file> [batch_size]" << std::endl;
        return 1;
    }

    std::string pcap_file = argv[1];
    uint32_t batch_size = DEFAULT_BATCH_SIZE;

    if (argc >= 3) {
        batch_size = std::atoi(argv[2]);
    }

    try {
        std::cout << "=== CUDA Packet Parser - Proof of Concept ===" << std::endl;
        std::cout << "PCAP file: " << pcap_file << std::endl;
        std::cout << "Batch size: " << batch_size << " packets\n" << std::endl;

        // Initialize components
        PcapReader reader(pcap_file);
        BatchManager batch_mgr(batch_size);
        GpuParser gpu_parser;

        // Allocate GPU buffers
        batch_mgr.allocate_buffers();

        // Statistics
        ParserStats stats = {0};

        // Storage for all parsed packets
        std::vector<ParsedPacket> all_parsed_packets;
        all_parsed_packets.reserve(1000000);  // Pre-allocate for performance

        // Temporary CPU storage
        std::vector<PacketMetadata> cpu_metadata;
        std::vector<uint8_t> cpu_packet_data;

        // CUDA events for timing
        cudaEvent_t start_gpu, stop_gpu, start_transfer, stop_transfer;
        cudaEventCreate(&start_gpu);
        cudaEventCreate(&stop_gpu);
        cudaEventCreate(&start_transfer);
        cudaEventCreate(&stop_transfer);

        // Process batches
        uint32_t batch_num = 0;
        auto cpu_start = std::chrono::high_resolution_clock::now();

        while (reader.read_batch(cpu_metadata, cpu_packet_data, batch_size)) {
            batch_num++;
            uint32_t packet_count = cpu_metadata.size();

            std::cout << "Processing batch " << batch_num
                      << " (" << packet_count << " packets, "
                      << cpu_packet_data.size() / 1024 << " KB)" << std::endl;

            // Copy to pinned host buffers
            memcpy(batch_mgr.get_host_metadata(), cpu_metadata.data(),
                   packet_count * sizeof(PacketMetadata));
            memcpy(batch_mgr.get_host_packet_buffer(), cpu_packet_data.data(),
                   cpu_packet_data.size());

            // Transfer to GPU
            cudaEventRecord(start_transfer);

            cudaMemcpy(batch_mgr.get_device_metadata(),
                      batch_mgr.get_host_metadata(),
                      packet_count * sizeof(PacketMetadata),
                      cudaMemcpyHostToDevice);

            cudaMemcpy(batch_mgr.get_device_packet_buffer(),
                      batch_mgr.get_host_packet_buffer(),
                      cpu_packet_data.size(),
                      cudaMemcpyHostToDevice);

            cudaEventRecord(stop_transfer);

            // Parse on GPU
            cudaEventRecord(start_gpu);

            gpu_parser.parse_batch(
                batch_mgr.get_device_packet_buffer(),
                batch_mgr.get_device_metadata(),
                batch_mgr.get_device_output(),
                packet_count,
                reader.get_datalink_type()
            );

            cudaEventRecord(stop_gpu);
            cudaEventSynchronize(stop_gpu);

            // Check for errors
            if (gpu_parser.get_last_error() != cudaSuccess) {
                std::cerr << "GPU parsing error!" << std::endl;
                return 1;
            }

            // Transfer results back
            cudaMemcpy(batch_mgr.get_host_output(),
                      batch_mgr.get_device_output(),
                      packet_count * sizeof(ParsedPacket),
                      cudaMemcpyDeviceToHost);

            // Calculate timing
            float gpu_ms, transfer_ms;
            cudaEventElapsedTime(&gpu_ms, start_gpu, stop_gpu);
            cudaEventElapsedTime(&transfer_ms, start_transfer, stop_transfer);

            stats.gpu_time_ms += gpu_ms;
            stats.transfer_time_ms += transfer_ms;
            stats.total_packets += packet_count;

            // Analyze results and store parsed packets
            for (uint32_t i = 0; i < packet_count; i++) {
                const auto& pkt = batch_mgr.get_host_output()[i];
                if (pkt.src_ip != 0) {
                    stats.packets_parsed++;
                    stats.packets_ipv4++;
                    if (pkt.protocol == 6) stats.packets_tcp++;
                    if (pkt.protocol == 17) stats.packets_udp++;

                    // Store parsed packet
                    all_parsed_packets.push_back(pkt);
                }
            }

            // Print sample from first batch
            if (batch_num == 1) {
                print_sample_packets(batch_mgr.get_host_output(), packet_count, 20);
            }
        }

        auto cpu_end = std::chrono::high_resolution_clock::now();
        stats.cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        stats.total_bytes = reader.get_total_bytes();

        // Print statistics
        print_stats(stats);

        // Write parsed packets to JSON file
        std::string json_output = pcap_file + ".json";
        write_packets_json(json_output, all_parsed_packets);

        // Cleanup
        cudaEventDestroy(start_gpu);
        cudaEventDestroy(stop_gpu);
        cudaEventDestroy(start_transfer);
        cudaEventDestroy(stop_transfer);

        std::cout << "Processing complete!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
