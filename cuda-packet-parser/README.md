# CUDA Packet Parser - Proof of Concept

A high-performance PCAP file parser that leverages NVIDIA CUDA for parallel packet processing.

## Features

- **GPU-Accelerated Parsing**: Processes thousands of packets in parallel on NVIDIA GPUs
- **Layer 2-4 Support**: Parses Ethernet, IPv4, TCP, and UDP headers
- **Memory-Mapped I/O**: Efficient PCAP file reading using mmap
- **Pinned Memory**: Optimized host-to-device transfers using cudaHostAlloc
- **Flow Hashing**: Computes 5-tuple flow identifiers for packet grouping

## Requirements

- **CUDA Toolkit**: 11.4 or later
- **GPU**: NVIDIA GPU with Compute Capability 5.0+ (Maxwell architecture or newer)
  - Tested: GTX 1060, RTX 3060, RTX 4090
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Libraries**: libpcap-dev
- **Compiler**: g++ 7.0+ or clang 6.0+
- **CMake**: 3.18 or later

## Installation

### 1. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y cmake libpcap-dev

# Install CUDA Toolkit (if not already installed)
# Download from: https://developer.nvidia.com/cuda-downloads
```

### 2. Build

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Usage

### Basic Usage

```bash
./cuda_packet_parser <pcap_file>
```

### With Custom Batch Size

```bash
./cuda_packet_parser <pcap_file> 50000
```

### Example

```bash
# Download sample PCAP
wget https://wiki.wireshark.org/SampleCaptures?action=AttachFile&do=get&target=http.cap

# Run parser
./cuda_packet_parser http.cap
```

## Output

The parser displays:

1. **PCAP File Info**: Endianness, data link type, timestamp precision
2. **Sample Packets**: First 20 parsed packets with IP addresses, ports, protocols
3. **Statistics**:
   - Total packets processed
   - Packet type breakdown (IPv4, TCP, UDP)
   - Performance metrics (CPU time, GPU time, transfer time, speedup)

### Sample Output

```
=== CUDA Packet Parser - Proof of Concept ===
PCAP file: capture.pcap
Batch size: 100000 packets

PCAP file opened successfully:
  File size: 524288000 bytes
  Endianness: Little
  Data link type: 1
  Precision: microsecond

========== Sample Parsed Packets ==========
Packet 0: 192.168.1.100:49684 -> 93.184.216.34:80 [TCP] Flags: 0x2 Hash: 0x12ab34cd
Packet 1: 93.184.216.34:80 -> 192.168.1.100:49684 [TCP] Flags: 0x12 Hash: 0x56ef78ab
...

========== Parsing Statistics ==========
Total packets:    750000
Total bytes:      524288000
Packets parsed:   748523
  IPv4 packets:   748523
  TCP packets:    612034
  UDP packets:    136489
Parse errors:     1477

========== Performance ==========
CPU time:         1250.45 ms
GPU time:         45.32 ms
Transfer time:    32.18 ms
Total GPU time:   77.50 ms
Speedup:          16.13x
GPU throughput:   54.23 Gbps
======================================
```

## Architecture

```
┌─────────────────────────┐
│   PCAP File (mmap)      │
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│   CPU: Batch Manager    │
│   - Read packets        │
│   - Prepare metadata    │
└───────────┬─────────────┘
            │ PCIe Transfer
┌───────────▼─────────────┐
│   GPU: Parsing Kernel   │
│   - L2: Ethernet        │
│   - L3: IPv4            │
│   - L4: TCP/UDP         │
│   - Flow hashing        │
└───────────┬─────────────┘
            │ PCIe Transfer
┌───────────▼─────────────┐
│   CPU: Results          │
│   - Statistics          │
│   - Output              │
└─────────────────────────┘
```

## Performance Notes

- **Batch Size**: Default 100K packets. Larger = better GPU utilization, but more VRAM
- **GPU Memory**: ~170 MB per 100K packet batch
- **Expected Speedup**: 10-20x for typical network captures
- **Bottleneck**: PCIe transfer for small packets, GPU compute for large packets

## Supported Packet Types

### Data Link (Layer 2)
- ✅ Ethernet II (DLT_ETHERNET = 1)
- ✅ Raw IP (DLT_RAW = 101)
- ✅ Linux Cooked Capture (DLT_LINUX_SLL = 113)
- ✅ VLAN (802.1Q)

### Network (Layer 3)
- ✅ IPv4
- ❌ IPv6 (planned)
- ❌ ARP (ignored)

### Transport (Layer 4)
- ✅ TCP (with flags)
- ✅ UDP
- ⚠️ ICMP (parsed but limited info)

## Limitations (POC Phase)

- IPv4 only (no IPv6)
- No protocol classification (HTTP, DNS, etc.)
- No TCP stream reassembly
- No packet filtering
- Single GPU only

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
./cuda_packet_parser file.pcap 50000
```

### No CUDA-Capable Device

Check GPU:
```bash
nvidia-smi
```

### Compilation Errors

Ensure CUDA is in PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Development Roadmap

### Phase 1: POC (Current)
- [x] PCAP reading with mmap
- [x] GPU memory management
- [x] L2/L3/L4 parsing kernel
- [x] Basic benchmarking

### Phase 2: MVP
- [ ] Protocol classification (HTTP, DNS, SSH, etc.)
- [ ] Flow aggregation with Thrust
- [ ] JSON/CSV output
- [ ] Multi-stream pipeline

### Phase 3: Production
- [ ] IPv6 support
- [ ] TCP stream reassembly
- [ ] Multi-GPU support
- [ ] PCAPNG format

## License

This is a proof-of-concept implementation for educational and research purposes.

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [libpcap Format](https://wiki.wireshark.org/Development/LibpcapFileFormat)
- [NetworkMiner Source](https://www.netresec.com/?page=NetworkMiner)

## Contributing

This is a POC project. Feedback and suggestions welcome!
