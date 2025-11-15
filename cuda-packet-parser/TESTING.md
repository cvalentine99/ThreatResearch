# Testing Guide

## Prerequisites

Before testing, ensure you have:

```bash
# CUDA installed and working
nvidia-smi
nvcc --version

# Project built
./build.sh
```

## Quick Validation Test

### 1. Download Sample Data

```bash
./download_samples.sh
```

### 2. Run Basic Test

```bash
# Test with small HTTP capture
./build/cuda_packet_parser data/http.cap
```

**Expected Output:**
- Should parse ~40-50 packets
- Show sample parsed packets with IPs and ports
- Display performance statistics
- Speedup should be 5-15x

### 3. Validate Correctness

Compare with tcpdump:

```bash
# Our parser
./build/cuda_packet_parser data/http.cap 2>&1 | grep "Packet 0"

# tcpdump reference
tcpdump -r data/http.cap -n -c 1
```

The IP addresses and ports should match!

## Comprehensive Tests

### Test 1: Different Batch Sizes

```bash
# Small batch
./build/cuda_packet_parser data/http.cap 10

# Medium batch
./build/cuda_packet_parser data/http.cap 100

# Large batch
./build/cuda_packet_parser data/http.cap 10000
```

**Validation:**
- All should produce same packet count
- Larger batches = better GPU utilization
- Very small batches may be slower (overhead)

### Test 2: Data Link Types

```bash
# Ethernet (DLT=1) - most common
./build/cuda_packet_parser data/http.cap

# If you have Raw IP captures (DLT=101)
./build/cuda_packet_parser data/raw_ip.pcap

# Linux SLL (DLT=113) from tcpdump -i any
./build/cuda_packet_parser data/linux_sll.pcap
```

### Test 3: Protocol Mix

Test with captures containing different protocols:

```bash
# HTTP traffic
./build/cuda_packet_parser data/http.cap

# DNS queries
./build/cuda_packet_parser data/dns.cap

# TCP handshake
./build/cuda_packet_parser data/tcp-3way-handshake.pcap
```

**Validation:**
- Check "TCP packets" vs "UDP packets" in stats
- DNS should show UDP packets on port 53
- HTTP should show TCP packets on port 80

### Test 4: Large File Performance

Generate a large synthetic PCAP:

```bash
# Method 1: Merge existing captures
mergecap -w large.pcap data/*.cap

# Method 2: Use tcpreplay to capture live traffic
sudo tcpdump -i eth0 -w large.pcap -c 100000
```

Then test:

```bash
time ./build/cuda_packet_parser large.pcap
```

**Expected Performance:**
- >100 MB file: 15-20x speedup
- >1 GB file: 18-25x speedup

### Test 5: Memory Stress Test

Test with maximum batch size for your GPU:

```bash
# 4GB GPU
./build/cuda_packet_parser large.pcap 250000

# 8GB GPU
./build/cuda_packet_parser large.pcap 500000

# 12GB GPU
./build/cuda_packet_parser large.pcap 1000000
```

Monitor with:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

**Validation:**
- Should not crash with OOM
- GPU memory usage should peak then release
- If OOM occurs, reduce batch size

## Correctness Validation

### Manual Verification

Compare key metrics with tcpdump:

```bash
# Total packets
tcpdump -r data/http.cap | wc -l
# Should match "Total packets" in our output

# TCP packets
tcpdump -r data/http.cap tcp | wc -l
# Should match "TCP packets"

# UDP packets
tcpdump -r data/http.cap udp | wc -l
# Should match "UDP packets"
```

### Detailed Packet Comparison

```bash
# First packet details
tcpdump -r data/http.cap -n -c 1 -v

# Compare with our "Packet 0" output
./build/cuda_packet_parser data/http.cap | grep "Packet 0"
```

**Check:**
- ✅ Source IP matches
- ✅ Destination IP matches
- ✅ Source port matches
- ✅ Destination port matches
- ✅ Protocol (TCP/UDP) matches

## Performance Benchmarking

### Baseline CPU Comparison

Time CPU-only parsing with tcpdump:

```bash
# tcpdump (CPU baseline)
time tcpdump -r large.pcap -n > /dev/null

# Our GPU parser
time ./build/cuda_packet_parser large.pcap
```

### Profiling with NVIDIA Tools

```bash
# Profile with nvprof (CUDA 10.x and earlier)
nvprof ./build/cuda_packet_parser data/http.cap

# Profile with Nsight Systems (CUDA 11+)
nsys profile ./build/cuda_packet_parser data/http.cap
```

**Look for:**
- Kernel execution time
- Memory transfer time
- GPU utilization percentage
- Warp efficiency

### Throughput Calculation

```
Throughput (Gbps) = (Total Bytes × 8) / (GPU Time in seconds)

Example:
- File: 500 MB
- GPU time: 0.05 seconds
- Throughput = (500 × 8) / 0.05 = 80 Gbps
```

## Common Issues & Solutions

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Reduce batch size
./build/cuda_packet_parser file.pcap 50000
```

### Issue: No speedup observed

**Possible causes:**
1. File too small (<10 MB) - overhead dominates
2. Old GPU (<Maxwell architecture)
3. PCIe bottleneck - check PCIe version

**Debug:**
```bash
# Check GPU info
nvidia-smi -q | grep -E "Name|Memory|PCIe"
```

### Issue: Incorrect packet counts

**Possible causes:**
1. Truncated PCAP file
2. Unsupported data link type
3. Malformed packets

**Debug:**
```bash
# Validate PCAP
capinfos data/http.cap

# Check for errors
./build/cuda_packet_parser data/http.cap 2>&1 | grep -i error
```

### Issue: Compile errors

**Solution:**
```bash
# Clean build
rm -rf build
./build.sh

# Check CUDA installation
nvcc --version
cmake --version
```

## Regression Tests

Create a simple test script:

```bash
#!/bin/bash
# test_regression.sh

echo "Running regression tests..."

# Test 1: HTTP capture
OUTPUT=$(./build/cuda_packet_parser data/http.cap 2>&1)
if echo "$OUTPUT" | grep -q "Speedup.*[0-9]"; then
    echo "✅ Test 1 passed: HTTP capture"
else
    echo "❌ Test 1 failed"
    exit 1
fi

# Test 2: Batch processing
OUTPUT=$(./build/cuda_packet_parser data/http.cap 100 2>&1)
if echo "$OUTPUT" | grep -q "Total packets"; then
    echo "✅ Test 2 passed: Batch processing"
else
    echo "❌ Test 2 failed"
    exit 1
fi

echo "All tests passed!"
```

Run with:
```bash
chmod +x test_regression.sh
./test_regression.sh
```

## GPU-Specific Tests

### Test on Different GPUs

Expected performance by GPU tier:

| GPU | VRAM | Expected Speedup | Max Batch Size |
|-----|------|------------------|----------------|
| GTX 1050 Ti | 4 GB | 8-12x | 100K |
| GTX 1060 | 6 GB | 10-15x | 200K |
| RTX 2060 | 6 GB | 12-18x | 250K |
| RTX 3060 | 12 GB | 15-20x | 500K |
| RTX 4090 | 24 GB | 20-30x | 1M |
| A100 | 40 GB | 25-40x | 2M |

### Compute Capability Check

```bash
# Get GPU compute capability
nvidia-smi -q | grep "Compute Capability"
```

**Minimum:** 5.0 (Maxwell)
**Recommended:** 7.5+ (Turing, Ampere, Ada)

## Acceptance Criteria

Before considering POC successful:

- [ ] Compiles without errors on CUDA 11.4+
- [ ] Parses sample PCAPs correctly (validated with tcpdump)
- [ ] Achieves >10x speedup on files >100 MB
- [ ] Handles batches up to 100K packets on 8GB GPU
- [ ] No memory leaks (validated with cuda-memcheck)
- [ ] GPU utilization >80% (check with nvidia-smi)

## Final Validation Checklist

```bash
# 1. Build check
./build.sh
echo $?  # Should be 0

# 2. Download samples
./download_samples.sh
ls data/*.cap  # Should list files

# 3. Run test
./build/cuda_packet_parser data/http.cap
# Verify output looks correct

# 4. Memory check
cuda-memcheck ./build/cuda_packet_parser data/http.cap
# Should show no errors

# 5. Performance check
time ./build/cuda_packet_parser large.pcap
# Should be faster than tcpdump
```

If all checks pass: **POC is validated and ready for Phase 2!**
