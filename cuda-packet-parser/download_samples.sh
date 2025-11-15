#!/bin/bash

set -e

echo "=== Downloading Sample PCAP Files ==="

mkdir -p data
cd data

# Small HTTP capture (~500 KB)
if [ ! -f "http.cap" ]; then
    echo "Downloading http.cap (HTTP traffic sample)..."
    wget -q https://wiki.wireshark.org/SampleCaptures?action=AttachFile\&do=get\&target=http.cap -O http.cap || true
fi

# DNS queries (~20 KB)
if [ ! -f "dns.cap" ]; then
    echo "Downloading dns.cap (DNS queries)..."
    wget -q https://wiki.wireshark.org/SampleCaptures?action=AttachFile\&do=get\&target=dns.cap -O dns.cap || true
fi

# TCP handshake
if [ ! -f "tcp-3way-handshake.pcap" ]; then
    echo "Downloading tcp-3way-handshake.pcap..."
    wget -q https://wiki.wireshark.org/SampleCaptures?action=AttachFile\&do=get\&target=tcp-3way-handshake.pcap -O tcp-3way-handshake.pcap || true
fi

echo ""
echo "Sample files downloaded to data/ directory:"
ls -lh *.cap *.pcap 2>/dev/null || echo "No files downloaded (network issue?)"
echo ""
echo "You can also create synthetic PCAPs with tcpreplay or scapy"
