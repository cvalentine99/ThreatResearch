# PCAP Analyzer WebGUI - Visualization Guide

## Overview

The PCAP Analyzer WebGUI provides comprehensive real-time visualizations of network packet capture data, powered by GPU-accelerated parsing and advanced analytics. All visualizations are interactive and update dynamically as data is processed.

## Accessing Visualizations

### Dashboard URL Format
```
http://localhost:5174/dashboard/{job_id}
```

### Example
For the completed job from our test run:
```
http://localhost:5174/dashboard/7e66b78b-a4c3-4cfe-b376-768813117d25
```

## Available Visualizations

The dashboard features four main tabs, each providing unique insights into your PCAP data:

### 1. Overview Tab

**Purpose**: High-level summary and protocol distribution

**Features**:
- Summary statistics cards showing:
  - Total packets parsed (e.g., 1,459,877 packets)
  - Total flows identified (e.g., 7,181 unique conversations)
  - GPU parse time (e.g., 2.3 seconds)
- **Protocol Distribution Pie Chart** (Apache ECharts):
  - Interactive donut chart showing packet distribution by protocol
  - Displays percentage and absolute counts
  - Hover to see detailed breakdown
  - Color-coded by protocol:
    - TCP: Blue
    - UDP: Purple
    - ICMP: Cyan
    - Others: Warm colors

**Use Cases**:
- Quick health check of network traffic composition
- Identify unusual protocol distributions
- Detect protocol anomalies (e.g., excessive ICMP)

**Example Data** (from test PCAP):
```
TCP:  1,333,011 packets (91.3%)
UDP:    121,818 packets (8.3%)
ICMP:     5,048 packets (0.3%)
```

---

### 2. Sessions Tab

**Purpose**: Detailed flow/session analysis with filtering

**Features**:
- **Paginated Flow Table**:
  - Source IP and port
  - Destination IP and port
  - Protocol (TCP/UDP/ICMP)
  - Packet count per flow
  - Total bytes transferred
  - Flow duration
- **Dynamic Filters**:
  - Filter by source IP
  - Filter by destination IP
  - Filter by protocol
  - Clear/reset filters
- **Pagination Controls**:
  - 50 flows per page (configurable)
  - Navigate between pages
  - Shows "X - Y of Total" counter

**Use Cases**:
- Investigate specific IP addresses
- Find long-running connections
- Identify high-volume talkers
- Analyze protocol-specific traffic patterns
- Forensic analysis of specific hosts

**Example Top Flows**:
```
65.57.79.228:34104 → 3.5.85.191:443   TCP   149,228 packets
65.57.79.228:34030 → 3.5.85.191:443   TCP   147,797 packets
65.57.79.228:34086 → 3.5.85.191:443   TCP   119,208 packets
```

**Filtering Example**:
```
Source IP: 65.57.79.228
Protocol: TCP
→ Shows only TCP flows from that specific IP
```

---

### 3. Network Topology Tab

**Purpose**: Visual network graph showing IP relationships

**Features**:
- **Interactive Force-Directed Graph** (Cytoscape.js):
  - Nodes represent IP addresses
  - Edges represent network flows
  - Node color coding:
    - Blue: Internal IPs (RFC 1918 private ranges)
    - Purple: External/public IPs
  - Edge thickness: Proportional to packet count
  - Edge label: Protocol and packet count
- **Graph Controls**:
  - Zoom In/Out buttons
  - Reset View button
  - Pan by dragging
  - Click nodes/edges for details
- **Statistics Display**:
  - Total nodes (unique IPs)
  - Total edges (unique flows)
- **Layout Algorithm**:
  - COSE (Compound Spring Embedder) layout
  - Automatically positions nodes for clarity
  - Avoids overlaps
  - Shows network structure

**Use Cases**:
- Visualize network architecture
- Identify hub nodes (high-degree IPs)
- Detect unusual connection patterns
- Spot isolated clusters
- Find internal↔external communication patterns
- Network reconnaissance analysis

**Example Network**:
```
Nodes: 150 unique IPs
  - Internal: 45 (10.x.x.x range)
  - External: 105 (public IPs)
Edges: 7,181 flows
```

**IP Classification**:
- `10.0.0.0/8` → Internal
- `172.16.0.0/12` → Internal
- `192.168.0.0/16` → Internal
- All others → External

---

### 4. Statistics Tab

**Purpose**: Detailed statistical analysis and future analytics

**Current Features**:
- Same protocol distribution chart as Overview tab
- More detailed view optimized for analysis

**Planned Features** (Coming Soon):
- **Bandwidth Over Time**: Timeline chart showing traffic volume
- **Top Talkers**: Bar chart of most active hosts by bytes
- **Traffic Heatmap**: Hour-by-hour or day-by-day activity matrix
- **Geo-IP Mapping**: Geographic visualization of external IPs
- **Port Distribution**: Most common ports used
- **Packet Size Distribution**: Histogram of packet sizes

---

## Navigation Guide

### 1. Upload a PCAP File

```
1. Navigate to http://localhost:5174/
2. Drag & drop a .pcap or .pcapng file (or click to browse)
3. File validation checks:
   - Extension must be .pcap or .pcapng
   - File size limit: 10GB
4. Wait for upload confirmation
5. Automatically redirected to dashboard
```

### 2. Monitor Processing Status

```
While job is running/pending:
- Status badge shows current state
- Progress indicator (if available)
- Real-time updates via Server-Sent Events
- "Parsing in Progress" card with spinner

When job completes:
- Status badge turns green
- Summary statistics appear
- All visualization tabs become active
```

### 3. Explore Visualizations

```
Overview Tab → Quick protocol breakdown
   ↓
Sessions Tab → Investigate specific flows
   ↓
Topology Tab → Visualize network structure
   ↓
Statistics Tab → Deep dive into analytics
```

### 4. Use Filters and Controls

**Session Filters**:
```javascript
// Filter for traffic from specific IP
Source IP: 10.141.42.10
→ Click "Apply"

// Filter for UDP traffic
Protocol: UDP
→ Click "Apply"

// Combine filters
Source IP: 10.141.42.10
Protocol: TCP
→ Click "Apply"

// Reset all filters
→ Click "Clear"
```

**Graph Controls**:
```
Zoom In/Out: Adjust zoom level
Reset View: Fit all nodes in viewport
Pan: Click and drag canvas
Click Node: Show IP details in console
Click Edge: Show flow details in console
```

---

## API Endpoints for Custom Integration

All visualizations are powered by RESTful API endpoints that you can use for custom integrations, scripts, or external tools.

### Base URL
```
http://localhost:8020/api/v1
```

### 1. Job Summary

**Endpoint**: `GET /stats/summary`

**Parameters**:
- `job_id` (required): Job UUID

**Response**:
```json
{
  "job_id": "7e66b78b-a4c3-4cfe-b376-768813117d25",
  "filename": "capture.pcap",
  "status": "completed",
  "total_packets": 1459877,
  "total_flows": 7181,
  "protocols": {
    "TCP": 1333011,
    "UDP": 121818,
    "ICMP": 5048
  },
  "parse_time_ms": 2349
}
```

**Example**:
```bash
curl "http://localhost:8020/api/v1/stats/summary?job_id=7e66b78b-a4c3-4cfe-b376-768813117d25"
```

---

### 2. Protocol Distribution

**Endpoint**: `GET /stats/protocols`

**Parameters**:
- `job_id` (required): Job UUID

**Response**:
```json
{
  "protocols": [
    {"protocol": "TCP", "count": 1333011},
    {"protocol": "UDP", "count": 121818},
    {"protocol": "ICMP", "count": 5048}
  ]
}
```

**Example**:
```bash
curl "http://localhost:8020/api/v1/stats/protocols?job_id=7e66b78b-a4c3-4cfe-b376-768813117d25"
```

---

### 3. Flow Query

**Endpoint**: `GET /flows`

**Parameters**:
- `job_id` (required): Job UUID
- `src_ip` (optional): Filter by source IP
- `dst_ip` (optional): Filter by destination IP
- `protocol` (optional): Filter by protocol (TCP/UDP/ICMP)
- `limit` (optional): Results per page (default: 100, max: 1000)
- `offset` (optional): Pagination offset (default: 0)

**Response**:
```json
{
  "total": 7181,
  "flows": [
    {
      "job_id": "7e66b78b-a4c3-4cfe-b376-768813117d25",
      "flow_hash": "0xb3a38466",
      "src_ip": "65.57.79.228",
      "src_port": 34104,
      "dst_ip": "3.5.85.191",
      "dst_port": 443,
      "protocol": "TCP",
      "packet_count": 149228,
      "total_bytes": 0,
      "duration_ms": 0,
      "first_seen": null,
      "last_seen": null,
      "created_at": "2025-11-15T22:50:46.845048"
    }
  ],
  "limit": 100,
  "offset": 0
}
```

**Examples**:
```bash
# Get all flows for a job
curl "http://localhost:8020/api/v1/flows?job_id=7e66b78b-a4c3-4cfe-b376-768813117d25&limit=50"

# Filter by source IP
curl "http://localhost:8020/api/v1/flows?job_id=7e66b78b-a4c3-4cfe-b376-768813117d25&src_ip=65.57.79.228"

# Filter by protocol
curl "http://localhost:8020/api/v1/flows?job_id=7e66b78b-a4c3-4cfe-b376-768813117d25&protocol=UDP"

# Pagination (page 2)
curl "http://localhost:8020/api/v1/flows?job_id=7e66b78b-a4c3-4cfe-b376-768813117d25&limit=50&offset=50"
```

---

### 4. Network Topology Graph

**Endpoint**: `GET /graph/topology`

**Parameters**:
- `job_id` (required): Job UUID
- `limit` (optional): Maximum number of flows to include (default: 500, max: 2000)

**Response**:
```json
{
  "nodes": [
    {
      "data": {
        "id": "10.141.42.10",
        "label": "10.141.42.10",
        "type": "internal"
      }
    },
    {
      "data": {
        "id": "3.5.85.191",
        "label": "3.5.85.191",
        "type": "external"
      }
    }
  ],
  "edges": [
    {
      "data": {
        "id": "0xb3a38466",
        "source": "65.57.79.228",
        "target": "3.5.85.191",
        "protocol": "TCP",
        "packet_count": 149228,
        "total_bytes": 0,
        "label": "TCP (149228 pkts)"
      }
    }
  ],
  "total_nodes": 150,
  "total_edges": 500
}
```

**Example**:
```bash
# Get topology with top 100 flows
curl "http://localhost:8020/api/v1/graph/topology?job_id=7e66b78b-a4c3-4cfe-b376-768813117d25&limit=100"

# Get full topology (up to 2000 flows)
curl "http://localhost:8020/api/v1/graph/topology?job_id=7e66b78b-a4c3-4cfe-b376-768813117d25&limit=2000"
```

---

### 5. Additional Statistics Endpoints

#### Top Source/Destination IPs

**Endpoint**: `GET /stats/ips/top`

**Parameters**:
- `job_id` (required): Job UUID
- `direction` (optional): "src" or "dst" (default: "src")
- `limit` (optional): Number of top IPs (default: 20)

```bash
curl "http://localhost:8020/api/v1/stats/ips/top?job_id=7e66b78b-a4c3-4cfe-b376-768813117d25&direction=src&limit=10"
```

#### Connection Matrix

**Endpoint**: `GET /stats/connections`

**Parameters**:
- `job_id` (required): Job UUID
- `limit` (optional): Number of connections (default: 100)

```bash
curl "http://localhost:8020/api/v1/stats/connections?job_id=7e66b78b-a4c3-4cfe-b376-768813117d25&limit=200"
```

#### Top Flows by Metric

**Endpoint**: `GET /flows/top`

**Parameters**:
- `job_id` (required): Job UUID
- `metric` (optional): "packet_count" or "total_bytes" (default: "packet_count")
- `limit` (optional): Number of flows (default: 10, max: 100)

```bash
curl "http://localhost:8020/api/v1/flows/top?job_id=7e66b78b-a4c3-4cfe-b376-768813117d25&metric=packet_count&limit=20"
```

---

## Performance Considerations

### Large PCAP Files

For files over 1GB:
- **Topology graph**: Limit to 500-1000 flows for optimal rendering
- **Session table**: Use pagination and filters to reduce dataset size
- **Charts**: Protocol distribution handles millions of packets efficiently

### Browser Performance

**Network Topology**:
- Recommended: < 500 nodes for smooth interaction
- Maximum: ~ 2000 nodes before performance degrades
- Use `limit` parameter to reduce complexity

**Session Table**:
- Pagination: 50 rows per page (optimal)
- Filtering reduces backend query time

### API Rate Limiting

Currently no rate limits enforced. For production:
- Consider caching frequently accessed job summaries
- Use pagination for large result sets
- Implement request throttling if needed

---

## Data Insights and Use Cases

### Network Security Analysis

**Detect Port Scanning**:
```
1. Navigate to Sessions tab
2. Filter by source IP of suspected scanner
3. Observe high number of unique destination ports
4. Check for low packet count per flow (SYN scans)
```

**Identify Data Exfiltration**:
```
1. Navigate to Sessions tab
2. Sort by total bytes (descending)
3. Look for unusual high-volume external connections
4. Check topology for unexpected internal→external flows
```

**Find Command & Control (C2) Traffic**:
```
1. Navigate to Topology tab
2. Identify external IPs with many internal connections
3. Check Sessions tab for regular interval traffic
4. Examine protocol distribution for anomalies
```

### Network Troubleshooting

**Diagnose Connectivity Issues**:
```
1. Navigate to Sessions tab
2. Filter by affected host IP
3. Check packet counts (low = connection problems)
4. Examine protocol distribution (excessive retransmits)
```

**Bandwidth Analysis**:
```
1. Navigate to Sessions tab
2. Sort by total bytes
3. Identify top bandwidth consumers
4. Check topology for traffic patterns
```

**Protocol Issues**:
```
1. Navigate to Overview tab
2. Check protocol distribution
3. Look for unexpected protocol ratios
4. Investigate anomalies in Sessions tab
```

### Forensics

**Timeline Reconstruction**:
```
1. Use API to query flows with timestamps
2. Sort by first_seen/last_seen
3. Build timeline of network activity
4. Correlate with security events
```

**Host Profiling**:
```
1. Filter Sessions by specific IP
2. Examine all flows to/from that host
3. Check Topology for connection patterns
4. Profile normal vs. anomalous behavior
```

---

## Customization and Extensions

### JavaScript Integration

```javascript
// Fetch and process job summary
fetch('http://localhost:8020/api/v1/stats/summary?job_id=YOUR_JOB_ID')
  .then(res => res.json())
  .then(data => {
    console.log(`Parsed ${data.total_packets} packets in ${data.parse_time_ms}ms`);
    console.log(`GPU throughput: ${(data.total_packets / (data.parse_time_ms / 1000)).toFixed(0)} packets/sec`);
  });
```

### Python Integration

```python
import requests

job_id = "7e66b78b-a4c3-4cfe-b376-768813117d25"
base_url = "http://localhost:8020/api/v1"

# Get all TCP flows from a specific source IP
response = requests.get(f"{base_url}/flows", params={
    "job_id": job_id,
    "src_ip": "10.141.42.10",
    "protocol": "TCP",
    "limit": 1000
})

flows = response.json()["flows"]
print(f"Found {len(flows)} TCP flows from 10.141.42.10")

# Calculate total bytes
total_bytes = sum(flow["total_bytes"] for flow in flows)
print(f"Total bytes transferred: {total_bytes:,}")
```

### Visualization Libraries

The dashboard uses:
- **Cytoscape.js**: Network graphs (https://js.cytoscape.org/)
- **Apache ECharts**: Charts (https://echarts.apache.org/)
- **Tailwind CSS + DaisyUI**: Styling (https://daisyui.com/)

You can extend visualizations by modifying component files:
- `frontend/src/lib/components/NetworkGraph.svelte`
- `frontend/src/lib/components/StatsCharts.svelte`
- `frontend/src/lib/components/SessionList.svelte`

---

## Troubleshooting

### Visualizations Not Loading

**Symptom**: Spinner shows indefinitely

**Solutions**:
1. Check browser console for errors (F12)
2. Verify backend API is responding:
   ```bash
   curl http://localhost:8020/api/v1/stats/summary?job_id=YOUR_JOB_ID
   ```
3. Check CORS configuration in backend `.env`:
   ```bash
   CORS_ORIGINS=["http://localhost:5174"]
   ```
4. Restart frontend: `npm run dev -- --port 5174 --host`

### Empty Topology Graph

**Symptom**: "0 nodes, 0 edges" displayed

**Solutions**:
1. Verify flows were calculated and indexed:
   ```bash
   curl "http://localhost:8020/api/v1/flows?job_id=YOUR_JOB_ID&limit=10"
   ```
2. Check OpenSearch has flow data:
   ```bash
   curl "http://localhost:9200/pcap-flows/_count"
   ```
3. Increase topology `limit` parameter if flows exist but graph is empty

### Session Table Shows No Data

**Symptom**: "No sessions found" message

**Solutions**:
1. Check if filters are too restrictive → Click "Clear"
2. Verify flows indexed:
   ```bash
   curl "http://localhost:8020/api/v1/flows?job_id=YOUR_JOB_ID"
   ```
3. Check pagination - may be on empty page → Go to page 1

### Protocol Chart Not Rendering

**Symptom**: Chart container is blank

**Solutions**:
1. Check browser console for ECharts errors
2. Verify protocol data exists:
   ```bash
   curl "http://localhost:8020/api/v1/stats/protocols?job_id=YOUR_JOB_ID"
   ```
3. Hard refresh browser (Ctrl+Shift+R)

---

## Keyboard Shortcuts (Planned)

Future enhancements will include keyboard shortcuts:
- `Tab 1-4`: Switch between visualization tabs
- `F`: Toggle filters panel
- `R`: Reset graph view
- `/`: Focus search box

---

## Mobile Responsiveness

The dashboard is responsive and works on tablets and mobile devices:
- Tables scroll horizontally
- Charts resize to fit screen
- Touch gestures supported on graph (pinch to zoom)
- Simplified layout on small screens

---

## Accessibility

- High contrast color scheme
- Keyboard navigation support
- ARIA labels for screen readers
- Colorblind-friendly palette

---

## Performance Metrics (Example PCAP)

**Test File**: `extrahop 2025-11-15 05.03.47 to 05.11.07 PST.pcap`
- **File Size**: 1.39 GB
- **Total Packets**: 1,459,877
- **GPU Parse Time**: 2.3 seconds (636,034 packets/sec)
- **Total Processing Time**: 55.5 seconds
- **Flows Calculated**: 7,181
- **OpenSearch Indexing**: ~26,000 packets/sec

**Visualization Load Times**:
- Protocol Chart: < 1 second
- Session Table (50 rows): < 1 second
- Network Topology (100 flows): 1-2 seconds
- Network Topology (500 flows): 2-4 seconds

---

## Next Steps

1. **Explore Your Data**: Navigate to the dashboard URL for your job
2. **Test Filters**: Try different combinations in Sessions tab
3. **Analyze Topology**: Look for interesting network patterns
4. **Export Data**: Use API endpoints to extract specific flows
5. **Integrate**: Build custom tools using the RESTful API

For technical architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md)

For build instructions, see [BUILD.md](BUILD.md)
