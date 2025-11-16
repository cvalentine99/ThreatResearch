# PCAP Analyzer WebGUI - Build & Deployment Guide

## Prerequisites

### System Requirements
- Ubuntu 20.04+ or compatible Linux distribution
- NVIDIA GPU with CUDA support (Compute Capability 6.0+)
- 16GB+ RAM recommended
- 50GB+ free disk space

### Required Software
- CUDA Toolkit 11.0+ (for GPU packet parsing)
- GCC/G++ 9.0+ compiler
- CMake 3.18+
- Node.js 18+ and npm
- Python 3.9+
- Docker and Docker Compose (for OpenSearch/Redis)
- Git

## Build Instructions

### 1. Clone the Repository

```bash
cd /home/cvalentine/ThreatResearch
git clone <repository-url> webgui
cd webgui
```

### 2. Build the CUDA Packet Parser

The GPU-accelerated packet parser must be built first:

```bash
cd /home/cvalentine/ThreatResearch/cuda-packet-parser

# Clean any previous builds
rm -rf build/
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build (use -j for parallel compilation)
make -j$(nproc)

# Verify the binary exists
ls -lh cuda_packet_parser

# Expected output: executable ~2-5MB
```

**Troubleshooting CUDA Build:**
- If CUDA not found: `export CUDA_HOME=/usr/local/cuda`
- If compute capability errors: Update CMakeLists.txt CUDA_ARCHITECTURES
- Missing libraries: `sudo apt install libpcap-dev`

### 3. Set Up Infrastructure Services

Start OpenSearch and Redis using Docker Compose:

```bash
cd /home/cvalentine/ThreatResearch/webgui

# Start services in detached mode
docker-compose up -d

# Verify services are running
docker-compose ps

# Check OpenSearch is ready (may take 30-60 seconds)
curl http://localhost:9200/_cluster/health?pretty

# Expected: status "green" or "yellow"
```

**OpenSearch Initial Setup:**

If disk usage is high (>85%), configure watermarks:

```bash
curl -X PUT "localhost:9200/_cluster/settings" -H 'Content-Type: application/json' -d'
{
  "persistent": {
    "cluster.routing.allocation.disk.watermark.low": "90%",
    "cluster.routing.allocation.disk.watermark.high": "95%",
    "cluster.routing.allocation.disk.watermark.flood_stage": "97%"
  }
}'
```

### 4. Set Up Python Backend

```bash
cd /home/cvalentine/ThreatResearch/webgui/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file with absolute paths
cat > .env << 'EOF'
# OpenSearch Configuration
OPENSEARCH_URL=http://localhost:9200

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# CUDA Parser (ABSOLUTE PATH REQUIRED)
CUDA_PARSER_PATH=/home/cvalentine/ThreatResearch/cuda-packet-parser/build/cuda_packet_parser

# CORS Origins (JSON array format)
CORS_ORIGINS=["http://localhost:5174", "http://localhost:5173", "http://localhost:3000"]

# File Upload Settings
MAX_UPLOAD_SIZE=10737418240

# Storage
UPLOAD_DIR=./uploads
EOF

# Create upload directory
mkdir -p uploads

# Verify configuration
python -c "from app.core.config import settings; print(f'CUDA Parser: {settings.CUDA_PARSER_PATH}')"
```

### 5. Set Up Frontend

```bash
cd /home/cvalentine/ThreatResearch/webgui/frontend

# Install dependencies
npm install

# Verify Vite configuration
cat vite.config.js

# Build for production (optional)
npm run build
```

## Running the Application

### Start All Services

You need 3 terminal windows/tabs:

**Terminal 1 - Celery Worker:**
```bash
cd /home/cvalentine/ThreatResearch/webgui/backend
source venv/bin/activate
celery -A app.workers.celery_config worker --loglevel=info
```

Expected output:
```
[tasks]
  . app.workers.tasks.parse_pcap_task

celery@hostname ready.
```

**Terminal 2 - FastAPI Backend:**
```bash
cd /home/cvalentine/ThreatResearch/webgui/backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8020 --reload
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8020
INFO:     Application startup complete.
```

**Terminal 3 - SvelteKit Frontend:**
```bash
cd /home/cvalentine/ThreatResearch/webgui/frontend
npm run dev -- --port 5174 --host
```

Expected output:
```
VITE v5.4.21  ready in 536 ms

  ➜  Local:   http://localhost:5174/
  ➜  Network: http://192.168.50.158:5174/
```

### Alternative: Use tmux for Process Management

```bash
# Start tmux session
tmux new -s pcap-analyzer

# Window 0: Celery worker
cd /home/cvalentine/ThreatResearch/webgui/backend
source venv/bin/activate
celery -A app.workers.celery_config worker --loglevel=info

# Create new window (Ctrl+b, c)
# Window 1: Backend API
cd /home/cvalentine/ThreatResearch/webgui/backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8020 --reload

# Create new window (Ctrl+b, c)
# Window 2: Frontend
cd /home/cvalentine/ThreatResearch/webgui/frontend
npm run dev -- --port 5174 --host

# Switch between windows: Ctrl+b, 0/1/2
# Detach: Ctrl+b, d
# Reattach: tmux attach -t pcap-analyzer
```

## Verification

### 1. Check Service Health

```bash
# OpenSearch
curl http://localhost:9200/_cluster/health?pretty

# Redis
redis-cli ping
# Expected: PONG

# Backend API
curl http://localhost:8020/api/v1/health
# Expected: {"status":"healthy"}

# Frontend
curl http://localhost:5174/
# Expected: HTML content
```

### 2. Test PCAP Upload via CLI

```bash
# Upload a test PCAP
curl -X POST http://localhost:8020/api/v1/upload/ \
  -F "file=@/path/to/test.pcap"

# Expected response:
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "test.pcap",
  "status": "queued",
  ...
}

# Check job status
curl http://localhost:8020/api/v1/upload/<job_id>/

# Monitor Celery worker logs for processing
```

### 3. Test via Web UI

1. Open browser: http://localhost:5174/
2. Drag & drop a .pcap or .pcapng file
3. Wait for "Upload successful" message
4. Redirected to dashboard showing job status
5. When status changes to "completed", view packets/flows/graphs

## Performance Tuning

### OpenSearch Heap Size

For large PCAP files (>1GB), increase heap:

```yaml
# docker-compose.yml
services:
  opensearch:
    environment:
      - "OPENSEARCH_JAVA_OPTS=-Xms4g -Xmx4g"  # 4GB heap
```

### Celery Concurrency

For multi-core systems:

```bash
# Run 4 worker processes
celery -A app.workers.celery_config worker --loglevel=info --concurrency=4
```

### CUDA Batch Size

Edit `cuda-packet-parser/src/main.cpp`:

```cpp
// Increase for larger GPUs with more memory
const uint32_t BATCH_SIZE = 262144;  // 256K packets per batch
```

Rebuild after changes.

## Common Issues

### CUDA Parser Not Found
```
Error: FileNotFoundError: cuda_packet_parser
```

**Fix:**
1. Verify absolute path in `.env`
2. Check file permissions: `chmod +x /path/to/cuda_packet_parser`
3. Test manually: `/path/to/cuda_packet_parser /path/to/test.pcap`

### OpenSearch Disk Full
```
Error: 429 cluster_block_exception
```

**Fix:**
```bash
# Remove read-only block
curl -X PUT "localhost:9200/_all/_settings" -H 'Content-Type: application/json' -d'
{
  "index.blocks.read_only_allow_delete": null
}'

# Free disk space (>10% required)
df -h
```

### Port Already in Use
```
Error: Address already in use
```

**Fix:**
```bash
# Find process using port 8020
lsof -i :8020

# Kill process if safe
kill <PID>

# Or use alternative port in .env and update frontend
```

### CORS Errors in Browser

**Fix:** Ensure `CORS_ORIGINS` in `.env` includes frontend URL:
```bash
CORS_ORIGINS=["http://localhost:5174"]
```

### Upload Hangs in Browser

**Symptom:** Upload shows "Uploading..." indefinitely

**Fix:** Frontend uses direct backend connection (not Vite proxy):
```javascript
// frontend/src/lib/api/client.js
const API_BASE = 'http://localhost:8020/api/v1';  // Direct connection
```

## Production Deployment

### Build Frontend for Production

```bash
cd /home/cvalentine/ThreatResearch/webgui/frontend
npm run build

# Serve with nginx
sudo cp -r build/* /var/www/pcap-analyzer/
```

### Run Backend with Gunicorn

```bash
cd /home/cvalentine/ThreatResearch/webgui/backend
source venv/bin/activate

gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8020 \
  --access-logfile /var/log/pcap-analyzer/access.log \
  --error-logfile /var/log/pcap-analyzer/error.log
```

### Systemd Service Files

Create `/etc/systemd/system/pcap-analyzer-backend.service`:

```ini
[Unit]
Description=PCAP Analyzer Backend API
After=network.target opensearch.service redis.service

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/home/cvalentine/ThreatResearch/webgui/backend
Environment="PATH=/home/cvalentine/ThreatResearch/webgui/backend/venv/bin"
ExecStart=/home/cvalentine/ThreatResearch/webgui/backend/venv/bin/gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8020

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/pcap-analyzer-worker.service`:

```ini
[Unit]
Description=PCAP Analyzer Celery Worker
After=network.target opensearch.service redis.service

[Service]
Type=forking
User=www-data
Group=www-data
WorkingDirectory=/home/cvalentine/ThreatResearch/webgui/backend
Environment="PATH=/home/cvalentine/ThreatResearch/webgui/backend/venv/bin"
ExecStart=/home/cvalentine/ThreatResearch/webgui/backend/venv/bin/celery -A app.workers.celery_config worker \
  --loglevel=info \
  --concurrency=4

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable pcap-analyzer-backend pcap-analyzer-worker
sudo systemctl start pcap-analyzer-backend pcap-analyzer-worker
```

## Maintenance

### View Logs

```bash
# Celery worker
tail -f /var/log/celery/worker.log

# Backend API
tail -f /var/log/pcap-analyzer/error.log

# OpenSearch
docker logs -f opensearch
```

### Clean Up Old Jobs

```bash
# Delete jobs older than 7 days
curl -X DELETE "localhost:9200/pcap-packets/_delete_by_query" -H 'Content-Type: application/json' -d'
{
  "query": {
    "range": {
      "created_at": {
        "lt": "now-7d"
      }
    }
  }
}'
```

### Backup OpenSearch Data

```bash
# Create snapshot repository
curl -X PUT "localhost:9200/_snapshot/backups" -H 'Content-Type: application/json' -d'
{
  "type": "fs",
  "settings": {
    "location": "/usr/share/opensearch/snapshots"
  }
}'

# Take snapshot
curl -X PUT "localhost:9200/_snapshot/backups/snapshot_1?wait_for_completion=true"
```

## Next Steps

- Configure nginx reverse proxy for production
- Set up SSL/TLS certificates
- Implement authentication/authorization
- Configure log rotation
- Set up monitoring (Prometheus, Grafana)
- Implement data retention policies

For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md)
