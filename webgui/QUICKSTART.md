# PCAP Analyzer WebGUI - Quick Start Guide

Get the complete stack running in 5 minutes!

## Prerequisites

- CUDA Packet Parser built (`cuda-packet-parser/build/cuda_packet_parser`)
- Docker & Docker Compose installed
- Python 3.10+ and Node.js 18+ installed

## Quick Start (Development)

### 1. Start Infrastructure (OpenSearch + Redis)

```bash
cd /home/cvalentine/ThreatResearch/webgui

# Start containers
docker-compose up -d

# Verify
docker ps
curl http://localhost:9200  # Should return OpenSearch info
```

### 2. Start Backend

**Terminal 1:**
```bash
cd /home/cvalentine/ThreatResearch/webgui/backend

# Create virtual environment (first time only)
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies (first time only)
pip install -r requirements.txt

# Start FastAPI server
uvicorn app.main:app --reload --port 8000
```

Access API docs: http://localhost:8000/docs

### 3. Start Celery Worker

**Terminal 2:**
```bash
cd /home/cvalentine/ThreatResearch/webgui/backend
source venv/bin/activate

# Start Celery worker
celery -A app.workers.celery_app worker --loglevel=info
```

### 4. Start Frontend

**Terminal 3:**
```bash
cd /home/cvalentine/ThreatResearch/webgui/frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

Access frontend: http://localhost:5173

## Usage

1. **Upload PCAP**:
   - Navigate to http://localhost:5173
   - Drag & drop or select a PCAP file
   - File will be uploaded and queued for parsing

2. **Monitor Progress**:
   - You'll be redirected to the dashboard
   - Real-time SSE updates show parsing progress
   - Watch the terminal running the Celery worker to see parsing

3. **View Results**:
   - Once parsing completes, explore:
     - **Overview Tab**: Protocol distribution charts
     - **Sessions Tab**: Filtered list of network flows
     - **Network Topology Tab**: Interactive graph of IP connections
     - **Statistics Tab**: Detailed analysis

## Troubleshooting

### OpenSearch won't start

```bash
# Check if port 9200 is already in use
sudo lsof -i :9200

# Increase vm.max_map_count for OpenSearch
sudo sysctl -w vm.max_map_count=262144
```

### Backend errors

```bash
# Check if CUDA parser exists
ls -la ../../cuda-packet-parser/build/cuda_packet_parser

# Check OpenSearch connection
curl http://localhost:9200

# Check Redis connection
redis-cli ping
```

### Worker not processing jobs

```bash
# Check Redis is running
docker ps | grep redis

# Check Celery can connect
cd backend
source venv/bin/activate
python -c "from app.workers.celery_app import celery_app; print(celery_app.control.inspect().active())"
```

### Frontend build errors

```bash
# Clear node_modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## Test with Sample PCAP

```bash
# Download sample PCAP
wget https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/http.cap -O /tmp/test.pcap

# Or use existing test file
cp ../cuda-packet-parser/data/http.cap /tmp/test.pcap
```

Then upload `/tmp/test.pcap` through the web interface.

## Stop Everything

```bash
# Stop frontend (Ctrl+C in Terminal 3)
# Stop worker (Ctrl+C in Terminal 2)
# Stop backend (Ctrl+C in Terminal 1)

# Stop Docker containers
docker-compose down

# Or to remove data volumes:
docker-compose down -v
```

## Architecture Overview

```
┌─────────────────┐
│  Browser        │
│  (localhost:5173)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SvelteKit      │
│  Frontend       │
└────────┬────────┘
         │ HTTP/SSE
         ▼
┌─────────────────┐      ┌──────────────┐
│  FastAPI        │◄─────┤  OpenSearch  │
│  Backend        │      │  :9200       │
│  :8000          │      └──────────────┘
└────────┬────────┘
         │ Celery Tasks
         ▼
┌─────────────────┐      ┌──────────────┐
│  Celery Worker  │◄─────┤  Redis       │
│                 │      │  :6379       │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│  CUDA Parser    │
│  (GPU)          │
└─────────────────┘
```

## Next Steps

- See `DEPLOYMENT.md` for production deployment
- See individual `README.md` files in `backend/` and `frontend/` for development details
- Check the main `README.md` for architecture overview

## Need Help?

Check logs:
- **Backend**: Terminal 1 output
- **Worker**: Terminal 2 output
- **Frontend**: Terminal 3 output
- **OpenSearch**: `docker logs pcap-opensearch`
- **Redis**: `docker logs pcap-redis`

Happy analyzing! 🚀
