# PCAP Analyzer WebGUI

Next-generation web interface for GPU-accelerated PCAP analysis.

## Architecture

- **Backend**: FastAPI (Python) - Async API with task queue
- **Frontend**: SvelteKit - High-performance compiler-based framework
- **Database**: OpenSearch - Full-text search and aggregations
- **Parser**: CUDA Packet Parser - GPU-accelerated processing
- **Queue**: Redis + Celery - Async task management
- **Proxy**: Nginx - Reverse proxy and static file serving

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 18+ (for frontend)
- Python 3.10+ (for backend)
- NVIDIA GPU with CUDA support

### 1. Start Infrastructure

```bash
# Start OpenSearch and Redis
docker-compose up -d
```

### 2. Start Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. Start Worker

```bash
cd backend
source venv/bin/activate
celery -A app.workers.celery_app worker --loglevel=info
```

### 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

### 5. Access Application

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- OpenSearch: http://localhost:9200

## Features

- ✅ PCAP file upload with progress tracking
- ✅ GPU-accelerated packet parsing (10x+ speedup)
- ✅ Real-time parsing status updates (SSE)
- ✅ Interactive network topology graph
- ✅ Statistical dashboards (bandwidth, protocols)
- ✅ Session list with advanced filtering
- ✅ Flow timeline visualization

## Project Structure

```
webgui/
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── models/         # Pydantic models
│   │   ├── services/       # Business logic
│   │   ├── workers/        # Celery tasks
│   │   └── core/           # Config, database
│   └── requirements.txt
├── frontend/               # SvelteKit application
├── docker-compose.yml      # Infrastructure
├── nginx/                  # Reverse proxy config
└── uploads/                # Temporary PCAP storage
```

## Development

See individual README files in `backend/` and `frontend/` directories.

## Deployment

See `docs/DEPLOYMENT.md` for Ubuntu 24 production deployment guide.
