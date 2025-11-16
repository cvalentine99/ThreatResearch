# PCAP Analyzer - Backend

FastAPI backend for GPU-accelerated PCAP analysis.

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` and edit:

```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Start Infrastructure

Ensure OpenSearch and Redis are running:

```bash
cd ..
docker-compose up -d
```

### 5. Run Development Server

```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Start Celery Worker (separate terminal)

```bash
celery -A app.workers.celery_app worker --loglevel=info
```

## API Documentation

Once running, access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Upload
- `POST /api/v1/upload` - Upload PCAP file
- `GET /api/v1/upload/{job_id}` - Get job status
- `GET /api/v1/upload` - List jobs

### Packets
- `GET /api/v1/packets` - Query packets
- `GET /api/v1/packets/conversation` - Get conversation

### Flows
- `GET /api/v1/flows` - Query flows
- `GET /api/v1/flows/top` - Top flows

### Statistics
- `GET /api/v1/stats/protocols` - Protocol distribution
- `GET /api/v1/stats/ips/top` - Top IPs
- `GET /api/v1/stats/connections` - Connection matrix
- `GET /api/v1/stats/summary` - Job summary

### Graph
- `GET /api/v1/graph/topology` - Network topology

### Streaming
- `GET /api/v1/stream/job/{job_id}` - SSE status stream

## Project Structure

```
backend/
├── app/
│   ├── api/              # API endpoints
│   │   ├── upload.py     # File upload
│   │   ├── packets.py    # Packet queries
│   │   ├── flows.py      # Flow queries
│   │   ├── stats.py      # Statistics
│   │   ├── stream.py     # SSE streaming
│   │   └── graph.py      # Graph/topology
│   ├── core/             # Core modules
│   │   ├── config.py     # Configuration
│   │   └── opensearch.py # DB client
│   ├── models/           # Pydantic models
│   │   ├── packet.py
│   │   ├── flow.py
│   │   └── job.py
│   ├── workers/          # Celery tasks
│   │   ├── celery_app.py
│   │   └── tasks.py      # PCAP parsing
│   └── main.py           # FastAPI app
└── requirements.txt
```

## Development

### Run Tests

```bash
pytest
```

### Format Code

```bash
black app/
isort app/
```

### Type Checking

```bash
mypy app/
```
