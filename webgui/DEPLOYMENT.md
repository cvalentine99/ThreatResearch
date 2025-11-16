# PCAP Analyzer - Production Deployment Guide

Complete guide for deploying the PCAP Analyzer on Ubuntu 24.

## Prerequisites

- Ubuntu 24 Server
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.4+
- Docker & Docker Compose
- Python 3.10+
- Node.js 18+
- Nginx

## Step-by-Step Deployment

### 1. System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install core dependencies
sudo apt install -y python3-pip python3-venv nginx git curl

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install -y docker-compose

# Configure firewall
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

### 2. Build CUDA Parser

```bash
cd /opt/ThreatResearch/cuda-packet-parser
./build.sh
```

Verify:
```bash
./build/cuda_packet_parser --version
```

### 3. Start Infrastructure (OpenSearch & Redis)

```bash
cd /opt/ThreatResearch/webgui
docker-compose up -d
```

Verify:
```bash
curl http://localhost:9200  # OpenSearch
redis-cli ping              # Redis
```

### 4. Deploy Backend

```bash
cd /opt/ThreatResearch/webgui/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with production values

# Test backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Create Systemd Service for Backend

```bash
sudo vim /etc/systemd/system/pcap-backend.service
```

Add:
```ini
[Unit]
Description=PCAP Analyzer FastAPI Backend
After=network.target docker.service

[Service]
User=your-username
WorkingDirectory=/opt/ThreatResearch/webgui/backend
Environment="PATH=/opt/ThreatResearch/webgui/backend/venv/bin"
ExecStart=/opt/ThreatResearch/webgui/backend/venv/bin/gunicorn \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000 \
    app.main:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable pcap-backend
sudo systemctl start pcap-backend
sudo systemctl status pcap-backend
```

#### Create Systemd Service for Celery Worker

```bash
sudo vim /etc/systemd/system/pcap-worker.service
```

Add:
```ini
[Unit]
Description=PCAP Analyzer Celery Worker
After=network.target redis.service

[Service]
User=your-username
WorkingDirectory=/opt/ThreatResearch/webgui/backend
Environment="PATH=/opt/ThreatResearch/webgui/backend/venv/bin"
ExecStart=/opt/ThreatResearch/webgui/backend/venv/bin/celery \
    -A app.workers.celery_app worker \
    --loglevel=info \
    --concurrency=2
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable pcap-worker
sudo systemctl start pcap-worker
sudo systemctl status pcap-worker
```

### 5. Deploy Frontend

```bash
cd /opt/ThreatResearch/webgui/frontend

# Install dependencies
npm install

# Build for production
npm run build

# Create web directory
sudo mkdir -p /var/www/pcap-analyzer/frontend

# Copy build files
sudo cp -R build/* /var/www/pcap-analyzer/frontend/

# Set permissions
sudo chown -R www-data:www-data /var/www/pcap-analyzer
```

### 6. Configure Nginx

```bash
# Copy nginx config
sudo cp /opt/ThreatResearch/webgui/nginx/pcap-analyzer.conf /etc/nginx/sites-available/

# Enable site
sudo ln -s /etc/nginx/sites-available/pcap-analyzer.conf /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

### 7. Configure SSL (Production)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d pcap.yourdomain.com

# Verify auto-renewal
sudo certbot renew --dry-run
```

### 8. Verify Deployment

```bash
# Check all services
sudo systemctl status pcap-backend
sudo systemctl status pcap-worker
sudo systemctl status nginx
docker ps

# Check logs
sudo journalctl -u pcap-backend -f
sudo journalctl -u pcap-worker -f
```

Access the application:
- HTTP: http://your-server-ip
- HTTPS: https://pcap.yourdomain.com

## Monitoring & Maintenance

### Check Logs

```bash
# Backend logs
sudo journalctl -u pcap-backend -n 100 --no-pager

# Worker logs
sudo journalctl -u pcap-worker -n 100 --no-pager

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# OpenSearch logs
docker logs pcap-opensearch
```

### Database Maintenance

```bash
# Check OpenSearch indices
curl http://localhost:9200/_cat/indices?v

# Clear old data (optional)
curl -X DELETE http://localhost:9200/pcap-packets
curl -X DELETE http://localhost:9200/pcap-flows
curl -X DELETE http://localhost:9200/pcap-jobs
```

### Backup Strategy

```bash
# Backup OpenSearch data
docker exec pcap-opensearch \
    curl -X PUT "localhost:9200/_snapshot/backup" \
    -H 'Content-Type: application/json' \
    -d '{"type": "fs", "settings": {"location": "/backup"}}'

# Backup configuration
tar -czf pcap-analyzer-config-backup.tar.gz \
    /opt/ThreatResearch/webgui/backend/.env \
    /etc/nginx/sites-available/pcap-analyzer.conf \
    /etc/systemd/system/pcap-*.service
```

## Troubleshooting

### Backend won't start

```bash
# Check Python environment
source /opt/ThreatResearch/webgui/backend/venv/bin/activate
python -c "import fastapi; print(fastapi.__version__)"

# Check dependencies
pip list

# Run manually for debugging
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Worker not processing jobs

```bash
# Check Redis connection
redis-cli ping

# Check Celery
cd /opt/ThreatResearch/webgui/backend
source venv/bin/activate
celery -A app.workers.celery_app inspect active

# Run worker manually
celery -A app.workers.celery_app worker --loglevel=debug
```

### OpenSearch issues

```bash
# Check container status
docker ps | grep opensearch

# Restart container
docker-compose restart opensearch

# Check memory
docker stats pcap-opensearch
```

### Upload failures

```bash
# Check upload directory permissions
ls -la /opt/ThreatResearch/webgui/uploads

# Check nginx file size limit
grep client_max_body_size /etc/nginx/sites-available/pcap-analyzer.conf

# Check disk space
df -h
```

## Performance Tuning

### Gunicorn Workers

Adjust workers based on CPU cores:
```bash
# In systemd service file
--workers $((2 * $(nproc) + 1))
```

### OpenSearch Memory

Adjust in docker-compose.yml:
```yaml
environment:
  - "OPENSEARCH_JAVA_OPTS=-Xms4g -Xmx4g"  # 4GB heap
```

### Nginx Caching

Add to nginx config:
```nginx
location ~* \.(js|css|png|jpg|jpeg|gif|ico)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

## Security Hardening

1. **Enable HTTPS** (required for production)
2. **Configure firewall** (only allow 80, 443, 22)
3. **Disable OpenSearch security plugin** (development only!)
4. **Use strong SECRET_KEY** in .env
5. **Limit CORS origins** to your domain
6. **Enable rate limiting** in nginx
7. **Regular security updates**: `sudo apt update && sudo apt upgrade`

## Scaling Considerations

- **Multi-GPU**: Modify CUDA parser to support multiple GPUs
- **Load Balancing**: Add multiple backend workers behind Nginx
- **OpenSearch Cluster**: Deploy multi-node cluster for HA
- **Redis Sentinel**: Add Redis HA for task queue
- **CDN**: Serve frontend assets via CDN

## Support

For issues, check:
- Backend logs: `sudo journalctl -u pcap-backend`
- Worker logs: `sudo journalctl -u pcap-worker`
- GitHub Issues: https://github.com/your-repo/issues
