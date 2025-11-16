"""Application configuration"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings"""

    # Application
    APP_NAME: str = "PCAP Analyzer WebGUI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # API
    API_V1_PREFIX: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-change-in-production"
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # OpenSearch
    OPENSEARCH_URL: str = "http://localhost:9200"
    OPENSEARCH_INDEX_PACKETS: str = "pcap-packets"
    OPENSEARCH_INDEX_FLOWS: str = "pcap-flows"
    OPENSEARCH_INDEX_JOBS: str = "pcap-jobs"

    # Redis & Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # CUDA Parser
    CUDA_PARSER_PATH: str = "../cuda-packet-parser/build/cuda_packet_parser"
    CUDA_BATCH_SIZE: int = 1000000  # 1M packets per batch for optimal performance

    # File Upload
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024 * 1024  # 10GB
    ALLOWED_EXTENSIONS: List[str] = [".pcap", ".pcapng"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
