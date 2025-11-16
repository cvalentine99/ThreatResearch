"""OpenSearch client and utilities"""

from opensearchpy import OpenSearch, helpers
from typing import Dict, List, Any, Optional
import logging
from .config import settings

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """OpenSearch client wrapper"""

    def __init__(self):
        self.client = OpenSearch(
            hosts=[settings.OPENSEARCH_URL],
            http_compress=True,
            use_ssl=False,  # Enable in production with proper certs
            verify_certs=False,
            timeout=30
        )
        self._ensure_indices()

    def _ensure_indices(self):
        """Create indices if they don't exist"""

        # Packets index mapping
        packets_mapping = {
            "mappings": {
                "properties": {
                    "job_id": {"type": "keyword"},
                    "packet_num": {"type": "long"},
                    "timestamp": {"type": "date"},
                    "src_ip": {"type": "ip"},
                    "dst_ip": {"type": "ip"},
                    "src_port": {"type": "integer"},
                    "dst_port": {"type": "integer"},
                    "protocol": {"type": "keyword"},
                    "length": {"type": "integer"},
                    "tcp_flags": {"type": "keyword"},
                    "flow_hash": {"type": "keyword"},
                    "parsed_at": {"type": "date"}
                }
            }
        }

        # Flows index mapping
        flows_mapping = {
            "mappings": {
                "properties": {
                    "job_id": {"type": "keyword"},
                    "flow_hash": {"type": "keyword"},
                    "src_ip": {"type": "ip"},
                    "dst_ip": {"type": "ip"},
                    "src_port": {"type": "integer"},
                    "dst_port": {"type": "integer"},
                    "protocol": {"type": "keyword"},
                    "packet_count": {"type": "long"},
                    "total_bytes": {"type": "long"},
                    "first_seen": {"type": "date"},
                    "last_seen": {"type": "date"},
                    "duration_ms": {"type": "long"},
                    "created_at": {"type": "date"}
                }
            }
        }

        # Jobs index mapping
        jobs_mapping = {
            "mappings": {
                "properties": {
                    "job_id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "file_size": {"type": "long"},
                    "status": {"type": "keyword"},
                    "total_packets": {"type": "long"},
                    "parsed_packets": {"type": "long"},
                    "error_message": {"type": "text"},
                    "created_at": {"type": "date"},
                    "started_at": {"type": "date"},
                    "completed_at": {"type": "date"},
                    "parse_time_ms": {"type": "long"}
                }
            }
        }

        indices = {
            settings.OPENSEARCH_INDEX_PACKETS: packets_mapping,
            settings.OPENSEARCH_INDEX_FLOWS: flows_mapping,
            settings.OPENSEARCH_INDEX_JOBS: jobs_mapping
        }

        for index_name, mapping in indices.items():
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(index=index_name, body=mapping)
                logger.info(f"Created index: {index_name}")

    def index_document(self, index: str, document: Dict[str, Any], doc_id: Optional[str] = None):
        """Index a single document"""
        return self.client.index(
            index=index,
            body=document,
            id=doc_id,
            refresh=False  # Set to True for immediate visibility
        )

    def bulk_index(self, index: str, documents: List[Dict[str, Any]]):
        """Bulk index multiple documents"""
        actions = [
            {
                "_index": index,
                "_source": doc
            }
            for doc in documents
        ]

        success, failed = helpers.bulk(
            self.client,
            actions,
            chunk_size=5000,
            request_timeout=60
        )

        return {"success": success, "failed": failed}

    def search(self, index: str, query: Dict[str, Any], size: int = 100, from_: int = 0):
        """Execute a search query"""
        return self.client.search(
            index=index,
            body=query,
            size=size,
            from_=from_
        )

    def count(self, index: str, query: Optional[Dict[str, Any]] = None):
        """Count documents matching query"""
        body = {"query": query} if query else None
        return self.client.count(index=index, body=body)

    def update_document(self, index: str, doc_id: str, document: Dict[str, Any]):
        """Update a document"""
        return self.client.update(
            index=index,
            id=doc_id,
            body={"doc": document}
        )

    def get_aggregations(self, index: str, aggs: Dict[str, Any], query: Optional[Dict[str, Any]] = None):
        """Get aggregations"""
        body = {
            "size": 0,
            "aggs": aggs
        }
        if query:
            body["query"] = query

        return self.client.search(index=index, body=body)


# Global client instance
opensearch_client = OpenSearchClient()
