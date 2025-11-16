"""Flow query endpoints"""

from fastapi import APIRouter, Query
from typing import Optional
from app.core.config import settings
from app.core.opensearch import opensearch_client

router = APIRouter()


@router.get("/flows")
async def get_flows(
    job_id: Optional[str] = None,
    src_ip: Optional[str] = None,
    dst_ip: Optional[str] = None,
    protocol: Optional[str] = None,
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    Query flows with filters
    """

    # Build query
    must_clauses = []

    if job_id:
        must_clauses.append({"term": {"job_id": job_id}})

    if src_ip:
        must_clauses.append({"term": {"src_ip": src_ip}})

    if dst_ip:
        must_clauses.append({"term": {"dst_ip": dst_ip}})

    if protocol:
        must_clauses.append({"term": {"protocol": protocol}})

    query = {
        "query": {
            "bool": {
                "must": must_clauses if must_clauses else [{"match_all": {}}]
            }
        },
        "sort": [{"packet_count": {"order": "desc"}}]
    }

    result = opensearch_client.search(
        index=settings.OPENSEARCH_INDEX_FLOWS,
        query=query,
        size=limit,
        from_=offset
    )

    flows = [hit['_source'] for hit in result['hits']['hits']]

    return {
        "total": result['hits']['total']['value'],
        "flows": flows,
        "limit": limit,
        "offset": offset
    }


@router.get("/flows/top")
async def get_top_flows(
    job_id: str,
    metric: str = Query("packet_count", regex="^(packet_count|total_bytes)$"),
    limit: int = Query(10, le=100)
):
    """
    Get top flows by packet count or bytes
    """

    query = {
        "query": {"term": {"job_id": job_id}},
        "sort": [{metric: {"order": "desc"}}]
    }

    result = opensearch_client.search(
        index=settings.OPENSEARCH_INDEX_FLOWS,
        query=query,
        size=limit
    )

    flows = [hit['_source'] for hit in result['hits']['hits']]

    return {
        "metric": metric,
        "flows": flows
    }
