"""Packet query endpoints"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from app.core.config import settings
from app.core.opensearch import opensearch_client
from app.models.packet import Packet

router = APIRouter()


@router.get("/packets", response_model=dict)
async def get_packets(
    job_id: Optional[str] = None,
    src_ip: Optional[str] = None,
    dst_ip: Optional[str] = None,
    protocol: Optional[str] = None,
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    Query packets with filters

    Returns paginated packet list with total count
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
        "sort": [{"packet_num": {"order": "asc"}}]
    }

    # Execute search
    result = opensearch_client.search(
        index=settings.OPENSEARCH_INDEX_PACKETS,
        query=query,
        size=limit,
        from_=offset
    )

    # Extract packets
    packets = []
    for hit in result['hits']['hits']:
        packets.append(hit['_source'])

    return {
        "total": result['hits']['total']['value'],
        "packets": packets,
        "limit": limit,
        "offset": offset
    }


@router.get("/packets/conversation")
async def get_conversation(
    job_id: str,
    src_ip: str,
    dst_ip: str,
    limit: int = Query(1000, le=10000)
):
    """
    Get all packets in a conversation between two IPs (bidirectional)
    """

    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"job_id": job_id}}
                ],
                "should": [
                    {
                        "bool": {
                            "must": [
                                {"term": {"src_ip": src_ip}},
                                {"term": {"dst_ip": dst_ip}}
                            ]
                        }
                    },
                    {
                        "bool": {
                            "must": [
                                {"term": {"src_ip": dst_ip}},
                                {"term": {"dst_ip": src_ip}}
                            ]
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "sort": [{"packet_num": {"order": "asc"}}]
    }

    result = opensearch_client.search(
        index=settings.OPENSEARCH_INDEX_PACKETS,
        query=query,
        size=limit
    )

    packets = [hit['_source'] for hit in result['hits']['hits']]

    return {
        "total": result['hits']['total']['value'],
        "packets": packets
    }
