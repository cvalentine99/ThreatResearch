"""Statistics and aggregation endpoints"""

from fastapi import APIRouter
from app.core.config import settings
from app.core.opensearch import opensearch_client

router = APIRouter()


@router.get("/stats/protocols")
async def get_protocol_distribution(job_id: str):
    """
    Get protocol distribution for a job
    """

    aggs = {
        "protocols": {
            "terms": {
                "field": "protocol",
                "size": 50
            }
        }
    }

    query = {"term": {"job_id": job_id}}

    result = opensearch_client.get_aggregations(
        index=settings.OPENSEARCH_INDEX_PACKETS,
        aggs=aggs,
        query=query
    )

    protocols = []
    for bucket in result['aggregations']['protocols']['buckets']:
        protocols.append({
            "protocol": bucket['key'],
            "count": bucket['doc_count']
        })

    return {"protocols": protocols}


@router.get("/stats/ips/top")
async def get_top_ips(job_id: str, direction: str = "src", limit: int = 20):
    """
    Get top source or destination IPs
    """

    field = f"{direction}_ip"

    aggs = {
        "top_ips": {
            "terms": {
                "field": field,
                "size": limit,
                "order": {"_count": "desc"}
            }
        }
    }

    query = {"term": {"job_id": job_id}}

    result = opensearch_client.get_aggregations(
        index=settings.OPENSEARCH_INDEX_PACKETS,
        aggs=aggs,
        query=query
    )

    ips = []
    for bucket in result['aggregations']['top_ips']['buckets']:
        ips.append({
            "ip": bucket['key'],
            "count": bucket['doc_count']
        })

    return {"direction": direction, "ips": ips}


@router.get("/stats/connections")
async def get_connection_matrix(job_id: str, limit: int = 100):
    """
    Get connection matrix (src_ip x dst_ip) for heatmap visualization
    """

    # Get unique flows
    query = {
        "query": {"term": {"job_id": job_id}},
        "size": limit,
        "sort": [{"packet_count": {"order": "desc"}}]
    }

    result = opensearch_client.search(
        index=settings.OPENSEARCH_INDEX_FLOWS,
        query=query,
        size=limit
    )

    # Build matrix data
    connections = []
    for hit in result['hits']['hits']:
        flow = hit['_source']
        connections.append({
            "src_ip": flow['src_ip'],
            "dst_ip": flow['dst_ip'],
            "packet_count": flow['packet_count'],
            "total_bytes": flow['total_bytes'],
            "protocol": flow['protocol']
        })

    return {"connections": connections}


@router.get("/stats/summary")
async def get_job_summary(job_id: str):
    """
    Get comprehensive summary statistics for a job
    """

    # Get job info
    job_result = opensearch_client.client.get(
        index=settings.OPENSEARCH_INDEX_JOBS,
        id=job_id
    )
    job = job_result['_source']

    # Get packet count
    packet_count = opensearch_client.count(
        index=settings.OPENSEARCH_INDEX_PACKETS,
        query={"term": {"job_id": job_id}}
    )

    # Get flow count
    flow_count = opensearch_client.count(
        index=settings.OPENSEARCH_INDEX_FLOWS,
        query={"term": {"job_id": job_id}}
    )

    # Get protocol distribution
    protocol_agg = opensearch_client.get_aggregations(
        index=settings.OPENSEARCH_INDEX_PACKETS,
        aggs={
            "protocols": {
                "terms": {"field": "protocol", "size": 10}
            }
        },
        query={"term": {"job_id": job_id}}
    )

    protocols = {}
    for bucket in protocol_agg['aggregations']['protocols']['buckets']:
        protocols[bucket['key']] = bucket['doc_count']

    return {
        "job_id": job_id,
        "filename": job['filename'],
        "status": job['status'],
        "total_packets": packet_count['count'],
        "total_flows": flow_count['count'],
        "protocols": protocols,
        "parse_time_ms": job.get('parse_time_ms', 0)
    }
