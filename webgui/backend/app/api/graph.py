"""Network graph/topology endpoints"""

from fastapi import APIRouter, Query
from typing import List, Dict, Any
from app.core.config import settings
from app.core.opensearch import opensearch_client

router = APIRouter()


@router.get("/graph/topology")
async def get_network_topology(
    job_id: str,
    limit: int = Query(500, le=2000)
):
    """
    Get network topology graph data for Cytoscape.js

    Returns nodes (IP addresses) and edges (flows) in Cytoscape format
    """

    # Get top flows
    query = {
        "query": {"term": {"job_id": job_id}},
        "sort": [{"packet_count": {"order": "desc"}}]
    }

    result = opensearch_client.search(
        index=settings.OPENSEARCH_INDEX_FLOWS,
        query=query,
        size=limit
    )

    # Extract unique IPs and flows
    nodes_set = set()
    edges = []

    for hit in result['hits']['hits']:
        flow = hit['_source']

        src_ip = flow['src_ip']
        dst_ip = flow['dst_ip']

        nodes_set.add(src_ip)
        nodes_set.add(dst_ip)

        # Create edge
        edges.append({
            "data": {
                "id": flow['flow_hash'],
                "source": src_ip,
                "target": dst_ip,
                "protocol": flow['protocol'],
                "packet_count": flow['packet_count'],
                "total_bytes": flow['total_bytes'],
                "label": f"{flow['protocol']} ({flow['packet_count']} pkts)"
            }
        })

    # Create nodes
    nodes = []
    for ip in nodes_set:
        # Classify node type (internal vs external)
        # Simple heuristic: private IPs are internal
        is_internal = _is_private_ip(ip)

        nodes.append({
            "data": {
                "id": ip,
                "label": ip,
                "type": "internal" if is_internal else "external"
            }
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges)
    }


def _is_private_ip(ip: str) -> bool:
    """Check if IP is in private range (RFC 1918)"""

    parts = ip.split('.')
    if len(parts) != 4:
        return False

    try:
        first = int(parts[0])
        second = int(parts[1])

        # 10.0.0.0/8
        if first == 10:
            return True

        # 172.16.0.0/12
        if first == 172 and 16 <= second <= 31:
            return True

        # 192.168.0.0/16
        if first == 192 and second == 168:
            return True

        return False

    except ValueError:
        return False
