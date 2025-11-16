"""Celery tasks for PCAP parsing"""

import subprocess
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any
from .celery_app import celery_app
from app.core.config import settings
from app.core.opensearch import opensearch_client
from app.models.job import JobStatus

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="tasks.parse_pcap")
def parse_pcap_task(self, job_id: str, filepath: str, filename: str):
    """
    Parse PCAP file using CUDA parser and index results to OpenSearch

    Args:
        job_id: Unique job identifier
        filepath: Path to uploaded PCAP file
        filename: Original filename
    """

    logger.info(f"Starting PCAP parse job {job_id} for file {filename}")

    # Update job status to RUNNING
    opensearch_client.update_document(
        index=settings.OPENSEARCH_INDEX_JOBS,
        doc_id=job_id,
        document={
            "status": JobStatus.RUNNING.value,
            "started_at": datetime.utcnow().isoformat()
        }
    )

    try:
        # Execute CUDA parser
        cuda_parser = settings.CUDA_PARSER_PATH
        batch_size = settings.CUDA_BATCH_SIZE

        cmd = [cuda_parser, filepath, str(batch_size)]

        logger.info(f"Executing: {' '.join(cmd)}")

        start_time = datetime.utcnow()

        # Run parser and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            check=True
        )

        end_time = datetime.utcnow()
        parse_time_ms = int((end_time - start_time).total_seconds() * 1000)

        # Read JSON output from CUDA parser
        json_output_path = filepath + ".json"
        parsed_data = _read_cuda_json_output(json_output_path)

        logger.info(f"CUDA parser completed: {parsed_data.get('total_packets', 0)} packets")

        # Index packets to OpenSearch
        packets_indexed = _index_packets(job_id, parsed_data.get('packets', []))

        # Calculate flows and index them
        flows_indexed = _index_flows(job_id, parsed_data.get('packets', []))

        # Update job status to COMPLETED
        opensearch_client.update_document(
            index=settings.OPENSEARCH_INDEX_JOBS,
            doc_id=job_id,
            document={
                "status": JobStatus.COMPLETED.value,
                "total_packets": parsed_data.get('total_packets', 0),
                "parsed_packets": packets_indexed,
                "completed_at": datetime.utcnow().isoformat(),
                "parse_time_ms": parse_time_ms
            }
        )

        logger.info(f"Job {job_id} completed: {packets_indexed} packets, {flows_indexed} flows")

        # Clean up uploaded file and JSON output
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Removed temporary file: {filepath}")

        if os.path.exists(json_output_path):
            os.remove(json_output_path)
            logger.info(f"Removed temporary JSON file: {json_output_path}")

        return {
            "job_id": job_id,
            "status": "completed",
            "packets_indexed": packets_indexed,
            "flows_indexed": flows_indexed,
            "parse_time_ms": parse_time_ms
        }

    except subprocess.TimeoutExpired:
        error_msg = f"PCAP parsing timed out after 1 hour"
        logger.error(f"Job {job_id} failed: {error_msg}")
        _update_job_failed(job_id, error_msg)
        raise

    except subprocess.CalledProcessError as e:
        error_msg = f"CUDA parser failed: {e.stderr}"
        logger.error(f"Job {job_id} failed: {error_msg}")
        _update_job_failed(job_id, error_msg)
        raise

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Job {job_id} failed: {error_msg}")
        _update_job_failed(job_id, error_msg)
        raise


def _read_cuda_json_output(json_path: str) -> Dict[str, Any]:
    """Read and parse JSON output from CUDA parser using streaming to avoid memory issues"""

    if not os.path.exists(json_path):
        logger.error(f"JSON output file not found: {json_path}")
        return {"packets": [], "total_packets": 0}

    try:
        import ijson

        packets = []
        packet_num = 0

        logger.info(f"Streaming JSON data from {json_path}")

        with open(json_path, 'rb') as f:
            # Stream parse the packets array
            parser = ijson.items(f, 'packets.item')

            for packet in parser:
                packet['packet_num'] = packet_num
                packet['length'] = 0  # Not available in ParsedPacket
                packets.append(packet)
                packet_num += 1

                # Log progress every 100k packets
                if packet_num % 100000 == 0:
                    logger.info(f"Parsed {packet_num} packets from JSON")

        logger.info(f"Completed JSON parsing: {len(packets)} packets")

        return {
            "packets": packets,
            "total_packets": len(packets)
        }

    except ImportError:
        # Fallback to standard json if ijson not available - but limit to prevent memory issues
        logger.warning("ijson not available, using standard json (may cause memory issues)")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            packets = data.get('packets', [])[:100000]  # Limit to 100k packets
            logger.warning(f"Limited to {len(packets)} packets due to memory constraints")

            for i, packet in enumerate(packets):
                packet['packet_num'] = i
                packet['length'] = 0

            return {
                "packets": packets,
                "total_packets": data.get('total', len(packets))
            }
        except Exception as e:
            logger.error(f"Failed to read JSON output: {e}")
            return {"packets": [], "total_packets": 0}

    except Exception as e:
        logger.error(f"Failed to stream JSON output: {e}")
        return {"packets": [], "total_packets": 0}


def _index_packets(job_id: str, packets: list) -> int:
    """Index packets to OpenSearch"""

    if not packets:
        return 0

    # Add job_id and timestamp to each packet
    now = datetime.utcnow().isoformat()
    documents = []

    for packet in packets:
        doc = {
            "job_id": job_id,
            "parsed_at": now,
            **packet
        }
        documents.append(doc)

    # Bulk index
    result = opensearch_client.bulk_index(
        index=settings.OPENSEARCH_INDEX_PACKETS,
        documents=documents
    )

    logger.info(f"Indexed {result['success']} packets for job {job_id}")
    return result['success']


def _index_flows(job_id: str, packets: list) -> int:
    """Calculate flow statistics and index to OpenSearch"""

    if not packets:
        return 0

    # Group packets by flow_hash
    flows = {}

    for packet in packets:
        flow_hash = packet.get('flow_hash')
        if not flow_hash:
            continue

        if flow_hash not in flows:
            flows[flow_hash] = {
                "flow_hash": flow_hash,
                "src_ip": packet['src_ip'],
                "dst_ip": packet['dst_ip'],
                "src_port": packet['src_port'],
                "dst_port": packet['dst_port'],
                "protocol": packet['protocol'],
                "packet_count": 0,
                "total_bytes": 0,
                "first_seen": None,
                "last_seen": None
            }

        flow = flows[flow_hash]
        flow["packet_count"] += 1
        flow["total_bytes"] += packet.get('length', 0)

    # Convert flows to documents
    now = datetime.utcnow().isoformat()
    documents = []

    for flow_hash, flow in flows.items():
        doc = {
            "job_id": job_id,
            "created_at": now,
            "first_seen": now,  # Placeholder - would need packet timestamps
            "last_seen": now,
            "duration_ms": 0,
            **flow
        }
        documents.append(doc)

    # Bulk index
    result = opensearch_client.bulk_index(
        index=settings.OPENSEARCH_INDEX_FLOWS,
        documents=documents
    )

    logger.info(f"Indexed {result['success']} flows for job {job_id}")
    return result['success']


def _update_job_failed(job_id: str, error_message: str):
    """Update job status to FAILED"""

    opensearch_client.update_document(
        index=settings.OPENSEARCH_INDEX_JOBS,
        doc_id=job_id,
        document={
            "status": JobStatus.FAILED.value,
            "error_message": error_message,
            "completed_at": datetime.utcnow().isoformat()
        }
    )
