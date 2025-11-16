"""Server-Sent Events (SSE) endpoints for real-time updates"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import asyncio
import json
from app.core.config import settings
from app.core.opensearch import opensearch_client

router = APIRouter()


async def job_status_stream(job_id: str) -> AsyncGenerator[str, None]:
    """
    Stream job status updates

    Polls OpenSearch every 2 seconds and sends updates to client
    """

    last_status = None

    while True:
        try:
            # Query job status
            result = opensearch_client.client.get(
                index=settings.OPENSEARCH_INDEX_JOBS,
                id=job_id
            )

            job_data = result['_source']
            current_status = job_data['status']

            # Send update if status changed
            if current_status != last_status:
                event_data = {
                    "job_id": job_id,
                    "status": current_status,
                    "parsed_packets": job_data.get('parsed_packets', 0),
                    "total_packets": job_data.get('total_packets', 0)
                }

                yield f"data: {json.dumps(event_data)}\n\n"
                last_status = current_status

                # Stop streaming if job completed or failed
                if current_status in ['completed', 'failed']:
                    break

            # Wait before next poll
            await asyncio.sleep(2)

        except Exception as e:
            # Send error event
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
            break


@router.get("/stream/job/{job_id}")
async def stream_job_status(job_id: str):
    """
    Stream real-time job status updates via SSE

    Client should use EventSource to connect to this endpoint
    """

    return StreamingResponse(
        job_status_stream(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
