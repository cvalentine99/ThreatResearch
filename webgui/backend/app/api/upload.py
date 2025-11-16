"""PCAP upload endpoints"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
from datetime import datetime
import aiofiles
from pathlib import Path

from app.core.config import settings
from app.core.opensearch import opensearch_client
from app.workers.tasks import parse_pcap_task
from app.models.job import Job, JobStatus, JobCreate

router = APIRouter()


@router.post("/", response_model=Job)
async def upload_pcap(file: UploadFile = File(...)):
    """
    Upload a PCAP file for parsing

    The file is saved temporarily and a Celery task is queued to parse it.
    Returns a job ID that can be used to track parsing progress.
    """

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    upload_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}{file_ext}")

    try:
        # Stream file to disk
        async with aiofiles.open(upload_path, 'wb') as f:
            while chunk := await file.read(8192):  # 8KB chunks
                await f.write(chunk)

        # Get file size
        file_size = os.path.getsize(upload_path)

        # Validate file size
        if file_size > settings.MAX_UPLOAD_SIZE:
            os.remove(upload_path)
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE} bytes"
            )

        # Create job record
        job = Job(
            job_id=job_id,
            filename=file.filename,
            file_size=file_size,
            status=JobStatus.PENDING,
            created_at=datetime.utcnow()
        )

        # Index job to OpenSearch
        opensearch_client.index_document(
            index=settings.OPENSEARCH_INDEX_JOBS,
            document=job.model_dump(mode='json'),
            doc_id=job_id
        )

        # Queue parsing task
        parse_pcap_task.delay(job_id, upload_path, file.filename)

        return job

    except Exception as e:
        # Clean up on error
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{job_id}", response_model=Job)
async def get_job_status(job_id: str):
    """
    Get the status of a parsing job
    """

    try:
        result = opensearch_client.client.get(
            index=settings.OPENSEARCH_INDEX_JOBS,
            id=job_id
        )

        job_data = result['_source']
        return Job(**job_data)

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.get("/", response_model=list[Job])
async def list_jobs(limit: int = 50, offset: int = 0):
    """
    List recent parsing jobs
    """

    query = {
        "query": {"match_all": {}},
        "sort": [{"created_at": {"order": "desc"}}]
    }

    result = opensearch_client.search(
        index=settings.OPENSEARCH_INDEX_JOBS,
        query=query,
        size=limit,
        from_=offset
    )

    jobs = []
    for hit in result['hits']['hits']:
        jobs.append(Job(**hit['_source']))

    return jobs
