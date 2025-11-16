"""Job data models"""

from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import Optional


class JobStatus(str, Enum):
    """Job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobCreate(BaseModel):
    """Job creation model"""
    filename: str
    file_size: int


class JobUpdate(BaseModel):
    """Job update model"""
    status: Optional[JobStatus] = None
    total_packets: Optional[int] = None
    parsed_packets: Optional[int] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parse_time_ms: Optional[int] = None


class Job(BaseModel):
    """Job model"""
    job_id: str
    filename: str
    file_size: int
    status: JobStatus
    total_packets: int = 0
    parsed_packets: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parse_time_ms: Optional[int] = None

    class Config:
        from_attributes = True
        use_enum_values = True
