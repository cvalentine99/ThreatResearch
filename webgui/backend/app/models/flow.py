"""Flow data models"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class FlowBase(BaseModel):
    """Base flow model"""
    job_id: str
    flow_hash: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str


class FlowStats(FlowBase):
    """Flow with statistics"""
    packet_count: int
    total_bytes: int
    first_seen: datetime
    last_seen: datetime
    duration_ms: int
    created_at: datetime

    class Config:
        from_attributes = True


class Flow(FlowStats):
    """Complete flow model"""
    pass
