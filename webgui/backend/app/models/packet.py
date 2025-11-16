"""Packet data models"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class PacketBase(BaseModel):
    """Base packet model"""
    job_id: str
    packet_num: int
    timestamp: Optional[datetime] = None
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    length: int
    tcp_flags: Optional[str] = None
    flow_hash: str


class PacketCreate(PacketBase):
    """Packet creation model"""
    pass


class Packet(PacketBase):
    """Packet model with metadata"""
    parsed_at: datetime

    class Config:
        from_attributes = True
