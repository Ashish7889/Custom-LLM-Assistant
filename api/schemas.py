"""
api/schemas.py -- Pydantic request and response schemas for FastAPI.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000, description="User financial query or command")
    session_id: Optional[str] = Field(None, description="Active session ID")
    user_id: str = Field("default_user", description="Authenticated user ID")
    model: Optional[str] = Field(None, description="Model key to use for inference: tinyllama, qwen, or deepseek_r1")
    web_search: bool = Field(True, description="Enable live internet search verification")


class ModelSwitchRequest(BaseModel):
    model: str = Field(..., description="Target model key: tinyllama, qwen, or deepseek_r1")


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    title: str
    created_at: float
    updated_at: float


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    device: str
    device_name: str
    decision_backend: str
    max_repairs: int


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    sentences: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    verified: bool
    abstained: bool
    created_at: float
