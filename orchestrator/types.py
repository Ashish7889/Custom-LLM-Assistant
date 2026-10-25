"""
orchestrator/types.py -- Typed definitions and data models for FinSage state machine.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class OrchestratorState(str, Enum):
    INGEST = "INGEST"
    ROUTE = "ROUTE"
    RETRIEVE_OR_TOOL = "RETRIEVE_OR_TOOL"
    GENERATE = "GENERATE"
    VERIFY = "VERIFY"
    REPAIR = "REPAIR"
    RESPOND = "RESPOND"
    ABSTAIN = "ABSTAIN"


class SourceChunk(BaseModel):
    id: int
    text: str
    source_name: str
    page: Optional[int] = None
    url: Optional[str] = None
    trusted_level: str = "verified"  # "official", "filings", "verified", "untrusted"
    date: Optional[str] = None


class ToolResult(BaseModel):
    tool_name: str
    input_args: Dict[str, Any]
    output_data: Dict[str, Any]
    formatted_text: str
    success: bool
    error: Optional[str] = None


class SentenceItem(BaseModel):
    text: str
    cites: List[int] = Field(default_factory=list)
    verified: bool = False
    warning: Optional[str] = None


class GeneratorOutput(BaseModel):
    abstain: bool = False
    sentences: List[SentenceItem] = Field(default_factory=list)
    raw_response: Optional[str] = None


class VerificationResult(BaseModel):
    passed: bool
    missing_citations: List[int] = Field(default_factory=list)
    unsupported_numbers: List[str] = Field(default_factory=list)
    claim_supported: bool = True
    feedback: Optional[str] = None


class ExecutionStep(BaseModel):
    state: OrchestratorState
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float


class StreamEvent(BaseModel):
    event_type: str  # "step", "decision", "tool_result", "token", "verified", "done", "error"
    payload: Dict[str, Any]
