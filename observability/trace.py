"""
observability/trace.py -- Structured per-request tracing for FinSage.
Tracks:
- Step-by-step state machine timeline and latency (ms)
- Verifier checks (numbers checked, citation validity, repairs attempted)
- Decision verdicts (task, choice, confidence, backend)
- Final delivery outcome (RESPOND vs ABSTAIN)
"""

import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("finsage.observability")


class TraceStep(BaseModel):
    state: str
    description: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class RequestTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    prompt: str
    created_at: float = Field(default_factory=time.time)
    steps: List[TraceStep] = Field(default_factory=list)
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    verifications: List[Dict[str, Any]] = Field(default_factory=list)
    repairs_count: int = 0
    final_verdict: str = "IN_PROGRESS"  # RESPOND, ABSTAIN, ERROR
    total_latency_ms: Optional[float] = None


class TraceManager:
    """In-memory trace recorder for monitoring and flywheel analysis."""

    def __init__(self):
        self._traces: Dict[str, RequestTrace] = {}

    def start_trace(self, session_id: str, user_id: str, prompt: str) -> RequestTrace:
        trace = RequestTrace(session_id=session_id, user_id=user_id, prompt=prompt)
        self._traces[trace.trace_id] = trace
        return trace

    def record_step_start(self, trace_id: str, state: str, description: str, details: Optional[Dict[str, Any]] = None) -> int:
        trace = self._traces.get(trace_id)
        if not trace:
            return -1
        step = TraceStep(
            state=state,
            description=description,
            start_time=time.time(),
            details=details or {},
        )
        trace.steps.append(step)
        return len(trace.steps) - 1

    def record_step_end(self, trace_id: str, step_index: int) -> None:
        trace = self._traces.get(trace_id)
        if not trace or step_index < 0 or step_index >= len(trace.steps):
            return
        step = trace.steps[step_index]
        step.end_time = time.time()
        step.duration_ms = round((step.end_time - step.start_time) * 1000.0, 2)

    def record_decision(self, trace_id: str, task: str, choice: str, confidence: float, backend: str) -> None:
        trace = self._traces.get(trace_id)
        if trace:
            trace.decisions.append({
                "task": task,
                "choice": choice,
                "confidence": confidence,
                "backend": backend,
                "timestamp": time.time(),
            })

    def record_verification(self, trace_id: str, passed: bool, unsupported_numbers: List[str], missing_cites: List[int], feedback: str) -> None:
        trace = self._traces.get(trace_id)
        if trace:
            trace.verifications.append({
                "passed": passed,
                "unsupported_numbers": unsupported_numbers,
                "missing_citations": missing_cites,
                "feedback": feedback,
                "timestamp": time.time(),
            })

    def complete_trace(self, trace_id: str, final_verdict: str) -> Optional[RequestTrace]:
        trace = self._traces.get(trace_id)
        if not trace:
            return None
        trace.final_verdict = final_verdict
        trace.total_latency_ms = round((time.time() - trace.created_at) * 1000.0, 2)
        logger.info(
            f"Trace {trace_id} completed: verdict={final_verdict} latency={trace.total_latency_ms}ms repairs={trace.repairs_count}"
        )
        return trace

    def get_trace(self, trace_id: str) -> Optional[RequestTrace]:
        return self._traces.get(trace_id)


tracer = TraceManager()
