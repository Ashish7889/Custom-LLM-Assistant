"""
api/routes.py -- FastAPI endpoints for FinSage.
Endpoints:
- POST /chat (SSE streaming with orchestrator step events and verified sentences)
- GET /health
- GET /sessions, POST /sessions, DELETE /sessions/{session_id}
- GET / (Serves UI)
"""

import json
import os
import torch
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from api.schemas import ChatRequest, HealthResponse, SessionResponse
from orchestrator.state_machine import OrchestratorStateMachine
from memory.db import db
from config import cfg

router = APIRouter()
orchestrator = OrchestratorStateMachine()


@router.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the minimal accessible FinSage web client."""
    ui_path = Path(__file__).resolve().parent.parent / "ui" / "index.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="UI index.html not found.")
    return ui_path.read_text(encoding="utf-8")


@router.get("/health", response_model=HealthResponse)
async def health():
    """System diagnostic and hardware status."""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return HealthResponse(
        status="healthy",
        app=cfg.app_name,
        version=cfg.app_version,
        device=cfg.device,
        device_name=device_name,
        decision_backend=cfg.decision_backend,
        max_repairs=cfg.max_repairs,
    )


@router.post("/chat")
async def chat_sse(request: ChatRequest):
    """Real-time Server-Sent Events (SSE) streaming endpoint."""
    session_id = request.session_id or db.create_session(request.user_id, title=request.prompt[:32])
    
    # Persist user prompt
    db.add_message(
        session_id=session_id,
        user_id=request.user_id,
        role="user",
        content=request.prompt,
    )

    async def event_generator():
        collected_sentences = []
        collected_sources = []
        collected_tools = []
        abstained = False

        try:
            yield {
                "event": "session",
                "data": json.dumps({"session_id": session_id}),
            }

            for event in orchestrator.run(request.prompt, session_id, request.user_id):
                if event.event_type == "sentence":
                    collected_sentences.append(event.payload)
                elif event.event_type == "source_retrieved":
                    collected_sources.append(event.payload)
                elif event.event_type == "tool_result":
                    collected_tools.append(event.payload)
                elif event.event_type == "done":
                    abstained = event.payload.get("abstain", False)

                yield {
                    "event": event.event_type,
                    "data": json.dumps(event.payload),
                }

            # Persist assistant response
            full_content = " ".join([s.get("text", "") for s in collected_sentences])
            db.add_message(
                session_id=session_id,
                user_id=request.user_id,
                role="assistant",
                content=full_content,
                sentences=collected_sentences,
                sources=collected_sources,
                tool_results=collected_tools,
                verified=not abstained,
                abstained=abstained,
            )

        except Exception as e:
            err_data = {"error": str(e)}
            yield {
                "event": "error",
                "data": json.dumps(err_data),
            }

    return EventSourceResponse(
        event_generator(),
        headers={"Content-Type": "text/event-stream; charset=utf-8"},
    )


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user_id: str = Form("default_user"),
):
    """Upload and ingest a PDF, DOCX, or TXT file into the session's isolated document store."""
    from ingest.parser import DocumentIngestionEngine, IngestionError
    from retrieval.engine import doc_store

    content_bytes = await file.read()
    filename = file.filename or "uploaded_document.txt"

    try:
        engine = DocumentIngestionEngine()
        chunks = engine.ingest(content_bytes, filename)
        doc_store.add_chunks(session_id, chunks)

        max_page = max([c.page for c in chunks if c.page is not None] or [1])
        return {
            "filename": filename,
            "session_id": session_id,
            "chunks_count": len(chunks),
            "pages": max_page,
            "status": "indexed",
        }
    except IngestionError as ie:
        raise HTTPException(status_code=400, detail=str(ie))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document parsing failed: {str(e)}")


@router.get("/sessions/{session_id}/sources")
async def get_session_sources(session_id: str):
    """Retrieve all ingested document sources for a session."""
    from retrieval.engine import doc_store
    chunks = doc_store.get_chunks(session_id)
    return [c.model_dump() for c in chunks]


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(user_id: str = Query("default_user")):
    return db.list_sessions(user_id)


@router.post("/sessions", response_model=dict)
async def create_session(user_id: str = Query("default_user"), title: str = "New Financial Session"):
    sess_id = db.create_session(user_id, title)
    return {"session_id": sess_id, "user_id": user_id, "title": title}


@router.delete("/sessions/{session_id}", response_model=dict)
async def delete_session(session_id: str, user_id: str = Query("default_user")):
    from retrieval.engine import doc_store
    success = db.delete_session(session_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or forbidden")
    # Clean up memory document store for this session
    doc_store.delete_session(session_id)
    return {"success": True, "message": "Session and uploaded documents deleted."}
