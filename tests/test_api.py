"""
tests/test_api.py -- Unit tests for FastAPI endpoints.
"""

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["app"] == "FinSage"


def test_sessions_lifecycle():
    # 1. Create session
    res = client.post("/sessions?user_id=test_api_user&title=TestSession")
    assert res.status_code == 200
    sess_id = res.json()["session_id"]
    assert sess_id is not None

    # 2. List sessions
    res_list = client.get("/sessions?user_id=test_api_user")
    assert res_list.status_code == 200
    sessions = res_list.json()
    assert any(s["session_id"] == sess_id for s in sessions)

    # 3. Delete session
    res_del = client.delete(f"/sessions/{sess_id}?user_id=test_api_user")
    assert res_del.status_code == 200
    assert res_del.json()["success"] is True


def test_chat_sse_endpoint():
    res = client.post(
        "/chat",
        json={"prompt": "What is compound interest?", "user_id": "test_api_user"},
    )
    assert res.status_code == 200
    # SSE stream format
    assert "text/event-stream" in res.headers["content-type"]
    text = res.text
    assert "event: step" in text
    assert "event: done" in text
