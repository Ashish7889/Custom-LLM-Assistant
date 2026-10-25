"""
tests/test_upload_api.py -- Unit tests for /upload endpoint and document-grounded QA.
"""

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_upload_txt_file():
    # 1. Create session
    sess_res = client.post("/sessions?user_id=upload_user&title=UploadSession")
    sess_id = sess_res.json()["session_id"]

    # 2. Upload file
    sample_content = b"The Indian Union Budget allocated 11.11 lakh crore for capital expenditure in 2024."
    files = {"file": ("budget_2024.txt", sample_content, "text/plain")}
    data = {"session_id": sess_id, "user_id": "upload_user"}

    res = client.post("/upload", files=files, data=data)
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["filename"] == "budget_2024.txt"
    assert res_data["chunks_count"] >= 1
    assert res_data["status"] == "indexed"

    # 3. Retrieve sources for session
    sources_res = client.get(f"/sessions/{sess_id}/sources")
    assert sources_res.status_code == 200
    sources = sources_res.json()
    assert len(sources) >= 1
    assert sources[0]["source_name"] == "budget_2024.txt"

    # 4. Clean up session and verify document deletion
    del_res = client.delete(f"/sessions/{sess_id}?user_id=upload_user")
    assert del_res.status_code == 200
    sources_after = client.get(f"/sessions/{sess_id}/sources").json()
    assert len(sources_after) == 0
