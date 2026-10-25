"""
server.py -- Production-grade web server for the Financial Assistant.

Features:
- Real-time Server-Sent Events (SSE) token streaming at /chat/stream
- Live market tool execution & RAG status badges
- User portfolio inspection and management endpoints
- Dynamic hardware/model status check
- Static frontend delivery for localpage.html
"""

import json
import os
import torch
from flask import Flask, Response, jsonify, request, stream_with_context
from agent_engine import FinancialAgentEngine

app = Flask(__name__, static_folder=None)
engine = FinancialAgentEngine()


@app.route("/")
def index():
    """Serve the local assistant interface."""
    path = os.path.join(app.root_path, "localpage.html")
    with open(path, encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/status", methods=["GET"])
def status():
    """Return backend device and engine status."""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return jsonify({
        "status": "ready",
        "device": str(engine.device),
        "device_name": device_name,
        "adapter": engine.adapter_path,
        "portfolio_count": len(engine.memory.portfolio),
        "discussed_tickers": list(engine.memory.discussed_tickers),
    })


@app.route("/chat", methods=["POST"])
def chat_sync():
    """Backward-compatible synchronous chat endpoint."""
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        response = engine.chat(prompt)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat/stream", methods=["GET", "POST"])
def chat_stream():
    """SSE streaming endpoint for real-time tokens and tool events."""
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or "").strip()
    else:
        prompt = (request.args.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    def generate_events():
        try:
            for event in engine.stream_chat(prompt):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            err_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(err_event)}\n\n"

    return Response(
        stream_with_context(generate_events()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/portfolio", methods=["GET", "POST"])
def manage_portfolio():
    """Inspect or update the user portfolio."""
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        ticker = (body.get("ticker") or "").upper().strip()
        shares = float(body.get("shares") or 0)
        cost_basis = float(body.get("cost_basis") or 0)

        if not ticker or shares <= 0:
            return jsonify({"error": "Invalid ticker or shares"}), 400

        engine.memory.add_to_portfolio(ticker, shares, cost_basis)
        return jsonify({
            "success": True,
            "portfolio": engine.memory.portfolio,
            "summary": engine.memory.get_portfolio_summary(),
        })

    return jsonify({
        "portfolio": engine.memory.portfolio,
        "summary": engine.memory.get_portfolio_summary(),
        "discussed_tickers": sorted(list(engine.memory.discussed_tickers)),
        "risk_tolerance": engine.memory.risk_tolerance,
    })


@app.route("/api/reset", methods=["POST"])
def reset_conversation():
    """Reset chat history and start a fresh session."""
    engine.memory.history = []
    engine.memory.discussed_tickers = set()
    engine.memory._save()
    return jsonify({"success": True, "message": "Conversation history cleared."})


if __name__ == "__main__":
    print("=" * 64)
    print("  Custom Financial LLM Assistant Server")
    print("  Status: Ready on http://localhost:5000")
    print("  Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("=" * 64)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
