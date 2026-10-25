"""
server.py -- a local web chat that runs your LoRA assistant.

Start it from your project folder and open http://localhost:5000 in a browser.

    python server.py

Two things run on http://localhost:5000:

1. The HTML page ./localpage.html (file at the project root) — a plain browser
   chat box. Type a question, press Enter, the page fetches /chat.
2. The /chat endpoint (returned as the page's /chat response) — takes
   the user text, runs the fine-tuned assistant, and returns the generated answer.

No internet needed after the first download — TinyLlama is cached locally.
"""
import os

import torch
from peft import PeftModel
from flask import Flask, jsonify, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = "finetune/adapter"

SYSTEM = "You are a helpful, friendly assistant."
GEN_KW = dict(max_length=None, do_sample=True, temperature=0.7, top_p=0.9,
              pad_token_id=None, eos_token_id=None, min_length=0, use_cache=True)

app = Flask(__name__)
_model = None
_tokenizer = None


def run(user_text):
    global _model, _tokenizer
    if _model is None:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        mdl = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16).to("cuda")
        _model = PeftModel.from_pretrained(mdl, ADAPTER)
        _model.eval()
        _tokenizer = tok
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}]
    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = _tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    mask = (ids != _tokenizer.pad_token_id).long()
    GEN_KW["max_length"] = ids.shape[1] + 300
    out = _model.generate(ids, attention_mask=mask, **GEN_KW)
    answer = _tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return answer


@app.route("/")
def index():
    with open(os.path.join(app.root_path, "localpage.html"), encoding="utf-8") as f:
        template = f.read()
    return template


@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    user_text = (body.get("prompt") or "").strip()
    if not user_text:
        return jsonify({"error": "no prompt"}), 400
    answer = run(user_text)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    print("Server starting on http://localhost:5000 — open it in a browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
