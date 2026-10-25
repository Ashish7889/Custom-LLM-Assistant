# FinSage: Evidence-Grounded Financial Intelligence Assistant

FinSage is a small-model financial assistant built for factual evidence-grounding, self-verification, and strict abstention when unverified.

**Hardware Target:** 6GB Laptop GPU (e.g., NVIDIA GeForce RTX 3060).  
**Default Budget:** Zero.  
**Core Rule:** The LLM does *not* plan or invent numbers; deterministic code drives the state machine and exact Python arithmetic drives calculations.

---

## 🔒 Non-Negotiable Principles

1. **Evidence & Truth:** Facts come strictly from verified sources or tools; math runs in verified Python functions. The model *never* invents numbers.
2. **Untrusted Data Isolation:** All retrieved text (web, documents, voice transcripts) is treated as untrusted *DATA*. It can never issue system instructions or invoke tools.
3. **No Simulated Quotes:** Real-time data uses `yfinance` with TTL caching; network or ticker failures yield explicit errors, never fabricated prices.
4. **Self-Verification & Abstention:** Responses are parsed into structured JSON claims and verified for numerical grounding and citation validity before reaching the user. If unverified, FinSage abstains.
5. **Regulatory Compliance:** Educational and explanatory only. Never provides personalized investment advice (no buy/sell mandates).

---

## 🏛️ System Architecture

```
[ User Input / Audio / File ]
             │
             ▼
     ┌───────────────┐
     │  FastAPI API  │ ◄── Rate limiting, CORS, SSE streaming
     └───────┬───────┘
             ▼
┌─────────────────────────┐
│ Orchestrator Machine    │ ◄── Explicit deterministic transitions
│                         │
│  [INGEST]               │ ──> MIME verification, chunking (300-500 words)
│     │                   │
│  [ROUTE]                │ ──> Decisions Layer: decide(task, state, options)
│     │                   │
│  [RETRIEVE | TOOL]      │ ──> BM25 + Embeddings / Safe Python Calculator / yfinance
│     │                   │
│  [GENERATE]             │ ──> Constrained Grammar JSON: {"abstain": bool, "sentences": [...]}
│     │                   │
│  [VERIFY]               │ ──> Verify numbers ∈ sources, check cites, NLI claim support
│     │                   │
│  (Fail -> REPAIR <= 2)  │ ──> Targeted error feedback to generator
│     │                   │
│  [RESPOND | ABSTAIN]    │ ──> Return verified answer with confidence or structured abstention
└─────────────────────────┘
```

---

## 🚀 Roadmap & Implementation Status
| Phase | Milestone | Status | Description |
| :--- | :--- | :---: | :--- |
| **Phase 0** | **Cleanup & Repo Architecture** | ✅ **Complete** | Truthful docs, typed configuration, and decoupled module tree. |
| **Phase 1** | **Orchestrator + API + UI** | ✅ **Complete** | Explicit deterministic state machine, FastAPI SSE `/chat`, decision engine, modern responsive UI. |
| **Phase 2** | **Ingest & Hybrid Retrieval** | ✅ **Complete** | PDF/DOCX/TXT parsers, magic byte validation, BM25 scoring, and isolated user document store. |
| **Phase 3** | **Verifier & Abstain Path** | ✅ **Complete** | Strict numerical containment (`numbers ⊆ sources ∪ tools`), citation verification, and auto-repair loop. |
| **Phase 4** | **Personal CA Tools & Debt Planner** | ✅ **Complete** | Budget 2024–25 Old vs New Tax Regime, Section 87A ₹0 rebate up to ₹7.75L, ITR form selector, and Newton-Raphson flat vs reducing APR solver. |
| **Phase 5** | **Voice Audio & Natural Dialogue** | ✅ **Complete** | Web Speech API microphone STT (`en-IN`), browser voice TTS readout, and multi-turn pronoun context resolution. |
| **Phase 6** | **Test Suite & Verification** | ✅ **Complete** | 33 comprehensive unit and integration tests covering tools, verifier, API, and state machine transitions. |
| **Phase 7** | **Personal CA SFT Fine-Tuning** | ✅ **Complete** | 1,330 curated tax/scheme triplets; TinyLlama-1.1B trained on RTX 3060 GPU with LoRA rank 32, loss converging to 0.3895. |

---

## 🇮🇳 Personal AI Chartered Accountant Capabilities

1. **Indian Income Tax Optimization (Budget 2024–25)**:
   - Compares Old vs New Tax Regime with ₹75,000 standard deduction for salaried individuals.
   - Accurately computes Section 87A rebate guaranteeing **₹0 tax** on taxable income up to ₹7,00,000 (gross up to ₹7,75,000 for salaried).
   - Automatically suggests the correct ITR form: `ITR-1 (Sahaj)` for salaried citizens, `ITR-4 (Sugam)` for gig workers, shopkeepers, and small businesses.

2. **Predatory Debt Trap Prevention & Prepayment**:
   - Newton-Raphson effective reducing APR solver exposing predatory loan tricks (e.g. 12% flat rate = 21.46% reducing APR).
   - Prepayment impact calculator showing exact months shaved and total interest saved.

3. **Government Welfare Scheme Intelligence**:
   - Official guidance for PM-KISAN, Ayushman Bharat, Atal Pension Yojana (APY), and Sukanya Samriddhi Yojana (SSY).

4. **Multi-Turn Context & Natural Speech**:
   - Resolves follow-up pronouns (*"who can apply to it"*, *"what are the eligibility criteria"*) without false abstentions.
   - Interactive voice input (`🎤`) and spoken voice readout (`🔊`) with Indian English cadence.

---

## 🛠️ Quick Start

### 1. Requirements
- Python 3.10+
- PyTorch with CUDA (6GB VRAM recommended, e.g. NVIDIA RTX 3060)
- Install dependencies:
  ```bash
  pip install fastapi uvicorn pydantic pytest sse-starlette yfinance transformers peft bitsandbytes
  ```

### 2. Run Test Suite
```bash
python -m pytest tests/ -v
```
*(All 33 tests pass in ~100s)*

### 3. Launch FinSage
```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000/`** to interact with FinSage via text, microphone voice input, or audio readout.