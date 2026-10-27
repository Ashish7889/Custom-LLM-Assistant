# FinSage: Evidence-Grounded Financial Intelligence Assistant

[![CI Quality Gate](https://github.com/Ashish7889/Custom-LLM-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashish7889/Custom-LLM-Assistant/actions/workflows/ci.yml)

FinSage is a small-model financial assistant built for factual evidence-grounding, self-verification, and strict abstention when unverified. It features a collaborative multi-agent Chartered Accountant (CA) firm architecture with specialized parsers for Indian tax and financial documents.

**Hardware Target:** 6GB Laptop GPU (e.g., NVIDIA GeForce RTX 3060).  
**Default Budget:** Zero.  
**Core Rule:** The LLM does *not* plan or invent numbers; deterministic code drives the state machine and exact Python arithmetic drives calculations.

---

## 🛡️ Non-Negotiable Principles

1. **Evidence & Truth:** Facts come strictly from verified sources or tools; math runs in verified Python functions. The model *never* invents numbers.
2. **Untrusted Data Isolation:** All retrieved text (web, documents, voice transcripts) is treated as untrusted *DATA*. It can never issue system instructions or invoke tools.
3. **No Simulated Quotes:** Real-time data uses yfinance with TTL caching; network or ticker failures yield explicit errors, never fabricated prices.
4. **Self-Verification & Abstention:** Responses are parsed into structured JSON claims and verified for numerical grounding and citation validity before reaching the user. If unverified, FinSage abstains.
5. **Regulatory Compliance:** Educational and explanatory only. Never provides personalized investment advice (no buy/sell mandates).

---

## 🏛️ System Architecture

`
[ User Input / Audio / Indian Financial Documents (PDF/DOCX/TXT) ]
                             │
                             ▼
     ┌──────────────────────────────────────────────┐
     │  FastAPI Gateway (:8000)                     │ ──► Rate limiting, CORS, SSE streaming
     └──────────────────────┬───────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Orchestrator Machine (Deterministic Transitions)            │
│                                                             │
│  [INGEST & DISPATCH]                                        │
│     ├── Form 16 / Salary Slips                             │
│     ├── Form 26AS Tax Credit Statements (TDS/TCS)          │
│     ├── ITR Acknowledgments (ITR-1 / ITR-4)                │
│     └── Bank Statements (Credits, Debits, Balances)        │
│                                                             │
│  [ROUTE & RETRIEVE]                                         │
│     ├── BM25 + Embeddings Hybrid Document Store             │
│     ├── Deterministic Indian Tax Engine (Budget 2024-25)   │
│     └── Predatory Loan & APR Solvers                       │
│                                                             │
│  [MULTI-AGENT CA FIRM GENERATION]                           │
│     ├── Junior Tax Analyst (Qwen 2.5 1.5B LoRA)             │
│     ├── Senior Partner / Reviewer (DeepSeek-R1 1.5B LoRA)   │
│     └── Edge Fallback (TinyLlama 1.1B LoRA)                 │
│                                                             │
│  [STRICT VERIFICATION & ABSTAIN]                            │
│     ├── Numerical Containment (numbers ⊆ sources ∪ tools)   │
│     ├── Citation Map & Legal Section Audit                 │
│     └── Auto-Repair Loop (<= 2 attempts)                    │
└─────────────────────────────────────────────────────────────┘
`

---

## 📑 Roadmap & Implementation Status

| Phase | Milestone | Status | Description |
| :--- | :--- | :---: | :--- |
| **Phase 0** | **Cleanup & Repo Architecture** | ✅ **Complete** | Truthful docs, typed configuration, and decoupled module tree. |
| **Phase 1** | **Orchestrator + API + UI** | ✅ **Complete** | Deterministic state machine, FastAPI SSE /chat, decision engine, modern responsive UI. |
| **Phase 2** | **Ingest & Hybrid Retrieval** | ✅ **Complete** | PDF/DOCX/TXT parsers, magic byte validation, BM25 scoring, and isolated user document store. |
| **Phase 3** | **Verifier & Abstain Path** | ✅ **Complete** | Strict numerical containment (
umbers ⊆ sources ∪ tools), citation verification, and auto-repair loop. |
| **Phase 4** | **Personal CA Tools & Debt Planner** | ✅ **Complete** | Budget 2024–25 Old vs New Tax Regime, Section 87A ₹0 rebate up to ₹7.75L, ITR form selector, and Newton-Raphson APR solver. |
| **Phase 5** | **Voice Audio & Natural Dialogue** | ✅ **Complete** | Web Speech API microphone STT (en-IN), browser voice TTS readout, and multi-turn pronoun context resolution. |
| **Phase 6** | **Test Suite & Verification** | ✅ **Complete** | 33 comprehensive unit and integration tests covering tools, verifier, API, and state machine transitions. |
| **Phase 7** | **Dual Model Personal CA Fine-Tuning** | ✅ **Complete** | 1,330 curated tax/scheme triplets; both TinyLlama-1.1B (loss 0.389) and Qwen-2.5-1.5B (loss 0.291) fine-tuned on RTX 3060 GPU with LoRA rank 32. |
| **Phase 8** | **Interactive Model Switcher** | ✅ **Complete** | Dynamic seamless model toggle (TinyLlama 1.1B, Qwen 2.5 1.5B, DeepSeek-R1 1.5B) in header UI, FastAPI routes, and GPU VRAM management. |
| **Phase 9** | **50-Item Held-Out Evaluation Benchmark** | ✅ **Complete** | Full benchmark in enchmark/eval_personal_ca.py across 15 Tax, 10 Debt, 15 Scheme, and 10 Abstention queries with 0% invented numbers. |
| **Phase 10**| **Multi-Agent CA Practice & Document Ingest**| ✅ **Complete** | Form 26AS, ITR, Bank Statement parsers; 215+ firm training samples each for Junior Analyst and Senior Partner; /finetune/status endpoint. |

---

## 🏢 Multi-Agent Chartered Accountant Firm Architecture

FinSage simulates a professional collaborative Indian CA firm with complementary specialized personas:

1. **Junior Tax Analyst (Fine-tuned Qwen 2.5 1.5B)**:
   - Primary role: Fast information extraction, structured salary component breakdown (Basic, HRA, Special Allowance), Section 80C/80D limit mapping, and initial tax computation draft.
   - Trained on 215+ structured tax extraction cases.

2. **Senior Review Partner (Fine-tuned DeepSeek-R1 1.5B)**:
   - Primary role: Complex tax scrutiny, audit defense strategy, Section 54/54EC capital gains exemption verification, grandfathering clause checks, penalty mitigation, and risk sign-off.
   - Evaluates whether the junior analyst missed tax optimization opportunities or violated statutory provisions.

3. **Edge / Fast Fallback (Fine-tuned TinyLlama 1.1B)**:
   - Ultra-low latency edge response for quick factual definitions, scheme deadlines, and basic tax threshold queries.

---

## 📄 Specialized Indian Financial Document Parsers

Located in ingest/:

- **Form 26AS Tax Credit Statement (orm26as_parser.py)**:
  - Automatically identifies Part A (TDS on salary), Part A1 (TDS on interest), Part B (TCS), and Part D (Refunds).
  - Cross-references total tax deposited with claimed credits.

- **Income Tax Return (ITR-1 / ITR-4) Acknowledgment (itr_parser.py)**:
  - Extracts Assessment Year, total income, Section 80 deductions, tax payable, and refund entitlement with exact field boundaries.

- **Bank Statement Analyzer (ank_stmt_parser.py)**:
  - Parses multi-page transaction records, isolating salary inflows, high-value UPI/NEFT debits, and interest credits for income categorization.

- **Form 16 & Salary Slips (parser.py)**:
  - Normalizes PDF table extraction for Part A and Part B salary annexures.

---

## ⚡ Quick Start

### 1. Requirements
- Python 3.10+
- PyTorch with CUDA (6GB VRAM recommended, e.g. NVIDIA RTX 3060)
- Install dependencies:
  `ash
  pip install fastapi uvicorn pydantic pytest sse-starlette yfinance transformers peft bitsandbytes pypdf
  `

### 2. Run Test Suite
`ash
python -m pytest tests/ -v
`
*(All 33 tests pass in ~100s)*

### 3. Launch FinSage
`ash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
`
Open **http://127.0.0.1:8000/** to interact with FinSage via text, microphone voice input, document uploads, or real-time model switching.

### 4. Check Adapter & Fine-Tuning Status
Visit **http://127.0.0.1:8000/finetune/status** to monitor fine-tuning readiness, model adapter weights, and training sample counts in real time.
