# FinSage: Evidence-Grounded Financial Intelligence Assistant

[![CI Quality Gate](https://github.com/Ashish7889/Custom-LLM-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashish7889/Custom-LLM-Assistant/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker Image](https://img.shields.io/badge/docker-ashishbhosle%2Ffinsage%3Alatest-2496ED.svg?logo=docker&logoColor=white)](https://hub.docker.com/r/ashishbhosle/finsage)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FinSage is a small-model, evidence-grounded financial intelligence assistant built for strict factual verification, statutory accuracy, and automated abstention when evidence is insufficient. It features a collaborative multi-agent Chartered Accountant (CA) firm architecture with dedicated parsers for Indian financial documents, deterministic tax engines, and predatory loan APR detectors.

**Primary Targets:**
* **Local Mode:** 6GB Laptop GPU (e.g., NVIDIA GeForce RTX 3060) running fine-tuned LoRA adapters (TinyLlama 1.1B, Qwen 2.5 1.5B, DeepSeek-R1 Distill 1.5B).
* **Cloud Mode:** Zero-cost CPU deployment (Render / Docker) running fast deterministic CA math engines and verified statutory rule grounding.

---

## 🛡️ Non-Negotiable Core Principles

1. **Evidence & Truth:** Facts derive strictly from verified document sources, regulatory portals, or deterministic tool execution; math runs in exact Python arithmetic. The model **never** invents numbers.
2. **Untrusted Data Isolation:** All retrieved text (web data, uploaded documents, voice transcripts) is treated as untrusted *DATA*. It cannot alter system instructions or invoke unauthorized tools.
3. **No Simulated Market Quotes:** Real-time financial data uses `yfinance` with TTL caching; network or ticker failures trigger explicit error boundaries, never hallucinated prices.
4. **Self-Verification & Abstention:** Responses are decomposed into structured claims and audited for numerical containment and citation validity. If unverified, FinSage strictly abstains.
5. **Regulatory Compliance & Privacy:** Educational and explanatory guidance only; never issues statutory audit certificates or personalized investment mandates. Document chunks are strictly in-memory and wiped when sessions end.

---

## 🏛️ System Architecture

```text
[ User Query / Microphone Voice / Financial Documents (PDF/DOCX/TXT) ]
                             │
                             ▼
     ┌───────────────────────────────────────────────────────────────┐
     │  FastAPI Gateway (:8000)      │ Rate limiting, CORS, SSE      │
     └───────────────────────────────┬───────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ Orchestrator Machine (Deterministic Transitions)                           │
│                                                                            │
│  [INGEST & SANITIZE]                                                       │
│     ├── Form 16 & Salary Slips (Part A & B Table Normalizer)               │
│     ├── Form 26AS Tax Credit Statements (TDS / TCS / Refunds)              │
│     ├── ITR Acknowledgments (ITR-1 Sahaj / ITR-4 Sugam)                    │
│     ├── Bank Statements (Credit / Debit Categorization)                    │
│     └── Prompt Injection Scanner (Payload Sanitization)                    │
│                                                                            │
│  [ROUTE & RETRIEVE]                                                        │
│     ├── In-Memory BM25 Hybrid Document Store (Isolated per Session)        │
│     ├── Deterministic Indian Tax Engine (Revised Section 87A Rebate)       │
│     └── Predatory Loan & APR Solvers (Newton-Raphson Effective APR)        │
│                                                                            │
│  [MULTI-AGENT CA FIRM INFERENCE]                                           │
│     ├── Agent 1: Intake & Routing (Deterministic / TinyLlama 1.1B)         │
│     ├── Agent 2: Junior Tax Analyst (Qwen 2.5 1.5B LoRA)                   │
│     └── Agent 3: Senior Partner CA (DeepSeek-R1 Distill 1.5B LoRA)         │
│                                                                            │
│  [STRICT VERIFICATION & ABSTAIN]                                           │
│     ├── Numerical Containment (All numbers ⊆ Sources ∪ Tools)              │
│     ├── Citation Map & Legal Section Audit                                 │
│     ├── Notice Risk Assessment (Section 143(1) Automated Scrutiny)         │
│     └── Auto-Repair Loop (≤ 2 attempts, then abstains)                     │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Roadmap & Implementation Status

| Phase | Milestone | Status | Description |
| :--- | :--- | :---: | :--- |
| **Phase 0** | **Cleanup & Repo Architecture** | ✅ **Complete** | Truthful docs, typed configuration, and decoupled module tree. |
| **Phase 1** | **Orchestrator + API + UI** | ✅ **Complete** | Deterministic state machine, FastAPI SSE `/chat`, decision engine, modern responsive UI. |
| **Phase 2** | **Ingest & Hybrid Retrieval** | ✅ **Complete** | PDF/DOCX/TXT parsers, magic byte validation, BM25 scoring, and isolated user document store. |
| **Phase 3** | **Verifier & Abstain Path** | ✅ **Complete** | Strict numerical containment (`numbers ⊆ sources ∪ tools`), citation verification, and auto-repair loop. |
| **Phase 4** | **Personal CA Tools & Debt Planner** | ✅ **Complete** | Old vs New Tax Regime, revised Section 87A ₹0 rebate up to ₹12.75L salaried, ITR selector, and Newton-Raphson APR solver. |
| **Phase 5** | **Voice Audio & Natural Dialogue** | ✅ **Complete** | Web Speech API microphone STT (en-IN), browser voice TTS readout, and multi-turn pronoun context resolution. |
| **Phase 6** | **Test Suite & Verification** | ✅ **Complete** | Comprehensive unit and integration test suite covering tools, verifier, API, and state machine transitions. |
| **Phase 7** | **Dual Model Personal CA Fine-Tuning** | ✅ **Complete** | Curated tax triplets; TinyLlama-1.1B and Qwen-2.5-1.5B fine-tuned on GPU with LoRA rank 32. |
| **Phase 8** | **Interactive Model Switcher** | ✅ **Complete** | Dynamic seamless model toggle (TinyLlama 1.1B, Qwen 2.5 1.5B, DeepSeek-R1 1.5B) in header UI, FastAPI routes, and VRAM management. |
| **Phase 9** | **Held-Out Evaluation Benchmark** | ✅ **Complete** | 50-item benchmark across 15 Tax, 10 Debt, 15 Scheme, and 10 Abstention queries with 0% invented numbers. |
| **Phase 10**| **Multi-Agent CA Practice & Cloud Deploy**| ✅ **Complete** | In-memory document parsers, cloud CPU fast-boot bypass, DockerHub image, GitHub Actions CI quality gate, and Render deployment. |

---

## 👥 Multi-Agent Chartered Accountant Firm Architecture

FinSage simulates a collaborative, professional Indian CA firm with specialized personas:

1. **Agent 1: Intake & Classification**:
   - Classifies queries into Tax, Debt, Scheme, Document QA, or Non-Finance categories.
   - Detects non-finance uploads (resumes, invoices, random text) and triggers immediate abstention.

2. **Agent 2: Junior Tax Analyst (Fine-tuned Qwen 2.5 1.5B / Cloud Engine)**:
   - Primary role: Information extraction, structured salary component breakdown (Basic, HRA, Special Allowance), Section 80C/80D limit mapping, and initial tax computation draft.

3. **Agent 3: Senior Review Partner (Fine-tuned DeepSeek-R1 1.5B / Cloud Engine)**:
   - Primary role: Audit defense, notice risk evaluation (Section 143(1) scrutiny flags), Section 54/54EC capital gains checks, and final compliance sign-off.

---

## 📑 Specialized Financial Document Parsers

Located in `ingest/`:

* **Form 16 & Salary Slips (`ingest/parser.py`)**:
  - Extracts Gross Salary, Standard Deduction, Taxable Income, and TDS deductions across Part A and Part B annexures.
* **Form 26AS Tax Credit Statement (`ingest/form26as_parser.py`)**:
  - Identifies Part A (Salary TDS), Part A1 (Interest TDS), Part B (TCS), and Part D (Refunds).
* **Income Tax Return (ITR-1 / ITR-4) Acknowledgment (`ingest/itr_parser.py`)**:
  - Extracts Assessment Year, total income, Chapter VI-A deductions, and net tax payable/refund.
* **Bank Statement Analyzer (`ingest/bank_stmt_parser.py`)**:
  - Normalizes transactions, separating salary inflows from recurring high-value debits.
* **Privacy Guarantee**:
  - All parsed chunks reside strictly in an ephemeral, in-memory `SessionDocStore`. When a session is closed or deleted, all chunks are purged from memory immediately.

---

## 🚀 Deployment

### Cloud Deployment (Render Free Tier)
FinSage is fully containerized and configured for automated continuous deployment on Render via Docker.

> [!NOTE]
> **Cold Start Advisory:** On Render's free tier, the web service enters standby after 15 minutes of inactivity. When waking from sleep, please allow ~45–60 seconds for the container to initialize.

### Docker Hub Image
The prebuilt container is publicly available:
```bash
docker pull ashishbhosle/finsage:latest
docker run -p 8000:8000 ashishbhosle/finsage:latest
```

---

## 🔄 CI/CD Quality Gate

Automated via GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):
* **Test Suite Quality Gate**: Every `push` and `pull_request` to `main` spins up an isolated Ubuntu runner and executes 18 automated statutory CA and math verification tests:
  ```bash
  python -m pytest tests/test_ca_tools.py tests/test_calculator.py tests/test_retrieval.py tests/test_ingest.py -v
  ```
* **Docker Container Build Integrity**: Validates multi-stage Docker build integrity on every commit to prevent deployment regressions.

---

## 💻 Local Quick Start

### 1. Requirements
* Python 3.10+
* (Optional) NVIDIA GPU with CUDA for local neural LoRA generation

### 2. Installation
```bash
git clone https://github.com/Ashish7889/Custom-LLM-Assistant.git
cd Custom-LLM-Assistant
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest tests/test_ca_tools.py tests/test_retrieval.py -v
```

### 4. Launch Application
```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```
Open **http://127.0.0.1:8000/** in your browser.

---

## ⚖️ Legal Disclaimer

FinSage is an educational and factual guidance assistant based on official statutory tax rates (Income Tax Department of India, RBI guidelines). It does not provide formal legal certification or statutory audit certificates. Users should reconcile output with Form 16 and official e-filing portals before filing statutory returns.
