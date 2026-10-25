# Custom LLM Assistant - Financial Edition

A comprehensive 10-step financial large language model roadmap implementation, ranging from basic financial Q&A to a production-ready financial assistant with retrieval, tool calling, memory, and evaluation.

## 📁 Project Structure

```
my-project/
├── model.py              # 10.7M-param GPT transformer from scratch
├── tokenizer.py          # Character-level tokenizer
├── data.py               # Shakespeare corpus downloader → train.bin/val.bin
├── train.py              # Training loop (5000 steps on Shakespeare)
├── chat.py               # Interactive chat with base model
├── README.md             # This file
├── finetune/
│   ├── train.py          # LoRA fine-tuning (TinyLlama → assistant)
│   ├── chat.py           # Q&A chat with LoRA adapter
│   └── adapter/          # Trained LoRA weights (~60 MB)
├── finetune_finance/     # Finance roadmap implementation (Steps 1-10)
│   └── train.py          # LoRA finance fine-tuning framework
├── diagnostics/          # Step 1 outputs
│   ├── failure_log.json
│   └── diagnostics_report.md
├── data/
│   ├── input.txt         # Shakespeare text
│   ├── train.bin, val.bin
│   ├── finance_instructions.jsonl  # 26 finance concepts
│   └── finance_template.jsonl      # Chat-formatted finance Q&A
├── evaluation/           # Steps 5-10 outputs
│   ├── evaluate_finance.py
│   ├── benchmark_finance.py
│   └── benchmark_report.json
├── localpage.html        # Browser UI for localhost chat
├── fix_repetition.py     # Repetitive footer trimming + guardrails
├── rag_finance.py        # RAG financial document retrieval
├── tool_calling.py       # Stock price, financial metrics, ticker search
├── conversation_memory.py# Persistent chat history, portfolio, risk tolerance
└── evaluate_finance.py   # Validation metrics framework
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- Git
- Optional: CUDA-enabled GPU for faster inference

### 2. Install Dependencies

```bash
pip install torch transformers peft yfinance numpy
```

### 3. Run the LocalChat Server

```bash
python server.py
```

Then open http://localhost:5000 in your browser.

### 4. Or Run Directly with Python

```bash
python chat.py
```

## 🛠️ Available Commands

| Command | Description |
|---------|-------------|
| `python server.py` | Start Flask dev server on http://localhost:5000 |
| `python rag_finance.py` | Test RAG financial document retrieval |
| `python tool_calling.py` | Test stock price and financial metrics lookups |
| `python conversation_memory.py` | Test conversation memory functionality |
| `python benchmark_finance.py` | Run evaluation benchmark suite |
| `python evaluate_finance.py` | Run financial validation metrics |

## 📋 10-Step Roadmap Status

| Step | Title | Status |
|------|-------|--------|
| 1 | Diagnose generation failures on financial queries | ✅ Complete |
| 2 | Build clean financial instruction dataset | ✅ Complete |
| 3 | Improve financial chat template system | ✅ Complete |
| 4 | Improve LoRA fine-tuning for finance | ✅ Complete |
| 5 | Add validation/evaluation with financial metrics | ✅ Complete |
| 6 | Fix repetition / EOS for financial text | ✅ Complete |
| 7 | Add RAG with financial document retrieval | ✅ Complete |
| 8 | Add tool calling (stock/lookup) | ✅ Complete |
| 9 | Add conversation memory for finance | ✅ Complete |
| 10 | Benchmark against baseline financial models | ✅ Complete |

## 🔧 Production Deployment

### Docker Deployment (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir torch transformers peft yfinance numpy flask

EXPOSE 5000
CMD ["python", "server.py"]
```

Build and run:
```bash
docker build -t custom-llm-assistant .
docker run -p 5000:5000 custom-llm-assistant
```

### Environment Variables

Create a `.env` file if needed:

```
# HF_TOKEN for higher rate limits on Hugging Face
HF_TOKEN=your_hf_token_here
```

## 📊 Model Information

- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B parameters)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) with rank 16
- **Parameters**: ~10.7M base model, ~2.25M LoRA parameters (0.20% of total)
- **Tokenizer**: Character-level tokenizer
- **Context Window**: Standard LLM context

## 💡 Usage Examples

### Financial Q&A

```python
from fix_repetition import generate_financial_answer
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
mdl = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype="bfloat16").to("cuda")

answer = generate_financial_answer(mdl, tok, "What is compound interest?")
print(answer)
```

### Stock Price Lookup

```python
from tool_calling import get_stock_price, format_tool_result

result = get_stock_price("AAPL")
print(format_tool_result("get_stock_price", result))
```

### RAG with Financial Context

```python
from rag_finance import augment_prompt_with_rag

augmented = augment_prompt_with_rag("What is a bull market?")
print(augmented)  # Includes retrieved context from knowledge base
```

### Conversation Memory

```python
from conversation_memory import FinanceMemory

mem = FinanceMemory(user_id="user123")
mem.add_message("user", "What do you think about AAPL?")
mem.add_to_portfolio("AAPL", 10, 185.0)
mem.add_message("user", "How has it performed?")
summary = mem.get_summary()
```

## 🧪 Evaluation

Run the full benchmark suite:

```bash
python benchmark_finance.py
```

Results saved to `evaluation/benchmark_report.json` with metrics on:
- Factual accuracy on financial concepts
- Number/percentage extraction
- Ticker symbol recognition

## 📈 Roadmap Progress

The project implements all 10 steps of the financial assistant progression:

1. **Foundation**: Diagnosed generation failures, built clean instruction dataset
2. **Template System**: Improved chat formatting with system prompts
3. **Fine-tuning**: LoRA framework for financial adaptation
4. **Evaluation**: Metrics framework for factual accuracy, numbers, tickers
5. **Repair**: Repetitive footer trimming and EOS behavior fixes
6. **RAG**: Keyword-based financial document retrieval
7. **Tools**: Stock price, financial metrics, ticker search
8. **Memory**: Persistent history, portfolio tracking, risk tolerance
9. **Benchmarking**: Comparison against baseline models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m "Add some improvement"`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is open source. See the repository for license details.

## 🆘 Need Help?

- Check the `diagnostics/` folder for troubleshooting common issues
- Review `evaluation/` for benchmark results and metrics
- Open an issue in the GitHub repository for bug reports or feature requests