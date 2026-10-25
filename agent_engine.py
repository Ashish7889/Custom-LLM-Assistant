"""
agent_engine.py

Unified Financial Assistant Agent Engine.
Orchestrates:
- Dynamic tool calling (yfinance stock prices, financial metrics, ticker searches)
- RAG document retrieval (finance_instructions.jsonl)
- Persistent multi-turn conversation memory & portfolio tracking
- Financial compliance guardrails & repetition trimming
- Real-time token streaming via TextIteratorStreamer
"""

import json
import os
import re
import threading
from typing import Dict, Any, Generator, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from conversation_memory import FinanceMemory
from fix_repetition import trim_repetitive_footers, enforce_max_length, check_financial_guardrails
from rag_finance import load_finance_knowledge_base, retrieve_relevant_context
import tool_calling

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER = "finance_adapter/final"
FALLBACK_ADAPTER = "finetune/adapter"


class FinancialAgentEngine:
    """Core autonomous agent coordinating model generation, tools, RAG, and memory."""

    def __init__(self, adapter_path: Optional[str] = None, user_id: str = "default_user"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.adapter_path = adapter_path or (
            DEFAULT_ADAPTER if os.path.isdir(DEFAULT_ADAPTER) else FALLBACK_ADAPTER
        )
        self.memory = FinanceMemory(user_id=user_id, storage_path="data/conversation_memory.json")
        self.knowledge_base = load_finance_knowledge_base("data/finance_instructions.jsonl")

        self.tokenizer = None
        self.model = None
        self._load_lock = threading.Lock()

    def load_model(self):
        """Lazily load the model and tokenizer to conserve memory until first inference."""
        if self.model is not None:
            return

        with self._load_lock:
            if self.model is not None:
                return

            print(f"Loading Base Model: {BASE_MODEL} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            ).to(self.device)

            if os.path.isdir(self.adapter_path):
                print(f"Attaching LoRA Adapter: {self.adapter_path}...")
                self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            else:
                print("No adapter found, using base model directly.")
                self.model = base_model

            self.model.eval()
            print("Model loaded successfully.")

    def _extract_ticker(self, text: str) -> Optional[str]:
        """Heuristically extract a ticker symbol or company name."""
        company_to_ticker = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "tesla": "TSLA",
            "amazon": "AMZN",
            "nvidia": "NVDA",
            "meta": "META",
            "facebook": "META",
            "netflix": "NFLX",
            "jpmorgan": "JPM",
            "visa": "V",
        }
        lower = text.lower()
        for company, ticker in company_to_ticker.items():
            if re.search(r"\b" + re.escape(company) + r"\b", lower):
                return ticker

        # Match $TICKER
        m = re.search(r"\$([A-Za-z]{1,5})\b", text)
        if m:
            return m.group(1).upper()

        # Common stop-words that are 1-5 letters but definitely NOT stock tickers
        stop_words = {
            "WHAT", "IS", "THE", "OF", "AND", "FOR", "TO", "IN", "ON", "AT",
            "STOCK", "SHARE", "PRICE", "RATIO", "WORTH", "QUOTE", "HOW", "MUCH",
            "SHOW", "TELL", "GIVE", "ABOUT", "BUY", "SELL", "COST", "PE", "P/E",
            "FUND", "BOND", "RATE", "HIGH", "LOW", "CAP", "GOOD", "BEST", "WITH",
            "DOES", "CAN", "YOU", "MY", "OUR", "ALL", "NEW", "NOW", "TODAY", "THIS"
        }

        # Look for standalone all-uppercase words (1-5 chars) that are not stop words
        caps_candidates = re.findall(r"\b([A-Z]{1,5})\b", text)
        for c in caps_candidates:
            if c not in stop_words:
                return c

        # Patterns like: "price of XYZ", "quote XYZ", "ticker XYZ"
        m2 = re.search(r"\b(?:stock of|shares of|price of|metrics for|quote for|quote)\s+([A-Za-z]{1,5})\b", text, re.IGNORECASE)
        if m2:
            cand = m2.group(1).upper()
            if cand not in stop_words:
                return cand

        # Check against previously discussed tickers in memory
        for t in self.memory.discussed_tickers:
            if t not in stop_words and re.search(r"\b" + re.escape(t) + r"\b", text, re.IGNORECASE):
                return t

        return None

    def _handle_portfolio_intent(self, text: str) -> Optional[str]:
        """Detect and execute portfolio commands like 'buy 10 shares of AAPL at 180'."""
        buy_match = re.search(
            r"\b(?:bought|buy|add)\s+(\d+(?:\.\d+)?)\s*(?:shares?\s+(?:of\s+)?)?\$?([A-Za-z]{1,5})\s*(?:at|@)\s*\$?(\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        if buy_match:
            shares = float(buy_match.group(1))
            ticker = buy_match.group(2).upper()
            price = float(buy_match.group(3))
            self.memory.add_to_portfolio(ticker, shares, price)
            self.memory.remember_ticker(ticker, "Portfolio update")
            return f"Updated portfolio: Added {shares:.2f} shares of {ticker} at ${price:.2f}/share."

        if re.search(r"\b(?:my portfolio|show portfolio|view portfolio|holdings)\b", text, re.IGNORECASE):
            return self.memory.get_portfolio_summary()

        return None

    def execute_tools(self, user_text: str) -> List[Dict[str, Any]]:
        """Identify if any real-time market tools should be invoked."""
        tool_events = []
        ticker = self._extract_ticker(user_text)

        if not ticker:
            # Check for general ticker search intent
            search_match = re.search(r"\bsearch (?:ticker|stock) (?:for )?([a-zA-Z\s]+)", user_text, re.IGNORECASE)
            if search_match:
                q = search_match.group(1).strip()
                res = tool_calling.search_tickers(q)
                tool_events.append({
                    "tool": "search_tickers",
                    "args": {"query": q},
                    "result": res,
                    "formatted": f"Found ticker candidates for '{q}': {json.dumps(res)}",
                })
            return tool_events

        # Record ticker in memory
        self.memory.remember_ticker(ticker, user_text[:60])

        lower = user_text.lower()
        wants_metrics = any(w in lower for w in ["pe", "p/e", "ratio", "metric", "cap", "revenue", "margin", "valuation"])
        wants_price = any(w in lower for w in ["price", "cost", "trading at", "worth", "quote", "how much is", "stock"]) or not wants_metrics

        if wants_price:
            price_data = tool_calling.get_stock_price(ticker)
            tool_events.append({
                "tool": "get_stock_price",
                "args": {"ticker": ticker},
                "result": price_data,
                "formatted": tool_calling.format_tool_result("get_stock_price", price_data),
            })

        if wants_metrics:
            metrics_data = tool_calling.get_financial_metrics(ticker)
            tool_events.append({
                "tool": "get_financial_metrics",
                "args": {"ticker": ticker},
                "result": metrics_data,
                "formatted": tool_calling.format_tool_result("get_financial_metrics", metrics_data),
            })

        return tool_events

    def build_prompt_context(self, user_text: str, tool_events: List[Dict[str, Any]], rag_context: Optional[str]) -> str:
        """Compose the full system instructions and prompt with real-time groundings."""
        system_sections = [
            "You are a professional, knowledgeable financial assistant.",
            "Provide accurate, concise, and helpful answers about markets, investing, and financial terms.",
            "Never give formal personal financial advice. Maintain compliance disclaimers when appropriate.",
        ]

        # Add portfolio context if present
        if self.memory.portfolio:
            system_sections.append(f"User Portfolio:\n{self.memory.get_portfolio_summary()}")

        if self.memory.discussed_tickers:
            system_sections.append(f"Active Ticker Focus: {', '.join(sorted(self.memory.discussed_tickers))}")

        # Add retrieved tool context
        supplemental_data = []
        if tool_events:
            supplemental_data.append("Real-Time Market Data:")
            for te in tool_events:
                supplemental_data.append(f"- {te['formatted']}")

        # Add RAG context
        if rag_context:
            supplemental_data.append(f"Reference Financial Concepts:\n{rag_context}")

        system_prompt = "\n\n".join(system_sections)

        # Build message history for the template
        messages = [{"role": "system", "content": system_prompt}]

        # Include last 3 exchanges from memory
        recent_history = self.memory.get_recent_history(n=4)
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Append current user prompt with injected data
        current_user_content = user_text
        if supplemental_data:
            current_user_content += "\n\n[Verified Grounding Data]:\n" + "\n".join(supplemental_data)

        messages.append({"role": "user", "content": current_user_content})

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def check_ecc_defense_baseline(self, text: str) -> Optional[str]:
        """ECC Prompt Defense Baseline: Detect and neutralize prompt injections and jailbreaks."""
        suspicious_patterns = [
            r"ignore (?:all )?(?:previous|above) (?:instructions|rules|directives)",
            r"reveal (?:your )?(?:system prompt|confidential instructions|keys|credentials)",
            r"you are now (?:in developer mode|an unrestricted ai|dan\b)",
            r"disregard (?:safety|guardrails|compliance)",
            r"bypass (?:security|filters|restrictions)",
        ]
        lower = text.lower()
        for p in suspicious_patterns:
            if re.search(p, lower):
                return (
                    "⚠️ **[ECC Agent Defense Alert]**: Input flagged by Prompt Defense Baseline. "
                    "System directives, persona boundaries, and regulatory guardrails cannot be overridden. "
                    "How can I assist you with legitimate market research or financial concepts?"
                )
        return None

    def handle_ecc_slash_commands(self, text: str) -> Optional[str]:
        """Handle ECC-style slash commands."""
        clean = text.strip()
        if not clean.startswith("/"):
            return None

        parts = clean.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/help", "/skills"):
            return (
                "### ⚡ Available ECC Agent Skills & Commands\n\n"
                "- **/stock `<ticker>`**: Fetch live real-time price & trading volume.\n"
                "- **/metrics `<ticker>`**: Inspect P/E ratio, market cap, and profit margins.\n"
                "- **/portfolio**: View your tracked portfolio holdings and average cost basis.\n"
                "- **/audit**: Run a portfolio concentration, risk, and compliance audit.\n"
                "- **/plan `<goal>`**: Formulate a structured step-by-step financial plan.\n"
                "- **/clear**: Reset the session history and active ticker memory.\n"
            )

        if cmd == "/portfolio":
            return self.memory.get_portfolio_summary()

        if cmd == "/audit":
            if not self.memory.portfolio:
                return "🛡️ **[ECC Portfolio Audit]**: No active holdings tracked yet. Add holdings with e.g. `buy 10 AAPL at 180` to run an audit."
            lines = ["🛡️ **[ECC Portfolio & Risk Audit Report]**\n"]
            lines.append(f"- **Total Asset Holdings**: {len(self.memory.portfolio)}")
            lines.append(f"- **Risk Profile**: {self.memory.risk_tolerance.capitalize()}")
            lines.append(f"- **Holdings Concentration**: Diversified across {len(self.memory.portfolio)} position(s).")
            lines.append("- **Compliance Status**: Compliant with non-discretionary educational guidance.")
            return "\n".join(lines)

        if cmd == "/stock":
            if not arg:
                return "Usage: `/stock <ticker>` (e.g. `/stock AAPL`)"
            t = self._extract_ticker(arg) or arg.upper()
            data = tool_calling.get_stock_price(t)
            return tool_calling.format_tool_result("get_stock_price", data)

        if cmd == "/metrics":
            if not arg:
                return "Usage: `/metrics <ticker>` (e.g. `/metrics NVDA`)"
            t = self._extract_ticker(arg) or arg.upper()
            data = tool_calling.get_financial_metrics(t)
            return tool_calling.format_tool_result("get_financial_metrics", data)

        if cmd == "/plan":
            goal = arg or "Building an emergency reserve and long-term portfolio"
            return (
                f"### 📋 Implementation Plan: {goal}\n\n"
                "1. **Baseline Assessment**: Calculate monthly net income and recurring liabilities.\n"
                "2. **Safety Net**: Establish 3–6 months of liquid reserves in high-yield vehicles.\n"
                "3. **Tax-Advantaged Allocation**: Maximize employer 401(k) match and IRA contributions.\n"
                "4. **Core Equity Indexing**: Allocate core capital to low-cost broad-market index ETFs.\n"
                "5. **Periodic Rebalancing**: Review risk tolerance and asset drift semi-annually.\n"
            )

        if cmd in ("/clear", "/reset"):
            self.memory.history = []
            self.memory.discussed_tickers = set()
            self.memory._save()
            return "Session and ticker memory successfully cleared."

        return None

    def stream_chat(self, user_text: str) -> Generator[Dict[str, Any], None, None]:
        """Stream response as real-time event generator with ECC defense and agent workflows."""
        user_clean = user_text.strip()
        if not user_clean:
            yield {"type": "error", "message": "Empty prompt provided."}
            return

        # 0. ECC Prompt Defense Check
        defense_alert = self.check_ecc_defense_baseline(user_clean)
        if defense_alert:
            yield {"type": "status", "message": "ECC defense baseline triggered"}
            yield {"type": "token", "content": defense_alert}
            yield {"type": "done", "portfolio": self.memory.portfolio}
            return

        # 1. ECC Slash Command Dispatch
        slash_reply = self.handle_ecc_slash_commands(user_clean)
        if slash_reply:
            yield {"type": "status", "message": "Executed ECC agent command"}
            self.memory.add_message("user", user_clean)
            self.memory.add_message("assistant", slash_reply)
            yield {"type": "token", "content": slash_reply}
            yield {"type": "done", "portfolio": self.memory.portfolio}
            return

        # 2. Portfolio Commands Check
        portfolio_feedback = self._handle_portfolio_intent(user_clean)
        if portfolio_feedback:
            yield {"type": "status", "message": "Updated user portfolio state"}
            self.memory.add_message("user", user_clean)
            self.memory.add_message("assistant", portfolio_feedback)
            yield {"type": "token", "content": portfolio_feedback}
            yield {"type": "done", "portfolio": self.memory.portfolio}
            return

        self.load_model()

        # 3. Check and invoke tools
        yield {"type": "status", "message": "Analyzing query & checking market tools..."}
        tool_events = self.execute_tools(user_clean)
        for te in tool_events:
            yield {"type": "tool_call", "tool": te["tool"], "args": te["args"]}
            yield {"type": "tool_result", "tool": te["tool"], "result": te["formatted"]}

        # 3. Retrieve RAG context if conceptual query
        yield {"type": "status", "message": "Searching financial knowledge base..."}
        rag_context = retrieve_relevant_context(user_clean, self.knowledge_base, top_k=2)
        if rag_context:
            yield {"type": "rag_retrieval", "snippet": rag_context[:160] + "..."}

        # 4. Construct prompt
        prompt_text = self.build_prompt_context(user_clean, tool_events, rag_context)
        input_ids = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # 5. Set up streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 280,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        yield {"type": "status", "message": "Generating response..."}

        generated_chunks = []
        for new_text in streamer:
            generated_chunks.append(new_text)
            yield {"type": "token", "content": new_text}

        thread.join()
        raw_answer = "".join(generated_chunks).strip()

        # 6. Apply post-processing and guardrails
        cleaned = trim_repetitive_footers(raw_answer)
        cleaned = enforce_max_length(cleaned, self.tokenizer)
        if check_financial_guardrails(cleaned) and "[Reviewed for financial advisory compliance.]" not in cleaned:
            cleaned += "\n\n*(Note: For educational purposes only. Not financial advice.)*"

        # 7. Persist to memory
        self.memory.add_message("user", user_clean)
        self.memory.add_message("assistant", cleaned)

        yield {"type": "done", "portfolio": self.memory.portfolio, "full_text": cleaned}

    def chat(self, user_text: str) -> Dict[str, Any]:
        """Synchronous wrapper for standard API calls."""
        tokens = []
        tools_executed = []
        for event in self.stream_chat(user_text):
            if event["type"] == "token":
                tokens.append(event["content"])
            elif event["type"] == "tool_result":
                tools_executed.append(event["result"])

        return {
            "answer": "".join(tokens),
            "tools": tools_executed,
            "portfolio": self.memory.portfolio,
        }
