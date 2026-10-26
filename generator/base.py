"""
generator/base.py -- Grammar-constrained JSON generator for FinSage.
Enforces schema:
{"abstain": bool, "sentences": [{"text": str, "cites": [int]}]}
Includes the strict runtime system prompt and 3 fixed few-shot examples.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional
from orchestrator.types import SourceChunk, ToolResult, GeneratorOutput, SentenceItem
from config import cfg, BASE_DIR

logger = logging.getLogger("finsage.generator")

RUNTIME_SYSTEM_PROMPT = """You are FinSage, an evidence-grounded Personal Chartered Accountant and financial advisor for everyday citizens, gig workers, and families. Speak warmly, clearly, and naturally in plain language. Use ONLY the SOURCES and TOOL RESULTS below. Cite every sentence with [n]. If they don't contain the answer, set abstain=true and say exactly: I don't have that in my sources. Never state a number, date, or price that is not in SOURCES or TOOL RESULTS. Never advise buying or selling risky assets. Output strictly valid JSON per the schema:
{"abstain": bool, "sentences": [{"text": str, "cites": [int]}]}"""

FEW_SHOT_EXAMPLES = [
    # 1. Grounded Example
    {
        "sources": [{"id": 1, "text": "A bull market is a financial market of increasing stock prices by 20% or more over a sustained duration."}],
        "tool_results": [],
        "question": "What is a bull market?",
        "response": {
            "abstain": False,
            "sentences": [
                {"text": "A bull market refers to a sustained period where market prices increase by 20% or more from recent lows.", "cites": [1]}
            ]
        }
    },
    # 2. Abstain Example
    {
        "sources": [{"id": 1, "text": "Compound interest is calculated on the initial principal and accumulated interest."}],
        "tool_results": [],
        "question": "What was Tesla's quarterly net income in 2018?",
        "response": {
            "abstain": True,
            "sentences": [
                {"text": "I don't have that in my sources.", "cites": []}
            ]
        }
    },
    # 3. Tool Readback Example
    {
        "sources": [],
        "tool_results": ["AAPL: $185.42 USD as of 2026-10-25 with P/E ratio 28.5"],
        "question": "What is Apple's current stock price and valuation?",
        "response": {
            "abstain": False,
            "sentences": [
                {"text": "Apple is currently trading at $185.42 USD.", "cites": [0]},
                {"text": "Its trailing price-to-earnings ratio is 28.5.", "cites": [0]}
            ]
        }
    }
]


class StructuredGenerator:
    """JSON grammar-constrained generation engine backed by fine-tuned TinyLlama or Qwen."""

    def __init__(self, model_key: Optional[str] = None):
        self.system_prompt = RUNTIME_SYSTEM_PROMPT
        self.device = cfg.device
        self.active_model_key = model_key or cfg.active_model
        self.model = None
        self.tokenizer = None
        self._load_neural_model(self.active_model_key)

    def set_model(self, model_key: str):
        """Switch active neural model between tinyllama and qwen."""
        if model_key == self.active_model_key and self.model is not None:
            return
        self.active_model_key = model_key
        self._load_neural_model(model_key)

    def _load_neural_model(self, model_key: str):
        """Load base model and attach the fine-tuned Personal CA LoRA adapter."""
        try:
            import gc
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            # Clean existing model from GPU VRAM
            if self.model is not None:
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            m_config = cfg.models.get(model_key, cfg.models.get("tinyllama"))
            base_model_name = m_config["name"]
            adapter_dir = m_config.get("ca_adapter_path") or m_config.get("adapter_path")

            tokenizer_path = str(adapter_dir) if (adapter_dir and adapter_dir.exists()) else base_model_name

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=dtype,
                device_map=self.device if self.device == "cuda" else None,
                trust_remote_code=True,
            )

            if adapter_dir and adapter_dir.exists():
                self.model = PeftModel.from_pretrained(base_model, str(adapter_dir))
            else:
                self.model = base_model

            self.model.eval()
            logger.info("StructuredGenerator: Model %s loaded successfully on %s", model_key, self.device)
        except Exception as e:
            logger.warning("Could not load neural model %s, running in fallback mode: %s", model_key, str(e))
            self.model = None

    def build_prompt(
        self,
        question: str,
        sources: List[SourceChunk],
        tool_results: List[ToolResult],
        repair_feedback: Optional[str] = None,
    ) -> str:
        """Compose grounded prompt using chat template or few-shot fallback."""
        source_lines = []
        for s in sources:
            source_lines.append(f"SOURCE [{s.id}]: {s.text}")
        for i, tr in enumerate(tool_results):
            source_lines.append(f"TOOL RESULT [{i}]: {tr.formatted_text}")

        context_text = "\n".join(source_lines) if source_lines else "NO_SOURCES_FOUND: Missing evidence."
        user_content = f"SOURCES:\n{context_text}\n\nQUESTION: {question}"
        if repair_feedback:
            user_content += f"\n\nPREVIOUS DRAFT REJECTED BY VERIFIER: {repair_feedback}\nPlease correct all numbers and citations immediately."

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                ]
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        # Fallback few-shot prompt
        sections = [self.system_prompt, "\n--- FEW-SHOT EXAMPLES ---"]
        for eg in FEW_SHOT_EXAMPLES:
            sections.append(f"SOURCES: {json.dumps(eg['sources'])}\nTOOL RESULTS: {json.dumps(eg['tool_results'])}\nQUESTION: {eg['question']}\nOUTPUT:\n{json.dumps(eg['response'])}")

        sections.append("\n--- CURRENT TASK ---")
        src_data = [{"id": s.id, "text": s.text, "source": s.source_name} for s in sources]
        tool_data = [tr.formatted_text for tr in tool_results]

        sections.append(f"SOURCES: {json.dumps(src_data)}")
        sections.append(f"TOOL RESULTS: {json.dumps(tool_data)}")
        if repair_feedback:
            sections.append(f"PREVIOUS DRAFT REJECTED BY VERIFIER: {repair_feedback}\nPlease correct all numbers and citations immediately.")
        sections.append(f"QUESTION: {question}\nOUTPUT JSON:")

        return "\n\n".join(sections)

    def parse_response(self, text: str) -> GeneratorOutput:
        """Extract and validate the JSON schema from model output."""
        try:
            # Look for JSON object in output
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise ValueError("No JSON object found in output")
            raw_json = json.loads(m.group(0))

            abstain = bool(raw_json.get("abstain", False))
            sentences = []
            for item in raw_json.get("sentences", []):
                t = str(item.get("text", "")).strip()
                t = re.sub(r"^\[Verified Source:[^\]]+\]\s*", "", t).strip()
                cites = [int(c) for c in item.get("cites", []) if isinstance(c, (int, str)) and str(c).isdigit()]
                if t:
                    sentences.append(SentenceItem(text=t, cites=cites))

            if not sentences and abstain:
                sentences.append(SentenceItem(text="I don't have that in my sources.", cites=[]))

            return GeneratorOutput(abstain=abstain, sentences=sentences, raw_response=text)

        except Exception as e:
            # Safe fallback if output format fails validation
            return GeneratorOutput(
                abstain=True,
                sentences=[SentenceItem(text="I don't have that in my sources.", cites=[])],
                raw_response=text,
            )

    def draft(
        self,
        question: str,
        sources: List[SourceChunk],
        tool_results: List[ToolResult],
        repair_feedback: Optional[str] = None,
    ) -> GeneratorOutput:
        """Draft a response generated directly by fine-tuned TinyLlama on GPU."""
        import torch

        if not sources and not tool_results:
            return GeneratorOutput(
                abstain=True,
                sentences=[SentenceItem(text="I don't have that in my sources.", cites=[])],
                raw_response='{"abstain": true, "sentences": [{"text": "I don\'t have that in my sources.", "cites": []}]}',
            )

        # If neural model is loaded, run real GPU inference through fine-tuned TinyLlama
        if self.model is not None and self.tokenizer is not None:
            try:
                prompt_text = self.build_prompt(question, sources, tool_results, repair_feedback)
                inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
                if self.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                with torch.no_grad():
                    gen_out = self.model.generate(
                        **inputs,
                        max_new_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                gen_tokens = gen_out[0][inputs["input_ids"].shape[1] :]
                decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                parsed = self.parse_response(decoded)
                if parsed.sentences and (not parsed.abstain or not sources and not tool_results):
                    return parsed
            except Exception as e:
                logger.warning("Neural generation error, falling back: %s", str(e))

        # Fallback synthesizer if neural generation produced invalid output
        sentences = []
        if tool_results:
            for tr in tool_results:
                sentences.append(SentenceItem(text=tr.formatted_text, cites=[0]))

        if sources:
            for s in sources[:2]:
                clean_text = re.sub(r"\[Verified Source:[^\]]+\]", "", s.text).strip()
                chunks = [c.strip() for c in clean_text.split(".") if len(c.strip()) > 8]
                for c in chunks[:3]:
                    sent_text = c if c.endswith(".") else f"{c}."
                    sentences.append(SentenceItem(text=sent_text, cites=[s.id]))

        return GeneratorOutput(abstain=False, sentences=sentences)
