"""
config.py -- Central configuration for FinSage.
Defines hardware settings, paths, limits, model configurations, and decision backends.
"""

import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field

# Prevent Windows Python 3.13 pyarrow Cython access-violation crash during dynamic imports
sys.modules["pyarrow"] = None

BASE_DIR = Path(__file__).resolve().parent

class FinSageConfig(BaseModel):
    # App & Server
    app_name: str = "FinSage"
    app_version: str = "0.1.0"
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False

    # Hardware & Model Architecture (3-Tier Multi-Agent: TinyLlama + Qwen + DeepSeek-R1)
    device: str = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"
    active_model: str = os.environ.get("FINSAGE_MODEL", "tinyllama")  # "tinyllama", "qwen", or "deepseek_r1"
    
    # Model Registry
    models: dict = {
        "tinyllama": {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "adapter_path": BASE_DIR / "finance_adapter" / "final",
            "ca_adapter_path": BASE_DIR / "adapters" / "tinyllama_ca",
            "template": "llama2",
            "load_in_4bit": False,  # TinyLlama fits easily in fp16 (~2GB)
        },
        "qwen": {
            "name": "Qwen/Qwen2.5-1.5B-Instruct",
            "adapter_path": BASE_DIR / "adapters" / "qwen_ca",
            "ca_adapter_path": BASE_DIR / "adapters" / "qwen_ca",
            "template": "chatml",
            "load_in_4bit": True,   # Qwen loaded in 4-bit via bitsandbytes (~1.2GB)
        },
        "deepseek_r1": {
            "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "adapter_path": BASE_DIR / "adapters" / "deepseek_ca",
            "ca_adapter_path": BASE_DIR / "adapters" / "deepseek_ca",
            "template": "deepseek_r1",
            "load_in_4bit": True,   # DeepSeek-R1 1.5B loaded in 4-bit (~1.3GB)
        },
    }

    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path: Path = BASE_DIR / "finance_adapter" / "final"
    fallback_adapter_path: Path = BASE_DIR / "finetune" / "adapter"
    vram_budget_gb: float = 6.0
    max_tokens: int = 400
    temperature: float = 0.1  # Low temperature for factual compliance

    # Storage & DB
    db_path: Path = BASE_DIR / "data" / "finsage.db"
    knowledge_base_path: Path = BASE_DIR / "data" / "finance_instructions.jsonl"
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    max_upload_size_bytes: int = 20 * 1024 * 1024  # 20 MB
    max_pdf_pages: int = 50

    # Decision Layer
    decision_backend: str = os.environ.get("FINSAGE_DECISION_BACKEND", "baseline")  # "baseline", "jev", "local"
    decision_threshold: float = 0.65

    # Verification & Repair
    max_repairs: int = 2

    # Tools
    market_cache_ttl_seconds: int = 300  # 5 min TTL cache for live quotes

cfg = FinSageConfig()
os.makedirs(cfg.db_path.parent, exist_ok=True)
os.makedirs(cfg.upload_dir, exist_ok=True)
