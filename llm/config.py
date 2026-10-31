"""Settings from configs/default.yaml plus environment variables (.env is loaded)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from llm.client import LLM, OpenAICompatibleLLM, ResponseCache, UsageMeter

ROOT = Path(__file__).resolve().parent.parent
ROLES = ("planner", "researcher", "critic", "synthesizer")


@dataclass
class Settings:
    raw: dict[str, Any]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> str:
        p = Path(self.get(dotted))
        return str(p if p.is_absolute() else ROOT / p)


def load_settings(path: str | None = None) -> Settings:
    load_dotenv(ROOT / ".env")
    with open(path or ROOT / "configs" / "default.yaml") as f:
        return Settings(yaml.safe_load(f))


def build_llms(settings: Settings, meter: UsageMeter | None = None) -> dict[str, LLM]:
    """One client per role, sharing a cache and a usage meter."""
    meter = meter or UsageMeter()
    provider = settings.get("llm.provider")
    p = settings.get(f"llm.providers.{provider}")
    api_key = os.environ.get(p["api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"{p['api_key_env']} is not set (put it in .env)")
    cache = ResponseCache(settings.path("llm.cache_path"))
    clients: dict[str, LLM] = {}
    shared: dict[str, OpenAICompatibleLLM] = {}
    for role in ROLES:
        model = settings.get(f"llm.models.{role}")
        # Roles on the same model share one client, so its pacing covers all of them.
        if model not in shared:
            shared[model] = OpenAICompatibleLLM(model, base_url=p["base_url"], api_key=api_key, meter=meter,
                                                cache=cache, requests_per_minute=p.get("requests_per_minute", 0),
                                                params=(settings.get("llm.model_params") or {}).get(model))
        clients[role] = shared[model]
    return clients
