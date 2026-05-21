from __future__ import annotations

import os
import shutil
from urllib.parse import urlparse


class SemanticBackendError(RuntimeError):
    pass


def _ollama_without_key_allowed(backends: dict) -> bool:
    ollama_url = os.environ.get(
        "OLLAMA_BASE_URL",
        backends.get("ollama", {}).get("base_url", ""),
    )
    try:
        host = (urlparse(ollama_url).hostname or "").lower()
    except Exception:
        host = ""
    return host in ("localhost", "127.0.0.1", "::1") or host.startswith("127.")


def _bedrock_without_key_allowed() -> bool:
    return bool(
        os.environ.get("AWS_PROFILE")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("AWS_ACCESS_KEY_ID")
    )


def resolve_semantic_backend(backend: str | None = None) -> str:
    from graphify.llm import (
        BACKENDS,
        detect_backend,
        _format_backend_env_keys,
        _get_backend_api_key,
    )

    chosen = backend or detect_backend()
    if chosen is None:
        raise SemanticBackendError(
            "no LLM API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY "
            "(gemini), MOONSHOT_API_KEY (kimi), ANTHROPIC_API_KEY (claude), "
            "OPENAI_API_KEY (openai), DEEPSEEK_API_KEY (deepseek), "
            "or pass --backend."
        )
    if chosen not in BACKENDS:
        raise SemanticBackendError(
            f"unknown backend '{chosen}'. Available: {', '.join(sorted(BACKENDS))}"
        )
    if _get_backend_api_key(chosen):
        return chosen
    if chosen == "ollama" and _ollama_without_key_allowed(BACKENDS):
        return chosen
    if chosen == "bedrock" and _bedrock_without_key_allowed():
        return chosen
    if chosen == "claude-cli":
        if shutil.which("claude") is not None:
            return chosen
        raise SemanticBackendError(
            "backend 'claude-cli' requires the `claude` CLI on $PATH "
            "(install Claude Code and run `claude` once to authenticate)."
        )
    raise SemanticBackendError(
        f"backend '{chosen}' requires {_format_backend_env_keys(chosen)} to be set."
    )
