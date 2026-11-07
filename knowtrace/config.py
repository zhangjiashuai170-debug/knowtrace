"""Explicit startup-only model configuration. Never loads a credential store."""
import ipaddress
import math
import os
from urllib.parse import urlsplit

from .generation import ModelConfig


def model_config_from_env(environ=None) -> ModelConfig:
    env = os.environ if environ is None else environ
    provider = env.get("KNOWTRACE_MODEL_PROVIDER", "disabled").strip()
    if provider == "disabled":
        # Do not even read unrelated model values when the feature is disabled.
        return ModelConfig()
    if provider not in {"openai_compatible", "ollama"}:
        raise ValueError("KNOWTRACE_MODEL_PROVIDER 配置无效")
    base = env.get("KNOWTRACE_MODEL_BASE_URL", "").strip().rstrip("/")
    model = env.get("KNOWTRACE_MODEL_NAME", "").strip()
    try:
        parsed = urlsplit(base)
        port = parsed.port
        host = parsed.hostname or ""
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host.lower() == "localhost"
        if (not host or parsed.scheme not in {"http", "https"} or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or (parsed.scheme == "http" and not local) or any(c.isspace() for c in base)):
            raise ValueError()
        if not model or len(model) > 200 or any(ord(c) < 32 for c in model):
            raise ValueError()
        timeout = float(env.get("KNOWTRACE_MODEL_TIMEOUT_SECONDS", "45"))
        if not math.isfinite(timeout) or not 1 <= timeout <= 120:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("模型地址、名称或超时配置无效；HTTP仅限本机，远程地址须使用HTTPS") from None
    key = env.get("KNOWTRACE_MODEL_API_KEY", "") if provider == "openai_compatible" else ""
    if any(ord(c) < 32 for c in key) or len(key) > 4096:
        raise ValueError("模型密钥格式无效")
    return ModelConfig(provider=provider, base_url=base, model=model, api_key=key, timeout_seconds=timeout)
