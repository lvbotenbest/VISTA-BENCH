#!/usr/bin/env python3
"""Minimal client for user-supplied OpenAI-compatible endpoints."""
from __future__ import annotations

from dataclasses import dataclass
import base64
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit

import requests


class APIError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None,
                 retryable: bool = True):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass(frozen=True)
class APIConfig:
    base_url: str
    endpoint: str
    model: str
    key_env: str
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("base_url must be an explicit HTTP(S) URL")
        if self.endpoint not in {"chat/completions", "responses"}:
            raise ValueError("endpoint must be chat/completions or responses")
        if not self.model.strip() or not self.key_env.strip():
            raise ValueError("model and key_env are required")

    @property
    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.endpoint}"

    def api_key(self) -> str:
        value = os.environ.get(self.key_env)
        if not value:
            raise APIError(f"environment variable {self.key_env!r} is not set", retryable=False)
        return value


def image_data_url(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xd8"):
        media_type = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "image/png"
    elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        media_type = "image/webp"
    else:
        raise APIError("unsupported image content")
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def _redact_error(text: str, config: APIConfig, secret: str) -> str:
    values = {secret, config.base_url, config.endpoint, config.url}
    parsed = urlsplit(config.base_url)
    if parsed.username:
        values.add(parsed.username)
    if parsed.password:
        values.add(parsed.password)
    for value in sorted((value for value in values if value), key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return re.sub(r"https?://[^\s'\"<>]+", "[REDACTED_URL]", text)


class OpenAICompatibleClient:
    def __init__(self, config: APIConfig, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()

    def generate(self, prompt: str, *, image: Path | None = None,
                 temperature: float = 0.0, max_output_tokens: int = 4096) -> str:
        key = self.config.api_key()
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if self.config.endpoint == "chat/completions":
            content: Any = prompt
            if image is not None:
                content = [{"type": "text", "text": prompt},
                           {"type": "image_url", "image_url": {"url": image_data_url(image)}}]
            payload = {"model": self.config.model, "messages": [{"role": "user", "content": content}],
                       "temperature": temperature, "max_tokens": max_output_tokens}
        else:
            content = [{"type": "input_text", "text": prompt}]
            if image is not None:
                content.append({"type": "input_image", "image_url": image_data_url(image)})
            payload = {"model": self.config.model,
                       "input": [{"role": "user", "content": content}],
                       "temperature": temperature, "max_output_tokens": max_output_tokens}
        try:
            response = self.session.post(self.config.url, headers=headers, json=payload,
                                         timeout=self.config.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            if self.config.endpoint == "chat/completions":
                text = data["choices"][0]["message"]["content"]
            else:
                text = data.get("output_text")
                if text is None:
                    text = "".join(item.get("text", "") for output in data.get("output", [])
                                   for item in output.get("content", []))
            if not isinstance(text, str) or not text.strip():
                raise APIError("endpoint returned no text")
            return text
        except APIError:
            raise
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            message = f"HTTP status {status}" if status is not None else "HTTP request failed"
            retryable = status is None or status in {408, 409, 425, 429} or status >= 500
            raise APIError(message, status_code=status, retryable=retryable) from None
        except Exception as exc:
            message = _redact_error(str(exc), self.config, key)
            raise APIError(message) from None
