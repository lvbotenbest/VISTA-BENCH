import os
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).parents[1]))
from api import APIConfig, APIError, OpenAICompatibleClient, image_data_url


class Response:
    def raise_for_status(self): pass
    def json(self): return {"choices": [{"message": {"content": "ok"}}]}


class Session:
    def __init__(self): self.request = None
    def post(self, url, **kwargs): self.request = (url, kwargs); return Response()


def test_requires_explicit_url_and_endpoint():
    try:
        APIConfig("example", "chat/completions", "m", "VISTA_API_KEY")
        assert False
    except ValueError:
        pass


def test_chat_request_uses_named_environment(monkeypatch):
    monkeypatch.setenv("VISTA_API_KEY", "test-secret")
    session = Session()
    client = OpenAICompatibleClient(APIConfig("https://example.test/v1", "chat/completions", "model", "VISTA_API_KEY"), session)
    assert client.generate("hello") == "ok"
    assert session.request[0] == "https://example.test/v1/chat/completions"
    assert session.request[1]["headers"]["Authorization"] == "Bearer test-secret"


class ResponsesResponse:
    def raise_for_status(self): pass
    def json(self): return {"output_text": "ok"}


class ResponsesSession(Session):
    def post(self, url, **kwargs):
        self.request = (url, kwargs)
        return ResponsesResponse()


def test_responses_request_and_output_text(monkeypatch):
    monkeypatch.setenv("VISTA_API_KEY", "test-secret")
    session = ResponsesSession()
    client = OpenAICompatibleClient(APIConfig(
        "https://example.test/v1", "responses", "model", "VISTA_API_KEY"), session)
    assert client.generate("hello") == "ok"
    assert session.request[0] == "https://example.test/v1/responses"
    assert session.request[1]["json"]["input"][0]["content"][0] == {
        "type": "input_text", "text": "hello"
    }


def test_image_mime_uses_magic_bytes_not_extension(tmp_path):
    samples = [
        ("wrong.png", b"\xff\xd8payload", "data:image/jpeg;base64,"),
        ("wrong.jpg", b"\x89PNG\r\n\x1a\npayload", "data:image/png;base64,"),
        ("wrong.jpeg", b"RIFF\x04\x00\x00\x00WEBP", "data:image/webp;base64,"),
    ]
    for name, data, prefix in samples:
        path = tmp_path / name
        path.write_bytes(data)
        assert image_data_url(path).startswith(prefix)


class FailingSession:
    def post(self, url, **kwargs):
        key = kwargs["headers"]["Authorization"].split()[-1]
        raise RuntimeError(f"request to {url} with key {key} failed")


def test_errors_redact_key_endpoint_url_userinfo_and_suppress_cause(monkeypatch):
    key = "private-test-key"
    monkeypatch.setenv("VISTA_API_KEY", key)
    config = APIConfig("https://private-user:private-pass@example.test/v1",
                       "chat/completions", "model", "VISTA_API_KEY")
    try:
        OpenAICompatibleClient(config, FailingSession()).generate("hello")
        assert False
    except APIError as exc:
        rendered = "".join(__import__("traceback").format_exception(exc))
        for private in (key, config.base_url, config.endpoint, config.url,
                        "private-user", "private-pass"):
            assert private not in str(exc)
            assert private not in rendered
        assert exc.__cause__ is None


class UnauthorizedResponse:
    status_code = 401

    def raise_for_status(self):
        raise requests.HTTPError("private URL", response=self)


class UnauthorizedSession:
    def post(self, url, **kwargs):
        return UnauthorizedResponse()


def test_http_status_is_carried_and_auth_is_not_retryable(monkeypatch):
    monkeypatch.setenv("VISTA_API_KEY", "test-secret")
    config = APIConfig("https://example.test/v1", "chat/completions", "model", "VISTA_API_KEY")
    with __import__("pytest").raises(APIError) as raised:
        OpenAICompatibleClient(config, UnauthorizedSession()).generate("hello")
    assert raised.value.status_code == 401
    assert raised.value.retryable is False
    assert "example.test" not in str(raised.value)
