import httpx
import pytest
from openai import OpenAI

from yt_sub_hub.translation import LMStudioTranslationProvider, TranslationError


@pytest.mark.parametrize("status", [200, 500])
def test_translation_closes_http_client_after_request(monkeypatch, status):
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={
        "choices": [{"message": {"content": '{"translated_text":"Привет"}'}}],
    }))
    http = httpx.Client(transport=transport)
    client = OpenAI(api_key="synthetic", base_url="http://localhost/v1", http_client=http, max_retries=0)
    provider = LMStudioTranslationProvider("http://localhost/v1", "synthetic", "model", retries=0)
    monkeypatch.setattr(provider, "_client", lambda: client)
    if status == 200:
        assert provider.translate_text("Hello", source_language="en") == "Привет"
    else:
        with pytest.raises(TranslationError):
            provider.translate_text("Hello", source_language="en")
    assert http.is_closed
