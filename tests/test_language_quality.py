import pytest

from yt_sub_hub.language_quality import validate_ru_text_field
from yt_sub_hub.translation import LMStudioTranslationProvider, TranslationError


@pytest.mark.parametrize("label", ["MCP", "API", "AI агенты", "Клиент MCP", "LLM"])
def test_russian_labels_preserve_known_technical_abbreviations(label):
    assert validate_ru_text_field("tag", label, 0.5, True, short_label=True) == []


@pytest.mark.parametrize("label", ["AI Agents", "Introduction", "Model MCP", "English text"])
def test_english_labels_still_require_translation(label):
    assert validate_ru_text_field("tag", label, 0.5, True, short_label=True)


def test_translation_keeps_technical_abbreviation_without_invented_russian_padding():
    item = {"path": "tags.0", "text": "MCP"}
    assert LMStudioTranslationProvider._parse_analysis_item_batch({"items": [item]}, [item], "ru") == {"tags.0": "MCP"}
    english = {"path": "tags.0", "text": "English text"}
    with pytest.raises(TranslationError):
        LMStudioTranslationProvider._parse_analysis_item_batch({"items": [english]}, [english], "ru")


@pytest.mark.parametrize("path,source", [
    ("abstract", "The protocol connects applications and data."),
    ("claims.0.text", "MCP"),
    ("tags.0", "Data integration"),
])
def test_abbreviation_exception_cannot_replace_prose_or_change_source(path, source):
    with pytest.raises(TranslationError):
        LMStudioTranslationProvider._parse_analysis_item_batch(
            {"items": [{"path": path, "text": "MCP"}]},
            [{"path": path, "text": source}], "ru",
        )


def test_single_item_shape_error_retries_as_text_without_model_owned_path(monkeypatch):
    provider = LMStudioTranslationProvider("http://localhost:1234/v1", "test-key", "gemma")
    responses = iter([{"items": []}, {"translated_text": "ИИ-агенты"}])
    monkeypatch.setattr(provider, "_translate_payload", lambda *args: next(responses))
    assert provider._translate_analysis_item_batch(
        [{"path": "tags.0", "text": "AI Agents"}], "en", "ru",
    ) == {"tags.0": "ИИ-агенты"}


def test_failed_single_item_repair_does_not_pad_untranslated_prose(monkeypatch):
    provider = LMStudioTranslationProvider("http://localhost:1234/v1", "test-key", "gemma")
    responses = iter([{"items": []}, {"translated_text": "English text"}])
    monkeypatch.setattr(provider, "_translate_payload", lambda *args: next(responses))
    with pytest.raises(TranslationError):
        provider._translate_analysis_item_batch(
            [{"path": "abstract", "text": "English text"}], "en", "ru",
        )


@pytest.mark.parametrize('label', ['ChatGPT', 'OpenAI', 'Claude', 'Python', 'Perplexity', 'NotebookLM'])
def test_product_names_remain_unchanged_in_russian_labels(label):
    item = {'path': 'tags.0', 'text': label}
    assert validate_ru_text_field('tag', label, 0.5, True, short_label=True) == []
    assert LMStudioTranslationProvider._parse_analysis_item_batch({'items': [item]}, [item], 'ru') == {'tags.0': label}
