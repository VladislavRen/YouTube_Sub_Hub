import json

from yt_sub_hub.llm import parse_analysis_response
from yt_sub_hub.models import Transcript, TranscriptSegment, VideoMetadata
from yt_sub_hub.llm import (
    LMStudioAnalyzer,
    build_analysis_json_schema_response_format,
    build_chunk_analysis_response_format,
    build_deep_graph_chunk_response_format,
    build_extractive_chunk_analysis,
    extract_json_payload,
    merge_chunk_analyses,
    parse_deep_graph_chunk_response,
)


def test_parse_analysis_response_validates_required_structured_fields():
    analysis = parse_analysis_response(
        {
            "language": "en",
            "cleaned_title": "A Lecture",
            "abstract": "Summary.",
            "tags": ["ai"],
            "outline": ["Intro"],
            "sections": [
                {
                    "heading": "Intro",
                    "summary": "Opening.",
                    "timestamp_seconds": 0,
                    "bullets": ["Context"],
                }
            ],
            "concepts": [{"name": "Model", "definition": "A representation."}],
            "claims": [{"text": "Models simplify.", "evidence": "Example from talk."}],
            "limitations": ["No benchmark."],
            "open_questions": ["What changes at scale?"],
            "related_topics": ["Evaluation"],
        }
    )

    assert analysis.cleaned_title == "A Lecture"
    assert analysis.concepts[0].slug == "model"


def test_lm_studio_prompt_uses_chunked_transcript_for_long_inputs():
    analyzer = LMStudioAnalyzer("http://localhost:1234/v1", "key", "qwen", "gemma", max_prompt_chars=80)
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(
        video_id="abc123",
        source="youtube",
        segments=[
            TranscriptSegment(start=0, duration=1, text="a" * 70),
            TranscriptSegment(start=1, duration=1, text="b" * 70),
        ],
    )

    prompt = analyzer._prompt(metadata, transcript)

    assert "Transcript chunks:" in prompt
    assert "Chunk 1:" in prompt
    assert "Chunk 2:" in prompt


def test_short_analysis_prompt_preserves_segment_timestamps_and_language():
    analyzer = LMStudioAnalyzer("http://localhost:1234/v1", "key", "qwen", "gemma")
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(
        video_id="abc123", source="youtube", language="ru",
        segments=[TranscriptSegment(12, 3, "Начало"), TranscriptSegment(187, 4, "Пример")],
    )
    prompt = analyzer._prompt(metadata, transcript)
    assert "[12s] Начало" in prompt
    assert "[187s] Пример" in prompt
    assert "Transcript language: ru" in prompt


def test_lm_studio_prompt_keeps_user_prompt_under_budget():
    analyzer = LMStudioAnalyzer("http://localhost:1234/v1", "key", "qwen", "gemma", max_prompt_chars=500)
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(
        video_id="abc123",
        source="youtube",
        segments=[TranscriptSegment(start=index, duration=1, text="x" * 120) for index in range(20)],
    )

    prompt = analyzer._prompt(metadata, transcript)

    assert len(prompt) <= 700


def test_lm_studio_long_transcript_analysis_sends_chunk_prompts_not_one_huge_prompt():
    calls = []

    class ChunkingAnalyzer(LMStudioAnalyzer):
        def _create_completion(self, client, **kwargs):
            calls.append(kwargs)
            content = {
                "heading": f"Chunk {len(calls)}",
                "summary": "Local chunk summary.",
                "timestamp_seconds": 0,
                "bullets": ["Local detail"],
                "concepts": [{"name": "Chunked analysis", "definition": "Analyze bounded transcript windows."}],
                "claims": [{"text": "The analyzer uses bounded prompts.", "evidence": "Captured prompt length."}],
                "limitations": [],
                "open_questions": [],
                "related_topics": ["analysis"],
            }
            message = type("Message", (), {"content": json.dumps(content)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    analyzer = ChunkingAnalyzer("http://localhost:1234/v1", "key", "qwen", "gemma", max_prompt_chars=1700)
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(
        video_id="abc123",
        source="youtube",
        segments=[TranscriptSegment(start=index, duration=1, text=f"segment {index} " + ("x" * 180)) for index in range(25)],
    )

    analysis = analyzer._analyze_long_transcript(object(), metadata, transcript)

    prompt_lengths = [len(call["messages"][1]["content"]) for call in calls]
    assert len(calls) > 1
    assert max(prompt_lengths) < len(transcript.full_text())
    assert all("Chunk " in call["messages"][1]["content"] for call in calls)
    assert analysis.sections


def test_lm_studio_uses_json_schema_response_format():
    response_format = build_analysis_json_schema_response_format()

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"]["required"] == [
        "language",
        "cleaned_title",
        "abstract",
        "tags",
        "outline",
        "sections",
        "concepts",
        "claims",
        "limitations",
        "open_questions",
        "related_topics",
    ]


def test_extract_json_payload_accepts_markdown_fenced_json():
    payload = extract_json_payload('```json\n{"ok": true}\n```')

    assert payload == '{"ok": true}'


def test_chunk_analysis_schema_has_required_section_fields():
    response_format = build_chunk_analysis_response_format()

    schema = response_format["json_schema"]["schema"]
    assert response_format["type"] == "json_schema"
    assert schema["required"] == [
        "heading",
        "summary",
        "timestamp_seconds",
        "bullets",
        "concepts",
        "claims",
        "limitations",
        "open_questions",
        "related_topics",
    ]


def test_deep_graph_chunk_schema_requires_graph_fields():
    response_format = build_deep_graph_chunk_response_format()

    schema = response_format["json_schema"]["schema"]
    assert response_format["type"] == "json_schema"
    assert "concept_relations" in schema["required"]
    assert "claim_relations" in schema["required"]
    concept_schema = schema["properties"]["concepts"]["items"]
    assert concept_schema["additionalProperties"] is False
    assert "confidence" in concept_schema["required"]


def test_deep_graph_chunk_schema_is_strict_for_all_nested_items():
    schema = build_deep_graph_chunk_response_format()["json_schema"]["schema"]

    expected_required = {
        "methods": {"local_id", "name", "description", "uses_tool_ids", "applies_to_concept_ids", "confidence", "evidence_quote", "timestamp_start", "timestamp_end"},
        "tools": {"local_id", "name", "description", "confidence", "evidence_quote", "timestamp_start", "timestamp_end"},
        "metrics": {"local_id", "name", "definition", "measured_concept_ids", "confidence", "evidence_quote", "timestamp_start", "timestamp_end"},
        "risks": {"local_id", "text", "related_concept_ids", "confidence", "evidence_quote", "timestamp_start", "timestamp_end"},
        "questions": {"local_id", "text", "about_concept_ids", "confidence", "evidence_quote", "timestamp_start", "timestamp_end"},
    }
    for field, required in expected_required.items():
        item_schema = schema["properties"][field]["items"]
        assert item_schema["additionalProperties"] is False
        assert set(item_schema["required"]) == required

    claim_relation_schema = schema["properties"]["claim_relations"]["items"]
    assert {"timestamp_start", "timestamp_end"} <= set(claim_relation_schema["required"])
    assert claim_relation_schema["properties"]["type"]["enum"] == ["SUPPORTS", "CONTRADICTS"]
    assert schema["properties"]["concept_relations"]["items"]["properties"]["type"]["enum"] == [
        "RELATED_TO",
        "PART_OF",
        "CAUSES",
        "CONTRASTS_WITH",
        "MEASURED_BY",
    ]


def test_merge_chunk_analyses_preserves_one_section_per_chunk():
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    chunks = [
        {
            "heading": "Part 1",
            "summary": "Summary 1",
            "timestamp_seconds": 0,
            "bullets": ["A", "B"],
            "concepts": [{"name": "Concept A", "definition": "Definition A"}],
            "claims": [{"text": "Claim A", "evidence": "Evidence A"}],
            "limitations": ["Limitation A"],
            "open_questions": ["Question A"],
            "related_topics": ["Topic A"],
        },
        {
            "heading": "Part 2",
            "summary": "Summary 2",
            "timestamp_seconds": 120,
            "bullets": ["C", "D"],
            "concepts": [{"name": "Concept B", "definition": "Definition B"}],
            "claims": [{"text": "Claim B", "evidence": "Evidence B"}],
            "limitations": ["Limitation B"],
            "open_questions": ["Question B"],
            "related_topics": ["Topic B"],
        },
    ]

    analysis = merge_chunk_analyses(metadata, chunks, model="gemma")

    assert len(analysis.sections) == 2
    assert analysis.sections[0].heading == "Part 1"
    assert analysis.sections[1].heading == "Part 2"
    assert len(analysis.concepts) == 2
    assert len(analysis.claims) == 2


def test_extractive_chunk_analysis_preserves_timestamp_and_transcript_lines():
    chunk = "[12s] First observation.\n[18s] Second observation."

    analysis = build_extractive_chunk_analysis(chunk, index=3)

    assert analysis["heading"] == "Transcript fragment 3"
    assert analysis["timestamp_seconds"] == 12
    assert analysis["bullets"] == ["First observation.", "Second observation."]
    assert "First observation" in analysis["summary"]


def test_parse_deep_graph_chunk_response_accepts_fenced_json_and_normalizes_metadata():
    payload = {
        "chunk_id": "wrong",
        "video_id": "wrong",
        "timestamp_start": 12,
        "timestamp_end": 20,
        "concepts": [
            {
                "local_id": "c1",
                "name": "Support",
                "definition": "A price area.",
                "aliases": [],
                "confidence": 0.9,
                "evidence_quote": "support zone",
                "timestamp_start": 12,
                "timestamp_end": 14,
            }
        ],
        "claims": [],
        "evidence": [],
        "concept_relations": [],
        "claim_relations": [],
        "methods": [],
        "tools": [],
        "metrics": [],
        "risks": [],
        "questions": [],
    }

    parsed = parse_deep_graph_chunk_response(
        "```json\n" + json.dumps(payload) + "\n```",
        expected_video_id="abc123",
        expected_chunk_id="chunk-1",
    )

    assert parsed["video_id"] == "abc123"
    assert parsed["chunk_id"] == "chunk-1"
    assert parsed["concepts"][0]["local_id"] == "c1"


def test_parse_deep_graph_chunk_response_rejects_invalid_json_with_chunk_metadata():
    parsed = parse_deep_graph_chunk_response(
        "not-json",
        expected_video_id="abc123",
        expected_chunk_id="chunk-1",
    )

    assert parsed["rejected"] is True
    assert parsed["chunk_id"] == "chunk-1"
    assert parsed["video_id"] == "abc123"
    assert "Invalid deep graph JSON" in parsed["error"]


def test_extract_deep_graph_chunks_sends_json_schema_and_max_tokens(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            content = {
                "chunk_id": "chunk-1",
                "video_id": "abc123",
                "timestamp_start": 0,
                "timestamp_end": 1,
                "concepts": [],
                "claims": [],
                "evidence": [],
                "concept_relations": [],
                "claim_relations": [],
                "methods": [],
                "tools": [],
                "metrics": [],
                "risks": [],
                "questions": [],
            }
            message = type("Message", (), {"content": json.dumps(content)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    class FakeClient:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.setattr("httpx.Client", FakeClient)
    analyzer = LMStudioAnalyzer(
        "http://localhost:1234/v1",
        "key",
        "qwen",
        "gemma",
        max_prompt_chars=3500,
        deep_graph_max_output_tokens=777,
    )
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(video_id="abc123", source="youtube", segments=[TranscriptSegment(0, 1, "Support zone.")])

    chunks = analyzer.extract_deep_graph_chunks(metadata, transcript, max_chunk_chars=2500)

    assert chunks[0]["chunk_id"] == "chunk-1"
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["max_tokens"] == 777


def test_deep_graph_invalid_json_retry_shrinks_prompt_and_uses_compact_system_prompt(monkeypatch):
    calls = []

    valid_content = {
        "chunk_id": "chunk-1",
        "video_id": "abc123",
        "timestamp_start": 0,
        "timestamp_end": 1,
        "concepts": [],
        "claims": [],
        "evidence": [],
        "concept_relations": [],
        "claim_relations": [],
        "methods": [],
        "tools": [],
        "metrics": [],
        "risks": [],
        "questions": [],
    }
    responses = ["not-json", json.dumps(valid_content)]

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            message = type("Message", (), {"content": responses.pop(0)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    class FakeClient:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.setattr("httpx.Client", FakeClient)
    analyzer = LMStudioAnalyzer(
        "http://localhost:1234/v1",
        "key",
        "qwen",
        "gemma",
        max_prompt_chars=2200,
        retries=1,
    )
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(
        video_id="abc123",
        source="youtube",
        segments=[
            TranscriptSegment(index, 1, f"Line {index} " + ("support breakout " * 8))
            for index in range(8)
        ],
    )

    chunks = analyzer.extract_deep_graph_chunks(metadata, transcript, max_chunk_chars=2500)

    assert len(calls) == 2
    assert len(calls[1]["messages"][1]["content"]) < len(calls[0]["messages"][1]["content"])
    assert "Prefer fewer high-confidence items" in calls[1]["messages"][0]["content"]
    assert chunks[0].get("rejected") is not True


def test_deep_graph_context_length_error_still_splits_chunk(monkeypatch):
    calls = []

    valid_content = {
        "chunk_id": "chunk-1",
        "video_id": "abc123",
        "timestamp_start": 0,
        "timestamp_end": 1,
        "concepts": [],
        "claims": [],
        "evidence": [],
        "concept_relations": [],
        "claim_relations": [],
        "methods": [],
        "tools": [],
        "metrics": [],
        "risks": [],
        "questions": [],
    }

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("context length exceeded")
            message = type("Message", (), {"content": json.dumps(valid_content)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    class FakeClient:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.setattr("httpx.Client", FakeClient)
    analyzer = LMStudioAnalyzer(
        "http://localhost:1234/v1",
        "key",
        "qwen",
        "gemma",
        max_prompt_chars=2200,
        retries=0,
    )
    metadata = VideoMetadata(video_id="abc123", canonical_url="https://youtu.be/abc123", title="Talk")
    transcript = Transcript(
        video_id="abc123",
        source="youtube",
        segments=[
            TranscriptSegment(index, 1, f"Line {index} " + ("context pressure " * 8))
            for index in range(8)
        ],
    )

    chunks = analyzer.extract_deep_graph_chunks(metadata, transcript, max_chunk_chars=2500)

    assert len(calls) >= 3
    assert all(chunk.get("rejected") is not True for chunk in chunks)
