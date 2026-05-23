from __future__ import annotations

import json
from dataclasses import replace

from yt_sub_hub.artifact_review import analysis_data_hash, review_ru_artifacts, transcript_hash
from yt_sub_hub.config import get_settings
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import Analysis, Claim, Concept, NoteSection, Transcript, TranscriptSegment, VideoMetadata
from yt_sub_hub.notes import render_markdown_note


def _analysis(language: str = "ru", *, count: int = 6, generic: bool = False, english_abstract: bool = False) -> Analysis:
    if generic:
        concepts = [
            Concept("Основная тема", "Главная тема видео описана в расшифровке."),
            Concept("Контекст видео", "Контекст видео взят из русской расшифровки."),
            Concept("Ключевой эпизод", "Ключевой эпизод выделен по русской расшифровке."),
        ]
        claims = [
            Claim("Раздел раскрывает важную часть темы видео.", "Русская расшифровка содержит это объяснение."),
            Claim("Видео содержит несколько важных эпизодов.", "Русская расшифровка показывает последовательность событий."),
            Claim("Материал требует проверки деталей.", "Автоматическая расшифровка может содержать неточности."),
        ]
        sections = [NoteSection(f"Раздел {index + 1}", "Русское резюме раздела с содержанием видео.", index * 10, []) for index in range(3)]
    else:
        concepts = [
            Concept(f"Понятие {index}", f"Связное русское определение понятия номер {index} для анализа видео.")
            for index in range(count)
        ]
        claims = [
            Claim(f"Русское утверждение номер {index} объясняет важную часть видео.", "В русской расшифровке приведено подробное подтверждение.")
            for index in range(count)
        ]
        sections = [
            NoteSection(f"Раздел {index + 1}", "Русское резюме раздела с подробным содержанием видео.", index * 10, ["Русский пункт раздела."])
            for index in range(max(3, count // 2))
        ]
    return Analysis(
        language=language,
        cleaned_title="Русский доклад",
        abstract="Scientific summary of 'Русский доклад'. The note is assembled from source transcript." if english_abstract else "Русский научный конспект с подробным описанием видео.",
        tags=["видео", "анализ"],
        outline=[section.heading for section in sections],
        sections=sections,
        concepts=concepts,
        claims=claims,
        limitations=["Ограничения основаны на автоматической расшифровке."],
        open_questions=["Какие детали требуют дополнительной проверки?"],
        related_topics=["видео", "анализ"],
    )


def _transcript(count: int = 12) -> Transcript:
    return Transcript(
        video_id="abc123def45",
        source="whisper",
        language="ru",
        segments=[TranscriptSegment(index * 5, 5, f"Русский сегмент {index} про понятие {index % 6}.") for index in range(count)],
    )


def _transcript_with_duration(count: int, duration_seconds: float) -> Transcript:
    segment_duration = duration_seconds / max(1, count)
    return Transcript(
        video_id="abc123def45",
        source="whisper",
        language="ru",
        segments=[
            TranscriptSegment(index * segment_duration, segment_duration, f"Русский сегмент {index} про покрытие расшифровки.")
            for index in range(count)
        ],
    )


def _write_review_video(store: LibraryStore, *, source_count: int = 12, ru_count: int = 6, generic: bool = False) -> tuple[Analysis, Transcript]:
    video_id = "abc123def45"
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://www.youtube.com/watch?v={video_id}", title="Talk", channel="Channel")
    transcript = _transcript(600 if source_count >= 40 else 12)
    source = _analysis("ru", count=source_count)
    ru = _analysis("ru", count=ru_count, generic=generic)
    store.save_video_artifacts(metadata, transcript, transcript, source, "# Черновик")
    store.save_translated_transcript(video_id, transcript)
    store.save_analysis_artifact(video_id, source, variant="source")
    source_data = store.load_analysis(video_id, variant="source")
    store.save_analysis_artifact(
        video_id,
        ru,
        variant="ru",
        compatibility=True,
        extra_fields={
            "source_analysis_hash": analysis_data_hash(source_data),
            "source_transcript_hash": transcript_hash(transcript),
            "generated_from": {"analysis": "analysis.source.json", "transcript": "transcript.ru.json", "target_language": "ru"},
        },
    )
    ru_data = store.load_analysis(video_id, variant="ru")
    note = render_markdown_note(
        metadata,
        ru,
        provenance={
            "analysis_hash": analysis_data_hash(ru_data),
            "transcript_hash": transcript_hash(transcript),
            "generated_from_analysis": "analysis.ru.json",
        },
    )
    store.save_note_artifact(video_id, note)
    store.save_graph_extraction(
        video_id,
        {
            "schema_version": "graph-extraction-v1",
            "video_id": video_id,
            "generated_from": {
                "transcript": "transcript.ru.json",
                "transcript_hash": transcript_hash(transcript),
                "analysis": "analysis.ru.json",
                "analysis_hash": analysis_data_hash(ru_data),
                "language": "ru",
            },
            "nodes": [{"id": f"video:{video_id}", "label": "Video"}]
            + [{"id": f"segment:{video_id}:{index}", "label": "SourceSegment", "text": segment.text} for index, segment in enumerate(transcript.segments)]
            + [{"id": f"concept:{concept.slug}", "label": "Concept", "name": concept.name} for concept in ru.concepts],
            "relationships": [{"id": "rel1", "from": f"video:{video_id}", "to": f"concept:{ru.concepts[0].slug}", "type": "MENTIONS", "source_video_id": video_id}],
            "quality_report": {},
        },
    )
    return ru, transcript


def _write_review_video_with_transcript(store: LibraryStore, transcript: Transcript, *, source_count: int = 12, ru_count: int = 12) -> tuple[Analysis, Transcript]:
    video_id = transcript.video_id
    metadata = VideoMetadata(video_id=video_id, canonical_url=f"https://www.youtube.com/watch?v={video_id}", title="Talk", channel="Channel")
    source = _analysis("ru", count=source_count)
    ru = _analysis("ru", count=ru_count)
    store.save_video_artifacts(metadata, transcript, transcript, source, "# Черновик")
    store.save_translated_transcript(video_id, transcript)
    store.save_analysis_artifact(video_id, source, variant="source")
    source_data = store.load_analysis(video_id, variant="source")
    store.save_analysis_artifact(
        video_id,
        ru,
        variant="ru",
        compatibility=True,
        extra_fields={
            "source_analysis_hash": analysis_data_hash(source_data),
            "source_transcript_hash": transcript_hash(transcript),
            "generated_from": {"analysis": "analysis.source.json", "transcript": "transcript.ru.json", "target_language": "ru"},
        },
    )
    ru_data = store.load_analysis(video_id, variant="ru")
    note = render_markdown_note(
        metadata,
        ru,
        provenance={
            "analysis_hash": analysis_data_hash(ru_data),
            "transcript_hash": transcript_hash(transcript),
            "generated_from_analysis": "analysis.ru.json",
        },
    )
    store.save_note_artifact(video_id, note)
    store.save_graph_extraction(
        video_id,
        {
            "schema_version": "graph-extraction-v1",
            "video_id": video_id,
            "generated_from": {
                "transcript": "transcript.ru.json",
                "transcript_hash": transcript_hash(transcript),
                "analysis": "analysis.ru.json",
                "analysis_hash": analysis_data_hash(ru_data),
                "language": "ru",
            },
            "nodes": [{"id": f"video:{video_id}", "label": "Video"}]
            + [{"id": f"segment:{video_id}:{index}", "label": "SourceSegment", "text": segment.text} for index, segment in enumerate(transcript.segments)]
            + [{"id": f"concept:{concept.slug}", "label": "Concept", "name": concept.name} for concept in ru.concepts],
            "relationships": [{"id": "rel1", "from": f"video:{video_id}", "to": f"concept:{ru.concepts[0].slug}", "type": "MENTIONS", "source_video_id": video_id}],
            "quality_report": {},
        },
    )
    return ru, transcript


def _replace_ru_analysis_data(store: LibraryStore, video_id: str, analysis_data: dict) -> None:
    paths = store.artifact_paths(video_id)
    LibraryStore._write_json(paths.analysis_ru, analysis_data)
    LibraryStore._write_json(paths.analysis, analysis_data)
    analysis_hash = analysis_data_hash(analysis_data)
    note = store.note_path(video_id).read_text(encoding="utf-8")
    note = "\n".join(
        f"analysis_hash: {analysis_hash}" if line.startswith("analysis_hash:") else line
        for line in note.splitlines()
    )
    store.note_path(video_id).write_text(note, encoding="utf-8")
    graph = store.load_graph_extraction(video_id)
    graph["generated_from"]["analysis_hash"] = analysis_hash
    store.save_graph_extraction(video_id, graph)


def _write_analysis_chunk(
    store: LibraryStore,
    video_id: str,
    *,
    index: int,
    total: int,
    start: int,
    end: int,
    source_hash: str,
    language: str = "ru",
    source_transcript: str = "transcript.ru.json",
) -> None:
    path = store.video_dir(video_id) / "analysis" / "chunks" / f"{index:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "chunk_index": index,
                "total_chunks": total,
                "source_transcript": source_transcript,
                "source_segment_start_index": start,
                "source_segment_end_index": end,
                "source_transcript_hash": source_hash,
                "language": language,
                "summary": "Русская сводка фрагмента.",
                "outline": ["Русский фрагмент"],
                "concepts": [],
                "claims": [],
                "evidence_segment_indexes": list(range(start, end + 1)),
                "limitations": [],
                "open_questions": [],
                "related_topics": [],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _settings(tmp_path):
    return replace(get_settings(), library_dir=tmp_path, enable_translation=True, target_language="ru")


def test_good_ru_artifacts_pass(tmp_path):
    store = LibraryStore(tmp_path)
    _write_review_video(store)

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert report.status == "passed"
    assert report.issues == []
    assert store.artifact_paths("abc123def45").ru_artifact_review.exists()


def test_english_abstract_with_cyrillic_title_fails(tmp_path):
    store = LibraryStore(tmp_path)
    _write_review_video(store)
    bad = _analysis("ru", count=6, english_abstract=True)
    source_data = store.load_analysis("abc123def45", variant="source")
    transcript = store.load_translated_transcript("abc123def45", "ru")
    store.save_analysis_artifact(
        "abc123def45",
        bad,
        variant="ru",
        compatibility=True,
        extra_fields={"source_analysis_hash": analysis_data_hash(source_data), "source_transcript_hash": transcript_hash(transcript)},
    )

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert report.status == "failed"
    assert any(issue.code == "ru_analysis_language_quality" for issue in report.issues)


def test_stale_note_hash_fails(tmp_path):
    store = LibraryStore(tmp_path)
    _write_review_video(store)
    store.note_path("abc123def45").write_text(store.note_path("abc123def45").read_text(encoding="utf-8").replace("analysis_hash:", "analysis_hash: stale-"), encoding="utf-8")

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert any(issue.code == "stale_markdown_analysis_hash" for issue in report.issues)


def test_stale_graph_hash_fails(tmp_path):
    store = LibraryStore(tmp_path)
    _write_review_video(store)
    graph = store.load_graph_extraction("abc123def45")
    graph["generated_from"]["analysis_hash"] = "stale"
    store.save_graph_extraction("abc123def45", graph)

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert any(issue.code == "stale_graph_analysis_hash" for issue in report.issues)


def test_sparse_generic_ru_analysis_fails_against_rich_source(tmp_path):
    store = LibraryStore(tmp_path)
    _write_review_video(store, source_count=40, ru_count=3, generic=True)

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert report.status == "failed"
    assert any(issue.code == "analysis_ru_sparse_concepts" for issue in report.issues)


def test_missing_graph_fails(tmp_path):
    store = LibraryStore(tmp_path)
    _write_review_video(store)
    store.artifact_paths("abc123def45").graph_extraction.unlink()

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert any(issue.code == "missing_graph" for issue in report.issues)


def test_source_language_leakage_in_markdown_fails(tmp_path):
    store = LibraryStore(tmp_path)
    _write_review_video(store)
    store.note_path("abc123def45").write_text(store.note_path("abc123def45").read_text(encoding="utf-8") + "\n\nEvidence: The video explains this.\n", encoding="utf-8")

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert any(issue.code == "markdown_language_quality" or issue.code == "markdown_source_language_leakage" for issue in report.issues)


def test_evidence_indexes_full_coverage_passes(tmp_path):
    store = LibraryStore(tmp_path)
    transcript = _transcript(100)
    _write_review_video_with_transcript(store, transcript)
    analysis_data = store.load_analysis("abc123def45", variant="ru")
    analysis_data["claims"][0]["segment_range"] = [0, 39]
    analysis_data["claims"][1]["source_segment_start_index"] = 40
    analysis_data["claims"][1]["source_segment_end_index"] = 79
    _replace_ru_analysis_data(store, "abc123def45", analysis_data)

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert report.coverage["coverage_source"] == "evidence_indexes"
    assert report.coverage["covered_segments_count"] == 80
    assert report.coverage["covered_segment_ratio"] == 0.8
    assert not any(issue.severity == "hard" for issue in report.issues)


def test_evidence_low_coverage_fails_for_long_transcript(tmp_path):
    store = LibraryStore(tmp_path)
    transcript = _transcript(600)
    _write_review_video_with_transcript(store, transcript, source_count=40, ru_count=12)
    analysis_data = store.load_analysis("abc123def45", variant="ru")
    analysis_data["claims"][0]["evidence_segment_indexes"] = list(range(100))
    _replace_ru_analysis_data(store, "abc123def45", analysis_data)

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert report.coverage["coverage_source"] == "evidence_indexes"
    assert report.coverage["covered_segments_count"] == 100
    assert any(issue.code == "low_transcript_coverage_evidence" and issue.severity == "hard" for issue in report.issues)


def test_two_evidence_indexes_are_not_treated_as_range(tmp_path):
    store = LibraryStore(tmp_path)
    transcript = _transcript(1000)
    _write_review_video_with_transcript(store, transcript, source_count=40, ru_count=12)
    analysis_data = store.load_analysis("abc123def45", variant="ru")
    analysis_data["claims"][0]["evidence_segment_indexes"] = [0, 999]
    _replace_ru_analysis_data(store, "abc123def45", analysis_data)

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert report.coverage["coverage_source"] == "evidence_indexes"
    assert report.coverage["covered_segments_count"] == 2
    assert report.coverage["covered_segment_ratio"] == 0.002
    assert any(issue.code == "low_transcript_coverage_evidence" and issue.severity == "hard" for issue in report.issues)


def test_chunk_coverage_passes_without_evidence_indexes(tmp_path):
    store = LibraryStore(tmp_path)
    transcript = _transcript(1000)
    _write_review_video_with_transcript(store, transcript, source_count=40, ru_count=12)
    source_hash = transcript_hash(transcript)
    _write_analysis_chunk(store, "abc123def45", index=0, total=2, start=0, end=499, source_hash=source_hash)
    _write_analysis_chunk(store, "abc123def45", index=1, total=2, start=500, end=999, source_hash=source_hash)

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert report.coverage["coverage_source"] == "analysis_chunks"
    assert report.coverage["covered_segments_count"] == 1000
    assert report.coverage["covered_segment_ratio"] == 1.0
    assert report.coverage["chunk_artifacts_total"] == 2
    assert report.coverage["chunk_artifacts_valid"] == 2
    assert not any(issue.code == "low_transcript_coverage_chunks" for issue in report.issues)


def test_chunk_with_out_of_bounds_end_is_stale_not_full_coverage(tmp_path):
    store = LibraryStore(tmp_path)
    transcript = _transcript(600)
    _write_review_video_with_transcript(store, transcript, source_count=40, ru_count=12)
    _write_analysis_chunk(store, "abc123def45", index=0, total=1, start=0, end=999, source_hash=transcript_hash(transcript))

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert report.coverage["chunk_artifacts_total"] == 1
    assert report.coverage["chunk_artifacts_valid"] == 0
    assert report.coverage["chunk_artifacts_stale"] == 1
    assert report.coverage["covered_segment_ratio"] is None
    assert any(issue.code == "stale_analysis_chunk_coverage" and issue.severity == "hard" for issue in report.issues)


def test_stale_chunks_are_ignored_and_fail_without_alternative_coverage(tmp_path):
    store = LibraryStore(tmp_path)
    transcript = _transcript(1000)
    _write_review_video_with_transcript(store, transcript, source_count=40, ru_count=12)
    _write_analysis_chunk(store, "abc123def45", index=0, total=1, start=0, end=999, source_hash="stale")

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert report.coverage["chunk_artifacts_total"] == 1
    assert report.coverage["chunk_artifacts_valid"] == 0
    assert report.coverage["chunk_artifacts_stale"] == 1
    assert any(issue.code == "stale_analysis_chunk_coverage" and issue.severity == "hard" for issue in report.issues)


def test_stale_chunks_do_not_block_timestamp_fallback_when_span_is_valid(tmp_path):
    store = LibraryStore(tmp_path)
    transcript = _transcript_with_duration(1000, 4000)
    _write_review_video_with_transcript(store, transcript, source_count=40, ru_count=10)
    _write_analysis_chunk(store, "abc123def45", index=0, total=1, start=0, end=999, source_hash="stale")
    analysis_data = store.load_analysis("abc123def45", variant="ru")
    for section, timestamp in zip(analysis_data["sections"], [0, 1000, 2000, 3000, 3500], strict=False):
        section["timestamp_seconds"] = timestamp
    _replace_ru_analysis_data(store, "abc123def45", analysis_data)

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert report.coverage["coverage_source"] == "section_timestamps"
    assert report.coverage["chunk_artifacts_stale"] == 1
    assert report.coverage["timestamp_span_ratio"] > 0.8
    assert not any(issue.code == "stale_analysis_chunk_coverage" for issue in report.issues)


def test_timestamp_fallback_does_not_divide_sections_by_segments(tmp_path):
    store = LibraryStore(tmp_path)
    transcript = _transcript_with_duration(1000, 4000)
    _write_review_video_with_transcript(store, transcript, source_count=40, ru_count=10)
    analysis_data = store.load_analysis("abc123def45", variant="ru")
    for section, timestamp in zip(analysis_data["sections"], [0, 1000, 2000, 3000, 3500], strict=False):
        section["timestamp_seconds"] = timestamp
    _replace_ru_analysis_data(store, "abc123def45", analysis_data)

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert report.coverage["coverage_source"] == "section_timestamps"
    assert report.coverage["timestamp_span_ratio"] > 0.8
    assert report.coverage["covered_segment_ratio"] is None
    assert report.coverage["covered_segment_ratio"] != 4 / 1000


def test_timestamp_fallback_low_span_fails_for_long_transcript(tmp_path):
    store = LibraryStore(tmp_path)
    transcript = _transcript_with_duration(1000, 4000)
    _write_review_video_with_transcript(store, transcript, source_count=40, ru_count=10)
    analysis_data = store.load_analysis("abc123def45", variant="ru")
    for section, timestamp in zip(analysis_data["sections"], [0, 50, 100, 150, 200], strict=False):
        section["timestamp_seconds"] = timestamp
    _replace_ru_analysis_data(store, "abc123def45", analysis_data)

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert report.coverage["coverage_source"] == "section_timestamps"
    assert report.coverage["timestamp_span_ratio"] < 0.25
    assert any(issue.code == "low_transcript_coverage_timestamps" and issue.severity == "hard" for issue in report.issues)


def test_live_like_pl6ziily1me_uses_chunk_coverage(tmp_path):
    store = LibraryStore(tmp_path)
    transcript = _transcript_with_duration(1415, 3823.36)
    _write_review_video_with_transcript(store, transcript, source_count=40, ru_count=12)
    source_hash = transcript_hash(transcript)
    ranges = [(0, 54), (55, 124), (125, 210), (211, 1414)]
    for index, (start, end) in enumerate(ranges):
        _write_analysis_chunk(store, "abc123def45", index=index, total=len(ranges), start=start, end=end, source_hash=source_hash)
    _write_analysis_chunk(store, "abc123def45", index=20, total=22, start=1328, end=1397, source_hash="stale", language="source", source_transcript="transcript.normalized.json")
    _write_analysis_chunk(store, "abc123def45", index=21, total=22, start=1398, end=1414, source_hash="stale", language="source", source_transcript="transcript.normalized.json")

    report = review_ru_artifacts("abc123def45", store, _settings(tmp_path))

    assert report.coverage["coverage_source"] == "analysis_chunks"
    assert report.coverage["covered_segment_ratio"] > 0.9
    assert report.coverage["chunk_artifacts_valid"] == 4
    assert report.coverage["chunk_artifacts_stale"] == 2
    assert not any(issue.severity == "hard" for issue in report.issues)
