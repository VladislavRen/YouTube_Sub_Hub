from yt_sub_hub.artifact_review import analysis_data_hash, canonical_hash, text_hash, transcript_hash
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import Analysis, Claim, Concept, NoteSection, Transcript, TranscriptSegment, VideoMetadata
from yt_sub_hub.quality import run_quality_gate


def russian_analysis():
    return Analysis(
        language="ru",
        cleaned_title="Русский доклад",
        abstract="Русский научный конспект",
        tags=["граф"],
        outline=["Введение"],
        sections=[NoteSection(heading="Введение", summary="Русское резюме", timestamp_seconds=0, bullets=[])],
        concepts=[Concept(name="Граф знаний", definition="Связная структура понятий")],
        claims=[Claim(text="Граф помогает анализу.", evidence="В докладе это объясняется.")],
        limitations=[],
        open_questions=[],
        related_topics=["знания"],
    )


def english_heading_note(abstract: str = "English abstract.") -> str:
    return "\n".join(
        [
            "---",
            'language: "ru"',
            "---",
            "",
            "# Русский доклад",
            "",
            "## Abstract",
            "",
            abstract,
            "",
            "## Detailed Notes",
            "",
            "Русское содержание.",
            "",
        ]
    )


def write_russian_video(store: LibraryStore, video_id: str = "abc123def45") -> None:
    metadata = VideoMetadata(
        video_id=video_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
        title="Talk",
        channel="Channel",
    )
    source = Transcript(
        video_id=video_id,
        source="youtube",
        language="en",
        segments=[TranscriptSegment(0, 1, "English text")],
    )
    translated = Transcript(
        video_id=video_id,
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(0, 1, "Русский текст", raw_text="English text")],
    )
    analysis = russian_analysis()
    store.save_video_artifacts(metadata, source, source, analysis, "# Русский конспект\n\nТекст.")
    store.save_translated_transcript(video_id, translated)
    store.save_analysis_artifact(video_id, analysis, variant="ru", compatibility=True)
    store.save_graph_extraction(
        video_id,
        {
            "schema_version": "graph-extraction-v1",
            "video_id": video_id,
            "generated_from": {
                "transcript": "transcript.ru.json",
                "analysis": "analysis.ru.json",
                "language": "ru",
            },
            "nodes": [
                {"id": f"video:{video_id}", "label": "Video"},
                {"id": "segment:abc123def45:0", "label": "SourceSegment"},
                {"id": "concept:graf-znanii", "label": "Concept"},
            ],
            "relationships": [
                {
                    "id": "rel1",
                    "from": f"video:{video_id}",
                    "to": "concept:graf-znanii",
                    "type": "MENTIONS",
                    "source_video_id": video_id,
                }
            ],
            "quality_report": {},
        },
    )
    paths = store.artifact_paths(video_id)
    paths.ru_artifact_review.parent.mkdir(parents=True, exist_ok=True)
    graph = store.load_graph_extraction(video_id)
    LibraryStore._write_json(
        paths.ru_artifact_review,
        {
            "video_id": video_id,
            "target_language": "ru",
            "status": "passed",
            "reviewer": {"kind": "deterministic", "version": "ru-artifact-review-v1"},
            "checked_artifacts": {
                "transcript": "transcript.ru.json",
                "analysis": "analysis.ru.json",
                "markdown": "note.md",
                "graph": "graph_extraction.json",
            },
            "hashes": {
                "transcript_ru_hash": transcript_hash(translated),
                "analysis_ru_hash": analysis_data_hash(store.load_analysis(video_id, variant="ru")),
                "note_hash": text_hash(store.note_path(video_id).read_text(encoding="utf-8")),
                "graph_hash": canonical_hash(graph),
            },
            "coverage": {
                "transcript_segments_total": 1,
                "analysis_sections": 1,
                "analysis_concepts": 1,
                "analysis_claims": 1,
                "covered_segment_ratio": 1.0,
            },
            "issues": [],
        },
    )


def test_quality_gate_passes_for_complete_russian_video(tmp_path):
    store = LibraryStore(tmp_path)
    write_russian_video(store)

    result = run_quality_gate(store, "abc123def45")

    assert result.status == "completed"
    assert result.hard_failures == []
    assert result.as_dict()["video_id"] == "abc123def45"


def test_quality_gate_fails_when_russian_analysis_or_note_is_invalid(tmp_path):
    store = LibraryStore(tmp_path)
    write_russian_video(store)
    bad_analysis = russian_analysis()
    bad_analysis.language = "en"
    store.save_analysis_artifact("abc123def45", bad_analysis, variant="ru", compatibility=True)
    store.note_path("abc123def45").write_text("# English note only", encoding="utf-8")

    result = run_quality_gate(store, "abc123def45")

    assert result.status == "failed"
    assert "analysis.ru.json language must be ru" in result.hard_failures
    assert "note.md must contain Cyrillic text" in result.hard_failures


def test_quality_gate_fails_when_analysis_ru_is_stale_non_cyrillic_content(tmp_path):
    store = LibraryStore(tmp_path)
    write_russian_video(store)
    stale = Analysis(
        language="ru",
        cleaned_title="English talk",
        abstract="English scientific summary",
        tags=["graph"],
        outline=["Intro"],
        sections=[NoteSection(heading="Intro", summary="English summary", timestamp_seconds=0, bullets=[])],
        concepts=[Concept(name="Knowledge graph", definition="Connected concept structure")],
        claims=[Claim(text="Graphs help analysis.", evidence="The talk explains it.")],
        limitations=[],
        open_questions=[],
        related_topics=["knowledge"],
    )
    store.save_analysis_artifact("abc123def45", stale, variant="ru", compatibility=True)

    result = run_quality_gate(store, "abc123def45")

    assert result.status == "failed"
    assert "analysis.ru.json abstract contains blocked English phrase: Scientific summary" in result.hard_failures
    assert any("analysis.ru.json sections[0].summary ru_word_ratio" in failure for failure in result.hard_failures)


def test_quality_gate_fails_when_analysis_ru_has_english_abstract_with_russian_title(tmp_path):
    store = LibraryStore(tmp_path)
    write_russian_video(store)
    stale = russian_analysis()
    stale.abstract = "Scientific summary of 'РУССКОЕ НАЗВАНИЕ'. The note is assembled from source transcript."
    store.save_analysis_artifact("abc123def45", stale, variant="ru", compatibility=True)

    result = run_quality_gate(store, "abc123def45")

    assert result.status == "failed"
    assert "analysis.ru.json abstract contains blocked English phrase: Scientific summary" in result.hard_failures
    assert "analysis.ru.json abstract contains blocked English phrase: The note is assembled" in result.hard_failures


def test_quality_gate_fails_when_note_md_has_english_abstract_heading(tmp_path):
    store = LibraryStore(tmp_path)
    write_russian_video(store)
    store.note_path("abc123def45").write_text(english_heading_note("Scientific summary of the talk."), encoding="utf-8")

    result = run_quality_gate(store, "abc123def45")

    assert result.status == "failed"
    assert "note.md contains English markdown heading: ## Abstract" in result.hard_failures
    assert "note.md contains blocked English phrase: Scientific summary" in result.hard_failures


def test_quality_gate_fails_when_claim_evidence_is_mixed_language(tmp_path):
    store = LibraryStore(tmp_path)
    write_russian_video(store)
    stale = russian_analysis()
    stale.claims[0].evidence = "The video explains this with one русский word."
    store.save_analysis_artifact("abc123def45", stale, variant="ru", compatibility=True)

    result = run_quality_gate(store, "abc123def45")

    assert result.status == "failed"
    assert any("analysis.ru.json claims[0].evidence ru_word_ratio" in failure for failure in result.hard_failures)


def test_quality_gate_fails_when_concept_definition_is_english(tmp_path):
    store = LibraryStore(tmp_path)
    write_russian_video(store)
    stale = russian_analysis()
    stale.concepts[0].definition = "Connected concept structure."
    store.save_analysis_artifact("abc123def45", stale, variant="ru", compatibility=True)

    result = run_quality_gate(store, "abc123def45")

    assert result.status == "failed"
    assert any("analysis.ru.json concepts[0].definition ru_word_ratio" in failure for failure in result.hard_failures)


def test_quality_gate_fails_when_transcript_ru_has_no_cyrillic(tmp_path):
    store = LibraryStore(tmp_path)
    write_russian_video(store)
    transcript = Transcript(
        video_id="abc123def45",
        source="youtube",
        language="ru",
        segments=[TranscriptSegment(0, 1, "English stale text", raw_text="English text")],
    )
    store.save_translated_transcript("abc123def45", transcript)

    result = run_quality_gate(store, "abc123def45")

    assert result.status == "failed"
    assert "transcript.ru.json must contain Cyrillic text" in result.hard_failures


def test_quality_gate_fails_when_graph_provenance_uses_source_artifacts(tmp_path):
    store = LibraryStore(tmp_path)
    write_russian_video(store)
    extraction = store.load_graph_extraction("abc123def45")
    extraction["generated_from"] = {
        "transcript": "transcript.normalized.json",
        "analysis": "analysis.source.json",
        "language": "ru",
    }
    store.save_graph_extraction("abc123def45", extraction)

    result = run_quality_gate(store, "abc123def45")

    assert result.status == "failed"
    assert "graph_extraction.generated_from.transcript must be transcript.ru.json" in result.hard_failures
    assert "graph_extraction.generated_from.analysis must be analysis.ru.json" in result.hard_failures


def test_quality_gate_fails_when_graph_analysis_hash_is_stale(tmp_path):
    store = LibraryStore(tmp_path)
    write_russian_video(store)
    extraction = store.load_graph_extraction("abc123def45")
    extraction["generated_from"]["analysis_hash"] = "stale-analysis-hash"
    store.save_graph_extraction("abc123def45", extraction)

    result = run_quality_gate(store, "abc123def45")

    assert result.status == "failed"
    assert "graph_extraction.generated_from.analysis_hash must match analysis.ru.json" in result.hard_failures


def test_quality_gate_warns_for_small_graph_but_keeps_required_russian_gates(tmp_path):
    store = LibraryStore(tmp_path)
    write_russian_video(store)
    extraction = store.load_graph_extraction("abc123def45")
    extraction["nodes"] = [{"id": "video:abc123def45", "label": "Video"}]
    extraction["relationships"] = []
    store.save_graph_extraction("abc123def45", extraction)
    report_path = store.artifact_paths("abc123def45").ru_artifact_review
    report = LibraryStore._read_json(report_path)
    report["hashes"]["graph_hash"] = canonical_hash(store.load_graph_extraction("abc123def45"))
    LibraryStore._write_json(report_path, report)

    result = run_quality_gate(store, "abc123def45")

    assert result.status == "completed_with_warning"
    assert "graph should contain SourceSegment nodes" in result.warnings
    assert "graph should contain at least one Concept, Claim, or Topic node" in result.warnings
