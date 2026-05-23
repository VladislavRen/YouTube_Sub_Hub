from pathlib import Path

from yt_sub_hub.library import LibraryStore
from yt_sub_hub.models import Analysis, Claim, Concept, NoteSection, Transcript, TranscriptSegment, VideoMetadata
from yt_sub_hub.notes import render_markdown_note


def sample_metadata() -> VideoMetadata:
    return VideoMetadata(
        video_id="abc123",
        canonical_url="https://www.youtube.com/watch?v=abc123",
        title="Scientific Talk",
        channel="Research Channel",
        duration_seconds=120,
    )


def sample_analysis() -> Analysis:
    return Analysis(
        language="en",
        cleaned_title="Scientific Talk",
        abstract="A compact scientific abstract.",
        tags=["science", "methods"],
        outline=["Motivation", "Method"],
        sections=[
            NoteSection(
                heading="Motivation",
                summary="Why the problem matters.",
                timestamp_seconds=12,
                bullets=["Defines the problem.", "States the research gap."],
            )
        ],
        concepts=[Concept(name="Causal inference", definition="Reasoning about cause and effect.")],
        claims=[Claim(text="The method improves identification.", evidence="Speaker compares assumptions.")],
        limitations=["The talk does not include empirical replication."],
        open_questions=["How robust is the method on noisy data?"],
        related_topics=["Experimental design"],
    )


def sample_russian_analysis() -> Analysis:
    return Analysis(
        language="ru",
        cleaned_title="Русский доклад",
        abstract="Русская аннотация подробно описывает главную тему видео.",
        tags=["наука", "методы"],
        outline=["Мотивация", "Метод"],
        sections=[
            NoteSection(
                heading="Мотивация",
                summary="Раздел объясняет, почему исследовательская проблема важна.",
                timestamp_seconds=12,
                bullets=["Определяется проблема.", "Формулируется исследовательский пробел."],
            )
        ],
        concepts=[Concept(name="Причинный вывод", definition="Рассуждение о причинах и следствиях.")],
        claims=[Claim(text="Метод улучшает идентификацию.", evidence="Докладчик сравнивает исходные предположения.")],
        limitations=["Доклад не включает эмпирическую репликацию."],
        open_questions=["Насколько метод устойчив к шумным данным?"],
        related_topics=["Экспериментальный дизайн"],
    )


def test_render_markdown_note_contains_required_scientific_sections_and_timestamps():
    markdown = render_markdown_note(sample_metadata(), sample_analysis())

    assert "---" in markdown
    assert "# Scientific Talk" in markdown
    assert "## Abstract" in markdown
    assert "## Detailed Notes" in markdown
    assert "[00:12](https://youtu.be/abc123?t=12)" in markdown
    assert "## Key Concepts" in markdown
    assert "## Claims And Arguments" in markdown
    assert "## Limitations" in markdown
    assert "## Open Questions" in markdown
    assert "## Related Topics" in markdown
    assert "**Claim:**" in markdown
    assert "Evidence:" in markdown


def test_render_markdown_note_uses_russian_labels_for_russian_analysis():
    markdown = render_markdown_note(sample_metadata(), sample_russian_analysis())

    assert "## Аннотация" in markdown
    assert "## План" in markdown
    assert "## Подробные заметки" in markdown
    assert "## Ключевые понятия" in markdown
    assert "## Утверждения" in markdown
    assert "## Ограничения" in markdown
    assert "## Открытые вопросы" in markdown
    assert "## Связанные темы" in markdown
    assert "**Утверждение:**" in markdown
    assert "Доказательство:" in markdown
    assert "## Abstract" not in markdown
    assert "Evidence:" not in markdown


def test_render_markdown_frontmatter_escapes_quotes_and_newlines():
    metadata = sample_metadata()
    metadata.title = 'He said "test"\nSecond line'

    markdown = render_markdown_note(metadata, sample_analysis())

    assert 'title: "He said \\"test\\" Second line"' in markdown


def test_library_store_writes_all_video_artifacts(tmp_path: Path):
    transcript = Transcript(
        video_id="abc123",
        source="youtube",
        segments=[TranscriptSegment(start=0, duration=2, text="Hello world")],
    )
    store = LibraryStore(tmp_path)

    paths = store.save_video_artifacts(sample_metadata(), transcript, transcript, sample_analysis(), "# Note")

    assert paths.note.read_text() == "# Note"
    assert paths.metadata.exists()
    assert paths.raw_transcript.exists()
    assert paths.normalized_transcript.exists()
    assert paths.analysis.exists()
    assert paths.video_dir == tmp_path / "videos" / "abc123"
