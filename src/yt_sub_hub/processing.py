from __future__ import annotations

from yt_sub_hub.models import Transcript


def chunk_transcript(transcript: Transcript, max_chars: int = 6000) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for segment in transcript.segments:
        line = f"[{int(segment.start)}s] {segment.text}"
        if current and current_size + len(line) > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_size = 0
        current.append(line)
        current_size += len(line)

    if current:
        chunks.append("\n".join(current))
    return chunks
