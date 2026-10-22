from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
import socket
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from yt_sub_hub.graph import GraphPayload, Neo4jGraphStore
from yt_sub_hub.library import LibraryStore
from yt_sub_hub.llm import parse_analysis_response
from yt_sub_hub.models import slugify


SCHEMA_VERSION = "research-enrichment-v1"
TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid", "igshid", "ref", "spm"}
SUPPORTED_CONTENT_TYPES = ("text/html", "text/plain")
DEFAULT_MAX_CONTENT_CHARS = 1_000_000
MAX_FETCH_REDIRECTS = 5


class ResearchValidationError(ValueError):
    pass


class ResearchStatus(Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    PASSED = "completed"
    COMPLETED_WITH_WARNING = "completed_with_warning"
    FAILED = "failed"


@dataclass(frozen=True)
class SafeUrlResult:
    allowed: bool
    url: str
    reason: str = ""
    hostname: str = ""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    rank: int
    query_id: str = ""
    provider: str = "duckduckgo"
    result_id: str = ""
    canonical_url: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical = canonicalize_url(self.canonical_url or self.url)
        object.__setattr__(self, "url", canonical)
        object.__setattr__(self, "canonical_url", canonical)
        if not self.result_id:
            object.__setattr__(self, "result_id", _stable_id("result", self.query_id, canonical, length=16))


@dataclass(frozen=True)
class ResearchQuery:
    query_id: str
    query: str
    intent: str
    source_claim_ids: list[str] = field(default_factory=list)
    priority: int = 100
    language: str = "ru"
    created_by: str = "deterministic"


@dataclass(frozen=True)
class FetchedSource:
    url: str
    title: str
    text: str
    source_id: str = ""
    result_ids: list[str] = field(default_factory=list)
    canonical_url: str = ""
    final_url: str = ""
    status_code: int | None = 200
    content_type: str = "text/plain"
    retrieved_at: str = ""
    content_hash: str = ""
    text_path: str = ""
    metadata_path: str = ""
    fetch_error: str | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        canonical = canonicalize_url(self.canonical_url or self.url)
        object.__setattr__(self, "url", canonical)
        object.__setattr__(self, "canonical_url", canonical)
        object.__setattr__(self, "final_url", self.final_url or canonical)
        if not self.source_id:
            object.__setattr__(self, "source_id", _stable_id("source", canonical, length=16))
        if not self.retrieved_at:
            object.__setattr__(self, "retrieved_at", utc_now())
        if not self.content_hash and self.text:
            object.__setattr__(self, "content_hash", text_hash(self.text))

    @property
    def fetched(self) -> bool:
        return self.fetch_error is None and self.status_code is not None and 200 <= int(self.status_code) < 300 and bool(self.text)


@dataclass(frozen=True)
class Citation:
    url: str
    quote: str
    source_id: str = ""
    citation_id: str = ""
    quote_start_char: int | None = None
    quote_end_char: int | None = None
    quote_validation: str = ""

    def __post_init__(self) -> None:
        canonical = canonicalize_url(self.url)
        object.__setattr__(self, "url", canonical)
        if not self.citation_id:
            object.__setattr__(self, "citation_id", _stable_id("citation", canonical, self.quote, length=16))


@dataclass(frozen=True)
class ResearchClaim:
    text: str
    citations: list[Citation]
    claim_id: str = ""
    finding_type: str = "context"
    confidence: float = 0.8
    accepted: bool = True

    def __post_init__(self) -> None:
        if not self.claim_id:
            object.__setattr__(self, "claim_id", _stable_id("claim", self.text, length=16))


@dataclass(frozen=True)
class ResearchReport:
    title: str
    claims: list[ResearchClaim]
    summary: str = ""
    external_context: str = ""
    confirmations: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    new_concepts: list[str] = field(default_factory=list)
    confidence: float = 0.8
    source_count: int = 0


@dataclass(frozen=True)
class QuoteValidation:
    status: str
    source_url: str
    confidence: float
    quote_start_char: int | None = None
    quote_end_char: int | None = None


@dataclass(frozen=True)
class ReviewIssue:
    severity: str
    code: str
    artifact: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchReviewReport:
    status: str
    ready_for_report: bool
    issues: list[ReviewIssue]
    checked_artifacts: dict[str, str] = field(default_factory=dict)
    hashes: dict[str, str | None] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    citation_counts: dict[str, int] = field(default_factory=dict)
    graph_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        return payload


class ResearchSearchProvider(Protocol):
    def search(self, query: str, limit: int) -> list[SearchResult]:
        ...


class DuckDuckGoSearchProvider:
    def search(self, query: str, limit: int) -> list[SearchResult]:
        try:
            from ddgs import DDGS
            from ddgs.exceptions import DDGSException
        except ImportError as exc:
            raise RuntimeError("ddgs package is required for DuckDuckGo research search") from exc
        try:
            rows = list(DDGS().text(query, max_results=max(1, int(limit))))
        except Exception as exc:
            if type(exc) is DDGSException and str(exc) == "No results found.":
                return []
            raise RuntimeError(f"DuckDuckGo search failed for query {query!r}: {exc}") from exc
        results: list[SearchResult] = []
        for index, row in enumerate(rows, start=1):
            url = str(row.get("href") or row.get("url") or "").strip()
            if not url:
                continue
            results.append(
                SearchResult(
                    query_id=_stable_id("query", query, length=12),
                    title=str(row.get("title") or url),
                    url=url,
                    snippet=str(row.get("body") or row.get("snippet") or ""),
                    rank=index,
                    raw=dict(row),
                )
            )
        return results


class SourceFetcher:
    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        user_agent: str = "yt-sub-hub-research/0.1",
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent
        self.max_content_chars = max(1, int(max_content_chars))
        self.max_content_bytes = self.max_content_chars * 4

    def fetch(self, result: SearchResult, source_id: str) -> FetchedSource:
        safe = validate_fetch_url(result.url)
        if not safe.allowed:
            return _blocked_source(result, source_id, safe.reason)
        try:
            with httpx.Client(
                follow_redirects=False,
                timeout=self.timeout_seconds,
                headers={"User-Agent": self.user_agent},
                trust_env=False,
            ) as client:
                current_url = safe.url
                status_code: int | None = None
                final_url = current_url
                content_type = ""
                body = b""
                byte_truncated = False
                for _ in range(MAX_FETCH_REDIRECTS + 1):
                    with client.stream("GET", current_url) as stream_response:
                        status_code = stream_response.status_code
                        location = stream_response.headers.get("location")
                        if status_code in {301, 302, 303, 307, 308} and location:
                            next_url = urljoin(str(stream_response.url), location)
                            next_safe = validate_fetch_url(next_url)
                            if not next_safe.allowed:
                                return _blocked_source(
                                    result,
                                    source_id,
                                    f"redirect target blocked: {next_safe.reason}",
                                    final_url=next_url,
                                    status_code=status_code,
                                )
                            current_url = next_safe.url
                            continue

                        final_url = str(stream_response.url)
                        final_safe = validate_fetch_url(final_url, resolve_host=False)
                        if not final_safe.allowed:
                            return _blocked_source(
                                result,
                                source_id,
                                f"redirect target blocked: {final_safe.reason}",
                                final_url=final_url,
                                status_code=status_code,
                            )
                        content_type = stream_response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                        if content_type not in SUPPORTED_CONTENT_TYPES:
                            return FetchedSource(
                                source_id=source_id,
                                result_ids=[result.result_id],
                                url=result.url,
                                title=result.title,
                                text="",
                                final_url=final_url,
                                status_code=status_code,
                                content_type=content_type,
                                fetch_error=f"unsupported content-type: {content_type or 'unknown'}",
                            )
                        body, byte_truncated = self._read_limited_body(stream_response)
                        encoding = getattr(stream_response, "encoding", None) or "utf-8"
                        break
                else:
                    return FetchedSource(
                        source_id=source_id,
                        result_ids=[result.result_id],
                        url=result.url,
                        title=result.title,
                        text="",
                        final_url=current_url,
                        status_code=status_code,
                        content_type="",
                        fetch_error="fetch failed: too many redirects",
                    )
        except Exception as exc:
            return FetchedSource(
                source_id=source_id,
                result_ids=[result.result_id],
                url=result.url,
                title=result.title,
                text="",
                status_code=None,
                content_type="",
                fetch_error=f"fetch failed: {exc.__class__.__name__}",
            )
        decoded = body.decode(encoding, errors="replace")
        char_truncated = len(decoded) > self.max_content_chars
        limited_text = decoded[: self.max_content_chars]
        text = clean_source_text(limited_text, content_type=content_type)
        title = extract_title(limited_text) or result.title
        return FetchedSource(
            source_id=source_id,
            result_ids=[result.result_id],
            url=result.url,
            title=title,
            text=text,
            final_url=final_url,
            status_code=status_code,
            content_type=content_type,
            fetch_error=None if 200 <= status_code < 300 and text.strip() else f"http status {status_code}",
            truncated=byte_truncated or char_truncated,
        )

    def _read_limited_body(self, response: httpx.Response) -> tuple[bytes, bool]:
        chunks: list[bytes] = []
        remaining = self.max_content_bytes
        truncated = False
        content_length = _parse_content_length(response.headers.get("content-length"))
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            if remaining <= 0:
                truncated = True
                break
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                truncated = True
                break
            chunks.append(chunk)
            remaining -= len(chunk)
            if remaining == 0:
                truncated = content_length is None or content_length > self.max_content_bytes
                break
        return b"".join(chunks), truncated


def validate_fetch_url(url: str, *, resolve_host: bool = True) -> SafeUrlResult:
    raw = (url or "").strip()
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        return SafeUrlResult(False, "", "blocked unsafe URL scheme")
    hostname = (parts.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        return SafeUrlResult(False, "", "blocked URL without host")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return SafeUrlResult(False, "", "blocked localhost host", hostname)
    if _unsafe_ip_literal(hostname):
        return SafeUrlResult(False, "", "blocked private or reserved IP address", hostname)
    if resolve_host:
        try:
            infos = socket.getaddrinfo(hostname, parts.port or (443 if scheme == "https" else 80), type=socket.SOCK_STREAM)
        except OSError:
            return SafeUrlResult(False, "", "blocked unresolved host", hostname)
        for info in infos:
            address = info[-1][0]
            if _unsafe_ip_literal(address):
                return SafeUrlResult(False, "", "blocked host resolving to private or reserved IP address", hostname)
    # Deduplication keys must not rewrite fetch/redirect targets: servers may
    # require www or a trailing slash and redirect back to that exact URL.
    return SafeUrlResult(True, urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, "")), "", hostname)


def _unsafe_ip_literal(hostname: str) -> bool:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
        or not address.is_global
    )


def _blocked_source(
    result: SearchResult,
    source_id: str,
    reason: str,
    *,
    final_url: str | None = None,
    status_code: int | None = None,
) -> FetchedSource:
    return FetchedSource(
        source_id=source_id,
        result_ids=[result.result_id],
        url=result.url,
        title=result.title,
        text="",
        final_url=final_url or result.url,
        status_code=status_code,
        content_type="",
        fetch_error=f"blocked unsafe URL: {_sanitize_fetch_error(reason)}",
    )


def _sanitize_fetch_error(reason: str) -> str:
    value = (reason or "blocked").strip()
    value = re.sub(r"[^\w\s:.-]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()[:160] or "blocked"


def _parse_content_length(value: str | None) -> int | None:
    try:
        parsed = int(str(value or "").strip())
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def canonicalize_url(url: str) -> str:
    parts = urlsplit((url or "").strip())
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    if scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query_items = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        lower_key = key.lower()
        if lower_key in TRACKING_QUERY_KEYS or any(lower_key.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES):
            continue
        query_items.append((key, value))
    query = urlencode(sorted(query_items))
    return urlunsplit((scheme, netloc, path, query, ""))


def dedupe_search_results(results: list[SearchResult], *, limit: int | None = None) -> list[SearchResult]:
    deduped: list[SearchResult] = []
    seen: set[str] = set()
    for result in sorted(results, key=lambda item: item.rank):
        canonical = canonicalize_url(result.url)
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(
            SearchResult(
                query_id=result.query_id,
                title=result.title,
                url=canonical,
                snippet=result.snippet,
                rank=result.rank,
                provider=result.provider,
                raw=result.raw,
            )
        )
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def validate_quote(quote: str, source: FetchedSource) -> QuoteValidation:
    source_text = source.text or ""
    normalized_quote = normalize_text(quote)
    normalized_source = normalize_text(source_text)
    if not normalized_quote or not normalized_source:
        return QuoteValidation("failed", source.url, 0.0)
    exact_start = normalized_source.find(normalized_quote)
    if exact_start >= 0:
        start, end = _locate_original_span(quote, source_text)
        return QuoteValidation("exact", source.url, 1.0, start, end)
    confidence = _best_quote_ratio(normalized_quote, normalized_source)
    if confidence >= 0.8:
        return QuoteValidation("fuzzy", source.url, confidence)
    return QuoteValidation("failed", source.url, confidence)


def review_research_enrichment(report: ResearchReport, fetched_sources: list[FetchedSource], graph_payload: dict[str, Any] | None = None) -> ResearchReviewReport:
    issues: list[ReviewIssue] = []
    sources_by_url = {source.url: source for source in fetched_sources if source.fetched}
    citation_total = 0
    valid_citations = 0
    seen_claims: set[str] = set()
    for claim in report.claims:
        finding_key = claim.claim_id if claim.finding_type == "unverified_context" else normalize_text(claim.text)
        if finding_key in seen_claims:
            issues.append(_warning("duplicate_finding", "findings.ru.json", f"duplicate finding: {claim.text[:80]}"))
        seen_claims.add(finding_key)
        if not claim.citations:
            issues.append(_hard("missing_citation", "findings.ru.json", "every finding must include at least one citation"))
            continue
        for citation in claim.citations:
            citation_total += 1
            source = sources_by_url.get(citation.url)
            if source is None:
                issues.append(_hard("hallucinated_url", "findings.ru.json", f"citation URL was not fetched: {citation.url}"))
                continue
            validation = validate_quote(citation.quote, source)
            if validation.status == "failed":
                issues.append(_hard("quote_validation_failed", "findings.ru.json", f"quote does not match fetched source text: {citation.url}"))
                continue
            valid_citations += 1
    if graph_payload is not None:
        for rel in graph_payload.get("relationships") or []:
            if not rel.get("research_run_id") or not rel.get("source_video_id") or not rel.get("source_url"):
                issues.append(_hard("missing_graph_provenance", "research_graph.json", "research graph relationships must include provenance"))
    hard = [issue for issue in issues if issue.severity == "hard"]
    status = "failed" if hard else ("warning" if issues else "passed")
    return ResearchReviewReport(
        status=status,
        ready_for_report=not hard,
        issues=issues,
        checked_artifacts={
            "report": "research_report.md",
            "findings": "findings.ru.json",
            "sources": "sources/*.json",
            "graph": "research_graph.json",
        },
        source_counts={
            "fetched": len([source for source in fetched_sources if source.fetched]),
            "total": len(fetched_sources),
        },
        citation_counts={"valid": valid_citations, "total": citation_total},
        graph_counts={
            "nodes": len((graph_payload or {}).get("nodes") or []),
            "relationships": len((graph_payload or {}).get("relationships") or []),
        },
    )


def render_research_report(report: ResearchReport, fetched_sources: list[FetchedSource]) -> str:
    review = review_research_enrichment(report, fetched_sources)
    if review.status == "failed":
        messages = "; ".join(issue.message for issue in review.issues)
        raise ResearchValidationError(f"research report citations are invalid: {messages}")
    sources_by_url = {source.url: source for source in fetched_sources if source.fetched}
    lines = [
        f"# Research Enrichment: {report.title}",
        "",
        f"## {report.title}",
        "",
        "## Краткий вывод",
        "",
        report.summary or f"Проверено внешних источников: {len(sources_by_url)}. Confidence: {report.confidence:.2f}.",
        "",
        "## Внешний контекст",
        "",
        report.external_context or "Внешний контекст построен только на скачанных и проверенных источниках.",
        "",
        "## Выдержки и контекст",
        "",
    ]
    for claim in report.claims:
        refs = []
        quote_lines = []
        for citation in claim.citations:
            source = sources_by_url.get(citation.url)
            if source is None:
                continue
            refs.append(f"[{source.title}]({source.url})")
            quote_lines.append(f"  - Цитата: \"{citation.quote}\"")
        lines.append(f"- {claim.text} ({'; '.join(refs)})")
        lines.extend(quote_lines)
    lines.extend(["", "## Противоречия / уточнения", ""])
    if report.contradictions:
        lines.extend(f"- {item}" for item in report.contradictions)
    else:
        lines.append("- Смысловая поддержка тезисов и противоречия автоматически не оценивались.")
    lines.extend(["", "## Новые понятия", ""])
    if report.new_concepts:
        lines.extend(f"- {item}" for item in report.new_concepts)
    else:
        lines.append("- Новые понятия не добавлены сверх проверенных тезисов.")
    lines.extend(["", "## Источники", ""])
    for index, source in enumerate(sources_by_url.values(), start=1):
        lines.append(f"{index}. [{source.title}]({source.url}) - {source.retrieved_at}, {source.source_id}")
    lines.append("")
    return "\n".join(lines)


def run_video_research(
    store: LibraryStore,
    video_id: str,
    *,
    provider: ResearchSearchProvider | None = None,
    fetcher: SourceFetcher | None = None,
    max_sources: int = 10,
) -> dict[str, Any]:
    research_dir = research_dir_for(store, video_id)
    research_dir.mkdir(parents=True, exist_ok=True)
    (research_dir / "review" / "research_artifact_review.json").unlink(missing_ok=True)
    _write_json(research_dir / "status.json", {"video_id": video_id, "status": ResearchStatus.RUNNING.value, "updated_at": utc_now()})
    try:
        _require_core_ready(store, video_id)
        metadata = store.load_video_metadata(video_id)
        analysis_data = store.load_analysis(video_id, variant="ru")
        if metadata is None or analysis_data is None:
            raise ResearchValidationError("metadata.json and analysis.ru.json are required")
        analysis = parse_analysis_response(analysis_data)
        note_path = store.note_path(video_id)
        plan = _build_research_plan(video_id, note_path, store.artifact_paths(video_id).analysis_ru, analysis_data)
        plan["limits"]["max_sources"] = max_sources
        queries = _build_queries(analysis)
        _write_json(research_dir / "research_plan.json", plan)
        _write_json(research_dir / "queries.json", [asdict(query) for query in queries])

        provider = provider or DuckDuckGoSearchProvider()
        raw_results: list[SearchResult] = []
        per_query_limit = max(1, min(max_sources, 5))
        for query in queries:
            for result in provider.search(query.query, per_query_limit):
                raw_results.append(
                    SearchResult(
                        query_id=query.query_id,
                        title=result.title,
                        url=result.url,
                        snippet=result.snippet,
                        rank=result.rank,
                        provider=result.provider,
                        raw=result.raw,
                    )
                )
        normalized_results = dedupe_search_results(raw_results, limit=max_sources)
        if not normalized_results:
            raise RuntimeError("research search returned no results")
        _write_json(research_dir / "search_results.raw.json", [_asdict(result) for result in raw_results])
        _write_json(research_dir / "search_results.normalized.json", [_asdict(result) for result in normalized_results])

        fetcher = fetcher or SourceFetcher()
        fetched_sources: list[FetchedSource] = []
        sources_dir = research_dir / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        for index, result in enumerate(normalized_results, start=1):
            source_id = f"source-{index:03d}"
            source = fetcher.fetch(result, source_id)
            text_name = f"{source.source_id}.txt"
            metadata_name = f"{source.source_id}.json"
            if source.text:
                (sources_dir / text_name).write_text(source.text, encoding="utf-8")
            source = FetchedSource(
                **{
                    **_asdict(source),
                    "text_path": str(Path("sources") / text_name) if source.text else "",
                    "metadata_path": str(Path("sources") / metadata_name),
                }
            )
            _write_json(sources_dir / metadata_name, _source_metadata(source))
            fetched_sources.append(source)
        usable_sources = [source for source in fetched_sources if source.fetched]
        if not usable_sources:
            raise RuntimeError("research source fetch produced no usable sources")

        source_summaries = _build_source_summaries(usable_sources, queries, normalized_results)
        _write_json(research_dir / "source_summaries.json", source_summaries)
        report = _build_report(metadata.title, analysis_data, usable_sources)
        graph = _build_research_graph(video_id, plan["research_run_id"], report, usable_sources)
        review = review_research_enrichment(report, fetched_sources, graph)
        if review.status == "failed":
            raise ResearchValidationError("research artifact review failed: " + "; ".join(issue.message for issue in review.issues))
        markdown = render_research_report(report, usable_sources)
        _write_json(research_dir / "findings.ru.json", [_claim_payload(claim, usable_sources) for claim in report.claims])
        (research_dir / "research_report.md").write_text(markdown, encoding="utf-8")
        _write_json(research_dir / "research_graph.json", graph)
        review_dir = research_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_payload = {**review.to_dict(), "research_run_id": plan["research_run_id"],
                          "scope": "source_and_quote_validation", "claim_support": "not_assessed",
                          "artifact_hashes": _research_artifact_hashes(research_dir)}
        _write_json(review_dir / "research_artifact_review.json", review_payload)
        status = ResearchStatus.COMPLETED.value if review.status == "passed" else ResearchStatus.COMPLETED_WITH_WARNING.value
        status_payload = {"video_id": video_id, "status": status, "review_status": review.status, "research_run_id": plan["research_run_id"], "updated_at": utc_now()}
        _write_json(research_dir / "status.json", status_payload)
        return build_research_payload(store, video_id)
    except Exception as exc:
        status_payload = {"video_id": video_id, "status": ResearchStatus.FAILED.value, "error": str(exc), "updated_at": utc_now()}
        _write_json(research_dir / "status.json", status_payload)
        raise


def build_research_payload(store: LibraryStore, video_id: str) -> dict[str, Any]:
    root = research_dir_for(store, video_id)
    status_path = root / "status.json"
    status = _read_json(status_path) if status_path.exists() else {"video_id": video_id, "status": ResearchStatus.NOT_STARTED.value}
    return {
        "video_id": video_id,
        "status": status,
        "artifacts": {
            "research_report": _artifact_info(root / "research_report.md", store.root),
            "findings": _artifact_info(root / "findings.ru.json", store.root),
            "research_graph": _artifact_info(root / "research_graph.json", store.root),
            "research_review": _artifact_info(root / "review" / "research_artifact_review.json", store.root),
            "sources": _artifact_info(root / "sources", store.root),
        },
    }


def load_research_report_markdown(store: LibraryStore, video_id: str) -> str:
    path = research_dir_for(store, video_id) / "research_report.md"
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def load_research_review(store: LibraryStore, video_id: str) -> dict[str, Any]:
    path = research_dir_for(store, video_id) / "review" / "research_artifact_review.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_json(path)


def sync_research_graph_to_neo4j(store: LibraryStore, video_id: str, *, settings: Any) -> dict[str, Any]:
    review = load_research_review(store, video_id)
    if review.get("status") != "passed":
        raise ResearchValidationError("neo4j research sync requires research_artifact_review status passed")
    require_current_research_run(store, video_id)
    path = research_dir_for(store, video_id) / "research_graph.json"
    if not path.exists():
        raise FileNotFoundError(path)
    graph_payload = _read_json(path)
    _validate_research_graph_for_sync(graph_payload, video_id)
    payload = GraphPayload(nodes=graph_payload.get("nodes") or [], relationships=graph_payload.get("relationships") or [])
    graph_store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    try:
        graph_store.delete_research_for_video(video_id)
        graph_store.upsert_payload(payload)
        consistency = graph_store.research_consistency(payload)
    finally:
        graph_store.close()
    result = {
        "video_id": video_id,
        "checked": bool(consistency.get("checked")),
        "missing": consistency.get("missing"),
        "extra": consistency.get("extra"),
        "node_count": len(payload.nodes),
        "relationship_count": len(payload.relationships),
    }
    if result["checked"] and result["missing"] == 0 and result["extra"] == 0:
        _record_research_sync_status(store, video_id, result)
    return result


def build_channel_research_summary(store: LibraryStore, channel_id: str) -> dict[str, Any]:
    channel_dir = store.root / "channels" / channel_id
    videos_path = channel_dir / "videos.json"
    if not videos_path.exists():
        raise FileNotFoundError(videos_path)
    channel_data = _read_json(videos_path)
    findings: list[dict[str, Any]] = []
    for video in channel_data.get("videos") or []:
        if not isinstance(video, dict):
            continue
        video_id = str(video.get("video_id") or "")
        if not video_id:
            continue
        findings_path = research_dir_for(store, video_id) / "findings.ru.json"
        if not findings_path.exists():
            continue
        for finding in json.loads(findings_path.read_text(encoding="utf-8")):
            if isinstance(finding, dict) and finding.get("accepted") is True:
                findings.append({"video_id": video_id, **finding})
    research_dir = channel_dir / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for finding in findings:
        key = normalize_text(str(finding.get("claim_ru") or finding.get("finding_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    _write_json(research_dir / "cross_video_findings.json", deduped)
    lines = [
        f"# Channel Research Enrichment: {channel_id}",
        "",
        "## Краткий вывод",
        "",
        f"Найдено принятых research findings: {len(deduped)}.",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {item.get('claim_ru', item.get('finding_id'))} ({item.get('video_id')})" for item in deduped)
    if not deduped:
        lines.append("- Принятые video-level research findings пока отсутствуют.")
    (research_dir / "channel_research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(
        research_dir / "channel_research_plan.json",
        {
            "schema_version": SCHEMA_VERSION,
            "channel_id": channel_id,
            "video_count": len(channel_data.get("videos") or []),
            "source": "accepted_video_research_findings",
        },
    )
    return {
        "channel_id": channel_id,
        "status": ResearchStatus.COMPLETED.value,
        "finding_count": len(deduped),
        "report_path": str(research_dir / "channel_research_report.md"),
    }


def research_dir_for(store: LibraryStore, video_id: str) -> Path:
    return store.video_dir(video_id) / "research"


def _validate_research_graph_for_sync(graph_payload: dict[str, Any], video_id: str) -> None:
    if str(graph_payload.get("video_id") or "") != video_id:
        raise ResearchValidationError("research graph video_id does not match requested video")
    if not str(graph_payload.get("research_run_id") or ""):
        raise ResearchValidationError("research graph requires research_run_id")
    allowed_quote_validation = {"exact", "fuzzy"}
    required = {"research_run_id", "source_video_id", "source_url", "provider", "evidence_quote", "quote_validation"}
    relationships = graph_payload.get("relationships") or []
    if not relationships:
        raise ResearchValidationError("research graph requires at least one research relationship")
    for index, relationship in enumerate(relationships, start=1):
        missing = [key for key in sorted(required) if not relationship.get(key)]
        if missing:
            raise ResearchValidationError(f"research graph relationship {index} missing provenance: {', '.join(missing)}")
        if relationship.get("source_video_id") != video_id:
            raise ResearchValidationError(f"research graph relationship {index} source_video_id mismatch")
        if relationship.get("quote_validation") not in allowed_quote_validation:
            raise ResearchValidationError(f"research graph relationship {index} quote_validation must be exact or fuzzy")


def _record_research_sync_status(store: LibraryStore, video_id: str, result: dict[str, Any]) -> None:
    root = research_dir_for(store, video_id)
    path = root / "status.json"
    status = _read_json(path) if path.exists() else {"video_id": video_id, "status": ResearchStatus.COMPLETED.value}
    status["research_sync"] = {
        **(status.get("research_sync") if isinstance(status.get("research_sync"), dict) else {}),
        "neo4j": {
            "checked": result.get("checked"),
            "missing": result.get("missing"),
            "extra": result.get("extra"),
            "node_count": result.get("node_count"),
            "relationship_count": result.get("relationship_count"),
            "synced_at": utc_now(),
        },
    }
    status["updated_at"] = utc_now()
    _write_json(path, status)


def clean_source_text(raw: str, *, content_type: str) -> str:
    text = raw
    if content_type == "text/html":
        text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", text)
        text = re.sub(r"(?is)<br\s*/?>", "\n", text)
        text = re.sub(r"(?is)</p\s*>", "\n", text)
        text = re.sub(r"(?is)<.*?>", " ", text)
        text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_title(raw_html: str) -> str | None:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_html or "")
    if not match:
        return None
    return clean_source_text(match.group(1), content_type="text/plain") or None


def normalize_text(value: str) -> str:
    value = (value or "").casefold()
    value = re.sub(r"[^\w\s]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def text_hash(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _build_research_plan(video_id: str, note_path: Path, analysis_path: Path, analysis_data: dict[str, Any]) -> dict[str, Any]:
    concepts = [str(item.get("name") or "") for item in analysis_data.get("concepts") or [] if isinstance(item, dict)]
    claims = [str(item.get("text") or "") for item in analysis_data.get("claims") or [] if isinstance(item, dict)]
    return {
        "schema_version": SCHEMA_VERSION,
        "research_run_id": _stable_id("research-run", video_id, time.time(), length=16),
        "video_id": video_id,
        "source_note_hash": text_hash(note_path.read_text(encoding="utf-8")) if note_path.exists() else None,
        "source_analysis_hash": text_hash(json.dumps(analysis_data, ensure_ascii=False, sort_keys=True)),
        "language": "ru",
        "goals": ["Проверить ключевые тезисы по внешним источникам"],
        "concepts": concepts[:12],
        "claims_to_check": claims[:8],
        "questions": [f"Какие источники подтверждают: {claim}" for claim in claims[:3]],
        "limits": {"max_sources": 10, "max_chunks_per_source": 5, "max_chars_per_chunk": 1800},
    }


def _build_queries(analysis: Any) -> list[ResearchQuery]:
    topic = " ".join(analysis.tags[:2]) or analysis.cleaned_title
    values = [claim.text for claim in analysis.claims[:3]]
    values.extend(f"{topic} {concept.name}".strip() for concept in analysis.concepts[:3])
    values.extend(analysis.open_questions[:2])
    queries: list[ResearchQuery] = []
    for index, value in enumerate(values, start=1):
        if not str(value).strip():
            continue
        queries.append(
            ResearchQuery(
                query_id=f"rq-{index:03d}",
                query=str(value).strip(),
                intent="verify_or_contextualize",
                priority=index,
            )
        )
    if not queries:
        queries.append(ResearchQuery(query_id="rq-001", query=analysis.cleaned_title, intent="context", priority=1))
    return queries


def _build_source_summaries(
    sources: list[FetchedSource], queries: list[ResearchQuery], results: list[SearchResult] | None = None,
) -> list[dict[str, Any]]:
    query_ids = {query.query_id for query in queries}
    results_by_id = {result.result_id: result for result in results or []}
    return [
        {
            "source_id": source.source_id,
            "relevant_query_ids": sorted({results_by_id[result_id].query_id for result_id in source.result_ids
                                          if result_id in results_by_id and results_by_id[result_id].query_id in query_ids}),
            "summary_ru": None,
            "summary_status": "not_generated",
            "excerpt_original": _first_sentence(source.text) or source.title,
            "language": "unknown",
            "key_points": [],
            "candidate_quotes": [_first_quote(source.text)],
            "chunk_refs": [{"chunk_id": f"{source.source_id}:000", "start_char": 0, "end_char": min(len(source.text), 500)}],
            "confidence": None,
        }
        for source in sources
    ]


def _build_report(title: str, analysis_data: dict[str, Any], sources: list[FetchedSource]) -> ResearchReport:
    # Matching a quote proves its origin, not that it supports a video claim.
    # The deterministic MVP provides source excerpts for human comparison only.
    claims: list[ResearchClaim] = []
    topic = " ".join(str(item.get("name") or "") for item in analysis_data.get("concepts") or [] if isinstance(item, dict))
    for source in sources:
        quote = _best_quote_for_claim(topic, source.text)
        validation = validate_quote(quote, source)
        claims.append(ResearchClaim(
            claim_id=_stable_id("source-context", source.url, length=16),
            text=f"Источник «{source.title}»: выдержка для сопоставления с видео; поддержка тезисов не оценивалась.",
            finding_type="unverified_context",
            citations=[Citation(source_id=source.source_id, url=source.url, quote=quote,
                                quote_start_char=validation.quote_start_char, quote_end_char=validation.quote_end_char,
                                quote_validation=validation.status)],
            confidence=0.0,
            accepted=False,
        ))
    return ResearchReport(
        title=title, claims=claims,
        summary=f"Собраны выдержки из {len(sources)} скачанных источников. Проверено происхождение цитат.",
        external_context="Это поисковый контекст для ручной проверки. Он не подтверждает и не опровергает тезисы видео.",
        confidence=0.0, source_count=len(sources),
    )


def _build_research_graph(video_id: str, research_run_id: str, report: ResearchReport, sources: list[FetchedSource]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    relationships: list[dict[str, Any]] = []
    video_node = f"video:{video_id}"
    nodes[video_node] = {"id": video_node, "label": "Video", "video_id": video_id}
    sources_by_url = {source.url: source for source in sources}
    for claim in report.claims:
        finding_id = f"research_finding:{research_run_id}:{claim.claim_id}"
        nodes[finding_id] = {
            "id": finding_id,
            "label": "ResearchFinding",
            "source_video_id": video_id,
            "research_run_id": research_run_id,
            "text": claim.text,
            "accepted": claim.accepted,
            "claim_support": "not_assessed" if not claim.accepted else "provided",
            "confidence": claim.confidence,
            "language": "ru",
        }
        first_source = sources_by_url.get(claim.citations[0].url) if claim.citations else None
        first_citation = claim.citations[0] if claim.citations else None
        if first_source is not None:
            relationships.append(
                _research_rel(video_node, finding_id, "RELATED_EXTERNAL", video_id, research_run_id, first_source, first_citation, claim.confidence)
            )
        for citation in claim.citations:
            source = sources_by_url.get(citation.url)
            if source is None:
                continue
            source_node = f"external_source:{video_id}:{research_run_id}:{source.source_id}"
            citation_node = f"citation:{video_id}:{research_run_id}:{citation.citation_id}"
            nodes[source_node] = {
                "id": source_node,
                "label": "ExternalSource",
                "source_video_id": video_id,
                "source_url": source.url,
                "title": source.title,
                "retrieved_at": source.retrieved_at,
                "provider": "duckduckgo",
                "research_run_id": research_run_id,
                "language": "ru",
            }
            nodes[citation_node] = {
                "id": citation_node,
                "label": "Citation",
                "source_video_id": video_id,
                "source_url": source.url,
                "evidence_quote": citation.quote,
                "quote_validation": citation.quote_validation or validate_quote(citation.quote, source).status,
                "research_run_id": research_run_id,
                "language": "ru",
            }
            relationships.append(_research_rel(finding_id, citation_node, "CITES", video_id, research_run_id, source, citation, claim.confidence))
            relationships.append(_research_rel(citation_node, source_node, "SOURCE_FOR", video_id, research_run_id, source, citation, claim.confidence))
    return {
        "schema_version": SCHEMA_VERSION,
        "research_run_id": research_run_id,
        "video_id": video_id,
        "nodes": list(nodes.values()),
        "relationships": relationships,
        "provenance": {"source": "research_enrichment", "provider": "duckduckgo"},
        "quality": {"source_count": len(sources), "finding_count": len(report.claims)},
    }


def _research_rel(
    from_id: str,
    to_id: str,
    rel_type: str,
    video_id: str,
    research_run_id: str,
    source: FetchedSource | None = None,
    citation: Citation | None = None,
    confidence: float = 0.8,
) -> dict[str, Any]:
    source_url = source.url if source is not None else ""
    evidence_quote = citation.quote if citation is not None else ""
    quote_validation = citation.quote_validation if citation is not None else ""
    return {
        "id": _stable_id("research-rel", from_id, rel_type, to_id, research_run_id, length=20),
        "from": from_id,
        "to": to_id,
        "type": rel_type,
        "source_video_id": video_id,
        "source_url": source_url,
        "retrieved_at": source.retrieved_at if source is not None else utc_now(),
        "provider": "duckduckgo",
        "research_run_id": research_run_id,
        "confidence": confidence,
        "evidence_quote": evidence_quote,
        "quote_validation": quote_validation,
        "language": "ru",
    }


RESEARCH_REVIEWED_ARTIFACTS = ("research_plan.json", "findings.ru.json", "source_summaries.json", "research_graph.json", "research_report.md")


def _research_artifact_hashes(root: Path) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RESEARCH_REVIEWED_ARTIFACTS}


def require_current_research_run(store: LibraryStore, video_id: str) -> str:
    root = research_dir_for(store, video_id)
    try:
        status = _read_json(root / "status.json")
        plan = _read_json(root / "research_plan.json")
        review = _read_json(root / "review" / "research_artifact_review.json")
    except (OSError, ValueError) as exc:
        raise ResearchValidationError("research requires a completed, reviewed run; rerun research-video") from exc
    if status.get("status") not in {"completed", "completed_with_warning"}:
        raise ResearchValidationError("research requires a completed run; rerun research-video")
    run_id = plan.get("research_run_id")
    if not run_id or status.get("research_run_id") != run_id or review.get("research_run_id") != run_id:
        raise ResearchValidationError("research run IDs do not match; rerun research-video")
    if review.get("status") != "passed":
        raise ResearchValidationError("research_artifact_review status passed is required")
    try:
        current_note_hash = text_hash(store.note_path(video_id).read_text(encoding="utf-8"))
        current_analysis_hash = text_hash(json.dumps(_read_json(store.artifact_paths(video_id).analysis_ru), ensure_ascii=False, sort_keys=True))
    except (OSError, ValueError) as exc:
        raise ResearchValidationError("research core artifacts are missing; rerun core and research-video") from exc
    if plan.get("source_note_hash") != current_note_hash or plan.get("source_analysis_hash") != current_analysis_hash:
        raise ResearchValidationError("research core artifacts changed; rerun research-video")
    try:
        current_hashes = _research_artifact_hashes(root)
    except OSError as exc:
        raise ResearchValidationError("research artifacts are missing; rerun research-video") from exc
    if review.get("artifact_hashes") != current_hashes:
        raise ResearchValidationError("research artifacts changed since review; rerun research-video")
    return str(run_id)


def _require_core_ready(store: LibraryStore, video_id: str) -> None:
    paths = store.artifact_paths(video_id)
    missing = [path.name for path in [paths.note, paths.analysis_ru, paths.ru_artifact_review] if not path.exists()]
    if missing:
        raise ResearchValidationError("research requires ready RU artifacts: " + ", ".join(missing))
    review = _read_json(paths.ru_artifact_review)
    if review.get("status") not in {"passed", "warning"}:
        raise ResearchValidationError("research requires ru_artifact_review passed or warning")


def _source_metadata(source: FetchedSource) -> dict[str, Any]:
    payload = _asdict(source)
    payload.pop("text", None)
    return payload


def _claim_payload(claim: ResearchClaim, sources: list[FetchedSource]) -> dict[str, Any]:
    sources_by_url = {source.url: source for source in sources}
    citations = []
    for citation in claim.citations:
        source = sources_by_url.get(citation.url)
        validation = validate_quote(citation.quote, source) if source is not None else QuoteValidation("failed", citation.url, 0.0)
        citations.append(
            {
                **_asdict(citation),
                "source_id": citation.source_id or (source.source_id if source is not None else ""),
                "quote_start_char": citation.quote_start_char if citation.quote_start_char is not None else validation.quote_start_char,
                "quote_end_char": citation.quote_end_char if citation.quote_end_char is not None else validation.quote_end_char,
                "quote_validation": citation.quote_validation or validation.status,
            }
        )
    return {
        "finding_id": claim.claim_id,
        "type": claim.finding_type,
        "claim_ru": claim.text,
        "source_ids": [item["source_id"] for item in citations if item["source_id"]],
        "citations": citations,
        "confidence": claim.confidence,
        "accepted": claim.accepted,
        "claim_support": "not_assessed" if not claim.accepted else "provided",
    }


def _artifact_info(path: Path, root: Path) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    try:
        relative = str(path.resolve(strict=False).relative_to(root.resolve(strict=False)))
    except ValueError:
        relative = str(path)
    return {
        "path": relative,
        "exists": exists,
        "size_bytes": stat.st_size if stat is not None and path.is_file() else None,
        "updated_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat().replace("+00:00", "Z") if stat is not None else None,
    }


def _first_sentence(text: str) -> str:
    match = re.search(r"(.{20,280}?[.!?])(?:\s|$)", text or "")
    return (match.group(1).strip() if match else (text or "")[:240].strip())


def _first_quote(text: str) -> str:
    words = (text or "").split()
    return " ".join(words[: min(len(words), 18)])


def _best_quote_for_claim(claim: str, source_text: str) -> str:
    source = source_text or ""
    claim_words = [word for word in normalize_text(claim).split() if len(word) > 2]
    sentences = re.split(r"(?<=[.!?])\s+", source)
    best = ""
    best_score = -1
    for sentence in sentences:
        normalized = normalize_text(sentence)
        score = sum(1 for word in claim_words if word in normalized)
        if score > best_score and 20 <= len(sentence.strip()) <= 500:
            best = sentence.strip()
            best_score = score
    return best[:500] if best else _first_quote(source)


def _best_quote_ratio(quote: str, source: str) -> float:
    quote_words = quote.split()
    if not quote_words:
        return 0.0
    window = max(1, len(quote_words))
    source_words = source.split()
    if not source_words:
        return 0.0
    best = 0.0
    for index in range(0, max(1, len(source_words) - window + 1)):
        candidate = " ".join(source_words[index : index + window])
        best = max(best, SequenceMatcher(None, quote, candidate).ratio())
    if len(source_words) < window:
        best = max(best, SequenceMatcher(None, quote, source).ratio())
    return best


def _locate_original_span(quote: str, source_text: str) -> tuple[int | None, int | None]:
    compact_quote = normalize_text(quote)
    if not compact_quote:
        return None, None
    for start in range(len(source_text)):
        if normalize_text(source_text[start : start + len(quote) + 64]).startswith(compact_quote):
            end = min(len(source_text), start + len(quote))
            return start, end
    lower_start = source_text.casefold().find(quote.casefold())
    if lower_start >= 0:
        return lower_start, lower_start + len(quote)
    return None, None


def _hard(code: str, artifact: str, message: str) -> ReviewIssue:
    return ReviewIssue("hard", code, artifact, message)


def _warning(code: str, artifact: str, message: str) -> ReviewIssue:
    return ReviewIssue("warning", code, artifact, message)


def _stable_id(*parts: Any, length: int = 24) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def _asdict(value: Any) -> dict[str, Any]:
    return asdict(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
