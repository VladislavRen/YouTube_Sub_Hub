---
title: Research Enrichment Layer for YouTube Sub Hub
status: accepted-for-mvp
date: 2026-05-15
tags: [architecture, research_enrichment, neo4j, quality_gate]
---

# Research Enrichment Layer for YouTube Sub Hub

## Decision Summary

Add `research_enrichment` as an optional post-core layer. It runs only after a video already has valid RU artifacts (`note.md`, `analysis.ru.json`, `review/ru_artifact_review.json`) and stores its own Markdown, JSON, review, and graph artifacts under `library/videos/<video_id>/research/`.

The layer must not downgrade core `product_quality`. If research search, fetch, synthesis, review, or Neo4j research sync fails, the core video can remain `product_quality=ready`.

## Problem

YouTube Sub Hub captures what a video says, creates a RU note, extracts a deterministic graph payload, and syncs Neo4j. It does not yet validate video claims against external web sources or add cited external context. The risk is that unchecked web findings can pollute the trusted core graph or make hallucinated URLs/quotes look authoritative.

## Goals

- Produce Russian research reports with citations.
- Preserve raw and normalized search/fetch artifacts.
- Validate every URL and quote before report/graph use.
- Keep research quality separate from core product quality.
- Add Neo4j research nodes only with explicit provenance.
- Support small context windows through chunking and compact source summaries.

## Non-Goals

- Do not embed Researcher-X as a Streamlit app.
- Do not require MongoDB, Gemini, Groq, LangGraph, or Streamlit.
- Do not silently invent URLs when search fails.
- Do not cite unfetched or unsupported sources.
- Do not run channel-level enrichment in the video MVP.

## Why Not Embed Researcher-X Directly

Researcher-X is a useful reference for the workflow shape: query planning -> search -> load documents -> synthesize report. It is not a production dependency for YouTube Sub Hub because its app shape and contracts do not match this repo:

- Streamlit UI rather than artifact-first CLI/API service.
- External API keys and providers such as Groq/Gemini/Google Serper.
- MongoDB chat/session memory.
- Broad dependency stack.
- No YouTube Sub Hub artifact, review, Neo4j, or RU quality contracts.

Reference sources:

- [Researcher-X repository](https://github.com/M-Abdelmegeed/Researcher-X)
- [Researcher-X README](https://github.com/M-Abdelmegeed/Researcher-X/blob/main/README.md)
- [Researcher-X main workflow](https://github.com/M-Abdelmegeed/Researcher-X/blob/main/main.py)

## Recommended Architecture

Research stages:

```text
quality_gate
research_query_plan
research_search
research_source_fetch
research_source_review
research_synthesis
research_report_render
research_graph_extract
research_artifact_review
neo4j_research_sync
```

MVP implements this as an explicit manual command/API flow. Runtime queue integration can later map the same stage names into `stage_plans`, but research status remains separate:

```text
not_started
running
completed
completed_with_warning
failed
```

## Artifact Schema

Video-level layout:

```text
library/videos/<video_id>/research/
  research_plan.json
  queries.json
  search_results.raw.json
  search_results.normalized.json
  sources/
    <source_id>.txt
    <source_id>.json
  source_summaries.json
  findings.ru.json
  research_report.md
  research_graph.json
  review/research_artifact_review.json
  status.json
```

Channel-level layout for later aggregation:

```text
library/channels/<channel_id>/research/
  channel_research_plan.json
  channel_research_report.md
  cross_video_findings.json
```

Core MVP contracts:

- `ResearchPlan`: `schema_version`, `research_run_id`, `video_id`, source hashes, goals, concepts, claims, questions, limits.
- `ResearchQuery`: query text, intent, source claim ids, priority, language.
- `SearchResult`: provider, rank, title, URL, canonical URL, snippet, raw payload.
- `FetchedSource`: source id, result ids, URL, final URL, status code, content type, retrieval time, hash, title, text path, metadata path, fetch error.
- `SourceSummary`: compact RU summary, key points, candidate quotes, chunk refs.
- `ResearchFinding`: finding type, RU claim, source ids, citations, confidence, accepted flag.
- `ResearchCitation`: source id, URL, quote, character offsets, quote validation status.
- `ResearchGraphPayload`: nodes, relationships, provenance, quality.
- `ResearchReviewReport`: status, issues, checked artifacts, hashes, source/citation/graph counts.

## Search Provider

Use a narrow provider contract:

```python
class ResearchSearchProvider:
    def search(query: str, limit: int) -> list[SearchResult]: ...
```

MVP provider is `DuckDuckGoSearchProvider`. It should use `ddgs` when installed. Results are capped to 10-15 normalized sources per video, deduped by canonical URL, and stored as both raw and normalized artifacts. If DuckDuckGo rate limits or fails, the stage fails or retries; it must not fall back to hallucinated source lists.

## Anti-Hallucination Strategy

- The LLM cannot create URLs.
- Every URL in report, findings, and graph must come from normalized search results.
- Every cited URL must be fetched successfully before use.
- Fetch records final URL after redirects, status code, content type, retrieval time, content hash, and extracted title.
- Unsupported or failed sources cannot be cited.
- Every quote must validate against saved source text with `exact` or `fuzzy`.
- `quote_validation=failed` is rejected from report and graph and blocks review.

## Safe URL Fetch Policy

Research fetch must treat search results as untrusted input:

- Allow only `http` and `https`.
- Reject missing hosts, `localhost`, and `*.localhost`.
- Reject loopback, private, link-local, multicast, unspecified, reserved, and non-global IPv4/IPv6 addresses, including `127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.0.0/16`, `::1`, unique-local, and link-local IPv6 ranges.
- Resolve hostnames before fetch and reject the URL if any resolved address is unsafe.
- Strip fragments during canonicalization.
- Use `trust_env=False` so local proxy/cookie environment does not affect fetch behavior.
- Follow redirects manually and validate each `Location` before the next request; a public URL redirecting to private network is rejected before fetching the private target.
- Cap fetched text before cleaning to bound memory use.
- Support only `text/html` and `text/plain` in MVP.

Blocked fetches produce sanitized `fetch_error` values and cannot become usable or citable sources.

## Context-Window Strategy

Do not send full web pages to the model. The intended pipeline is:

1. Fetch source.
2. Clean text.
3. Chunk source into bounded windows.
4. Score chunks against research queries and source video concepts.
5. Summarize each source from selected chunks.
6. Synthesize final report only from original video concepts/claims, compact source summaries, and validated quotes.

MVP limits:

- max sources: configurable, default 10.
- max chunks per source: configurable later.
- max chars per chunk: configurable later.
- final synthesis receives summaries and quotes, not raw full pages.

## Research Report

Report is Markdown in Russian:

```md
# Research Enrichment: <video title>

## Краткий вывод
## Внешний контекст
## Подтверждения
## Противоречия / уточнения
## Новые понятия
## Источники
```

Every claim must have citations. The source list must match fetched sources. Confidence and source count must be visible.

## Graph Schema

Research graph is separate from the core deterministic graph. It may use:

- `ExternalSource`
- `ResearchFinding`
- `ResearchQuery`
- `Citation`
- `ResearchTopic`

Relationships:

- `FOUND_BY_QUERY`
- `CITES`
- `SUPPORTS`
- `CONTRADICTS`
- `EXPANDS`
- `RELATED_EXTERNAL`
- `SOURCE_FOR`
- `MENTIONS_EXTERNAL`

Required provenance properties on research relationships include `source_video_id`, `source_url`, `retrieved_at`, `provider`, `research_run_id`, `confidence`, `evidence_quote`, `quote_validation`, and `language`.

## Neo4j Research Sync Contract

Neo4j sync is an explicit user-facing step, not part of the default `research-video` run.

- `review/research_artifact_review.json` must be `passed`.
- `research_graph.json` must exist and must match the requested `video_id`.
- Every research relationship must include `research_run_id`, `source_video_id`, `source_url`, `provider`, `evidence_quote`, and `quote_validation` of `exact` or `fuzzy`.
- Before inserting the current graph, the store deletes previous research relationships and research-owned nodes for the same `source_video_id`; core `Video` and core deterministic graph nodes are preserved.
- After upsert, sync runs consistency against the current `research_run_id`.
- Sync succeeds only when `checked=true`, `missing=0`, and `extra=0`.

Successful sync records `research_sync.neo4j` in `status.json` with consistency counts and `synced_at`.

## Review Gate

`research_artifact_review` checks:

- all findings have valid `source_id`;
- all source ids exist;
- all URLs came from search/fetch layers;
- cited URLs were fetched;
- quotes validate;
- RU report has no unsupported English leakage;
- no duplicate sources/findings;
- graph payload has provenance;
- no unsupported claims without citation;
- report source list matches structured sources.

Status values: `passed`, `warning`, `failed`. MVP Neo4j sync requires `passed`; warning tolerance can be revisited after warning classes are explicit.

## Product/API/CLI

CLI:

```bash
yt-sub-hub research-video <video_id>
yt-sub-hub research-channel <channel_id>
yt-sub-hub review-research <video_id>
yt-sub-hub research-status <video_id> --json
yt-sub-hub research-sync-neo4j <video_id> --json
```

API:

```text
POST /videos/{video_id}/research/run
GET  /videos/{video_id}/research
GET  /videos/{video_id}/research/report
GET  /videos/{video_id}/research/review
POST /videos/{video_id}/research/retry
POST /videos/{video_id}/research/sync-neo4j
```

Artifact browser should list research report, findings, review, and graph artifacts alongside core artifacts.

## MVP Scope

MVP is manual video-level enrichment:

- derive queries from `analysis.ru.json`;
- use DuckDuckGo provider abstraction;
- fetch text/html and text/plain sources;
- create structured artifacts;
- validate quotes and URLs;
- render RU Markdown;
- create `research_graph.json`;
- run explicit Neo4j research sync after passed review;
- expose CLI/API/artifact browser surfaces.

Automatic staged queue integration, retries, locks, and cross-video channel research are later phases.

## Phased Implementation Plan

Phase 0 - Discovery/ADR:

- record this decision;
- verify `ddgs` package behavior;
- keep Researcher-X as reference only.

Phase 1 - Artifacts and provider contracts:

- add schemas/dataclasses;
- add DuckDuckGo provider;
- add fetcher, URL validation, quote validation.

Phase 2 - Video research MVP:

- build query planner;
- summarize fetched sources;
- generate report, findings, status, CLI.

Phase 3 - Review and graph:

- implement research review;
- create research graph payload;
- add isolated Neo4j research sync support;
- expose API/browser artifacts.

Phase 4 - Runtime integration:

- add staged queue integration, retry/recovery, locks, progress.

Phase 5 - Channel-level enrichment:

- aggregate accepted video findings;
- dedupe sources/findings;
- generate channel report.

## Acceptance Criteria

- A ready video can run `yt-sub-hub research-video <video_id>`.
- Search returns normalized results.
- Fetch succeeds for usable text sources.
- `research_report.md` is Russian.
- Citation URLs exist in normalized/fetched sources.
- Quotes validate against fetched source text.
- `research_artifact_review.json` is `passed`.
- `research_graph.json` exists.
- `yt-sub-hub research-sync-neo4j <video_id> --json` succeeds.
- Neo4j research sync returns `checked=true`, `missing=0`, `extra=0`.
- Research failure does not downgrade core `product_quality`.

## Known Follow-ups

- Improve chunk ranking and source summarization for large pages.
- Add runtime staged queue integration with retry/recovery/locks.
- Define domain allow/block policy and source quality preferences.
- Decide if PDF extraction is needed for scientific usefulness.

## Open Questions

- Should research auto-run after every ready video or stay opt-in?
- Should PDF extraction enter the MVP?
- Which source domains should be preferred or blocked?
- Which warning types, if any, are allowed for research sync after MVP?
