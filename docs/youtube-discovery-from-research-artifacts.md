---
title: YouTube Discovery From Research Artifacts
status: accepted-for-mvp
date: 2026-05-21
tags: [architecture, youtube_discovery, research_enrichment, neo4j, quality_gate]
---

# YouTube Discovery From Research Artifacts

## 1. Problem Statement

YouTube Sub Hub can process videos/channels, create RU transcript, RU analysis, `note.md`, `graph_extraction.json`, review RU artifacts, sync the core graph to Neo4j, and run Research Enrichment over `note.md` / `analysis.ru.json`.

The missing product layer is discovery: after research enrichment, use the saved descriptions, findings, source summaries, and graph context to find additional relevant YouTube videos and channels. The layer must expand the knowledge graph and create a controlled `discovery -> review -> enqueue` path without turning the system into an uncontrolled crawler.

Non-goals:

- No automatic processing of discovered videos.
- No recursive crawl loop through channels or related videos.
- No `yt-dlp`, browser cookies, or generated YouTube URLs in discovery.
- No downgrade of core `product_quality` when discovery fails.

## 2. Current State

Inputs already available per video:

- `analysis.ru.json`: RU concepts, claims, topics, tags.
- `note.md`: human-readable RU note.
- `research/research_report.md`: cited research report.
- `research/findings.ru.json`: accepted findings with citations and source ids.
- `research/source_summaries.json`: compact source summaries and key points.
- `research/queries.json`: existing web research queries.
- `research/search_results.normalized.json`: normalized external web search results.
- `research/research_graph.json`: research nodes/relationships for Neo4j.
- Neo4j research graph nodes and relationships after explicit research sync.

Research is an optional post-core layer and stores artifacts under `library/videos/<video_id>/research/`. Discovery follows the same artifact-first style but writes under `library/videos/<video_id>/youtube_discovery/`.

## 3. Target Architecture

Module: `yt_sub_hub.youtube_discovery`.

Components:

- `DiscoverySeedExtractor`: extracts topics, entities, concepts, claims, findings, and external context from bounded fields, not whole long source texts. Every seed stores `source_artifact`, `source_field`, `source_ids`, and input artifact hashes.
- `YouTubeQueryPlanner`: builds short RU/EN YouTube search queries, separates intents `same_topic`, `counterpoint`, `deep_dive`, `source_author`, `channel_discovery`, `recent_updates`, and dedupes weak or repeated queries.
- `YouTubeDiscoveryProvider`: uses only YouTube Data API methods `search_videos`, `search_channels`, `get_video_details`, `get_channel_details`. It does not use `yt-dlp`, cookies, or invented URLs.
- `CandidateNormalizer`: accepts only verified IDs returned by Data API and builds canonical URLs from those IDs.
- `CandidateScorer`: computes relevance, overlap, alignment, novelty, risk, language, availability, caption readiness, authority, freshness, and final `discovery_score`.
- `DiscoveryReviewer`: blocks missing IDs, hallucinated URLs, and missing provenance; marks candidates as `accepted`, `rejected`, or `needs_review`.
- `DiscoveryArtifactsWriter`: writes `seeds.json`, `queries.json`, `search_results.raw.json`, `video_candidates.json`, `channel_candidates.json`, `review.json`, `graph.json`, `discovery_report.md`, and `status.json`.
- `DiscoveryGraphBuilder`: emits discovery-specific Neo4j nodes/relationships while preserving research/run provenance.
- `DiscoveryEnqueuePolicy`: explicit manual enqueue only.

## 4. Runtime / Swarm / Parallelism

Discovery stage names:

- `youtube_discovery_seed_extract`
- `youtube_discovery_query_expand`
- dynamic `youtube_discovery_query_000...NNN`
- `youtube_discovery_candidate_fetch`
- `youtube_discovery_candidate_score`
- `youtube_discovery_review`
- `youtube_discovery_graph`
- `youtube_discovery_report`
- optional `youtube_discovery_enqueue`

Locks:

- `youtube_data_api`: default 3 slots, configurable in RuntimeStore settings.
- `local_candidate_scoring`: 4 slots.
- `llm_discovery_review`: 1 slot.
- `neo4j_sync`: 1 slot.
- `queue_mutation`: 1 slot.

BeeAI can coordinate agents (`SeedExtractorAgent`, `QueryPlannerAgent`, `CandidateSearchAgent`, `RelevanceCriticAgent`, `ContradictionCriticAgent`, `GraphSyncAgent`), but RuntimeStore remains canonical for locks, retries, leases, cancellation, and status. BeeAI must not invent candidates, statuses, IDs, or provenance.

## 5. Completeness / Relevance / Non-Contradiction

Completeness gate:

- Key concepts are represented by selected seeds.
- Findings are mapped to at least one query when possible.
- At least 10 video candidates or explicit warning reason.
- At least 3 channel candidates or explicit warning reason.

Relevance gate:

- Title, description, channel title, or tags overlap with seed/query terms.
- Candidate has seed/query provenance.
- Report explains why each accepted candidate is relevant.

Non-contradiction gate:

- Candidate relation is one of `supports`, `expands`, `contradicts`, `unclear`.
- Counterpoints are marked as `contradicts`; they are not mixed into plain related recommendations.
- Report shows supports/expands/contradicts/unclear separately through review status and reasons.

Anti-hallucination:

- YouTube URL is built only from Data API IDs.
- Source URLs come from fetched research/search artifacts.
- LLM cannot create IDs.
- LLM outputs, if added later, must pass schema validation.

Context-window control:

- Do not send full source files to LLM.
- Use top-k seeds and source summaries.
- Prefer deterministic extraction first.
- Use LLM only for review/reranking after schema validation.

## 6. Data Contracts

All discovery JSON artifacts include `schema_version`, `video_id`, `research_run_id`, `discovery_run_id`, `created_at`, `updated_at`, `input_hashes`, and `artifact_hash` where applicable.

`youtube_discovery/seeds.json`:

- Required root fields: common fields plus `items`.
- Item fields: `seed_id`, `kind`, `text`, `language`, `source_artifact`, `source_field`, `source_ids`, `confidence`, `selected`.

`youtube_discovery/queries.json`:

- Item fields: `query_id`, `seed_ids`, `intent`, `language`, `query`, `priority`, `dedupe_key`, `quota_estimate`, `rejected_reason`.

`youtube_discovery/video_candidates.json`:

- Item fields: `candidate_id`, `kind=video`, `youtube_video_id`, `canonical_url`, `title`, `description`, `channel_id`, `channel_title`, `published_at`, `duration`, `view_count`, `language`, `tags`, `thumbnails`, all score fields, `query_ids`, `seed_ids`, `source_ids`, `relationship`, `review_status`, `processed_status`, `reason`.

`youtube_discovery/channel_candidates.json`:

- Item fields: `candidate_id`, `kind=channel`, `youtube_channel_id`, `canonical_url`, `title`, `description`, `published_at`, `subscriber_count`, `video_count`, `view_count`, `language`, `thumbnails`, all score fields, `query_ids`, `seed_ids`, `source_ids`, `relationship`, `review_status`, `reason`.

`youtube_discovery/review.json`:

- Fields: `status`, `issues`, `accepted_candidate_ids`, `rejected_candidate_ids`, `needs_review_candidate_ids`, `completeness`, `relevance`, `non_contradiction`.

`youtube_discovery/graph.json`:

- Fields: `nodes`, `relationships`.
- Required relationship provenance: `source_video_id`, `research_run_id`, `discovery_run_id`, `seed_id`, `query_id`, `youtube_video_id` or `youtube_channel_id`, `score`, `review_status`.

`youtube_discovery/discovery_report.md`:

- Sections: summary, video candidates, channel candidates, review issues.
- Candidate rows include status, score, title, canonical URL, and reason.

## 7. CLI / API

CLI:

- `yt-sub-hub discover-youtube <video_id> --limit 20 --max-quota-units 900`
- `yt-sub-hub review-discovery <video_id>`
- `yt-sub-hub discovery-status <video_id> --json`
- `yt-sub-hub enqueue-discovered-videos <video_id> --min-score 0.75 --limit 5`
- `yt-sub-hub enqueue-discovered-channel <video_id> --channel-id ...`
- `yt-sub-hub discovery-sync-neo4j <video_id>`

API:

- `POST /videos/{video_id}/discovery/youtube/run`
- `GET /videos/{video_id}/discovery/youtube`
- `GET /videos/{video_id}/discovery/youtube/review`
- `POST /videos/{video_id}/discovery/youtube/sync-neo4j`
- `POST /videos/{video_id}/discovery/youtube/enqueue`
- `POST /videos/{video_id}/discovery/youtube/enqueue-channel`

## 8. Product UX / Artifact Browser

Artifact Browser includes discovery seeds, queries, video candidates, channel candidates, review, graph, and report.

The UI should show:

- Found videos/channels.
- Score and reason.
- Seed/query provenance.
- Caption readiness score.
- Processed status: already processed, queued, new candidate.
- Actions: enqueue video, enqueue channel, reject candidate, mark reviewed, sync discovery graph.

## 9. Security / Safety / Quotas

Discovery uses only YouTube Data API. It never uses browser cookies or `yt-dlp`.

Safety controls:

- Max quota units per run.
- Max candidates per run.
- No auto-enqueue.
- No recursive channel expansion.
- Allowlist/blocklist support can be added around candidate review before enqueue.
- Existing RuntimeStore dedupes standalone video jobs by video ID.
- Enqueue writes `enqueue_audit.json`.

Quota policy:

- Default run budget: `--max-quota-units 900`.
- `search.list`: 100 units per request.
- `videos.list` / `channels.list`: low-cost detail lookups.
- Quota/rate errors fail discovery without mutating queue.

## 10. Implementation Backlog

### Epic 1. Architecture and contracts

- Story: Add ADR and artifact schemas.
- Scope: docs, JSON root contracts, status values.
- Out of scope: full JSON Schema package.
- Files: `docs/youtube-discovery-from-research-artifacts.md`, `src/yt_sub_hub/youtube_discovery.py`.
- Acceptance: artifacts include common metadata and hashes.
- Tests: discovery artifact serialization tests.
- DoD: contracts documented and enforced by tests.

### Epic 2. Discovery seed extraction

- Story: Extract bounded seeds from analysis/research artifacts.
- Scope: concepts, claims, findings, source summary key points.
- Out of scope: full raw source text ingestion.
- Files: `youtube_discovery.py`.
- Acceptance: seeds have provenance and selected flags.
- Tests: passed research review required; missing artifacts fail.
- DoD: no LLM-generated seeds required for MVP.

### Epic 3. YouTube Data API discovery provider

- Story: Add fakeable provider for search and details.
- Scope: `search.list`, `videos.list`, `channels.list`, quota budget.
- Out of scope: `yt-dlp`, cookies.
- Acceptance: provider returns normalized dicts with verified IDs.
- Tests: fake provider, quota overflow, sanitized API errors.
- DoD: no live network in unit tests.

### Epic 4. Candidate normalization and dedupe

- Story: Convert provider results into canonical candidates.
- Scope: video/channel canonical URLs, self-video exclusion, dedupe by ID.
- Acceptance: no hallucinated URLs.
- Tests: duplicate and self-video cases.
- DoD: only Data API IDs become candidates.

### Epic 5. Candidate scoring and review

- Story: Score and gate candidates.
- Scope: deterministic score fields and review statuses.
- Acceptance: accepted/rejected/needs_review split.
- Tests: invalid ID/provenance fails review.
- DoD: enqueue cannot bypass review.

### Epic 6. Runtime/BeeAI staged orchestration

- Story: Reserve discovery stage names and locks.
- Scope: stage metadata and resource locks.
- Out of scope: autonomous BeeAI candidate creation.
- Acceptance: stage lookup resolves discovery locks.
- Tests: stage_by_name/runtime lock tests.
- DoD: RuntimeStore remains source of truth.

### Epic 7. Discovery artifacts and report

- Story: Write JSON artifacts and Markdown report.
- Scope: `youtube_discovery/*`.
- Acceptance: status and artifact browser expose files.
- Tests: artifact browser tests.
- DoD: report includes links/reasons/issues.

### Epic 8. Neo4j discovery graph sync

- Story: Add discovery graph labels/relationships and sync function.
- Scope: graph payload, delete discovery for video, consistency.
- Acceptance: `missing=0`, `extra=0`.
- Tests: fake graph store sync.
- DoD: sync requires non-failed review.

### Epic 9. CLI/API/product status

- Story: Wire commands and endpoints.
- Scope: run, status, review, sync, enqueue.
- Acceptance: controlled errors, JSON output.
- Tests: CLI/API delegation tests.
- DoD: commands documented in ADR.

### Epic 10. Enqueue policy and recovery

- Story: Enqueue accepted candidates explicitly.
- Scope: videos and one selected channel.
- Acceptance: idempotent RuntimeStore video jobs, audit log.
- Tests: enqueue idempotency.
- DoD: discovery run never creates queue jobs.

### Epic 11. Tests and live acceptance

- Story: Verify MVP with fake providers and targeted suite.
- Scope: no live API by default.
- Acceptance: targeted tests pass.
- DoD: live Data API run remains opt-in through CLI/API.

## 11. Acceptance Criteria

MVP GO:

- Discovery runs on a video with completed research artifacts.
- Creates seeds, queries, video candidates, channel candidates, review, report.
- Candidates have verified YouTube IDs from Data API/provider contract.
- No hallucinated URLs.
- Fewer than 10 video candidates or 3 channel candidates produce explicit warnings.
- Review splits candidates into accepted/rejected/needs_review.
- Report contains YouTube links and relevance reasons.
- Neo4j sync uses discovery graph and consistency check.
- Enqueue works only after non-failed review.
- Discovery run itself does not enqueue.
- Locks exist for YouTube API, local scoring, LLM review, Neo4j, queue mutation.
- Tests pass.

## 12. Final Output

Implementation entrypoints:

- Module: `src/yt_sub_hub/youtube_discovery.py`
- CLI/API: `src/yt_sub_hub/cli.py`, `src/yt_sub_hub/api.py`
- Graph: `src/yt_sub_hub/graph.py`
- Browser: `src/yt_sub_hub/artifact_browser.py`
- Runtime locks/stages: `src/yt_sub_hub/runtime.py`, `src/yt_sub_hub/stages.py`

Verdict: ready for MVP implementation and controlled live testing with a YouTube Data API token.
