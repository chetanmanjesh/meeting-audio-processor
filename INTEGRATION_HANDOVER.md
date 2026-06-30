# Integration Handover — Folding Meeting Audio Processor into the Studio Dashboard

**Purpose:** This document is for the session that integrates the standalone Meeting Audio Processor into the Studio Portal dashboard. It captures the current state of both systems, the integration decisions to make, what must not break, and a recommended phased plan.

**Date:** 2026-06-30 · **Status of the LLM question:** CLOSED — Sonnet 4.6 is the *recommended* default for the client deliverable, but the model is env-controlled (set manually on Render), not locked (see §6).

---

## 0. The two codebases

| | Meeting Audio Processor (source) | Studio Portal (target) |
|---|---|---|
| Path | `~/Desktop/dashboard_peggy/meeting-audio-processor/` | `~/Desktop/dashboard_peggy/studio-portal-beta/` |
| Frontend | Plain HTML/JS embedded in `server.py` (`INDEX_HTML`) | React + Vite, single-file `src/App.jsx` (~8k lines, intentional) |
| Backend | FastAPI `server.py` (1736 lines) | FastAPI `api_server.py` on port **8787** |
| DB / state | In-memory `JOBS` dict + R2 cache (no DB) | **Supabase** (Postgres + storage) |
| Storage | **Cloudflare R2** (S3-compatible, India POP) | Supabase storage |
| Hosting | Render free tier | (see its own `HANDOVER.md`) |
| Auth | `?token=<ACCESS_TOKEN>` query param | Real dashboard auth + roles (`chief_designer`, etc.) |

Read both repos' `HANDOVER.md` first. The source app's full architecture (pipeline, endpoints, R2 layout, design decisions, costs, gotchas) is in `meeting-audio-processor/HANDOVER.md` — **do not duplicate it here; read it.** This file only covers the *integration*.

---

## 1. What the processor does (one paragraph)

Audio in → structured, branded Minutes-of-Meeting `.docx` out (Gurkar & Associates letterhead). Pipeline: browser uploads audio directly to R2 (presigned PUT) → Render worker downloads → optional ffmpeg stitch → transcribe (Deepgram default / Sarvam for Indian languages) → extract per-speaker 8s samples → **Claude Sonnet 4.6** extracts a themed MoM JSON → render markdown + docx from `template.docx`. Opt-in: concise version, speaker identification. It is a **client-facing deliverable generator**, not an internal transcriber.

---

## 2. The reusable backend surface (what to wire into `api_server.py`)

The processing logic is already cleanly separated from the web layer — these modules are import-and-call, no HTTP needed:

- `process.py` — `extract_mom()`, `condense_to_concise()`, transcription (Deepgram), the `LLM_PROVIDER` dispatcher (`_llm_stream_json`). **This is where `CLAUDE_MODEL` now lives (env-configurable, default `claude-sonnet-4-6`).**
- `process_template.py` — `fill_template()`, `fill_template_to_bytes()`, `auto_meta()`, `build_discussion_blocks()`, `build_open_items_blocks()`. Turns MoM JSON → docx.
- `process_sarvam.py` — Sarvam Saarika ASR wrapper.
- `template.docx` — the branded template. Must travel with the integration.

The HTTP endpoints in `server.py` are the contract the current frontend uses — replicate this surface in `api_server.py` (paths can change to match dashboard conventions):

| Method | Path | What |
|---|---|---|
| POST | `/upload-url` | Mint presigned R2 PUT URL. `{filename, content_type}` → `{url, key}` |
| POST | `/jobs` | Create job. `{r2_keys, filenames, provider}` → `{job_id}`. Worker = background thread. |
| GET | `/jobs/{id}` | Lightweight status poll (status/step/progress) |
| GET | `/jobs/{id}/result` | Full result (transcript + both MoM blocks) |
| GET | `/jobs/{id}/docx?style=detailed\|concise` | Stream the .docx |
| POST | `/jobs/{id}/retry` | Re-run failed job from R2-cached intermediates (no re-pay) |
| POST | `/jobs/{id}/condense` | Generate concise version on demand |
| GET | `/jobs/{id}/speaker-samples` | Speaker samples + presigned audio URLs + name mapping |
| POST | `/jobs/{id}/identify-speakers` | Apply `{label: name}` → re-render all outputs |
| POST | `/jobs/{id}/reset-speakers` | Restore original `Speaker N` labels |

---

## 3. Integration decisions (recommendations, but confirm with Chetan)

**3a. Backend: in-process vs microservice.**
Recommend **in-process** — import `process.py` / `process_template.py` into `api_server.py` and run the worker as a background task/thread there. The modules are already decoupled and this avoids a second deployment. Keep the standalone Render app alive until the dashboard version is verified, then retire it.

**3b. Job state: in-memory dict → Supabase table.**
The standalone app's `JOBS` dict dies on Render restart (mitigated only by R2 caching). The dashboard has Supabase — move job state into a `meeting_jobs` table (id, status, step, progress, r2_keys, result_json, speaker_map, created_by, created_at). This is the single biggest robustness upgrade the integration unlocks: resumable, multi-user, survives restarts. **Keep the R2 intermediate cache** regardless — it's what makes retries free.

**3c. Storage: keep R2, do NOT move uploads to Supabase storage.**
The direct browser→R2 presigned upload exists specifically because **Render's edge proxy times out long uploads from India** (Bangalore→Singapore latency). R2 also has an India POP and zero egress. Moving uploads to Supabase storage risks reintroducing the timeout. Recommendation: audio + samples + cache stay on R2; only *job metadata and final docx references* go in Supabase. Revisit only if you later host everything on the Oracle VM.

**3d. Frontend: React port, not iframe.**
Rebuild the `INDEX_HTML` UI as a React view inside `App.jsx`, calling the endpoints above. Iframing the old app would bypass dashboard auth and look bolted-on. The flows to port: multi-file upload + progress polling, provider selector (Deepgram/Sarvam), detailed/concise download, the Identify-Speakers modal (per-speaker audio playback + name inputs).

**3e. Auth: drop `?token=`, gate behind dashboard auth.**
Remove the `ACCESS_TOKEN` query-param scheme. Gate the feature behind the dashboard's existing auth + appropriate role. Mint R2 presigned URLs server-side only for authenticated users.

---

## 4. Constraints that must NOT break (hard-won)

1. **Direct browser→R2 upload** — do not route audio through the backend (Render/India timeout). §3c.
2. **Resumable R2 caching** — each step (transcript / detailed / concise) caches to R2 keyed by job_id; retry recomputes only missing steps. Claude steps are expensive; keep this.
3. **Concise is opt-in** — second LLM pass, ~₹6 + ~2 min. Don't make it automatic.
4. **British English everywhere** — cousin's preference, baked into prompts. Keep.
5. **Themed MoM schema** — `decisions: [{theme, items:[{lead, detail}]}]` (not flat lists). This is what produces the scannable bold-lead structure matching her existing MoM style. Don't flatten.
6. **`OPEN ITEMS` template lookup uses `startswith`, not substring** — a prior bug where an LLM summary containing "open items" clobbered DISCUSSION NOTES. Be defensive about substring matches on any new template field.
7. **`template.docx` is the firm's branded asset** — template-per-firm is the productization model; keep it swappable, not hardcoded.

---

## 5. Environment variables to carry over

From `meeting-audio-processor/.env.example` / `render.yaml` into the dashboard's env:

- `DEEPGRAM_API_KEY`, `SARVAM_API_KEY` (ASR)
- `ANTHROPIC_API_KEY` + `CLAUDE_MODEL` (default `claude-sonnet-4-6`)
- `LLM_PROVIDER` (keep `claude`), and the now-supported `DEEPSEEK_*` / `GEMINI_*` if you want the A/B harness available
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`
- Drop `ACCESS_TOKEN` (replaced by dashboard auth)

R2 CORS must include the dashboard's origin (currently allows only the Render URL + localhost:8000) — see `HANDOVER.md` §"R2 bucket setup".

---

## 6. The LLM model question is settled — use Sonnet 4.6

A cost-optimisation investigation across **four** cheaper models is closed. All failed; Sonnet 4.6 is the permanent default. Benchmark = one on-site clubhouse meeting. The verdict came from **reading the full docx outputs**, not bullet-counting (which proved misleading — see the Haiku note). Decision figures below are rough bullet counts.

| Model | Decisions | Sign-offs | Materials | Actions | Both safety items? | Verdict |
|---|---|---|---|---|---|---|
| **Sonnet 4.6** | 12 | 2 | 4 | 8 | yes — as urgent ACTION ITEMS | **baseline / KEEP** |
| Gemini Flash 2.5 (free) | 3 | 0 | 4 | 5 | yes | FAIL (collapse to 3, 0 sign-offs) |
| DeepSeek V3 (paid) | 3 | 1 | 1 | 5 | no | FAIL |
| Claude Haiku 4.5 | 8 | 2 | 4 | 8 | yes — but demoted to Open Items | FAIL (qualitative) |
| DeepSeek-Reasoner R1 | 5 | 3 | 1 | 5 | fire only, landing dropped | FAIL |

**Why Haiku fails despite being close (it is NOT a safety drop):** Haiku keeps both safety items and is even more precise than Sonnet on some MEP detail. It fails for three qualitative reasons that matter for a *client-facing accountability* deliverable: (1) it **demotes the fire-exit + landing-height from urgent Action Items to parked "Open Items / past defects with no resolution path"** — Sonnet makes them explicit urgent actions; (2) **weaker speaker-role inference** (mislabels the client/owner as "architect," leaks a raw "project_manager" label); (3) a **likely meeting-date error** (defaulted to today vs Sonnet's correct extraction) plus a subtle commitment change ("identify a maintenance person to be trained" → "provide training documentation"). For internal notes Haiku would be defensible; for the client deliverable, Sonnet wins.

**Takeaway:** the discriminator isn't raw counts or even "is the safety item in the doc" — it's *how* the deliverable handles safety items (urgent action vs parked note), speaker-role reliability, and metadata accuracy. That package is a Sonnet-class capability here. **Recommendation: keep the integration on Sonnet for client-facing output**, but the model stays env-controlled (`CLAUDE_MODEL` / `LLM_PROVIDER`, set manually on Render) — don't hardcode it, and Chetan may choose a cheaper model for internal/lower-stakes runs.

The infra for *future* tests is in place (commit `8c43b7e` on `main`): `CLAUDE_MODEL` is env-configurable and `deepseek-reasoner` is unblocked (`response_format` applied only for `deepseek-chat*`). To benchmark any future model: flip env vars, generate a docx, and **evaluate by reading the full docx** (a section/bullet-count script wrongly flagged Haiku as a safety-drop). Reference docx baselines are in `~/Downloads/mom_detailed_*.docx`.

---

## 7. Recommended phased plan

1. **Schema + storage** — add a `meeting_jobs` table in Supabase; keep R2 for audio/samples/cache. Add R2 + Anthropic + ASR keys to the dashboard env. Update R2 CORS for the dashboard origin.
2. **Backend** — import `process.py` / `process_template.py` / `process_sarvam.py` into `api_server.py`; replicate the endpoint surface (§2) behind dashboard auth; back job state with the Supabase table; preserve R2 caching + retry.
3. **Frontend** — port the upload/progress/download/identify-speakers flows into a React view in `App.jsx`.
4. **Verify** — run a real meeting end-to-end through the dashboard; confirm the docx matches the standalone app's output (same Sonnet baseline). Confirm resumability survives an `api_server.py` restart mid-job.
5. **Cutover** — once verified, retire the standalone Render service (or keep as fallback).

---

## 8. Open questions for Chetan

- **Multi-studio now or later?** Currently single-firm (Gurkar & Associates). If the dashboard already has workspace/firm isolation, wire `template.docx` + branding per firm now; otherwise keep single-template and defer.
- **Who can access the feature?** Which dashboard roles see Meeting Processor (all users vs `chief_designer`)?
- **Retire Render or keep as fallback?** Affects whether `server.py` / `render.yaml` stay maintained.
- **Cold-start / long-job hosting:** Render free tier sleeps after 15 min idle and kills running jobs if the tab closes. The dashboard's hosting must handle minutes-long background jobs — confirm it does, or plan the Oracle VM move noted in `HANDOVER.md` §"What's next".

---

*Companion docs: `HANDOVER.md` (full processor architecture), `SESSION_HANDOVER.md` (last session's LLM-test state + comparison-script pattern), and the dashboard's own `HANDOVER.md`.*
