# Meeting Audio Processor — Project Handover

## What it is

A web app that turns meeting audio recordings (1-file or multi-file stitched) into structured Minutes of Meeting deliverables — a downloadable `.docx` formatted to match the Gurkar & Associates studio template, plus markdown + JSON sidecar files. Built for **Chetan's cousin's interior design studio in Bangalore** (Gurkar & Associates). She records client/site meetings on her phone (mp4/Voice Memos), uploads them here, and gets back a polished MoM she can send to clients or use internally for BOQ prep.

Repo lives at: `~/Desktop/dashboard_peggy/meeting-audio-processor/`
Deployed at: `https://meeting-audio-processor.onrender.com/?token=<ACCESS_TOKEN>`

---

## Pipeline (what happens to an uploaded audio)

```
1. Browser → R2 (direct upload via presigned URL — bypasses Render's edge timeout)
2. R2 → Render worker (downloads audio to /tmp)
3. (If multi-file) ffmpeg-stitch into one file via imageio-ffmpeg
4. Transcribe:
     - Deepgram (default, cheaper, English-leaning)  -OR-
     - Sarvam Saarika v2.5 (Indian-English, Kannada/Hindi/Tamil/etc, ~3× pricier per minute)
5. Extract speaker audio samples (~8s clip per unique speaker) → upload to R2
6. LLM extracts a "detailed" structured MoM JSON (Claude Sonnet 4.6 by default;
   Gemini Flash 2.0 also wired up as an alternative provider)
7. Render detailed markdown + detailed docx (filled into template.docx)
8. (Opt-in) User can click "Generate concise version" → second LLM pass
   tightens prose without dropping facts → concise markdown + docx
9. (Opt-in) User can click "Identify speakers" → modal with each speaker's
   sample audio + name input → string-replace "Speaker N" with names across
   transcript / MoM JSON / markdown / docx. Originals cached; reset available.
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Backend | **FastAPI** (`server.py`) | Single-file, async, simple |
| Frontend | Plain HTML/JS embedded in `server.py` (`INDEX_HTML`) | No build step |
| Hosting | **Render** free tier | Free, with caveats (see below) |
| Object storage | **Cloudflare R2** (S3-compatible) | Free tier, zero egress, edge POP in India |
| ASR (transcription) | **Deepgram nova-3** OR **Sarvam Saarika v2.5** (user-selectable) | Deepgram cheap+fast; Sarvam catches Indian-English + Kannada/Hindi that Deepgram drops |
| LLM (extraction) | **Anthropic Claude Sonnet 4.6** by default; **Google Gemini Flash 2.0** as alt | Sonnet is most reliable for structured JSON + multilingual reading; Gemini Flash is the free-tier alternative |
| Audio toolchain | **imageio-ffmpeg** (bundles a static ffmpeg binary, no apt needed) | Works on Render's Python buildpack |
| Docx generation | **python-docx** | Fills `template.docx` (Gurkar & Associates branded) |

---

## File structure

```
meeting-audio-processor/
├── server.py               ← FastAPI app + embedded HTML/JS UI. 1736 lines.
├── process.py              ← Core pipeline: stitch, transcribe (Deepgram),
│                             MoM prompts + extraction, condense, render markdown.
│                             574 lines. The LLM_PROVIDER dispatcher lives here.
├── process_sarvam.py       ← Sarvam Saarika ASR wrapper.
├── process_template.py     ← Fills the docx template from a MoM JSON.
│                             Has fill_template(), auto_meta(), build_discussion_blocks(),
│                             build_open_items_blocks(), fill_template_to_bytes().
├── process_multi_formats.py ← CLI variant for stitching mixed-format files locally.
├── template.docx           ← The Gurkar & Associates studio's MoM template (provided
│                             by Chetan's cousin). Has fixed sections: PURPOSE/SUMMARY,
│                             DISCUSSION NOTES, ACTION ITEMS (table), OPEN ITEMS table.
├── requirements.txt        ← All Python deps.
├── render.yaml             ← Render deploy config (env var stubs).
├── .env.example            ← Documented env vars.
├── README.md               ← Original brief docs (light).
└── HANDOVER.md             ← This file.
```

---

## Environment variables

| Key | Required | Notes |
|---|---|---|
| `DEEPGRAM_API_KEY` | yes (if user picks Deepgram) | nova-3 transcription |
| `SARVAM_API_KEY` | yes (if user picks Sarvam) | Saarika batch ASR |
| `ANTHROPIC_API_KEY` | yes (if LLM_PROVIDER=claude) | Claude Sonnet for MoM extraction + condense |
| `LLM_PROVIDER` | optional; default `claude` | `claude`, `gemini`, or `deepseek` |
| `CLAUDE_MODEL` | optional; default `claude-sonnet-4-6` | `claude-haiku-4-5` is ~5x cheaper; A/B-test quality before defaulting |
| `GEMINI_API_KEY` | if LLM_PROVIDER=gemini | From https://aistudio.google.com/app/apikey |
| `GEMINI_MODEL` | optional; default `gemini-2.0-flash` | Try `gemini-2.5-flash` if free-tier quota issues |
| `DEEPSEEK_API_KEY` | if LLM_PROVIDER=deepseek | From https://platform.deepseek.com/api_keys |
| `DEEPSEEK_MODEL` | optional; default `deepseek-chat` | `deepseek-chat` is V3; `deepseek-reasoner` is R1 (now supported — `response_format` applied only for `deepseek-chat*`) |
| `R2_ACCOUNT_ID` | yes | Cloudflare account ID |
| `R2_ACCESS_KEY_ID` | yes | R2 API token |
| `R2_SECRET_ACCESS_KEY` | yes | R2 API token secret |
| `R2_BUCKET_NAME` | yes | Bucket holding uploaded audio + samples + intermediate caches |
| `ACCESS_TOKEN` | recommended | URL query param `?token=` required if set. Without it, anyone with the URL burns the API budget. |
| `PYTHON_VERSION` | yes | `3.11.10` (Render buildpack reads this) |

---

## R2 bucket setup

Bucket: `meeting-audio-uploads` (Cloudflare APAC region).

**Required CORS** (Settings → CORS Policy):
```json
[
  {
    "AllowedOrigins": [
      "https://meeting-audio-processor.onrender.com",
      "http://localhost:8000"
    ],
    "AllowedMethods": ["PUT", "GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3000
  }
]
```

**Required lifecycle rules** (Settings → Object lifecycle rules):
- Prefix `uploads/` → delete after 1 day (audio files; deleted on success anyway, this catches orphans)
- Prefix `samples/` → delete after 7 days (per-speaker audio clips; used by the Identify Speakers feature)
- Prefix `cache/` → optional cleanup (intermediates are explicitly deleted on success)

Storage layout in R2:
- `uploads/<random-uuid>/<filename>` — uploaded audio (deleted on success)
- `samples/<job_id>/speaker_<N>.mp3` — per-speaker 8s clips
- `cache/<job_id>/transcript.txt` — cached transcript (for retry)
- `cache/<job_id>/detailed.json` — cached detailed MoM
- `cache/<job_id>/concise.json` — cached concise MoM (if generated)
- `cache/<job_id>/speaker_samples.json` — sample metadata

---

## HTTP endpoints (server.py)

| Method | Path | What |
|---|---|---|
| GET | `/` | Serves the HTML UI |
| POST | `/upload-url` | Mint a presigned R2 PUT URL for one file. Body: `{filename, content_type}` → `{url, key}` |
| POST | `/jobs` | Create a processing job. Body: `{r2_keys, filenames, provider}` → `{job_id}`. Worker runs in background thread. |
| GET | `/jobs/{id}` | Lightweight status poll. Returns status + step + progress, strips heavy fields. |
| GET | `/jobs/{id}/result` | Full result when done — transcript + both MoM blocks |
| GET | `/jobs/{id}/docx?style=detailed\|concise` | Stream the rendered .docx |
| POST | `/jobs/{id}/retry` | Re-run a failed job. Uses cached intermediates so no re-pay. |
| POST | `/jobs/{id}/condense` | Generate the concise version on demand (opt-in) |
| GET | `/jobs/{id}/speaker-samples` | List speaker samples + presigned audio URLs + current name mapping |
| POST | `/jobs/{id}/identify-speakers` | Apply `{label: name}` mapping → re-render all outputs |
| POST | `/jobs/{id}/reset-speakers` | Restore original Speaker N labels |

All endpoints require `?token=<ACCESS_TOKEN>` if that env var is set.

---

## Branches

| Branch | What | Status |
|---|---|---|
| `main` | Production. Contains everything below up to speaker identification. | **Live on Render.** Origin and local are in sync. |
| `gemini-llm-provider` | Adds Gemini Flash 2.0 as an alternative LLM provider behind `LLM_PROVIDER` env var. | Local only — not pushed yet. |

Older branches (`speaker-identification`) are now redundant with `main` and can be deleted.

---

## Key design decisions (and why)

1. **Direct browser → R2 upload, not via Render.** Render's edge proxy times out long uploads from India (Bangalore → Singapore region is high-latency for large files). Bypassing it solved persistent upload failures.

2. **In-memory JOBS dict for state** (not Redis/DB). Render free tier wipes memory on restart; we live with that and use R2 for intermediate caching so retries survive restarts.

3. **Resumable jobs.** Each pipeline step (transcript, detailed MoM, concise MoM) caches its output to R2 keyed by job_id. On retry, the worker checks JOBS → falls back to R2 cache → only recomputes missing steps. **Critical** because Claude steps are expensive and Render free tier flakes occasionally.

4. **Concise version is opt-in.** Saves ~₹6 and ~2 min per meeting since most use cases only need detailed.

5. **Shared MoM prompt for both providers.** `process.py`'s `MOM_SYSTEM_PROMPT` and `CONCISE_SYSTEM_PROMPT` are provider-agnostic. Switching `LLM_PROVIDER` doesn't change schemas or outputs (in theory — quality differs).

6. **British English everywhere.** Cousin's preference. The prompts explicitly instruct it, and it sticks reliably.

7. **The MoM schema has themed grouping** (`decisions: [{theme, items: [{lead, detail}]}]`) instead of flat lists. Forces Claude/Gemini to produce scannable structure matching the cousin's existing MoM style — bold scannable leads + thematic sub-headings inside DISCUSSION NOTES.

---

## Costs per meeting (rough, 1-hour audio)

| Step | Claude default | Gemini Flash (free tier) |
|---|---|---|
| Deepgram transcription | ~₹22 | same |
| OR Sarvam transcription | ~₹4–8 (varies) | same |
| Detailed MoM extraction | ~₹20 | **₹0** (under daily cap) |
| Concise condense (opt-in) | ~₹6 | **₹0** |
| Render hosting | free tier | same |
| R2 storage / egress | free tier | same |
| **Total all-in (with concise)** | **~₹50** | **~₹4–22** |

---

## Known issues / gotchas

1. **Gemini free-tier quota currently shows `limit: 0`** for `gemini-2.0-flash` on Chetan's project. Likely fix: try `GEMINI_MODEL=gemini-2.5-flash`, or re-generate the API key from https://aistudio.google.com/app/apikey (not Cloud Console — that ties to billing). In progress — Chetan hasn't tested the fix yet.

2. **Render free tier sleeps after 15 min idle.** Cold start ~30s. If a job is running when sleep hits (browser tab closed → no polling), the job dies. Mitigation: cousin should keep the tab open during processing. Permanent fix: $7/month Render Starter, or external pings (Cloudflare Worker cron) — both noted but not done.

3. **R2 metadata cleanup on success deletes only transcript/detailed/concise cache.** Speaker samples + their metadata JSON stay around until R2 lifecycle rule cleans them (7 days). Intentional — Identify Speakers feature needs them after job completes.

4. **`token=` query param protects the app** but is visible in URL/history. For real-world use, prefer rotating the token periodically or moving to proper auth eventually.

5. **Sarvam pricing is high.** ~₹8 per request average from Chetan's testing. Free tier credits run out fast. For most studio meetings (English-dominant), Deepgram is the better default — the new MoM prompt squeezes most of the value out of Deepgram's transcripts even though Sarvam captures ~2× more raw words.

6. **OPEN ITEMS placeholder lookup**: was buggy (`"OPEN ITEMS" in t.upper()` substring match) — the LLM-generated summary would sometimes contain the phrase "open items", which matched and clobbered the DISCUSSION NOTES section. Fixed in `process_template.py` — now uses `startswith("OPEN ITEMS")`. Worth being defensive about this pattern in any future template fields.

---

## Run locally

```bash
cd ~/Desktop/dashboard_peggy/meeting-audio-processor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in keys in .env
uvicorn server:app --reload --port 8000
# Open http://localhost:8000
```

For local testing without R2, the CLI scripts still work:
```bash
python process.py meeting.mp3                                    # Deepgram
python process_sarvam.py meeting.mp3                             # Sarvam
python process_template.py meeting.mp3 --provider deepgram       # Full pipeline + docx
```

---

## Deploy

Render auto-deploys on push to `main` (`render.yaml` is the source of truth for the service config). After any code change:

```bash
git add -A
git commit -m "..."
git push
# Wait ~3 min for build + ~30s boot.
```

Render Dashboard → service → Logs to see what happened.

---

## What's next (likely roadmap)

1. **Confirm Gemini Flash works** (model swap → re-key from AI Studio if needed). If Gemini's free tier is usable, default `LLM_PROVIDER=gemini` in production to eliminate per-meeting Claude cost.
2. **Self-host IndicConformer + pyannote** for transcription. AI4Bharat's open model, same lineage as Sarvam, free if running on an Oracle Cloud free-tier ARM Ampere VM (Chetan has one provisioned but hasn't deployed to it yet).
3. **Move web app to the same Oracle VM** for no cold starts. Render free tier was the right MVP, not the right production.
4. **Persistence beyond R2 cache.** JOBS dict is in-memory; once we move to Oracle or upgrade Render, consider SQLite on disk for job state.
5. **Multi-user / multi-studio.** Currently single shared instance for one studio. Real productization would add user auth + workspace isolation.

---

## Related context (other Chetan projects)

- **Studio Portal** (`~/Desktop/dashboard_peggy/studio-portal-beta/`) — the main dashboard this meeting processor is eventually expected to fold into. React + Vite frontend, FastAPI backend on port 8787, Supabase. See its own `HANDOVER.md`.
- **CAD single→double-line converter** (`~/Desktop/cad_project/`) — separate client project, Shapely-based.

---

*Last updated: 2026-06-20. Author: Claude (with Chetan).*
