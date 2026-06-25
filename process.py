"""
Meeting audio processor — CLI.

Usage:
    python process.py path/to/audio.mp3
    python process.py path/to/audio.mp3 --out output/

Pipeline:
    1. Send audio to Deepgram (nova-3) with diarization.
    2. Format the diarized transcript as "Speaker N: ..." paragraphs.
    3. Send transcript to Claude for structured MoM extraction.
    4. Write transcript.txt and mom.json (and a pretty mom.md) to the output dir.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
from anthropic import Anthropic
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = "claude-sonnet-4-6"

# LLM provider switch — controls extract_mom() and condense_to_concise().
#   "claude":   Anthropic Sonnet. ~₹25/meeting. Most reliable + polished.
#   "gemini":   Google Gemini Flash. Free up to daily cap but capacity-bursty.
#   "deepseek": DeepSeek V3. ~₹2/meeting (paid but cheap). Quality close to Sonnet.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "claude").strip().lower()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")  # 'deepseek-chat' = V3; 'deepseek-reasoner' = R1

MOM_SYSTEM_PROMPT = """You extract DETAILED, GRANULAR, and WELL-STRUCTURED minutes of meetings (MoM) from diarized transcripts of design / construction / project meetings.

GOAL: Produce a MoM that a busy client can SKIM in 60 seconds and a project team can REFER to for ground truth. Both audiences matter.

CORE RULES:
1. BE EXHAUSTIVE. Capture every salient detail — decisions, sign-offs, commitments, material specs, vendor names, dimensions, prices, brand names, room/space names, dates. Err on the side of MORE rather than less.
2. NEVER CONSOLIDATE distinct items into one. If six spaces are signed off, list six separate sign-off entries. If five materials are specified, list five separate spec entries.
3. GROUP THEMATICALLY. Decisions and discussion notes are grouped under short thematic headings (e.g. "Execution Priority", "Material Specifications", "Coordination Challenges"). Each item within a group has a short bold "lead" phrase followed by the full detail — so a reader can scan the leads and only dive into the detail when relevant.
4. PRESERVE CONDITIONS AND ORDER. If a commitment has a follow-up ("physical sign-off happens on-site tomorrow"), keep that condition verbatim. For sequential sign-offs, preserve the order.
5. HUMANIZE TIMINGS. For action items, use natural language that matches what was actually said — "This week", "At the earliest — High priority", "By next Tuesday", or "10 June 2026" if a hard date was given. Don't invent ISO dates that weren't committed.

The transcript labels speakers as "Speaker 0", "Speaker 1", etc. The transcript may contain mixed Kannada, Hindi, Tamil, Telugu, or other Indian languages alongside English. Read all scripts directly. Produce all output in **British English** (use 'finalise', 'organise', 'colour', 'centre', 'analyse', 'metre', 'realise', 'specialised', etc. — not the American spellings). Use British conventions for dates (e.g. "10 June 2026") and punctuation. Indian English idioms ("revert", "kindly", "do the needful") are acceptable where natural.

Return ONLY valid JSON matching this schema:
{
  "summary": "4-6 sentence dense overview: context, participants, key decisions, sign-offs achieved, what remains open, critical commitments",
  "meeting_mode": "Online | Google Meet / Online | Zoom / In-person, on-site / Phone / Hybrid / null — whatever the transcript suggests",
  "speakers": [
    {"label": "Speaker 0", "inferred_name": "name or null", "role_guess": "client / designer / architect / project_manager / vendor / consultant / null", "talking_points": ["specific positions raised, decided, or committed to"]}
  ],
  "decisions": [
    {
      "theme": "Short thematic label (e.g. 'Execution Priority', 'Material Specifications', 'Drawing Reference Authority')",
      "items": [
        {"lead": "2-5 word scannable label (e.g. 'Phase 1 focus', 'Washroom ceilings', 'Single source of truth')", "detail": "Full one-sentence decision with all specifics"}
      ]
    }
  ],
  "sign_offs": [
    {"space_or_item": "specific space or design element (e.g. 'Gym design', 'Restaurant FOH layout')", "lead": "2-5 word scannable label of what's signed off (e.g. 'Gym design — verbal')", "status": "signed off / partially signed off / pending", "verbal_or_physical": "verbal / physical / pending physical", "by_whom": "Speaker N or inferred name", "conditions_or_notes": "any condition, follow-up plan, or null"}
  ],
  "materials_and_specs": [
    {"space_or_element": "specific space or component", "material_or_spec": "exact material / finish / dimension / brand", "notes": "vendor, condition, or null"}
  ],
  "discussion_notes": [
    {
      "theme": "Short thematic label (e.g. 'Coordination Challenges', 'Timeline Constraint', 'Drawing Deliverables Structure')",
      "items": ["Each bullet is one sentence covering rationale, context, or sub-issues raised — NOT decisions themselves"]
    }
  ],
  "action_items": [
    {"task": "specific action with full context", "owner": "Speaker N or inferred name or 'unassigned'", "due": "HUMANIZED timing string — 'This week' / 'At the earliest — High priority' / 'By 10 June 2026' / null", "blockers_or_dependencies": "blocker description or null"}
  ],
  "open_questions": [
    {"topic": "2-5 word topic label (e.g. 'Vendor Selection & Lead Times', 'Electrical Coordination')", "what_is_pending": "one-sentence description of what's blocked and what's needed to resolve it"}
  ],
  "topics_discussed": ["short topic labels covering everything the meeting touched on, in chronological order"]
}

Be faithful — never invent items that weren't discussed. But if you're hesitating about whether something is salient enough to include, INCLUDE IT. Your bias should be toward more detail, not less. Empty sections return empty arrays."""


def stitch_audio(audio_paths: list[Path], out_path: Path) -> Path:
    """Concatenate multiple audio files using ffmpeg.

    Fast path: concat demuxer with `-c copy` (no re-encoding, takes seconds).
    Works when all inputs share the same codec/sample rate — typically true for
    back-to-back recordings from the same device. Output extension is forced to
    match the input codec (.m4a for mp4/aac, otherwise .mp3 etc).

    Slow fallback: concat filter with re-encode to mp3. Used if the fast path
    fails (e.g. mixed input formats).
    """
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError(
            "ffmpeg not found. Install it (e.g. `brew install ffmpeg` locally, "
            "or include it in your Docker image for deploys)."
        )

    n = len(audio_paths)

    # Pick output extension based on first input. mp4/m4a → m4a; everything else → mp3.
    first_ext = audio_paths[0].suffix.lower()
    if first_ext in (".mp4", ".m4a", ".aac"):
        fast_out = out_path.with_suffix(".m4a")
    else:
        fast_out = out_path.with_suffix(first_ext or ".mp3")

    # Build a concat list file for the demuxer.
    list_file = out_path.parent / "_concat_list.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in audio_paths))

    fast_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy", "-vn",
        str(fast_out),
    ]
    print(f"→ Stitching {n} files (fast path, no re-encode)...", file=sys.stderr)
    result = subprocess.run(fast_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        list_file.unlink(missing_ok=True)
        return fast_out

    # Fallback: re-encode via concat filter.
    print(f"  fast path failed ({result.stderr.strip().splitlines()[-1] if result.stderr else 'unknown'})", file=sys.stderr)
    print("  falling back to re-encode (slower)...", file=sys.stderr)
    list_file.unlink(missing_ok=True)

    inputs = []
    for p in audio_paths:
        inputs.extend(["-i", str(p)])
    filter_expr = "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[out]"
    slow_out = out_path.with_suffix(".mp3")
    slow_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_expr,
        "-map", "[out]",
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(slow_out),
    ]
    subprocess.run(slow_cmd, check=True)
    return slow_out


def transcribe(audio_path: Path, timeout_seconds: float = 1800.0) -> dict:
    """Send audio to Deepgram, return the raw response dict.

    timeout_seconds applies to read/write/pool. Long meetings take Deepgram a
    few to many minutes server-side; default is 30 min, override via the CLI.
    """
    key = os.environ["DEEPGRAM_API_KEY"]
    client = DeepgramClient(key)

    with open(audio_path, "rb") as f:
        payload: FileSource = {"buffer": f.read()}

    options = PrerecordedOptions(
        model="nova-3",
        diarize=True,
        punctuate=True,
        paragraphs=True,
        smart_format=True,
        language="en",
    )

    print(
        f"→ Sending {audio_path.name} to Deepgram (timeout {int(timeout_seconds)}s)...",
        file=sys.stderr,
    )
    timeout = httpx.Timeout(timeout_seconds, connect=15.0)
    response = client.listen.rest.v("1").transcribe_file(payload, options, timeout=timeout)
    return response.to_dict()


def format_transcript(dg_response: dict) -> str:
    """Turn Deepgram's word-level output into 'Speaker N: ...' paragraphs."""
    words = dg_response["results"]["channels"][0]["alternatives"][0]["words"]
    if not words:
        return ""

    lines = []
    current_speaker = words[0].get("speaker", 0)
    current_words = []

    for w in words:
        spk = w.get("speaker", 0)
        token = w.get("punctuated_word") or w["word"]
        if spk != current_speaker:
            lines.append(f"Speaker {current_speaker}: {' '.join(current_words)}")
            current_speaker = spk
            current_words = [token]
        else:
            current_words.append(token)

    if current_words:
        lines.append(f"Speaker {current_speaker}: {' '.join(current_words)}")

    return "\n\n".join(lines)


def _strip_json_fences(text: str) -> str:
    """Strip markdown fences a model may have wrapped the JSON in."""
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence (with optional language tag) and the closing fence.
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if text.startswith("json\n"):
            text = text[5:]
    return text


def _claude_stream_json(system_prompt: str, user_content: str, max_tokens: int, on_progress=None) -> dict:
    """Stream a JSON response from Claude and parse it."""
    client = Anthropic()
    pieces = []
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        for text in stream.text_stream:
            pieces.append(text)
            if on_progress:
                try:
                    on_progress(sum(len(p) for p in pieces))
                except Exception:
                    pass
    return json.loads(_strip_json_fences("".join(pieces)))


def _gemini_stream_json(system_prompt: str, user_content: str, max_tokens: int, on_progress=None) -> dict:
    """Stream a JSON response from Gemini Flash and parse it.

    Uses response_mime_type='application/json' so Gemini constrains the output
    to a JSON value (no markdown fences). System instruction holds the schema.
    Retries on 503 / overload errors with exponential backoff — Google's free-tier
    capacity is bursty and transient 503s are common.
    """
    try:
        from google import genai
        from google.genai import types as gtypes
        from google.genai import errors as gerrors
    except ImportError as e:
        raise RuntimeError(
            "google-genai not installed. `pip install google-genai` first."
        ) from e
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set on server (LLM_PROVIDER=gemini requires it).")
    client = genai.Client(api_key=api_key)
    cfg = gtypes.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        max_output_tokens=max_tokens,
        temperature=0.2,
    )

    # Retry transient errors. 503 (overload) and 429 (rate limit) get a few
    # tries with backoff; everything else fails fast.
    import time as _time
    delays = [3, 8, 20, 45]  # ~75 seconds of total backoff budget
    last_err = None
    for attempt, sleep_s in enumerate([0] + delays):
        if sleep_s:
            print(f"  Gemini retry in {sleep_s}s (attempt {attempt+1})...", file=sys.stderr)
            _time.sleep(sleep_s)
        try:
            pieces = []
            stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=user_content,
                config=cfg,
            )
            for chunk in stream:
                try:
                    t = chunk.text
                except Exception:
                    t = None
                if not t:
                    continue
                pieces.append(t)
                if on_progress:
                    try:
                        on_progress(sum(len(p) for p in pieces))
                    except Exception:
                        pass
            return json.loads(_strip_json_fences("".join(pieces)))
        except gerrors.ClientError as e:
            # 429 RESOURCE_EXHAUSTED is retryable; everything else (e.g. 400 bad request) is not.
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if str(code) != "429":
                raise
            last_err = e
        except gerrors.ServerError as e:
            # 5xx including 503 UNAVAILABLE
            last_err = e
        except Exception as e:
            # Network blips / connection resets — also retry
            last_err = e
    # Out of retries
    raise last_err


def _deepseek_stream_json(system_prompt: str, user_content: str, max_tokens: int, on_progress=None) -> dict:
    """Stream a JSON response from DeepSeek (V3 by default) and parse it.

    DeepSeek's API is OpenAI-compatible — we use the `openai` SDK pointed at
    https://api.deepseek.com. The `response_format={'type': 'json_object'}`
    parameter constrains output to a JSON value (similar to Gemini's
    response_mime_type). When using JSON mode, the prompt must mention "json"
    somewhere — our system prompt already does.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai SDK not installed. `pip install openai` first."
        ) from e
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set on server (LLM_PROVIDER=deepseek requires it).")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    pieces = []
    stream = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
        stream=True,
        response_format={"type": "json_object"},
    )
    for chunk in stream:
        try:
            t = chunk.choices[0].delta.content
        except (AttributeError, IndexError):
            t = None
        if not t:
            continue
        pieces.append(t)
        if on_progress:
            try:
                on_progress(sum(len(p) for p in pieces))
            except Exception:
                pass
    return json.loads(_strip_json_fences("".join(pieces)))


def _llm_stream_json(system_prompt: str, user_content: str, max_tokens: int, on_progress=None) -> dict:
    """Dispatch to the configured LLM provider."""
    if LLM_PROVIDER == "gemini":
        return _gemini_stream_json(system_prompt, user_content, max_tokens, on_progress)
    if LLM_PROVIDER == "deepseek":
        return _deepseek_stream_json(system_prompt, user_content, max_tokens, on_progress)
    return _claude_stream_json(system_prompt, user_content, max_tokens, on_progress)


def extract_mom(transcript: str, on_progress=None) -> dict:
    """Stream MoM extraction from the configured LLM. on_progress(chars_so_far) called as tokens arrive."""
    print(f"→ Extracting MoM with {LLM_PROVIDER.title()} (streaming)...", file=sys.stderr)
    return _llm_stream_json(
        MOM_SYSTEM_PROMPT,
        f"Transcript:\n\n{transcript}",
        max_tokens=16384,
        on_progress=on_progress,
    )


CONCISE_SYSTEM_PROMPT = """You receive a DETAILED minutes-of-meeting (MoM) JSON object. Produce an AGGRESSIVELY CONCISE version in the SAME JSON schema, by (a) tightening prose AND (b) MERGING items that state the same fact in different ways.

ABSOLUTE RULE — NO DISTINCT FACTS DROPPED.

Every distinct fact in the detailed input MUST appear in the concise output, in some form. Identical or near-identical facts repeated across multiple items SHOULD be consolidated. But unique facts MUST be retained.

BEFORE you decide to drop or omit any detailed item, run this test: "Can I point to the specific concise item that already contains this fact?" If no — keep the detailed item (compressed). If yes — merging is fine.

ANTI-EXAMPLE (do NOT do this):
- Detailed has: "FOH electrical — architect governs", "FOH plumbing — architect governs", "Lighting designer scope — fixtures and intensity only"
- Wrong concise: "FOH — architect governs" (drops the lighting designer scope fact entirely — these are SEPARATE facts: override hierarchy vs scope demarcation)
- Right concise: "FOH electrical and plumbing — architect governs" + "Lighting designer scope — fixtures and intensity only"

TARGET: roughly 65-75% the length of the detailed input. Achieve this through (a) tight prose, (b) merging of truly-identical-fact items (like the sign-offs case below), and (c) NOT through dropping unique facts.

WHEN TO MERGE — examples (do this enthusiastically):

- Detailed has 8 sign-offs, each "verbally signed off by Vasu, physical on-site next day", differing only in space name → 1 sign-off entry listing all 8 spaces with the shared conditions. e.g. {"space_or_item": "Gym, Restaurant, Common Washrooms, GF Lounge, Badminton Court, Locker Room", "lead": "Six clubhouse spaces — verbal sign-off", "status": "signed off", "verbal_or_physical": "verbal", "by_whom": "Vasu", "conditions_or_notes": "Physical sign-off to happen on-site the following day. No further design changes."}
- Detailed has 3 decisions all under "Drawing Authority" saying variations of "architect overrides consultant on FOH for X, Y, Z" → 1 consolidated bullet listing all overrides.
- Detailed has 3 action items all for architect, due "This week", all about issuing some drawing → 1 action item bundling them.
- Detailed has 4 discussion_notes bullets explaining the same coordination gap from different angles → 1 tight bullet.
- Detailed has 5 materials_and_specs entries all describing the gym (flooring, walls, ceiling, lighting, signage) → optionally combine related ones into a per-space line, OR keep separate if each spec is meaningfully distinct. Use judgement: if a reader needs to find the flooring quickly, keep it as its own row.

WHEN TO KEEP SEPARATE — examples:

- "Gym flooring: rubber tiles" and "Restaurant flooring: ceramic tiles" → KEEP separate. Different materials, different spaces.
- "Architect to share GFC by Friday" and "Architect to share material samples by Monday" → KEEP separate. Different deliverables, different timing.
- "HVAC layout not finalised — blocks MEP" and "Brewery HVAC pending — blocks brewery GFC" → KEEP separate (different scopes), though both can be expressed tightly.

PROSE-LEVEL COMPRESSION (apply throughout):
- Cut filler words ("really", "actually", "the fact that", "in terms of"), soft framing ("it was discussed that", "the team agreed to"), and redundant phrasing.
- Use abbreviations once established (FOH/BOH after first mention; GFC after first mention).
- "₹4 crores" not "approximately ₹4 crores costed at around ₹4 crores".
- Example: "The lobby area has a tighter budget; options include MS laminate, TMS plate boxing, or wooden Formica pasting; the design needs to land on the most economical option" → "Lobby — tight budget; options: MS laminate, TMS plate boxing, or wooden Formica pasting."

SECTION-BY-SECTION RULES:

- summary: 2-3 dense sentences, not 4-6.
- decisions: merge ONLY when items state literally the same fact (e.g. two "FOH X — architect governs" items). Keep items that state distinct facts even if related. Expect ~85-90% of detailed item count.
- sign_offs: aggressive merging when multiple sign-offs share status + verbal_or_physical + by_whom + conditions (only the space differs). Often 8 items → 2-3. This is the section where merging delivers the most savings.
- materials_and_specs: keep all unique element/material pairings. Only merge if multiple entries describe IDENTICAL spec for IDENTICAL element. Expect ~95% of detailed count.
- discussion_notes: merge bullets that re-explain the same coordination gap or rationale. Often 5 → 2-3.
- action_items: merge ONLY when same owner + same timing + same deliverable family (e.g. multiple drawings of the same type). Keep distinct deliverables separate. Expect ~85-90% of detailed count.
- open_questions: each is usually distinct — minimal merging, tighten prose.
- topics_discussed: KEEP ALL ENTRIES. They are short labels — they cost nothing to retain and serve as the table of contents.
- speakers: keep all speakers with labels/names/roles. For talking_points, drop those fully restated under decisions/sign-offs/action_items, keep only speaker-specific positions. Tighten what remains.
- meeting_mode: keep as-is.

British English throughout. Same JSON schema.

Return ONLY the concise JSON object. No commentary, no markdown fences."""


def condense_to_concise(detailed_mom: dict, on_progress=None) -> dict:
    """Take a detailed MoM and produce a concise version (no facts dropped, tighter prose)."""
    print(f"→ Condensing to concise MoM with {LLM_PROVIDER.title()} (streaming)...", file=sys.stderr)
    return _llm_stream_json(
        CONCISE_SYSTEM_PROMPT,
        f"Detailed MoM JSON:\n\n{json.dumps(detailed_mom, ensure_ascii=False)}",
        max_tokens=16384,
        on_progress=on_progress,
    )


def render_markdown(mom: dict) -> str:
    """Pretty-print MoM JSON as a readable markdown doc.

    Output structure mirrors a professional MoM: thematic groups with bold
    scannable leads, distinct decisions vs discussion notes, and tables for
    action items / open questions."""
    out = ["# Minutes of meeting", ""]

    meta_bits = []
    if mom.get("meeting_mode"):
        meta_bits.append(f"**Mode:** {mom['meeting_mode']}")
    if meta_bits:
        out.extend(meta_bits)
        out.append("")

    out.append("## Purpose / Summary")
    out.append("")
    out.append(mom.get("summary", "—"))
    out.append("")

    # Decisions — thematically grouped with bold leads
    if mom.get("decisions"):
        out.append("## Decisions agreed")
        out.append("")
        for group in mom["decisions"]:
            out.append(f"### {group.get('theme', 'General')}")
            out.append("")
            for item in group.get("items", []):
                lead = item.get("lead", "").strip()
                detail = item.get("detail", "").strip()
                if lead and detail:
                    out.append(f"- **{lead}**: {detail}")
                elif lead:
                    out.append(f"- **{lead}**")
                elif detail:
                    out.append(f"- {detail}")
            out.append("")

    # Sign-offs — flat, but each with a scannable lead
    if mom.get("sign_offs"):
        out.append("## Sign-offs")
        out.append("")
        for s in mom["sign_offs"]:
            lead = s.get("lead") or s.get("space_or_item", "—")
            tail_bits = []
            if s.get("status"): tail_bits.append(s["status"])
            if s.get("verbal_or_physical"): tail_bits.append(s["verbal_or_physical"])
            if s.get("by_whom"): tail_bits.append(f"by {s['by_whom']}")
            line = f"- **{lead}**"
            if tail_bits:
                line += f" — {', '.join(tail_bits)}"
            if s.get("conditions_or_notes"):
                line += f". {s['conditions_or_notes']}"
            out.append(line)
        out.append("")

    # Materials & specs — flat per-element
    if mom.get("materials_and_specs"):
        out.append("## Materials and specs")
        out.append("")
        for m in mom["materials_and_specs"]:
            elem = m.get("space_or_element", "—")
            spec = m.get("material_or_spec", "")
            line = f"- **{elem}**: {spec}"
            if m.get("notes"):
                line += f" — {m['notes']}"
            out.append(line)
        out.append("")

    # Discussion notes — thematically grouped, plain bullets (no leads)
    if mom.get("discussion_notes"):
        out.append("## Design rationale & discussion notes")
        out.append("")
        for group in mom["discussion_notes"]:
            out.append(f"### {group.get('theme', 'General')}")
            out.append("")
            for it in group.get("items", []):
                out.append(f"- {it}")
            out.append("")

    # Action items — table for scanability
    if mom.get("action_items"):
        out.append("## Action items")
        out.append("")
        out.append("| Owner | Action | Timing |")
        out.append("|---|---|---|")
        for a in mom["action_items"]:
            owner = (a.get("owner") or "unassigned").replace("|", "\\|")
            task = (a.get("task") or "").replace("|", "\\|").replace("\n", " ")
            due = (a.get("due") or "—").replace("|", "\\|")
            if a.get("blockers_or_dependencies"):
                task += f" — blocked on: {a['blockers_or_dependencies']}"
            out.append(f"| {owner} | {task} | {due} |")
        out.append("")

    # Open questions — table with topic + what's pending
    if mom.get("open_questions"):
        out.append("## Open items & pending clarifications")
        out.append("")
        out.append("| Topic | What is pending / needed to resolve |")
        out.append("|---|---|")
        for q in mom["open_questions"]:
            topic = (q.get("topic") or "—").replace("|", "\\|")
            pending = (q.get("what_is_pending") or "").replace("|", "\\|").replace("\n", " ")
            out.append(f"| **{topic}** | {pending} |")
        out.append("")

    if mom.get("topics_discussed"):
        out.append("## Topics covered (overview)")
        out.append("")
        out.extend(f"- {t}" for t in mom["topics_discussed"])
        out.append("")

    if mom.get("speakers"):
        out.append("## Participants")
        out.append("")
        for s in mom["speakers"]:
            name = s.get("inferred_name") or s["label"]
            role = f" — *{s['role_guess']}*" if s.get("role_guess") else ""
            out.append(f"### {name}{role}")
            out.append("")
            for tp in s.get("talking_points", []):
                out.append(f"- {tp}")
            out.append("")

    out.append("---")
    out.append("")
    out.append("*Please review the points above and revert with any corrections or additions within 3 working days. In the absence of comments, these minutes will be taken as accepted.*")

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Process meeting audio → transcript + MoM")
    parser.add_argument(
        "audio", type=Path, nargs="+",
        help="Path(s) to audio file(s). Multiple files are stitched in the given order.",
    )
    parser.add_argument("--out", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument(
        "--name", type=str, default=None,
        help="Output file stem. Defaults to first input's stem (with _combined if multiple).",
    )
    parser.add_argument(
        "--timeout", type=float, default=1800.0,
        help="Deepgram HTTP timeout in seconds (default 1800 = 30 min). Bump for very long meetings.",
    )
    parser.add_argument(
        "--no-concise", action="store_true",
        help="Skip the concise second pass. By default both _detailed and _concise versions are produced.",
    )
    args = parser.parse_args()

    for p in args.audio:
        if not p.exists():
            sys.exit(f"Audio file not found: {p}")

    for var in ("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"Missing {var} in environment. Copy .env.example to .env and fill it in.")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.name:
        stem = args.name
    elif len(args.audio) == 1:
        stem = args.audio[0].stem
    else:
        stem = f"{args.audio[0].stem}_combined"

    # Stitch if needed, then run the same pipeline.
    with tempfile.TemporaryDirectory() as tmp:
        if len(args.audio) > 1:
            stitched = stitch_audio(args.audio, Path(tmp) / stem)
            # Also save the stitched audio next to outputs so it can be re-used.
            kept = args.out / f"{stem}{stitched.suffix}"
            shutil.copy2(stitched, kept)
            print(f"✓ Stitched:   {kept}", file=sys.stderr)
            audio_to_process = stitched
        else:
            audio_to_process = args.audio[0]

        dg_response = transcribe(audio_to_process, timeout_seconds=args.timeout)

    (args.out / f"{stem}.deepgram.json").write_text(json.dumps(dg_response, indent=2))

    transcript = format_transcript(dg_response)
    (args.out / f"{stem}.transcript.txt").write_text(transcript)
    print(f"✓ Transcript: {args.out / f'{stem}.transcript.txt'}", file=sys.stderr)

    mom = extract_mom(transcript)
    (args.out / f"{stem}.mom_detailed.json").write_text(json.dumps(mom, indent=2, ensure_ascii=False))
    (args.out / f"{stem}.mom_detailed.md").write_text(render_markdown(mom))
    print(f"✓ MoM (detailed): {args.out / f'{stem}.mom_detailed.md'}", file=sys.stderr)

    if not args.no_concise:
        concise = condense_to_concise(mom)
        (args.out / f"{stem}.mom_concise.json").write_text(json.dumps(concise, indent=2, ensure_ascii=False))
        (args.out / f"{stem}.mom_concise.md").write_text(render_markdown(concise))
        print(f"✓ MoM (concise):  {args.out / f'{stem}.mom_concise.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
