"""
Parallel pipeline using Sarvam Saarika ASR instead of Deepgram.

Designed for meetings with mixed Indian languages + English (Kannada, Hindi,
etc.) which Deepgram nova-3 can't transcribe. Produces both the faithful
multi-script original transcript AND an English translation of it.

Outputs (in --out, default ./output/):
    <stem>.sarvam.json              raw Sarvam job result
    <stem>.transcript.txt           original diarized transcript (mixed script)
    <stem>.transcript.english.txt   English translation (preserves Speaker N: structure)
    <stem>.mom.json                 structured MoM extracted by Claude
    <stem>.mom.md                   readable MoM

Usage:
    python process_sarvam.py meeting.mp3
    python process_sarvam.py part1.mp4 part2.mp4 part3.mp4

Notes:
- Sarvam's batch API is async — the script submits the job and polls until done.
  Long meetings take several minutes server-side; this is normal.
- I wrote this against my best memory of the sarvamai SDK shape; if the first
  run errors on an SDK call, the fix is usually a one-line tweak — paste the
  error and we'll adjust.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

import process as base

load_dotenv()

CLAUDE_MODEL = "claude-sonnet-4-6"

# Use the canonical MoM prompt from process.py — single source of truth for both pipelines.
MOM_SYSTEM_PROMPT = base.MOM_SYSTEM_PROMPT

TRANSLATE_SYSTEM_PROMPT = """You are a translator. You will receive a diarized meeting transcript that may contain mixed Kannada, Hindi, Tamil, Telugu, or other Indian languages alongside English.

Translate the entire transcript into natural, fluent English. Rules:
- Preserve the 'Speaker N:' line structure exactly.
- Translate Indian-language portions to English.
- Leave English portions as-is.
- Do not summarise, paraphrase aggressively, or add commentary.
- Do not add a preamble or footer. Output only the translated transcript."""


# ---------- Sarvam transcription ----------

def transcribe_sarvam(audio_path: Path, poll_interval: int = 5) -> dict:
    """Send audio to Sarvam Saarika batch API with diarization. Block until done."""
    from sarvamai import SarvamAI

    key = os.environ["SARVAM_API_KEY"]
    client = SarvamAI(api_subscription_key=key)

    print(f"→ Submitting {audio_path.name} to Sarvam Saarika (batch)...", file=sys.stderr)

    job = client.speech_to_text_job.create_job(
        model="saarika:v2.5",
        language_code="unknown",       # auto-detect mixed languages
        with_diarization=True,
        with_timestamps=True,
    )
    job.upload_files(file_paths=[str(audio_path)])
    job.start()

    print(f"→ Polling Sarvam job {job.job_id} (long audio can take several minutes)...", file=sys.stderr)
    # The SDK's default timeout is 600s; bump for long meetings (1 hour cap).
    job.wait_until_complete(poll_interval=poll_interval, timeout=3600)

    if not job.is_successful():
        try:
            details = job.get_file_results()
        except Exception:
            details = None
        raise RuntimeError(f"Sarvam job failed. Details: {details}")

    # Download the result JSON(s) to a temp dir and load the first one.
    with tempfile.TemporaryDirectory() as out_dir:
        job.download_outputs(out_dir)
        out_files = sorted(Path(out_dir).rglob("*.json"))
        if not out_files:
            all_files = list(Path(out_dir).rglob("*"))
            raise RuntimeError(f"No JSON in Sarvam output. Got: {[str(p) for p in all_files]}")
        return json.loads(out_files[0].read_text())


def format_sarvam_transcript(sarvam_response: dict) -> str:
    """Turn Sarvam's diarized output into 'Speaker N: ...' paragraphs.

    Sarvam's response shape (best guess): contains a list of segments/diarized_entries
    with fields like {speaker_id, transcript, start_time, end_time}.
    """
    # Try common keys defensively
    segments = (
        sarvam_response.get("diarized_transcript", {}).get("entries")
        or sarvam_response.get("segments")
        or sarvam_response.get("diarized_entries")
        or sarvam_response.get("transcript_segments")
    )

    if not segments:
        # Last resort: just return any 'transcript' field as one block
        flat = sarvam_response.get("transcript")
        if flat:
            return f"Speaker 0: {flat}"
        return ""

    lines = []
    current_speaker = None
    current_text_parts = []
    for seg in segments:
        spk = seg.get("speaker_id") or seg.get("speaker") or 0
        text = seg.get("transcript") or seg.get("text") or ""
        if spk != current_speaker:
            if current_text_parts:
                lines.append(f"Speaker {current_speaker}: {' '.join(current_text_parts).strip()}")
            current_speaker = spk
            current_text_parts = [text]
        else:
            current_text_parts.append(text)
    if current_text_parts:
        lines.append(f"Speaker {current_speaker}: {' '.join(current_text_parts).strip()}")

    return "\n\n".join(lines)


# ---------- Claude steps ----------

def translate_to_english(transcript: str, on_progress=None) -> str:
    """Stream translation. on_progress(chars_so_far) called as tokens arrive."""
    client = Anthropic()
    print("→ Translating transcript to English with Claude (streaming)...", file=sys.stderr)
    pieces = []
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=32000,
        system=TRANSLATE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript}],
    ) as stream:
        for text in stream.text_stream:
            pieces.append(text)
            if on_progress:
                try:
                    on_progress(sum(len(p) for p in pieces))
                except Exception:
                    pass
    return "".join(pieces).strip()


def extract_mom_multilingual(transcript: str, on_progress=None) -> dict:
    """Thin alias for base.extract_mom. The shared prompt already handles
    multilingual input — keep this name for callers that already use it."""
    return base.extract_mom(transcript, on_progress=on_progress)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Audio → transcript + English translation + MoM (Sarvam Saarika)")
    parser.add_argument("audio", type=Path, nargs="+", help="Path(s) to audio file(s)")
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--name", type=str, default=None,
                        help="Output file stem. Defaults to first input's stem.")
    parser.add_argument("--poll-interval", type=int, default=10,
                        help="Seconds between Sarvam job status polls (default 10).")
    parser.add_argument("--no-concise", action="store_true",
                        help="Skip the concise second pass. Default: produce both _detailed and _concise.")
    args = parser.parse_args()

    for p in args.audio:
        if not p.exists():
            sys.exit(f"Audio file not found: {p}")

    for var in ("SARVAM_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            sys.exit(f"Missing {var} in environment. Update your .env file.")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.name:
        stem = args.name
    elif len(args.audio) == 1:
        stem = args.audio[0].stem
    else:
        stem = f"{args.audio[0].stem}_combined"

    # Reuse process.py's stitch logic so multi-file input works the same way.
    with tempfile.TemporaryDirectory() as tmp:
        if len(args.audio) > 1:
            stitched = base.stitch_audio(args.audio, Path(tmp) / stem)
            kept = args.out / f"{stem}{stitched.suffix}"
            shutil.copy2(stitched, kept)
            print(f"✓ Stitched:    {kept}", file=sys.stderr)
            audio_to_process = stitched
        else:
            audio_to_process = args.audio[0]

        sarvam_response = transcribe_sarvam(audio_to_process, poll_interval=args.poll_interval)

    (args.out / f"{stem}.sarvam.json").write_text(json.dumps(sarvam_response, indent=2, ensure_ascii=False))

    # Original (faithful, possibly mixed-script) transcript — sent directly to Claude.
    # No separate translate-to-English step: Claude reads Kannada/Hindi natively and
    # produces all MoM output in English regardless of source language.
    transcript = format_sarvam_transcript(sarvam_response)
    (args.out / f"{stem}.transcript.txt").write_text(transcript)
    print(f"✓ Transcript:  {args.out / f'{stem}.transcript.txt'}", file=sys.stderr)

    # Detailed MoM
    mom = base.extract_mom(transcript)
    (args.out / f"{stem}.mom_detailed.json").write_text(json.dumps(mom, indent=2, ensure_ascii=False))
    (args.out / f"{stem}.mom_detailed.md").write_text(base.render_markdown(mom))
    print(f"✓ MoM (detailed): {args.out / f'{stem}.mom_detailed.md'}", file=sys.stderr)

    # Concise MoM (condensed from detailed — no facts dropped, tighter prose)
    if not args.no_concise:
        concise = base.condense_to_concise(mom)
        (args.out / f"{stem}.mom_concise.json").write_text(json.dumps(concise, indent=2, ensure_ascii=False))
        (args.out / f"{stem}.mom_concise.md").write_text(base.render_markdown(concise))
        print(f"✓ MoM (concise):  {args.out / f'{stem}.mom_concise.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
