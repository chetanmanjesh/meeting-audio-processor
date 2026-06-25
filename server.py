"""
Web app: record meeting audio in the browser → transcript + MoM.

Single-file FastAPI app. Serves the record UI at `/` and runs the processing
pipeline as a background thread. Uses an in-memory job store (jobs are lost on
restart — fine for a free-tier MVP).

Run locally:
    uvicorn server:app --reload --port 8000

Deploy on Render: see render.yaml in this directory.

Environment:
    DEEPGRAM_API_KEY        required if user selects Deepgram
    SARVAM_API_KEY          required if user selects Sarvam
    ANTHROPIC_API_KEY       required for both (MoM generation)
    ACCESS_TOKEN            optional. If set, every request needs ?token=<value>.

    R2_ACCOUNT_ID           required. Cloudflare R2 account ID.
    R2_ACCESS_KEY_ID        required. R2 API token access key.
    R2_SECRET_ACCESS_KEY    required. R2 API token secret.
    R2_BUCKET_NAME          required. The R2 bucket holding uploaded audio.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import boto3
import imageio_ffmpeg
from botocore.client import Config as BotoConfig
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

import process as base
import process_sarvam as sarvam_pipeline
from process_template import auto_meta, fill_template_to_bytes

DEFAULT_TEMPLATE = Path(__file__).parent / "template.docx"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

load_dotenv()

app = FastAPI(title="Meeting audio → MoM")

# In-memory job store. Lost on restart; that's OK for an MVP free-tier deploy.
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")


def _check_token(token: Optional[str]) -> None:
    if ACCESS_TOKEN and token != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


# ---------- R2 (Cloudflare object storage, S3-compatible) ----------

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
R2_ENABLED = all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME])


def _r2_client():
    if not R2_ENABLED:
        return None
    return boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=BotoConfig(signature_version="s3v4"),
    )


def _require_r2():
    if not R2_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="R2 storage not configured on server (set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME).",
        )


def _r2_download(key: str, local_path: Path) -> None:
    """Download an R2 object to a local file."""
    s3 = _r2_client()
    s3.download_file(R2_BUCKET_NAME, key, str(local_path))


def _r2_delete(key: str) -> None:
    """Best-effort delete; swallow errors."""
    try:
        s3 = _r2_client()
        if s3:
            s3.delete_object(Bucket=R2_BUCKET_NAME, Key=key)
    except Exception:
        pass


def _r2_put_json(key: str, data) -> None:
    """Best-effort write JSON to R2 (used to cache intermediate outputs for retry)."""
    try:
        s3 = _r2_client()
        if not s3:
            return
        s3.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=key,
            Body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception:
        pass


def _r2_put_text(key: str, text: str) -> None:
    try:
        s3 = _r2_client()
        if not s3:
            return
        s3.put_object(
            Bucket=R2_BUCKET_NAME, Key=key,
            Body=text.encode("utf-8"), ContentType="text/plain; charset=utf-8",
        )
    except Exception:
        pass


def _r2_get_json(key: str):
    """Read JSON from R2, return None if not found."""
    try:
        s3 = _r2_client()
        if not s3:
            return None
        obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def _r2_get_text(key: str):
    try:
        s3 = _r2_client()
        if not s3:
            return None
        obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
        return obj["Body"].read().decode("utf-8")
    except Exception:
        return None


def _cache_keys(job_id: str) -> dict:
    """The R2 keys we use to cache intermediates per job for resumability."""
    return {
        "transcript": f"cache/{job_id}/transcript.txt",
        "detailed":   f"cache/{job_id}/detailed.json",
        "concise":    f"cache/{job_id}/concise.json",
        "samples":    f"cache/{job_id}/speaker_samples.json",
    }


# ---------- Speaker sample extraction (for the Identify Speakers feature) ----------

def _r2_put_file(key: str, local_path: Path, content_type: str = "application/octet-stream") -> bool:
    try:
        s3 = _r2_client()
        if not s3:
            return False
        s3.upload_file(str(local_path), R2_BUCKET_NAME, key, ExtraArgs={"ContentType": content_type})
        return True
    except Exception:
        return False


def _extract_speaker_samples(transcript_response, provider: str, audio_path: Path, job_id: str) -> dict:
    """Find the longest contiguous run per speaker, clip ~8s from it, upload to R2.

    Returns: {label: {key, start_s, end_s, duration_s}} keyed by 'Speaker N'.
    Best-effort — failures during clipping are swallowed; the feature just won't
    be available for speakers we couldn't sample.
    """
    # Gather (speaker_id, start_s, end_s) for each contiguous run.
    runs: list[tuple] = []
    if provider == "deepgram":
        try:
            words = (transcript_response.get("results", {})
                                       .get("channels", [{}])[0]
                                       .get("alternatives", [{}])[0]
                                       .get("words", []) or [])
        except Exception:
            words = []
        if words:
            cur_spk = words[0].get("speaker", 0)
            cur_start = float(words[0].get("start", 0.0))
            cur_end = float(words[0].get("end", cur_start))
            for w in words[1:]:
                spk = w.get("speaker", 0)
                if spk == cur_spk:
                    cur_end = float(w.get("end", cur_end))
                else:
                    runs.append((cur_spk, cur_start, cur_end))
                    cur_spk = spk
                    cur_start = float(w.get("start", 0.0))
                    cur_end = float(w.get("end", cur_start))
            runs.append((cur_spk, cur_start, cur_end))
    else:  # sarvam
        def _find_segs(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, list) and v and isinstance(v[0], dict) and "start_time_seconds" in v[0]:
                        return v
                    r = _find_segs(v)
                    if r:
                        return r
            return None
        segs = _find_segs(transcript_response) or []
        for s in segs:
            try:
                runs.append((
                    str(s.get("speaker_id", 0)),
                    float(s.get("start_time_seconds", 0.0)),
                    float(s.get("end_time_seconds", 0.0)),
                ))
            except Exception:
                continue

    # Pick the longest run per speaker.
    best: dict = {}
    for spk, start, end in runs:
        duration = end - start
        if spk not in best or duration > (best[spk][1] - best[spk][0]):
            best[spk] = (start, end)

    samples: dict = {}
    for spk, (start, end) in best.items():
        end = min(end, start + 8.0)  # cap clip length
        if end - start < 1.5:  # too short to be useful
            continue
        sample_key = f"samples/{job_id}/speaker_{spk}.mp3"
        clip_fd, clip_name = tempfile.mkstemp(suffix=".mp3")
        os.close(clip_fd)
        clip_path = Path(clip_name)
        try:
            subprocess.run(
                [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                 "-i", str(audio_path),
                 "-vn", "-ac", "1", "-b:a", "48k",
                 str(clip_path)],
                check=True, timeout=30,
            )
            if _r2_put_file(sample_key, clip_path, content_type="audio/mpeg"):
                samples[f"Speaker {spk}"] = {
                    "key": sample_key,
                    "start_s": round(start, 2),
                    "end_s": round(end, 2),
                    "duration_s": round(end - start, 2),
                }
        except Exception:
            pass
        finally:
            clip_path.unlink(missing_ok=True)

    return samples


# ---------- Speaker-name mapping helpers ----------

def _apply_mapping_to_str(s: str, mapping: dict) -> str:
    if not s or not mapping:
        return s
    # Replace in length order (longer labels first) so "Speaker 10" doesn't get
    # half-replaced by a "Speaker 1" rule.
    out = s
    for label in sorted(mapping.keys(), key=len, reverse=True):
        out = out.replace(label, mapping[label])
    return out


def _apply_mapping_to_obj(obj, mapping: dict):
    if isinstance(obj, str):
        return _apply_mapping_to_str(obj, mapping)
    if isinstance(obj, list):
        return [_apply_mapping_to_obj(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: _apply_mapping_to_obj(v, mapping) for k, v in obj.items()}
    return obj


def stitch_files(audio_paths: list[Path], out_stem: Path) -> Path:
    """Stitch multiple audio files via ffmpeg. Fast path (no re-encode) first,
    fallback to re-encode for mixed formats. Returns the stitched file path."""
    if len(audio_paths) == 1:
        return audio_paths[0]

    list_file = out_stem.parent / f"{out_stem.name}_concat.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in audio_paths))

    first_ext = audio_paths[0].suffix.lower()
    if first_ext in (".mp4", ".m4a", ".aac"):
        fast_out = out_stem.with_suffix(".m4a")
    else:
        fast_out = out_stem.with_suffix(first_ext or ".mp3")

    fast_cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy", "-vn",
        str(fast_out),
    ]
    result = subprocess.run(fast_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        list_file.unlink(missing_ok=True)
        return fast_out

    # Re-encode fallback for mixed formats
    inputs = []
    for p in audio_paths:
        inputs.extend(["-i", str(p)])
    n = len(audio_paths)
    filter_expr = "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[out]"
    slow_out = out_stem.with_suffix(".mp3")
    slow_cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_expr,
        "-map", "[out]",
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(slow_out),
    ]
    subprocess.run(slow_cmd, check=True)
    list_file.unlink(missing_ok=True)
    return slow_out


# ---------- background processing ----------

def _process_job(job_id: str, r2_keys: list[str], filenames: list[str], provider: str) -> None:
    """Run the full pipeline. Downloads each R2 key to a local temp file,
    stitches if needed, transcribes, extracts MoM. No standalone translation
    step — Claude reads the multilingual transcript directly."""
    def update(**fields):
        with JOBS_LOCK:
            JOBS[job_id].update(fields)

    last_emit = {"t": 0.0}
    def make_progress(step_name: str):
        def cb(chars: int):
            now = time.time()
            if now - last_emit["t"] < 0.5:
                return
            last_emit["t"] = now
            update(progress_chars=chars, progress_step=step_name)
        return cb

    audio_paths: list[Path] = []
    stitched_path: Optional[Path] = None
    ck = _cache_keys(job_id)

    # Resume support: load any already-completed intermediates from in-memory JOBS
    # first, then fall back to R2 cache (survives Render restarts/redeploys).
    with JOBS_LOCK:
        job_snapshot = dict(JOBS.get(job_id, {}))
    transcript = job_snapshot.get("transcript") or _r2_get_text(ck["transcript"])
    detailed_mom = job_snapshot.get("detailed_mom") or _r2_get_json(ck["detailed"])
    concise_mom = job_snapshot.get("concise_mom") or _r2_get_json(ck["concise"])
    speaker_samples = job_snapshot.get("speaker_samples") or _r2_get_json(ck["samples"]) or {}

    try:
        # ---- Step 1: Download + transcribe (skip if cached) ----
        if not transcript:
            update(status="downloading", step="downloading", progress_chars=0, progress_step=None)
            for key, fname in zip(r2_keys, filenames):
                suffix = Path(fname or "audio.webm").suffix or ".webm"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.close()
                _r2_download(key, Path(tmp.name))
                audio_paths.append(Path(tmp.name))

            if len(audio_paths) > 1:
                update(status="stitching", step="stitching", progress_chars=0)
                stem = audio_paths[0].with_suffix("")
                stitched_path = stitch_files(audio_paths, stem)
                audio_to_process = stitched_path
            else:
                audio_to_process = audio_paths[0]

            update(status="transcribing", step="transcription", progress_chars=0, progress_step=None)
            asr_response = None
            if provider == "deepgram":
                if not os.environ.get("DEEPGRAM_API_KEY"):
                    raise RuntimeError("DEEPGRAM_API_KEY not set on server")
                asr_response = base.transcribe(audio_to_process, timeout_seconds=1800.0)
                transcript = base.format_transcript(asr_response)
            else:
                if not os.environ.get("SARVAM_API_KEY"):
                    raise RuntimeError("SARVAM_API_KEY not set on server")
                asr_response = sarvam_pipeline.transcribe_sarvam(audio_to_process)
                transcript = sarvam_pipeline.format_sarvam_transcript(asr_response)

            update(transcript=transcript)
            _r2_put_text(ck["transcript"], transcript)

            # Extract per-speaker audio samples while we still have the local file.
            # Best-effort: failures here don't fail the job.
            if not speaker_samples and asr_response is not None:
                try:
                    speaker_samples = _extract_speaker_samples(
                        asr_response, provider, audio_to_process, job_id
                    )
                    if speaker_samples:
                        update(speaker_samples=speaker_samples)
                        _r2_put_json(ck["samples"], speaker_samples)
                except Exception as e:
                    print(f"Speaker sample extraction skipped: {e}", file=sys.stderr)

        # ---- Step 2: Detailed MoM (skip if cached) ----
        if not detailed_mom:
            update(status="extracting_detailed", step="mom_extraction_detailed",
                   progress_chars=0, progress_step="mom_extraction_detailed")
            detailed_mom = base.extract_mom(transcript, on_progress=make_progress("mom_extraction_detailed"))
            update(detailed_mom=detailed_mom)
            _r2_put_json(ck["detailed"], detailed_mom)
        detailed_md = base.render_markdown(detailed_mom)

        # ---- Step 3: Concise MoM is now opt-in via POST /jobs/{id}/condense ----
        # We skip it in the main pipeline. If a previous run already cached one
        # (e.g. resuming from R2), still surface it; otherwise leave it None.
        concise_md = base.render_markdown(concise_mom) if concise_mom else None

        # Render detailed docx. Concise docx only if we already have a concise MoM
        # (i.e. a previous /condense call populated it).
        update(status="rendering_docx", step="docx_rendering", progress_chars=None, progress_step=None)
        detailed_docx = fill_template_to_bytes(DEFAULT_TEMPLATE, detailed_mom, auto_meta(detailed_mom))
        concise_docx = (
            fill_template_to_bytes(DEFAULT_TEMPLATE, concise_mom, auto_meta(concise_mom))
            if concise_mom else None
        )

        update(
            status="done",
            step=None,
            progress_chars=None,
            progress_step=None,
            transcript=transcript,
            detailed_mom=detailed_mom,
            detailed_markdown=detailed_md,
            concise_mom=concise_mom,
            concise_markdown=concise_md,
            detailed_docx=detailed_docx,
            concise_docx=concise_docx,
            llm_provider=base.LLM_PROVIDER,
            finished_at=time.time(),
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # Always print the full traceback to stdout so it shows up in Render logs.
        print(tb, file=sys.stderr, flush=True)
        # Friendly-ify common provider errors so the UI isn't a wall of HTTP headers.
        raw = str(e)
        friendly = raw
        low = raw.lower()
        if "insufficient_quota" in low or "no credits available" in low:
            friendly = "Sarvam account has no credits left. Top up at https://dashboard.sarvam.ai and retry."
        elif "401" in raw and "deepgram" in low:
            friendly = "Deepgram rejected the request (auth). Check DEEPGRAM_API_KEY on the server."
        elif "401" in raw and "anthropic" in low:
            friendly = "Anthropic rejected the request (auth). Check ANTHROPIC_API_KEY on the server."
        elif "rate_limit" in low or "429" in raw:
            friendly = "Rate limit hit on a provider. Wait a minute and retry."
        update(status="failed", error=friendly, error_raw=tb[:4000], finished_at=time.time())
    finally:
        # Local temp files always cleaned (free disk on Render's small instances).
        for p in audio_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        if stitched_path and stitched_path not in audio_paths:
            try:
                stitched_path.unlink(missing_ok=True)
            except Exception:
                pass
        # R2 audio cleanup: only on SUCCESS. On failure we keep the audio + the
        # cached intermediates so the user can hit Retry without re-paying for
        # Sarvam/Deepgram and Claude. (R2 lifecycle rule auto-deletes after 1 day
        # as a safety net.)
        with JOBS_LOCK:
            final_status = JOBS.get(job_id, {}).get("status")
        if final_status == "done":
            for key in r2_keys:
                _r2_delete(key)
            # Delete transcript/detailed/concise cache on success.
            # Keep speaker_samples.json + the samples/{job_id}/*.mp3 clips
            # around so the Identify-Speakers UI works on the result page.
            ck = _cache_keys(job_id)
            for cache_kind in ("transcript", "detailed", "concise"):
                _r2_delete(ck[cache_kind])


# ---------- routes ----------

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/upload-url")
def get_upload_url(
    payload: dict = Body(...),
    token: Optional[str] = Query(None),
):
    """Mint a presigned PUT URL for the browser to upload one file directly to R2.

    Request body: {"filename": "meeting.mp3", "content_type": "audio/mpeg"}
    Response:     {"url": "https://...", "key": "uploads/<uuid>/meeting.mp3"}
    """
    _check_token(token)
    _require_r2()
    filename = (payload.get("filename") or "audio.bin").replace("/", "_").replace("\\", "_")
    content_type = payload.get("content_type") or "application/octet-stream"
    key = f"uploads/{uuid.uuid4().hex}/{filename}"
    s3 = _r2_client()
    url = s3.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": R2_BUCKET_NAME,
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=3600,  # 1 hour to upload
    )
    return {"url": url, "key": key}


@app.post("/jobs")
def create_job(
    payload: dict = Body(...),
    token: Optional[str] = Query(None),
):
    """Create a processing job from already-uploaded R2 keys.

    Request body:
      {
        "r2_keys": ["uploads/abc.../file1.mp3", "uploads/def.../file2.mp3"],
        "filenames": ["file1.mp3", "file2.mp3"],
        "provider": "deepgram" | "sarvam"
      }
    """
    _check_token(token)
    _require_r2()

    provider = payload.get("provider")
    r2_keys = payload.get("r2_keys") or []
    filenames = payload.get("filenames") or [Path(k).name for k in r2_keys]

    if provider not in ("deepgram", "sarvam"):
        raise HTTPException(status_code=400, detail="provider must be 'deepgram' or 'sarvam'")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")
    if not r2_keys:
        raise HTTPException(status_code=400, detail="r2_keys must be a non-empty list")

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "provider": provider,
            "status": "queued",
            "step": None,
            "n_files": len(r2_keys),
            "created_at": time.time(),
            "error": None,
            # Stored so /jobs/{id}/retry can re-invoke _process_job without re-uploading.
            "_r2_keys": r2_keys,
            "_filenames": filenames,
        }

    thread = threading.Thread(
        target=_process_job, args=(job_id, r2_keys, filenames, provider), daemon=True
    )
    thread.start()
    return {"job_id": job_id}


_HEAVY_KEYS = {
    "transcript", "detailed_mom", "detailed_markdown", "concise_mom",
    "concise_markdown", "detailed_docx", "concise_docx",
    "_r2_keys", "_filenames",
    "speaker_samples",
    "_original_transcript", "_original_detailed_mom", "_original_detailed_markdown",
    "_original_detailed_docx", "_original_concise_mom", "_original_concise_markdown",
    "_original_concise_docx",
}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, token: Optional[str] = Query(None)):
    _check_token(token)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        # Keep status polls light — strip the heavy payloads.
        return {k: v for k, v in job.items() if k not in _HEAVY_KEYS}


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str, token: Optional[str] = Query(None)):
    _check_token(token)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] != "done":
            raise HTTPException(status_code=409, detail=f"job not finished (status: {job['status']})")
        return JSONResponse({
            "id": job["id"],
            "provider": job["provider"],
            "transcript": job.get("transcript"),
            "detailed": {
                "mom": job.get("detailed_mom"),
                "markdown": job.get("detailed_markdown"),
            },
            "concise": {
                "mom": job.get("concise_mom"),
                "markdown": job.get("concise_markdown"),
            },
        })


def _condense_job(job_id: str) -> None:
    """Background worker that runs the condense pass + renders the concise docx.
    Idempotent: skips if concise outputs already exist on the job."""
    def update(**fields):
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id].update(fields)

    last_emit = {"t": 0.0}
    def progress(chars: int):
        now = time.time()
        if now - last_emit["t"] < 0.5:
            return
        last_emit["t"] = now
        update(condense_progress_chars=chars)

    try:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            detailed_mom = job and job.get("detailed_mom")
        if not detailed_mom:
            ck = _cache_keys(job_id)
            detailed_mom = _r2_get_json(ck["detailed"])
        if not detailed_mom:
            update(condense_status="failed", condense_error="No detailed MoM available to condense from.")
            return

        update(condense_status="running", condense_progress_chars=0)
        concise_mom = base.condense_to_concise(detailed_mom, on_progress=progress)
        concise_md = base.render_markdown(concise_mom)
        concise_docx = fill_template_to_bytes(DEFAULT_TEMPLATE, concise_mom, auto_meta(concise_mom))

        # Persist to R2 cache for resumability
        ck = _cache_keys(job_id)
        _r2_put_json(ck["concise"], concise_mom)

        update(
            concise_status="done",
            condense_status="done",
            condense_progress_chars=None,
            concise_mom=concise_mom,
            concise_markdown=concise_md,
            concise_docx=concise_docx,
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        update(condense_status="failed", condense_error=str(e), condense_error_raw=tb[:4000])


@app.post("/jobs/{job_id}/condense")
def start_condense(job_id: str, token: Optional[str] = Query(None)):
    """Generate the concise MoM + docx on demand. Cheap pass (~₹6 / 2 min)."""
    _check_token(token)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job.get("status") != "done":
            raise HTTPException(status_code=409, detail=f"main job not finished (status: {job.get('status')})")
        if job.get("concise_mom"):
            return {"status": "done", "already": True}
        if job.get("condense_status") == "running":
            return {"status": "running", "already": True}
        job["condense_status"] = "queued"
        job["condense_progress_chars"] = 0
        job.pop("condense_error", None)

    thread = threading.Thread(target=_condense_job, args=(job_id,), daemon=True)
    thread.start()
    return {"status": "queued"}


@app.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, token: Optional[str] = Query(None)):
    """Re-run a failed job, skipping any step whose output is already cached
    (in-memory in JOBS or persistently in R2). Saves Claude/Sarvam credits when
    a tail-end step fails."""
    _check_token(token)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            # Job lost from memory (Render restart). Try to rehydrate from R2 cache.
            ck = _cache_keys(job_id)
            cached_transcript = _r2_get_text(ck["transcript"])
            cached_detailed = _r2_get_json(ck["detailed"])
            cached_concise = _r2_get_json(ck["concise"])
            if not (cached_transcript or cached_detailed or cached_concise):
                raise HTTPException(status_code=404, detail="job not found and no cache exists")
            # Need provider + r2_keys + filenames to retry — these are lost on restart.
            # For now, signal that the user has to resubmit, but we'll preserve the cache.
            raise HTTPException(
                status_code=409,
                detail="Job was wiped from memory (Render restart). Resubmit the same audio; cached intermediates will still be used.",
            )
        if job["status"] not in ("failed",):
            raise HTTPException(status_code=409, detail=f"can only retry failed jobs (current status: {job['status']})")
        # Reset error fields, keep cached intermediates (transcript, detailed_mom, concise_mom)
        job["status"] = "queued"
        job["step"] = None
        job["progress_chars"] = None
        job["progress_step"] = None
        job["error"] = None
        job.pop("error_raw", None)
        # We need r2_keys, filenames, and provider from the original submission.
        # Stored on the job dict by create_job; if missing (older runs), reject.
        r2_keys = job.get("_r2_keys")
        filenames = job.get("_filenames")
        provider = job["provider"]
        if not r2_keys:
            raise HTTPException(status_code=409, detail="job has no stored audio references; resubmit")

    thread = threading.Thread(
        target=_process_job, args=(job_id, r2_keys, filenames, provider), daemon=True
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}/speaker-samples")
def get_speaker_samples(job_id: str, token: Optional[str] = Query(None)):
    """Return presigned GET URLs for each speaker's audio sample, plus any
    existing speaker-name mapping."""
    _check_token(token)
    _require_r2()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        samples = dict(job.get("speaker_samples") or {})
        mapping = dict(job.get("speaker_mapping") or {})

    s3 = _r2_client()
    speakers = []
    for label, info in samples.items():
        try:
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": R2_BUCKET_NAME, "Key": info["key"]},
                ExpiresIn=3600,
            )
        except Exception:
            url = None
        speakers.append({
            "label": label,
            "current_name": mapping.get(label, ""),
            "audio_url": url,
            "start_s": info.get("start_s"),
            "end_s": info.get("end_s"),
            "duration_s": info.get("duration_s"),
        })
    # Sort by label so "Speaker 0", "Speaker 1", "Speaker 2" come in order.
    speakers.sort(key=lambda s: s["label"])
    return {"speakers": speakers, "mapping": mapping}


def _apply_speaker_mapping_and_rerender(job_id: str, mapping: dict) -> dict:
    """Apply name mapping to transcript + detailed/concise MoM (if present),
    re-render markdown + docx. Caches originals on first call. Returns the
    updated field dict (kept caller-side for atomic JOBS update)."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")

        # Cache originals once so /reset-speakers can restore them.
        if "_original_transcript" not in job:
            job["_original_transcript"] = job.get("transcript")
            job["_original_detailed_mom"] = job.get("detailed_mom")
            job["_original_detailed_markdown"] = job.get("detailed_markdown")
            job["_original_detailed_docx"] = job.get("detailed_docx")
            if job.get("concise_mom"):
                job["_original_concise_mom"] = job.get("concise_mom")
                job["_original_concise_markdown"] = job.get("concise_markdown")
                job["_original_concise_docx"] = job.get("concise_docx")

        orig_transcript = job.get("_original_transcript")
        orig_detailed = job.get("_original_detailed_mom")
        orig_concise = job.get("_original_concise_mom")

    new_transcript = _apply_mapping_to_str(orig_transcript or "", mapping)
    new_detailed_mom = _apply_mapping_to_obj(orig_detailed, mapping) if orig_detailed else None
    new_concise_mom = _apply_mapping_to_obj(orig_concise, mapping) if orig_concise else None

    updates: dict = {
        "transcript": new_transcript,
        "speaker_mapping": mapping,
    }
    if new_detailed_mom:
        updates["detailed_mom"] = new_detailed_mom
        updates["detailed_markdown"] = base.render_markdown(new_detailed_mom)
        updates["detailed_docx"] = fill_template_to_bytes(
            DEFAULT_TEMPLATE, new_detailed_mom, auto_meta(new_detailed_mom)
        )
    if new_concise_mom:
        updates["concise_mom"] = new_concise_mom
        updates["concise_markdown"] = base.render_markdown(new_concise_mom)
        updates["concise_docx"] = fill_template_to_bytes(
            DEFAULT_TEMPLATE, new_concise_mom, auto_meta(new_concise_mom)
        )
    return updates


@app.post("/jobs/{job_id}/identify-speakers")
def identify_speakers(job_id: str, payload: dict = Body(...), token: Optional[str] = Query(None)):
    """Apply a {label: name} mapping to all outputs.

    Body: {"mapping": {"Speaker 0": "Priya", "Speaker 1": "Rohit"}}
    Empty/blank names in the mapping are ignored — that speaker keeps the
    label as-is.
    """
    _check_token(token)
    raw = payload.get("mapping") or {}
    mapping = {k: v.strip() for k, v in raw.items() if isinstance(v, str) and v.strip()}

    updates = _apply_speaker_mapping_and_rerender(job_id, mapping)
    with JOBS_LOCK:
        JOBS[job_id].update(updates)
    return {
        "status": "ok",
        "mapping": mapping,
        "detailed_updated": "detailed_mom" in updates,
        "concise_updated": "concise_mom" in updates,
    }


@app.post("/jobs/{job_id}/reset-speakers")
def reset_speakers(job_id: str, token: Optional[str] = Query(None)):
    """Restore the original Speaker N labels (undo a previous identify call)."""
    _check_token(token)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if "_original_detailed_mom" not in job:
            return {"status": "ok", "noop": True}
        job["transcript"] = job["_original_transcript"]
        job["detailed_mom"] = job["_original_detailed_mom"]
        job["detailed_markdown"] = job["_original_detailed_markdown"]
        job["detailed_docx"] = job["_original_detailed_docx"]
        if "_original_concise_mom" in job:
            job["concise_mom"] = job["_original_concise_mom"]
            job["concise_markdown"] = job["_original_concise_markdown"]
            job["concise_docx"] = job["_original_concise_docx"]
        job["speaker_mapping"] = {}
    return {"status": "ok"}


@app.get("/jobs/{job_id}/docx")
def get_docx(job_id: str, style: str = Query("detailed"), token: Optional[str] = Query(None)):
    """Stream the rendered docx for a finished job. ?style=detailed|concise"""
    _check_token(token)
    if style not in ("detailed", "concise"):
        raise HTTPException(status_code=400, detail="style must be 'detailed' or 'concise'")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] != "done":
            raise HTTPException(status_code=409, detail=f"job not finished (status: {job['status']})")
        data = job.get(f"{style}_docx")
    if not data:
        raise HTTPException(status_code=404, detail=f"{style} docx not available")
    from fastapi.responses import Response
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="mom_{style}_{job_id}.docx"'},
    )


# ---------- HTML (embedded so deploy is a single file) ----------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Meeting audio → Minutes</title>
<style>
  :root { --bg:#f6f5f1; --fg:#222; --muted:#666; --accent:#3a5a40; --danger:#a04040; --border:#d8d4c8; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--fg); }
  main { max-width: 720px; margin: 0 auto; padding: 32px 20px 80px; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  .sub { color: var(--muted); margin-bottom: 28px; }
  section { background: white; border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 16px; }
  section h2 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0 0 12px; }
  .provider-grid { display: grid; gap: 10px; }
  .provider { display: flex; gap: 12px; align-items: flex-start; padding: 12px; border: 1px solid var(--border); border-radius: 8px; cursor: pointer; transition: border-color .15s, background .15s; }
  .provider:hover { background: #faf8f3; }
  .provider.selected { border-color: var(--accent); background: #f0f4ef; }
  .provider input { margin: 4px 0 0; }
  .provider .name { font-weight: 600; }
  .provider .desc { color: var(--muted); font-size: 14px; margin-top: 2px; }
  .style-tabs { display: inline-flex; gap: 0; background: #f3f0e8; padding: 3px; border-radius: 8px; margin-bottom: 12px; }
  .style-tab { padding: 6px 14px; border: none; background: transparent; cursor: pointer; font: inherit; border-radius: 6px; font-size: 14px; color: var(--muted); }
  .style-tab.active { background: white; color: var(--fg); font-weight: 600; box-shadow: 0 1px 2px rgba(0,0,0,.06); }
  .tabs { display: flex; gap: 4px; margin-bottom: 16px; border-bottom: 1px solid var(--border); }
  .tab { padding: 8px 14px; border: none; background: none; cursor: pointer; color: var(--muted); border-bottom: 2px solid transparent; margin-bottom: -1px; font: inherit; }
  .tab.active { color: var(--fg); border-bottom-color: var(--accent); font-weight: 600; }
  .recorder { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  .uploader { display: flex; flex-direction: column; gap: 12px; }
  .filedrop { display: block; border: 2px dashed var(--border); border-radius: 10px; padding: 24px; text-align: center; color: var(--muted); cursor: pointer; transition: border-color .15s, background .15s; }
  .filedrop:hover, .filedrop.dragover { border-color: var(--accent); background: #faf8f3; color: var(--fg); }
  /* Truly hide the file input — display:none alone can leave browser-extension overlays
     visible for some users; the absolute-positioning trick gets rid of those too. */
  .filedrop input[type="file"] { position: absolute; left: -9999px; width: 1px; height: 1px; opacity: 0; pointer-events: none; }
  .fileinfo { background: #f3f0e8; padding: 10px 12px; border-radius: 8px; font-size: 14px; display: flex; justify-content: space-between; align-items: center; gap: 12px; }
  .fileinfo .clear { color: var(--danger); cursor: pointer; font-size: 13px; }
  button { font: inherit; padding: 10px 16px; border-radius: 8px; border: 1px solid var(--border); background: white; cursor: pointer; }
  button.primary { background: var(--accent); color: white; border-color: var(--accent); }
  button.danger { color: var(--danger); border-color: var(--danger); }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .timer { font-variant-numeric: tabular-nums; font-size: 28px; font-weight: 600; min-width: 90px; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #ccc; margin-right: 8px; vertical-align: middle; }
  .dot.recording { background: #d4423a; animation: pulse 1.2s infinite; }
  .dot.paused { background: #d4a03a; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
  .status { color: var(--muted); font-size: 14px; margin-top: 12px; }
  .result h3 { font-size: 16px; margin: 18px 0 6px; }
  .result pre { background: #f3f0e8; padding: 12px; border-radius: 8px; white-space: pre-wrap; word-break: break-word; max-height: 320px; overflow: auto; font-size: 13px; line-height: 1.45; }
  .download-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
  .download-row a { padding: 8px 12px; border: 1px solid var(--border); border-radius: 8px; text-decoration: none; color: var(--fg); font-size: 14px; }
  .download-row a:hover { background: #faf8f3; }
  .error { color: var(--danger); background: #fbeeee; border: 1px solid #f0c8c8; padding: 10px; border-radius: 8px; margin-top: 12px; }
</style>
</head>
<body>
<main>
  <h1>Meeting audio → Minutes of Meeting</h1>
  <div class="sub">Record a meeting, pause/resume as needed, submit when done. You'll get a diarized transcript and structured minutes.</div>

  <section>
    <h2>1. Pick a transcription provider</h2>
    <div class="provider-grid" id="providers">
      <label class="provider selected" data-provider="deepgram">
        <input type="radio" name="provider" value="deepgram" checked>
        <div>
          <div class="name">Deepgram (nova-3) <span style="color:var(--accent); font-size:12px; font-weight:600;">— default</span></div>
          <div class="desc">Fastest and cheapest. Reliably captures high-level decisions, sign-offs, and action items — the bulk of any MoM. <strong>Pick for:</strong> routine internal coordination meetings where you mainly need the decision log. <strong>Skip when:</strong> the meeting includes vendor selections, exact specs, or substantial Kannada — Deepgram tends to drop named brands, dimensions, and code-switched bits that are only said once.</div>
        </div>
      </label>
      <label class="provider" data-provider="sarvam">
        <input type="radio" name="provider" value="sarvam">
        <div>
          <div class="name">Sarvam (Saarika) <span style="color:var(--muted); font-size:12px;">— pick when detail matters</span></div>
          <div class="desc">Captures vendor names ("EBIT Flow"), brand mentions, exact specs ("50mm cement board"), and single-mention details reliably. Handles Kannada / Hindi / Tamil / Telugu natively. <strong>Pick for:</strong> client-facing MoMs, BOQ-prep meetings, vendor selection calls — anything where granular detail will feed procurement or contracts. Slower and ~3-4× costlier per minute.</div>
        </div>
      </label>
    </div>
  </section>

  <section>
    <h2>2. Provide audio</h2>
    <div class="tabs">
      <button class="tab active" data-tab="record">Record live</button>
      <button class="tab" data-tab="upload">Upload file</button>
    </div>

    <div id="recordTab">
      <div class="recorder">
        <span><span id="dot" class="dot"></span><span id="state">Idle</span></span>
        <span class="timer" id="timer">00:00</span>
        <button id="recordBtn" class="primary">● Record</button>
        <button id="pauseBtn" disabled>⏸ Pause</button>
        <button id="stopBtn" disabled>⏹ Stop</button>
        <button id="resetBtn" disabled>↩ Reset</button>
      </div>
      <div class="status" id="recStatus">Click Record to start. Pause/resume as many times as needed.</div>
    </div>

    <div id="uploadTab" style="display:none;">
      <label class="filedrop" id="filedrop">
        <div><strong>Drop audio file(s) here</strong> or click to choose</div>
        <div style="font-size:13px; margin-top:6px;">mp3, m4a, wav, mp4, webm, ogg, flac… Multiple files will be stitched in the order shown.</div>
        <input type="file" id="fileInput" multiple accept="audio/*,video/mp4,.m4a,.mp3,.wav,.webm,.ogg,.flac">
      </label>
      <div id="fileInfo" class="fileinfo" style="display:none;">
        <span id="fileInfoText"></span>
        <span class="clear" id="fileClear">remove</span>
      </div>
    </div>

    <audio id="preview" controls style="display:none; margin-top:14px; width:100%;"></audio>
  </section>

  <section>
    <h2>3. Submit for processing</h2>
    <button id="submitBtn" class="primary" disabled>📤 Submit recording</button>
    <div class="status" id="submitStatus">Stop the recording first to enable submission.</div>
    <div id="error" class="error" style="display:none;"></div>
  </section>

  <section id="resultSection" style="display:none;">
    <h2>4. Results</h2>
    <div class="result">
      <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
        <div class="style-tabs">
          <button class="style-tab active" data-style="detailed">Detailed</button>
          <button class="style-tab" data-style="concise">Concise</button>
        </div>
        <button id="identifyBtn" type="button" style="margin-left:auto;">
          🗣 Identify speakers
        </button>
      </div>
      <div id="speakerMapBadge" style="display:none; margin:8px 0; font-size:13px; color:var(--muted);"></div>
      <h3>Minutes of meeting</h3>
      <pre id="momMd"></pre>
      <div class="download-row" id="momDownloads"></div>
      <h3 style="margin-top:24px;">Diarized transcript</h3>
      <pre id="transcript"></pre>
      <div class="download-row">
        <a href="#" id="dlTranscript" download="transcript.txt">Download transcript</a>
      </div>
    </div>
  </section>

  <!-- Speaker identification modal -->
  <div id="speakerModal" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.5); z-index:1000; align-items:center; justify-content:center; padding:20px;">
    <div style="background:white; border-radius:12px; max-width:600px; width:100%; max-height:90vh; overflow:auto; padding:24px;">
      <div style="display:flex; justify-content:space-between; align-items:start; margin-bottom:8px;">
        <div>
          <h2 style="margin:0 0 4px; font-size:18px;">Identify speakers</h2>
          <div style="color:var(--muted); font-size:13px;">Play each clip and type the speaker's name. Save to replace 'Speaker N' labels across the MoM, transcript, and downloadable files. You can reset anytime.</div>
        </div>
        <button id="speakerModalClose" type="button" style="border:none; background:none; font-size:24px; line-height:1; cursor:pointer; color:var(--muted); padding:0 4px;">×</button>
      </div>
      <div id="speakerList" style="margin-top:16px; display:flex; flex-direction:column; gap:14px;"></div>
      <div id="speakerError" class="error" style="display:none; margin-top:12px;"></div>
      <div style="display:flex; gap:10px; margin-top:18px; justify-content:flex-end;">
        <button id="speakerResetBtn" type="button" class="danger" style="display:none;">↺ Reset to Speaker IDs</button>
        <button id="speakerSaveBtn" type="button" class="primary">Save names</button>
      </div>
    </div>
  </div>
</main>

<script>
const $ = (id) => document.getElementById(id);
const accessToken = new URLSearchParams(location.search).get("token");

// ---------- provider selection ----------
document.querySelectorAll(".provider").forEach((el) => {
  el.addEventListener("click", () => {
    document.querySelectorAll(".provider").forEach((x) => x.classList.remove("selected"));
    el.classList.add("selected");
    el.querySelector("input").checked = true;
  });
});

// ---------- input mode tabs ----------
let inputMode = "record"; // "record" | "upload"
document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    inputMode = t.dataset.tab;
    $("recordTab").style.display = inputMode === "record" ? "block" : "none";
    $("uploadTab").style.display = inputMode === "upload" ? "block" : "none";
    // Reset shared state when switching modes so we don't accidentally submit the wrong thing
    resetAudioState();
  });
});

function resetAudioState() {
  finalFiles = [];
  if (blobUrl) { URL.revokeObjectURL(blobUrl); blobUrl = null; }
  if (typeof perFileBlobUrls !== "undefined") {
    perFileBlobUrls.forEach((u) => URL.revokeObjectURL(u));
    perFileBlobUrls = [];
  }
  $("preview").style.display = "none";
  $("preview").src = "";
  $("submitBtn").disabled = true;
  $("submitStatus").textContent = inputMode === "record"
    ? "Stop the recording first to enable submission."
    : "Pick file(s) to enable submission.";
  $("fileInfo").style.display = "none";
  $("fileInput").value = "";
}

// ---------- file upload ----------
const filedrop = $("filedrop"), fileInput = $("fileInput");
fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) acceptFiles(Array.from(e.target.files));
});
["dragenter", "dragover"].forEach((ev) => filedrop.addEventListener(ev, (e) => {
  e.preventDefault(); filedrop.classList.add("dragover");
}));
["dragleave", "drop"].forEach((ev) => filedrop.addEventListener(ev, (e) => {
  e.preventDefault(); filedrop.classList.remove("dragover");
}));
filedrop.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length > 0) acceptFiles(Array.from(e.dataTransfer.files));
});
$("fileClear").addEventListener("click", (e) => { e.preventDefault(); resetAudioState(); });

// Track per-file blob URLs so we can revoke them on reset
let perFileBlobUrls = [];

function acceptFiles(files) {
  files.sort((a, b) => a.name.localeCompare(b.name));
  finalFiles = files;

  // Clean up any previous blob URLs
  if (blobUrl) { URL.revokeObjectURL(blobUrl); blobUrl = null; }
  perFileBlobUrls.forEach((u) => URL.revokeObjectURL(u));
  perFileBlobUrls = [];

  const totalMb = files.reduce((s, f) => s + f.size, 0) / (1024 * 1024);

  if (files.length === 1) {
    // Single file — use the main preview element
    blobUrl = URL.createObjectURL(files[0]);
    $("preview").src = blobUrl;
    $("preview").style.display = "block";
    const sizeMb = (files[0].size / (1024 * 1024)).toFixed(1);
    $("fileInfoText").textContent = `${files[0].name} — ${sizeMb} MB`;
    $("submitStatus").textContent = "File ready. Click Submit to process.";
  } else {
    // Multi-file — hide the main preview and render one mini-player per file
    $("preview").style.display = "none";
    $("preview").src = "";
    const listHtml = files.map((f, i) => {
      const url = URL.createObjectURL(f);
      perFileBlobUrls.push(url);
      const sizeMb = (f.size / (1024 * 1024)).toFixed(1);
      return `<div style="margin-top:6px;">
        <div style="font-size:13px;"><strong>${i + 1}.</strong> ${escapeHtml(f.name)} <span style="color:var(--muted);">(${sizeMb} MB)</span></div>
        <audio controls preload="none" src="${url}" style="width:100%; height:30px; margin-top:4px;"></audio>
      </div>`;
    }).join("");
    $("fileInfoText").innerHTML =
      `<strong>${files.length} files, ${totalMb.toFixed(1)} MB total — stitch order:</strong>${listHtml}`;
    $("submitStatus").textContent = `${files.length} files ready. They'll be stitched server-side in the order shown.`;
  }
  $("fileInfo").style.display = "flex";
  $("submitBtn").disabled = false;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- recording ----------
let mediaRecorder, chunks = [], startTime = 0, elapsedBeforePause = 0, timerInterval = null, blobUrl = null, finalFiles = [];

function fmtTime(ms) {
  const s = Math.floor(ms / 1000);
  return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
}
function tick() { $("timer").textContent = fmtTime(elapsedBeforePause + (Date.now() - startTime)); }
function setState(label, dotClass) {
  $("state").textContent = label;
  $("dot").className = "dot " + (dotClass || "");
}

$("recordBtn").addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    chunks = [];
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
      // Wrap blob as a File so it has a name when posted in FormData
      const file = new File([blob], "recording.webm", { type: blob.type });
      finalFiles = [file];
      if (blobUrl) URL.revokeObjectURL(blobUrl);
      blobUrl = URL.createObjectURL(blob);
      $("preview").src = blobUrl;
      $("preview").style.display = "block";
      $("submitBtn").disabled = false;
      $("submitStatus").textContent = "Recording ready. Click Submit to process.";
    };
    mediaRecorder.start();
    startTime = Date.now();
    elapsedBeforePause = 0;
    timerInterval = setInterval(tick, 200);
    setState("Recording", "recording");
    $("recordBtn").disabled = true;
    $("pauseBtn").disabled = false;
    $("stopBtn").disabled = false;
    $("resetBtn").disabled = true;
    $("submitBtn").disabled = true;
    $("preview").style.display = "none";
    $("recStatus").textContent = "Recording… click Pause to pause, Stop when done.";
  } catch (err) {
    $("recStatus").textContent = "Could not access microphone: " + err.message;
  }
});

$("pauseBtn").addEventListener("click", () => {
  if (mediaRecorder.state === "recording") {
    mediaRecorder.pause();
    elapsedBeforePause += Date.now() - startTime;
    clearInterval(timerInterval);
    setState("Paused", "paused");
    $("pauseBtn").textContent = "▶ Resume";
    $("recStatus").textContent = "Paused. Click Resume to continue or Stop to finish.";
  } else if (mediaRecorder.state === "paused") {
    mediaRecorder.resume();
    startTime = Date.now();
    timerInterval = setInterval(tick, 200);
    setState("Recording", "recording");
    $("pauseBtn").textContent = "⏸ Pause";
    $("recStatus").textContent = "Recording…";
  }
});

$("stopBtn").addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    if (mediaRecorder.state === "recording") elapsedBeforePause += Date.now() - startTime;
    mediaRecorder.stop();
  }
  clearInterval(timerInterval);
  setState("Stopped", "");
  $("recordBtn").disabled = true;
  $("pauseBtn").disabled = true;
  $("stopBtn").disabled = true;
  $("resetBtn").disabled = false;
  $("pauseBtn").textContent = "⏸ Pause";
});

$("resetBtn").addEventListener("click", () => {
  resetAudioState();
  $("timer").textContent = "00:00";
  setState("Idle", "");
  $("recordBtn").disabled = false;
  $("pauseBtn").disabled = true;
  $("stopBtn").disabled = true;
  $("resetBtn").disabled = true;
  $("recStatus").textContent = "Click Record to start.";
});

// ---------- submit & poll ----------
// Direct-to-R2 upload with progress, using XMLHttpRequest so we get upload events.
function uploadFileToR2(file, presignedUrl, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", presignedUrl, true);
    xhr.setRequestHeader("Content-Type", file.type || "application/octet-stream");
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded, e.total);
    });
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else reject(new Error(`R2 upload failed: ${xhr.status} ${xhr.responseText || ""}`));
    };
    xhr.onerror = () => reject(new Error("Network error during R2 upload"));
    xhr.ontimeout = () => reject(new Error("R2 upload timed out"));
    xhr.send(file);
  });
}

async function getUploadUrl(file) {
  const tokenParam = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
  const r = await fetch(`/upload-url${tokenParam}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: file.name || "audio.webm",
      content_type: file.type || "application/octet-stream",
    }),
  });
  if (!r.ok) throw new Error("Could not get upload URL: " + (await r.text()));
  return r.json(); // { url, key }
}

$("submitBtn").addEventListener("click", async () => {
  if (!finalFiles.length) return;
  $("submitBtn").disabled = true;
  $("error").style.display = "none";
  $("resultSection").style.display = "none";
  const provider = document.querySelector('input[name="provider"]:checked').value;

  // Phase 1: upload each file directly to R2 with live progress
  const totalBytes = finalFiles.reduce((s, f) => s + f.size, 0);
  const fmtMB = (b) => (b / (1024 * 1024)).toFixed(1);
  const keys = [];
  const filenames = [];
  try {
    let totalLoaded = 0;
    for (let i = 0; i < finalFiles.length; i++) {
      const f = finalFiles[i];
      const { url, key } = await getUploadUrl(f);
      let lastLoaded = 0;
      await uploadFileToR2(f, url, (loaded) => {
        const delta = loaded - lastLoaded;
        lastLoaded = loaded;
        totalLoaded += delta;
        const pct = Math.min(100, Math.floor((totalLoaded / totalBytes) * 100));
        $("submitStatus").textContent =
          `Uploading file ${i + 1}/${finalFiles.length} — ${fmtMB(totalLoaded)} / ${fmtMB(totalBytes)} MB (${pct}%)`;
      });
      keys.push(key);
      filenames.push(f.name || `audio_${i}.webm`);
    }
  } catch (err) {
    showError("Upload failed: " + err.message);
    $("submitBtn").disabled = false;
    return;
  }

  // Phase 2: tell the server the keys are ready
  $("submitStatus").textContent = "Uploads complete — starting job…";
  try {
    const tokenParam = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
    const r = await fetch(`/jobs${tokenParam}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ r2_keys: keys, filenames, provider }),
    });
    if (!r.ok) throw new Error(await r.text());
    const { job_id } = await r.json();
    $("submitStatus").textContent = `Job ${job_id} submitted. Processing…`;
    pollJob(job_id);
  } catch (err) {
    showError("Could not start job: " + err.message);
    $("submitBtn").disabled = false;
  }
});

async function pollJob(jobId) {
  const stepLabel = {
    queued: "Job queued",
    downloading: "Fetching audio from storage",
    stitching: "Stitching audio files",
    transcribing: "Transcribing audio",
    extracting_detailed: "Extracting detailed minutes",
    condensing: "Condensing to concise version",
    rendering_docx: "Rendering Word documents",
  };
  const tokenParam = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
  const startedAt = Date.now();
  while (true) {
    try {
      const resp = await fetch(`/jobs/${jobId}${tokenParam}`);
      if (!resp.ok) throw new Error(await resp.text());
      const job = await resp.json();
      const label = stepLabel[job.status] || job.status;
      const elapsed = Math.floor((Date.now() - startedAt) / 1000);
      const elapsedStr = `${Math.floor(elapsed/60)}:${String(elapsed%60).padStart(2,"0")} elapsed`;
      let progressStr = "";
      if (job.progress_chars && job.progress_chars > 0) {
        progressStr = ` — ${job.progress_chars.toLocaleString()} chars streamed`;
      }
      $("submitStatus").textContent = `${label}… (${elapsedStr}${progressStr})`;
      if (job.status === "done") {
        await loadResult(jobId);
        return;
      }
      if (job.status === "failed") {
        showErrorWithRetry(jobId, job.error || "unknown");
        $("submitBtn").disabled = false;
        return;
      }
    } catch (err) {
      showError("Polling failed: " + err.message);
      $("submitBtn").disabled = false;
      return;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
}

let resultData = null;   // cache of last loaded result so style toggling is instant
let currentJobId = null;

async function loadResult(jobId) {
  const tokenParam = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
  const resp = await fetch(`/jobs/${jobId}/result${tokenParam}`);
  if (!resp.ok) { showError("Could not load result: " + (await resp.text())); return; }
  resultData = await resp.json();
  currentJobId = jobId;

  $("transcript").textContent = resultData.transcript || "";
  $("dlTranscript").href = textBlobUrl(resultData.transcript || "");

  // Show detailed by default
  document.querySelectorAll(".style-tab").forEach((b) => b.classList.toggle("active", b.dataset.style === "detailed"));
  renderStyle("detailed");

  $("resultSection").style.display = "block";
  $("submitStatus").textContent = "Done.";
  $("submitBtn").disabled = false;
}

function renderStyle(style) {
  if (!resultData) return;
  const block = resultData[style] || {};
  const dl = $("momDownloads");
  dl.innerHTML = "";

  if (style === "concise" && !block.mom) {
    // No concise version yet — show the on-demand generation UI
    $("momMd").textContent = "";
    const wrap = document.createElement("div");
    wrap.style.cssText = "padding: 24px; text-align: center;";
    const note = document.createElement("div");
    note.style.cssText = "color: var(--muted); font-size: 14px; margin-bottom: 12px; line-height: 1.5;";
    note.innerHTML = "A concise version is a tightened, client-friendly rewrite of the detailed MoM.<br>Skips redundant detail and merges related sign-offs. Takes ~2 minutes.";
    const btn = document.createElement("button");
    btn.textContent = "Generate concise version";
    btn.className = "primary";
    btn.id = "genConciseBtn";
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.textContent = "Starting…";
      try {
        const tokenParam = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
        const r = await fetch(`/jobs/${currentJobId}/condense${tokenParam}`, { method: "POST" });
        if (!r.ok) throw new Error(await r.text());
        pollCondense();
      } catch (err) {
        btn.disabled = false;
        btn.textContent = "Generate concise version";
        const e = document.createElement("div");
        e.style.cssText = "color: var(--danger); margin-top: 10px; font-size: 13px;";
        e.textContent = "Failed: " + err.message;
        wrap.appendChild(e);
      }
    });
    wrap.appendChild(note);
    wrap.appendChild(btn);
    $("momMd").innerHTML = "";
    $("momMd").appendChild(wrap);
    return;
  }

  $("momMd").textContent = block.markdown || "";
  const tokenParam = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
  function addLink(href, label, dlname) {
    const a = document.createElement("a");
    a.href = href;
    if (dlname) a.download = dlname;
    a.textContent = label;
    dl.appendChild(a);
  }
  addLink(textBlobUrl(block.markdown || ""), `Download ${style} MoM (markdown)`, `mom_${style}.md`);
  addLink(textBlobUrl(JSON.stringify(block.mom || {}, null, 2), "application/json"), `Download ${style} MoM (json)`, `mom_${style}.json`);
  addLink(`/jobs/${currentJobId}/docx?style=${style}${tokenParam ? "&" + tokenParam.slice(1) : ""}`, `Download ${style} MoM (Word)`, `mom_${style}.docx`);
}

async function pollCondense() {
  const tokenParam = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
  const btn = $("genConciseBtn");
  const t0 = Date.now();
  while (true) {
    try {
      const r = await fetch(`/jobs/${currentJobId}${tokenParam}`);
      if (!r.ok) throw new Error(await r.text());
      const job = await r.json();
      const elapsed = Math.floor((Date.now() - t0) / 1000);
      const elapsedStr = `${Math.floor(elapsed/60)}:${String(elapsed%60).padStart(2,"0")}`;
      const chars = job.condense_progress_chars || 0;
      if (btn) {
        btn.textContent = chars > 0
          ? `Condensing… ${elapsedStr} elapsed — ${chars.toLocaleString()} chars streamed`
          : `Condensing… ${elapsedStr} elapsed`;
      }
      if (job.condense_status === "done") {
        // Reload the full result to get the new concise content + docx
        await loadResult(currentJobId);
        renderStyle("concise");
        return;
      }
      if (job.condense_status === "failed") {
        if (btn) {
          btn.disabled = false;
          btn.textContent = "Generate concise version";
        }
        showError("Concise generation failed: " + (job.condense_error || "unknown"));
        return;
      }
    } catch (err) {
      // Transient; try again
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
}

document.querySelectorAll(".style-tab").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".style-tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    renderStyle(b.dataset.style);
  });
});

function textBlobUrl(text, type = "text/plain") {
  return URL.createObjectURL(new Blob([text], { type }));
}

function showError(msg) {
  const e = $("error");
  e.textContent = msg;
  e.style.display = "block";
}

function showErrorWithRetry(jobId, msg) {
  const e = $("error");
  e.innerHTML = "";
  const text = document.createElement("div");
  text.textContent = "Processing failed: " + msg;
  e.appendChild(text);
  const note = document.createElement("div");
  note.style.cssText = "font-size:13px; color:var(--muted); margin-top:6px;";
  note.textContent = "Cached intermediates will be reused — retrying won't re-pay for transcription or any extraction steps already completed.";
  e.appendChild(note);
  const btn = document.createElement("button");
  btn.textContent = "↻ Retry";
  btn.className = "primary";
  btn.style.marginTop = "10px";
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "Retrying…";
    const tokenParam = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
    try {
      const r = await fetch(`/jobs/${jobId}/retry${tokenParam}`, { method: "POST" });
      if (!r.ok) throw new Error(await r.text());
      e.style.display = "none";
      $("submitStatus").textContent = "Retrying job…";
      pollJob(jobId);
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "↻ Retry";
      showError("Retry failed: " + err.message);
    }
  });
  e.appendChild(btn);
  e.style.display = "block";
}

// ---------- Speaker identification ----------

function updateSpeakerBadge(mapping) {
  const badge = $("speakerMapBadge");
  const entries = Object.entries(mapping || {});
  if (entries.length === 0) {
    badge.style.display = "none";
    return;
  }
  const txt = entries.map(([label, name]) => `${label} → ${name}`).join(", ");
  badge.textContent = `Speakers identified: ${txt}`;
  badge.style.display = "block";
}

async function openSpeakerModal() {
  if (!currentJobId) return;
  const modal = $("speakerModal");
  const list = $("speakerList");
  const err = $("speakerError");
  const resetBtn = $("speakerResetBtn");
  err.style.display = "none";
  list.innerHTML = "<div style='color:var(--muted); font-size:14px;'>Loading speaker samples…</div>";
  modal.style.display = "flex";
  try {
    const tokenParam = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
    const r = await fetch(`/jobs/${currentJobId}/speaker-samples${tokenParam}`);
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    list.innerHTML = "";
    if (!data.speakers || data.speakers.length === 0) {
      list.innerHTML = "<div style='color:var(--muted); font-size:14px;'>No speaker samples were captured for this meeting.</div>";
      resetBtn.style.display = "none";
      return;
    }
    data.speakers.forEach((spk) => {
      const row = document.createElement("div");
      row.style.cssText = "background:#faf8f3; border:1px solid var(--border); border-radius:8px; padding:12px;";
      const head = document.createElement("div");
      head.style.cssText = "display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;";
      const label = document.createElement("div");
      label.innerHTML = `<strong>${spk.label}</strong> <span style="color:var(--muted); font-size:12px;">— ${spk.duration_s}s sample from ${Math.floor(spk.start_s)}s in</span>`;
      head.appendChild(label);
      row.appendChild(head);
      if (spk.audio_url) {
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "none";
        audio.src = spk.audio_url;
        audio.style.cssText = "width:100%; height:32px;";
        row.appendChild(audio);
      }
      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = "Name (leave blank to keep as Speaker N)";
      input.value = spk.current_name || "";
      input.dataset.label = spk.label;
      input.style.cssText = "width:100%; margin-top:8px; padding:8px 10px; border:1px solid var(--border); border-radius:6px; font:inherit;";
      row.appendChild(input);
      list.appendChild(row);
    });
    resetBtn.style.display = Object.keys(data.mapping || {}).length > 0 ? "inline-block" : "none";
  } catch (e) {
    list.innerHTML = "";
    err.textContent = "Could not load speaker samples: " + e.message;
    err.style.display = "block";
  }
}

function closeSpeakerModal() {
  $("speakerModal").style.display = "none";
}

async function saveSpeakerMapping() {
  const inputs = document.querySelectorAll("#speakerList input[type='text']");
  const mapping = {};
  inputs.forEach((inp) => {
    const v = inp.value.trim();
    if (v) mapping[inp.dataset.label] = v;
  });
  const btn = $("speakerSaveBtn");
  const err = $("speakerError");
  err.style.display = "none";
  btn.disabled = true;
  btn.textContent = "Saving…";
  try {
    const tokenParam = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
    const r = await fetch(`/jobs/${currentJobId}/identify-speakers${tokenParam}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mapping }),
    });
    if (!r.ok) throw new Error(await r.text());
    // Reload the result so the new mapping flows through to UI + downloads.
    closeSpeakerModal();
    await loadResult(currentJobId);
    updateSpeakerBadge(mapping);
  } catch (e) {
    err.textContent = "Save failed: " + e.message;
    err.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.textContent = "Save names";
  }
}

async function resetSpeakerMapping() {
  const btn = $("speakerResetBtn");
  const err = $("speakerError");
  err.style.display = "none";
  btn.disabled = true;
  btn.textContent = "Resetting…";
  try {
    const tokenParam = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
    const r = await fetch(`/jobs/${currentJobId}/reset-speakers${tokenParam}`, { method: "POST" });
    if (!r.ok) throw new Error(await r.text());
    closeSpeakerModal();
    await loadResult(currentJobId);
    updateSpeakerBadge({});
  } catch (e) {
    err.textContent = "Reset failed: " + e.message;
    err.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.textContent = "↺ Reset to Speaker IDs";
  }
}

$("identifyBtn").addEventListener("click", openSpeakerModal);
$("speakerModalClose").addEventListener("click", closeSpeakerModal);
$("speakerSaveBtn").addEventListener("click", saveSpeakerMapping);
$("speakerResetBtn").addEventListener("click", resetSpeakerMapping);
$("speakerModal").addEventListener("click", (e) => {
  if (e.target === $("speakerModal")) closeSpeakerModal();
});

// Surface any existing speaker mapping when result loads
const _originalLoadResult = loadResult;
loadResult = async function(jobId) {
  await _originalLoadResult(jobId);
  // Pull current mapping from the result for badge display
  try {
    const tokenParam = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
    const r = await fetch(`/jobs/${jobId}/speaker-samples${tokenParam}`);
    if (r.ok) {
      const data = await r.json();
      updateSpeakerBadge(data.mapping || {});
    }
  } catch (e) { /* non-fatal */ }
};
</script>
</body>
</html>
"""
