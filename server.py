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
    try:
        # 1) Download each R2 object to a local temp file.
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

        if provider == "deepgram":
            if not os.environ.get("DEEPGRAM_API_KEY"):
                raise RuntimeError("DEEPGRAM_API_KEY not set on server")
            dg = base.transcribe(audio_to_process, timeout_seconds=1800.0)
            transcript = base.format_transcript(dg)
        else:  # sarvam
            if not os.environ.get("SARVAM_API_KEY"):
                raise RuntimeError("SARVAM_API_KEY not set on server")
            sv = sarvam_pipeline.transcribe_sarvam(audio_to_process)
            transcript = sarvam_pipeline.format_sarvam_transcript(sv)

        # Detailed extraction (shared prompt — same for Deepgram and Sarvam)
        update(status="extracting_detailed", step="mom_extraction_detailed",
               progress_chars=0, progress_step="mom_extraction_detailed")
        detailed_mom = base.extract_mom(transcript, on_progress=make_progress("mom_extraction_detailed"))
        detailed_md = base.render_markdown(detailed_mom)

        # Concise condense (same source, tighter prose + merging of same-fact items)
        update(status="condensing", step="mom_extraction_concise",
               progress_chars=0, progress_step="mom_extraction_concise")
        concise_mom = base.condense_to_concise(detailed_mom, on_progress=make_progress("mom_extraction_concise"))
        concise_md = base.render_markdown(concise_mom)

        # Render both docx (template metadata auto-derived from the MoM itself)
        update(status="rendering_docx", step="docx_rendering", progress_chars=None, progress_step=None)
        detailed_docx = fill_template_to_bytes(DEFAULT_TEMPLATE, detailed_mom, auto_meta(detailed_mom))
        concise_docx = fill_template_to_bytes(DEFAULT_TEMPLATE, concise_mom, auto_meta(concise_mom))

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
            finished_at=time.time(),
        )
    except Exception as e:
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
        update(status="failed", error=friendly, error_raw=raw[:2000], finished_at=time.time())
    finally:
        # Clean up local temp files
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
        # Clean up R2 objects (best-effort — they have a lifecycle rule too as a safety net)
        for key in r2_keys:
            _r2_delete(key)


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
        }

    thread = threading.Thread(
        target=_process_job, args=(job_id, r2_keys, filenames, provider), daemon=True
    )
    thread.start()
    return {"job_id": job_id}


_HEAVY_KEYS = {
    "transcript", "detailed_mom", "detailed_markdown", "concise_mom",
    "concise_markdown", "detailed_docx", "concise_docx",
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
      <label class="provider selected" data-provider="sarvam">
        <input type="radio" name="provider" value="sarvam" checked>
        <div>
          <div class="name">Sarvam (Saarika) <span style="color:var(--accent); font-size:12px; font-weight:600;">— pick when detail matters</span></div>
          <div class="desc">Captures vendor names ("EBIT Flow"), brand mentions, exact specs ("50mm cement board"), and single-mention details reliably. Handles Kannada / Hindi / Tamil / Telugu natively. <strong>Pick for:</strong> client-facing MoMs, BOQ-prep meetings, vendor selection calls — anything where granular detail will feed procurement or contracts. Slower and ~3-4× costlier per minute.</div>
        </div>
      </label>
      <label class="provider" data-provider="deepgram">
        <input type="radio" name="provider" value="deepgram">
        <div>
          <div class="name">Deepgram (nova-3)</div>
          <div class="desc">Fastest and cheapest. Reliably captures high-level decisions, sign-offs, and action items — the bulk of any MoM. <strong>Pick for:</strong> routine internal coordination meetings where you mainly need the decision log. <strong>Skip when:</strong> the meeting includes vendor selections, exact specs, or substantial Kannada — Deepgram tends to drop named brands, dimensions, and code-switched bits that are only said once.</div>
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
      <div class="style-tabs">
        <button class="style-tab active" data-style="detailed">Detailed</button>
        <button class="style-tab" data-style="concise">Concise</button>
      </div>
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
        showError("Processing failed: " + (job.error || "unknown"));
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
  $("momMd").textContent = block.markdown || "";
  // Build the download row inline: markdown + json + docx for the active style
  const tokenParam = accessToken ? `?token=${encodeURIComponent(accessToken)}` : "";
  const dl = $("momDownloads");
  dl.innerHTML = "";
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
</script>
</body>
</html>
"""
