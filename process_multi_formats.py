"""
Same as process.py, but the stitcher always re-encodes through libmp3lame.

Use this when your input files have different codecs / sample rates / channel
counts (e.g. one mp3 + one m4a + one wav). It's slower than the fast-path copy
in process.py, but it works regardless of format mismatch.

Usage:
    python process_multi_formats.py file1.mp3 file2.m4a file3.wav [--name custom]
"""

import shutil
import subprocess
import sys
from pathlib import Path

import process as base


def stitch_audio_reencode(audio_paths: list[Path], out_path: Path) -> Path:
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found. Install it with: brew install ffmpeg")

    n = len(audio_paths)
    inputs = []
    for p in audio_paths:
        inputs.extend(["-i", str(p)])
    filter_expr = "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[out]"

    final = out_path.with_suffix(".mp3")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_expr,
        "-map", "[out]",
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(final),
    ]
    print(f"→ Stitching {n} files with re-encode (handles mixed formats)...", file=sys.stderr)
    subprocess.run(cmd, check=True)
    return final


# Override the stitcher used by base.main()
base.stitch_audio = stitch_audio_reencode


if __name__ == "__main__":
    base.main()
