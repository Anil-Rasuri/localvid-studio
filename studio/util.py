"""Small helpers: slugs, image fitting, ffmpeg wrappers, GPU cleanup."""
from __future__ import annotations

import gc
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(text).strip()).strip("-").lower()
    return slug or "untitled"


def fit_size(aspect_w: int, aspect_h: int, long_side: int, multiple: int) -> tuple[int, int]:
    """Size with the aspect ratio of aspect_w:aspect_h whose long side is ~long_side, rounded to `multiple`."""
    ratio = aspect_w / aspect_h
    if ratio >= 1:
        w, h = long_side, long_side / ratio
    else:
        w, h = long_side * ratio, long_side

    def snap(v: float) -> int:
        return max(multiple, int(round(v / multiple)) * multiple)

    return snap(w), snap(h)


def fit_cover(img: Image.Image, width: int, height: int) -> Image.Image:
    """Resize + centre-crop so the image fills width x height exactly (never stretches)."""
    img = img.convert("RGB")
    scale = max(width / img.width, height / img.height)
    new_size = (max(width, round(img.width * scale)), max(height, round(img.height * scale)))
    img = img.resize(new_size, Image.LANCZOS)
    left = (img.width - width) // 2
    top = (img.height - height) // 2
    return img.crop((left, top, left + width, top + height))


def frames_for(seconds: float, fps: int, step: int, lo: int, hi: int) -> int:
    """Frame count of the form step*k+1 closest to seconds*fps, clamped to [lo, hi]."""
    n = int(round((seconds * fps - 1) / step)) * step + 1
    return max(lo, min(hi, n))


# ---------------------------------------------------------------- ffmpeg
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ffmpeg was not found. Install it (Windows: `winget install Gyan.FFmpeg`) "
            "and open a new terminal."
        ) from exc


def run_ffmpeg(args: list, cwd: str | Path | None = None) -> None:
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *[str(a) for a in args]]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + " ".join(cmd) + "\n" + result.stderr[-2500:])


def media_duration(path: str | Path) -> float:
    """Duration in seconds of an audio/video file (parsed from `ffmpeg -i`, so no ffprobe needed)."""
    result = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise RuntimeError(f"Could not read the duration of {path}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------- GPU
def free_gpu() -> None:
    """Release cached GPU memory so the next model has room (important on 8 GB cards)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass
