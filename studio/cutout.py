"""Cutout mode: guaranteed-identical characters.

The AI paints only the BACKGROUND of each shot. Your own character images are cut out of their
backgrounds and animated on top (idle sway, bounce while speaking, slow camera move). Because the
characters are your pictures, they cannot change between shots or frames.

Character images should be clean pictures on a plain (white/light) or transparent background,
showing ONE character. If the picture contains several figures, the largest one is used.
"""
from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageFilter

from .project import Project, Shot
from .util import ffmpeg_exe, fit_cover, slugify


# ---------------------------------------------------------------- cutting out
def remove_background(img: Image.Image, tolerance: float = 40.0) -> Image.Image:
    """Return the character as RGBA, cropped to its outline. Uses the image's own transparency if it
    has any; otherwise removes the plain background that touches the image border."""
    from scipy import ndimage

    arr = np.array(img.convert("RGBA"))
    alpha = arr[..., 3]
    if (alpha < 250).mean() > 0.02:  # already has transparency
        mask = alpha.copy()
    else:
        rgb = arr[..., :3].astype(np.float32)
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        bg = np.median(border, axis=0)
        near = np.sqrt(((rgb - bg) ** 2).sum(axis=2)) < tolerance
        labels, _ = ndimage.label(near)
        touching = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
        fg = ~np.isin(labels, list(touching))
        # keep only the largest figure (gaps of a few pixels, e.g. a thin tail, are bridged first)
        bridged = ndimage.binary_dilation(fg, iterations=6)
        comp, n = ndimage.label(bridged)
        if n > 1:
            sizes = ndimage.sum(fg, comp, range(1, n + 1))
            fg = fg & (comp == (int(np.argmax(sizes)) + 1))
        fg = ndimage.binary_fill_holes(fg)
        soft = Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))
        mask = np.array(soft.filter(ImageFilter.GaussianBlur(1.0)))
    arr[..., 3] = mask
    out = Image.fromarray(arr, "RGBA")
    bbox = out.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    return out.crop(bbox) if bbox else out


def load_cutout(project: Project, character, cfg: dict) -> Image.Image | None:
    """Cut-out image of a character (cached in work/cutouts, re-made when the picture changes)."""
    if not character or not character.reference_image:
        return None
    src = project.abs(character.reference_image)
    if not src.exists():
        return None
    key = hashlib.md5(src.read_bytes()).hexdigest()[:8]
    out = project.dir("work/cutouts") / f"{slugify(character.name)}_{key}.png"
    if out.exists():
        return Image.open(out).convert("RGBA")
    cut = remove_background(Image.open(src), cfg["clips"]["cutout"]["tolerance"])
    cut.save(out)
    return cut


# ---------------------------------------------------------------- backgrounds without AI
def plain_background(width: int, height: int) -> Image.Image:
    """Simple sky + grass picture, used when no AI background is available (tests, offline use)."""
    y = np.linspace(0, 1, height)[:, None, None]
    sky = np.array([135, 195, 235]) * (1 - y) + np.array([205, 232, 210]) * y
    grass = np.array([105, 170, 85]) * (1 - y) + np.array([70, 130, 60]) * y
    horizon = 0.68
    arr = np.where(y < horizon, sky, grass).astype(np.uint8)
    return Image.fromarray(np.broadcast_to(arr, (height, width, 3)).copy(), "RGB")


# ---------------------------------------------------------------- animation helpers
def _speech_envelope(project: Project, shot: Shot, name: str, n_frames: int, fps: int) -> np.ndarray:
    """0..1 loudness of this character's speech for every video frame (drives the talking bounce)."""
    env = np.zeros(n_frames, dtype=np.float32)
    for line in shot.dialogue:
        if line.speaker.lower() != name.lower() or not line.audio_path:
            continue
        path = project.abs(line.audio_path)
        data = np.zeros(0, dtype=np.float32)
        sr = 24000
        if path.exists():
            data, sr = sf.read(str(path), dtype="float32")
            data = data.mean(axis=1) if data.ndim > 1 else data
        hop = max(1, int(sr / fps))
        n_here = max(1, int(round(line.duration * fps))) if line.duration else len(data) // hop
        local = np.array([np.sqrt(np.mean(data[i * hop:(i + 1) * hop] ** 2)) if len(data) > (i + 1) * hop else 0.0
                          for i in range(n_here)], dtype=np.float32)
        if local.max() < 1e-4:  # silent voice: fake a talking rhythm for the length of the line
            t = np.arange(n_here) / fps
            local = (0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * t)).astype(np.float32)
        else:
            local = np.clip(local / (local.max() * 0.7), 0, 1)
        start = int(round(line.start * fps))
        for i, v in enumerate(local):
            f = start + i
            if 0 <= f < n_frames:
                env[f] = max(env[f], v)
    if env.max() > 0:
        env = np.convolve(env, np.ones(3, dtype=np.float32) / 3, mode="same")
    return env


def _framing(text: str) -> float:
    t = (text or "").lower()
    if "close-up" in t or "close up" in t or "closeup" in t:
        return 1.55
    if "wide shot" in t or "wide-angle" in t or "far away" in t:
        return 0.8
    return 1.0


def render_cutout_clip(project: Project, cfg: dict, shot: Shot, background_path: Path,
                       out_path: Path, seconds: float) -> float:
    c = cfg["clips"]["cutout"]
    fps = project.fps
    W, H = project.width - project.width % 2, project.height - project.height % 2
    n = max(2, int(round(seconds * fps)))

    # background, a little larger than the frame so the camera can move
    big = fit_cover(Image.open(background_path), int(W * 1.15), int(H * 1.15))

    # characters on screen (only those that have a picture)
    cast = []
    for name in shot.characters:
        ch = project.character(name)
        cut = load_cutout(project, ch, cfg)
        if cut is not None:
            cast.append((ch, cut))

    k = _framing(shot.visual)
    base_h = H * c["char_height"] * k
    ground = H * (c["ground"] + (k - 1.0) * 0.18)
    bases, shadows, envs = [], [], []
    for ch, cut in cast:
        scale = base_h / cut.height
        size = (max(2, int(cut.width * scale)), max(2, int(base_h)))
        bases.append(cut.resize(size, Image.LANCZOS))
        sh = Image.new("RGBA", (int(size[0] * 1.0), int(size[0] * 0.22)), (0, 0, 0, 0))
        from PIL import ImageDraw
        ImageDraw.Draw(sh).ellipse((0, 0, sh.width - 1, sh.height - 1), fill=(0, 0, 0, 70))
        shadows.append(sh.filter(ImageFilter.GaussianBlur(6)))
        envs.append(_speech_envelope(project, shot, ch.name, n, fps))

    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
           "-pix_fmt", "yuv420p", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    zoom_in = shot.id % 2 == 1
    try:
        for f in range(n):
            t, p = f / fps, f / max(1, n - 1)
            # camera: slow push-in (or pull-out on even shots), slight drift
            view = 1.10 - 0.10 * p if zoom_in else 1.0 + 0.10 * p
            cw, ch_ = W * view, H * view
            left = (big.width - cw) / 2 + math.sin(p * math.pi) * W * 0.01
            top = (big.height - ch_) / 2
            frame = big.crop((int(left), int(top), int(left + cw), int(top + ch_))).resize((W, H), Image.BILINEAR)
            frame = frame.convert("RGBA")

            count = len(cast)
            for i, base in enumerate(bases):
                cx = W * (i + 1) / (count + 1)
                phase = i * 1.7
                e = float(envs[i][f]) if f < len(envs[i]) else 0.0
                idle = abs(math.sin(2 * math.pi * 0.8 * t + phase)) * H * 0.006
                talk = abs(math.sin(2 * math.pi * 4.5 * t)) * e * H * 0.03
                squash = 1.0 + 0.035 * e * math.sin(2 * math.pi * 4.5 * t)
                grow = 0.88 + 0.12 * min(1.0, t / 0.4) ** 0.5   # pop in at the start of the shot
                sx, sy = grow / squash, grow * squash
                img = base.resize((max(2, int(base.width * sx)), max(2, int(base.height * sy))), Image.BILINEAR)
                angle = math.sin(2 * math.pi * 0.55 * t + phase) * 1.4 + (math.sin(2 * math.pi * 4.5 * t) * 1.5 * e)
                img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
                lift = idle + talk
                sh = shadows[i]
                frame.alpha_composite(sh, (int(cx - sh.width / 2), int(ground - sh.height * 0.5)))
                frame.alpha_composite(img, (int(cx - img.width / 2), int(ground - img.height - lift)))
            proc.stdin.write(frame.convert("RGB").tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while rendering the cutout clip:\n" + err[-1500:])
    except BrokenPipeError:
        err = proc.stderr.read().decode("utf-8", "replace")
        raise RuntimeError("ffmpeg stopped early:\n" + err[-1500:])
    finally:
        if proc.poll() is None:
            proc.kill()
    return n / fps
