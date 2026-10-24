"""Background music: ONE track for the whole video (so it cannot change between shots).

mode "file"      use your own song / track
mode "generate"  make a track with MusicGen from a text prompt (looped to cover the video)
mode "none"      no music
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .project import Project
from .util import free_gpu


def prepare_music(project: Project, cfg: dict, total_seconds: float, force: bool = False) -> Path | None:
    m = project.music
    if m.mode == "file":
        if not m.file or not project.abs(m.file).exists():
            raise RuntimeError("Music mode is 'file' but no music file was found. Upload one in the Music tab.")
        return project.abs(m.file)
    if m.mode != "generate":
        return None

    seconds = int(min(max(total_seconds, 8), cfg["music"]["max_generate_seconds"]))
    key = hashlib.md5(f"{m.prompt}|{seconds}|{project.seed}".encode()).hexdigest()[:8]
    out = project.dir("work/music") / f"music_{key}.wav"
    if out.exists() and not force:
        return out

    import torch
    from scipy.io import wavfile
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(cfg["music"]["model"])
    model = MusicgenForConditionalGeneration.from_pretrained(cfg["music"]["model"]).to(device)
    torch.manual_seed(project.seed)
    inputs = processor(text=[m.prompt], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=int(seconds * 50))
    rate = model.config.audio_encoder.sampling_rate
    wavfile.write(str(out), rate, audio[0, 0].float().cpu().numpy())
    del model, processor
    free_gpu()
    return out
