"""Configuration: built-in defaults, overridden by config.yaml (if present)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "projects"

DEFAULT_CONFIG: dict = {
    "device": "cuda",          # "cuda" or "cpu"
    "low_vram": True,          # CPU offload so 8 GB GPUs can run the big models (slower, but fits)
    "shot": {
        "min_seconds": 3.0,    # shortest a shot may be
        "max_seconds": 5.0,    # longest AI-generated clip; longer dialogue is ping-pong looped
        "dialogue_gap": 0.35,  # pause between two lines in one shot
        "tail_pad": 0.4,       # silence after the last line
    },
    "llm": {                   # optional: writes the script from a one-line idea (needs Ollama)
        "host": "http://localhost:11434",
        "model": "qwen2.5:7b",
        "shot_seconds": 4.0,
    },
    "keyframes": {
        "backend": "sdxl",     # sdxl | none  (none = only use images you upload)
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "vae": "madebyollin/sdxl-vae-fp16-fix",
        "variant": "fp16",
        "long_side": 1024,
        "steps": 28,
        "guidance": 6.5,
        "use_ip_adapter": True,
        "ip_adapter_repo": "h94/IP-Adapter",
        "ip_adapter_subfolder": "sdxl_models",
        "ip_adapter_weight": "ip-adapter_sdxl.bin",
        "ip_adapter_scale": 0.75,
        "negative": "blurry, low quality, deformed, extra limbs, text, watermark, inconsistent character",
    },
    "clips": {
        "backend": "ltx",      # ltx | wan | kenburns
        "model": "Lightricks/LTX-Video",
        "long_side": 640,      # lower to 512 if you run out of GPU memory
        "fps": 24,
        "steps": 30,
        "guidance": 3.0,
        "min_frames": 49,      # 8k+1 for LTX
        "max_frames": 121,
        "motion_suffix": "subtle natural motion, stable camera, consistent character appearance",
        "negative": "worst quality, inconsistent motion, blurry, jittery, distorted, morphing, changing character",
        "fallback_on_error": True,   # if the AI clip fails (e.g. out of memory) use a camera-move clip instead
        "wan": {               # experimental backend
            "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "long_side": 704,
            "steps": 30,
            "guidance": 5.0,
        },
    },
    "voice": {
        "target_rms": 0.08,    # every line is normalised to this loudness so voices stay level
    },
    "music": {
        "model": "facebook/musicgen-small",
        "max_generate_seconds": 30,   # longer videos loop the generated track
    },
}


def _deep_update(base: dict, new: dict) -> dict:
    for key, value in (new or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict:
    """Return defaults merged with config.yaml (or the file given in `path`)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            _deep_update(cfg, yaml.safe_load(fh) or {})
    return cfg
