"""Clips: animate a keyframe into a short video (about 3-5 seconds).

Backends
  ltx       LTX-Video image-to-video through diffusers (default; documented API)
  wan       Wan 2.2 TI2V-5B through diffusers (experimental; may need more than 8 GB without extra tuning)
  kenburns  no AI: a slow zoom/pan over the keyframe. Instant, rock-solid character consistency.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .project import Project, Shot
from .prompts import video_prompt
from .util import fit_cover, fit_size, frames_for, free_gpu, media_duration, run_ffmpeg


def ken_burns(image_path: Path, out_path: Path, seconds: float, width: int, height: int, fps: int, variant: int = 0) -> None:
    """Slow zoom (alternating in / out per shot) over a still image, rendered with ffmpeg."""
    frames = max(1, int(round(seconds * fps)))
    if variant % 2 == 0:
        zoom = "min(zoom+0.0012,1.18)"
    else:
        zoom = "if(eq(on,0),1.18,max(zoom-0.0012,1.0))"
    vf = (
        f"scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,crop={width * 2}:{height * 2},"
        f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={width}x{height}:fps={fps},"
        "format=yuv420p"
    )
    run_ffmpeg(["-i", image_path, "-vf", vf, "-frames:v", frames, "-c:v", "libx264", "-crf", "18",
                "-preset", "medium", out_path])


class ClipGenerator:
    def __init__(self, project: Project, cfg: dict):
        self.p = project
        self.cfg = cfg
        self.c = cfg["clips"]
        self.backend = self.c["backend"]
        self.pipe = None

    # -------------------------------------------------------- model loading
    def _load_ltx(self) -> None:
        import torch
        from diffusers import LTXImageToVideoPipeline

        pipe = LTXImageToVideoPipeline.from_pretrained(self.c["model"], torch_dtype=torch.bfloat16)
        if self.cfg["low_vram"]:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.cfg["device"])
        pipe.vae.enable_tiling()
        self.pipe = pipe

    def _load_wan(self) -> None:
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        model = self.c["wan"]["model"]
        vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(model, vae=vae, torch_dtype=torch.bfloat16)
        if self.cfg["low_vram"]:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.cfg["device"])
        self.pipe = pipe

    # -------------------------------------------------------- main
    def generate(self, shot: Shot, keyframe: Path, out_path: Path, seconds: float) -> float:
        """Write the clip to out_path and return its duration in seconds."""
        out_path = Path(out_path)
        c = self.c
        if self.backend == "kenburns":
            w, h = fit_size(self.p.width, self.p.height, 1280, 2)
            ken_burns(keyframe, out_path, seconds, w, h, self.p.fps, variant=shot.id)
            return media_duration(out_path)

        import torch
        from diffusers.utils import export_to_video

        fps = c["fps"]
        if self.backend == "wan":
            if self.pipe is None:
                self._load_wan()
            w, h = fit_size(self.p.width, self.p.height, c["wan"]["long_side"], 32)
            frames = frames_for(seconds, fps, 4, 33, 121)
            steps, guidance = c["wan"]["steps"], c["wan"]["guidance"]
        else:
            if self.pipe is None:
                self._load_ltx()
            w, h = fit_size(self.p.width, self.p.height, c["long_side"], 32)
            frames = frames_for(seconds, fps, 8, c["min_frames"], c["max_frames"])
            steps, guidance = c["steps"], c["guidance"]

        image = fit_cover(Image.open(keyframe), w, h)
        result = self.pipe(
            image=image,
            prompt=video_prompt(self.p, shot, c["motion_suffix"]),
            negative_prompt=c["negative"],
            width=w, height=h,
            num_frames=frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator("cpu").manual_seed(shot.seed),
        )
        export_to_video(result.frames[0], str(out_path), fps=fps)
        return frames / fps

    def unload(self) -> None:
        self.pipe = None
        free_gpu()
