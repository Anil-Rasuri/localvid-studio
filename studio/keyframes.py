"""Keyframes: the first image of every shot.

Character consistency comes from three layers (use as many as you have):
1. your own image for the shot (always used as-is)            -> `shot.user_keyframe`
2. a character LoRA trained on your design (strongest)         -> `character.lora_path`
3. IP-Adapter conditioning on the character's reference image  -> `character.reference_image`
plus the fixed style / character description text added to every prompt.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageOps

from .project import Project, Shot
from .prompts import BG_NEGATIVE, background_prompt, keyframe_prompt
from .util import fit_cover, fit_size, free_gpu


def reference_sheet(images: list[Image.Image], size: int = 512) -> Image.Image:
    """Square collage of several reference images (IP-Adapter looks at a square crop)."""
    if len(images) == 1:
        return ImageOps.contain(images[0].convert("RGB"), (size, size))
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    cw, ch = size // cols, size // rows
    sheet = Image.new("RGB", (size, size), "white")
    for i, img in enumerate(images):
        cell = ImageOps.contain(img.convert("RGB"), (cw, ch))
        x, y = (i % cols) * cw + (cw - cell.width) // 2, (i // cols) * ch + (ch - cell.height) // 2
        sheet.paste(cell, (x, y))
    return sheet


class KeyframeGenerator:
    def __init__(self, project: Project, cfg: dict):
        self.p = project
        self.cfg = cfg
        self.k = cfg["keyframes"]
        self.size = fit_size(project.width, project.height, self.k["long_side"], 8)
        self.pipe = None
        self._loras: dict[str, str] = {}
        self.bg_mode = cfg["clips"]["backend"] == "cutout"   # background-only images

    # -------------------------------------------------------- helpers
    def _references(self, shot: Shot) -> list[Path]:
        refs = []
        for name in shot.characters:
            c = self.p.character(name)
            if c and c.reference_image and self.p.abs(c.reference_image).exists():
                refs.append(self.p.abs(c.reference_image))
        return refs

    def _load(self) -> None:
        if self.pipe is not None:
            return
        import torch
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline

        k = self.k
        dtype = torch.float16
        kwargs = {"torch_dtype": dtype, "use_safetensors": True}
        if k.get("variant"):
            kwargs["variant"] = k["variant"]
        if k.get("vae"):
            kwargs["vae"] = AutoencoderKL.from_pretrained(k["vae"], torch_dtype=dtype)
        pipe = StableDiffusionXLPipeline.from_pretrained(k["model"], **kwargs)
        if k["use_ip_adapter"] and not self.bg_mode:
            pipe.load_ip_adapter(k["ip_adapter_repo"], subfolder=k["ip_adapter_subfolder"],
                                 weight_name=k["ip_adapter_weight"])
        if self.cfg["low_vram"]:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.cfg["device"])
        pipe.vae.enable_tiling()
        self.pipe = pipe

    def _apply_loras(self, shot: Shot) -> None:
        paths = []
        for name in shot.characters:
            c = self.p.character(name)
            if c and c.lora_path and self.p.abs(c.lora_path).exists():
                paths.append(self.p.abs(c.lora_path))
        for lp in paths:
            if str(lp) not in self._loras:
                adapter = f"lora{len(self._loras)}"
                self.pipe.load_lora_weights(str(lp.parent), weight_name=lp.name, adapter_name=adapter)
                self._loras[str(lp)] = adapter
        if paths:
            self.pipe.enable_lora()
            weight = 1.0 if len(paths) == 1 else 0.8
            self.pipe.set_adapters([self._loras[str(lp)] for lp in paths], adapter_weights=[weight] * len(paths))
        elif self._loras:
            self.pipe.disable_lora()

    # -------------------------------------------------------- main
    def generate(self, shot: Shot, out_path: Path) -> Path:
        out_path = Path(out_path)
        w, h = self.size

        if shot.user_keyframe:  # 1) your own image wins (in cutout mode it is used as the background)
            fit_cover(Image.open(self.p.abs(shot.user_keyframe)), w, h).save(out_path)
            return out_path

        if self.bg_mode:  # cutout mode: paint ONLY the background, characters are added later
            return self._background(shot, out_path, w, h)

        refs = self._references(shot)
        if self.k["backend"] == "none":
            if not refs:
                raise RuntimeError(
                    f"Shot {shot.id} has no keyframe. Upload an image for this shot, give a character a "
                    "reference image, or set keyframes.backend to 'sdxl' in config.yaml."
                )
            fit_cover(Image.open(refs[0]), w, h).save(out_path)
            return out_path

        import torch

        self._load()
        self._apply_loras(shot)
        kwargs = dict(
            prompt=keyframe_prompt(self.p, shot),
            negative_prompt=(self.p.negative + ", " if self.p.negative else "") + self.k["negative"],
            width=w, height=h,
            num_inference_steps=self.k["steps"],
            guidance_scale=self.k["guidance"],
            generator=torch.Generator("cpu").manual_seed(shot.seed),
        )
        if self.k["use_ip_adapter"]:
            if refs:
                self.pipe.set_ip_adapter_scale(self.k["ip_adapter_scale"])
                kwargs["ip_adapter_image"] = reference_sheet([Image.open(r) for r in refs])
            else:  # adapter is loaded, so it must get *an* image; scale 0 switches it off
                self.pipe.set_ip_adapter_scale(0.0)
                kwargs["ip_adapter_image"] = Image.new("RGB", (224, 224), "gray")
        self.pipe(**kwargs).images[0].save(out_path)
        return out_path

    def _background(self, shot: Shot, out_path: Path, w: int, h: int) -> Path:
        if self.k["backend"] == "none":
            from .cutout import plain_background

            plain_background(w, h).save(out_path)
            return out_path
        import torch

        self._load()
        self.pipe(
            prompt=background_prompt(self.p, shot),
            negative_prompt=(self.p.negative + ", " if self.p.negative else "") + BG_NEGATIVE + ", " + self.k["negative"],
            width=w, height=h,
            num_inference_steps=self.k["steps"],
            guidance_scale=self.k["guidance"],
            generator=torch.Generator("cpu").manual_seed(shot.seed),
        ).images[0].save(out_path)
        return out_path

    def unload(self) -> None:
        self.pipe = None
        self._loras.clear()
        free_gpu()
