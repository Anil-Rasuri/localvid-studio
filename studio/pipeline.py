"""The pipeline: voices -> keyframes -> clips -> music -> assemble.

Voices run first because their length decides how long each clip must be.
Each stage caches its output on disk, so re-running only does missing work, and a single
shot can be redone without touching the others.
"""
from __future__ import annotations

from pathlib import Path

from .assemble import assemble
from .config import load_config
from .music import prepare_music
from .project import Project, Shot
from .script import assign_seeds
STAGES = ["voices", "keyframes", "clips", "music", "assemble"]


class Pipeline:
    def __init__(self, project: Project, cfg: dict | None = None, progress=None):
        self.p = project
        self.cfg = cfg or load_config()
        self._progress = progress or (lambda frac, msg: None)
        self.messages: list[str] = []

    # -------------------------------------------------------- helpers
    def log(self, msg: str, frac: float | None = None) -> None:
        print(msg, flush=True)
        self.messages.append(msg)
        self._progress(frac, msg)

    def shot_seconds(self, shot: Shot) -> float:
        s = self.cfg["shot"]
        need = shot.audio_duration + s["tail_pad"] if shot.audio_duration else 0.0
        return min(max(need, s["min_seconds"]), s["max_seconds"])

    def _targets(self, only: list[int] | None) -> list[Shot]:
        return [s for s in self.p.shots if only is None or s.id in only]

    # -------------------------------------------------------- stages
    def stage_voices(self, force: bool = False, only: list[int] | None = None) -> None:
        from .voices import VoiceEngine, build_shot_audio

        engine = VoiceEngine(self.cfg)
        shots = self._targets(only)
        for i, shot in enumerate(shots):
            self.log(f"Voices: shot {shot.id}/{len(self.p.shots)}", i / max(1, len(shots)))
            build_shot_audio(self.p, shot, engine, self.cfg, force=force)
        engine.unload()
        self.p.save()

    def stage_keyframes(self, force: bool = False, only: list[int] | None = None) -> None:
        from .keyframes import KeyframeGenerator

        if not all(s.seed for s in self.p.shots):
            assign_seeds(self.p)
        gen = KeyframeGenerator(self.p, self.cfg)
        out_dir = self.p.dir("work/keyframes")
        shots = self._targets(only)
        try:
            for i, shot in enumerate(shots):
                path = out_dir / f"shot_{shot.id:02d}.png"
                if path.exists() and not force and not shot.user_keyframe:
                    shot.keyframe_path = self.p.rel(path)
                    continue
                self.log(f"Keyframes: shot {shot.id}/{len(self.p.shots)}", i / max(1, len(shots)))
                gen.generate(shot, path)
                shot.keyframe_path = self.p.rel(path)
                self.p.save()
        finally:
            gen.unload()

    def stage_clips(self, force: bool = False, only: list[int] | None = None) -> None:
        from .clips import ClipGenerator, ken_burns
        from .util import fit_size as _fit

        gen = ClipGenerator(self.p, self.cfg)
        out_dir = self.p.dir("work/clips")
        shots = self._targets(only)
        try:
            for i, shot in enumerate(shots):
                path = out_dir / f"shot_{shot.id:02d}.mp4"
                if path.exists() and not force:
                    shot.clip_path = self.p.rel(path)
                    continue
                kf = self.p.abs(shot.keyframe_path) if shot.keyframe_path else None
                if not kf or not kf.exists():
                    raise RuntimeError(f"Shot {shot.id} has no keyframe yet. Run the keyframe stage first.")
                seconds = self.shot_seconds(shot)
                self.log(f"Clips ({gen.backend}): shot {shot.id}/{len(self.p.shots)}", i / max(1, len(shots)))
                try:
                    gen.generate(shot, kf, path, seconds)
                except Exception as exc:  # noqa: BLE001
                    if gen.backend == "kenburns" or not self.cfg["clips"]["fallback_on_error"]:
                        raise
                    self.log(f"  AI clip failed for shot {shot.id} ({type(exc).__name__}: {str(exc)[:160]}). "
                             "Using a camera-move clip instead.")
                    gen.unload()
                    w, h = _fit(self.p.width, self.p.height, 1280, 2)
                    ken_burns(kf, path, seconds, w, h, self.p.fps, variant=shot.id)
                shot.clip_path = self.p.rel(path)
                self.p.save()
        finally:
            gen.unload()

    def stage_music(self, force: bool = False) -> Path | None:
        total = sum(self.shot_seconds(s) for s in self.p.shots)
        if self.p.music.mode == "generate":
            self.log("Music: generating one track for the whole video")
        return prepare_music(self.p, self.cfg, total, force=force)

    def stage_assemble(self, music: Path | None = None) -> Path:
        self.log("Assembling final video", 0.0)
        out = assemble(self.p, self.cfg, music, log=lambda m: self.log(m))
        self.p.save()
        self.log(f"Done: {out}", 1.0)
        return out

    # -------------------------------------------------------- public entry points
    def run_all(self, force: bool = False) -> Path:
        if not self.p.shots:
            raise RuntimeError("The project has no shots. Build them from a script first.")
        if not all(s.seed for s in self.p.shots):
            assign_seeds(self.p)
        self.stage_voices(force=force)
        self.stage_keyframes(force=force)
        self.stage_clips(force=force)
        music = self.stage_music(force=force)
        return self.stage_assemble(music)

    def run_stage(self, stage: str, force: bool = False) -> Path | None:
        if stage == "all":
            return self.run_all(force)
        if stage == "voices":
            self.stage_voices(force)
        elif stage == "keyframes":
            self.stage_keyframes(force)
        elif stage == "clips":
            self.stage_clips(force)
        elif stage == "music":
            self.stage_music(force)
        elif stage == "assemble":
            return self.stage_assemble(self.stage_music())
        else:
            raise ValueError(f"Unknown stage: {stage}")
        return None

    def regenerate_shot(self, shot_id: int, redo_keyframe: bool, new_seed: bool = True) -> Path:
        """Redo one shot (new seed by default) and rebuild the final video."""
        shot = next((s for s in self.p.shots if s.id == shot_id), None)
        if shot is None:
            raise ValueError(f"No shot {shot_id}")
        if new_seed:
            shot.seed += 7919
        for attr in (["keyframe_path"] if redo_keyframe else []) + ["clip_path"]:
            old = getattr(shot, attr)
            if old and self.p.abs(old).exists():
                self.p.abs(old).unlink()
            setattr(shot, attr, "")
        self.p.save()
        self.stage_voices(only=[shot_id])
        self.stage_keyframes(only=[shot_id])
        self.stage_clips(only=[shot_id])
        return self.stage_assemble(self.stage_music())
