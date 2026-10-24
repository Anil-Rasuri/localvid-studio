"""Text-to-speech with ONE fixed voice per character.

Why the voices cannot drift:
* every line of a character is made from the same voice settings (same Kokoro voice id,
  or the same reference clip for cloning)
* every line is loudness-normalised to the same level
* results are cached by (voice settings + text), so re-running never changes a voice
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .project import Project, Shot, Voice

SR = 24000  # all dialogue audio is kept at 24 kHz mono until the final mix


def normalize(audio: np.ndarray, target_rms: float, peak: float = 0.95) -> np.ndarray:
    audio = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if rms < 1e-6:
        return audio
    audio = audio * (target_rms / rms)
    top = float(np.max(np.abs(audio)))
    if top > peak:
        audio = audio * (peak / top)
    return audio


def to_mono_sr(audio: np.ndarray, sr: int, target_sr: int = SR) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr and audio.size:
        n = int(round(len(audio) * target_sr / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


class VoiceEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._kokoro: dict = {}

    # -------------------------------------------------------- backends
    def _kokoro_pipe(self, lang: str):
        if lang not in self._kokoro:
            try:
                from kokoro import KPipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Kokoro is not installed. Run: pip install kokoro misaki[en]  (see docs/SETUP.md)"
                ) from exc
            self._kokoro[lang] = KPipeline(lang_code=lang)
        return self._kokoro[lang]

    def _render_kokoro(self, voice: Voice, text: str) -> tuple[np.ndarray, int]:
        lang = voice.lang or voice.voice_id[:1]
        chunks = []
        for result in self._kokoro_pipe(lang)(text, voice=voice.voice_id, speed=voice.speed):
            audio = result.audio if hasattr(result, "audio") else result[2]
            if audio is None:
                continue
            chunks.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for: {text!r}")
        return np.concatenate(chunks).astype(np.float32), 24000

    def _render_f5(self, voice: Voice, text: str, project_root: Path, tmp: Path) -> tuple[np.ndarray, int]:
        """Voice cloning with F5-TTS (experimental): same reference clip => same voice every time."""
        exe = shutil.which("f5-tts_infer-cli")
        if not exe:
            raise RuntimeError("F5-TTS is not installed. Run: pip install f5-tts  (see docs/SETUP.md)")
        ref = Path(voice.ref_audio)
        ref = ref if ref.is_absolute() else project_root / ref
        if not ref.exists():
            raise RuntimeError(f"Reference audio not found: {ref}")
        cmd = [exe, "--model", "F5TTS_v1_Base", "--ref_audio", str(ref), "--gen_text", text,
               "--output_dir", str(tmp.parent), "--output_file", tmp.name]
        if voice.ref_text:
            cmd += ["--ref_text", voice.ref_text]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0 or not tmp.exists():
            raise RuntimeError("F5-TTS failed:\n" + result.stderr[-1500:])
        audio, sr = sf.read(str(tmp), dtype="float32")
        return audio, sr

    @staticmethod
    def _render_silent(text: str) -> tuple[np.ndarray, int]:
        seconds = max(0.8, len(text.split()) / 2.6)
        return np.zeros(int(seconds * SR), dtype=np.float32), SR

    # -------------------------------------------------------- public
    def synth(self, voice: Voice, text: str, out_path: Path, project_root: Path) -> None:
        if voice.backend == "silent":
            audio, sr = self._render_silent(text)
        elif voice.backend == "f5":
            audio, sr = self._render_f5(voice, text, project_root, out_path.with_suffix(".f5tmp.wav"))
            out_path.with_suffix(".f5tmp.wav").unlink(missing_ok=True)
        else:
            audio, sr = self._render_kokoro(voice, text)
        audio = normalize(to_mono_sr(audio, sr), self.cfg["voice"]["target_rms"])
        sf.write(str(out_path), audio, SR)

    def unload(self) -> None:
        self._kokoro.clear()


def build_shot_audio(project: Project, shot: Shot, engine: VoiceEngine, cfg: dict, force: bool = False) -> None:
    """Make one wav per dialogue line (cached), then join them into one wav for the shot."""
    audio_dir = project.dir("work/audio")
    gap = cfg["shot"]["dialogue_gap"]
    pieces: list[np.ndarray] = []
    t = 0.0
    for idx, line in enumerate(shot.dialogue, 1):
        char = project.character(line.speaker)
        voice = char.voice if char else Voice()
        key = hashlib.md5(json.dumps([asdict(voice), line.text], sort_keys=True).encode()).hexdigest()[:8]
        wav = audio_dir / f"shot{shot.id:02d}_line{idx:02d}_{key}.wav"
        if force or not wav.exists():
            engine.synth(voice, line.text, wav, project.root)
        data, sr = sf.read(str(wav), dtype="float32")
        data = to_mono_sr(data, sr)
        if pieces:
            pieces.append(np.zeros(int(gap * SR), dtype=np.float32))
            t += gap
        line.audio_path = project.rel(wav)
        line.start = round(t, 3)
        line.duration = round(len(data) / SR, 3)
        pieces.append(data)
        t += len(data) / SR
    if not pieces:
        shot.audio_path, shot.audio_duration = "", 0.0
        return
    combined = np.concatenate(pieces)
    out = audio_dir / f"shot{shot.id:02d}.wav"
    sf.write(str(out), combined, SR)
    shot.audio_path = project.rel(out)
    shot.audio_duration = round(len(combined) / SR, 3)
