"""Project data model. A project is a folder with project.json + assets/ + work/ + output/.

Everything that must stay the same across the whole video (characters, their reference
images, their voices, the style text, the music) lives here and is reused by every shot.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .config import PROJECTS_DIR
from .util import slugify


@dataclass
class Voice:
    backend: str = "kokoro"   # kokoro | f5 (clone from ref_audio) | silent
    voice_id: str = "af_heart"
    lang: str = ""            # Kokoro language code; empty = taken from the first letter of voice_id
    speed: float = 1.0
    ref_audio: str = ""       # reference clip for cloning backends (relative to project root)
    ref_text: str = ""        # transcript of ref_audio (optional)


@dataclass
class Character:
    name: str
    description: str = ""     # appearance text injected into EVERY prompt this character is in
    reference_image: str = ""
    lora_path: str = ""       # optional character LoRA (.safetensors) for the strongest consistency
    voice: Voice = field(default_factory=Voice)


@dataclass
class Line:
    speaker: str
    text: str
    audio_path: str = ""
    start: float = 0.0        # seconds from the start of the shot
    duration: float = 0.0


@dataclass
class Shot:
    id: int
    visual: str = ""
    dialogue: list[Line] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    user_keyframe: str = ""   # an image you supplied for this shot (always wins over generation)
    keyframe_path: str = ""
    clip_path: str = ""
    audio_path: str = ""
    audio_duration: float = 0.0
    seed: int = 0


@dataclass
class MusicSettings:
    mode: str = "none"        # none | generate | file
    prompt: str = "gentle playful orchestral music, light and cheerful"
    file: str = ""            # your own song / track (relative to project root or absolute)
    volume: float = 0.18
    duck: bool = True         # lower the music while someone speaks


def _pick(cls, data: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in (data or {}).items() if k in names}


@dataclass
class Project:
    name: str
    style: str = "2D cartoon animation, bright colors, clean outlines, consistent character design"
    negative: str = ""
    width: int = 1280
    height: int = 720
    fps: int = 24
    seed: int = 1234
    subtitles: bool = True
    script: str = ""
    characters: list[Character] = field(default_factory=list)
    shots: list[Shot] = field(default_factory=list)
    music: MusicSettings = field(default_factory=MusicSettings)
    root: Path = field(default=Path("."), repr=False)

    # ------------------------------------------------------------ paths
    def abs(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    def rel(self, p: str | Path) -> str:
        p = Path(p)
        try:
            return p.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(p)

    def dir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def import_asset(self, src: str | Path) -> str:
        """Copy a file into assets/ (unless it is already inside the project) and return its relative path."""
        src = Path(src)
        try:
            inside = src.resolve().is_relative_to(self.root.resolve())
        except AttributeError:  # Python < 3.9 (not supported, kept for safety)
            inside = str(src.resolve()).startswith(str(self.root.resolve()))
        if inside:
            return self.rel(src)
        dest = self.dir("assets") / src.name
        i = 1
        while dest.exists() and dest.read_bytes() != src.read_bytes():
            dest = dest.with_name(f"{src.stem}_{i}{src.suffix}")
            i += 1
        if not dest.exists():
            shutil.copy2(src, dest)
        return self.rel(dest)

    # ------------------------------------------------------------ lookup
    def character(self, name: str) -> Character | None:
        for c in self.characters:
            if c.name.lower() == str(name).strip().lower():
                return c
        return None

    # ------------------------------------------------------------ (de)serialise
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("root", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        chars = []
        for c in d.get("characters", []):
            voice = Voice(**_pick(Voice, c.get("voice", {})))
            chars.append(Character(**{**_pick(Character, c), "voice": voice}))
        shots = []
        for s in d.get("shots", []):
            lines = [Line(**_pick(Line, ln)) for ln in s.get("dialogue", [])]
            shots.append(Shot(**{**_pick(Shot, s), "dialogue": lines}))
        music = MusicSettings(**_pick(MusicSettings, d.get("music", {})))
        base = _pick(cls, d)
        for key in ("characters", "shots", "music", "root"):
            base.pop(key, None)
        return cls(**base, characters=chars, shots=shots, music=music)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "project.json").write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, ref: str | Path) -> "Project":
        """`ref` is a project folder name inside projects/ or a path to a project folder."""
        folder = PROJECTS_DIR / str(ref)
        if not folder.is_dir():
            folder = Path(ref)
        if not (folder / "project.json").exists():
            raise FileNotFoundError(f"No project.json in {folder}")
        proj = cls.from_dict(json.loads((folder / "project.json").read_text(encoding="utf-8")))
        proj.root = folder.resolve()
        return proj

    @classmethod
    def create(cls, name: str, root: str | Path | None = None, **kwargs) -> "Project":
        folder = Path(root) if root else PROJECTS_DIR / slugify(name)
        if (folder / "project.json").exists():
            return cls.load(folder)
        proj = cls(name=name, root=folder.resolve(), **kwargs)
        proj.save()
        return proj


def list_projects() -> list[str]:
    if not PROJECTS_DIR.exists():
        return []
    return sorted(p.name for p in PROJECTS_DIR.iterdir() if (p / "project.json").exists())
