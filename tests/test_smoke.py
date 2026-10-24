"""CPU-only smoke tests: script parsing and a full run with the no-AI backends
(your own images + camera-move clips + silent voices + a synthetic music track).

    pip install pytest
    python -m pytest -q
"""
import re
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf
from PIL import Image, ImageDraw

from studio.cli import build_shots
from studio.config import load_config
from studio.pipeline import Pipeline
from studio.project import Character, Project, Voice
from studio.script import parse_script
from studio.util import ffmpeg_exe, media_duration

SCRIPT = """
# comment
SCENE: Bobo and Mimi sit on a branch at sunrise.
Bobo: Look, the sun is coming up!
Mimi: "It is beautiful."
SCENE: Bobo finds a banana.
Bobo: Mine!
SCENE: A wide shot of the jungle. Mimi laughs in the distance.
"""


def test_parse_script():
    shots = parse_script(SCRIPT, ["Bobo", "Mimi"])
    assert len(shots) == 3
    assert [(ln.speaker, ln.text) for ln in shots[0].dialogue] == [
        ("Bobo", "Look, the sun is coming up!"),
        ("Mimi", "It is beautiful."),
    ]
    assert shots[0].characters == ["Bobo", "Mimi"]
    assert shots[2].dialogue == []
    assert shots[2].characters == ["Mimi"]  # mentioned in the scene text only


def _picture(path, color):
    img = Image.new("RGB", (800, 600), color)
    ImageDraw.Draw(img).ellipse((250, 150, 550, 450), fill=(240, 220, 180))
    img.save(path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_end_to_end_without_ai_models(tmp_path):
    root = tmp_path / "proj"
    proj = Project.create("smoke", root=root, width=640, height=360, fps=24)
    for name, color in (("Bobo", (90, 140, 60)), ("Mimi", (60, 90, 150))):
        img = root / "assets" / f"{name}.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        _picture(img, color)
        proj.characters.append(Character(name=name, reference_image=f"assets/{name}.png",
                                         voice=Voice(backend="silent")))
    build_shots(proj, SCRIPT)

    t = np.linspace(0, 10, 10 * 24000, endpoint=False)
    sf.write(str(root / "assets" / "song.wav"), 0.3 * np.sin(2 * np.pi * 220 * t), 24000)
    proj.music.mode, proj.music.file = "file", "assets/song.wav"
    proj.save()

    cfg = load_config()
    cfg["keyframes"]["backend"] = "none"
    cfg["clips"]["backend"] = "kenburns"
    final = Pipeline(proj, cfg).run_all()

    assert final.exists()
    total = media_duration(final)
    expected = sum(max(cfg["shot"]["min_seconds"], s.audio_duration + cfg["shot"]["tail_pad"]) for s in proj.shots)
    assert abs(total - expected) < 0.6, (total, expected)

    info = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(final)], capture_output=True, text=True).stderr
    assert re.search(r"Video: h264", info) and re.search(r"Audio: aac", info)
    assert (root / "work" / "shots" / "subs.srt").exists()
