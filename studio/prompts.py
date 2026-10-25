"""Prompt building. The fixed style text and each character's fixed description are injected
into every shot, which is the cheapest and most reliable way to keep characters from drifting."""
from __future__ import annotations

from .project import Project, Shot


def _join(parts: list[str]) -> str:
    return ". ".join(p.strip().rstrip(".") for p in parts if p and p.strip())


def character_text(project: Project, shot: Shot) -> list[str]:
    out = []
    for name in shot.characters:
        c = project.character(name)
        if c:
            out.append(f"{c.name}: {c.description}" if c.description.strip() else c.name)
    return out


BG_NEGATIVE = "character, person, people, animal, creature, monkey, figure, face, text, watermark"


def background_prompt(project: Project, shot: Shot) -> str:
    """Prompt for a background with nobody in it (cutout mode): the scene text minus character names."""
    import re

    scene = shot.visual
    for c in project.characters:
        scene = re.sub(rf"\b{re.escape(c.name)}(?:'s)?\b", "", scene, flags=re.IGNORECASE)
    scene = re.sub(r"\b(close-?up|wide shot)\b( of)?", "", scene, flags=re.IGNORECASE)
    scene = re.sub(r"\s{2,}", " ", scene).strip(" ,.")
    return _join([project.style, "empty background scenery, no characters, no animals, no people, wide view", scene])


def keyframe_prompt(project: Project, shot: Shot) -> str:
    return _join([project.style, *character_text(project, shot), shot.visual])


def video_prompt(project: Project, shot: Shot, suffix: str) -> str:
    return _join([shot.visual, *character_text(project, shot), project.style, suffix])
