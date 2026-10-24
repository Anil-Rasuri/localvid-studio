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


def keyframe_prompt(project: Project, shot: Shot) -> str:
    return _join([project.style, *character_text(project, shot), shot.visual])


def video_prompt(project: Project, shot: Shot, suffix: str) -> str:
    return _join([shot.visual, *character_text(project, shot), project.style, suffix])
