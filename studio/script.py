"""Script handling.

Script format (plain text, one thing per line):

    SCENE: Two monkeys sit on a branch at sunrise.      <- starts a new shot (what we SEE)
    BOBO: Look, the sun is coming up!                   <- a spoken line in that shot
    MIMI: It is beautiful.
    SCENE: Bobo finds a banana.
    # lines starting with # are comments

Optionally, a local LLM (Ollama) can write this script from a one-line idea.
"""
from __future__ import annotations

import re

from .project import Character, Line, Project, Shot, Voice

SCRIPT_HELP = """SCENE: <what we see: place, who is there, what happens>
NAME: <what that character says>
NAME: <another line>
SCENE: <next shot>   (each SCENE becomes one ~4 second clip)
# comment lines start with #"""

SCENE_RE = re.compile(r"^\s*(?:scene|shot)\s*\d*\s*[:\-]\s*(.*)$", re.IGNORECASE)
DIALOGUE_RE = re.compile(r"^\s*([^\W\d_][\w .'\-]{0,29}?)\s*[:：]\s*(.+)$")

DEFAULT_VOICES = ["af_heart", "am_adam", "bf_emma", "bm_george", "af_bella", "am_michael"]


def _strip_quotes(text: str) -> str:
    return text.strip().strip('"“”').strip()


def parse_script(text: str, known_names: list[str] | tuple = ()) -> list[Shot]:
    """Turn script text into shots (dialogue, who is on screen). Deterministic, no AI involved."""
    known = {n.lower(): n for n in known_names}
    shots: list[Shot] = []
    current: Shot | None = None

    def new_shot(visual: str = "") -> Shot:
        shot = Shot(id=len(shots) + 1, visual=visual.strip())
        shots.append(shot)
        return shot

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = SCENE_RE.match(line)
        if m:
            current = new_shot(m.group(1))
            continue
        m = DIALOGUE_RE.match(line)
        if m:
            name = m.group(1).strip()
            speaker = known.get(name.lower(), name)
            if current is None:
                current = new_shot()
            current.dialogue.append(Line(speaker=speaker, text=_strip_quotes(m.group(2))))
            continue
        # a plain line continues the scene description
        if current is None:
            current = new_shot(line)
        else:
            current.visual = f"{current.visual} {line}".strip()

    names = list(known_names) + [s for s in speakers(shots) if s.lower() not in known]
    for shot in shots:
        on_screen: list[str] = []
        for ln in shot.dialogue:
            if ln.speaker not in on_screen:
                on_screen.append(ln.speaker)
        for name in names:
            if name not in on_screen and re.search(rf"\b{re.escape(name)}\b", shot.visual, re.IGNORECASE):
                on_screen.append(name)
        shot.characters = on_screen
    return shots


def speakers(shots: list[Shot]) -> list[str]:
    seen: list[str] = []
    for shot in shots:
        for ln in shot.dialogue:
            if ln.speaker not in seen:
                seen.append(ln.speaker)
    return seen


def ensure_characters(project: Project) -> list[str]:
    """Create a character (with a distinct default voice) for every speaker that does not have one yet."""
    added = []
    for name in speakers(project.shots):
        if project.character(name) is None:
            voice_id = DEFAULT_VOICES[len(project.characters) % len(DEFAULT_VOICES)]
            project.characters.append(Character(name=name, voice=Voice(voice_id=voice_id)))
            added.append(name)
    return added


def assign_seeds(project: Project) -> None:
    for shot in project.shots:
        shot.seed = project.seed + shot.id


# ---------------------------------------------------------------- optional LLM
def write_script_with_llm(idea: str, character_names: list[str], total_seconds: int, cfg: dict) -> str:
    """Ask a local Ollama model to write a script in the format above."""
    import requests

    llm = cfg["llm"]
    shot_len = float(llm.get("shot_seconds", 4.0))
    n_scenes = max(2, round(total_seconds / shot_len))
    if character_names:
        cast = f"Use only these characters: {', '.join(character_names)}. Use exactly these spellings."
    else:
        cast = "Invent 1 to 3 characters with short one-word names."
    prompt = (
        f"You are a screenwriter for a short animated video of about {total_seconds} seconds.\n"
        "Write the script in EXACTLY this plain-text format and nothing else:\n\n"
        "SCENE: <one sentence describing what we SEE: setting, characters, action>\n"
        "NAME: <spoken line>\n"
        "NAME: <spoken line>\n\n"
        "Rules:\n"
        f"- {cast}\n"
        f"- Write {n_scenes} scenes (about {shot_len:.0f} seconds each).\n"
        "- At most two spoken lines per scene, each at most 12 words. Some scenes may have no dialogue.\n"
        "- Scene descriptions must be concrete and filmable. Do not mention sound or music in them.\n"
        "- No markdown, no numbering, no commentary before or after.\n\n"
        f"Story idea: {idea}\n"
    )
    try:
        resp = requests.post(
            f"{llm['host']}/api/generate",
            json={"model": llm["model"], "prompt": prompt, "stream": False, "options": {"temperature": 0.7}},
            timeout=900,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {llm['host']}. Install it from https://ollama.com, run "
            f"`ollama pull {llm['model']}`, then try again. (You can also write the script by hand.)"
        ) from exc
    text = resp.json().get("response", "")
    text = re.sub(r"^```[a-z]*\s*$", "", text, flags=re.MULTILINE).strip()
    return text
