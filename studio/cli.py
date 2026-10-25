"""Command line interface (no UI needed).

    python -m studio.cli new "My story"
    python -m studio.cli build my-story --script projects/my-story/script.txt
    python -m studio.cli run my-story                 # everything
    python -m studio.cli run my-story --stage clips   # one stage
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import STAGES, Pipeline
from .project import Project
from .script import assign_seeds, ensure_characters, parse_script


def _drop_cached(project: Project, kind: str, shot_id: int) -> None:
    """Delete saved images ('image') or clips ('clip') of a shot, whichever backend made them."""
    work = project.root / "work"
    names = ("keyframes", "backgrounds") if kind == "image" else tuple(
        d.name for d in work.glob("clips*") if d.is_dir())
    for name in names:
        for f in (work / name).glob(f"shot_{shot_id:02d}.*"):
            f.unlink(missing_ok=True)


def build_shots(project: Project, text: str) -> list[str]:
    """Parse script text into shots.

    Work that is still valid is kept (an unchanged scene keeps its image, an unchanged scene +
    dialogue keeps its clip); anything whose script text changed is deleted so it gets redone.
    """
    old = {s.id: s for s in project.shots}
    project.script = text
    project.shots = parse_script(text, [c.name for c in project.characters])
    added = ensure_characters(project)  # a default voice for every speaker without a character yet
    assign_seeds(project)
    for shot in project.shots:
        prev = old.get(shot.id)
        same_scene = prev is not None and prev.visual == shot.visual
        same_all = same_scene and [(ln.speaker, ln.text) for ln in prev.dialogue] == \
            [(ln.speaker, ln.text) for ln in shot.dialogue]
        if same_scene:
            shot.user_keyframe, shot.keyframe_path, shot.seed = prev.user_keyframe, prev.keyframe_path, prev.seed
        if same_all:
            shot.clip_path = prev.clip_path
        if not same_scene:
            _drop_cached(project, "image", shot.id)
        if not same_all:
            _drop_cached(project, "clip", shot.id)
    for sid in set(old) - {s.id for s in project.shots}:  # shots that no longer exist
        _drop_cached(project, "image", sid)
        _drop_cached(project, "clip", sid)
    project.save()
    return added


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m studio.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to a config yaml (default: config.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new", help="create a project")
    p_new.add_argument("name")

    p_build = sub.add_parser("build", help="turn a script file into shots")
    p_build.add_argument("project")
    p_build.add_argument("--script", required=True)

    p_run = sub.add_parser("run", help="generate the video")
    p_run.add_argument("project")
    p_run.add_argument("--stage", default="all", choices=["all", *STAGES])
    p_run.add_argument("--force", action="store_true", help="redo work even if cached")

    args = ap.parse_args(argv)
    if args.cmd == "new":
        proj = Project.create(args.name)
        print(f"Project ready: {proj.root}")
    elif args.cmd == "build":
        proj = Project.load(args.project)
        added = build_shots(proj, Path(args.script).read_text(encoding="utf-8"))
        print(f"{len(proj.shots)} shots built.")
        if added:
            print("New characters were created for: " + ", ".join(added) + " (edit their voices/images in project.json or the UI).")
    elif args.cmd == "run":
        proj = Project.load(args.project)
        out = Pipeline(proj, load_config(args.config)).run_stage(args.stage, force=args.force)
        if out:
            print(f"\nFinished: {out}")


if __name__ == "__main__":
    main()
