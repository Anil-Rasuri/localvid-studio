"""LocalVid Studio - browser UI.   Run:  python app.py   (or double-click run.bat on Windows)"""
from __future__ import annotations

import traceback
from pathlib import Path

import gradio as gr

from studio import script as scr
from studio.cli import build_shots
from studio.config import load_config
from studio.pipeline import Pipeline
from studio.project import Character, Project, Voice, list_projects

MAX_CHARS = 5  # character slots in the UI (the engine itself has no limit)
VOICES = ["af_heart", "af_bella", "af_nicole", "af_sky", "am_adam", "am_michael",
          "bf_emma", "bm_george", "bm_lewis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
          "silent (no audio)"]
SIZES = {
    "Landscape 1280x720": (1280, 720),
    "Vertical 720x1280": (720, 1280),
    "Square 1024x1024": (1024, 1024),
}


# ------------------------------------------------------------------ helpers
def _need(name: str) -> Project:
    if not name:
        raise gr.Error("Create or pick a project first (tab 1).")
    return Project.load(name)


def _exists(p: Project, rel: str):
    return str(p.abs(rel)) if rel and p.abs(rel).exists() else None


def _adopt(p: Project, path) -> str:
    """Copy an uploaded file into the project's assets folder (no-op if already there)."""
    return p.import_asset(path) if path else ""


def _size_label(p: Project) -> str:
    for label, (w, h) in SIZES.items():
        if (w, h) == (p.width, p.height):
            return label
    return next(iter(SIZES))


def _shots_table(p: Project) -> list[list]:
    rows = []
    for s in p.shots:
        talk = " | ".join(f"{ln.speaker}: {ln.text}" for ln in s.dialogue)
        done = "".join([
            "voice " if s.audio_path else "",
            "image " if s.keyframe_path else "",
            "clip" if s.clip_path else "",
        ]).strip() or "-"
        rows.append([s.id, s.visual, talk, done])
    return rows


def _gallery(p: Project) -> list[str]:
    return [x for s in p.shots if (x := _exists(p, s.keyframe_path))]


def _final(p: Project):
    f = p.root / "output" / "final.mp4"
    return str(f) if f.exists() else None


# ------------------------------------------------------------------ project tab
def create_project(new_name: str):
    if not new_name.strip():
        raise gr.Error("Type a project name.")
    p = Project.create(new_name.strip())
    return gr.update(choices=list_projects(), value=p.root.name), ""


def load_project(name: str):
    """Fill every field in the UI from the saved project (order must match LOAD_OUTPUTS)."""
    if not name:
        blank_chars = ["", "", None, VOICES[0], None, ""] * MAX_CHARS
        return ["", "", next(iter(SIZES)), 24, 1234, True, *blank_chars,
                "", "none", "", None, 0.18, True, [], None, []]
    p = Project.load(name)
    out: list = [p.style, p.negative, _size_label(p), p.fps, p.seed, p.subtitles]
    for i in range(MAX_CHARS):
        if i < len(p.characters):
            c = p.characters[i]
            vid = "silent (no audio)" if c.voice.backend == "silent" else c.voice.voice_id
            out += [c.name, c.description, _exists(p, c.reference_image), vid,
                    _exists(p, c.voice.ref_audio), c.lora_path]
        else:
            out += ["", "", None, VOICES[0], None, ""]
    out += [p.script, p.music.mode, p.music.prompt, _exists(p, p.music.file), p.music.volume, p.music.duck,
            _shots_table(p), _final(p), _gallery(p)]
    return out


def save_settings(name, style, negative, size, fps, seed, subtitles):
    p = _need(name)
    p.style, p.negative, p.fps, p.seed, p.subtitles = style, negative, int(fps), int(seed), bool(subtitles)
    p.width, p.height = SIZES[size]
    p.save()
    return "Saved."


# ------------------------------------------------------------------ characters tab
def save_characters(name, *vals):
    p = _need(name)
    old = {c.name.lower(): c for c in p.characters}
    chars = []
    for i in range(MAX_CHARS):
        cname, desc, ref_img, voice_id, ref_audio, lora = vals[i * 6:(i + 1) * 6]
        if not (cname or "").strip():
            continue
        cname = cname.strip()
        prev = old.get(cname.lower())
        speed = prev.voice.speed if prev else 1.0
        if ref_audio:
            voice = Voice(backend="f5", voice_id=voice_id, ref_audio=_adopt(p, ref_audio),
                          ref_text=prev.voice.ref_text if prev else "", speed=speed)
        elif str(voice_id).startswith("silent"):
            voice = Voice(backend="silent", voice_id="silent", speed=speed)
        else:
            voice = Voice(backend="kokoro", voice_id=voice_id, speed=speed)
        chars.append(Character(name=cname, description=(desc or "").strip(),
                               reference_image=_adopt(p, ref_img), lora_path=(lora or "").strip(), voice=voice))
    p.characters = chars
    p.save()
    if p.script.strip():  # re-detect who is on screen with the new cast (unchanged scenes keep their work)
        build_shots(p, p.script)
    return (f"Saved {len(chars)} character(s). Voices and looks are fixed for every shot. "
            "If you changed a character's look, tick 'Redo everything' in tab 5 so old images are replaced.")


# ------------------------------------------------------------------ story tab
def llm_script(name, idea, seconds):
    p = _need(name)
    if not idea.strip():
        raise gr.Error("Describe the story in a sentence or two.")
    try:
        return scr.write_script_with_llm(idea, [c.name for c in p.characters], int(seconds), load_config())
    except RuntimeError as exc:
        raise gr.Error(str(exc))


def make_shots(name, script_text):
    p = _need(name)
    if not script_text.strip():
        raise gr.Error("The script is empty.")
    added = build_shots(p, script_text)
    note = f"{len(p.shots)} shots built."
    if added:
        note += " New characters created: " + ", ".join(added) + " - open tab 2 to give them images/voices."
    return _shots_table(p), note


# ------------------------------------------------------------------ music tab
def save_music(name, mode, prompt, file, volume, duck):
    p = _need(name)
    p.music.mode, p.music.prompt = mode, prompt
    p.music.volume, p.music.duck = float(volume), bool(duck)
    p.music.file = _adopt(p, file) if file else p.music.file
    p.save()
    return "Saved."


# ------------------------------------------------------------------ generate tab
def _result(p: Project, pipe: Pipeline):
    p = Project.load(p.root)
    return "\n".join(pipe.messages), _final(p), _gallery(p), _shots_table(p)


def _run(name, stage, force, progress):
    p = _need(name)
    if not p.shots:
        raise gr.Error("No shots yet - build them from a script in tab 3.")
    pipe = Pipeline(p, load_config(), progress=lambda frac, msg: progress(frac, desc=msg))
    try:
        pipe.run_stage(stage, force=force)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise gr.Error(f"{type(exc).__name__}: {exc}")
    return _result(p, pipe)


def run_everything(name, force, progress=gr.Progress()):
    return _run(name, "all", force, progress)


def run_voices(name, progress=gr.Progress()):
    return _run(name, "voices", False, progress)


def run_images(name, progress=gr.Progress()):
    return _run(name, "keyframes", False, progress)


def run_clips(name, progress=gr.Progress()):
    return _run(name, "clips", False, progress)


def run_assemble(name, progress=gr.Progress()):
    return _run(name, "assemble", False, progress)


def redo_shot(name, shot_id, mode, new_image, progress=gr.Progress()):
    p = _need(name)
    sid = int(shot_id)
    shot = next((s for s in p.shots if s.id == sid), None)
    if shot is None:
        raise gr.Error(f"There is no shot {sid}.")
    if new_image:
        shot.user_keyframe = _adopt(p, new_image)
    redo_keyframe = mode.startswith("New image") or bool(new_image)
    p.save()
    pipe = Pipeline(p, load_config(), progress=lambda frac, msg: progress(frac, desc=msg))
    try:
        pipe.regenerate_shot(sid, redo_keyframe=redo_keyframe, new_seed=True)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise gr.Error(f"{type(exc).__name__}: {exc}")
    return _result(p, pipe)


# ------------------------------------------------------------------ layout
with gr.Blocks(title="LocalVid Studio") as demo:
    gr.Markdown("# LocalVid Studio\nPrompt / script + your images -> a short video with fixed characters, "
                "fixed voices and one music track. Everything runs on this computer.")

    with gr.Row():
        project_dd = gr.Dropdown(choices=list_projects(), label="Project", scale=3)
        new_name = gr.Textbox(label="New project name", scale=3)
        new_btn = gr.Button("Create", scale=1)

    # ---- 1. look
    with gr.Tab("1. Look & format"):
        style = gr.Textbox(label="Style (added to every shot)", lines=2)
        negative = gr.Textbox(label="Things to avoid (optional)")
        with gr.Row():
            size = gr.Dropdown(list(SIZES), value=next(iter(SIZES)), label="Video size")
            fps = gr.Slider(12, 30, step=1, value=24, label="Frames per second")
            seed = gr.Number(value=1234, precision=0, label="Seed")
            subtitles = gr.Checkbox(value=True, label="Burn in subtitles")
        s_btn = gr.Button("Save", variant="primary")
        s_msg = gr.Markdown()

    # ---- 2. characters
    char_inputs: list = []
    with gr.Tab("2. Characters"):
        gr.Markdown("Each character gets **one** appearance and **one** voice, reused in every shot. "
                    "Describe the look in words and add a reference image. To clone a voice, upload a ~10 s clean "
                    "clip (needs F5-TTS, see docs/SETUP.md); otherwise pick a built-in voice.")
        for i in range(MAX_CHARS):
            with gr.Accordion(f"Character {i + 1}", open=(i < 2)):
                with gr.Row():
                    c_name = gr.Textbox(label="Name (as used in the script)")
                    c_voice = gr.Dropdown(VOICES, value=VOICES[0], allow_custom_value=True, label="Voice")
                c_desc = gr.Textbox(label="Look (e.g. 'small brown monkey, red scarf, big round eyes')", lines=2)
                with gr.Row():
                    c_img = gr.Image(type="filepath", label="Reference image", height=200)
                    c_aud = gr.Audio(type="filepath", sources=["upload"], label="Voice sample to clone (optional)")
                c_lora = gr.Textbox(label="Character LoRA file path (optional, strongest consistency)")
                char_inputs += [c_name, c_desc, c_img, c_voice, c_aud, c_lora]
        c_btn = gr.Button("Save characters", variant="primary")
        c_msg = gr.Markdown()

    # ---- 3. story
    with gr.Tab("3. Story"):
        gr.Markdown("Write the script yourself, or let a local LLM (Ollama) draft it from an idea.")
        with gr.Row():
            idea = gr.Textbox(label="Story idea", scale=4, placeholder="Two monkeys race to a banana tree...")
            secs = gr.Slider(10, 60, step=5, value=30, label="Length (s)", scale=1)
            llm_btn = gr.Button("Draft with local LLM", scale=1)
        script_box = gr.Textbox(label="Script", lines=14, placeholder=scr.SCRIPT_HELP)
        shots_btn = gr.Button("Build shots from script", variant="primary")
        shots_msg = gr.Markdown()
        shots_tbl = gr.Dataframe(headers=["#", "What we see", "Dialogue", "Done"], interactive=False, wrap=True)

    # ---- 4. music
    with gr.Tab("4. Music"):
        m_mode = gr.Radio(["none", "generate", "file"], value="none", label="Music")
        m_prompt = gr.Textbox(label="Describe the music (for 'generate')")
        m_file = gr.Audio(type="filepath", sources=["upload"], label="Your own song / track (for 'file')")
        with gr.Row():
            m_vol = gr.Slider(0.02, 0.6, step=0.02, value=0.18, label="Music volume")
            m_duck = gr.Checkbox(value=True, label="Lower music while characters speak")
        m_btn = gr.Button("Save", variant="primary")
        m_msg = gr.Markdown()

    # ---- 5. generate
    with gr.Tab("5. Generate"):
        with gr.Row():
            go_btn = gr.Button("Make the whole video", variant="primary", scale=2)
            force = gr.Checkbox(label="Redo everything (ignore saved work)", scale=1)
        with gr.Row():
            b_v = gr.Button("Voices only")
            b_k = gr.Button("Images only")
            b_c = gr.Button("Clips only")
            b_a = gr.Button("Assemble only")
        log = gr.Textbox(label="Log", lines=8, interactive=False)
        video = gr.Video(label="Final video")
        gallery = gr.Gallery(label="Keyframes", columns=4, height=260)
        status_tbl = gr.Dataframe(headers=["#", "What we see", "Dialogue", "Done"], interactive=False, wrap=True)
        gr.Markdown("### Not happy with one shot?")
        with gr.Row():
            r_id = gr.Number(value=1, precision=0, label="Shot #")
            r_mode = gr.Radio(["Same image, new motion", "New image + motion"], value="Same image, new motion",
                              label="What to redo")
            r_img = gr.Image(type="filepath", label="...or use my own image for this shot", height=140)
            r_btn = gr.Button("Redo this shot")

    # ---- wiring
    load_outputs = [style, negative, size, fps, seed, subtitles, *char_inputs,
                    script_box, m_mode, m_prompt, m_file, m_vol, m_duck, shots_tbl, video, gallery]
    new_btn.click(create_project, new_name, [project_dd, new_name])
    project_dd.change(load_project, project_dd, load_outputs)
    s_btn.click(save_settings, [project_dd, style, negative, size, fps, seed, subtitles], s_msg)
    c_btn.click(save_characters, [project_dd, *char_inputs], c_msg)
    llm_btn.click(llm_script, [project_dd, idea, secs], script_box)
    shots_btn.click(make_shots, [project_dd, script_box], [shots_tbl, shots_msg])
    m_btn.click(save_music, [project_dd, m_mode, m_prompt, m_file, m_vol, m_duck], m_msg)
    gen_outputs = [log, video, gallery, status_tbl]
    go_btn.click(run_everything, [project_dd, force], gen_outputs)
    for btn, fn in ((b_v, run_voices), (b_k, run_images), (b_c, run_clips), (b_a, run_assemble)):
        btn.click(fn, project_dd, gen_outputs)
    r_btn.click(redo_shot, [project_dd, r_id, r_mode, r_img], gen_outputs)

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
