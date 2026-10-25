# LocalVid Studio

Make short animated videos (up to about 1 minute) **on your own computer, for free**, from a prompt or script plus your own character images.

The goal that shapes the whole design: **the characters, their voices and the music must not change from the first shot to the last.**

```
idea / script ──► shots ──► voices ──► keyframe images ──► animated clips ──► music ──► final MP4
                  (fixed cast)  (one voice per character)  (same look every shot)       (one track, ducked under speech)
```

## What it does

- **Any story, any style.** You define a cast (name, look, reference image, voice) and write or generate a script. Monkeys, robots, kids' tales, anything.
- **Characters stay exactly the same (cutout mode, default).** The AI paints only the *background*; your own character images are cut out and animated on top (idle sway, bounce while speaking, camera move), so a character cannot change between shots. Fully AI-animated clips (`ltx`, `wan`) are available but characters can drift.
- **Voices stay the same.** One fixed voice per character, loudness-matched, cached so re-runs never change them. You can clone a voice from a short clip (experimental).
- **One music track for the whole video.** Generate it from a prompt, or use your own song. It is automatically lowered while someone speaks.
- **Subtitles** burned in (optional), **one-shot redo** for any shot you don't like, and every stage is **cached** so you never redo finished work.
- **Works offline** after the first model download. Browser UI (`run.bat`) and a command line.

## Quick start (Windows 11, NVIDIA GPU)

1. Install **Python 3.12**, **Git** and a recent **NVIDIA driver** (details in [docs/SETUP.md](docs/SETUP.md)).
2. ```
   git clone https://github.com/<you>/localvid-studio.git
   cd localvid-studio
   setup.bat        # one time: venv + PyTorch (CUDA 12.8) + packages + GPU check
   run.bat          # opens the app in your browser
   ```
3. In the app: **create a project** → add **characters** → paste a **script** (try `projects/example/script.txt`) → **Make the whole video**.

Not sure your install works yet? Set `clips.backend: kenburns` and `keyframes.backend: none` in `config.yaml` and use your own images: that path uses no AI models and finishes in seconds, so you can check the voices, music and assembly first.

## Script format

```
SCENE: Wide shot of a jungle at sunrise. Bobo and Mimi sit on a branch.
Bobo: What a beautiful morning!
Mimi: And I am very hungry.
SCENE: Close-up of Bobo sniffing the air.
Bobo: I smell bananas!
```

Each `SCENE:` becomes one 3-5 second clip. `NAME:` lines are spoken by that character. Or give the app a one-line idea and let a local LLM ([Ollama](https://ollama.com)) draft the script.

## How consistency is enforced

| What | How it stays the same |
|---|---|
| Look | **Cutout mode: your own character images are pasted into every frame, so they cannot change.** In the fully-AI modes: fixed style text + each character's description in **every** prompt; reference image through IP-Adapter; optional character **LoRA**; or **your own image** for any shot (always used as-is) |
| Voice | One Kokoro voice id (or one reference clip) per character; every line loudness-normalised; cached by voice + text |
| Music | One track for the whole video: your own file, or one generated track looped, with a fade in/out |
| Motion | Each clip starts from its keyframe and uses a "stable character" motion prompt; short clips limit drift |

Honest limits: AI video models can still wobble a character inside a clip, and two characters in one AI-generated image are the hardest case. The most reliable recipe is reference image + LoRA (or your own keyframes) and short shots. See [docs/USAGE.md](docs/USAGE.md).

## Hardware

Built for a laptop with an **8 GB NVIDIA GPU** (target: RTX 5050, 24 GB RAM). Models run one at a time with CPU offload and are unloaded between stages. Bigger GPUs can set `low_vram: false` for speed.

Rough expectation: an AI clip takes minutes, so a one-minute video is on the order of an hour or more. These are estimates, not measurements on your machine; run a single shot first to see your real speed.

## What has been tested

- Automated tests (`pip install pytest`, then `python -m pytest -q`) run **without a GPU or models**: script parsing, caching/invalidation rules, and a full run that builds a video with voices, camera-move clips, ducked music and subtitles using real ffmpeg.
- The AI stages (SDXL + IP-Adapter, LTX-Video, Kokoro, MusicGen, F5-TTS) use the documented `diffusers` / `transformers` / Kokoro APIs but **have not yet been run on an RTX 5050**. Expect to fix small first-run issues; if something fails, the log in the app (or the terminal) shows exactly where. The Wan 2.2 and F5-TTS backends are marked experimental.

## Project layout

```
app.py                  browser UI (Gradio)
config.yaml             the few settings you will actually change
studio/                 the pipeline (script, voices, keyframes, clips, music, assemble)
scripts/                Windows setup + GPU check
docs/                   SETUP, USAGE, ARCHITECTURE
projects/example/       a ready-to-run example story
tests/                  CPU-only smoke tests
```

## Documentation

- [docs/SETUP.md](docs/SETUP.md): install, first-run downloads, troubleshooting
- [docs/USAGE.md](docs/USAGE.md): workflow, consistency tips, command line, settings
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how the pieces fit, how to add a model backend

## License

MIT for this code (see [LICENSE](LICENSE)). The AI models it downloads have their own licenses; check them before commercial use.
