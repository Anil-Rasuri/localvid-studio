# Setup (Windows 11, NVIDIA GPU)

Written for an RTX 5050 Laptop GPU (8 GB), but works for any recent NVIDIA card.

## 1. Install the basics (once)

| Tool | Why | How |
|---|---|---|
| **NVIDIA driver, 570 or newer** | RTX 50-series needs a recent driver for CUDA 12.8 | nvidia.com/drivers (or GeForce Experience / NVIDIA app) |
| **Python 3.12** (3.11 also works) | the app | python.org/downloads. Tick **"Add python.exe to PATH"** and keep the **py launcher** option. Avoid 3.13+ (some packages lag behind). |
| **Git** | to clone / push the project | git-scm.com |
| **ffmpeg** | joins video + audio | `winget install Gyan.FFmpeg`, then open a **new** terminal. (A bundled fallback copy is installed by `setup.bat`, but a real install is more reliable.) |

## 2. Install the app

```
git clone https://github.com/<you>/localvid-studio.git
cd localvid-studio
setup.bat
```

`setup.bat` creates `.venv`, installs **PyTorch built for CUDA 12.8** first, then the other packages, then runs `scripts\check_gpu.py`.

> **Why PyTorch is installed separately:** the RTX 50-series (Blackwell, compute capability 12.0) only works with PyTorch builds made for CUDA 12.8 or newer. Older guides and the default `pip install torch` can give "no kernel image is available". If the cu128 index ever stops working, get the current command from <https://pytorch.org/get-started/locally/> and run it with `.venv\Scripts\python.exe -m pip ...`.

The GPU check must end with **"All good."** If it does not, fix that before going further.

## 3. Start

```
run.bat
```

The first time you generate, models download automatically from Hugging Face and are cached (plan for **tens of GB** in total: SDXL, the IP-Adapter, the video model, Kokoro, MusicGen). To keep them on another drive, set an environment variable before starting, for example `setx HF_HOME D:\hf_cache`, then open a new terminal.

## 4. Optional extras

**Script writing with a local LLM**
Install [Ollama](https://ollama.com), then `ollama pull qwen2.5:7b`. Keep Ollama running. The "Draft with local LLM" button in the Story tab will use it. You can change the model in `config.yaml` (`llm.model`).

**Voice cloning (experimental)**
`.venv\Scripts\python.exe -m pip install f5-tts`
This can replace your PyTorch build with a different one. Afterwards run
`.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
and `.venv\Scripts\python.exe scripts\check_gpu.py` again. Then upload a clean ~10 second voice sample for a character in tab 2.

**Character LoRA**
Not trained by this app. Train one on your character images with any SDXL LoRA trainer (kohya_ss, ai-toolkit, or the diffusers training script), then put the `.safetensors` path in the character's "LoRA" field.

## Languages for dialogue

- English (US/UK) and Hindi work out of the box through Kokoro (voice ids starting `af_ am_ bf_ bm_` for English, `hf_ hm_` for Hindi).
- Telugu and many other languages are not covered by the default voice engine. Voice cloning with a language-capable model, or adding another TTS backend in `studio/voices.py`, is the route; see docs/ARCHITECTURE.md.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no kernel image is available for execution on the device` | Wrong PyTorch build. Reinstall with the cu128 command above. |
| `CUDA out of memory` | In `config.yaml`: `clips.long_side: 512`, keep `low_vram: true`, close games/browsers. The app already falls back to a camera-move clip for that shot if `fallback_on_error` is true. |
| Very slow clips | Normal on an 8 GB laptop GPU. Use fewer / shorter shots, or `clips.backend: kenburns` for calm shots. Check Task Manager that the GPU (not CPU) is busy. |
| `ffmpeg was not found` | `winget install Gyan.FFmpeg`, then open a new terminal. |
| Kokoro / espeak error | Install espeak-ng from its GitHub releases page (espeak-ng.msi), then restart the app. |
| `setup.bat` window closes instantly | Run it from a terminal: `powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1` to see the error. |
| Port 7860 already in use | Close the old app window, or set `GRADIO_SERVER_PORT=7861` before `run.bat`. |
| Hugging Face download fails or is gated | Re-run; downloads resume. Some models need you to accept a license on huggingface.co and run `huggingface-cli login`. |

If you hit an error that is not listed, copy the last 30 lines of the terminal window; the stage name ("Clips: shot 3/11") tells you where it failed.
