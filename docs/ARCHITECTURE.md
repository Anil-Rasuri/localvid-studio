# Architecture

## Pipeline

```
script text ──parse──► Project (cast + shots) ──► stage_voices ──► stage_keyframes ──► stage_clips ──► stage_music ──► stage_assemble
                                                  wav per line      png per shot       mp4 per shot     one wav        final.mp4
                                                  (decides clip     (user image,       (LTX / Wan /     (MusicGen or
                                                   length)           or SDXL+IP-Adapter  Ken Burns)       your file)
                                                                     +LoRA)
```

Voices run first because dialogue length decides how long each clip must be. Heavy models are loaded one at a time and unloaded after their stage (`free_gpu()`), which is what makes 8 GB workable.

## Modules

| File | Responsibility |
|---|---|
| `studio/project.py` | Data model: `Project`, `Character`, `Voice`, `Shot`, `Line`, `MusicSettings`; JSON save/load; paths stored relative to the project |
| `studio/script.py` | Script text → shots (deterministic parser); optional Ollama script writer; default voices |
| `studio/prompts.py` | Builds keyframe and video prompts; injects the fixed style and character text into every shot |
| `studio/voices.py` | One voice per character; Kokoro, F5-TTS (clone), or silent; loudness normalisation; cache by voice+text |
| `studio/keyframes.py` | User image → LoRA → IP-Adapter/SDXL; reference collage for multi-character shots |
| `studio/clips.py` | Image → video: LTX-Video, Wan 2.2 5B, or Ken Burns (ffmpeg) |
| `studio/music.py` | One music track: your file or MusicGen |
| `studio/assemble.py` | ffmpeg: per-shot encode (identical settings), lossless join, music ducking, subtitles |
| `studio/pipeline.py` | Runs the stages, caching, per-shot redo, fallback when an AI clip fails |
| `studio/cli.py` | Command line + `build_shots` (keeps still-valid work when the script changes) |
| `app.py` | Gradio UI over the same functions |

## Why consistency holds

- **Identity lives in the project, not in the shot.** Style text, character descriptions, reference images, LoRA paths and voices are defined once and read by every stage.
- **Determinism.** Seeds are `project.seed + shot id`; voice files are named by a hash of (voice, text); music by a hash of (prompt, length, seed).
- **Uniform encoding.** Every shot is encoded to the same codec, size, fps and audio format before joining, so the join is lossless and audio never changes character between shots.

## Choosing models (and why)

- **LTX-Video via diffusers** is the default clip model because its image-to-video pipeline (`LTXImageToVideoPipeline`) is documented in diffusers, supports CPU offload and VAE tiling, and the original 2B-class model is realistic for an 8 GB card with offload and reduced resolution. Docs: <https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx_video>.
- **Wan 2.2 TI2V-5B** is the other commonly recommended 8 GB image-to-video model; it is easiest to run through ComfyUI (GGUF). The `wan` backend here is experimental. A 2026 overview of local video models: <https://insiderllm.com/guides/local-ai-video-generation/>. Newer, larger LTX releases exist but need much more than 8 GB.
- **Kokoro** is a small, fast TTS with many fixed voices; a fixed voice id is the simplest guarantee of a stable voice.
- **MusicGen-small** for generated music; or bring your own track.
- Model landscape changes quickly. Re-check these choices before a big investment of time.

## Adding a backend

- **A different clip model:** add `_load_xxx` and a branch in `ClipGenerator.generate` (`studio/clips.py`), and a `clips.backend` name.
- **A different TTS (e.g. another language):** add a `_render_xxx(voice, text)` method returning `(numpy_audio, sample_rate)` to `VoiceEngine` and a branch in `VoiceEngine.synth` keyed on `voice.backend`. Everything else (normalisation, caching, timing, subtitles) is shared.
- **A different image generator:** `KeyframeGenerator.generate` only has to write one image to `out_path`.

## Known gaps / ideas

- Lip-sync / mouth animation timed to the dialogue audio
- Training a character LoRA from inside the app
- Cache invalidation when character look or style changes (currently manual: "Redo everything")
- Cross-shot continuity: using the last frame of one clip as the next keyframe
- More languages through additional TTS backends
