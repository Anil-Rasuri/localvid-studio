# Using LocalVid Studio

## The workflow (tabs in the app)

1. **Project and look.** Create a project. Set the style text (added to every shot), video size, subtitles.
2. **Characters.** For each character: name (exactly as in the script), a short fixed description of how they look, a reference image, and a voice. Save.
3. **Story.** Paste a script, or type an idea and press *Draft with local LLM*. Press *Build shots*. Unknown speaker names get a character with a default voice automatically.
4. **Music.** `none`, `generate` (MusicGen from your prompt), or `file` (your own song).
5. **Generate.** *Make the whole video*, or run one stage at a time. Redo any single shot at the bottom.

Output: `projects/<name>/output/final.mp4`. Intermediate files are in `projects/<name>/work/`.

## Cutout mode (default): characters that never change

`clips.backend: cutout` paints only the backgrounds with AI and animates **your character images** on top.

- Give each character **one clean picture of a single character** on a plain (white/light) or transparent background. The app removes the background automatically; if the picture holds several figures, the largest is used. A transparent PNG gives the cleanest edges.
- Movement is simple "puppet" animation: idle sway, a bounce timed to the voice while the character speaks, a pop-in at the start, a slow camera move. There is no body or lip animation.
- Characters named in a shot's `SCENE:` text or speaking in it appear, spread evenly across the screen. `Close-up` makes them larger and `Wide shot` smaller.
- Use `Redo everything` after changing a character image. Backgrounds are made without characters, so write scenes as places ("jungle clearing at sunrise").
- Want your own background for a shot? Upload it as that shot's image in tab 5.

## Getting the most consistent characters (fully-AI modes)

Roughly from most to least reliable:

1. **Your own image per shot.** Upload it for the shot (Generate tab → "use my own image"). The app uses it as-is and only animates it.
2. **Character LoRA** trained on your design (see SETUP). Put the path in the character's LoRA field. Add the LoRA's trigger word to the character's description.
3. **Reference image + description.** Use one clear image per character: full body or head and shoulders, plain background, same outfit every time. Keep the description short and concrete ("small brown monkey, red scarf, big round eyes") and never reword it between shots; the app already injects the same text each time.
4. Keep shots **short** (3-5 s) and the **motion simple**. Long, busy clips drift the most.
5. Prefer **one character per image** when possible. Two characters in one AI image is the hardest case: the reference images are combined into one collage for the adapter, which helps but is not perfect.
6. If a shot comes out wrong, **redo only that shot** (new seed). Don't regenerate the whole video.

## Consistent voices

- Pick a voice per character in tab 2 and leave it. Lines are cached by *(voice settings + text)*, so re-running never changes a voice.
- All lines are normalised to the same loudness.
- Keep each line to about **12 words or less** (≈ 4 seconds). If dialogue is longer than the clip, the clip is played forwards then backwards, then held.
- Voice cloning: upload a clean ~10 s clip (no music, no noise) as the character's voice sample. It needs the optional F5-TTS install (docs/SETUP.md).
- Lip-sync is not included: cartoon mouths are not moved to match the words. Subtitles help the viewer follow.

## Music and songs

- `file`: your own track plays under the whole video (looped if shorter), fades in and out, and dips while characters speak. A *song with lyrics* will compete with dialogue; keep its volume low or use an instrumental.
- `generate`: one MusicGen track (up to 30 s), looped for longer videos. Change the prompt and seed to get a different tune; it stays the same across re-runs with the same prompt and seed.

## When things are redone (caching)

Each stage saves its output, so pressing *Make the whole video* again only does missing work. This means:

- Editing the **script** → unchanged scenes keep their image and clip; changed scenes are redone automatically.
- Changing a **character's look, the style text, or a seed** does **not** invalidate old images. Tick *Redo everything* (or delete `work/keyframes` and `work/clips`) after such a change.
- Changing a **voice** re-makes that character's audio automatically (voice settings are part of the cache key).

## Command line

```
python -m studio.cli new "My story"
python -m studio.cli build my-story --script projects/my-story/script.txt
python -m studio.cli run my-story                  # everything
python -m studio.cli run my-story --stage clips    # one stage: voices, keyframes, clips, music, assemble
python -m studio.cli run my-story --force          # ignore saved work
python -m studio.cli --config other.yaml run my-story
```

Edit `projects/<name>/project.json` directly if you want finer control (voice speed, per-shot seeds, music volume).

## Settings worth knowing (config.yaml)

| Setting | Effect |
|---|---|
| `clips.backend` | `ltx` (AI, default), `wan` (experimental), `kenburns` (no AI, instant, perfectly consistent) |
| `clips.long_side` | Clip resolution. Lower (512) if you run out of GPU memory. |
| `keyframes.backend` | `sdxl` generates images; `none` only uses images you supply |
| `shot.min_seconds` / `max_seconds` | Clip length limits |
| `low_vram` | CPU offload. Keep `true` on 8 GB. |

## A good first test

Use `projects/example` but run only the first 2-3 scenes (delete the rest from the script), and set `clips.backend: kenburns` the first time. When voices, music and assembly look right, switch to `ltx` and time one AI clip before committing to a full minute.
