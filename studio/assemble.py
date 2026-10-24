"""Final assembly with ffmpeg: per-shot video + dialogue, join, background music (ducked under
speech), optional burned-in subtitles. Every shot is encoded with identical settings so the
join is lossless and the audio format never changes between shots."""
from __future__ import annotations

from pathlib import Path

from .project import Project
from .util import media_duration, run_ffmpeg, srt_time

AUDIO_RATE = 48000


def render_shot(project: Project, cfg: dict, shot, out_path: Path) -> float:
    """Combine the shot's clip + dialogue into one mp4. Returns its duration."""
    clip = project.abs(shot.clip_path)
    vdur = media_duration(clip)
    need = shot.audio_duration + cfg["shot"]["tail_pad"] if shot.audio_duration else 0.0
    dur = max(vdur, need, cfg["shot"]["min_seconds"] if vdur < cfg["shot"]["min_seconds"] else 0.0)
    w, h, fps = project.width, project.height, project.fps

    # Dialogue longer than the clip: play the clip forwards then backwards (ping-pong), then hold.
    if dur > vdur + 0.05:
        chain = "[0:v]split=2[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0[pp];[pp]"
        vdur = vdur * 2
    else:
        chain = "[0:v]"
    chain += (
        f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h},fps={fps},setsar=1,"
        f"tpad=stop_mode=clone:stop_duration={max(0.0, dur - vdur) + 0.1:.3f}[v]"
    )
    audio_chain = f"[1:a]aresample={AUDIO_RATE},aformat=channel_layouts=stereo,apad[a]"
    if shot.audio_path:
        audio_in = ["-i", project.abs(shot.audio_path)]
    else:
        audio_in = ["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]

    run_ffmpeg([
        "-i", clip, *audio_in,
        "-filter_complex", f"{chain};{audio_chain}",
        "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", AUDIO_RATE, "-ac", "2",
        out_path,
    ])
    return dur


def write_srt(project: Project, durations: list[float], out_path: Path) -> bool:
    entries, offset, n = [], 0.0, 1
    for shot, dur in zip(project.shots, durations):
        for line in shot.dialogue:
            start = offset + line.start
            end = start + max(line.duration, 0.8)
            entries.append(f"{n}\n{srt_time(start)} --> {srt_time(end)}\n{line.speaker}: {line.text}\n")
            n += 1
        offset += dur
    if entries:
        out_path.write_text("\n".join(entries), encoding="utf-8")
    return bool(entries)


def assemble(project: Project, cfg: dict, music_path: Path | None, log=print) -> Path:
    work = project.dir("work/shots")
    out_dir = project.dir("output")
    final = out_dir / "final.mp4"

    shot_files, durations = [], []
    for shot in project.shots:
        if not shot.clip_path or not project.abs(shot.clip_path).exists():
            raise RuntimeError(f"Shot {shot.id} has no clip yet. Run the clip stage first.")
        path = work / f"shot_{shot.id:02d}.mp4"
        log(f"Assembling shot {shot.id}/{len(project.shots)}")
        durations.append(render_shot(project, cfg, shot, path))
        shot_files.append(path)

    # join (all shots share codec/size/fps/audio format, so stream copy is safe)
    (work / "list.txt").write_text("".join(f"file '{p.name}'\n" for p in shot_files), encoding="utf-8")
    joined = work / "joined.mp4"
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", joined.name], cwd=work)
    total = sum(durations)

    have_subs = project.subtitles and write_srt(project, durations, work / "subs.srt")
    if not music_path and not have_subs:
        joined.replace(final)
        return final

    args: list = ["-i", joined]
    if music_path:
        m = project.music
        fade_start = max(0.0, total - 1.5)
        args += ["-stream_loop", "-1", "-i", music_path]
        music = f"[1:a]volume={m.volume},afade=t=in:d=1.0,afade=t=out:st={fade_start:.2f}:d=1.5"
        if m.duck:
            fc = (f"{music}[m0];[0:a]asplit=2[dlg][sc];"
                  "[m0][sc]sidechaincompress=threshold=0.02:ratio=10:attack=20:release=500[md];"
                  "[dlg][md]amix=inputs=2:duration=first:dropout_transition=0,volume=2[a]")
        else:
            fc = (f"{music}[m0];[0:a][m0]amix=inputs=2:duration=first:dropout_transition=0,volume=2[a]")
        args += ["-filter_complex", fc, "-map", "0:v", "-map", "[a]"]
    else:
        args += ["-map", "0:v", "-map", "0:a"]

    if have_subs:
        style = "FontSize=18,Outline=2,Shadow=0,MarginV=24"
        args += ["-vf", f"subtitles=subs.srt:force_style='{style}'",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
    else:
        args += ["-c:v", "copy"]
    args += ["-c:a", "aac", "-b:a", "192k", "-t", f"{total:.3f}", final]
    run_ffmpeg(args, cwd=work)  # cwd = work/shots so the subtitle path needs no escaping
    return final
