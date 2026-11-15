#!/usr/bin/env python3
"""
Render a video clip of a recording from the 3D replay viewer.

Steps the PAUSED timeline frame by frame (the sequence mode of
scripts/viewer_snapshot.py: same Chrome flags, same seek helpers), writes
one PNG per frame and encodes them with ffmpeg to an MP4 (H.264) and,
with --gif, an optimised GIF. Everything in the viewer animates on battle
time, so a frame is the same however long it took to render.

Camera, one of:
  --director              the viewer's auto director picks the shots (the
                          default when nothing else is given)
  --focus SHIP            orbit camera on a ship (--dist/--az/--el, plus
                          --orbit deg per battle second), re-framed every frame
  --camera-path FILE      explicit path: JSON {"keys": [{"t": 98,
                          "camera": [x, y, z], "target": [x, y, z]}, ...],
                          "ease": "smooth"|"linear", "relative_to": "ship id"}
                          in km (scene units); with relative_to the keys are
                          offsets from that ship

Software GL renders a frame in seconds to minutes, so the render is
RESUMABLE: frames already on disk are skipped (the frame directory holds a
clip.json with the parameters; a mismatch is refused unless --restart) and
an interrupted run still encodes the frames it has.

Examples:
    # 6 s of a kill, directed, 10 fps
    uv run python scripts/viewer_clip.py --recording p5_tactics_s4.json \
        --start 98 --duration 6 --fps 10 --width 960 --height 540 \
        --director --out /tmp/kill.mp4 --gif

    # the frames are already there: just encode again
    uv run python scripts/viewer_clip.py --recording p5_tactics_s4.json \
        --start 98 --duration 6 --fps 10 --out /tmp/kill.mp4 --encode-only

If no ffmpeg with an H.264 encoder is found (system `ffmpeg`, then
Playwright's bundled one, which only has VP8), the PNG sequence is kept and
the exact ffmpeg command is printed instead.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viewer_snapshot import (  # noqa: E402
    FRAME_JS, ViewerSession, page_url_for, start_vite, stop_vite)

CAMERA_JS = """
({ camera, target, rel }) => {
  const v = window.visualizer;
  v.cameraController?.setDirector?.(false);
  let o = [0, 0, 0];
  if (rel) {
    const ship = v.scene.ships.get(rel);
    if (ship) o = ship.position.toArray();
  }
  v.scene.camera.position.set(camera[0] + o[0], camera[1] + o[1], camera[2] + o[2]);
  v.scene.controls.target.set(target[0] + o[0], target[1] + o[1], target[2] + o[2]);
  v.scene.controls.update();
}
"""

SHOT_JS = """
(() => { const s = window.visualizer.cameraController?.directorState?.();
  return s && s.active ? `${s.type}/${s.kind}${s.subject ? ' ' + s.subject : ''}` : null; })()
"""


def frame_times(start: float, duration: float, fps: float) -> list[float]:
    """Battle time of every frame: start, start + 1/fps, ... (duration * fps frames)."""
    n = max(1, int(round(duration * fps)))
    return [round(start + i / fps, 6) for i in range(n)]


def frame_name(i: int) -> str:
    return f"f_{i:05d}.png"


def missing_frames(frames_dir: Path, n: int) -> list[int]:
    return [i for i in range(n) if not (frames_dir / frame_name(i)).exists()]


def contiguous_prefix(frames_dir: Path, n: int) -> int:
    """Number of frames on disk from frame 0 without a hole (what ffmpeg can encode)."""
    i = 0
    while i < n and (frames_dir / frame_name(i)).exists():
        i += 1
    return i


def load_camera_path(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        data = {"keys": data}
    keys = sorted(data.get("keys") or [], key=lambda k: k["t"])
    if not keys:
        raise ValueError("camera path has no keys")
    for k in keys:
        if len(k.get("camera", [])) != 3 or len(k.get("target", [])) != 3:
            raise ValueError("every camera path key needs t, camera [x,y,z] and target [x,y,z]")
    return {"keys": keys, "ease": data.get("ease", "smooth"), "relative_to": data.get("relative_to")}


def camera_at(path: dict, t: float) -> dict:
    """Camera and target at battle time t: held before the first key and
    after the last, eased (smoothstep) or linear in between."""
    keys = path["keys"]
    if t <= keys[0]["t"]:
        a = b = keys[0]
        u = 0.0
    elif t >= keys[-1]["t"]:
        a = b = keys[-1]
        u = 0.0
    else:
        j = next(i for i in range(1, len(keys)) if keys[i]["t"] >= t)
        a, b = keys[j - 1], keys[j]
        span = b["t"] - a["t"]
        u = (t - a["t"]) / span if span > 0 else 0.0
        if path.get("ease", "smooth") == "smooth":
            u = u * u * (3 - 2 * u)
    mix = lambda p, q: [p[i] + (q[i] - p[i]) * u for i in range(3)]  # noqa: E731
    return {"camera": mix(a["camera"], b["camera"]), "target": mix(a["target"], b["target"]),
            "rel": path.get("relative_to")}


def find_ffmpeg() -> tuple[str | None, bool]:
    """(path, has H.264 encoder): the system ffmpeg, else Playwright's."""
    candidates = []
    system = shutil.which("ffmpeg")
    if system:
        candidates.append(system)
    cache = Path.home() / ".cache" / "ms-playwright"
    candidates += sorted(str(p) for p in cache.glob("ffmpeg-*/ffmpeg*"))
    first = None
    for exe in candidates:
        try:
            out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True,
                                 text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        first = first or exe
        if "libx264" in out:
            return exe, True
    return first, False


def mp4_command(ffmpeg: str, frames_dir: Path, fps: float, out: Path, n: int, crf: int = 18) -> list[str]:
    return [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-framerate", f"{fps:g}",
            "-start_number", "0", "-i", str(frames_dir / "f_%05d.png"), "-frames:v", str(n),
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
            # H.264 4:2:0 needs even dimensions
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-movflags", "+faststart", str(out)]


def gif_command(ffmpeg: str, frames_dir: Path, fps: float, out: Path, n: int, width: int) -> list[str]:
    """One pass with a clip-wide palette (diff stats: the moving parts get
    the colours) and rectangle diffing, which keeps a mostly-black space
    scene small."""
    vf = (f"fps={fps:g},scale={width}:-2:flags=lanczos,split[a][b];"
          "[a]palettegen=max_colors=160:stats_mode=diff[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    return [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-framerate", f"{fps:g}",
            "-start_number", "0", "-i", str(frames_dir / "f_%05d.png"), "-frames:v", str(n),
            "-filter_complex", vf, "-loop", "0", str(out)]


def encode(frames_dir: Path, n: int, fps: float, out: Path, gif: bool, gif_width: int, crf: int) -> int:
    if n < 1:
        print("no frames to encode")
        return 1
    ffmpeg, h264 = find_ffmpeg()
    if not ffmpeg or not h264:
        cmd = mp4_command("ffmpeg", frames_dir, fps, out, n, crf)
        print(f"no ffmpeg with an H.264 encoder found ({ffmpeg or 'no ffmpeg at all'}); "
              f"the PNG sequence is in {frames_dir}. Encode it with:")
        print("  " + " ".join(f"'{c}'" if any(ch in c for ch in " ()[];,") else c for c in cmd))
        return 2
    subprocess.run(mp4_command(ffmpeg, frames_dir, fps, out, n, crf), check=True)
    print(f"wrote {out}  {out.stat().st_size / 1024:.0f} KiB  {n} frames  {n / fps:.2f} s at {fps:g} fps")
    if gif:
        gif_out = out.with_suffix(".gif")
        subprocess.run(gif_command(ffmpeg, frames_dir, fps, gif_out, n, gif_width), check=True)
        print(f"wrote {gif_out}  {gif_out.stat().st_size / 1024:.0f} KiB")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", required=True,
                   help="file name under visualizer/public/recordings/ (or a full URL)")
    p.add_argument("--start", type=float, required=True, help="battle time of the first frame (s)")
    p.add_argument("--duration", type=float, required=True, help="battle seconds to render")
    p.add_argument("--fps", type=float, default=24.0, help="frames per battle second")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--gfx", choices=["low", "high", "auto"],
                   help="graphics quality (default: viewer auto-detect; software GL -> low)")
    p.add_argument("--hud", action="store_true", help="keep the HUD panels (default: canvas only)")
    p.add_argument("--director", action="store_true", help="auto director camera (the default)")
    p.add_argument("--focus", help="ship id to orbit instead of the director")
    p.add_argument("--dist", type=float, default=3.0, help="--focus camera distance in hull lengths")
    p.add_argument("--az", type=float, default=35.0, help="--focus azimuth at the first frame (deg)")
    p.add_argument("--el", type=float, default=18.0, help="--focus elevation (deg)")
    p.add_argument("--orbit", type=float, default=0.0, help="--focus azimuth sweep, deg per battle second")
    p.add_argument("--camera-path", help="JSON camera path (see the module docstring)")
    p.add_argument("--out", required=True, help="output .mp4")
    p.add_argument("--gif", action="store_true", help="also write an optimised GIF next to the MP4")
    p.add_argument("--gif-width", type=int, default=640)
    p.add_argument("--crf", type=int, default=18, help="H.264 quality (lower = better, bigger)")
    p.add_argument("--frames-dir", help="PNG sequence directory (default: <out>_frames)")
    p.add_argument("--restart", action="store_true", help="discard frames from an earlier run")
    p.add_argument("--encode-only", action="store_true", help="encode the frames on disk, render nothing")
    p.add_argument("--preroll", type=float, default=2.0,
                   help="battle seconds stepped through before the first rendered frame, so "
                        "effects that began earlier are in flight (also on resume)")
    p.add_argument("--substep", type=float, default=0.5, help="largest seek step during the pre-roll")
    p.add_argument("--port", type=int, default=5173)
    p.add_argument("--init-timeout", type=float, default=600.0,
                   help="seconds to wait for the viewer to load the recording")
    p.add_argument("--query", help="extra viewer URL parameters")
    args = p.parse_args()

    out = Path(args.out).resolve()
    if out.suffix.lower() != ".mp4":
        p.error("--out must end in .mp4")
    if args.focus and args.camera_path:
        p.error("--focus and --camera-path are different cameras: give one")
    director = not (args.focus or args.camera_path)
    if args.director and not director:
        p.error("--director cannot be combined with --focus / --camera-path")
    path = load_camera_path(args.camera_path) if args.camera_path else None
    frames_dir = Path(args.frames_dir).resolve() if args.frames_dir else out.with_name(out.stem + "_frames")
    times = frame_times(args.start, args.duration, args.fps)
    n = len(times)

    meta = {"recording": args.recording, "start": args.start, "fps": args.fps, "width": args.width,
            "height": args.height, "gfx": args.gfx, "hud": args.hud, "query": args.query,
            "camera": "director" if director else
            ({"path": path} if path else
             {"focus": args.focus, "dist": args.dist, "az": args.az, "el": args.el, "orbit": args.orbit})}
    meta_file = frames_dir / "clip.json"
    if args.restart and frames_dir.exists() and not args.encode_only:
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if meta_file.exists():
        old = json.loads(meta_file.read_text())
        if old != json.loads(json.dumps(meta)) and not args.encode_only:
            print(f"{frames_dir} holds frames of a different clip ({meta_file}); "
                  "use --restart to discard them or another --frames-dir")
            return 2
    else:
        meta_file.write_text(json.dumps(meta, indent=1))
    for stale in frames_dir.glob("*.tmp.png"):
        stale.unlink()

    todo = [] if args.encode_only else missing_frames(frames_dir, n)
    if todo:
        print(f"{n} frames ({args.duration:g} s at {args.fps:g} fps, {args.width}x{args.height}); "
              f"{n - len(todo)} already on disk, rendering {len(todo)}")
        vite = start_vite(args.port)
        page_url = page_url_for(args.port, args.recording, args.gfx, director, args.query)
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as pw:
                session = ViewerSession(pw, page_url, args.width, args.height,
                                        init_timeout_s=args.init_timeout)
                page = session.page
                if not args.hud:
                    session.hide_ui()

                def place_camera(i: int) -> None:
                    if path:
                        page.evaluate(CAMERA_JS, camera_at(path, times[i]))
                    elif args.focus:
                        page.evaluate(FRAME_JS, {
                            "focus": args.focus, "dist": args.dist, "el": args.el,
                            "az": args.az + args.orbit * (times[i] - args.start),
                            "camera": None, "target": None, "director": False})

                at = None        # battle time the page is at
                started = time.time()
                done = 0
                try:
                    for i in todo:
                        t = times[i]
                        if at is None or t < at or t - at > max(args.substep, 1.5 / args.fps) + 1e-6:
                            # First frame, or a hole after frames already on
                            # disk: run in from a little earlier
                            at = max(0.0, t - args.preroll)
                            session.clean_seek(at)
                        while at < t - 1e-6:
                            at = min(t, at + args.substep)
                            session.seek(at)
                        place_camera(i)
                        session.wait_frames(1)
                        tmp = frames_dir / f"{frame_name(i)}.tmp.png"
                        page.screenshot(path=str(tmp), timeout=300_000)
                        os.replace(tmp, frames_dir / frame_name(i))   # a frame on disk is a whole frame
                        done += 1
                        shot = page.evaluate(SHOT_JS) if director else None
                        rate = (time.time() - started) / done
                        print(f"frame {i + 1}/{n}  t={session.battle_time():.2f}"
                              f"{'  ' + shot if shot else ''}  ({rate:.0f} s/frame, "
                              f"~{rate * (len(todo) - done) / 60:.0f} min left)", flush=True)
                except KeyboardInterrupt:
                    print("interrupted: encoding the frames rendered so far")
                except Exception as e:   # renderer stalled, page crashed: keep what we have
                    print(f"render stopped: {e}")
                session.print_errors()
                try:
                    session.close()
                except Exception:
                    pass
        finally:
            if vite:
                stop_vite(vite)

    have = contiguous_prefix(frames_dir, n)
    if have < n:
        print(f"{have}/{n} frames on disk without a hole - run the same command again to resume; "
              "encoding that prefix")
    code = encode(frames_dir, have, args.fps, out, args.gif, args.gif_width, args.crf)
    return code if code else (0 if have == n else 3)


if __name__ == "__main__":
    sys.exit(main())
