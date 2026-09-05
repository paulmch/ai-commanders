#!/usr/bin/env python3
"""
Headless screenshots of the 3D replay viewer via Playwright.

Starts the Vite dev server in visualizer/ (or reuses one already listening
on the port), loads a recording from visualizer/public/recordings/, seeks
the timeline, frames the camera and writes PNGs. Used to verify rendering
changes without a GPU or a human at the keyboard - WSL2 llvmpipe is slow
but produces the same pixels a real GPU would.

Examples:
    # one frame, camera orbiting alpha_4 two hull-lengths out
    uv run python scripts/viewer_snapshot.py --recording fleet_heur.json \
        --time 140 --focus alpha_4 --dist 2.5 --out /tmp/a4.png

    # a destruction sequence: seek to 142.5, play in real time, grab a
    # frame every 0.8 s for 12 s
    uv run python scripts/viewer_snapshot.py --recording fleet_heur.json \
        --time 142.5 --focus alpha_4 --dist 4 --sequence 12 --every 0.8 \
        --out /tmp/kill_%02d.png

    # explicit camera
    uv run python scripts/viewer_snapshot.py --recording torpedo.json \
        --time 60 --camera -40,25,80 --target 0,0,0 --out /tmp/wide.png

The viewer exposes window.visualizer; camera framing is done with
page.evaluate so hidden ships (destroyed hulks) are framed from their
destruction effect instead.
"""

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
VIS_DIR = ROOT / "visualizer"

CHROME_ARGS = [
    # llvmpipe on WSL2: without --ignore-gpu-blocklist THREE fails with
    # "Could not create a WebGL context"
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--enable-unsafe-swiftshader",
    "--disable-gpu-sandbox",
]

FRAME_JS = """
({ focus, dist, az, el, camera, target }) => {
  const v = window.visualizer;
  const s = v.scene;
  const T = s.THREE || null;
  let center = null, L = 8;
  if (focus) {
    const ship = s.ships.get(focus);
    if (ship) {
      L = ship.userData.size?.length || 8;
      center = ship.position.clone();
      if (!ship.visible) {
        const eff = (s.destructionEffects || []).find(e => e.shipId === focus);
        if (eff) center = eff.position.clone();
      }
    }
  }
  const cam = s.camera;
  const ctl = s.controls;
  if (camera) {
    cam.position.set(camera[0], camera[1], camera[2]);
    ctl.target.set(target[0], target[1], target[2]);
  } else if (center) {
    const r = L * dist;
    const azr = az * Math.PI / 180, elr = el * Math.PI / 180;
    cam.position.set(
      center.x + r * Math.cos(elr) * Math.sin(azr),
      center.y + r * Math.sin(elr),
      center.z + r * Math.cos(elr) * Math.cos(azr));
    ctl.target.copy(center);
  }
  ctl.update();
  return center ? [center.x, center.y, center.z, L] : null;
}
"""


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_vite(port: int) -> subprocess.Popen | None:
    if port_open(port):
        return None
    proc = subprocess.Popen(
        ["npx", "vite", "--port", str(port), "--strictPort", "--host", "127.0.0.1"],
        cwd=VIS_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        if port_open(port):
            return proc
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError("vite did not start")


def parse_vec(s: str | None):
    if not s:
        return None
    return [float(x) for x in s.split(",")]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", required=True,
                   help="file name under visualizer/public/recordings/ (or a full URL)")
    p.add_argument("--time", type=float, default=0.0, help="seek time in seconds")
    p.add_argument("--focus", help="ship id to frame")
    p.add_argument("--dist", type=float, default=3.0, help="camera distance in hull lengths")
    p.add_argument("--az", type=float, default=35.0, help="camera azimuth deg")
    p.add_argument("--el", type=float, default=18.0, help="camera elevation deg")
    p.add_argument("--camera", help="x,y,z camera position (km) - overrides --focus")
    p.add_argument("--target", default="0,0,0", help="x,y,z look-at (km) with --camera")
    p.add_argument("--follow", action="store_true",
                   help="re-frame the focus ship before every sequence frame")
    p.add_argument("--sequence", type=float, default=0.0,
                   help="play this many seconds of battle time after seeking")
    p.add_argument("--every", type=float, default=1.0, help="seconds between frames")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed for --sequence")
    p.add_argument("--substep", type=float, default=0.5,
                   help="seek granularity (battle s) between sequence frames")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--port", type=int, default=5173)
    p.add_argument("--hide-ui", action="store_true", help="hide the HUD panels")
    p.add_argument("--out", required=True,
                   help="output PNG; use %%02d for --sequence frames")
    p.add_argument("--eval", help="extra JS to run after framing (window.visualizer in scope)")
    args = p.parse_args()

    vite = start_vite(args.port)
    url = args.recording if "://" in args.recording else f"/recordings/{args.recording}"
    page_url = f"http://127.0.0.1:{args.port}/?recording={url}"

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=CHROME_ARGS)
            page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                    device_scale_factor=1)
            errors: list[str] = []
            page.on("console", lambda m: errors.append(m.text)
                    if m.type in ("error", "warning") else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(page_url, wait_until="domcontentloaded")
            page.wait_for_function("window.visualizer && window.visualizer.isInitialized",
                                   timeout=120_000)
            # initializeBattle() auto-plays ~300 ms after isInitialized flips
            page.wait_for_timeout(700)
            page.evaluate("window.visualizer.timeController.pause()")
            # llvmpipe frames can take >1 s; the viewer's stalled-tab guard
            # would otherwise skip every update
            page.evaluate("window.visualizer.slowFrameOk = true")

            def wait_frames(n: int, timeout_s: float = 60.0) -> None:
                start = page.evaluate("window.visualizer.scene.frameIndex")
                t0 = time.time()
                while page.evaluate("window.visualizer.scene.frameIndex") < start + n:
                    if time.time() - t0 > timeout_s:
                        raise RuntimeError("renderer stalled")
                    page.wait_for_timeout(50)
            if args.hide_ui:
                page.add_style_tag(content="#app > *:not(canvas){display:none !important}")

            frame_opts = {
                "focus": args.focus, "dist": args.dist, "az": args.az, "el": args.el,
                "camera": parse_vec(args.camera), "target": parse_vec(args.target),
            }

            def seek(t: float) -> None:
                page.evaluate("t => window.visualizer.timeController.seek(t)", t)
                wait_frames(1)

            def shoot(path: str) -> None:
                page.evaluate(FRAME_JS, frame_opts)
                if args.eval:
                    page.evaluate(args.eval)
                wait_frames(2)
                page.evaluate(FRAME_JS, frame_opts)
                wait_frames(2)
                page.screenshot(path=path)
                print("wrote", path)

            # A clean seek: jump a little before the target so the first
            # visible frame has interpolated state, then land on it.
            seek(max(0.0, args.time - 0.05))
            seek(args.time)

            if args.sequence <= 0:
                shoot(args.out)
            else:
                # Deterministic stepping: stay paused and seek in small
                # sub-steps so event crossings spawn at their true times;
                # effects animate on battle time, not the wall clock.
                page.evaluate(FRAME_JS, frame_opts)
                n = int(args.sequence / args.every) + 1
                sub = args.substep
                t = args.time
                for i in range(n):
                    target = args.time + i * args.every
                    while t < target - 1e-6:
                        t = min(target, t + sub)
                        seek(t)
                    if args.follow or i == 0:
                        page.evaluate(FRAME_JS, frame_opts)
                    wait_frames(2)
                    path = args.out % i if "%" in args.out else args.out
                    bt = page.evaluate("window.visualizer.timeController.currentTime")
                    page.screenshot(path=path)
                    print(f"wrote {path}  battle t={bt:.1f}")
            if errors:
                print("--- console errors/warnings ---")
                for e in errors[:20]:
                    print(e[:300])
            browser.close()
    finally:
        if vite:
            vite.terminate()
            try:
                vite.wait(timeout=5)
            except subprocess.TimeoutExpired:
                vite.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
