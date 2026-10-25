# README screenshots

Captured from the Three.js viewer on October 25, 2026 with headless Chromium
(Playwright, SwiftShader WebGL) at 1600 × 900 with `--gfx high`. These are
application screenshots: ship states, ordnance, damage, HUD text and commander
messages come from the recorded battles. Nothing is painted in or retouched;
the only post-processing is lossless PNG optimisation (and JPEG quality 95,
4:4:4, for `reactor-breach.jpg`, whose particle field does not fit a PNG under
1.5 MB). Framing choices are camera placement, lens (field of view), panel
filters/scrolling, and for `command-net.png` a 2× device-scale crop.

| File | Recording | Battle time | View |
| --- | --- | ---: | --- |
| `hero.png` | Showcase s0 (200 pt, dreadnoughts vs torpedo swarm) | 96.5s | `alpha_1` (dreadnought) at full thrust, PD beams on inbound torpedoes; 4.6 hull lengths, azimuth −36°, elevation 9°, 20° lens; HUD hidden |
| `hull-lineup.png` | `rcs_showroom.json` as an empty stage | 5s | Showroom: every hull class spawned with the viewer's `createShip`, alpha column facing beta; top to bottom siege dreadnought, dreadnought, battleship, battlecruiser, torpedo cruiser, cruiser, destroyer, frigate, corvette; 12° lens; HUD hidden |
| `fleet-battle.png` | Claude Sonnet 5 vs DeepSeek V4.1 Flash, 150 pt | 151s | Fleet view one second after the first kill; scoreboard, kill toast, command net, registry, timeline lanes; track legend collapsed |
| `torpedo-salvo.png` | Showcase s7 (200 pt, battleships vs torpedo swarm) | 203.3s | Two battleships' PD fans on an inbound torpedo, kinetic strike on `alpha_1`; explicit camera; HUD hidden |
| `reactor-breach.jpg` | Showcase s7 | 205.7s | `alpha_1` battleship reactor breach 1.7 s after the kill, the sister ship's plume crossing the fireball; 5.5 hull lengths, azimuth −20°, elevation 20°; HUD hidden |
| `dying-hulk.png` | Showcase s7 | 114s | `alpha_7` destroyer as a dying hulk (killed at 110 s, reactor goes at 118 s): molten cracks, venting, sputtering torch; 1.35 hull lengths, azimuth 20°, elevation 15°; HUD hidden |
| `command-net.png` | DeepSeek V4.1 Flash mirror, 150 pt (pass-4 engine) | 216s | OCS Lancer-2 and its captain's DEVIATE report to the admiral (merge-floor reasoning, compliance chip), both admirals' standing directives; REPORTS filter; 1200 × 660 viewport at device scale 2, cropped to the right 800 × 450 CSS px |
| `intent-consequence.png` | Qwen–GPT-OSS | 184.8s | `alpha_2` decision `e152` pinned; camera on the crippled `beta_1` (reactor in 14 s), 2 hull lengths, azimuth 35°, elevation 18°; Decisions panel scrolled to ship execution and observed consequences |
| `live-deliberation.png` | Claude Sonnet 5 vs DeepSeek V4.1 Flash, replayed through the live endpoints | CP5 (150s) | `?live=1&decisions=1`: live bar, COMMAND DELIBERATION card (both sides' progress, spend), alpha admiral's salvo directive, the salvo one second from impact |
| `draft-splash.png` | Same, live endpoints | T+4s | Draft splash after both sealed picks: rosters, formations and each admiral's reasoning |

## Recordings

- Claude Sonnet 5 vs DeepSeek V4.1 Flash: `sonnet5_vs_dsv41flash_150pt_20260926.json` (live run of September 26, 2026)
- DeepSeek V4.1 Flash mirror: `p4_new_flash_mirror_150pt.json` (pass-4 engine with captain reports)
- [Qwen–GPT-OSS](../../../data/recordings/intent_qwen_gptoss.json)
- Showcase s0 / s7: free heuristic draft battles on the pass-4 engine, regenerated
  deterministically (identical trace and events run to run):

  ```bash
  uv run python scripts/run_draft_battle.py --auto-draft --no-admirals --trace \
    --seed 0 --budget 200 --time-limit 420    # -> stage as showcase_s0_200pt.json
  uv run python scripts/run_draft_battle.py --auto-draft --no-admirals --trace \
    --seed 7 --budget 200 --time-limit 420    # -> stage as showcase_s7_200pt.json
  ```

Stage recordings in `visualizer/public/recordings/` as described in the
[root README](../../../README.md#watch-a-replay).

## Capture commands

Each replay image is one `scripts/viewer_snapshot.py` command (add `--port` if
your dev server is not on 5173). `capture-helpers.js` holds the JavaScript the
commands pass through `--eval`: it only calls the viewer's own API
(`window.visualizer`).

```bash
uv run playwright install chromium
SNAP="uv run python scripts/viewer_snapshot.py --gfx high --width 1600 --height 900"
H=docs/images/readme/capture-helpers.js
TRACKS_OFF="window.visualizer.trackControl.set('off')"

# hero.png
$SNAP --recording showcase_s0_200pt.json --time 96.5 --focus alpha_1 --dist 4.6 --az -36 --el 9 --hide-ui \
  --eval "(() => { $TRACKS_OFF; const c = window.visualizer.scene.camera; c.fov = 20; c.updateProjectionMatrix(); })()" \
  --out hero.png

# hull-lineup.png
$SNAP --recording rcs_showroom.json --time 5 --hide-ui --eval "$(cat $H); readmeLineup()" --out hull-lineup.png

# fleet-battle.png
$SNAP --recording sonnet5_vs_dsv41flash_150pt_20260926.json --time 151 \
  --eval "window.visualizer.showFleetView(); window.visualizer.trackHud.setLegend(false)" --out fleet-battle.png

# torpedo-salvo.png
$SNAP --recording showcase_s7_200pt.json --time 203.3 \
  --camera=-100.27,-14.827,34.354 --target=-112.369,-24.971,1.112 --hide-ui --eval "$TRACKS_OFF" --out torpedo-salvo.png

# reactor-breach.jpg (captured as PNG, then JPEG q95 4:4:4)
$SNAP --recording showcase_s7_200pt.json --time 205.7 --focus alpha_1 --dist 5.5 --az -20 --el 20 \
  --hide-ui --eval "$TRACKS_OFF" --out reactor-breach.png

# dying-hulk.png
$SNAP --recording showcase_s7_200pt.json --time 114 --focus alpha_7 --dist 1.35 --az 20 --el 15 \
  --hide-ui --eval "$TRACKS_OFF" --out dying-hulk.png

# command-net.png: 1200x660 CSS at device scale 2, right-hand 800x450 CSS crop
uv run python scripts/viewer_snapshot.py --gfx high --recording p4_new_flash_mirror_150pt.json \
  --time 216 --focus beta_2 --dist 2.6 --az 165 --el 12 --width 1200 --height 660 --scale 2 \
  --clip 392,117,800,450 --eval "$(cat $H); $TRACKS_OFF; readmeCommsTo('Lancer-2')" --out command-net.png

# intent-consequence.png (the ship is deselected after load so its off-screen
# marker does not stack on the fleet marker)
$SNAP --recording intent_qwen_gptoss.json --query "ship=alpha_2&decisions=1&decision=e152&t=184.8" \
  --time 184.8 --focus beta_1 --dist 2 --az 35 --el 18 \
  --eval "$(cat $H); window.visualizer.deselectShip(); readmeDecisionTo('03')" --out intent-consequence.png
```

The live images replay the Sonnet–DeepSeek recording through the live
endpoints with `scripts/fake_live_server.py` (40 s synthetic draft: alpha
commits at 60 %, beta at 90 %; each checkpoint held 14 s for deliberation):

```bash
uv run python scripts/fake_live_server.py \
  visualizer/public/recordings/sonnet5_vs_dsv41flash_150pt_20260926.json \
  --port 8825 --speed 6 --draft-seconds 40 --pause-seconds 14
```

A 1600 × 900 Playwright page then opens
`/?live=1&live_base=http://127.0.0.1:8825/live&gfx=high` (plus `&decisions=1`
for `live-deliberation.png`), with `window.visualizer.slowFrameOk = true` once
the battle starts. `draft-splash.png` is taken about 2.5 s after the battle
starts (the splash still up, both fleets committed); for
`live-deliberation.png` the splash is dismissed, playback set to 4×, the track
legend collapsed, and the frame taken 1.5 s after the deliberation card for
checkpoint 5 appears. The live pacing and deliberation progress come from the
fake server's replay of the recording, not from a running battle.
