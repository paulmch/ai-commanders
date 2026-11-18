# README screenshots

Captured from the Three.js viewer on November 15, 2026 (pass-5 viewer: sun
shadows, image-based fill, ambient occlusion, new hulls and effects) with
headless Chromium (Playwright, SwiftShader WebGL) at 1600 × 900 with
`--gfx high`. `damage-bow.png`, `damage-mounts.jpg` and `damage-cookoff.png`
were captured on November 17, 2026; `hero.png`, `dreadnought.png`,
`plume-wash.png`, `damage-hulk.png` and the new `launch-sequence.png` on
November 18, 2026 (the one-piece drive plume, the torpedo launch sequence), all at the viewer's default hull scale
(`auto`: real-size hulls up close); see [Hull scale](#hull-scale). The two live-mode images (`live-deliberation.png`,
`draft-splash.png`) are still the October 25, 2026 captures. These are
application screenshots: ship states, ordnance, damage, HUD text and commander
messages come from the recorded battles. Nothing is painted in or retouched;
the only post-processing is lossless PNG re-compression (Pillow,
`optimize=True`) and, for the images whose particle fields or star fields do
not fit a PNG under about 1.5 MB (`reactor-breach.jpg` and any other `.jpg`
below), JPEG quality 95, 4:4:4. Framing choices are camera placement, lens (field of view), panel
filters/scrolling, and for `command-net.png` a 2× device-scale crop.

| File | Recording | Battle time | View |
| --- | --- | ---: | --- |
| `hero.png` | Showcase s0 (200 pt, dreadnoughts vs torpedo swarm) | 96.5s | `alpha_1` (dreadnought) at full thrust, PD lasers on inbound torpedoes; 4.6 hull lengths (1.27 km), azimuth −36°, elevation 9°, 20° lens; HUD hidden; scale auto |
| `hull-lineup.png` | `rcs_showroom.json` as an empty stage (any recording works: the hulls are spawned far from its ships) | 5s | Showroom: every hull class spawned with the viewer's `createShip`, alpha column facing beta; top to bottom siege dreadnought, dreadnought, battleship, battlecruiser, torpedo cruiser, cruiser, destroyer, frigate, corvette; 12° lens; HUD hidden |
| `fleet-battle.png` | Claude Sonnet 5 vs DeepSeek V4.1 Flash, 150 pt | 151s | Fleet view one second after the first kill; scoreboard, kill toast, command net, registry, timeline lanes; track legend collapsed |
| `torpedo-salvo.png` | Showcase s7 (200 pt, battleships vs torpedo swarm) | 203.3s | Two battleships' PD fans on an inbound torpedo, kinetic strike on `alpha_1`; explicit camera; HUD hidden |
| `reactor-breach.jpg` | Showcase s7 | 205.7s | `alpha_1` battleship reactor breach 1.7 s after the kill, shock ring and debris, the sister ship's plume at the fireball's left edge; 5.5 hull lengths, azimuth −20°, elevation 20°; HUD hidden |
| `dying-hulk.png` | Showcase s7 | 114s | `alpha_7` destroyer as a dying hulk (killed at 110 s, reactor goes at 118 s): molten cracks, venting, sputtering torch; 1.35 hull lengths, azimuth 20°, elevation 15°; HUD hidden |
| `command-net.png` | DeepSeek V4.1 Flash mirror, 150 pt (pass-4 engine) | 216s | OCS Lancer-2 and its captain's DEVIATE report to the admiral (merge-floor reasoning, compliance chip), both admirals' standing directives; REPORTS filter; 1200 × 660 viewport at device scale 2, cropped to the right 800 × 450 CSS px |
| `intent-consequence.png` | Qwen–GPT-OSS | 184.8s | `alpha_2` decision `e152` pinned; camera on the crippled `beta_1` (reactor in 14 s), 2 hull lengths, azimuth 35°, elevation 18°; Decisions panel scrolled to ship execution and observed consequences |
| `blast-shell.jpg` | Pass-5 damage demo | 152.6s | Reactor blast of `beta_5` (destroyer, killed at 150 s) 2.6 s on: fireball inside the blast shell drawn to the recorded 13 km radius, two escorts beside it, PD lasers still working; 6 hull lengths, azimuth 30°, elevation 18°; HUD hidden |
| `fittings-hud.png` | Pass-5 damage demo | 180s | OCS Falchion-1 (`beta_2`) selected one second after the frigate TIS Ward-1's reactor blast caught it: kill toast with the blast radius, CAUGHT IN BLAST and DISARMED toasts, fitting pips in the ship registry, FITTINGS block (guns, PD with one damaged turret, drive) in the telemetry card; 5 hull lengths, azimuth 30°, elevation 18° |
| `dreadnought.png` | Showcase s0 | 96.5s | `alpha_1`, the pass-5 dreadnought, three-quarter view from the sunlit side at full thrust; 3 hull lengths (0.83 km), azimuth −10°, elevation 35°, 20° lens; HUD hidden; scale auto |
| `launch-sequence.png` | Launch showcase (`p5_plume2_launch_showcase.json.gz`, scripted `launch_showcase` battle) | 34.1s | The torpedo cruiser `alpha`, flank to its target, 4.1 s after emptying one cell of each VLS bank: three rounds lighting their torches 55-75 m off the hull, the round from the bank facing away still turning about on its RCS (it lights at 37 s); open hatches on the banks; 1.6 hull lengths, azimuth 60°, elevation 20°; HUD hidden; scale auto |
| `plume-wash.png` | Pass-5 plume demo (`p5_plume_demo.json`, scripted `plume_wash` battle) | 62s | From beside the stern of the battleship `alpha_1` (lower left, torch lit) down its exhaust: one beam from the bells, fading with distance, to the destroyer `alpha_2` 6 km astern, glowing in the wash; camera 1.1 hull lengths ahead of the battleship's centre, 0.5 above, 1.5 to port, looking 14 lengths astern; HUD hidden; scale auto |
| `damage-bow.png` | Sonnet 5 vs Sonnet 5.5, battle d | 297.6s | Bow of `alpha_1` (Sonnet 5's battlecruiser) after 35 nose hits, 8 of them penetrations (the last at 297 s), all five mounts wrecked; its magazine burns at 298 s. Camera 0.66 hull lengths ahead of the centre, 0.2 above, 0.3 to port, looking at the bow; HUD hidden; scale auto |
| `damage-mounts.jpg` | Sonnet 5.5 vs Sonnet 5, battle c | 478s | `beta_1` (Sonnet 5's battleship) from above the bow, 27 s after its forward magazine cooked off (451 s): scorched deck, heavy battery `weapon_3` (destroyed at 453 s) smoking, two breaches venting; camera 0.35 lengths ahead, 0.55 above, 0.45 to port; HUD hidden; scale auto |
| `damage-cookoff.png` | Sonnet 5.5 vs Sonnet 5, battle c | 451.9s | The forward magazine of `beta_1` cooking off (`magazine_cookoff` at 451 s); camera 1.1 lengths ahead, 0.3 above, 0.8 to port; HUD hidden; scale auto |
| `damage-hulk.png` | Sonnet 5.5 vs Sonnet 5, battle c | 510.5s | `beta_1` as a dying hulk (`ship_dying` at 484 s, reactor at 514 s): molten cracks, venting, a secondary explosion, sputtering torch; camera 1.0 lengths ahead, 0.35 above, 0.7 to port; HUD hidden; scale auto |
| `torpedo-strike.png` | Pass-5 tactics showcase | 100.5s | Synchronized torpedo salvo landing on the battleship `beta_1` half a second before its reactor goes; 2.6 hull lengths, azimuth 35°, elevation 18°; HUD hidden |
| `live-deliberation.png` | Claude Sonnet 5 vs DeepSeek V4.1 Flash, replayed through the live endpoints | CP5 (150s) | `?live=1&decisions=1`: live bar, COMMAND DELIBERATION card (both sides' progress, spend), alpha admiral's salvo directive, the salvo one second from impact |
| `draft-splash.png` | Same, live endpoints | T+4s | Draft splash after both sealed picks: rosters, formations and each admiral's reasoning |

## Hull scale

The viewer draws hulls at their real size up close by default (`scale=auto`,
see the root README). Images captured before November 17, 2026 used what is
now `scale=cinematic` (every hull about 60 times oversize); their commands
below carry `--query scale=cinematic` so they still reproduce.

| Scale | Images |
| --- | --- |
| auto (default: real-size hulls up close) | `hero.png`, `dreadnought.png`, `plume-wash.png`, `launch-sequence.png`, `damage-bow.png`, `damage-mounts.jpg`, `damage-cookoff.png`, `damage-hulk.png` |
| cinematic (hulls ~60x oversize) | `hull-lineup.png`, `fleet-battle.png`, `torpedo-salvo.png`, `reactor-breach.jpg`, `dying-hulk.png`, `command-net.png`, `intent-consequence.png`, `blast-shell.jpg`, `fittings-hud.png`, `torpedo-strike.png`, `live-deliberation.png`, `draft-splash.png` |

What was re-shot and why. `hero.png` and `dreadnought.png` are close-ups of
one hull framed in hull lengths: the same commands at the new default give
the same framing of the same hull, so they were re-shot and their commands
reproduce them as written. What differs from the old frames: point-defense
beams are drawn whole, the second dreadnought that sat behind `alpha_1` in
`hero.png` (oversize, 60 km away) is now far too small to see, and a faint
exhaust-jet haze continues the torch plumes. `plume-wash.png` was re-shot with a new camera: the old frame put
a destroyer inside the flame of an oversize battleship, which is not what the
default view shows any more (at real size the destroyer is 6 km astern, in the
faint exhaust jet). `torpedo-salvo.png` was kept as a cinematic-scale image:
its composition (two battleships 20 km apart sharing a frame with a
torpedo's whole terminal run) only exists in that mode, and the README
captions it as such. The other images are HUD shots or scenes where the scale
mode does not change what the caption claims; the root README marks the ones
where oversize hulls are visible against real-size blasts as cinematic.

November 18 re-shoot (drive plume, same commands). The exhaust is now one
light field from the bells outward (`visualizer/src/render/PlumeProfile.js`):
the flame and the spear beyond it both start at the nozzle and only fade, so
the November 17 frames' detached haze is gone. In `hero.png` and
`dreadnought.png` the eight dreadnought plumes are hotter white-blue lances
with a glow at the bells; in
`plume-wash.png` the battleship's exhaust is a single beam that fades from
the bells to the washed destroyer instead of a flame, a dark gap and a pale
cloud around the destroyer. The two dreadnought frames were taken from the
staged copy of the same battle, `p4c_vfix_showcase_s0.json` (substitute it
for `showcase_s0_200pt.json` in the commands if that file is not staged).
The other frames with a lit torch were re-shot the same day with their
commands as written: `dying-hulk.png`, `damage-hulk.png`, `reactor-breach.jpg`,
`torpedo-salvo.png`, `blast-shell.jpg`, `fittings-hud.png`, `torpedo-strike.png`
and `intent-consequence.png` (the showcase s7 frames from its staged copy
`p4c_vfix_showcase_s7.json`). Besides the torch they show the current viewer's
point-defence beams and HUD layout. `fleet-battle.png` and `command-net.png`
(torches a few pixels long) were not re-shot: their recordings were not
staged.

## Recordings

- Claude Sonnet 5 vs DeepSeek V4.1 Flash: `sonnet5_vs_dsv41flash_150pt_20260926.json.gz` (live run of September 26, 2026)
- DeepSeek V4.1 Flash mirror: `p4_new_flash_mirror_150pt.json` (pass-4 engine with captain reports)
- [Qwen–GPT-OSS](../../../data/recordings/intent_qwen_gptoss.json.gz)
- Sonnet 5.5 vs Sonnet 5, battles [c](../../../data/recordings/p5_sonnet_battles_20261116/p5_sonnet55_vs_sonnet5_c/battle_claude_sonnet_5_5_vs_claude_sonnet_5_20261116_001820.json.gz)
  and [d](../../../data/recordings/p5_sonnet_battles_20261116/p5_sonnet5_vs_sonnet55_d/battle_claude_sonnet_5_vs_claude_sonnet_5_5_20261116_003206.json.gz):
  two of the four LLM fleet battles of November 16, 2026 described in
  [`docs/pass-5-findings.md`](../../pass-5-findings.md), staged for the commands below as
  `p5_sonnet55_vs_sonnet5_c.json` and `p5_sonnet5_vs_sonnet55_d.json` (plain or `.json.gz`)
- Pass-5 damage demo: `p5_damage_demo.json`, a seeded heuristic fleet battle
  (regeneration command in
  [`docs/pass-5-recording-additions.md`](../../pass-5-recording-additions.md))
- Launch showcase: `p5_plume2_launch_showcase.json.gz` (`scripts/generate_test_battle.py launch_showcase`,
  also in `data/recordings/`)
- Pass-5 plume demo: `p5_plume_demo.json` (`scripts/generate_test_battle.py plume_wash`,
  see [`docs/pass-5-recording-additions.md`](../../pass-5-recording-additions.md))
- Pass-5 tactics showcase: `p5_readme_torpedo_strike.json` (heuristic captains, `tactical`
  doctrine)
- Showcase s0 / s7: free heuristic draft battles recorded on the pass-4 engine
  with the commands below. The frames here are taken from those pass-4
  recordings; later engine changes can alter seeded battles, so re-running
  the commands today may give a different battle and different timestamps:

  ```bash
  uv run python scripts/run_draft_battle.py --auto-draft --no-admirals --trace \
    --seed 0 --budget 200 --time-limit 420 --doctrine legacy  # -> showcase_s0_200pt.json
  uv run python scripts/run_draft_battle.py --auto-draft --no-admirals --trace \
    --seed 7 --budget 200 --time-limit 420 --doctrine legacy  # -> showcase_s7_200pt.json
  ```

Stage recordings in `visualizer/public/recordings/` as described in the
[root README](../../../README.md#watch-a-replay).

## Capture commands

Each replay image is one `scripts/viewer_snapshot.py` command (add `--port` if
your dev server is not on 5173, and `--init-timeout 420` on a loaded machine:
software GL bakes the sky and compiles every shader at load). `capture-helpers.js` holds the JavaScript the
commands pass through `--eval`: it only calls the viewer's own API
(`window.visualizer`).

```bash
uv run playwright install chromium
SNAP="uv run python scripts/viewer_snapshot.py --gfx high --width 1600 --height 900"
H=docs/images/readme/capture-helpers.js
TRACKS_OFF="window.visualizer.trackControl.set('off')"
CINE="--query scale=cinematic"   # images captured before the real-size default (see Hull scale)

# hero.png
$SNAP --recording showcase_s0_200pt.json --time 96.5 --focus alpha_1 --dist 4.6 --az -36 --el 9 --hide-ui \
  --eval "(() => { $TRACKS_OFF; const c = window.visualizer.scene.camera; c.fov = 20; c.updateProjectionMatrix(); })()" \
  --out hero.png

# hull-lineup.png
$SNAP $CINE --recording rcs_showroom.json --time 5 --hide-ui --eval "$(cat $H); readmeLineup()" --out hull-lineup.png

# fleet-battle.png
$SNAP $CINE --recording sonnet5_vs_dsv41flash_150pt_20260926.json.gz --time 151 \
  --eval "window.visualizer.showFleetView(); window.visualizer.trackHud.setLegend(false)" --out fleet-battle.png

# torpedo-salvo.png
$SNAP $CINE --recording showcase_s7_200pt.json --time 203.3 \
  --camera=-100.27,-14.827,34.354 --target=-112.369,-24.971,1.112 --hide-ui --eval "$TRACKS_OFF" --out torpedo-salvo.png

# reactor-breach.jpg (captured as PNG, then JPEG q95 4:4:4)
$SNAP $CINE --recording showcase_s7_200pt.json --time 205.7 --focus alpha_1 --dist 5.5 --az -20 --el 20 \
  --hide-ui --eval "$TRACKS_OFF" --out reactor-breach.png

# dying-hulk.png
$SNAP $CINE --recording showcase_s7_200pt.json --time 114 --focus alpha_7 --dist 1.35 --az 20 --el 15 \
  --hide-ui --eval "$TRACKS_OFF" --out dying-hulk.png

# command-net.png: 1200x660 CSS at device scale 2, right-hand 800x450 CSS crop
uv run python scripts/viewer_snapshot.py --gfx high $CINE --recording p4_new_flash_mirror_150pt.json \
  --time 216 --focus beta_2 --dist 2.6 --az 165 --el 12 --width 1200 --height 660 --scale 2 \
  --clip 392,117,800,450 --eval "$(cat $H); $TRACKS_OFF; readmeCommsTo('Lancer-2')" --out command-net.png

# intent-consequence.png (the ship is deselected after load so its off-screen
# marker does not stack on the fleet marker)
$SNAP --recording intent_qwen_gptoss.json.gz --query "ship=alpha_2&decisions=1&decision=e152&t=184.8&scale=cinematic" \
  --time 184.8 --focus beta_1 --dist 2 --az 35 --el 18 \
  --eval "$(cat $H); window.visualizer.deselectShip(); readmeDecisionTo('03')" --out intent-consequence.png

# blast-shell.jpg (captured as PNG, then JPEG q95 4:4:4)
$SNAP $CINE --recording p5_damage_demo.json --time 152.6 --focus beta_5 --dist 6 --az 30 --el 18 \
  --hide-ui --eval "$TRACKS_OFF" --out blast-shell.png

# fittings-hud.png
$SNAP --recording p5_damage_demo.json --query "ship=beta_2&scale=cinematic" --time 180 --focus beta_2 --dist 5 --az 30 --el 18 \
  --out fittings-hud.png

# dreadnought.png
$SNAP --recording showcase_s0_200pt.json --time 96.5 --focus alpha_1 --dist 3.0 --az -10 --el 35 --hide-ui \
  --eval "(() => { $TRACKS_OFF; const c = window.visualizer.scene.camera; c.fov = 20; c.updateProjectionMatrix(); })()" \
  --out dreadnought.png

# launch-sequence.png
$SNAP --recording p5_plume2_launch_showcase.json.gz --time 34.1 --focus alpha --dist 1.6 --az 60 --el 20 \
  --hide-ui --eval "$TRACKS_OFF" --out launch-sequence.png

# plume-wash.png
$SNAP --recording p5_plume_demo.json --time 62 --hide-ui \
  --eval "$(cat $H); $TRACKS_OFF; readmeNear('alpha_1', -1.1, 0.5, 1.5, -14)" --out plume-wash.png

# damage-bow.png
$SNAP --recording p5_sonnet5_vs_sonnet55_d.json --time 297.6 --hide-ui \
  --eval "$(cat $H); $TRACKS_OFF; readmeNear('alpha_1', -0.66, 0.2, 0.3, 0.36)" --out damage-bow.png

# damage-mounts.jpg (captured as PNG, then JPEG q95 4:4:4)
$SNAP --recording p5_sonnet55_vs_sonnet5_c.json --time 478 --hide-ui \
  --eval "$(cat $H); $TRACKS_OFF; readmeNear('beta_1', -0.35, 0.55, 0.45, 0.15)" --out damage-mounts.png

# damage-cookoff.png
$SNAP --recording p5_sonnet55_vs_sonnet5_c.json --time 451.9 --hide-ui \
  --eval "$(cat $H); $TRACKS_OFF; readmeNear('beta_1', -1.1, 0.3, 0.8, 0.2)" --out damage-cookoff.png

# damage-hulk.png
$SNAP --recording p5_sonnet55_vs_sonnet5_c.json --time 510.5 --hide-ui \
  --eval "$(cat $H); $TRACKS_OFF; readmeNear('beta_1', -1.0, 0.35, 0.7, 0)" --out damage-hulk.png

# torpedo-strike.png
$SNAP $CINE --recording p5_readme_torpedo_strike.json --time 100.5 --focus beta_1 --dist 2.6 --az 35 --el 18 \
  --hide-ui --eval "$TRACKS_OFF" --out torpedo-strike.png
```

The live images replay the Sonnet–DeepSeek recording through the live
endpoints with `scripts/fake_live_server.py` (40 s synthetic draft: alpha
commits at 60 %, beta at 90 %; each checkpoint held 14 s for deliberation):

```bash
uv run python scripts/fake_live_server.py \
  visualizer/public/recordings/sonnet5_vs_dsv41flash_150pt_20260926.json.gz \
  --port 8825 --speed 6 --draft-seconds 40 --pause-seconds 14
```

A 1600 × 900 Playwright page then opens
`/?live=1&live_base=http://127.0.0.1:8825/live&gfx=high` (plus `&decisions=1`
for `live-deliberation.png`; add `&scale=cinematic` today: both were captured
before the real-size default), with `window.visualizer.slowFrameOk = true` once
the battle starts. `draft-splash.png` is taken about 2.5 s after the battle
starts (the splash still up, both fleets committed); for
`live-deliberation.png` the splash is dismissed, playback set to 4×, the track
legend collapsed, and the frame taken 1.5 s after the deliberation card for
checkpoint 5 appears. The live pacing and deliberation progress come from the
fake server's replay of the recording, not from a running battle.

`p5_readme_torpedo_strike.json` is a frozen copy of the tactical-doctrine showcase battle
(`run_draft_battle.py --auto-draft --no-admirals --trace --seed 4 --budget 200 --time-limit 600`)
as recorded on November 15, 2026, before later engine and price changes; re-running the command
on a later engine gives a different battle.
