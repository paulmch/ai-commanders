# Admiral Vision: Battle Images for LLM Commanders

Prototype and assessment for giving admirals/captains a *graphical* view of
the battle alongside the numeric status reports.

## Why

Text reports carry scalars well (hull %, distance, heat) but geometry poorly.
"Two torpedo streams converging from bearing 040 and 320 while the enemy
kites away" is one glance in a picture and a paragraph of coordinates in
text. Multimodal models (Claude, Gemini, GPT-4-class) can consume that
glance directly.

## What exists now

Two working render paths, both driven by the same recording schema
(`sim_trace` frames + events):

### 1. Tactical plot (matplotlib) - `scripts/render_admiral_view.py`

```bash
uv run --extra viz python scripts/render_admiral_view.py \
    --recording data/recordings/battle_X.json --time 95 --out /tmp/view.png
```

Top-down plot with oriented ship markers + stat cards (name, class, hull,
thrust state, speed, maneuver), faction-colored torpedo diamonds with
25s trails, disabled torpedoes greyed out, live PD beam lines, projectile
dots, ship-to-ship range, in-flight ordnance counts, and a recent-events
ticker. Dark, high-contrast, monospace - tuned for a vision model reader.

- Cost: ~150 ms in-process per frame (matplotlib Agg), no browser, no GPU.
- `render_frame()` takes a plain frame dict - exactly what the live
  simulation produces each tick - so it works mid-battle, not just in
  replays. `matplotlib` is behind the `viz` optional dependency group.

### 2. Cinematic 3D frame (viewer + headless Chrome)

The replay viewer (now with torpedo, continuous-PD-beam and improved
rendering) can be driven headlessly - the Playwright verification harness
does exactly this: load recording → seek → screenshot. Beautiful, but each
frame needs a vite server + Chrome (~seconds, GPU-less WebGL via
SwiftShader/ANGLE is flaky under parallel load). Right for post-battle
highlight reels and docs, wrong for per-checkpoint admiral calls.

## Integration (WIRED)

Implemented in `src/llm/admiral_view.py` + hooks in `battle_runner.py` and
`admiral.py`:

1. When any admiral has vision, the runner keeps an `AdmiralViewState`:
   torpedo positions sampled every sim step (25 s trail window) and a
   rolling event buffer (PD beams, ticker events).
2. At each admiral checkpoint, `_get_admiral_decision` renders the live sim
   to a PNG - left panel annotated top-down plot, right panel a true 3D
   perspective (Z gets a minimum visual thickness so planar battles stay
   readable; tick labels carry real values). Frames are also saved to
   `data/recordings/vision/<timestamp>/cpNNN_<faction>.png` for post-battle
   review.
3. The PNG rides into the admiral's Phase-1 directive call as an
   `image_url` content part (base64 data URL) with a legend in the text.
   Phase-2 per-ship order calls stay text-only (image cost x N ships was
   the budget concern). OpenRouter forwards the parts to multimodal models.
4. Gated by (a) `"vision": true` on the admiral config (or the
   `--admiral-vision` CLI flag on `run_llm_battle.py`; draft mode enables
   it by default), (b) `is_vision_model()` - a conservative model-family
   allowlist, and (c) matplotlib being installed (`viz` extra). Any miss
   falls back silently to text-only.

Resolved questions:

- **Honesty**: the plot shows the same positions/velocities the text
  snapshots already give both admirals - geometry comprehension, not new
  intelligence. A per-faction sensor filter is still future work if sensor
  limits ever land.
- **Token cost**: paid only on the one directive call per checkpoint.

Still open:

- **A/B value test**: run the same battle with/without the image and
  compare admiral decision quality (cheap with scripted captains).
