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

### 1. Tactical plot (matplotlib) - `src/llm/admiral_view.py`

```bash
# the image an admiral with vision got (or would have got) at checkpoint 3
uv run --extra viz python scripts/render_admiral_view.py data/recordings/battle_X.json.gz \
    --checkpoint 3 --faction alpha --out /tmp/cp003_alpha.png
# every checkpoint, both sides
uv run --extra viz python scripts/render_admiral_view.py battle_X.json.gz --all --out-dir /tmp/plots
```

The admiral's text turn state carries every number; the picture is for the
geometry text cannot carry. Redesigned in pass 5 (lane `admplot`) after an
audit found no admiral reply in 1,229 calls that referred to the old plot
(16 stat cards over the map, an unreadable 3D panel, one diamond per torpedo).

One fixed **1280 x 720** PNG (~1,229 image tokens at w x h / 750, the same
budget as before; under every model's downscale cap, so the model reads it
pixel for pixel). The legend lives in the cached system prompt
(`prompts.ADMIRAL_VISION_LEGEND`), not in the image.

- **Header line**: time, checkpoint N of M, which colour is you (your fleet is
  always cyan, the enemy orange), torpedoes in flight per side (unguided
  counted apart), the score when `AdmiralConfig.show_score` is on.
- **Plan** (top-down, world X right / Y up: the axes of the turn state's
  world-frame headings), km grid and scale bar.
  - Ships: dart glyphs pointing along the nose (white nose dot), length by
    class (15 px corvette .. 31 px dreadnought), filled from the tail in
    proportion to hull integrity, a torch flame when the drive produces
    thrust (applied throttle >= 0.1). The plan projection of the nose is
    honest: a ship pointing out of the plane draws foreshortened.
  - Labels: the ship's exact full name (what the text and `issue_order` use;
    pinned through `fleet_intel.resolve_ship_ref`), hull % when damaged,
    `SIDE` / `TAIL` when that is the face it shows the other fleet's centre
    (the engine's 30/150 deg hit cones, as the turn state's enemy facing),
    `Z+N` beyond 50 km above/below the viewer's fleet median height. Placed by
    a greedy collision-avoiding layout (8 directions on growing rings, leader
    lines beyond the first ring, preferring the side away from the other
    fleet): never over a glyph, the zoom panel or another label.
  - Motion: a thin line to a hollow ghost dart where the ship will be at the
    next checkpoint if it coasts; the ghost turns yellow when it is within
    150 km of an enemy ghost (the doctrine's gun band).
  - Dead ships: crosses; a white cross and `NAME DESTROYED` for a loss since
    the previous checkpoint (diffed against that checkpoint's frame).
  - Envelopes: per fleet, the outline of the union of 150 km circles around
    its gun ships (solid) and of its PD range around PD ships (dotted, 250 km
    from fleet data). 150 km is the doctrine's effective band ("beyond ~150
    km a dodger is rarely hit"); the guns' nominal 900 km is not drawn.
  - Torpedoes: one dot per round (hollow grey = unguided: blinded seeker,
    dry or target lost, but still on a hull's line; spent rounds on nobody's line are only counted) and
    ONE arrow per (launching group, target): `N -> TARGET Ts`, first impact
    in T s, `+k unguided`, `NEW` for a round launched since the last checkpoint.
    A launching group is a side's ships linked within 150 km. State, target
    and ETA follow the turn state's own model (`fleet_intel.gather_rounds`:
    the guidance law's pursuit for guided rounds, the engine's closest-approach
    test for ballistic ones); a trace frame derives the round's mass from its
    recorded delta-v.
  - Closest approach of the two fleet centres if both coast (within 600 s).
  - **Zoom panel**: when ships of both fleets sit closer than 24 px on the
    plan, a 400 x 225 panel in the free corner (or top/bottom middle) with
    the fewest glyphs magnifies the crowded region (at least x2, dashed box on
    the plan); ships inside the box are labelled there.
- **Side view** strip under the plan: the same X columns, height Z up, at the
  plan's scale or compressed by 2, 4, 8 ... when the height spread does not fit
  (never exaggerated); ships standing out of the viewer's fleet plane are named.

**Framing rule** (stateless, `choose_view`): the plan width is the smallest
step of a fixed ladder (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8 x 10^k km) that
holds every live ship, ghost, ship lost since the last checkpoint and
threatening round at least 100 px (x) / 30 px (y) inside the edges, with the
centre snapped to 1/20 of that width. Same state, same frame; consecutive
checkpoints share a frame until the content outgrows or shrinks below a step.

Rendering is deterministic (Agg, bundled DejaVu Sans, no wall clock, PNG
written without the matplotlib version stamp; tests compare bytes of two
renders) and takes ~0.16 s per frame (median, loaded 16-core box) including the torpedo forecasts (the old
two-panel plot: ~0.76 s, same frames and machine). Live and replay share one code path:
`build_live_frame` (live sim) and `frame_from_trace` (recording) produce the
same frame schema, `build_scene` computes everything shown (pure),
`layout_scene` places it (pure), `render_layout_png` draws it.
`tests/test_admiral_plot.py` pins labels vs glyphs for random fleets, names,
framing, wave aggregation and ETAs against the turn state, determinism.

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
   the frame each side saw at its previous checkpoint (what changed since).
2. At each admiral checkpoint, `_render_admiral_frame` renders the live sim
   to the plot above, with the admiral's checkpoint number, the battle
   clock's total and (score flag only) the score. Frames are also saved to
   `data/recordings/vision/<timestamp>/cpNNN_<faction>.png` for post-battle
   review; `scripts/render_admiral_view.py` re-renders any checkpoint from a
   `--trace` recording.
3. The PNG rides into the admiral's Phase-1 directive call as an
   `image_url` content part (base64 data URL); its legend is in the cached
   system prompt (`prompts.ADMIRAL_VISION_LEGEND`).
   Phase-2 per-ship order calls stay text-only (image cost x N ships was
   the budget concern). OpenRouter forwards the parts to multimodal models.
4. Gated by (a) `"vision": true` on the admiral config (or the
   `--admiral-vision` CLI flag on `run_llm_battle.py`; draft mode enables
   it by default), (b) `is_vision_model()` - a conservative model-family
   allowlist, and (c) matplotlib being installed (`viz` extra). Any miss
   falls back silently to text-only. `AdmiralConfig.image_input` overrides
   (b) with a catalog-verified answer: `scripts/run_live_draft_battle.py
   --admiral-vision` refuses to start unless OpenRouter lists `image` in
   both models' `input_modalities`, then sets it, so vision is never
   silently dropped in a paid live battle. Its frames land in
   `data/recordings/<name>/vision/`.
5. Budgeted battles (`RequestBudget`, `src/llm/spending.py`) reserve each
   image as prompt tokens from its pixel size - Anthropic ~w*h/750 capped
   at 4784 (the 2576 px high-res limit), other providers at least 3,000 -
   priced at the higher of the prompt and advertised `image` rate. The
   base64 bytes never count as text.

Resolved questions:

- **Honesty**: the plot shows the same positions/velocities the text
  snapshots already give both admirals - geometry comprehension, not new
  intelligence. A per-faction sensor filter is still future work if sensor
  limits ever land.
- **Token cost**: paid only on the one directive call per checkpoint.

Vision-capable allowlist additions (2026-08-06): `kimi-k3` and
`qwen3.8-max`. First competitive match with vision on BOTH sides:
Kimi K3 vs GPT-5.6 Terra (frames in `data/recordings/vision/20260806_155054/`).

Still open:

- **A/B value test**: run the same battle with/without the image and
  compare admiral decision quality (cheap with scripted captains).
