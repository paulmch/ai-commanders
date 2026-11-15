# AI Commanders - Development Guide

## Project Overview
Terra Invicta-inspired space battle simulator where LLM captains and admirals
command fleets: real Newtonian physics and combat simulation, with tactical
decisions made by LLMs (via OpenRouter) or humans (via MCP) every 30s checkpoint.

## Setup

```bash
# Use uv for Python package management
uv sync --locked --all-extras   # --all-extras pulls in the mcp extra (mcp, aiohttp)

# LLM battles need an OpenRouter key (MCP-controlled fleets don't)
echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here" > .env
```

## Running Tests

```bash
uv run pytest tests/ -v   # ~2400 tests, ~2-3 minutes
```

CI (`.github/workflows/tests.yml`) runs the same suite on Python 3.10-3.12 with
`uv sync --locked --all-extras`.

## Running Battles

```bash
uv run python scripts/run_llm_battle.py -v                                  # 1v1 duel
uv run python scripts/run_llm_battle.py --fleet-config data/fleet_config_claude_vs_gemini.json -v
uv run python scripts/mcp_battle.py --config data/fleet_config_mcp_example.json  # human via MCP
uv run python scripts/run_draft_battle.py --auto-draft --no-admirals --trace --seed 3  # free battle
```

Free battles fly `HeuristicCaptain` (`src/llm/heuristic_captain.py`). Its default
"tactical" doctrine plans per fleet in `src/llm/heuristic_tactics.py` (reach-aware focus
fire; gun ships nose-first - closure cap, no braking under fire, coast through a
controlled pass facing the nearest gun, HOLD_RANGE once closure is gone; torpedo boats
hold a standoff and fire every launchable round, ranked by target size/agility/PD umbrella; evade
overlay vs torpedoes; crippled or disarmed ships open range, dead drives leave the
formation, ships burn clear of dying hulks' blast radius; plan announced on the FLEET NET);
`--doctrine legacy` (or `BattleConfig.heuristic_doctrine`) keeps the pass-4 captain the
fire-control fixture's heuristic battles were flown with. `scripts/tactics_eval.py`
runs/analyses seeded grids (kills, closest approach, rounds inside 150 km, hit rate).

## Project Structure

```
ai-commanders/
├── data/
│   ├── fleet_ships.json    # Ship specs, weapons, armor - source of truth for hulls
│   └── fleet_config_*.json # Fleet battle configurations
├── src/
│   ├── simulation.py       # Battle engine, threat-aware evasion (RUN/PRESENT)
│   ├── physics.py          # Newtonian mechanics, trajectories, rotation
│   ├── combat.py           # Weapons, armor, damage resolution
│   ├── torpedo.py          # APN guidance, No-Escape-Zone, terminal burn
│   ├── pointdefense.py     # Continuous-dwell PD lasers, per-turret capacitors
│   ├── projectile.py / firecontrol.py / targeting.py / maneuvers.py
│   ├── damage.py / modules.py / geometry.py / thermal.py / power.py
│   ├── battle_damage.py    # damage consequences: surface fittings (mounts, PD,
│   │                       # tubes, drive), reactor blast, magazine cook-off
│   ├── drive_plume.py      # drive plume wash: heat/armour/fittings for ships in a
│   │                       # burning torch's exhaust cone (no RNG, one snapshot per tick)
│   ├── trajectory_prediction.py  # movement-only shadow sim -> per-frame sim_trace 'plan'
│   └── llm/                # OpenRouter client, captain/admiral agents (+vision),
│                           # draft mode + heuristic captains, prompts, tools,
│                           # battle runner/recorder, MCP servers, commander
│                           # notebooks (notebook.py, cross-battle memory),
│                           # range_control.py (anti-merge headings/warnings),
│                           # maneuver_orders.py (standoff/HOLD_RANGE/toward_*),
│                           # communication.py + fleet_comms.py (fleet net,
│                           # captain reports, engine datalink), fleet_intel.py
│                           # (admiral turn state), fire_solutions.py (captain)
├── tests/                  # pytest suite; test_fixes_*.py pin past bug fixes
├── scripts/
│   ├── run_llm_battle.py   # CLI for AI vs AI battles (--fleet-config, --admiral-vision)
│   ├── run_draft_battle.py # draft mode: point-budget fleets + formations
│   ├── draft_tournament.py # seeded round robin of fleet archetypes at equal points (price balance)
│   ├── generate_test_battle.py  # scripted no-LLM recordings for the viewer
│   ├── viewer_snapshot.py  # headless Playwright screenshots of the 3D viewer
│   ├── viewer_clip.py      # frame-stepped MP4/GIF clips (director or camera path, resumable)
│   ├── mcp_battle.py       # CLI for MCP-controlled battles
│   ├── refine_commander.py # Post-battle lesson distillation + rematch gating
│   ├── fire_control_calibration.py  # gun P(hit) vs engine outcomes (recordings + seeded harness)
│   ├── perf_check.py       # engine speed + bit-equality: seeded battles and a calibration
│   │                       # slice hashed against perf_check_golden.json, CPU time per case,
│   │                       # --profile CASE. An optimisation must leave every hash identical;
│   │                       # re-record (--record) only on a commit that changes behaviour, with
│   │                       # the Python minor version the golden is compared under (3.11 and
│   │                       # 3.12 give different low bits: use --golden PATH for another one)
│   └── calculate_shots_to_kill.py  # Regenerates docs/ships.md combat tables
├── docs/ships.md           # Ship specs + simulated shots-to-kill tables
└── visualizer/             # Three.js 3D battle replay viewer; src/render/ holds
                            # the PBR hull kit (Hulls + HullLights/Markings/Rigs),
                            # Projectiles (in-flight weapons), DamageFX, Rcs,
                            # Tracks, Environment/SkyTextures, PostFX/Bloom/Quality,
                            # HullWear, TargetingLines, Torch (volumetric plumes),
                            # GunLooks, BlastShell, sunShadow; src/ holds the
                            # spectator HUD (HudIndex, Scoreboard, KillFeed,
                            # CommsFeed, TimelineMarkers, Registry, FittingsIndex
                            # + FittingsHud: per-ship fitting/crew state), the
                            # auto director (Director, DirectorCamera) and live mode
                            # (LiveSession, LiveCards, DraftSplash,
                            # OffscreenIndicators, ViewInsets)
```

## Key Constants

- Exhaust velocity: 10,256 km/s
- Main thrust: 58.56 MN (Protium Converter Torch x6)
- Combat thrust vectoring: 1° deflection
- Target delta-v: 500 km/s
- Trident torpedo: 250 kg penetrator, 14 km/s delta-v, 12g; guidance holds a
  ~12 km/s closure floor and dumps remaining delta-v in the terminal burn
- PD laser: 250 km envelope, continuous dwell; one turret blinds ~5 km/s of closure
- Coilguns fire Terra Invicta salvos: batteries 5 shots 10 s apart then 20 s
  cooldown (5 per 60 s), spinals 5 per 72 s, siege coiler 3-shot salvo
- Heat (TI-derived): Mk3 guns 0.7 efficient (coil 0.31 / heavy 0.52 / spinal
  1.84 / siege 3.11 GJ per shot), PD 30 MW while dwelling; every hull has 60 t
  radiators (780 MW extended) + a 320 GJ sink, and full-throttle drive heat
  equals radiator rejection. Ladder: guns hold at 90%, radiators forced out at
  100%, reactor meltdown at `REACTOR_MELTDOWN_HEAT_FRACTION` (150%, src/thermal.py)
- Damage consequences (`src/battle_damage.py`, `tests/test_battle_damage.py`,
  recorded per `docs/pass-5-recording-additions.md`): any hull hit can strike an
  external fitting (15% nose / 25% flank or tail; gun mounts and torpedo tubes
  35% health per GJ, PD turrets 60%; a damaged mount cycles up to 2x slower, a
  wrecked tube loses its rounds; 30% of tail hits strike the drive, 15% thrust
  per GJ). Every death is a reactor blast (15 km for a cruiser, x sqrt(dry
  mass)): facing armour, heat pulse, extended radiators, fittings, dying hulks'
  fuses; torpedoes die inside 25% of the radius and go blind inside 60%. A
  destroyed magazine burns half the rounds and blows non-critical neighbours.
  These rolls use `CombatSimulation._damage_rng` (seed-derived), never the
  battle RNG that decides hits
- Drive plume wash (`src/drive_plume.py`, `tests/test_drive_plume.py`, recorded as
  `plume_wash` per `docs/pass-5-recording-additions.md`): a burning torch (300 TW
  jet, the same on every hull) washes any ship within 7.5° of dead astern out to
  100 km - on axis at full burn 40 GJ/s at 5 km, 16 at 10 km, 4.7 at 20 km, 0.8
  at 50 km (Gaussian 3° core, scaled by the throttle actually produced); above
  4 GJ/s it also burns armour and fittings off the washed facing. Heat stops at
  149% (never a kill by itself). Above 0.78 GJ/s the wash also shreds EXTENDED
  radiator panels (`radiator_damaged`, cause `plume_wash`; retracted ones are
  safe) and the heat ladder does not force radiators out while the ship is in
  such a jet (`drive_plume.radiators_in_jet`, gate in
  `_update_radiator_automation`). A turning emitter sweeps its jet (axis sampled
  every 1.5° between ticks, so a sweep cannot tunnel past a ship). Crew
  radiation dose (`crew_dose_sv`, permanent): wash rate x 0.03 Sv/GJ x
  exp(-armour cm / 100) on the washed facing; 2 Sv = impaired (guns and tubes
  cycle x1.5), 8 Sv = incapacitated (`crew_incapacitated`: coasts, never fires,
  cannot deal damage for the victory rule); `crew_radiation` events. No RNG,
  order-independent, off in the torpedo forecast shadow
  (`sim.plume_wash_enabled`). Drive interlock (`interlock_caps` /
  `apply_interlock`, always on, also while evading; exemption constant
  `PLUME_INTERLOCK_EXEMPT_TORPEDO_EVASION` is off): every ship's throttle is
  capped each tick, from the start-of-tick snapshot, so its exhaust puts at
  most 0.5 GJ/s on a FRIENDLY ship; enemies never limit it; `drive_interlock`
  events, "DRIVE INTERLOCK: throttle held ..." status. An incapacitated crew is
  asked nothing by the runner (UNABLE report, compliance "not executable"),
  MCP commands to it are rejected; crew state and heavy wash go out on the
  FLEET NET datalink (`CREW_STATE`, `PLUME_WASH`). Torpedoes are affected only behind
  `PLUME_AFFECTS_TORPEDOES` (off). Tactical heuristic captains ease the drive
  for a friendly astern (`heuristic_tactics.plume_safe_throttle`); commanders
  see PLUME WASH / PLUME RISK status lines
- Two-stage kills: a non-reactor killing blow leaves an untargetable dying hulk
  (tumbling, torch sputtering) that detonates 0-30 s later; hits on the hulk
  shorten the fuse, a reactor hit detonates it at once (`test_dying_ships.py`)

## Control Architecture

Hierarchical control:
1. **Admiral LLM** (fleet battles): two-phase orders every checkpoint - fleet
   directive plus per-captain orders, including coordinated torpedo salvos.
   Keeps a private standing battle plan (set_battle_plan) echoed back each
   checkpoint - its only cross-checkpoint memory of its own intent
2. **Captain LLM**: tool calls every 30s decision cycle (maneuver, weapons order,
   torpedo launch, radiators, captain's-log notes via log_note, and comms:
   report_to_admiral - DEVIATE/UNABLE/request/contact, acting on orders is the
   ACK; respond_to_orders is its alias - fleet_signal to its own fleet's FLEET
   NET, send_message on the open channel the ENEMY reads, discuss_with_admiral).
   Tool surface is derived from the hull's actual armament - gun tools only on
   gun ships, torpedo tools only on torpedo ships

Fleet comms (src/llm/communication.py, fleet_comms.py): captain reports reach
the admiral at the next checkpoint (CAPTAIN REPORTS, with the engine's
ordered-vs-executed compliance); fleet_signal traffic, admiral notes and the
engine datalink (launches with target and ETA, kills, losses, PD losses) reach
captains as FLEET NET, enemy open-channel traffic as ENEMY TRANSMISSIONS.
Recorded as comms_message / comms_delivered / captain_report events. A failed
directive call leaves captains "(no new fleet directive this checkpoint)" and
is recorded as admiral_directive_failed.

Within a checkpoint, independent LLM calls (both admirals, per-ship orders, all
captains) run concurrently (BattleConfig.parallel_llm, on by default).
Cross-battle: scripts/refine_commander.py distills notebook lessons from
recordings, rematch-gates them, and battles opt in via use_notebooks/--notebooks.
3. **Rule-based layer**: executes between checkpoints - fire control modes,
   automatic point defense, threat-aware evasion (jink vs guns, RUN from guided
   torpedoes, PRESENT thickest armor when a hit is unavoidable; with no threat
   EVADE weaves around the ship's CURRENT course and never chases)

Range control (anti-merge): HEADING turns first, then burns (throttle 0 =
attitude only, RCS, velocity untouched). `src/llm/range_control.py` gives
captains, admirals and MCP clients, per enemy group, closing speed, closest
approach, stopping distance, a merge warning and ready-made world-frame headings
(cancel closure - "hold range" once the closure is gone / open range / re-engage
after a pass / pass wide / flank left|right / coast & shoot - "coast & launch" on
gunless torpedo hulls) whose stated effects are pinned by
`tests/test_range_control_and_evade.py`. An admiral
`issue_order(ship_name="all ships", heading, throttle)` is flown as a formation:
throttles matched to the slowest hull, burn start synced (MCP: ship_id "all").
A per-ship heading/maneuver REPLACES it for that ship (MAINTAIN/PADLOCK = engine
off; given with a heading - PADLOCK only at throttle > 0 - the ship flies that
HEADING instead); `evade` alone rides on it - or, with no fleet order, on the
ship's current maneuver (`src/llm/maneuver_orders.py`, `tests/test_order_semantics.py`).
The evade overlay burns at the order's throttle, at full throttle under a
throttle-0 / engine-off order (`CombatSimulation._evasion_overlay`).
Plain INTERCEPT never brakes (it merges); the anti-merge forms
(`src/llm/maneuver_orders.py`; the same fields on captain set_maneuver /
set_heading, admiral issue_order and MCP set_maneuver): INTERCEPT with
standoff_km (+ arrival_closing_kps) closes, flips and holds that range;
HOLD_RANGE keeps range_km to a target; HEADING toward_point flies to a position
and stops there, toward_ship falls in alongside a ship; evade ('full'|'dodge')
overlays threat evasion on any other maneuver.

MCP mode replaces the admiral+captains of a fleet with any MCP client
(`src/llm/mcp_server.py`, one battle HTTP API on port 8765, `--faction alpha|beta`).
A `draft` block in the fleet config (or `--draft` on `scripts/mcp_battle.py`)
inserts a pre-battle draft phase: MCP clients buy fleets and place formations
via `get_draft_state`/`select_fleet`/`set_formation`/`ready`
(`src/llm/mcp_draft.py`), non-MCP sides draft via their admiral LLM or
auto-draft. The same HTTP server serves live spectator endpoints
(`GET /live/recording` incremental via `?since_t=`, `GET /live/predictions`
with per-ship constant-acceleration paths to the next checkpoint); the
visualizer's `?live=1` mode polls them through a Vite proxy, and
`scripts/fake_live_server.py` replays a saved recording through the same
endpoints for viewer development.

## Ship Classes

| Ship | Accel | 90° Turn (TV) | 90° Turn (RCS) |
|------|-------|---------------|----------------|
| Corvette | 3.0g | 12.1s | 54.2s |
| Frigate | 3.0g | 15.1s | 83.3s |
| Destroyer | 2.0g | 20.6s | 127.6s |
| Cruiser | 1.5g | 28.2s | 206.3s |
| Torpedo Cruiser | 1.5g | 28.2s | 206.3s |
| Battlecruiser | 1.5g | 28.2s | 206.3s |
| Battleship | 1.0g | 36.9s | 288.7s |
| Dreadnought | 0.75g | 49.9s | 458.4s |

The corvette (1 launcher) and torpedo cruiser (4 launchers, 12 rounds each) are the
torpedo-armed hulls; all others fight with coilguns.

## Recording Files

New recordings are gzip (`battle_*.json.gz`: compact JSON, deterministic bytes:
mtime 0, no file name); old plain `.json` recordings stay valid everywhere. All
Python reads/writes go through `src/recording_io.py` (`load_recording` sniffs the
gzip magic, `dump_recording` is atomic, `sidecar_path` gives `battle_x.draft.json`);
never `json.load(open(recording))`. The viewer decodes through
`visualizer/src/RecordingCodec.js` (magic-byte sniff + `DecompressionStream`).
`BattleConfig.compress_recording` / `save_recording`; `--plain-json` on the battle
scripts, `run_draft_battle.py --no-recording`; `scripts/compress_recordings.py`
converts a folder (verify before delete). `BattleRecording.to_json` (plain text,
hashed by the perf harness) is unchanged.

## Viewer Rendering Notes

- All custom `ShaderMaterial`s must include `<common>` + the `logdepthbuf_*`
  chunks (the renderer uses a logarithmic depth buffer; without them the
  shader fails to compile or depth-tests wrongly).
- `IcosahedronGeometry(r, detail)` subdivides linearly (20*(d+1)^2 faces) -
  use detail 12-28 for smooth blast spheres, not 3-5.
- The hull kit (render/Hulls.js) builds parts from `chamferBoxGeometry` and
  `octSectionGeometry` (shouldered sections, wedge bows), all non-indexed and
  merged per material. Plated materials (`hull`, `trim`, `paint`, `dark`,
  `barrel`) use vertex colours and one hull-space UV lattice: geometry added
  outside `HullKit.add` needs a `color` attribute and UVs at `kit.texel` scale
  or it renders black. `mats.bellHeat` is a MeshLambertMaterial; drive only its
  `emissiveIntensity`.
- Effects animate on battle time (spawnTime vs currentTime) so timeline
  scrubbing is deterministic; wall-clock only drives texture turbulence.
- Verify visually with `uv run python scripts/viewer_snapshot.py` (see
  README); llvmpipe frames take >1 s, so the harness sets
  `visualizer.slowFrameOk` to bypass the stalled-tab update guard.
  Graphics quality (render/Quality.js; G key, `?gfx=low|high|auto`) auto-picks
  low on software GL (no MSAA/god rays/dust) - pass `--gfx high` for beauty shots.
- `--trace` recordings carry optional per-ship frame fields: `acc`, `omega`,
  `alpha`, `turn` ("tv"|"rcs"), `evade`, `tgt`, and `plan` {t_cp, dt, path,
  sigma, fwd, mode, stale?} (predicted path to the next checkpoint; sigma is
  calibrated per evasion mode; checkpoint frames are re-predicted after the
  new orders apply, `stale` marks them until then); header `decision_interval_s`.
  Torpedo frames add thruster telemetry `thr`, `fwd` (engine burning) and
  `rcs` (lateral RCS thrust vector, rare: only small precision trims) -
  recording only, never read by guidance/physics. Torpedoes have no fins.
  Every consumer (TrajectoryPredictor, Rcs) must fall back gracefully on old
  recordings without them. Prediction must never draw from the battle RNG or
  change results - `tests/test_trajectory_prediction.py` pins seeded on/off
  equality.
- Render globals (lighting, shadows, IBL, post) live in render/Environment.js,
  render/PostFX.js, render/Bloom.js, render/sunShadow.js and render/Quality.js;
  effects and hull materials are tuned against them, so change them in one
  place and re-shoot the reference frames. High tier only: sun shadow map
  (box fitted each frame to the hulls on screen, `chooseShadowFocus`; hull
  meshes are flagged cast/receive by material type in
  `SceneManager._flagShadowMeshes`, so any opaque MeshStandardMaterial hull
  part casts), screen-space AO (reads the LOGARITHMIC depth texture:
  w = exp2(d * log2(far + 1)) - 1; held off emitters by luminance), the
  anamorphic streak. Both tiers: the print grade and exposure adaptation
  (metered from the coarsest bloom level, stateless so scrubbing stays
  deterministic). Shadow-side fill is image-based (PMREM: cool anti-sun lobe,
  planet bounce, floor - `ENV_FILL`/`ENV_BOUNCE`/`ENV_AMBIENT`), never drawn
  on the visible sky.
- Auto director: `src/Director.js` turns the recording into a shot list (pure,
  no THREE; `lookahead: false` in live mode is causal), `src/DirectorCamera.js`
  frames each shot type as a pure function of (shot, battle time, positions), and
  `CameraController` applies it plus blast shake (`Director.shakeAt`, battle time).
  Never add wall-clock easing to a director pose: `tests/director.test.js` pins
  seek-away-and-back equality. Shake is on the camera only for the draw (`main.js`
  wraps `scene.render()` in `applyShake` / `removeShake`). Kill shots and shake are
  keyed on `DamageTimeline.shipDetonationTime` (the frame the viewer blows the ship
  up), the pull-back distance on `Explosion.blastExtent` - change a blast's curves
  in `Blast.update` and that function together.
- Ownership: in-flight weapons live in render/Projectiles.js (WeaponTimeline
  decides what exists at time t); persistent hit aftermath and deaths in
  DamageTimeline.js + render/DamageFX.js; hulls expose `userData.rcsMounts` /
  `pdMounts` for Rcs.js and the PD beams.

## Doc/Prompt Invariants

- Numbers quoted in captain/admiral doctrine (`src/llm/prompts.py`) are bound to
  engine constants or seeded runs by `tests/test_doctrine_accuracy.py` - change the
  engine and the tests will tell you which prompt claims drifted.
- Gun P(hit) (`firecontrol.calculate_hit_probability` = conversion x track, via
  `CombatSimulation.gun_firing_solution` for the FIRE_WHEN_OPTIMAL gate, captain/MCP
  forecasts and shot_fired) is pinned against what the engine delivers by
  `tests/test_fire_control_calibration.py`, per target maneuver class (EVASIVE, evade
  overlay, HEADING turning / burning, INTERCEPT, PADLOCK) x 100 km so errors cannot
  cancel. The dispersion roll consumes only the conversion part; shot_fired records the
  forecast's inputs (`fc`: dodge, missed jink burn, pending HEADING burn). If the gate,
  the roll, evasion/dodging or the lead changes, re-run `scripts/fire_control_calibration.py`
  (harness + table) and regenerate the fixture; the doctrine P(hit) table is bound by
  `tests/test_prompt_structure.py`. Known residual: LLM fleets dodging a whole fleet's
  fire were hit even less than forecast (firecontrol.JINK_ACCEL_FRACTION).
- Side symmetry: the engine must not care which fleet is alpha or was added
  first. `CombatSimulation.step()` resolves weapons fire, point defence and
  the ship updates (`_update_ships`: every ship steers on the start-of-step
  state of the others, new states are staged and installed together) each from
  one snapshot. `tests/test_side_symmetry.py` pins it exactly: mirrored
  RNG-free (or flat-RNG) engagements stay mirror images and do not depend on
  insertion order. Per-tick logic that reads another ship must never see a
  ship already updated this tick (a mirrored HOLD_RANGE pair used to split
  into CLOSE and HOLD that way, and mirror battles went to one side by roster).
- After any armor or damage-model change, regenerate the combat tables in
  `docs/ships.md` with `uv run python scripts/calculate_shots_to_kill.py`.
- Draft point costs are DERIVED from `data/fleet_ships.json`, not hand-typed
  (`ship_point_cost` in `src/llm/fleet_draft.py`): armour tonnage + PD turrets +
  gun energy throughput + torpedo magazine depth x per-round yield. Change a
  hull's armour or magazine and its price moves on its own; the cost table in
  `docs/draft_mode.md` then needs regenerating. Never re-add a hard-coded
  `SHIP_POINT_COSTS` literal - torpedo armament was nearly free that way.
  After a change to torpedo guidance, point defence, the heuristic captains or
  the time-limit scoring, re-run `scripts/draft_tournament.py` (cached by
  engine source + rosters + hull prices + seed) before touching
  `TORPEDO_POINTS_PER_GJ`. Every hull must stay draftable at the default
  100-point budget (pinned), and the time-limit verdict values a hull at its
  draft price (`ship_class_value`; `SHIP_CLASS_VALUES` is only the fallback).
