# Pass 5 recording additions

New facts the engine writes into battle recordings in pass 5, for the viewer
lanes. Everything here is **additive**: old recordings simply do not contain
these events/fields, and every consumer must treat them as optional.

General rules that apply to every event below:

- Events sit in the recording's top-level `events` array with the usual
  envelope: `event_id`, `sequence`, `timestamp` (battle seconds), `event_type`,
  `ship_id`, `data`.
- `data.target_id` is always present (often `null`) - the recorder adds it to
  every engine event. `data.command_ids` / `data.decision_ids` /
  `data.ordnance_id` may appear on any event (replay-evidence enrichment) and
  can be ignored.
- Positions are metres in the battle frame (same frame as `sim_trace[].ships[].pos`),
  distances in the named unit (`_km`, `_cm`), energies in GJ, powers in kW.
- All events are emitted on the tick they happen (1 s ticks); several can share
  one `timestamp`. Within one timestamp, `sequence` gives the engine order.

## Damage lane (`src/battle_damage.py`)

Demo recording containing every event type below:
`visualizer/public/recordings/p5_damage_demo.json`. It is **not committed**
(`visualizer/public/recordings/` is gitignored): regenerate it with the
command under "Demo recording" at the end of this section, which also lists
its timestamps; "Variants not in the demo recording" covers the four field
shapes it does not contain.

Magnitudes are named constants in `src/battle_damage.py`; the numbers quoted
here are the values at the time of writing.

### `mount_damaged` - an external fitting was damaged or knocked out

`ship_id` = the ship that owns the fitting.

Emitted when:

- a hull hit (coilgun slug `hit` or `torpedo_impact`, penetrating or not) also
  strikes an external fitting (`cause: "surface_hit"`). It follows the `hit` /
  `torpedo_impact` event of the same impact on the same `timestamp`. Chance
  per hull hit: 15% nose, 25% lateral, 25% tail (a gun, tube or PD turret,
  picked uniformly among the surviving ones; the spinal mount only on a nose
  hit). A tail hit additionally has a 30% chance to strike the drive instead
  (`kind: "engine"`).
- a reactor blast catches the fitting (`cause: "reactor_blast"`); it then
  follows that victim's `blast_damage` event and precedes the `reactor_blast`
  summary.

One event per fitting per impact. Fitting health is absolute, so the **latest
`mount_damaged` for a given (`ship_id`, `kind`, `index`) at or before time t is
that fitting's state at t**; a fitting with no event is intact (100).

| field | type | meaning |
|---|---|---|
| `kind` | string | `"gun"` (coilgun mount), `"pd"` (point-defence turret), `"launcher"` (torpedo tube), `"engine"` (drive assembly) |
| `slot` | string | `gun`/`launcher`: the weapon slot id (`weapon_0`...), same ids as `shot_fired.weapon_slot` and the fleet spec `weapons` keys. `pd`: the turret name (`pd_0`...), the same string as `pd_fired.turret_name` (the key the hull's PD rigs use). `engine`: the engine module name (`"Main Engine Assembly"`) |
| `index` | int | `gun`: position of the slot in the ship's weapon list (fleet-spec order, `weapon_0` = 0). `pd`: index into the ship's PD turret list (`pd_N` = N). `launcher`: Nth torpedo tube of the hull (0-based). `engine`: Nth engine module (0 on every current hull) |
| `weapon_name` | string | display name (`"Coilgun Battery Mk3"`, `"PD Laser"`, `"Torpedo Launcher"`, engine module name) |
| `health_pct` | float 0-100 | fitting health after this event |
| `health_lost_pct` | float | health removed by this event |
| `destroyed` | bool | `true` exactly when `health_pct` is 0 (a fitting left under 0.5 % is destroyed): the gun/tube/turret never fires again, a destroyed engine gives no thrust |
| `cooldown_factor` | float or null | `gun`/`launcher`, while not destroyed: the mount now needs this many times its normal cooldown between shots (1.0-2.0 = `min(2, 100/health_pct)`). Always `1.0` for a damaged `pd` turret (it keeps its full output until it is destroyed). `null` for `engine` and for every destroyed fitting |
| `thrust_fraction` | float 0-1 | `engine` only: the ship's available thrust after this event. It equals engine health / 100 while health is above 25%; at 25% or less the drive stops producing thrust altogether (0 = adrift), before it is formally destroyed |
| `torpedoes_lost` | int | `launcher` only, only when `destroyed`: rounds lost with the tube |
| `torpedoes_remaining` | int | `launcher` only: rounds left on the whole ship after this event |
| `cause` | string | `"surface_hit"`, `"reactor_blast"` or `"plume_wash"` (a fitting burned by another ship's drive plume, see "Plume lane"; `source_ship_id` is then the ship whose exhaust it is) |
| `hit_location` | string or null | struck facing: `"nose"`, `"lateral"`, `"tail"` |
| `source_ship_id` | string or null | shooter (surface hit) or the ship whose reactor detonated (blast) |

Effects a spectator can see afterwards: a degraded gun/tube fires at a slower
cadence, a destroyed one falls silent; the ship's PD beam count drops by one
per destroyed turret; an engine hit lowers the ship's acceleration (frames'
`thrust` falls below `requested_thrust`). A destroyed engine is additionally
recorded as a `module_destroyed` (`module_name` = engine module,
`cause: "surface_hit"`).

Not every silenced weapon has a `mount_damaged`: the older path still exists,
where a penetrating hit destroys a weapon *module* inside the hull
(`module_destroyed` with a weapon module name such as `Spinal Coiler Mount`,
`Dorsal Turret Mount`, `Coilgun Battery`, `PD Laser Dorsal`) and the engine
disables the matching weapon without a `mount_damaged` event.

Example (gun mount, demo recording, t = 170 s):

```json
{"event_id": "e3383", "sequence": 3383, "timestamp": 170.0, "event_type": "mount_damaged", "ship_id": "alpha_4", "data": {"kind": "gun", "slot": "weapon_0", "index": 0, "weapon_name": "Coilgun Battery Mk3", "health_pct": 4.6, "health_lost_pct": 95.4, "destroyed": false, "cooldown_factor": 2.0, "cause": "surface_hit", "hit_location": "lateral", "source_ship_id": "beta_2", "command_ids": ["c190", "c189"], "decision_ids": ["e2361"], "ordnance_id": "proj_beta_2_0020", "target_id": null}}
```

Example (PD turret, demo recording, t = 99 s):

```json
{"event_id": "e1166", "sequence": 1166, "timestamp": 99.0, "event_type": "mount_damaged", "ship_id": "beta_7", "data": {"kind": "pd", "slot": "pd_0", "index": 0, "weapon_name": "PD Laser", "health_pct": 0.0, "health_lost_pct": 100.0, "destroyed": true, "cooldown_factor": null, "cause": "surface_hit", "hit_location": "nose", "source_ship_id": "alpha_1", "target_id": null}}
```

Example (torpedo tube, demo recording, t = 216 s):

```json
{"event_id": "e5010", "sequence": 5010, "timestamp": 216.0, "event_type": "mount_damaged", "ship_id": "alpha_6", "data": {"kind": "launcher", "slot": "weapon_0", "index": 0, "weapon_name": "Torpedo Launcher", "health_pct": 0.0, "health_lost_pct": 100.0, "destroyed": true, "cooldown_factor": null, "cause": "surface_hit", "hit_location": "lateral", "source_ship_id": "beta_1", "torpedoes_lost": 1, "torpedoes_remaining": 0, "command_ids": ["c221", "c253"], "decision_ids": ["e3761", "e4730"], "ordnance_id": "proj_beta_1_0058", "target_id": null}}
```

Example (drive, demo recording, t = 324 s):

```json
{"event_id": "e7646", "sequence": 7646, "timestamp": 324.0, "event_type": "mount_damaged", "ship_id": "alpha_1", "data": {"kind": "engine", "slot": "Main Engine Assembly", "index": 0, "weapon_name": "Main Engine Assembly", "health_pct": 61.8, "health_lost_pct": 38.2, "destroyed": false, "cooldown_factor": null, "thrust_fraction": 0.618, "cause": "surface_hit", "hit_location": "tail", "source_ship_id": "beta_4", "command_ids": ["c317", "c316"], "decision_ids": ["e7075"], "ordnance_id": "proj_beta_4_0113", "target_id": null}}
```

### `reactor_blast` - a dead ship's reactor detonation

`ship_id` = the ship that just died. Emitted once for **every**
`ship_destroyed` (all three causes: `reactor_destroyed`, `reactor_detonation`,
`reactor_overheat`), on the same `timestamp`, after the `ship_destroyed` event
and after the `blast_damage` / `mount_damaged` / `torpedo_intercepted` events
it caused. A chain detonation (a hulk whose fuse the blast burned down) gets
its own `ship_destroyed` + `reactor_blast` on the next tick.

| field | type | meaning |
|---|---|---|
| `ship_type` | string | hull class of the dead ship |
| `cause` | string | the `ship_destroyed.cause` |
| `position` | [x, y, z] m | blast centre (the ship's position at death) |
| `radius_km` | float | outer radius of any effect on ships: `15 km x sqrt(dry mass / 3790 t)` - corvette/frigate 10.6, destroyer 13.0, cruiser/battlecruiser/torpedo cruiser 15.0, battleship 18.4, dreadnought 21.2 |
| `torpedo_kill_radius_km` | float | torpedoes inside are destroyed (`0.25 x radius_km`) |
| `torpedo_blind_radius_km` | float | torpedoes inside lose their seeker (`0.60 x radius_km`) |
| `yield_gj` | float | nominal yield (0.15 GJ per tonne of dry mass); for sizing the flash only |
| `ships_hit` | [ship_id] | ships that received a `blast_damage` event (may be empty) |
| `torpedoes_destroyed` | [torpedo_id] | each also has a `torpedo_intercepted` event (below) |
| `torpedoes_blinded` | [torpedo_id] | seeker dead from this tick on; the torpedo keeps flying ballistically. With `--trace`, its frame entry shows `disabled: true` from the next frame. No separate event |

Example (demo recording, t = 150 s):

```json
{"event_id": "e2646", "sequence": 2646, "timestamp": 150.0, "event_type": "reactor_blast", "ship_id": "beta_5", "data": {"ship_type": "destroyer", "cause": "reactor_destroyed", "position": [130698.3, 1727.9, 1251.1], "radius_km": 12.99, "torpedo_kill_radius_km": 3.25, "torpedo_blind_radius_km": 7.79, "yield_gj": 426.4, "ships_hit": ["beta_6"], "torpedoes_destroyed": [], "torpedoes_blinded": ["torp_beta_6_0040"], "target_id": null}}
```

### `blast_damage` - one ship caught in a reactor blast

`ship_id` = the victim (friend or foe of the dead ship). One event per ship
inside `radius_km` with intensity >= 0.01, emitted just before the
`reactor_blast` summary. Intensity is `(1 - distance / radius)^2`.

| field | type | meaning |
|---|---|---|
| `source_ship_id` | string | the ship whose reactor detonated |
| `distance_km` | float | victim's distance from the blast centre |
| `intensity` | float 0-1 | blast intensity at the victim |
| `hit_location` | string | armour facing toward the blast: `"nose"`, `"lateral"`, `"tail"` (absent for a dying hulk) |
| `armor_ablation_cm` | float | armour burned off that facing (`intensity x 15 cm`, capped at what was left) |
| `armor_remaining_cm` | float | thickness left on that facing |
| `heat_gj` | float | heat dumped into the victim's sink (`intensity x 120 GJ`, capped so the sink never passes 149 % of its rating: a blast alone never causes a reactor meltdown; 0 when the victim was already that hot) |
| `heat` | float | victim's heat fraction after the pulse (same scale as frame `heat`) |
| `radiators` | [object] | one entry per **extended** panel, all of which are damaged: `radiator_position` (`tail_port`/`tail_starboard`/`tail_dorsal`/`tail_ventral`), `health_pct`, `destroyed`, `capacity_lost_kw`. Empty when the panels were retracted (shielded) |
| `fuse_cut_s` | float | only when the victim is a dying hulk: seconds the blast took off its detonation fuse (`intensity x 20 s`). Such an event carries only `source_ship_id`, `distance_km`, `intensity`, `fuse_cut_s`, `detonation_in_s` |
| `detonation_in_s` | float | dying hulk only: fuse left after the cut (0 = it detonates next tick) |

Blast armour loss has **no** `armor_damage` event: a consumer that keeps an
armour ledger from events must also apply `blast_damage.armor_remaining_cm`
to the `hit_location` facing (the frames' `armor` already include it).

A blast never damages hull modules, so `hull` in the frames does not change.
(An `engine` fitting hit does lower `hull` a little: the engine is one of the
modules that percentage averages.)

Example (demo recording, t = 150 s):

```json
{"event_id": "e2645", "sequence": 2645, "timestamp": 150.0, "event_type": "blast_damage", "ship_id": "beta_6", "data": {"source_ship_id": "beta_5", "distance_km": 3.56, "intensity": 0.527, "hit_location": "lateral", "armor_ablation_cm": 7.9, "armor_remaining_cm": 28.5, "heat": 0.369, "heat_gj": 63.2, "radiators": [], "target_id": null}}
```

### `torpedo_intercepted` with `cause: "reactor_blast"`

The existing event type (the viewer already removes the torpedo on it), now
actually emitted by the engine: a torpedo inside `torpedo_kill_radius_km` of a
detonation is destroyed. `ship_id` = the ship that launched the torpedo,
`data.target_id` = the torpedo's target.

| field | type | meaning |
|---|---|---|
| `torpedo_id` | string | the destroyed round |
| `cause` | string | `"reactor_blast"` (the engine emits no other cause today) |
| `blast_ship_id` | string | the ship whose reactor detonated |
| `distance_km` | float | torpedo's distance from the blast centre |
| `total_heat_absorbed_j` | float | PD heat it had already absorbed |

Example (demo recording, t = 150 s):

```json
{"event_id": "e2647", "sequence": 2647, "timestamp": 150.0, "event_type": "torpedo_intercepted", "ship_id": "beta_8", "data": {"torpedo_id": "torp_beta_8_0031", "cause": "reactor_blast", "blast_ship_id": "alpha_7", "distance_km": 2.41, "total_heat_absorbed_j": 10220965.174477061, "command_ids": ["c133"], "decision_ids": ["e683"], "target_id": "alpha_6"}}
```

### `magazine_cookoff` - a destroyed magazine burned off

`ship_id` = the ship. Emitted when a penetrating hit destroys a module of type
magazine (`Main Magazine`, `Forward Magazine`, `Magazine Forward`,
`Missile Magazine`; corvettes and frigates have no magazine module). It
follows that module's `module_destroyed` on the same `timestamp`; the
`module_damaged` / `module_destroyed` events of the secondary explosion (each
with `cause: "magazine_cookoff"`) come between the two.

| field | type | meaning |
|---|---|---|
| `module_name` | string | the magazine module |
| `torpedoes_lost` | int | torpedoes burned: half (rounded up) of each tube's remaining rounds |
| `torpedoes_remaining` | int | rounds left on the ship |
| `gun_rounds_lost` | int | coilgun rounds burned: half of each gun's remaining ammunition |
| `secondary_gj` | float | secondary explosion energy: 0.75 GJ per torpedo lost, capped at 6 GJ; 1.5 GJ for a gun magazine |
| `modules_damaged` | [module_name] | neighbouring compartments the secondary damaged (same layer, then one layer fore/aft; never bridge or reactor) |
| `modules_destroyed` | [module_name] | subset of the above that were destroyed |
| `source_ship_id` | string or null | shooter whose hit destroyed the magazine |

Example (demo recording, t = 100 s):

```json
{"event_id": "e1209", "sequence": 1209, "timestamp": 100.0, "event_type": "magazine_cookoff", "ship_id": "beta_7", "data": {"module_name": "Main Magazine", "torpedoes_lost": 0, "torpedoes_remaining": 0, "gun_rounds_lost": 1125, "secondary_gj": 1.5, "modules_damaged": ["Dorsal Turret Mount"], "modules_destroyed": [], "source_ship_id": "alpha_2", "target_id": null}}
```

### New fields on existing events

`module_damaged` (existing): two optional fields.

| field | type | meaning |
|---|---|---|
| `health_pct` | float 0-100 | module health after this event (previously the event only carried `damage_gj` and `destroyed`) |
| `cause` | string | only present for secondary damage: `"magazine_cookoff"` |

`module_destroyed` (existing): optional `cause` (`"magazine_cookoff"` or
`"surface_hit"`); absent for ordinary penetration damage.

Example (demo recording, t = 100 s):

```json
{"event_id": "e1208", "sequence": 1208, "timestamp": 100.0, "event_type": "module_damaged", "ship_id": "beta_7", "data": {"module_name": "Dorsal Turret Mount", "damage_gj": 1.5, "destroyed": false, "health_pct": 25.0, "cause": "magazine_cookoff", "target_id": null}}
```

`radiator_damaged` (existing): three new fields. The old
`dissipation_lost_kw` is the **live** rejection lost and is 0 whenever the
panel was retracted when hit (the usual case in combat), even if the panel was
destroyed.

| field | type | meaning |
|---|---|---|
| `capacity_lost_kw` | float | rated rejection this panel can never give again (panel rating x health lost); nonzero for every hit |
| `health_pct` | float 0-100 | the panel's health after the hit |
| `array_health_pct` | float 0-100 | surviving rejection of the whole 4-panel array as a percentage of its design rating |

Example (demo recording, t = 169 s):

```json
{"event_id": "e3330", "sequence": 3330, "timestamp": 169.0, "event_type": "radiator_damaged", "ship_id": "beta_2", "data": {"radiator_position": "tail_port", "damage_gj": 0.648, "destroyed": false, "dissipation_lost_kw": 0.0, "capacity_lost_kw": 25272.0, "health_pct": 87.0, "array_health_pct": 96.8, "target_id": null}}
```

### Unchanged on purpose

- `sim_trace` frames carry no new fields. Thrust loss shows in the existing
  `thrust` vs `requested_thrust`, heat pulses in `heat`, armour loss in
  `armor`, blinded torpedoes in the torpedo entry's `disabled`.
- `ship_destroyed`, `ship_dying`, `pd_torpedo_disabled`, `pd_torpedo_destroyed`
  are unchanged; PD events are never emitted for a blast's work.

### Demo recording

`visualizer/public/recordings/p5_damage_demo.json` - a seeded heuristic fleet
battle with `--trace` (no LLM), 420 s long. Not in the repository; regenerate
it (fully offline, no API key, about half a minute) and copy it into place:

```bash
uv run python scripts/run_draft_battle.py --auto-draft --no-admirals --trace --seed 4 --budget 200 --time-limit 420 --doctrine legacy --recording-dir /tmp/p5_demo
cp "$(ls -t /tmp/p5_demo/battle_*[0-9].json | head -1)" visualizer/public/recordings/p5_damage_demo.json
```

The rosters, deaths and counts below were taken from the first version of
this lane. Later engine changes (the review fixes: fuse-expiry blast origin,
blast heat cap, the combat-ineffective battle end) can move timestamps and
counts in a regenerated file; the event shapes are unchanged.

Alpha: alpha_1 cruiser_torpedo, alpha_2 corvette, alpha_3 corvette, alpha_4 frigate, alpha_5 cruiser_torpedo, alpha_6 corvette, alpha_7 frigate.
Beta: beta_1 battleship, beta_2 destroyer, beta_3 corvette, beta_4 battleship, beta_5 destroyer, beta_6 corvette, beta_7 destroyer, beta_8 corvette.
Ship deaths (each followed by a `reactor_blast`): 109 s beta_7 (reactor_detonation); 150 s beta_5 (reactor_destroyed); 150 s alpha_7 (reactor_destroyed); 157 s beta_6 (reactor_destroyed); 179 s alpha_4 (reactor_destroyed); 194 s beta_2 (reactor_destroyed); 258 s alpha_6 (reactor_detonation).

| event | count | when (first occurrences) |
|---|---|---|
| `mount_damaged` gun | 2 | 170 s alpha_4 weapon_0 -> 5%; 174 s alpha_4 weapon_0 -> 0% (destroyed) |
| `mount_damaged` pd | 11 | 99 s beta_7 pd_0 -> 0% (destroyed); 151 s beta_8 pd_0 -> 0% (destroyed); 176 s alpha_4 pd_1 -> 0% (destroyed); 176 s alpha_4 pd_0 -> 0% (destroyed); 179 s beta_2 pd_1 -> 92%; 217 s alpha_6 pd_0 -> 0% (destroyed) |
| `mount_damaged` launcher | 5 | 216 s alpha_6 weapon_0 -> 0% (destroyed); 249 s alpha_3 weapon_0 -> 91%; 279 s alpha_5 weapon_1 -> 39%; 295 s alpha_5 weapon_3 -> 62%; 341 s alpha_1 weapon_3 -> 20% |
| `mount_damaged` engine | 2 | 324 s alpha_1 thrust -> 62%; 325 s alpha_1 thrust -> 53% |
| `mount_damaged` with `cause: "reactor_blast"` | 1 | 179 s beta_2 pd pd_1 |
| `reactor_blast` | 7 | 109 s beta_7; 150 s beta_5 (ships 1, torpedoes 0 destroyed / 1 blinded); 150 s alpha_7 (ships 0, torpedoes 1 destroyed / 1 blinded); 157 s beta_6 (ships 0, torpedoes 1 destroyed / 0 blinded); 179 s alpha_4 (ships 1, torpedoes 0 destroyed / 0 blinded); 194 s beta_2 (ships 0, torpedoes 2 destroyed / 0 blinded); 258 s alpha_6 |
| `blast_damage` | 2 | 150 s beta_6 at 3.6 km from beta_5; 179 s beta_2 at 7.3 km from alpha_4 |
| `torpedo_intercepted` (`cause: "reactor_blast"`) | 4 | 150 s torp_beta_8_0031; 157 s torp_beta_6_0040; 194 s torp_alpha_5_0037; 194 s torp_alpha_2_0042 |
| `magazine_cookoff` | 2 | 100 s beta_7 (0 torpedoes, 1125 gun rounds); 150 s beta_5 (0 torpedoes, 1125 gun rounds) |
| `radiator_damaged` (new fields) | 5 | 169 s beta_2 tail_port; 176 s alpha_4 tail_port; 187 s beta_2 tail_port; 277 s alpha_5 tail_starboard; 295 s alpha_5 tail_starboard |
| `module_damaged` with `health_pct` | 128 | first at 100 s beta_7 |

### Variants not in the demo recording

The demo recording does not contain these four shapes. Shapes 1 and 2 did not
occur in any of the 20 seeded 200-point heuristic battles run for this lane;
shape 3 occurs in other seeds (seed 11 at the same settings) and shape 4 was
observed in earlier tuning runs. The JSON below is real engine output through
the recorder, but from a **scripted setup** (ships placed and damaged by hand,
as `tests/test_battle_damage.py` does), not from a recording file; the
envelope is shortened to the fields that matter.

1. `blast_damage` on a **dying hulk** - a different, shorter field set (no
   armour, heat or radiator fields):

```json
{"timestamp": 120.0, "event_type": "blast_damage", "ship_id": "alpha_2", "data": {"source_ship_id": "alpha_1", "distance_km": 3.0, "intensity": 0.591, "fuse_cut_s": 11.8, "detonation_in_s": 10.2, "target_id": null}}
```

2. `blast_damage` with a non-empty `radiators` list (the victim had its
   panels extended):

```json
{"timestamp": 120.0, "event_type": "blast_damage", "ship_id": "beta_1", "data": {"source_ship_id": "alpha_1", "distance_km": 4.0, "intensity": 0.479, "hit_location": "lateral", "armor_ablation_cm": 7.19, "armor_remaining_cm": 34.11, "heat": 0.18, "heat_gj": 57.5, "radiators": [{"radiator_position": "tail_dorsal", "health_pct": 42.5, "destroyed": false, "capacity_lost_kw": 112088.8}, {"radiator_position": "tail_port", "health_pct": 42.5, "destroyed": false, "capacity_lost_kw": 112088.8}, {"radiator_position": "tail_starboard", "health_pct": 42.5, "destroyed": false, "capacity_lost_kw": 112088.8}, {"radiator_position": "tail_ventral", "health_pct": 42.5, "destroyed": false, "capacity_lost_kw": 112088.8}], "target_id": null}}
```

3. A **destroyed drive**: `mount_damaged` (`kind: "engine"`, `destroyed: true`,
   `thrust_fraction: 0`) immediately followed by a `module_destroyed` with
   `cause: "surface_hit"`. (The demo has two engine hits on alpha_1 that leave
   it at 62% and then 53% thrust.)

```json
{"timestamp": 130.0, "event_type": "mount_damaged", "ship_id": "beta_1", "data": {"kind": "engine", "slot": "Main Engine Assembly", "index": 0, "weapon_name": "Main Engine Assembly", "health_pct": 0.0, "health_lost_pct": 100.0, "destroyed": true, "cooldown_factor": null, "thrust_fraction": 0.0, "cause": "surface_hit", "hit_location": "tail", "source_ship_id": "alpha_3", "target_id": null}}
{"timestamp": 130.0, "event_type": "module_destroyed", "ship_id": "beta_1", "data": {"module_name": "Main Engine Assembly", "cause": "surface_hit", "target_id": null}}
```

4. `magazine_cookoff` on a **torpedo hull** (`torpedoes_lost` > 0; the two
   cook-offs in the demo are gun magazines):

```json
{"timestamp": 130.0, "event_type": "magazine_cookoff", "ship_id": "beta_2", "data": {"module_name": "Main Magazine", "torpedoes_lost": 24, "torpedoes_remaining": 24, "gun_rounds_lost": 0, "secondary_gj": 6.0, "modules_damaged": ["Forward Hull Section", "Primary Sensor Array", "Targeting Sensors", "Bridge Armored Bulkhead Port", "Bridge Armored Bulkhead Starboard"], "modules_destroyed": ["Forward Hull Section", "Primary Sensor Array"], "source_ship_id": "alpha_3", "target_id": null}}
```

## Hunt lane

### `torpedo_miss.reason`

- **Event:** `torpedo_miss` (existing event type).
- **New field:** `reason`, a string, present only when set. Its one value so far is
  `"left_battle"`.
- **When it is emitted:** the engine culls a torpedo once it is more than 10,000 km from
  its target or has flown for more than an hour. Before pass 5 the cull removed the round
  without any event. A long battle could lose a tenth of its torpedoes this way: in seed
  8, budget 200, 900 s, 9 of 78 rounds disappeared, all of them seeker-killed and coasting
  away. A culled round now records a `torpedo_miss` with `reason: "left_battle"`. Its
  `closest_approach_km` (km) is the closest the round came to its target, and its
  `hit_probability` is 0.
- **Consumers:** `launched = impact + miss + intercepted + in_flight_at_end` now balances
  per side. Viewers that ignore the field are unaffected.

## Plume lane (`src/drive_plume.py`)

A ship that produces thrust projects a narrow exhaust cone astern (exactly
opposite its nose vector; the engine never deflects the nozzle). Any other
ship inside the cone - friend or foe - takes heat into its sink, its crew
takes a radiation dose, and where the wash is strong it loses armour on the
facing it shows the jet and has the fittings on that facing burned. No RNG;
the wash never kills by itself (the heat stops at 149 % of the sink, just
under the meltdown line).

Demo recording: `visualizer/public/recordings/p5_plume_demo.json` (not
committed; regenerate with the command under "Demo recording" below).

Magnitudes are named constants in `src/drive_plume.py`; the numbers quoted
here are the values at the time of writing. On the exhaust axis of a torch at
full throttle (all hulls mount the same 58.56 MN / 300 TW drive; every column
is proportional to the throttle actually produced):

| distance astern | heat into the victim | armour loss on the washed facing | crew dose, unshielded | behind a destroyer's bow (151 cm) | behind a destroyer's flank (26 cm) |
|---|---|---|---|---|---|
| 2 km | 69 GJ/s | 1.95 cm/s | 2.07 Sv/s | 0.46 Sv/s | 1.60 Sv/s |
| 5 km | 40 GJ/s | 1.08 cm/s | 1.20 Sv/s | 0.26 Sv/s | 0.93 Sv/s |
| 10 km | 16 GJ/s | 0.36 cm/s | 0.48 Sv/s | 0.11 Sv/s | 0.37 Sv/s |
| 20 km | 4.7 GJ/s | 0.02 cm/s | 0.14 Sv/s | 0.031 Sv/s | 0.11 Sv/s |
| 30 km | 2.2 GJ/s | 0 | 0.065 Sv/s | 0.014 Sv/s | 0.050 Sv/s |
| 50 km | 0.79 GJ/s | 0 | 0.024 Sv/s | 0.005 Sv/s | 0.018 Sv/s |
| 100 km | 0.20 GJ/s | 0 | 0.006 Sv/s | 0.001 Sv/s | 0.005 Sv/s |

Nothing at all beyond 100 km. Off the axis every column falls as
`exp(-(angle / 3 deg)^2)` and is exactly zero beyond 7.5 deg. For drawing the
cone: half-angle 3 deg (core) / 7.5 deg (edge), length 100 km, brightness
proportional to the emitter's `thrust` frame field. For making the washed
hull glow: `hit_location` is the facing, `peak_heat_rate_gj_s` its intensity.

### `plume_wash` - a ship is in another ship's drive exhaust

`ship_id` = the washed ship. One *episode* per (emitter, victim) pair, from
the tick the victim enters the jet to the tick it is no longer washed:

- `phase: "enter"` on the first washed tick (values of that tick);
- `phase: "sustain"` every 5 s of battle time while the episode lasts;
- `phase: "exit"` on the first tick without wash (the victim left the cone,
  the emitter stopped burning, or one of them died), carrying the totals of
  the whole episode. A battle that ends mid-episode has no `exit`; the last
  `sustain` then holds the running totals.

A one-tick crossing therefore always gives an `enter` and, one tick later, an
`exit` (same facing, the peak rate of the crossing). A turning emitter sweeps
its jet: within a tick the thrust axis is sampled every 1.5 deg between the
previous and the present nose direction, so a fast sweep cannot step over a
ship between two ticks; the victim takes the mean over the samples (its dwell
share) and the episode is flagged `swept`.

Emitted after the tick's `reactor_blast` events. A ship in two exhausts at
once has two concurrent episodes. A victim that is a dying hulk is tracked
(events are emitted) but takes nothing: its `heat_gj` / `armor_ablation_cm` /
`dose_sv` stay 0, `hit_location` is `null` and the crew fields are absent.

| field | type | meaning |
|---|---|---|
| `source_ship_id` | string | the ship whose exhaust it is (may be a dying hulk whose torch still sputters) |
| `phase` | string | `"enter"`, `"sustain"`, `"exit"` (or `"torpedo_blinded"`, see below) |
| `friendly` | bool | emitter and victim are of the same faction |
| `distance_km` | float | emitter-to-victim distance on the last washed tick |
| `off_axis_deg` | float | victim's angle off the exhaust axis on the last washed tick (0 = dead astern, at most 7.5; for a swept tick the closest the axis came) |
| `heat_rate_gj_s` | float | heat rate of the wash on the last washed tick (mean over the tick); `0.0` on `exit` |
| `peak_heat_rate_gj_s` | float | highest instantaneous rate of the episode so far |
| `emitter_throttle` | float 0-1 | thrust fraction the emitter produced on the last washed tick (engine damage and dry tanks included) |
| `duration_s` | float | length of the episode so far (washed ticks x tick length) |
| `heat_gj` | float | heat added to the victim's sink over the episode so far (capped: the wash never takes a sink past 149 %) |
| `armor_ablation_cm` | float | armour burned off over the episode so far |
| `armor_remaining_cm` | float | thickness left on the washed facing (absent for a hulk) |
| `hit_location` | string or null | facing the victim shows the jet: `"nose"`, `"lateral"`, `"tail"` |
| `fittings_hit` | [string] | fittings damaged by this episode so far, as `"<kind>:<slot>"` (`"gun:weapon_0"`, `"pd:pd_1"`, `"launcher:weapon_2"`); each damage step also has its own `mount_damaged` event with `cause: "plume_wash"` |
| `swept` | bool | the jet swept across the ship on at least one tick of the episode (the emitter was turning) |
| `radiators` | [object] | extended radiator panels the wash has damaged in this episode so far, the same shape as `blast_damage.radiators`: `radiator_position`, `health_pct`, `destroyed`, `capacity_lost_kw` (cumulative over the episode). Empty when the panels were retracted or the wash was weaker than 0.78 GJ/s |
| `dose_sv` | float | crew radiation dose received in this episode so far (Sv) |
| `crew_dose_sv` | float | the victim crew's total dose in the battle so far (permanent; absent for a hulk) |
| `crew_state` | string | `"fit"`, `"impaired"` or `"incapacitated"` (absent for a hulk) |
| `heat` | float | victim's heat fraction after the tick (absent for a hulk) |

Fittings: only the part of the heat rate above 4 GJ/s (inside ~22 km of a
full burn) damages them, 0.5 % health per GJ, applied in steps of at least
25 % (so one `mount_damaged` per fitting every few seconds, not one per tick)
to every working gun mount, torpedo tube and PD turret of the ship whatever
the facing - the jet is far wider than a hull; only the spinal mount is
facing-gated (reached when the victim shows its nose). `hit_location` on
these `mount_damaged` events is the washed facing.

Example (entry, demo recording, t = 30 s):

```json
{"event_id": "e69", "sequence": 69, "timestamp": 30.0, "event_type": "plume_wash", "ship_id": "alpha_2", "data": {"source_ship_id": "alpha_1", "phase": "enter", "friendly": true, "distance_km": 10.99, "off_axis_deg": 0.77, "heat_rate_gj_s": 12.83, "peak_heat_rate_gj_s": 12.83, "emitter_throttle": 1.0, "duration_s": 1.0, "heat_gj": 12.8, "armor_ablation_cm": 0.26, "fittings_hit": [], "hit_location": "nose", "swept": false, "dose_sv": 0.085, "crew_dose_sv": 0.085, "crew_state": "fit", "armor_remaining_cm": 150.94, "heat": 0.043, "target_id": null}}
```

Example (sustained, demo recording, t = 55 s):

```json
{"event_id": "e98", "sequence": 98, "timestamp": 55.0, "event_type": "plume_wash", "ship_id": "alpha_2", "data": {"source_ship_id": "alpha_1", "phase": "sustain", "friendly": true, "distance_km": 7.57, "off_axis_deg": 1.94, "heat_rate_gj_s": 16.01, "peak_heat_rate_gj_s": 16.42, "emitter_throttle": 1.0, "duration_s": 26.0, "heat_gj": 385.4, "armor_ablation_cm": 8.44, "fittings_hit": ["gun:weapon_0", "gun:weapon_1", "pd:pd_0", "pd:pd_1"], "hit_location": "nose", "swept": false, "dose_sv": 2.662, "crew_dose_sv": 2.662, "crew_state": "impaired", "armor_remaining_cm": 142.76, "heat": 1.256, "target_id": null}}
```

Example (exit with totals, demo recording, t = 69 s):

```json
{"event_id": "e180", "sequence": 180, "timestamp": 69.0, "event_type": "plume_wash", "ship_id": "alpha_2", "data": {"source_ship_id": "alpha_1", "phase": "exit", "friendly": true, "distance_km": 3.69, "off_axis_deg": 6.72, "heat_rate_gj_s": 0.0, "peak_heat_rate_gj_s": 16.42, "emitter_throttle": 0.7, "duration_s": 39.0, "heat_gj": 453.3, "armor_ablation_cm": 9.99, "fittings_hit": ["gun:weapon_0", "gun:weapon_1", "pd:pd_0", "pd:pd_1"], "hit_location": "nose", "swept": false, "dose_sv": 3.326, "radiators": [{"radiator_position": "tail_dorsal", "health_pct": 0.0, "destroyed": true, "capacity_lost_kw": 195000.0}, {"radiator_position": "tail_port", "health_pct": 0.0, "destroyed": true, "capacity_lost_kw": 195000.0}, {"radiator_position": "tail_starboard", "health_pct": 0.0, "destroyed": true, "capacity_lost_kw": 195000.0}, {"radiator_position": "tail_ventral", "health_pct": 0.0, "destroyed": true, "capacity_lost_kw": 195000.0}], "crew_dose_sv": 3.326, "crew_state": "impaired", "armor_remaining_cm": 141.21, "heat": 1.5, "target_id": null}}
```

(The entry and sustained examples above were taken before the `radiators`
field existed; they now also carry `"radiators": []`. The `swept` example
below is from the previous version of the demo recording, in which the
destroyer survived; the current one has no swept episode.)

Example (a weak episode that ended with the emitter jinking, `swept`, demo
recording, t = 154 s):

```json
{"event_id": "e1598", "sequence": 1598, "timestamp": 154.0, "event_type": "plume_wash", "ship_id": "alpha_1", "data": {"source_ship_id": "alpha_2", "phase": "exit", "friendly": true, "distance_km": 34.34, "off_axis_deg": 4.35, "heat_rate_gj_s": 0.0, "peak_heat_rate_gj_s": 0.6, "emitter_throttle": 0.4, "duration_s": 57.0, "heat_gj": 25.5, "armor_ablation_cm": 0.0, "fittings_hit": [], "hit_location": "nose", "swept": true, "dose_sv": 0.061, "crew_dose_sv": 0.061, "crew_state": "fit", "armor_remaining_cm": 252.84, "heat": 0.384, "target_id": null}}
```

Example (`mount_damaged` caused by the wash, demo recording, t = 35 s; the
same mount is destroyed by the fourth step at t = 50 s):

```json
{"event_id": "e70", "sequence": 70, "timestamp": 35.0, "event_type": "mount_damaged", "ship_id": "alpha_2", "data": {"kind": "gun", "slot": "weapon_0", "index": 0, "weapon_name": "Spinal Coiler Mk3", "health_pct": 72.1, "health_lost_pct": 27.9, "destroyed": false, "cooldown_factor": 1.39, "cause": "plume_wash", "hit_location": "nose", "source_ship_id": "alpha_1", "target_id": null}}
```

### Radiators in the jet: `radiator_damaged` with `cause: "plume_wash"`

Extended radiator panels stand in the jet. While the wash is above 0.78 GJ/s
(what the 780 MW array could reject itself) every EXTENDED panel takes
0.7 GJ of damage per GJ of wash above that rate, through the reactor blast's
mechanism (`thermal.apply_radiator_damage`; a panel has 5 GJ of health): gone
in one tick 10 km astern of a full burn, in two at 20 km, untouched at 50 km.
Retracted panels are shielded by the hull. The existing `radiator_damaged`
event is emitted with its usual fields plus `cause: "plume_wash"` and
`source_ship_id` (the emitter) each time a panel loses another quarter of its
health and when it is destroyed (not once per tick of a weak wash); the
panels are also listed in the episode's `plume_wash.radiators`. Existing
`radiator_damaged` events from gun hits carry no `cause`.

While a ship is washed above that rate, the heat ladder's forced extension at
100 % heat is suppressed and the `auto` radiator mode does not move the
panels (state of the previous tick's wash); a commander's explicit
`set_radiators` extend is obeyed, and the panels are then shredded. Once the
ship is clear of the jet the ladder forces them out as before.

Example (demo recording, t = 60 s: the legacy captain extends its radiators
at the checkpoint, inside the jet; all four panels go on that tick):

```json
{"event_id": "e171", "sequence": 171, "timestamp": 60.0, "event_type": "radiator_damaged", "ship_id": "alpha_2", "data": {"radiator_position": "tail_dorsal", "damage_gj": 5.0, "destroyed": true, "dissipation_lost_kw": 195000.0, "capacity_lost_kw": 195000.0, "health_pct": 0.0, "array_health_pct": 75.0, "cause": "plume_wash", "source_ship_id": "alpha_1", "target_id": null}}
```

### `crew_radiation` - a crew's plume radiation dose crossed a threshold

`ship_id` = the irradiated ship. The dose accumulates while the ship is
washed (dose rate = wash heat rate x 0.03 Sv/GJ x `exp(-armour_cm / 100)`,
with the armour left on the washed facing; the full wash rate counts, also
when the sink is already at its ceiling) and never decreases. Emitted once
per ship per threshold, on the tick it is crossed, just before that tick's
`plume_wash` event of the episode (if one is due):

- `state: "impaired"` at 2 Sv: from now on every gun and torpedo tube of the
  ship needs `cooldown_factor` (1.5) times its normal cooldown, on top of any
  mount damage. PD is unaffected.
- `state: "incapacitated"` at 8 Sv: the ship stops fighting and maneuvering
  for the rest of the battle. It coasts (frames show `thrust: 0`, maneuver
  none), fires nothing and launches nothing; automatic PD and radiators keep
  working; it stays targetable and counts as unable to deal damage for the
  victory rule and as combat-ineffective for the time-limit score.

| field | type | meaning |
|---|---|---|
| `state` | string | `"impaired"` or `"incapacitated"` |
| `dose_sv` | float | the crew's total dose after the tick |
| `threshold_sv` | float | the threshold crossed (2.0 / 8.0) |
| `cooldown_factor` | float or null | `impaired`: weapon cooldown multiplier (1.5); `null` for `incapacitated` |
| `source_ship_id` | string | the ship whose exhaust delivered the dose that crossed the line |

Example (demo recording, t = 50 s):

```json
{"event_id": "e92", "sequence": 92, "timestamp": 50.0, "event_type": "crew_radiation", "ship_id": "alpha_2", "data": {"state": "impaired", "dose_sv": 2.08, "threshold_sv": 2.0, "cooldown_factor": 1.5, "source_ship_id": "alpha_1", "target_id": null}}
```

The demo recording has no `incapacitated` event (same shape, `state:
"incapacitated"`, `threshold_sv: 8.0`, `cooldown_factor: null`).

### `drive_interlock` - the engine held a ship's throttle for a friendly in its jet

`ship_id` = the ship whose main drive is being limited. Every ship has the
interlock, whoever commands it (LLM, MCP, rule-based; also while evading):
each tick its throttle is capped so that its exhaust puts at most 0.5 GJ/s on
any friendly ship (same faction; a surrendered friendly counts, a dying hulk
does not; enemies never limit it). The caps of all ships come from one
start-of-tick snapshot. Frames show it as `thrust` below `requested_thrust`.

- `phase: "enter"` on the first tick the cap is below the throttle the
  maneuver wants;
- `phase: "exit"` on the first tick it no longer is (the friendly cleared the
  jet, or the ship stopped wanting to burn). A battle that ends while a ship
  is held has no `exit`.

| field | type | meaning |
|---|---|---|
| `phase` | string | `"enter"` or `"exit"` |
| `friendly_ship_id` | string | the friendly the cap protects (the most limiting one; on `exit` the last one) |
| `throttle_cap` | float 0-1 | `enter`: the throttle allowed (0 = drive held off) |
| `ordered_throttle` | float 0-1 | `enter`: the throttle the maneuver wanted |
| `distance_km` | float | `enter`: distance to that friendly |
| `off_axis_deg` | float | `enter`: its angle off the exhaust axis |
| `evading` | bool | `enter`: the ship was in a threat-evasion mode (EVADE / RUN / PRESENT / TANK / BRACE) |
| `mode` | string | `enter`: the ship's execution mode at that moment |
| `duration_s` | float | `exit`: how long the throttle was held |
| `min_throttle_cap` | float | `exit`: the lowest cap of the episode |
| `evading_s` | float | `exit`: seconds of it spent in a threat-evasion mode |

Example (demo `p5_plume_interlock_demo.json`, t = 30 s and 60 s):

```json
{"event_id": "e64", "sequence": 64, "timestamp": 30.0, "event_type": "drive_interlock", "ship_id": "alpha_1", "data": {"phase": "enter", "friendly_ship_id": "alpha_2", "throttle_cap": 0.037, "ordered_throttle": 1.0, "distance_km": 11.0, "off_axis_deg": 1.04, "evading": false, "mode": "COAST", "target_id": null}}
{"event_id": "e147", "sequence": 147, "timestamp": 60.0, "event_type": "drive_interlock", "ship_id": "alpha_1", "data": {"phase": "exit", "friendly_ship_id": "alpha_2", "duration_s": 30.0, "min_throttle_cap": 0.0, "evading_s": 0.0, "target_id": null}}
```

Demo: `visualizer/public/recordings/p5_plume_interlock_demo.json` (not
committed) - the column battle of the plume demo below with the interlock on:

```bash
uv run python scripts/generate_test_battle.py drive_interlock --recording-dir /tmp/p5_interlock --out visualizer/public/recordings/p5_plume_interlock_demo.json
```

- t = 30-60 s: battleship `alpha_1` is held at 4 % -> 0 % of its ordered
  100 % while destroyer `alpha_2` is 11 km astern in its jet;
- t = 69-161 s: `alpha_2`, now ahead, is held at 7-32 % of its ordered 70 %
  while `alpha_1` is astern of it (again briefly at 162-166 s);
- friendly `plume_wash` never exceeds 0.49 GJ/s, the largest crew dose is
  0.07 Sv, no `mount_damaged`, no `crew_radiation`.

### Fleet-net datalink messages (`comms_message`, `sender_role: "engine"`)

Two new `kind`s on the FLEET NET, recorded like the existing datalink
traffic (`LAUNCH`, `KILL`, `LOSS`, `PD_LOSS`, `TARGET`):

- `CREW_STATE` - a crew crossed a radiation threshold since the last
  checkpoint. To the ship's own fleet (all but the ship itself): "TIS B CREW
  IMPAIRED by plume radiation (2.4 Sv): its weapons cycle x1.5 slower" /
  "TIS B CREW INCAPACITATED by plume radiation (8.8 Sv): it cannot fight,
  maneuver or execute orders" (`structured`: `subject_id`, `crew_state`,
  `dose_sv`, `source_ship_id`). To the other fleet, as seen from outside:
  "TIS B took heavy plume wash from TIS A: crew impaired, slower fire" /
  "TIS B has stopped maneuvering and firing after plume wash from TIS A: crew
  incapacitated" (`structured.observed: true`, no dose).
- `PLUME_WASH` - a friendly took a wash that peaked above 4 GJ/s since the
  last checkpoint: "TIS B is taking heavy plume wash in TIS A's exhaust (11 s,
  peak +32.8 GJ/s, 361 GJ, crew dose 8.8 Sv)" ("took" once the episode has
  ended). Own fleet only.

A `captain_report` for a ship whose crew is incapacitated is filed by the
runner without a captain call: `status: "UNABLE"`, `implicit: true`, and its
`compliance` carries `executable: false`, `followed: false` and the leading
deviation `"not executable: crew incapacitated by plume radiation"`.

### Torpedoes in the plume (switch OFF in pass 5)

`drive_plume.PLUME_AFFECTS_TORPEDOES` is `False`: recordings made with the
default engine contain none of the following. With the switch on, a torpedo
flying in a burning ship's exhaust (never its own launching ship's)
accumulates a dose (the ship-equivalent heat it flew through):

- at 2 GJ its seeker dies: `plume_wash` with `phase: "torpedo_blinded"`,
  `ship_id` = the emitter, `data.target_id` = the torpedo's target, and
  `torpedo_id`, `torpedo_source_ship_id`, `distance_km`, `off_axis_deg`,
  `dose_gj`. With `--trace` the torpedo's frame entry shows `disabled: true`
  from the next frame;
- at 10 GJ it is destroyed: `torpedo_intercepted` with `cause: "plume_wash"`,
  `plume_ship_id` (the emitter), `distance_km`, `off_axis_deg` (same shape as
  the reactor-blast variant above).

### Demo recording

`visualizer/public/recordings/p5_plume_demo.json` - a seeded battle from
fixed start positions with per-tick frames and predicted paths (no LLM),
180 s long, legacy heuristic captains (they know nothing of the plume).
**The drive interlock is switched off for this recording** (the generator's
default for this scenario), otherwise the battleship's throttle would be held
and nothing below would happen; see `drive_interlock` above for the same
battle with it on.
Regenerate it (offline, no API key, a few seconds):

```bash
uv run python scripts/generate_test_battle.py plume_wash --recording-dir /tmp/p5_plume --out visualizer/public/recordings/p5_plume_demo.json
```

Alpha: battleship `alpha_1`, destroyer `alpha_2` 11 km astern of it and
0.2 km off its exhaust axis, cruiser `alpha_3` 14 km abeam (never washed).
Beta: a destroyer and a cruiser 400 km away.

- t = 30-69 s: `alpha_2` flies up `alpha_1`'s exhaust at twice its
  acceleration (11 km -> 3.7 km, 13-16 GJ/s): `enter` at 30 s, `sustain`
  every 5 s, `exit` at 69 s with 453 GJ of heat, 10 cm of nose armour and
  3.3 Sv. Its heat climbs from 4 % past the wash's 149 % ceiling
  (`heat_warning` 0.9 at 50 s, 1.0 at 51 s, 1.25 at 56 s); from 51 s its
  radiators are NOT forced out (it is in the jet).
- `mount_damaged` (`cause: "plume_wash"`) on its spinal mount, turret and
  both PD turrets at 35, 40, 45 s (72 %, 47 %, 20 % health) and at 50 s
  (all four destroyed).
- t = 50 s: `crew_radiation`, `alpha_2` impaired (2.08 Sv).
- t = 60 s: its captain extends the radiators at the checkpoint;
  `radiator_damaged` x4 (`cause: "plume_wash"`), all panels destroyed.
- t = 70 s: one second after leaving the jet, at 149 % with no radiator and
  its drive still burning, `alpha_2` melts down (`ship_destroyed`,
  `reactor_overheat`, followed by its `reactor_blast`). The earlier version
  of this recording, in which it survived and later washed `alpha_1`, no
  longer exists.

## Surrender and end reasons (`src/llm/battle_runner.py`)

### `surrender` - a ship struck its colours

- **Event:** `surrender` (the existing 1v1 event type, now also recorded in fleet
  battles). Before, a fleet battle showed a surrender only as the `surrender` tool
  call inside the captain's `captain_decision`; viewers should prefer this event and
  fall back to the tool call for older recordings.
- **When it is emitted:** once per ship, at the checkpoint where the surrender takes
  effect (`timestamp` = that checkpoint, the same as the `captain_decision` holding
  the tool call). Sources: a captain's `surrender` tool call (fleet and 1v1), and an
  MCP commander's `surrender_fleet`, which records one event per ship still afloat
  (destroyed and dying ships are not surrenders and get no event). Several ships
  surrendering at one checkpoint follow each other in fleet order.
- **`ship_id`:** the ship that surrendered.

| field | type | meaning |
|---|---|---|
| `captain_name` | string | the captain (or, for an MCP fleet surrender, the MCP commander) |
| `faction` | string | `"alpha"` / `"beta"` |
| `reason` | string or null | the captain's stated reason; `null` when none was given (the LLM `surrender` tool takes no argument, MCP `surrender_fleet` neither; heuristic captains give one) |
| `hull_pct` | float 0-100 | hull integrity when the ship surrendered |
| `could_deal_damage` | bool | the ship still had an operational gun with ammunition or a torpedo tube with rounds (`battle_damage.ship_can_deal_damage`, read before the surrender) |

A surrendered ship drifts, is untargetable and is out of the fight for every victory
rule; it is not destroyed and gets no `ship_destroyed`.

Example, from `uv run python scripts/run_draft_battle.py --auto-draft --no-admirals
--seed 4 --budget 200 --time-limit 600` (tactical heuristic captains: a disarmed ship
with no armed friend left surrenders; this battle ends at 150 s):

```json
{"event_id": "e3415", "sequence": 3415, "timestamp": 150.0, "event_type": "surrender", "ship_id": "beta_3",
 "data": {"captain_name": "Captain Dart-1", "faction": "beta",
          "reason": "all weapons out, no armed ship left to protect",
          "hull_pct": 100.0, "could_deal_damage": false}}
```

(beta_6 and beta_8 follow at the same timestamp with the same fields.)

### `result_reason` - how a fleet battle ended

Header field `result_reason` (also `GET /live/recording` `live.result.reason`).
Winner and outcome logic are unchanged; only the text now says what happened.
`<Side>` is `Alpha` / `Beta`. A ship is counted *destroyed* when it was destroyed or
was a dying hulk, otherwise *surrendered* when it surrendered.

| reason | when |
|---|---|
| `<Side> fleet eliminated` | every ship of that fleet was destroyed |
| `<Side> fleet surrendered (N ships struck their colours)` | no ship of that fleet was destroyed; all N surrendered (`1 ship struck its colours` for one) |
| `<Side> fleet destroyed or surrendered (D destroyed, S surrendered)` | a mix: D destroyed, the remaining S surrendered |
| `Mutual destruction/surrender: <alpha reason>; <beta reason>` | neither fleet has a ship left in the fight (draw); each half is one of the three above |
| `<Side> fleet combat-ineffective (no weapon left that can deal damage)` | that fleet could deal damage at the start and no longer can (no operational gun with ammunition, no tube with rounds, nothing of its own in flight); the other side still can |
| `Both fleets combat-ineffective - on points: <tactical reason>` | neither side can deal damage any more; decided on points |

Unchanged: `Draw accepted - ...`, `<Side> tactical advantage (...)`, `Too close to
call (...)`, and the 1v1 reasons. The seed-4 battle above ends `winner: "alpha"`,
`result_reason: "Beta fleet destroyed or surrendered (5 destroyed, 3 surrendered)"`
(before: `"Beta fleet eliminated"` with three beta corvettes afloat at 100 % hull).

## File format

New recordings are written gzip-compressed by default (lane `gzip`):
`battle_<alpha>_vs_<beta>_<date>.json.gz`. The draft sidecar stays plain:
`battle_<...>.draft.json` (named from the stem, not `<recording>.draft.json`).

- **Inside the gzip:** the same recording object as before, as compact JSON
  (`separators=(',', ':')`, no indent; `BattleRecording.to_compact_json()`). It
  parses to exactly the object the plain `to_json()` text parses to. No field,
  event or unit changed.
- **Deterministic:** the gzip header has mtime 0, no FNAME/FCOMMENT/FEXTRA, and
  the level is fixed (9): the same battle gives the same bytes on a given zlib
  build (`tests/test_recording_io.py`).
- **Plain JSON** (`--plain-json` on `run_draft_battle.py`, `run_llm_battle.py`,
  `mcp_battle.py`, `generate_test_battle.py`; `BattleConfig.compress_recording =
  False`) writes the indented `to_json()` text as before, byte for byte.
- **Readers** decide by the first two bytes (`1f 8b` = gzip), not by the name:
  Python `src/recording_io.load_recording`, viewer `RecordingCodec.js`. A dev
  server may send `.json.gz` with `Content-Encoding: gzip` (Vite does: the browser
  hands the viewer inflated text) or as raw bytes (the viewer inflates them with
  `DecompressionStream('gzip')`); both load.
- **Sizes** (committed recordings, plain indented -> gzip): 37.8 MB -> 1.0 MB,
  15.8 MB -> 0.64 MB, 13.7 MB -> 0.68 MB; all of `data/recordings/` 263 MB ->
  17 MB (`scripts/compress_recordings.py data/recordings --dry-run`). The ratio
  varies: one LLM recording with 10 MB of replay-audit `assets` (model prompts
  and responses) went 31.8 MB -> 8.0 MB.
- **`GET /live/recording`** is unchanged (an HTTP JSON payload, not a file).

## Arcs lane (`src/weapon_arcs.py`)

Weapon arcs and turret traverse: a battery or a PD laser fires only in a direction its
own hull leaves open, and only once the mount has slewed onto it. The masks come from
`data/hull_mounts.json` (generated from the viewer's hull kit by
`visualizer/scripts/export_hull_mounts.mjs`; keyed by hull class, never by faction, ship
or seed). Arcs decide WHETHER and WHEN a round or a beam leaves; hit rolls, lead and
evasion are untouched. Simulation flags `weapon_arcs` and `roll_to_unmask` (both default
on); a recording made with arcs off carries none of the fields below, and every consumer
must fall back (the viewer keeps its own aiming and its own roll).

### Body frame

x = forward (nose), z = up (dorsal), y = z cross x (port). Azimuth = atan2(y, x), 0 at the
nose, positive to port, (-180, 180]; elevation = asin(z), positive dorsal.

The engine's `kinematic_state.up` is a NAVIGATION vector (evasion picks jink directions
from it); the hull's real attitude is that frame rolled about `fwd` by the ship's roll
angle. The trace records the result, so a consumer never needs the roll angle itself.

### Constants a consumer needs

| what | constant (`src/weapon_arcs.py`) | value |
|---|---|---|
| PD laser traverse | `PD_TRAVERSE_DEG_S` | 120 deg/s |
| light battery traverse | `LIGHT_BATTERY_TRAVERSE_DEG_S` | 45 deg/s |
| coilgun battery traverse | `BATTERY_TRAVERSE_DEG_S` | 30 deg/s |
| heavy battery traverse | `HEAVY_BATTERY_TRAVERSE_DEG_S` | 15 deg/s |
| on-target tolerance | `AIM_TOLERANCE_DEG` | 0.5 deg |
| roll authority | `ROLL_RCS_AUTHORITY_FACTOR` | 3.3 x the hull's RCS pitch acceleration and rate cap |
| roll re-plan lattice | `ROLL_REPLAN_INTERVAL_S` | every 5 s of battle time |
| roll goal commit | `ROLL_GOAL_COMMIT_S` | a goal stands >= 15 s; a gun-only change also waits out the standing goal's planned roll time |
| roll PD hysteresis | `ROLL_PD_HYSTERESIS_SAMPLES` | one whole turret-on-torpedo pair (3 samples); an attitude earns nothing against a torpedo that impacts before the hull could roll there |
| masked-report delay | `MOUNT_MASKED_REPORT_S` | 3 s |

Why these rates: a PD director must hand over between the rounds of one salvo (a few
degrees: free) and swing onto a second salvo from another quarter inside a tick or two;
at 120 deg/s and the 1 s engine tick any retarget within 120 degrees costs no dwell and
the far side costs one tick (5 % of the ~20 s a round spends in the 250 km envelope at
the 12 km/s closure floor), so the pinned per-turret PD figures hold for a turret that
bears. A battery (30 deg/s, 90 degrees in 3 s) loses at most one round of its 10 s salvo
cadence to a 90 degree retarget; a heavy battery (15 deg/s, 90 degrees in 6 s) gives up
most of a round and still out-turns every hull that carries it. Spinals have no traverse:
the gimbal follows the lead inside `pivot_range_deg` and the hull is the traverse.

Slew is along the great circle in the body frame. Per tick a mount moves up to
rate x dt and fires in the tick in which it arrives. A mount whose target is masked
creeps to the last open direction on that great circle (half-cell steps) and waits.

Two things are NOT rate limited, on purpose:
- A mount's FIRST designation is free: the first time a mount is given a target it is
  simply on it (if the direction is open). In draft battles the captains name their first
  gun targets at the 30 s checkpoint, 400-500 km out and minutes before the first round, so
  a recording shows the turrets snap from their stowed pose (along the mount axis) onto
  the enemy in that one frame. Every later change of target or attitude is slewed.
- `CombatSimulation.settle_arcs` (run once inside the first step, and callable by scripted
  scenarios): hulls that ALREADY have gun targets at t = 0 start rolled for them. Draft
  battles have no targets at t = 0, so there it does nothing: every hull starts unrolled
  (`up` = +Z) and all roll seen in a recording is flown at the RCS roll rate.

Roll: re-planned every 5 s (inbound torpedoes first, then gun energy, candidates scored at
the angle and 7.5 degrees to either side so a hull settles with margin), flown with a
rate profile that changes by at most the roll acceleration per tick. `up` is NOT the roll
angle alone: it also moves whenever the hull turns its nose (the body frame turns with
`fwd`), so a flip or a hard turn swings `up` much faster than the roll rate.

### Trace frame fields (per ship, `--trace` recordings)

| field | type | meaning |
|---|---|---|
| `up` | [x, y, z], 4 decimals | hull dorsal axis in world coordinates, roll included; unit, perpendicular to `fwd`. A dying hulk's `up` tumbles with it. |
| `tur` | list of `[az_deg, el_deg, state]`, 0.1 degree | one entry per mount in `data/hull_mounts.json` order for the hull class (launchers included: always state 0, along their axis). Aim direction in the body frame above. `state`: 0 idle / stowed (no target), 1 slewing, 2 on target and clear, 3 target masked by the own hull (spinal: outside its gimbal). |

**Timing of `tur` against events (one frame offset, by design).** The trace is recorded
after each engine step: the frame stamped `T` holds positions, attitude and `tur` at the
END of the tick that ran from `T - dt` to `T`, while that tick's events (`shot_fired`,
`pd_fired`, `mount_masked`, ...) are stamped `T - dt`. So the mount state that gated an
event stamped `t` is in the frame stamped `t + dt` (in the 2026-11-15 demo every
`shot_fired`/`pd_fired` has state 2 there; the frame stamped `t` shows the state BEFORE
the tick and can be 1 or 3 for a mount that arrived during it). That is what an
interpolating viewer wants: `visualizer/src/Interpolator.js` blends `tur` between the
frames around the replay time (`render/HullRigs.js` poses from the blend), so during an
event's tick the turret visibly finishes its slew onto the target - the engine fires a
mount in the tick in which it arrives. The engine therefore keeps this stamping; a
consumer that wants "the state that allowed this event" reads the next frame.

Real frame (alpha_1 at t = 100.0 s in the demo; mounts in `hull_mounts.json` order):

```json
{"t": 100.0, "ships": {"alpha_1": {"fwd": [1.0, 0.0006, 0.0], "up": [0.0003, -0.5773, 0.8165],
   "tur": [[0.5, -0.4, 2], [0.7, 0.0, 3], [0.5, -0.4, 2], [0.4, 15.0, 3], [0.4, -5.0, 3], [90.0, 30.0, 0], [0.0, -90.0, 0], [-90.0, 30.0, 0]], "...": "the other per-ship fields"}}}
```

### `mount_masked` - a mount with a target is masked by its own hull

Emitted once when a mount has been masked for `MOUNT_MASKED_REPORT_S` with a target
(`masked: true`), and once when it clears (`masked: false`): the target moved into the
arc, the ship rolled or turned, the mount lost its target, or the ship left the fight
(destroyed, dying, surrendered, or - guns only - crew incapacitated: the exit then carries
`cause` and every mount of that ship reports `tur` state 0 from then on). `ship_id` is the ship,
`target_id` the ship it wants to shoot (PD: the torpedo / slug id is in `target_id`).

| field | type | meaning |
|---|---|---|
| `slot` | string | `weapon_N` / `pd_N`, the `hull_mounts.json` slot |
| `kind` | string | `gun`, `pd` or `spinal` |
| `masked` | bool | true on entry, false on exit |
| `reason` | string (entry only) | `hull` (own structure in the way) or `gimbal` (spinal: target outside `pivot_range_deg`) |
| `since` | float s (entry only) | battle time the mount became masked (3 s before the event) |
| `duration_s` | float s (exit only) | how long it was masked |
| `cause` | string (exit only, optional) | `destroyed`, `dying`, `surrendered` or `crew_incapacitated` when the ship leaving the fight closed the episode |

```json
{"event_id": "e120", "sequence": 120, "timestamp": 33.0, "event_type": "mount_masked", "ship_id": "alpha_1", "data": {"slot": "weapon_3", "kind": "gun", "masked": true, "reason": "hull", "since": 30.0, "target_id": "beta_4"}}
{"event_id": "e4842", "sequence": 4842, "timestamp": 285.0, "event_type": "mount_masked", "ship_id": "alpha_1", "data": {"slot": "weapon_3", "kind": "gun", "masked": false, "duration_s": 255.0, "target_id": "beta_3"}}
{"event_id": "e238", "sequence": 238, "timestamp": 85.0, "event_type": "mount_masked", "ship_id": "beta_1", "data": {"slot": "pd_0", "kind": "pd", "masked": true, "reason": "hull", "since": 82.0, "target_id": "torp_alpha_3_0001"}}
{"event_id": "e5837", "sequence": 5837, "timestamp": 333.0, "event_type": "mount_masked", "ship_id": "beta_1", "data": {"slot": "weapon_0", "kind": "spinal", "masked": true, "reason": "gimbal", "since": 330.0, "target_id": "alpha_1"}}
```

### `weapon_status` reasons

`weapon_status.reason` gains `masked` (the lead direction is behind the own hull) and
`slewing` (the mount is still swinging onto it). Like every `weapon_status` they are
emitted on transitions only.

```json
{"event_id": "e1280", "sequence": 1280, "timestamp": 165.0, "event_type": "weapon_status", "ship_id": "beta_3", "data": {"weapon_slot": "weapon_1", "reason": "masked", "command_ids": ["c19", "c101"], "decision_ids": ["e20", "e947"], "target_id": "alpha_1"}}
{"event_id": "e5778", "sequence": 5778, "timestamp": 330.0, "event_type": "weapon_status", "ship_id": "alpha_1", "data": {"weapon_slot": "weapon_2", "reason": "slewing", "command_ids": ["c184", "c201"], "decision_ids": ["e5070", "e5712"], "target_id": "beta_1"}}
```

### What did NOT change

`shot_fired` / `projectile_launched` and `pd_engaged` already name the slot / turret;
their fields are unchanged. `fire_direction` on a shot is the world direction the round
left in - by construction an open direction of that slot's mask in the frame built from
the same frame's `fwd` + `up`.

### Demo recording

`visualizer/public/recordings/p5_arcs_demo.json.gz` (gzip, read it with
`src/recording_io.load_recording`; seeded, `--trace`, heuristic captains,
no LLM):

- `scripts/run_draft_battle.py --auto-draft --no-admirals --seed 5 --trace --time-limit 420`: Draft Battle (100 pts); winner `alpha`, `Alpha tactical advantage (127.1 vs 117.9)`; 420 trace frames, 7232 events.
- Every trace frame carries `up` and `tur` for every ship. Hulls start unrolled (`up` = +Z); the first gun targets are named at the 30 s checkpoint (turrets snap from stow onto them in that frame: the free first designation) and the rolls begin there.
- Roll to unmask: `up` leaves +Z on 8 of 9 ships; first on `beta_4` at 35 s (its `up` is 15 degrees off +Z by 65 s, a pure roll: the nose has not turned). Later `up` also swings when hulls turn or flip, so it is not the roll angle alone.
- PD defence against a salvo: 8 torpedoes launched at 30-114 s; PD dwell on them (`pd_fired`, `target_type: "torpedo"`) from 82 s to 176 s, 8 seekers blinded (`pd_torpedo_disabled` at 98, 107, 118, 126, 138, 150, 163, 176 s), 4 impacts (99, 109, 119, 181 s).
- Gun fight: 241 rounds (`shot_fired`) from 124 s to 419 s; `mount_masked` 218 events (20 gun entries, 86 PD entries, 4 spinal gimbal entries); `weapon_status` `masked` x18, `slewing` x1.
- First masked battery: `alpha_1` `weapon_3` at 33 s, clear at 285 s after 255 s.
- Kills: `beta_3` at 285 s.
- (Recorded before the arcs review fixes: there a mount of a ship that died while masked got no exit event and its `tur` simply stopped changing; now the exit is emitted with `cause` and the ship's mounts go to state 0. The turrets also snapped onto their first targets at 30 s; now the engagement is settled at that checkpoint and later first designations slew.)

## Anthropic lane (`src/llm/anthropic_client.py`, `src/llm/providers.py`)

Battles may now run bare Anthropic model ids (`claude-sonnet-5-5`, ...) on the
Anthropic API directly, next to OpenRouter ids. Nothing in the simulation
changed; the additions are in the replay-audit events and the provenance. All
are optional for consumers: OpenRouter recordings and older recordings lack them.

### New fields on `model_call`

Written by `src/llm/replay_audit.AuditedClient` for every model call (as before).
On calls served by the Anthropic API:

- `provider`: `"anthropic"` (absent on OpenRouter calls).
- `stop_reason`: the API's raw `stop_reason` (`tool_use`, `end_turn`,
  `max_tokens`, `refusal`, ...). `finish_reason` keeps the OpenRouter vocabulary
  the callers use: `tool_calls`, `stop`, `length` (from `max_tokens`), `refusal`.
- `model`: the model the API says served the call (`response.model`).
- `request`: the request settings without messages/tools: `model`, `max_tokens`,
  `thinking` (`{"type": "adaptive"}`), `output_config` (`{"effort": ...}`),
  `tool_choice`, plus `provider`, `cache_ttl` (`"1h"`) and `api_version`
  (the `anthropic-version` header, e.g. `"2026-10-21"`). Never headers or keys.
- `usage`: OpenRouter-shaped (`prompt_tokens` = uncached + cache writes + cache
  reads, `completion_tokens`, `prompt_tokens_details.cached_tokens` /
  `.cache_write_tokens`, `cost` in USD computed from the price table) plus
  `usage.anthropic` with the API's own fields: `input_tokens`,
  `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`,
  `cache_creation.ephemeral_5m_input_tokens` / `.ephemeral_1h_input_tokens`,
  `thinking_tokens`.

### `model_refusal` - a model declined a call

Emitted right after the `model_call` whose `finish_reason` is `refusal`
(HTTP 200, `stop_reason: "refusal"`). The commander issues no orders from that
call and it is not re-asked: a captain loses the turn (its `captain_decision`
has `call_failed: true`, its lost-turn note says the model refused), an admiral's
checkpoint has no directive (`admiral_directive_failed`, `reason` starts
`refused by the model`). The battle goes on.

| field | type | meaning |
|---|---|---|
| `actor` | `"captain"` / `"admiral"` | who was refused |
| `faction` | `"alpha"` / `"beta"` | side |
| `model` | string | model that refused |
| `category` | string or null | `stop_details.category` (`cyber`, `bio`, `general_harms`, ...) |
| `explanation` | string or null | `stop_details.explanation` |
| `call_event_id` | string | the `model_call` event it belongs to |

`ship_id` is the captain's ship (null for an admiral).

### `provenance.llm` - who served each side

Set when the battle has an LLM client: `{"alpha": [entry, ...], "beta": [...]}`,
one entry per distinct LLM model id the side's admiral and captains use.
Anthropic entry: `provider` (`"anthropic"`), `model` (exact id), `effort`,
`reasoning_effort_setting`, `thinking`, `max_tokens`, `tool_choice`, `cache_ttl`,
`fallbacks` (always `false` in battles: a refused turn is never re-run on another
model), `sdk` (`"anthropic-python 2.8.0"`), `api_version`, `prices_as_of`.
OpenRouter entry: `provider` (`"openrouter"`), `model`, `reasoning_effort`,
`temperature`, `max_tokens`, `cache_ttl`.

### `provenance.billing` (live draft battles) gains

`refusals`, `refusals_by_side`, `computed_cost_usd` (sum of per-call `usage.cost`),
`cache` (one-line cache summary), `prompt_tokens`, `cached_tokens`,
`cache_write_tokens`. For Anthropic models `pricing[model]` is the price-table
entry (USD per million tokens, `as_of`).

## Admiral command-interface lane (`cmdadm`: admiral, fleet_intel, fleet_draft, admiral_view)

All additions are optional; recordings without them load everywhere as before.

### `draft_auto_trimmed` - an invalid draft trimmed instead of aborting

Recorded on the draft recorder (live draft battles merge it into the battle
recording) when an admiral's three `select_fleet` attempts were all invalid and its
LAST fleet was trimmed to validity (`fleet_draft.trim_selection`: unknown hulls,
then the cheapest hulls, dropped until the fleet fits). `timestamp` 0, `ship_id`
null. `data`: `faction`, `submitted` (the last `ships` argument as the model sent
it), `dropped` (hull types, in drop order), `kept` (hull types), `points`,
`budget`, `max_ships`. The draft sidecar (`draft_summary_dict`) carries the same
dict as `auto_trimmed`. `run_admiral_draft(allow_trim=False)` turns it off.

### Draw offers on the diplomacy channel

`comms_message` on channel `admiral_diplomacy` (as `message_enemy_admiral`
traffic) with `kind` `DRAW_PROPOSAL` (once per offer), `DRAW_ACCEPT` (an open
enemy offer accepted: the battle then ends as a draw) or `DRAW_REJECT` (the
offer is closed and the proposer is told at its next checkpoint). `text` is one
sentence ("Admiral X rejects the fleet draw."). An accept with no open offer is
not recorded (the acceptor is told nothing was open).

### Not recorded, prompt only

The enemy facing, the HEADINGS facing fields, the torpedo defence forecast, the
clock line, the non-compliance escalation and the (default-off,
`AdmiralConfig.show_score`) score line are turn-state text; they reach the
recording only inside `model_call` message assets.

## Command-interface runner lane (`src/llm/call_policy.py`, `src/llm/cost_report.py`)

Experiment plumbing for paid comparisons. Every mode is OFF by default; a
battle without the new flags sends byte-identical requests
(`tests/test_call_policy.py::test_default_battle_requests_are_unchanged`) and
its events differ only by the always-written `role` / `effort` fields below.
All fields are optional for consumers; older recordings lack them
(`cost_report.cost_report()` and `scripts/replay_turn.py` infer the role from
the offered tools).

### New fields on `model_call`

| field | when | meaning |
|---|---|---|
| `role` | always | `draft` (select_fleet / set_formation), `directive` (admiral Phase 1), `order` (admiral per-ship call), `captain`, `other` (personality selection, discuss answers) - `call_policy.call_role` |
| `effort` | when the request carries one | the effort actually sent (`request.output_config.effort`; OpenRouter `reasoning.effort`) |
| `effort_reason` | effort policy set | `default` (battle-wide effort), `role` (a per-role setting), `trigger:<reasons>` (captain trigger effort; reasons from `call_policy.captain_triggers`: `first`, `torpedo`, `merge`, `conflict`, `damage`, `new_order`), `explicit` (the caller's own effort, e.g. the admiral's retry ladder) |
| `thinking_summary` | `--thinking-summary` | the API's summarized thinking (`thinking.display: "summarized"`), joined text of the reply's thinking blocks. Audit only: no prompt is ever built from it |
| `cache_warmup` | `--cache-warmup` | `{usage, cost}` of the `max_tokens: 0` pre-warm this call waited for (billed on its own; `usage.cost` of the call excludes it) |

### `captain_turn_skipped` - a quiet turn (`--quiet-turns`)

The captain was not called this checkpoint: no trigger, no new FLEET NET
traffic, applied admiral order unchanged. The ship keeps flying its current
maneuver, renewed for the window as a lost turn renews it. No
`captain_decision` and no `captain_report` are written for that ship and
checkpoint.

| field | type | meaning |
|---|---|---|
| `faction` | string | side |
| `captain_name` | string | |
| `reason` | string | starts `quiet:` |
| `kept_maneuver` | string or null | e.g. `INTERCEPT@40%` |
| `renewed` | bool | false when the maneuver was applied this checkpoint (an admiral heading) and flies on unrenewed |

### `battle_stopped` in every runner

`LLMBattleRunner.check_budget()` now stops fleet (sync and MCP) and duel
battles too when the spend cap refuses a request: `battle_stopped`
(`reason` starts `Spending ceiling reached`), result DRAW with no winner and
that reason as `result_reason`; the partial recording is saved as usual. Live
draft battles keep their own stop (raise, `record_stop`).

### `admiral_order` text in intent-only mode (`--intent-only`)

Orders of an intent-only admiral are made by the runner, not by a model call:
`order_text` `[INTENT-ONLY] No per-ship order: ...`, no heading or maneuver,
`torpedo_salvo` / `torpedo_target` only when the directive call ordered a salvo
for that ship.

### `status.json` (live draft) and `--status-file` (run_llm_battle)

`cost_report`: `{"rows": [{side, role, calls, input, cache_read, cache_write,
output, thinking, usd, thinking_calls, thinking_share}], "by_side": {...},
"total": {...}}`; role `warmup` rows are the cache pre-warms. Live draft
`status.json` also gains `role_models` (`{side: {admiral, captain}}`),
`effort_policy` and `experiment` (the mode flags).

## Command-interface review (`cmdrev`)

### `provenance.label` and `provenance.config.label`

`--label TEXT` on `scripts/run_live_draft_battle.py` and `scripts/run_llm_battle.py`
(`BattleConfig.label`): a free-text note naming the experiment the run belongs to.
Stored as `provenance.label` (absent without `--label`), in `provenance.config`, in
`status.json` / `--status-file` (`label`) and on both spend-ledger lines (`label`). Never
part of a prompt.

### `battle_stopped` / `result_reason` after Ctrl-C (live draft)

One SIGINT now closes the request budget (no new model call), lets the calls in flight
settle, records `battle_stopped` with `reason` `Interrupted (Ctrl-C): ...` and closes the
spend-ledger line with `status: "interrupted"`. The ledger counts a closed run at
`max(usd, committed_usd)`.

### `captain_turn_skipped`: kills and losses count as new traffic

A quiet turn is skipped only when no datalink KILL or LOSS line reached the captain this
checkpoint either (launches, target switches and other datalink lines still do not count).

## Evade-burn lane (`evadeburn`: the evade overlay on a burning order)

Owner's rule (src/simulation.py module docstring): an evade overlay riding on
an order that burns keeps the burn the whole time and evades by swinging the
thrust axis inside `EVADE_BURN_CONE_DEG` (20 deg) of the burn line. Nothing
new is emitted; two existing records gain a field. Recordings without them
read as before (no swing).

### `execution_state.swing`

| field | type | meaning |
|---|---|---|
| `swing` | bool, only when true | the evade overlay has the helm (`mode` EVADE / TANK / BRACE / PRESENT; EVADE also when only an armed enemy within 400 km keeps it swinging) and is swinging the order's burn - the drive lit and the aim off the line - instead of holding it: `applied_throttle` is the throttle the order burns this tick (x engine fraction; `requested_throttle` for a fixed heading, the range keeper's own demand for HOLD_RANGE / standoff), the swing is aimed inside the cone of the burn line (the nose can overshoot it by a few degrees on carried spin; see `off_line`). Absent: the older overlay (an engine-off order or the EVADE maneuver - BRACE drive off, TANK half), or no evasion. |

| `off_line` | bool, only when true (with `swing`) | the nose is still more than the cone plus 10 deg off the burn line (the order's first turn under fire, or turning back after a RUN): the drive burns at the order's throttle while thrust vectoring turns the ship (not along the line yet), however far off the line the nose points (a range keeper notes its correction scaled by cos(angle off)) |

The transition key now includes both, so a change from a held BRACE to a
swung BRACE at the same throttles is a new `execution_state` event.

### Trace frame field `swing` (per ship, `--trace` recordings)

`swing: 1` next to `evade` on frames where the overlay swings a burn (omitted
otherwise). The viewer (`render/Rcs.js` `jinkOf`) shows a swung BRACE as an
armour swing (`JINK_FACE`) instead of a ship holding still.

### `shot_fired.fc.swing`

The fire-control forecast's input for a target swinging a burn: its sideways
acceleration `burn x sin(cone)` (m/s^2), present only when > 0
(`TargetMotion.swing_ms2`, priced by `firecontrol.SWING_*_FRACTION`).

## Retarget lane: torpedo launch profiles and the datalink (`src/torpedo.py`, `src/torpedo_link.py`)

### `torpedo_launched.profile`

| field | type | meaning |
|---|---|---|
| `profile` | string, only when not the default | the round's launch profile, `"reserve"`. Absent = `"sprint"` (the default, and every recording made before profiles existed), so a sprint launch is byte-identical to before. |

`sprint` builds closure early up to the 12 km/s cruise floor
(`MIN_CLOSING_SPEED_KPS`); `reserve` cruises at `RESERVE_CRUISE_CLOSING_KPS`
(8 km/s) and keeps the rest of its delta-v until `RESERVE_TERMINAL_WINDOW_S`
(20 s) before impact. Trace frames carry the round's `dv_remaining` as before,
so a viewer can show the held tank.

### `torpedo_retargeted` from the datalink: `reason`, `group`, `group_size`, `profile`

Every reassignment by the datalink is a `torpedo_retargeted` event with the
existing fields (`torpedo_id`, `old_target_id`, `new_target_id`,
`distance_km`, `time_to_intercept_s`, `dv_remaining_kps`, `selection_mode`,
`seeker_heat_fraction`) plus:

| field | type | meaning |
|---|---|---|
| `reason` | `"target_lost"` / `"target_surrendered"` / `"datalink_surplus"` | `target_lost`: the round's target was destroyed or became a dying hulk and the side's rounds were reassigned as a group; `target_surrendered`: the same for a target that struck its colours (owner decision: a surrender releases its inbound rounds like a hulk; rounds nothing else is reachable for keep homing on it); `datalink_surplus`: a `reserve`-profile round diverted BEFORE its target died because more rounds were inbound on it than it needs and another hull still needed rounds |
| `group` | string | the decision it was part of: `"<faction>@<sim time s>:<reason>"`; every round reassigned in the same tick for the same reason shares it |
| `group_size` | int | rounds reassigned in that group |
| `profile` | string | the round's launch profile (`sprint` / `reserve`) |

Events without `reason` are the pre-link per-round rule (recordings before
pass 5, or the measurement switch `AIC_TORPEDO_DATALINK=0`). The viewer reads
only `ship_id`, `torpedo_id` and `new_target_id` (HudIndex "RETARGET" line);
the new fields are optional for every consumer.

### Rules the events record

- **The board and what a target needs** (`torpedo_link.need_for` / `kill_count`,
  `CombatSimulation.torpedo_board`): a target's need is PD soak + rounds to kill,
  sized on the rounds' FORECAST impact (closure now + the delta-v they will still
  spend, less 0.5 km/s kept; within ~1 % of the captain's flown shot from a launcher
  at rest or closing 3 km/s), plus `LINK_KILL_MARGIN_ROUNDS` = 1 (from flown strikes,
  `scripts/torpedo_link_margin.py`). The captain's OVERKILL line and the admiral's
  TORPEDO ECONOMY quote the same board (`tests/test_torpedo_datalink.py`
  `TestOneNeedEstimate` pins that the three agree at the same instant).
- **Group reassignment** (always on): every tick, per side, the live-seeker
  rounds (not blinded, not dry) whose target is destroyed, dying or surrendered are planned
  together, most constrained round first (fewest reachable hulls, then torpedo
  id), each to the reachable hull with a positive deficit - its need minus the
  rounds already inbound that ARRIVE BEFORE this one (forecast arrival) minus the
  rounds assigned to it this tick - with the most delta-v left (a PD-singed seeker:
  the fastest intercept); with no deficit anywhere, the least over-subscribed hull.
  Reachable: `can_intercept` within 600 s, the steer flown as a finite burn
  (impossible above `a * t_go / 2`, cost `a t (1 - sqrt(1 - 2 dv / (a t)))`) and at
  least `min(1.12, 0.25 + 0.02 * t)` km/s left after it. Rounds with nothing reachable
  keep homing on a dying hulk, or coast when the target is gone. Ties go to the
  smaller ship id; the outcome does not depend on ship or round insertion order and
  is mirror symmetric.
- **Surplus diversion** (`reserve` rounds only, every 5 s per side): on a hull with
  more live rounds inbound than it needs + 1, the later arrivals beyond that,
  not inside 12 s of impact (`LINK_COMMITTED_S`), go to hulls with a positive
  deficit they can reach. Sprint rounds are never diverted.
- After a reassignment the round's cruise floor is the closure it already
  has toward the new hull plus whatever delta-v it holds beyond the
  reassignment reserve (never above its profile's floor).
- A reserve round's terminal window is 20 s or the time its held tank takes to
  burn at its present acceleration, whichever is longer.

### FLEET NET: `comms_message` kind `RETASK`

The engine datalink (sender `FLEET DATALINK`, channel fleet net) tells the
launching side at the next checkpoint, one line per (reason, old target, new
target): "5 rounds re-tasked OCS Halberd-1 -> OCS Halberd-3 (target lost), first
impact ~T+141". Structured: `reason`, `old_target_id`, `new_target_id`, `rounds`,
`first_impact_t` (s, battle time: reassignment time + time to intercept).

### `captain_report.from_reason` (lane cmdcap, round 2)

Optional, `true` only. The report was not a `report_to_admiral` call: the
captain gave a `reason` on an action call (set_maneuver / set_heading /
set_primary_target / launch_torpedo). Such a report is filed as an implicit
ACK with the reasons in `intent`, and becomes a DEVIATE (implicit false, the
reasons in `reason`) only when the engine's compliance finds a deviation in
what a reasoned call did (maneuver / target / salvo). Recordings before this
change filed every reason-only report as an explicit ACK. The viewer's
command net shows a `CALL REASON` chip on such rows.

### Schedule: checkpoint 1 at T+0 (`first_checkpoint_s`)

Since the T+0 schedule (owner decision, pass 5) checkpoint 1 is at T+0 and
checkpoint k at (k-1) x `decision_interval_s`; every checkpoint's orders fly a
full interval, the last one too, so N checkpoints end at T+N x interval (20 x
30 s: T+600, was T+630 in live draft battles and T+600 with unflown last orders
in the others).

| field | where | meaning |
|---|---|---|
| `first_checkpoint_s` | recording header | `0.0`; absent on older recordings, whose first checkpoint is one interval in (T+30) |
| `first_checkpoint_t` | live `/live/recording` block (and `scripts/fake_live_server.py`) | the same for spectators; absent from older servers |

Consumers: `scripts/replay_turn.py` numbers checkpoints from it, the viewer
counts orders at t=0 as checkpoint 1 (the engine's own `decision_point_reached`
at t=0 is still not a checkpoint) and numbers the live NEXT CHECKPOINT marker
from `first_checkpoint_t`. A trace frame is recorded at T+0 before the first
orders (its plans are re-predicted once they apply, as at every checkpoint).

## Launch-sequence lane (`src/torpedo_launch.py`)

A torpedo no longer lights its torch at its ship's centre on the launch tick.
It leaves its own rack round / VLS cell, drifts clear on its manoeuvring
thrusters, turns onto its first guidance heading and only then ignites. Old
recordings (no fields below) are rounds whose torch burned from the launch
tick at the ship's centre; every field here is optional and recording only
(nothing reads it back into the game).

### The sequence (what a viewer animates)

| phase | duration | the round |
|---|---|---|
| `eject` | first `LAUNCH_EJECT_S` = 1.0 s | slides off its rail / out of its cell: starts at the round's position on the hull (`data/hull_mounts.json` launcher `rounds[k].pos`, body frame) with the ship's velocity + the hull's spin at that point (omega x r) + `LAUNCH_EJECT_SPEED_MPS` = 10 m/s along `rounds[k].eject`; nose = `rounds[k].nose` turned with the ship (roll included); keeps the ship's spin; no thruster fires |
| `separate` | until ignition | lateral RCS (the round's own 5 % of main thrust, ~0.6 g) pushes it off the line through its ship along its heading (or brakes it away from a hull it is closing on), and it turns its nose onto the first guidance heading at up to 30 deg/s (30 deg/s^2) |
| boost | rest of the flight | the torch lights at the START of the first tick on which (start-of-tick ship snapshot): the round is 50 m from its own hull's surface (capsule of `ShipGeometry`), 50 m from every friendly hull within 2 km, its exhaust cone (15 deg half angle, to 5 km) misses its own hull, and its nose is within 10 deg of the heading. After 8 s it lights once it is 50 m off its own hull whatever else holds; at 15 s regardless. A round whose ship is destroyed has no hull to clear. On that first boost tick the torch first cancels the sideways drift the sequence gave the round (velocity off its launcher's, square to the line of sight; ~0.2 s of thrust) |

Seconds from launch to ignition, measured (`scripts/launch_sequence_facts.py`,
target 500 km, PD off):

| launcher / geometry | corvette (rail rounds, nose forward) | torpedo cruiser (VLS cells, nose radial) |
|---|---|---|
| coasting, target ahead | 4 | 4 |
| coasting, target abeam | 4 | 4 (the cell facing away: 7) |
| coasting, target astern | 7 (turns about) | 4 |
| full burn toward the target | 3 (the ship pulls away) | 4 |
| full burn away from the target | 8 (turns about; 8 s timeout) | 4 |
| mid-turn / spinning at launch | 4 | 4-5 |
| two hulls 120 m apart launching together | 4 and 8 (friendly clearance, timeout) | 4 and 9 |
| 50 km shot | 4 | 4 |

Nothing in these cases puts a round inside a hull or in its ship's drive
plume (closest approach to its own hull after the first tick 11.7 m, after
ignition 52.8 m; time inside the 7.5 deg plume cone while the drive burns: 0 s),
the four rounds of a cruiser salvo stay > 50 m apart, so the engine has no
collision model for launches.

### `torpedo_launched`: new fields

| field | type | meaning |
|---|---|---|
| `mount` | string or null | the launcher's slot in `data/hull_mounts.json` (`weapon_0` ...); null when the hull class has no mount data |
| `round` | int | which of that mount's `rounds` left (index into `hull_mounts.json` `rounds`, i.e. `Hulls.js` `magazineRound` call order). The engine does NOT fire them in index order: of the rounds still on the tube it takes the one nearest the target's side of the hull (body position along the line of sight), then the one farthest from the nearest friendly hull, then the lowest index - geometry, so a fleet and its mirror image (rotated or reflected) fire mirror-image rounds. A corvette with its target off the port bow empties its port rail first; dead ahead with no friend near, index order (port and starboard alternating) |
| `eject_pos_m` | [x, y, z] m, body frame | the round's start point (x forward, y port, z dorsal; = `rounds[round].pos`) |
| `eject_dir` | [x, y, z] unit, body frame | direction it is ejected (= `rounds[round].eject`: rack outboard, cell out of the hatch) |
| `nose` | [x, y, z] unit, body frame | its nose at rest (rack: forward along the rail; cell: out of the tube) |
| `eject_speed_mps` | float | ejection speed relative to the hull point (10) |
| `profile` | string | (now actually recorded: it was dropped between the engine event and the recording before) `"reserve"` when not sprint |

A launch with the sequence switched off (`CombatSimulation.launch_sequence =
False`, `AIC_LAUNCH_SEQUENCE=0`, measurement only) has none of the sequence fields.

Viewer note: `HullRigs.assignLaunches` deals a ship's launches to its magazine
banks round robin by launch count and empties each bank in index order; the
engine picks the first ready tube and the round by geometry (above), so the
round the viewer removes can differ from the one that left. `mount` (bank =
launcher index in `hull_mounts` order) and `round` give the exact rack round /
cell; `rounds[round].pos` is where the animation starts. Without these fields
(old recordings) keep the old dealing.

### `torpedo_ignition` (new event)

Emitted on the tick the torch lights. `ship_id` = launcher, `target_id` = the
round's target.

| field | type | meaning |
|---|---|---|
| `torpedo_id` | string | the round |
| `delay_s` | float s | launch to ignition |
| `reason` | `clear` / `free` / `timeout` / `hard_timeout` | all rules held / its ship was destroyed (no hull to clear) / the 8 s timeout (clear of its own hull, not aligned or a friendly too near) / the 15 s cap |
| `own_gap_m` | float m or null | distance from its own hull's surface at ignition (null: ship destroyed) |
| `nose` | [x, y, z] unit, world | nose at ignition |
| `heading` | [x, y, z] unit, world | the first guidance heading it turned onto |

### Trace frame fields (per torpedo, `--trace` recordings)

| field | when | meaning |
|---|---|---|
| `ph` | only while the torch is cold | `"eject"` or `"separate"`; absent = burning (or an old recording) |
| `nose` | with `ph` | world unit nose direction (the round has an attitude only in its sequence; once burning its nose is the thrust direction `fwd`) |
| `rcs` | while the RCS fires (unchanged field) | lateral RCS thrust, unit direction x throttle: the separation push / brake (throttle 1) |
| `thr` | always (unchanged) | 0 during the sequence; > 0 from the ignition frame on |

The frame at time t describes the tick that ended at t: the first frame after
the launch shows `ph: "eject"` with the round already 10 m off its rack (one
tick of drift); the ignition frame has no `ph` and `thr` > 0. Interpolate the
round between frames from `pos`/`vel`; slerp `nose` between sequence frames and
hand over to `fwd` at ignition.

### Rules the engine keeps around the sequence

- Nobody treats a round still in its sequence as an inbound threat: point
  defence does not engage it, the target's evasion, roll-to-unmask, radiator
  automation, evade status and the rule-based captain's threat check ignore it
  (they see it from its ignition tick).
- Forecasts include the delay: the captain's shadow flight (`fly_torpedo`, its
  TOF and the PROFILES line) flies the sequence; the admiral's 1-D pursuit
  (`fleet_intel.solve_pursuit(hold_s=)`, `torpedo_solution`) coasts for the
  forecast hold (`torpedo_launch.predict_launch_hold_s`: the per-tick sequence
  from a coasting ship, within one tick of the flown ignition); the datalink
  board (`torpedo_link.round_forecast`), `Torpedo.can_intercept` (powered
  branch) and the trajectory-prediction ghosts add the hold left
  (`Torpedo.launch_hold_s()`).
- A round in its sequence whose target is lost is reassigned by the datalink
  like any other and continues its sequence; an orphaned or blinded round in
  its sequence coasts and never strikes its own launcher.
- The tube's reload cycle starts at ejection (unchanged); a tube wrecked after
  the round left does not take it.
