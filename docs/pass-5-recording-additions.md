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
`visualizer/public/recordings/p5_damage_demo.json` (see "Demo recording" at
the end of this section for timestamps, and "Variants not in the demo
recording" for the four field shapes it does not contain).

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
| `cause` | string | `"surface_hit"` or `"reactor_blast"` |
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
battle with `--trace` (no LLM), 420 s long, generated with:

```bash
uv run python scripts/run_draft_battle.py --auto-draft --no-admirals --trace --seed 4 --budget 200 --time-limit 420
```

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
