# Pass 4: command, communication and fire control

Pass 4 (October 25, 2026) set out to make the admirals and captains decide better, by fixing what
they are told, how they talk to each other and how honest the fire-control numbers are, and to make
a live or replayed battle something a spectator can follow. This document records what the audits
found, what changed, and what three measured battles show.

**One caveat up front:** each code state was measured with a single DeepSeek V4.1 Flash mirror
battle. The fleets were drafted by the admiral models, so the rosters and the battle geometry
differ between runs. The numbers below are evidence, not proof. The test that matters most, a
Sonnet 5 rematch against the [September 26 baseline](../data/recordings/sonnet5_vs_dsv41flash_150pt_20260926/),
is still to be run.

## What the audits found

Seven audits covered the admiral and captain prompts, the comms path, past battles, the live viewer,
the hulls and the effects. The prompt audits rebuilt every admiral and captain prompt from recorded
battles, the Sonnet 5 vs DeepSeek battle among them, and the viewer audits drove the viewer headlessly.
Most failures came from information the models got wrong, could not use or never received. Missing
doctrine mattered less.

- **The command net did not work.** Captains answered orders 238 times in two battles, including
  correct deviation reports such as "target is SEPARATING at 2.82 km/s". None of those texts reached
  a later admiral or captain prompt, yet a quarter of the admiral orders asked captains to report
  back. Fleet broadcasts went to the enemy, labelled "ENEMY TRANSMISSION". Messages addressed by
  ship name were never delivered, and the admiral answered captains' questions without any battle
  state, so it made up explanations.
- **Admirals were misinformed.** The projectile list had no side attribution and told the admiral
  to evade its own outgoing torpedoes. A directive call that failed (provider error or token cap)
  looked like a deliberate "no new directive", and captains got the previous checkpoint's
  directive with nothing marking it as stale. The per-ship order prompt quoted the range from the
  nearest friendly ship as if it were the range from the ship being ordered: 86 km instead of 429 km.
- **Captains were misinformed.** "Incoming threats" counted every blinded torpedo in the battle,
  your own side's included, which put phantom threats into a third of the prompts. The enemy weapons
  summary dropped torpedo launchers, so torpedo corvettes read as "1x PD". Captains were never told
  their own position, so 9 of 12 `set_heading` calls in one battle passed a position as a direction.
- **One tool-call round.** `captain.decide()` never followed up. Models that answered only with
  `respond_to_orders` drifted for the whole battle: 31% of captain turns across three recordings
  issued no action at all.
- **Maneuvers could not stop.** INTERCEPT had no arrival-speed control, and 28% of admiral orders
  asked for INTERCEPT and EVADE at once. Opposing fleets passed through each other at 3.5-7 km/s in
  5 of 7 battles.
- **Fire control overstated the odds.** The P(hit) shown to commanders, and used by the
  FIRE_WHEN_OPTIMAL gate, ignored the geometric gate a round must pass first. Captains were shown
  12-19% at 435-490 km while 0 of 609 rounds beyond 200 km connected.
- **Spectators could not follow a battle.** The decisions panel locked onto its first entry. There
  was no scoreboard, kill feed or comms feed, live sim time arrived in bursts, and the combat log
  repeated events. On the hulls, the dreadnought's spinal gun fired from empty space and every class
  wore the same pyramid prow. Kinetic torpedo hits were drawn as nuclear fireballs.

## What changed

**Communication.** A faction-checked fleet channel delivers by name, by `NAME [id]` and to all
friendlies. `report_to_admiral` (the old `respond_to_orders` still works) files ACK, DEVIATE or
UNABLE with a reason and an optional contact report, and the engine attaches compliance: ordered vs
executed maneuver, target and salvo. `fleet_signal` lets captains claim targets and call for fire,
and an engine datalink posts friendly launches, kills and losses. Each captain's turn shows a FLEET
NET block, and the admiral's turn opens with CAPTAIN REPORTS. All of it is recorded as
`comms_message`, `comms_delivered` and `captain_report` events.

**Admiral intel.** A new turn state (`src/llm/fleet_intel.py`): what needs a decision NOW, captain
reports, what happened since the last checkpoint, fleet geometry, a threat table and the torpedo
economy. When a directive call fails, captains read "(no new fleet directive this checkpoint)"
instead of the previous directive, and the failure is recorded as `admiral_directive_failed`.

**Captain intel.** A new turn format with per-target fire solutions (`src/llm/fire_solutions.py`),
positions in orders turned into directions from the ship, threats counted per side, torpedo
launchers in the enemy summary, and follow-up rounds when a reply contains only an acknowledgement.
Each turn shows the order text next to the maneuver the engine actually applied, and warns when the
two disagree.

**Maneuvers.** INTERCEPT takes a standoff range and an arrival closing speed, HOLD_RANGE keeps a
set range, `set_heading` accepts `toward_point` and `toward_ship`, and evasion can overlay any
maneuver. With none of the new fields set, seeded battles are bit-identical to before.

**Fire control.** P(hit) is now conversion × track: the old roll probability times the chance that
the target is still inside the gate after the flight, given its acceleration and its dodge. Its
constants were fitted on recorded battles and 85k live-engine rounds, and it is checked per maneuver
class and 100 km band
(`tests/test_fire_control_calibration.py`). Engine hit resolution is unchanged, with one exception:
targets flying a HEADING or BURN order are now led by their drive's actual output instead of the
ordered thrust. In replays of 432 rounds at such targets 200+ km out, that raised the gate pass
rate from 32% to 40%.

**Viewer.** Live mode gained pacing, a deliberation card, a draft splash, end-of-battle and
disconnect cards, a camera director (key A) and off-screen indicators. A spectator HUD adds a
scoreboard, kill feed, comms feed (key N), a fixed combat log and timeline markers. Every hull
class got its own design, including a dreadnought with a spinal keel. Torpedo hits are now kinetic
strikes, hits stay attached to the hull, armor shows wear, and targeting lines show who is shooting
whom. RCS plumes are sized to their blocks and are suppressed where a nozzle is blocked.

## Three battles, one setup

All three runs used DeepSeek V4.1 Flash on both sides, a 150-point draft, seed 20260906, reasoning
low, admiral vision and a $1 spend cap. Every run ended at the cap.

- **Before:** `viz-pass-3` at 9400302, the code before pass 4.
- **Mid:** pass 4 after the first wave (communication, admiral and captain intel, maneuvers), before
  the fire-control and order fixes.
- **After:** the final pass-4 code, merged here.

| Rosters | Alpha | Beta |
|---|---|---|
| Before | 2 battleships, 2 battlecruisers, 2 frigates | 4 cruisers, 4 frigates |
| Mid | 2 battleships, 2 destroyers, 4 frigates | 4 battlecruisers, 2 destroyers, 2 frigates |
| After | 2 battleships, cruiser, 2 frigates, corvette | 4 battlecruisers, 2 destroyers, 2 frigates |

### Decisions and communication

| | Before | Mid | After |
|---|---|---|---|
| Checkpoints before the cap | 15 | 13 | 14 |
| Spend | $0.71 | $0.85 | $0.95 |
| Captain turns | 210 | 208 | 193 |
| Turns with no action tool | 37% | 11% | **5%** |
| Turns with no maneuver tool | 90% | 58% | 80% |
| Reports that reached the admiral | 0 (153 stored, never read) | 34 (14 ACK, 20 DEVIATE) | 23 (11 ACK, 11 DEVIATE, 1 UNABLE) |
| Fleet-net messages | none | none | 9 captain signals (8 target claims), 14 datalink |
| Captain prompt / completion, median tokens | 11,010 / 1,225 | 13,072 / 2,153 | 12,937 / 2,546 |
| Admiral prompt / completion, median tokens | 2,505 / 1,251 | 3,566 / 1,002 | 3,850 / 1,231 |

The drop in action-less turns, from 37% to 5%, is the clearest decision-level change. The captains
now act every turn. That comes from the follow-up round and from turns that state what is already
applied.

The maneuver share went the other way after the mid run, and this run does not explain why. In both
pass-4 runs every admiral order carried a structured maneuver that the engine applies directly
(222 of 223 mid, 211 of 211 after), so a captain flying its ordered maneuver does not need the tool.
A likely cause is the order work between the two runs: the turn now shows the applied maneuver, so
captains may no longer re-issue it just to be safe. That is not verified.

### Gunnery

| | Before | Mid | After |
|---|---|---|---|
| Rounds fired | 508 | 827 | 424 |
| Fired beyond 300 km | 139 (27%) | 525 (63%) | **40 (9%)** |
| Hits | 87 (17.1%) | 13 (1.6%) | 133 (31.4%) |
| Mean forecast P(hit) at fire time | 41.3% | 35.9% | **32.5%** |
| Closest approach | 123.5 km | 51.4 km | 20.5 km |
| Kills | 0 | 0 | 1 (a cruiser, at 357 s) |
| Torpedoes launched | 0 | 0 | 0 |

The overall hit rate mostly reflects geometry. The after battle closed to 20 km and fought inside
200 km, while the mid battle spent its rounds at 400+ km. Do not read 17% → 31% as pass 4
doubling accuracy.

Two results do hold up:

- **The forecast is now honest.** Before pass 4, fire control promised 41% and delivered 17%. Mid,
  it promised 36% and delivered 1.6%. After, it promised 32.5% and 31.4% landed. By range:

  | Range | Before: forecast / observed | After: forecast / observed (n) |
  |---|---|---|
  | 0-99 km | no rounds | 46.0% / 39.8% (118) |
  | 100-199 km | 55.8% / 22.6% | 36.5% / 33.2% (202) |
  | 200-299 km | 40.6% / 17.9% | 11.2% / 23.4% (64) |
  | 300-399 km | 31.6% / 11.5% | 7.6% / 13.3% (30) |
  | 400+ km | no rounds | 1.8% / 0% (10) |

  Beyond 200 km the new forecast errs on the cautious side, on small samples.
- **Captains stopped shooting at hopeless range.** Shown honest numbers, they fired 9% of their
  rounds beyond 300 km, against 63% in the mid run. They spent their heat inside 200 km, where it
  connects.

### Cost

Spend per checkpoint rose from $0.047 to $0.068 (+44%), mostly from longer completions: captains
reason more when they get more to reason about. DeepSeek reported no cache writes in the pass-4 runs
(its caching is automatic), and only 10% of prompt tokens were read from cache after. The one-hour cache TTL added in
pass 4 applies to Anthropic `cache_control` calls only.

13 of the after run's 14 failed calls were budget-cap aborts at the final checkpoint. Twice an
admiral directive call spent all 8,192 output tokens on reasoning and was retried with a
short-answer nudge.

## Known issues

- **No torpedoes.** No torpedoes were fired in any of the three runs. In the after run the admirals
  never ordered a salvo, and one order told the corvette to hold all 8 of its torpedoes.
- **Range discipline under Flash.** The fleets closed to 20 km although the admirals ordered
  HOLD_RANGE at 150-200 km and INTERCEPT standoffs of 220-250 km. More than half of their orders
  were PADLOCK, which does not hold range.
- **Dodgers under massed fire.** Ships dodging a whole fleet's fire are still harder to hit than the
  forecast says. In the flash mirror they moved 2-4x farther across the line of fire than in seeded
  runs, and nothing visible at fire time predicts it. The residual is pinned by a test and stated
  in the doctrine (`src/firecontrol.py`, JINK_ACCEL_FRACTION).
- **Viewer and fixtures.** The viewer does not yet render `admiral_directive_failed` events. The
  bow RCS nozzles on the corvette and the torpedo cruiser are blocked by their rails and launcher
  banks, so those blocks never fire. The HUD comms fixture records launch signals twice.
- **MCP.** `set_weapons_order` is accepted but does nothing on a hull without guns.

## Next

- The Sonnet 5 vs DeepSeek V4.1 Flash rematch against the September 26 baseline, the real test of
  whether a strong model decides better with the pass-4 turn state. A pass-4 checkpoint costs about
  40% more than before, so budget about $8-10 for a 15-checkpoint Sonnet battle.
- Cross-model battles on the pass-4 engine.
- Torpedo doctrine and range keeping for cheap models.
