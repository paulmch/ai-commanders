# Intent and consequence replays

Open a recording in the viewer and select **Decisions** (shortcut **D**).
Choose a ship and checkpoint to read its admiral context, captain orders,
execution feedback, and observed consequences. The checkpoint arrows seek and
pause playback. Click a supporting event to seek to its timestamp and focus the
involved ship. The originating decision stays selected when a round hits during
a later checkpoint. Evidence is limited to the current playhead; scrubbing back
removes later outcomes from the summary.

Share a specific view with these query parameters:

```text
?recording=/recordings/example.json&decisions=1&ship=alpha_1&decision=e123&t=180
```

Older recordings remain readable. They display existing admiral orders, captain
decisions, discussions and logs, with events from the following decision window
labelled as chronological context. Missing tool calls, execution reasons and
attribution are never reconstructed as facts.

## Recording version 3.0

Every recorded event has a stable `event_id` and increasing `sequence` in arrival
order. `timestamp` remains simulation time; several model calls and commands can
arrive at the same timestamp while the simulation is paused.

| Record | Evidence retained |
| --- | --- |
| `model_call` | Actor, faction/ship, requested and returned model, sampling settings, maximum tokens, generation ID, finish reason, usage/cost, public output/tool calls, and references to exact message/tool inputs |
| `captain_decision` | Requested tool calls, resolved command payloads, tool errors, model failure status, and parent events for admiral context/model calls |
| `command_status` | Stable command ID, originating decision, originating tool/model call where available, command payload, and issued/accepted/rejected/queued/executed/superseded/expired/cancelled status |
| `weapon_status` | Changes in hold, cooldown, empty magazine, damage, capacitor charge, target availability, arc, intercept, range/probability gate, and firing status |
| `execution_state` | Requested and applied throttle, engine and fuel limits, and actual automatic evasion mode |
| Combat events | Original engine details, projectile/torpedo identity, originating command and decision IDs through flight, interception, impact, armor/module damage and destruction |

`accepted` means the engine accepted an order. A launch event proves a round was
launched. Persistent gun policies retain their source decision after the next
checkpoint. A changed primary target can be a second dependency of a later shot.
Queued salvo rounds retain their original command across tube reloads. Ship
damage emitted during impact resolution inherits the impacting round's identity.
The execution ledger records transitions rather than emitting every wait reason
on every simulation tick.

The `assets` dictionary deduplicates exact model messages and tool schemas by
SHA-256. Provider private reasoning is excluded from replay commentary and
public-output records. The recorder contains battle prompts and public model
responses; HTTP authorization headers are never part of its input capture.
The metadata includes the simulation seed, actual decision interval, git revision,
working source hash, ship-data hash and runtime options. These identify the run;
they are not a saved simulator/RNG snapshot suitable for branching a battle.

Duels, synchronous fleets, asynchronous fleets and the HTTP MCP runner share
the ledger. MCP inputs record the submitted command payloads and an execution-time
snapshot. The battle server cannot capture the external MCP client's private
model conversation, and marks that input as unavailable.

## Live streaming

To let both models buy their own fleets and watch the battle as it runs:

```bash
uv run python scripts/run_live_draft_battle.py \
  --alpha-model z-ai/glm-5.3-flash --beta-model openai/gpt-5.6-luna \
  --points 200 --max-ships 8 --turns 40 --spend-limit 8
```

Open `http://localhost:5173/?live=1&decisions=1` with the Vite viewer running.
The spectator server uses port 8765 and stays available after the battle ends.
The admirals independently choose fleet composition and formation; every captain
uses its faction's model. Draft calls and their public outputs are recorded too.
An invalid model draft stops the launch instead of substituting an automatic fleet.
Defaults are high reasoning, 8,192 output tokens per request, text observations,
500 km anchor separation, and a shared $8 ceiling covering drafting and battle.

Each decision receives a full 30-second execution window, including the last.
With the initial 30-second coast, 40 turns allow up to 1,230 simulated seconds;
elimination, surrender, or an agreed draw can end the battle earlier. The worker
runs independently of HTTP polling so model calls cannot freeze the spectator API.
Snapshots are saved after each turn under `visualizer/public/recordings/`; detailed
drafts, configuration, and a continuously updated `status.json` with spending live
under `data/recordings/<run name>/`. The spending ceiling stops further requests
and preserves a partial replay if exhausted.

For a free smoke run, add `--offline --turns 2 --port 8767 --no-linger` and use
`?live=1&live_base=http://localhost:8767/live`. This uses automatic fleets and
heuristic captains. A custom `--name` must be new, to protect existing recordings.

The viewer polls `/live/recording?since_t=<last frame>&since_seq=<last event>`.
Frames use simulation time; events use their arrival sequence. This preserves
late orders at a paused checkpoint and lets repeated/final full fetches merge
without duplicate events. Incremental responses include assets referenced by new
model calls. Clients using only `since_t` retain the earlier endpoint behaviour.
The fake live server supplies stable sequence IDs for legacy recordings too.

## Summaries and optional model commentary

The default summaries are calculated locally from recorded evidence. Each claim
links to its supporting events: commands issued, launches, impacts, misses or
interceptions, module losses, unresolved rounds and weapon wait reasons. Incoming
impacts during a ship's decision window are labelled as context. No summary
claims that a different order would have saved a ship or changed the winner.

`src/llm/replay_commentary.py` optionally adds short model observations over a
bounded set of linked events. Unknown/missing citations are rejected. The viewer
labels accepted prose **Model interpretation**, reveals it only once all cited
events are in the past, and displays its citations. Citation validation establishes
that referenced events exist; it does not prove that a model's interpretation is
correct. Deterministic summaries remain available without API calls.

Captains and admirals also receive compact engine execution feedback in their
next prompts. This works with recording disabled. They can see the last observed
weapon/helm transitions and rejected, cancelled or expired commands.

## Recorded model trials

The repository includes these completed examples:

| Battle | Recording | Result | Reported model cost |
| --- | --- | --- | --- |
| GLM 5.3 Flash vs GPT-5.6 Luna; independently drafted 200-point fleets | [Replay](../data/recordings/glm_luna_200pt_live_20260906/battle_glm_5_3_flash_vs_gpt_5_6_luna_20260906_140449.json), [drafts](../data/recordings/glm_luna_200pt_live_20260906/battle_glm_5_3_flash_vs_gpt_5_6_luna_20260906_140449.draft.json) | GLM won; Beta eliminated at 233s, after 7 turns | $0.1946 |
| Qwen 3.8 Flash vs GPT-OSS 120B | [Replay](../data/recordings/intent_qwen_gptoss.json) | Qwen won on tactical score at 300s | $0.0691 |
| DeepSeek V3.2 vs GPT-4.1 mini | [Replay](../data/recordings/intent_deepseek_mini.json) | GPT-4.1 mini won on tactical score at 360s | $0.1406 |

These are single recorded matches, not a model ranking. The earlier trial costs
include optional commentary calls. A [heuristic recording](../data/recordings/intent_heuristic.json)
also demonstrates the execution evidence without paid models.

Load any of these files through the viewer's file selector. To restore the
convenient local replay URLs after cloning:

```bash
mkdir -p visualizer/public/recordings
cp data/recordings/intent_*.json visualizer/public/recordings/
cp data/recordings/glm_luna_200pt_live_20260906/battle_glm_5_3_flash_vs_gpt_5_6_luna_20260906_140449.json \
  visualizer/public/recordings/glm_luna_200pt_live_20260906.json
```

```bash
uv run python scripts/run_replay_trials.py --budget 2
```

This runs Qwen Flash versus GPT-OSS, then DeepSeek versus GPT-4.1 mini, including
admirals and gun/torpedo captains, saves viewer-ready recordings and adds optional
commentary. Prices are fetched from OpenRouter's current model catalog. All
requests share a thread-safe budget: each attempt reserves a conservative maximum
input/output cost; successful responses settle against reported usage, while
timeouts or missing usage retain their reservation. Provider routing is also constrained
to the priced envelope with `provider.max_price`. Provider costs are reported
using [OpenRouter usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting).
The report includes reported spend and conservative committed spend separately.
The text-only budget estimator rejects image/audio inputs.


### Trials recorded on 2026-09-06

| Replay | Battle length | Recorded decisions | Reported cost, including commentary |
| --- | --- | --- | --- |
| `intent_qwen_gptoss.json` | 300 s | 33 | $0.0691 |
| `intent_deepseek_mini.json` | 360 s | 64 | $0.1406 |

Total reported spend was **$0.2097**; conservative committed spend was **$0.2607**.
There were 242 recorded commander calls, six optional commentary calls and five
retried requests. No commander calls failed permanently or returned truncated or
malformed tool arguments. One optional commentary passed citation validation;
uncited/invalid commentary was discarded. Both replays have deterministic evidence
summaries. [Machine-readable trial report](replay-trial-results.json).

The Qwen corvette decision `e152` at 120 s is a useful example: its torpedoes
impact at 175 s and 184 s after the ship's maneuver expires. Open it with:

```text
http://localhost:5173/?recording=/recordings/intent_qwen_gptoss.json&decisions=1&ship=alpha_2&decision=e152&t=184
```

Validation: **1,690 Python tests**, **15 viewer tests**, production viewer build,
wide/600 px browser checks, legacy replay checks, live incremental streaming,
model-input inspection, outcome seeking/camera focus, and three repeated seeded
heuristic scenarios. Recording enabled/disabled preserved physical outcomes in
those scenarios. Recordings live in the existing ignored recordings directory;
the guide, runner and result manifest are project files.
