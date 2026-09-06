# Command a fleet through MCP

MCP clients can command either or both sides of a battle. The battle process
hosts an HTTP API on port 8765; a small stdio MCP server bridges each client to
its faction. The browser viewer reads the spectator endpoints on the same API.

Install the dependencies from the repository root:

```bash
uv sync --locked --all-extras
```

## Start a battle

Choose one configuration:

```bash
# Draft Alpha yourself against an automatic fleet with heuristic captains.
uv run python scripts/mcp_battle.py \
  --config data/fleet_config_mcp_draft.json \
  --draft-budget 200 --draft-max-ships 8

# Or connect separate MCP clients to both preconfigured fleets.
uv run python scripts/mcp_battle.py \
  --config data/fleet_config_mcp_vs_mcp.json
```

For an OpenRouter opponent, use
`data/fleet_config_mcp_draft_vs_llm.json` or
`data/fleet_config_mcp_example.json`, review its model settings, and configure
`OPENROUTER_API_KEY` locally. The MCP battle launcher does not apply the
budgeted live draft runner's dollar ceiling.

## Connect a client

For clients accepting an `mcpServers` configuration, add an entry like this:

```json
{
  "mcpServers": {
    "ai-commanders-alpha": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/ai-commanders",
        "run", "python", "-m", "src.llm.mcp_server",
        "--faction", "alpha",
        "--http", "http://localhost:8765"
      ]
    }
  }
}
```

Replace the directory with your checkout's absolute path. For the second side,
use a separate client entry with `--faction beta`. MCP-controlled sides make no
OpenRouter calls through the game; the external client's own model billing is
separate.

## Draft, then command

In a draft-enabled battle, call `get_draft_state()` to inspect the budget,
ship catalog, and formation rules. Use `select_fleet` to buy ships and
`set_formation` to place them, then call `ready()` to commit. Both sides must
finish before the simulation starts.

During a battle:

1. Call `get_battle_state()` to inspect ships, contacts, resources, and messages.
2. Issue maneuvers, targets, gun orders, torpedo launches, or radiator commands.
3. Call `ready()` to finish the turn. The simulation advances when both sides
   are ready.

For example, using the tool names and arguments:

```python
get_battle_state()
set_maneuver(ship_id="alpha_1", maneuver_type="INTERCEPT",
             target_id="beta_1", throttle=1.0)
set_weapons_order(ship_id="alpha_1", spinal_mode="FIRE_IMMEDIATE",
                  turret_mode="FIRE_IMMEDIATE")
ready()
```

Ship IDs depend on the fleet. Use the IDs returned by the state tools. The
server exposes the full tool schemas, including communication, surrender, draw,
and draft tools. Maneuvers last for the configured decision window; gun firing
policies persist until changed.

## Watch and inspect

Start the viewer in another terminal:

```bash
cd visualizer
npm ci
npm run dev
```

Open `http://localhost:5173/?live=1&decisions=1`. The sample configurations enable
recording and simulation traces; custom configurations need both
`record_battle` and `record_sim_trace` for the full live view. Spectators can
scrub backward, inspect decision evidence, and return to the live head without
changing the battle.

The game waits for MCP readiness without imposing a turn timeout. If it appears
stuck at a checkpoint, inspect which faction has not called `ready()`.

See the [draft guide](draft_mode.md), [replay guide](intent-replays.md), and
[MCP tool implementation](../src/llm/mcp_server.py) for details.
