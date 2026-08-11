#!/usr/bin/env python3
"""
Post-battle refinement: distill commander-notebook lessons from a recording,
then gate them on rematch results before they reach any prompt.

The loop (evidence-backed, continual-harness style):
  1. analyze   - read a finished recording, ask the refined model itself to
                 distill 1-2 durable lessons (admiral- and/or captain-level);
                 saved as PENDING.
  2. validate  - the rematch gate: play N battles WITH the lesson injected and
                 N WITHOUT (COSTS REAL LLM CALLS). Accept only if it wins more
                 with than without.
  3. accept /  - manual override for either direction.
     reject
Battles only see ACCEPTED entries, and only when run with notebooks enabled
(fleet config "use_notebooks": true, or --notebooks on the battle scripts).

Examples:
    # Distill lessons for every LLM admiral in a recording (pending)
    uv run python scripts/refine_commander.py analyze data/recordings/battle_x.json

    # Captain-level lessons for one model
    uv run python scripts/refine_commander.py analyze data/recordings/battle_x.json \\
        --model anthropic/claude-sonnet-5 --role captain --side beta

    # Inspect notebooks
    uv run python scripts/refine_commander.py list

    # Rematch-gate a pending entry (2x3 battles - costs money!)
    uv run python scripts/refine_commander.py validate anthropic/claude-sonnet-5 \\
        a1b2c3d4 --config data/fleet_config_x.json --battles 3 --auto
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
from src.llm.captain import LLMCaptainConfig
from src.llm.client import CaptainClient
from src.llm.fleet_config import BattleFleetConfig
from src.llm.notebook import (
    DEFAULT_NOTEBOOK_DIR,
    add_entry,
    build_battle_digest,
    distill_lessons,
    get_entry,
    infer_side_for_model,
    load_notebook,
    model_slug,
    notebook_path,
    set_status,
)


def _print_entry(entry: dict, indent: str = "  ") -> None:
    v = entry.get("validation", {})
    validated = (f" | rematches with {v.get('with_wins', 0)}W/{v.get('with_losses', 0)}L"
                 f" vs without {v.get('without_wins', 0)}W/{v.get('without_losses', 0)}L"
                 if any(v.values()) else "")
    print(f"{indent}[{entry['id']}] ({entry['status']}, {entry.get('role', 'any')})"
          f"{validated}")
    print(f"{indent}  {entry['text']}")
    if entry.get("source_outcome"):
        print(f"{indent}  from: {entry['source_outcome']}")


def cmd_analyze(args: argparse.Namespace) -> int:
    recording = json.loads(Path(args.recording).read_text())
    notebook_dir = Path(args.notebook_dir) if args.notebook_dir else None

    # Build the (model, role, side) jobs to refine.
    jobs = []
    if args.model:
        side = args.side or infer_side_for_model(recording, args.model)
        if not side:
            print(f"Cannot infer which side {args.model} commanded - pass --side.",
                  file=sys.stderr)
            return 1
        role = args.role or (
            "admiral" if (recording.get(f"{side}_fleet") or {})
            .get("admiral", {}).get("model") == args.model else "captain")
        jobs.append((args.model, role, side))
    else:
        for side in ("alpha", "beta"):
            admiral = (recording.get(f"{side}_fleet") or {}).get("admiral") or {}
            model = admiral.get("model")
            if model and model != "heuristic":
                jobs.append((model, "admiral", side))
        if not jobs:  # 1v1 recording: captains only
            for side in ("alpha", "beta"):
                model = recording.get(f"{side}_model")
                if model:
                    jobs.append((model, "captain", side))
    if not jobs:
        print("No LLM commanders found in this recording.", file=sys.stderr)
        return 1

    winner = recording.get("winner")
    for model, role, side in jobs:
        outcome = ("won" if winner == side else "drew" if not winner else "lost")
        enemy_side = "beta" if side == "alpha" else "alpha"
        enemy_admiral = ((recording.get(f"{enemy_side}_fleet") or {})
                         .get("admiral") or {}).get("model")
        enemy_label = enemy_admiral or recording.get(f"{enemy_side}_model") or "unknown"
        source_outcome = f"{outcome} vs {enemy_label}"

        print(f"\n=== Distilling {role} lessons for {model} ({side}, {outcome}) ===")
        digest = build_battle_digest(recording, side)
        client = CaptainClient(
            model=args.analyst_model or model,
            session_id=f"ai-commanders-refine-{uuid.uuid4().hex[:16]}",
        )
        lessons = distill_lessons(client, model, role, digest,
                                  analyst_model=args.analyst_model)
        if not lessons:
            print("  (analyst recorded no lessons)")
            continue
        for lesson in lessons:
            entry = add_entry(
                model=model,
                text=lesson["text"],
                role=lesson["role"],
                source_battle=str(args.recording),
                source_outcome=source_outcome,
                status="accepted" if args.auto_accept else "pending",
                notebook_dir=notebook_dir,
            )
            _print_entry(entry)
        print(f"  -> {notebook_path(model, notebook_dir)}")
    if not args.auto_accept:
        print("\nEntries are PENDING. Gate them with 'validate' (rematch A/B) "
              "or force with 'accept'.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    notebook_dir = Path(args.notebook_dir) if args.notebook_dir else DEFAULT_NOTEBOOK_DIR
    paths = ([notebook_path(args.model, notebook_dir)] if args.model
             else sorted(notebook_dir.glob("*.json")))
    found = False
    for path in paths:
        if not path.exists():
            continue
        notebook = json.loads(path.read_text())
        if not notebook.get("entries"):
            continue
        found = True
        print(f"\n{notebook['model']}  ({path})")
        for entry in notebook["entries"]:
            _print_entry(entry)
    if not found:
        print("No notebook entries yet. Create some with 'analyze'.")
    return 0


def cmd_set_status(args: argparse.Namespace, status: str) -> int:
    notebook_dir = Path(args.notebook_dir) if args.notebook_dir else None
    entry = set_status(args.model, args.entry_id, status, notebook_dir=notebook_dir)
    _print_entry(entry)
    return 0


def _model_side_in_config(fleet_config: BattleFleetConfig, model: str) -> str:
    for side, fleet in (("alpha", fleet_config.alpha_fleet),
                        ("beta", fleet_config.beta_fleet)):
        if fleet.admiral and fleet.admiral.model == model:
            return side
        if any(s.model == model for s in fleet.ships):
            return side
    raise SystemExit(f"Model {model} does not appear in this fleet config - "
                     "the rematch gate needs it commanding one side.")


def _run_gate_battle(config_path: str, seed: int, max_checkpoints: int) -> str:
    """One validation battle with notebooks enabled. Returns winner or ''."""
    fleet_config = BattleFleetConfig.from_json(config_path)
    fleet_config.use_notebooks = True
    client_model = (fleet_config.alpha_fleet.ships[0].model
                    if fleet_config.alpha_fleet.ships else "anthropic/claude-sonnet-5")
    client = CaptainClient(
        model=client_model,
        session_id=f"ai-commanders-gate-{uuid.uuid4().hex[:16]}",
    )
    battle_config = BattleConfig(
        initial_distance_km=fleet_config.initial_distance_km,
        time_limit_s=fleet_config.time_limit_s,
        max_checkpoints=max_checkpoints,
        verbose=False,
        personality_selection=False,  # keep the A/B about the lesson, not vibes
        record_battle=False,
        seed=seed,
    )
    dummy = LLMCaptainConfig(name="-", ship_name="-", model=client_model)
    runner = LLMBattleRunner(config=battle_config, alpha_config=dummy,
                             beta_config=dummy, client=client,
                             fleet_config=fleet_config)
    result = runner.run_fleet_battle(load_fleet_data())
    return result.winner or ""


def cmd_validate(args: argparse.Namespace) -> int:
    entry = get_entry(args.model, args.entry_id)
    if not entry:
        print(f"No entry {args.entry_id} for {args.model}.", file=sys.stderr)
        return 1
    fleet_config = BattleFleetConfig.from_json(args.config)
    side = _model_side_in_config(fleet_config, args.model)

    total = 2 * args.battles
    print(f"Rematch gate for [{entry['id']}] \"{entry['text'][:70]}...\"")
    print(f"  {args.battles} battles WITH the lesson vs {args.battles} WITHOUT "
          f"({total} full LLM battles - this costs real API usage).")

    original_status = entry["status"]
    tallies = dict(entry.get("validation") or {})
    try:
        for phase, status in (("with", "accepted"), ("without", "rejected")):
            set_status(args.model, args.entry_id, status)
            for i in range(args.battles):
                seed = (args.seed or 0) + i
                winner = _run_gate_battle(args.config, seed, args.max_checkpoints)
                won = winner == side
                key = f"{phase}_{'wins' if won else 'losses'}"
                tallies[key] = tallies.get(key, 0) + 1
                print(f"  [{phase} {i + 1}/{args.battles}] winner: "
                      f"{winner or 'draw'} ({'WIN' if won else 'no win'} for {side})")
    finally:
        # Whatever happened, the entry goes back to a deliberate state with
        # the tallies recorded - never left accidentally accepted.
        set_status(args.model, args.entry_id, original_status, validation=tallies)

    with_w, without_w = tallies.get("with_wins", 0), tallies.get("without_wins", 0)
    print(f"\nResult: with {with_w}/{args.battles} wins, "
          f"without {without_w}/{args.battles} wins.")
    if with_w > without_w:
        verdict = "accept"
    elif with_w < without_w:
        verdict = "reject"
    else:
        verdict = "pending"
    if args.auto and verdict != "pending":
        set_status(args.model, args.entry_id,
                   "accepted" if verdict == "accept" else "rejected",
                   validation=tallies)
        print(f"Auto-{verdict}ed entry {args.entry_id}.")
    else:
        print(f"Recommendation: {verdict.upper()} "
              f"(apply with '{verdict} {args.model} {args.entry_id}')"
              if verdict != "pending" else
              "Recommendation: tie - left pending; run more battles.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="Distill pending lessons from a recording")
    p.add_argument("recording", help="Path to a battle recording JSON")
    p.add_argument("--model", help="Refine only this model id")
    p.add_argument("--role", choices=["admiral", "captain"],
                   help="Lesson level (default: inferred)")
    p.add_argument("--side", choices=["alpha", "beta"],
                   help="Side the model commanded (default: inferred)")
    p.add_argument("--analyst-model",
                   help="Model to run the analysis (default: the refined model itself)")
    p.add_argument("--auto-accept", action="store_true",
                   help="Skip the gate and accept immediately (not recommended)")
    p.add_argument("--notebook-dir", help="Override data/notebooks (mostly for tests)")

    p = sub.add_parser("list", help="Show notebooks")
    p.add_argument("--model", help="Only this model")
    p.add_argument("--notebook-dir")

    for name in ("accept", "reject"):
        p = sub.add_parser(name, help=f"{name.title()} an entry")
        p.add_argument("model")
        p.add_argument("entry_id")
        p.add_argument("--notebook-dir")

    p = sub.add_parser("validate",
                       help="Rematch-gate a pending entry (runs real battles)")
    p.add_argument("model")
    p.add_argument("entry_id")
    p.add_argument("--config", required=True,
                   help="Fleet config JSON for the rematches")
    p.add_argument("--battles", type=int, default=3,
                   help="Battles per arm (default 3 -> 6 total)")
    p.add_argument("--max-checkpoints", type=int, default=15)
    p.add_argument("--seed", type=int, default=None,
                   help="Base RNG seed; arm i uses seed+i so both arms see the "
                        "same combat dice")
    p.add_argument("--auto", action="store_true",
                   help="Apply the verdict automatically")

    args = parser.parse_args()
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "accept":
        return cmd_set_status(args, "accepted")
    if args.command == "reject":
        return cmd_set_status(args, "rejected")
    if args.command == "validate":
        return cmd_validate(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
