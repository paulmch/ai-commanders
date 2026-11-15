#!/usr/bin/env python3
"""
Draft-mode battle: admirals buy fleets from a point budget, place them in
formation, then command them through the battle while cheap AI captains fly
the ships.

Examples:
    # Two LLM admirals draft and fight with heuristic (zero-cost) captains
    uv run python scripts/run_draft_battle.py \\
        --alpha-admiral anthropic/claude-opus-5 \\
        --beta-admiral google/gemini-3.5-pro --trace

    # Fully offline smoke run: deterministic auto-drafts, no LLM calls at all
    uv run python scripts/run_draft_battle.py --auto-draft --no-admirals --trace

    # Cheap LLM captains instead of the rule-based AI
    uv run python scripts/run_draft_battle.py --captain-model anthropic/claude-haiku-4.5
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.battle_runner import BattleConfig, LLMBattleRunner, load_fleet_data
from src.llm.captain import LLMCaptainConfig
from src.llm.fleet_config import AdmiralConfig, BattleFleetConfig, _get_short_model_name
from src.llm.fleet_draft import (
    DEFAULT_MAX_SHIPS,
    DEFAULT_POINT_BUDGET,
    auto_draft,
    draft_summary_dict,
    draft_to_fleet_definition,
    run_admiral_draft,
)
from src.llm.heuristic_captain import DEFAULT_DOCTRINE, DOCTRINES, HEURISTIC_MODEL
from src.llm.prompts import CaptainPersonality
from src.recording_io import sidecar_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha-admiral", default="anthropic/claude-opus-5",
                        help="Admiral model for alpha fleet")
    parser.add_argument("--beta-admiral", default="anthropic/claude-sonnet-5",
                        help="Admiral model for beta fleet")
    parser.add_argument("--budget", type=int, default=DEFAULT_POINT_BUDGET,
                        help=f"Draft point budget per side (default {DEFAULT_POINT_BUDGET})")
    parser.add_argument("--max-ships", type=int, default=DEFAULT_MAX_SHIPS,
                        help=f"Max hulls per fleet (default {DEFAULT_MAX_SHIPS})")
    parser.add_argument("--captain-model", default=HEURISTIC_MODEL,
                        help=f"Captain control: '{HEURISTIC_MODEL}' (rule-based, free) "
                             "or any LLM model id for cheap LLM captains")
    parser.add_argument("--alpha-captain-model", default=None,
                        help="Override captain model for alpha ships only "
                             "(e.g. the alpha admiral's own model so it flies "
                             "its fleet directly)")
    parser.add_argument("--beta-captain-model", default=None,
                        help="Override captain model for beta ships only")
    parser.add_argument("--doctrine", choices=DOCTRINES, default=DEFAULT_DOCTRINE,
                        help="Doctrine of the rule-based captains: 'tactical' (focus fire, "
                             "range discipline, massed torpedo salvos) or 'legacy' (the "
                             f"pass-4 captain). Default {DEFAULT_DOCTRINE}")
    parser.add_argument("--alpha-doctrine", choices=DOCTRINES, default=None,
                        help="Override --doctrine for alpha's rule-based captains")
    parser.add_argument("--beta-doctrine", choices=DOCTRINES, default=None,
                        help="Override --doctrine for beta's rule-based captains")
    parser.add_argument("--mirror-draft", action="store_true",
                        help="With --auto-draft: both sides get the same fleet "
                             "(beta drafts with alpha's seed)")
    parser.add_argument("--distance", type=float, default=500.0,
                        help="Initial fleet separation in km (default 500)")
    parser.add_argument("--time-limit", type=float, default=900.0,
                        help="Battle time limit in seconds (default 900)")
    parser.add_argument("--max-checkpoints", type=int, default=25,
                        help="Checkpoint limit (default 25)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducible combat rolls")
    parser.add_argument("--trace", action="store_true",
                        help="Record full sim trace (for the 3D replay viewer)")
    parser.add_argument("--no-predictions", action="store_true",
                        help="With --trace: skip the per-frame predicted path "
                             "to the next checkpoint (sim_trace 'plan')")
    parser.add_argument("--recording-dir", default="data/recordings",
                        help="Where to save the recording (default data/recordings)")
    parser.add_argument("--plain-json", action="store_true",
                        help="Write the recording as plain indented JSON (battle_*.json) "
                             "instead of the default gzip (battle_*.json.gz)")
    parser.add_argument("--no-recording", action="store_true",
                        help="Write no recording or draft sidecar (outcome-only sweeps: "
                             "the result line and exit code are all you get)")
    parser.add_argument("--auto-draft", action="store_true",
                        help="Skip the LLM draft; both sides get deterministic auto-fleets")
    parser.add_argument("--no-admirals", action="store_true",
                        help="No admirals during the battle (captains fly autonomously)")
    parser.add_argument("--no-vision", action="store_true",
                        help="Disable tactical-plot images on admiral checkpoints")
    parser.add_argument("--notebooks", action="store_true",
                        help="Inject accepted commander-notebook lessons "
                             "(data/notebooks) into each model's prompts")
    parser.add_argument("--reasoning-effort", default=None,
                        help="Reasoning effort for LLM calls (low/medium/high/xhigh/max). On bare Anthropic ids it is sent as output_config.effort on every request (default medium there); on OpenRouter ids as reasoning.effort (default: unset)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    args = parser.parse_args()
    if args.mirror_draft and not args.auto_draft:
        parser.error("--mirror-draft needs --auto-draft (it gives both sides the same auto-draft)")

    verbose = not args.quiet
    fleet_data = load_fleet_data()

    alpha_captains = args.alpha_captain_model or args.captain_model
    beta_captains = args.beta_captain_model or args.captain_model
    needs_llm = (not args.auto_draft or not args.no_admirals
                 or any(model != HEURISTIC_MODEL for model in (alpha_captains, beta_captains)))
    client = None
    if needs_llm:
        from src.llm.providers import create_llm_client
        # A bare Anthropic id (claude-sonnet-5-5) goes to the Anthropic API, a
        # vendor/model id to OpenRouter; each side may use either.
        admirals_used = not (args.auto_draft and args.no_admirals)
        used = ([args.alpha_admiral, args.beta_admiral] if admirals_used else []) + [
            alpha_captains, beta_captains]
        client = create_llm_client(
            next(m for m in used if m != HEURISTIC_MODEL), used,
            session_id=f"ai-commanders-draft-{uuid.uuid4().hex[:16]}",
            reasoning_effort=args.reasoning_effort,
        )

    # ------------------------------------------------------------------
    # Draft phase
    # ------------------------------------------------------------------
    drafts = {}
    for faction, model in (("alpha", args.alpha_admiral), ("beta", args.beta_admiral)):
        admiral_name = f"Admiral {_get_short_model_name(model)}"
        if args.auto_draft:
            seed = (args.seed or 0) + (0 if faction == "alpha" or args.mirror_draft else 1)
            drafts[faction] = auto_draft(
                faction, budget=args.budget, max_ships=args.max_ships, seed=seed)
            if verbose:
                roster = ", ".join(s.ship_name for s in drafts[faction].ships)
                print(f"[DRAFT {faction}] auto-draft "
                      f"({drafts[faction].points_spent}/{args.budget} pts): {roster}")
        else:
            if verbose:
                print(f"\n=== DRAFT: {admiral_name} ({faction}) ===")
            drafts[faction] = run_admiral_draft(
                client=client,
                model=model,
                admiral_name=admiral_name,
                faction=faction,
                fleet_data=fleet_data,
                budget=args.budget,
                max_ships=args.max_ships,
                initial_distance_km=args.distance,
                verbose=verbose,
                seed=args.seed,
            )

    # ------------------------------------------------------------------
    # Battle phase
    # ------------------------------------------------------------------
    def admiral_cfg(model: str) -> AdmiralConfig | None:
        if args.no_admirals:
            return None
        return AdmiralConfig(
            model=model,
            vision=not args.no_vision,
            name=f"Admiral {_get_short_model_name(model)}",
        )

    fleet_config = BattleFleetConfig(
        battle_name=f"Draft Battle ({args.budget} pts)",
        alpha_fleet=draft_to_fleet_definition(
            drafts["alpha"], args.distance,
            captain_model=alpha_captains,
            admiral_config=admiral_cfg(args.alpha_admiral)),
        beta_fleet=draft_to_fleet_definition(
            drafts["beta"], args.distance,
            captain_model=beta_captains,
            admiral_config=admiral_cfg(args.beta_admiral)),
        time_limit_s=args.time_limit,
        initial_distance_km=args.distance,
        record_battle=True,
        record_sim_trace=args.trace,
        personality_selection=False,
        use_notebooks=args.notebooks,
    )

    battle_config = BattleConfig(
        initial_distance_km=args.distance,
        time_limit_s=args.time_limit,
        max_checkpoints=args.max_checkpoints,
        verbose=verbose,
        personality_selection=False,
        record_battle=not args.no_recording,
        record_sim_trace=args.trace and not args.no_recording,
        record_predictions=not args.no_predictions,
        recording_dir=args.recording_dir,
        compress_recording=not args.plain_json,
        seed=args.seed,
        heuristic_doctrine={"alpha": args.alpha_doctrine or args.doctrine,
                            "beta": args.beta_doctrine or args.doctrine},
    )

    # The 1v1 captain configs are unused in fleet mode but required by the
    # runner's constructor.
    dummy_cfg = LLMCaptainConfig(
        name="-", ship_name="-", model=args.captain_model,
        personality=CaptainPersonality.BALANCED)

    runner = LLMBattleRunner(
        config=battle_config,
        alpha_config=dummy_cfg,
        beta_config=dummy_cfg,
        client=client,
        fleet_config=fleet_config,
    )
    result = runner.run_battle(fleet_data)

    print(f"\n=== DRAFT BATTLE RESULT ===")
    print(f"Winner: {result.winner or 'Draw'} - {result.reason}")
    print(f"Duration: {result.duration_s:.0f}s, checkpoints: {result.checkpoints_used}")
    if result.recording_file:
        print(f"Recording: {result.recording_file}")
        sidecar = sidecar_path(result.recording_file)
        sidecar.write_text(json.dumps({
            "budget": args.budget,
            "alpha_captain_model": alpha_captains,
            "beta_captain_model": beta_captains,
            "alpha_doctrine": args.alpha_doctrine or args.doctrine,
            "beta_doctrine": args.beta_doctrine or args.doctrine,
            "alpha": draft_summary_dict(drafts["alpha"]),
            "beta": draft_summary_dict(drafts["beta"]),
        }, indent=2))
        print(f"Draft sidecar: {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
