#!/usr/bin/env python3
"""
Run an LLM-controlled space battle.

Usage:
    python scripts/run_llm_battle.py --verbose
    python scripts/run_llm_battle.py --alpha-model openai/gpt-5.6-terra --beta-model anthropic/claude-sonnet-5
    python scripts/run_llm_battle.py --fleet-config data/fleet_config.json
"""

import argparse
import sys
import uuid
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.providers import create_llm_client
from src.llm.captain import LLMCaptainConfig
from src.llm.prompts import CaptainPersonality
from src.llm.battle_runner import LLMBattleRunner, BattleConfig, load_fleet_data
from src.llm.fleet_config import BattleFleetConfig, _get_short_model_name
from src.llm.call_policy import parse_effort_specs


def _available_ship_types() -> list:
    """Ship classes the simulation can actually build, straight from fleet data."""
    try:
        import json
        from pathlib import Path as _Path
        data = json.loads((_Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text())
        return sorted(data.get("ships", {}).keys())
    except Exception:
        return ["corvette", "frigate", "destroyer", "cruiser",
                "battlecruiser", "battleship", "dreadnought"]


def fleet_models(fleet_config) -> list:
    """Every model a fleet config names: captains, then enabled admirals."""
    return ([s.model for s in fleet_config.get_all_ships()] +
            [f.admiral.model for f in (fleet_config.alpha_fleet, fleet_config.beta_fleet)
             if f.admiral and f.admiral.enabled])


def apply_model_overrides(fleet_config, args) -> None:
    """--{side}-admiral-model / --{side}-captain-model rewrite a loaded fleet config."""
    for side in ("alpha", "beta"):
        fleet = getattr(fleet_config, f"{side}_fleet")
        captain = getattr(args, f"{side}_captain_model", None)
        if captain:
            for ship in fleet.ships:
                ship.model = captain
        admiral = getattr(args, f"{side}_admiral_model", None)
        if admiral:
            if fleet.admiral is None:
                raise ValueError(f"--{side}-admiral-model: the {side} fleet has no admiral")
            fleet.admiral.model = admiral
            fleet.admiral.name = f"Admiral {_get_short_model_name(admiral)}"


def main():
    parser = argparse.ArgumentParser(
        description="Run an LLM-controlled space battle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_llm_battle.py --verbose
    python scripts/run_llm_battle.py --alpha-model openai/gpt-5.6-terra --max-checkpoints 10
    python scripts/run_llm_battle.py --alpha-personality aggressive --beta-personality cautious
        """,
    )

    # Model settings
    parser.add_argument(
        "--alpha-model",
        default="openrouter/anthropic/claude-sonnet-5",
        help="Model for alpha captain (default: claude-sonnet-5)",
    )
    parser.add_argument(
        "--beta-model",
        default="openrouter/anthropic/claude-sonnet-5",
        help="Model for beta captain (default: claude-sonnet-5)",
    )

    # Captain names
    parser.add_argument(
        "--alpha-name",
        default=None,
        help="Name for alpha captain",
    )
    parser.add_argument(
        "--beta-name",
        default=None,
        help="Name for beta captain",
    )

    # Ship names
    parser.add_argument(
        "--alpha-ship",
        default=None,
        help="Name for alpha ship",
    )
    parser.add_argument(
        "--beta-ship",
        default=None,
        help="Name for beta ship",
    )

    # Personalities
    parser.add_argument(
        "--alpha-personality",
        choices=["aggressive", "cautious", "balanced", "berserker", "survivor", "cosmic_wit"],
        default="balanced",
        help="Personality for alpha captain",
    )
    parser.add_argument(
        "--beta-personality",
        choices=["aggressive", "cautious", "balanced", "berserker", "survivor", "cosmic_wit"],
        default="balanced",
        help="Personality for beta captain",
    )

    # Ship types. Derived from the fleet data rather than hardcoded: the literal
    # list had drifted and omitted "corvette", so the only torpedo boat in the
    # game could not be selected from the CLI at all.
    ship_types = _available_ship_types()
    parser.add_argument(
        "--alpha-ship-type",
        choices=ship_types,
        default="destroyer",
        help="Ship class for alpha (default: destroyer)",
    )
    parser.add_argument(
        "--beta-ship-type",
        choices=ship_types,
        default="destroyer",
        help="Ship class for beta (default: destroyer)",
    )

    # Battle settings
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Initial distance in km (default: 500)",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Maximum LLM checkpoints (default: 40)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Time limit in seconds (default: 1200)",
    )
    parser.add_argument("--decision-interval", type=float, default=None,
                        help="Seconds between decisions, 20-60 (default: 30)")
    parser.add_argument("--plain-json", action="store_true",
                        help="Write the recording as plain indented JSON (battle_*.json) "
                             "instead of the default gzip (battle_*.json.gz)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed combat randomness for reproducible scripted battles")

    # Fleet configuration (for multi-ship battles with Admirals)
    parser.add_argument(
        "--fleet-config",
        type=str,
        help="Path to JSON fleet configuration file (enables fleet mode with Admirals)",
    )

    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Reasoning effort for LLM calls (low/medium/high/xhigh/max). On bare Anthropic ids it is sent as output_config.effort on every request (default medium there); on OpenRouter ids as reasoning.effort (default: unset)",
    )

    parser.add_argument(
        "--effort", action="append", default=[], metavar="[SIDE.]ROLE=LEVEL",
        help="Effort for one role (directive, order, captain, trigger) on one side (alpha, beta; "
             "omitted = both); repeatable, later wins, overrides the fleet config's per-fleet "
             "'effort' blocks. 'trigger' = captain turns with torpedoes inbound, a merge warning, "
             "damage, a changed order kind or the first turn. Unset roles run at --reasoning-effort",
    )
    for side in ("alpha", "beta"):
        parser.add_argument(f"--{side}-admiral-model", default=None,
                            help=f"Fleet mode: replace the {side} admiral's model from the fleet config")
        parser.add_argument(f"--{side}-captain-model", default=None,
                            help=f"Fleet mode: replace every {side} captain's model (e.g. 'heuristic' "
                                 "for rule-based captains under an LLM admiral)")

    # Battle modes
    parser.add_argument(
        "--unlimited",
        action="store_true",
        help="Unlimited mode: fight until destruction, surrender, or mutual draw",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Record detailed sim trace (position/velocity of all objects every step). WARNING: Large files!",
    )
    parser.add_argument(
        "--no-personality-selection",
        action="store_true",
        help="Skip personality selection phase (use preset personalities from --alpha/beta-personality)",
    )
    parser.add_argument(
        "--admiral-vision",
        action="store_true",
        help=(
            "Attach a rendered tactical-plot image to every admiral checkpoint "
            "prompt (fleet mode only; needs matplotlib and a vision-capable "
            "admiral model - silently text-only otherwise)"
        ),
    )
    parser.add_argument(
        "--notebooks",
        action="store_true",
        help=(
            "Inject each model's accepted commander-notebook lessons "
            "(data/notebooks, built by scripts/refine_commander.py) into its "
            "prompts. Off by default so battles measure the raw model."
        ),
    )

    # Output
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (only show result)",
    )

    args = parser.parse_args()

    # Map personality strings to enums
    personality_map = {
        "aggressive": CaptainPersonality.AGGRESSIVE,
        "cautious": CaptainPersonality.CAUTIOUS,
        "balanced": CaptainPersonality.BALANCED,
        "berserker": CaptainPersonality.BERSERKER,
        "survivor": CaptainPersonality.SURVIVOR,
        "cosmic_wit": CaptainPersonality.COSMIC_WIT,
    }

    try:
        # Check for fleet config mode
        fleet_config = None
        if args.fleet_config:
            # Load fleet configuration
            fleet_config = BattleFleetConfig.from_json(args.fleet_config)
            if args.admiral_vision:
                for fleet in (fleet_config.alpha_fleet, fleet_config.beta_fleet):
                    if fleet.admiral:
                        fleet.admiral.vision = True
            if args.notebooks:
                fleet_config.use_notebooks = True
            apply_model_overrides(fleet_config, args)
            # An explicit CLI value overrides JSON; omitted flags preserve it.
            for option, field in ((args.distance, "initial_distance_km"),
                                  (args.time_limit, "time_limit_s"),
                                  (args.decision_interval, "decision_interval_s")):
                if option is not None:
                    setattr(fleet_config, field, option)
            if not args.quiet:
                print(f"FLEET MODE: Loading configuration from {args.fleet_config}")
                print(f"Battle: {fleet_config.battle_name}")

        # Short display names (shared implementation - version-aware)
        alpha_short = _get_short_model_name(args.alpha_model)
        beta_short = _get_short_model_name(args.beta_model)

        # Create captain configs with model names (used for legacy mode)
        alpha_config = LLMCaptainConfig(
            name=args.alpha_name or f"Captain {alpha_short}",
            ship_name=args.alpha_ship or f"TIS {alpha_short}",
            model=args.alpha_model,
            personality=personality_map[args.alpha_personality],
        )

        beta_config = LLMCaptainConfig(
            name=args.beta_name or f"Captain {beta_short}",
            ship_name=args.beta_ship or f"HFS {beta_short}",
            model=args.beta_model,
            personality=personality_map[args.beta_personality],
        )

        # Create battle config
        config_overrides = dict(
            verbose=not args.quiet,
            alpha_ship_type=args.alpha_ship_type,
            beta_ship_type=args.beta_ship_type,
            fleet_config_path=args.fleet_config,
        )
        if args.max_checkpoints is not None:
            config_overrides["max_checkpoints"] = args.max_checkpoints
            if fleet_config:
                fleet_config.max_checkpoints = args.max_checkpoints
        if args.seed is not None:
            config_overrides["seed"] = args.seed
        if args.unlimited:
            config_overrides["unlimited_mode"] = True
        if args.trace:
            config_overrides["record_sim_trace"] = True
        if args.no_personality_selection:
            config_overrides["personality_selection"] = False
        if args.notebooks:
            config_overrides["use_notebooks"] = True
        if args.plain_json:
            config_overrides["compress_recording"] = False
        if args.effort:
            policy = parse_effort_specs(args.effort)
            if fleet_config:
                base = BattleConfig.from_fleet_config(fleet_config).effort_policy
                for side, roles in policy.items():
                    base.setdefault(side, {}).update(roles)
                policy = base
            config_overrides["effort_policy"] = policy
        if fleet_config:
            battle_config = BattleConfig.from_fleet_config(fleet_config, **config_overrides)
        else:
            for value, field in ((args.distance, "initial_distance_km"),
                                 (args.time_limit, "time_limit_s"),
                                 (args.decision_interval, "decision_interval_s")):
                if value is not None:
                    config_overrides[field] = value
            battle_config = BattleConfig(**config_overrides)

        # Create client with appropriate model
        if fleet_config:
            # The client's default model: the first LLM model of the battle
            # (a heuristic first captain is not one).
            llm = [m for m in fleet_models(fleet_config) if m not in ("heuristic", "mcp", "dummy")]
            client_model = llm[0] if llm else "openrouter/anthropic/claude-sonnet-5"
        else:
            client_model = args.alpha_model

        # Stable per-run sticky-routing key. OpenRouter otherwise derives one by
        # hashing the first system + first non-system message; our per-checkpoint
        # turn changes every time, so an explicit id keeps the whole battle pinned
        # to one provider endpoint and its warm cache.
        session_id = f"ai-commanders-{uuid.uuid4().hex[:16]}"
        models = fleet_models(fleet_config) if fleet_config else [args.alpha_model, args.beta_model]
        client = (create_llm_client(client_model, models, session_id=session_id,
                                    reasoning_effort=args.reasoning_effort)
                  if any(model != "heuristic" for model in models) else None)

        if args.unlimited and not args.quiet:
            print("UNLIMITED MODE: Battle will continue until destruction, surrender, or mutual draw")

        # Load fleet data
        fleet_data = load_fleet_data()

        # Create and run battle
        runner = LLMBattleRunner(
            config=battle_config,
            alpha_config=alpha_config,
            beta_config=beta_config,
            client=client,
            fleet_config=fleet_config,
        )

        result = runner.run_battle(fleet_data)

        # Print final result if quiet mode
        if args.quiet:
            print(f"Winner: {result.winner or 'Draw'}")
            print(f"Reason: {result.reason}")
            print(f"Duration: {result.duration_s:.0f}s ({result.checkpoints_used} checkpoints)")

        # Print messages if verbose
        if args.verbose and result.messages:
            print("\n--- All Communications ---")
            for msg in result.messages:
                print(f"  {msg}")

        # Return appropriate exit code
        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
