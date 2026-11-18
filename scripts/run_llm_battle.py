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

from src.llm.providers import check_required_provider, create_llm_client, llm_models, provider_for_model
from src.llm.spending import RequestBudget, SpendLedger

DEFAULT_LEDGER = Path(__file__).resolve().parent.parent / "data/recordings/spend_ledger.jsonl"
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


MOCK_DEFAULT_MODELS = ("claude-sonnet-5-5", "claude-sonnet-5")


def openrouter_pricing(models) -> dict:
    """
    OpenRouter catalog prices for the vendor/model ids among `models` (a free
    metadata call; RequestBudget.reserve needs them). Bare Anthropic ids are
    priced by the client's own table and need nothing here.
    """
    wanted = [m for m in models if provider_for_model(m) == "openrouter"]
    if not wanted:
        return {}
    import httpx
    response = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
    response.raise_for_status()
    catalog = {m["id"]: m for m in response.json()["data"]}
    from src.llm.client import _strip_prefix
    pricing = {}
    for model in wanted:
        entry = catalog.get(_strip_prefix(model)) or catalog.get(model)
        if not entry:
            raise ValueError(f"--spend-limit: {model} is not in the OpenRouter catalog (cannot be priced)")
        pricing[model] = entry["pricing"]
        pricing[_strip_prefix(model)] = entry["pricing"]
    return pricing


def fleet_models(fleet_config) -> list:
    """Every model a fleet config names: captains, then enabled admirals."""
    return ([s.model for s in fleet_config.get_all_ships()] +
            [f.admiral.model for f in (fleet_config.alpha_fleet, fleet_config.beta_fleet)
             if f.admiral and f.admiral.enabled])


def _sides(value):
    return () if not value else ("alpha", "beta") if value == "both" else (value,)


def apply_model_overrides(fleet_config, args) -> None:
    """--{side}-admiral-model / --{side}-captain-model / --no-admiral / --intent-only
    rewrite a loaded fleet config."""
    for side in _sides(getattr(args, "no_admiral", None)):
        getattr(fleet_config, f"{side}_fleet").admiral = None
    choice = getattr(args, "show_score", None)
    if choice is not None:
        # --show-score picks the sides that see it (default: every admiral,
        # AdmiralConfig.show_score); "none" turns it off for both.
        shown = set(_sides(None if choice == "none" else choice))
        for side in ("alpha", "beta"):
            admiral = getattr(fleet_config, f"{side}_fleet").admiral
            if admiral is None:
                if side in shown:
                    raise ValueError(f"--show-score {side}: that fleet has no admiral")
                continue
            admiral.show_score = side in shown
    for side in _sides(getattr(args, "intent_only", None)):
        admiral = getattr(fleet_config, f"{side}_fleet").admiral
        if admiral is None:
            raise ValueError(f"--intent-only {side}: that fleet has no admiral")
        admiral.intent_only = True
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


def main(argv=None):
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
    # No default models (launch safety): a duel names both, a fleet battle
    # takes them from its --fleet-config. --mock-llm alone defaults to
    # claude-sonnet-5-5 vs claude-sonnet-5 (free, in-process).
    parser.add_argument(
        "--alpha-model", default=None,
        help="Model for the alpha captain of a duel (required without --fleet-config)",
    )
    parser.add_argument(
        "--beta-model", default=None,
        help="Model for the beta captain of a duel (required without --fleet-config)",
    )
    parser.add_argument(
        "--require-provider", choices=("anthropic", "openrouter"), default=None,
        help="Refuse to start when any role's model is served by another provider "
             "(overrides the fleet JSON's \"require_provider\")",
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

    parser.add_argument(
        "--spend-limit", type=float, default=None,
        help="Hard dollar cap (RequestBudget): every request reserves an upper bound on its cost "
             "first; once a reservation is refused the battle stops at that checkpoint "
             "(battle_stopped event, no-verdict result, partial recording saved)",
    )
    parser.add_argument(
        "--ledger", default=None,
        help="Cumulative spend ledger (JSONL); paid runs append to it "
             "(default data/recordings/spend_ledger.jsonl), mock runs only when named here",
    )
    parser.add_argument(
        "--ledger-cap", type=float, default=None,
        help="Refuse to start when the ledger total plus this run's --spend-limit would exceed it",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Per-call output ceiling (default: the client's; Anthropic ids clamp at 16000). "
             "Lower values shrink each in-flight call's cap reservation",
    )
    parser.add_argument(
        "--recording-dir", default=None,
        help="Where the recording is written (default data/recordings)",
    )
    parser.add_argument(
        "--mock-llm", action="store_true",
        help="Offline: every Anthropic-id call is answered by the in-process mock "
             "(src/llm/mock_anthropic.py); no key, no network, no cost",
    )
    parser.add_argument("--mock-thinking-tokens", type=int, default=20, help=argparse.SUPPRESS)
    sides = ("alpha", "beta", "both")
    parser.add_argument("--no-admiral", choices=sides, default=None,
                        help="Experiment: drop that side's admiral; its LLM captains fly on their own")
    parser.add_argument("--intent-only", choices=sides, default=None,
                        help="Experiment: that side's admiral issues the directive and battle plan "
                             "(and may order torpedo salvos) but no per-ship orders; captains decide")
    parser.add_argument("--show-score", choices=sides + ("none",), default=None,
                        help="Fleet mode: which admirals see the time-limit verdict as it stands "
                             "each checkpoint, with the draw-margin verdict (AdmiralConfig.show_score). "
                             "Default: every admiral (or the fleet JSON's \"show_score\"); "
                             "'none' turns it off; captains never get it")
    parser.add_argument("--label", default=None,
                        help="Free-text note naming the experiment this run belongs to: stored in the "
                             "--status-file, the recording provenance and the ledger lines")
    parser.add_argument("--quiet-turns", action="store_true",
                        help="Experiment: skip a captain call when its turn has no trigger, no new "
                             "FLEET NET traffic and an unchanged applied order (captain_turn_skipped)")
    parser.add_argument("--cache-warmup", action="store_true",
                        help="Experiment (Anthropic ids): pre-warm each cold cached prefix with one "
                             "max_tokens=0 call before a parallel fan-out reads it")
    parser.add_argument("--status-file", default=None,
                        help="Write the end-of-battle status (cost report per role and side) here")
    parser.add_argument(
        "--thinking-summary", action="store_true",
        help="Anthropic ids: request thinking display 'summarized' and record the summary per "
             "model_call (thinking_summary; audit only, never fed back). Billed the same as the "
             "default omitted display",
    )

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

    args = parser.parse_args(argv)
    if not args.fleet_config and not (args.alpha_model and args.beta_model):
        if args.mock_llm:
            args.alpha_model = args.alpha_model or MOCK_DEFAULT_MODELS[0]
            args.beta_model = args.beta_model or MOCK_DEFAULT_MODELS[1]
        else:
            print("Error: name both models (--alpha-model and --beta-model) or a --fleet-config; "
                  "there are no default models, so a mistyped command never spends on one",
                  file=sys.stderr)
            return 2
    if args.fleet_config:
        # Display names only (the fleet config carries the real models).
        args.alpha_model = args.alpha_model or "fleet"
        args.beta_model = args.beta_model or "fleet"

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
        if args.recording_dir:
            config_overrides["recording_dir"] = args.recording_dir
        if args.quiet_turns:
            config_overrides["quiet_turns"] = True
        if args.label:
            config_overrides["label"] = args.label
        if not fleet_config and args.show_score not in (None, "none"):
            raise ValueError("--show-score needs --fleet-config (fleet battles have admirals)")
        if not fleet_config and (args.no_admiral or args.intent_only):
            raise ValueError("--no-admiral / --intent-only need --fleet-config")
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
        anthropic_options = {}
        if args.thinking_summary:
            anthropic_options["thinking_display"] = "summarized"
        if args.cache_warmup:
            anthropic_options["cache_warmup"] = True
        llm_ids = llm_models(models)
        check_required_provider(
            args.require_provider or (fleet_config.require_provider if fleet_config else None), llm_ids)
        if args.mock_llm:
            not_anthropic = [m for m in llm_ids if provider_for_model(m) != "anthropic"]
            if not_anthropic:
                raise ValueError(f"--mock-llm answers bare Anthropic ids only, not {not_anthropic}")
            import anthropic
            from src.llm.mock_anthropic import mock_anthropic_transport
            anthropic_options.update(api_key="mock-no-network", http_client=anthropic.DefaultHttpxClient(
                transport=mock_anthropic_transport(seed=args.seed or 0,
                                                   thinking_tokens=args.mock_thinking_tokens)))
        if args.ledger_cap is not None and args.spend_limit is None:
            raise ValueError("--ledger-cap needs --spend-limit (the run's cap is what it reserves)")
        client_kwargs = {"max_tokens": args.max_tokens} if args.max_tokens else {}
        client = (create_llm_client(client_model, models, session_id=session_id,
                                    reasoning_effort=args.reasoning_effort,
                                    anthropic_options=anthropic_options or None, **client_kwargs)
                  if llm_ids else None)
        budget = None
        if args.spend_limit is not None and client is not None:
            budget = RequestBudget(args.spend_limit, openrouter_pricing(llm_ids))
            client.budget = budget
        ledger = ledger_run = None
        if client is not None and (not args.mock_llm or args.ledger):
            ledger = SpendLedger(args.ledger or DEFAULT_LEDGER)
            ledger.check(args.spend_limit or 0.0, args.ledger_cap)
            if args.spend_limit is None and not args.mock_llm:
                # Nothing caps this run, and its open ledger line reserves $0: a run
                # that dies without closing it counts as nothing in the ledger.
                print("WARNING: no --spend-limit: this paid battle has NO dollar cap and its ledger "
                      "line reserves $0 until it closes", file=sys.stderr, flush=True)

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

        if ledger is not None:
            # Reserved only now, right before the first request.
            ledger_run = ledger.open_run(
                script="run_llm_battle", name=Path(args.fleet_config or "duel").stem, models=llm_ids,
                cap_usd=args.spend_limit if args.spend_limit is not None else 0.0,
                ledger_cap_usd=args.ledger_cap, mock=args.mock_llm, label=args.label)
        status = "error"
        try:
            result = runner.run_battle(fleet_data)
            status = "stopped" if getattr(runner, "stop_reason", None) else "done"
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            if status in ("error", "interrupted") and budget is not None:
                # No new request from a worker thread that outlives the
                # interrupt; calls in flight count at their reservation.
                budget.stop(status)
            if ledger_run is not None:
                stats = client.stats
                ledger.close_run(ledger_run, calls=stats.calls,
                                 tokens={"prompt": stats.prompt_tokens, "cache_read": stats.cached_tokens,
                                         "cache_write": stats.cache_write_tokens},
                                 usd=budget.reported_usd if budget else stats.cost_usd,
                                 committed_usd=budget.committed_usd if budget else None, status=status)

        # Cost per role and side (model_call events), stdout + --status-file.
        if runner.recorder is not None and client is not None:
            from src.llm.cost_report import cost_report, format_cost_report
            report = cost_report(runner.recorder.events)
            print(format_cost_report(report))
            if args.status_file:
                import json as _json
                status = {"label": args.label, "result": {"winner": result.winner, "reason": result.reason,
                                     "checkpoints": result.checkpoints_used},
                          "recording": runner.recording_file, "stopped": getattr(runner, "stop_reason", None),
                          "spend": {"reported_usd": budget.reported_usd if budget else client.stats.cost_usd,
                                    "committed_usd": budget.committed_usd if budget else None,
                                    "limit_usd": args.spend_limit, "calls": client.stats.calls},
                          "effort_policy": battle_config.effort_policy, "cost_report": report}
                Path(args.status_file).parent.mkdir(parents=True, exist_ok=True)
                Path(args.status_file).write_text(_json.dumps(status, indent=1, default=str))

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
