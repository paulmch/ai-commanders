#!/usr/bin/env python3
"""
Run an LLM-controlled space battle.

Usage:
    python scripts/run_llm_battle.py --verbose
    python scripts/run_llm_battle.py --alpha-model openai/gpt-4 --beta-model anthropic/claude-3.5-sonnet
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.client import CaptainClient
from src.llm.captain import LLMCaptainConfig
from src.llm.prompts import CaptainPersonality
from src.llm.battle_runner import LLMBattleRunner, BattleConfig, load_fleet_data


def main():
    parser = argparse.ArgumentParser(
        description="Run an LLM-controlled space battle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_llm_battle.py --verbose
    python scripts/run_llm_battle.py --alpha-model openai/gpt-4 --max-checkpoints 10
    python scripts/run_llm_battle.py --alpha-personality aggressive --beta-personality cautious
        """,
    )

    # Model settings
    parser.add_argument(
        "--alpha-model",
        default="openrouter/anthropic/claude-3.5-sonnet",
        help="Model for alpha captain (default: claude-3.5-sonnet)",
    )
    parser.add_argument(
        "--beta-model",
        default="openrouter/anthropic/claude-3.5-sonnet",
        help="Model for beta captain (default: claude-3.5-sonnet)",
    )

    # Captain names
    parser.add_argument(
        "--alpha-name",
        default="Commander Chen",
        help="Name for alpha captain",
    )
    parser.add_argument(
        "--beta-name",
        default="Captain Volkov",
        help="Name for beta captain",
    )

    # Ship names
    parser.add_argument(
        "--alpha-ship",
        default="TIS Relentless",
        help="Name for alpha ship",
    )
    parser.add_argument(
        "--beta-ship",
        default="HFS Determination",
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

    # Ship types
    parser.add_argument(
        "--alpha-ship-type",
        choices=["frigate", "destroyer", "cruiser", "battlecruiser", "battleship", "dreadnought"],
        default="destroyer",
        help="Ship class for alpha (default: destroyer)",
    )
    parser.add_argument(
        "--beta-ship-type",
        choices=["frigate", "destroyer", "cruiser", "battlecruiser", "battleship", "dreadnought"],
        default="destroyer",
        help="Ship class for beta (default: destroyer)",
    )

    # Battle settings
    parser.add_argument(
        "--distance",
        type=float,
        default=500.0,
        help="Initial distance in km (default: 500)",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=40,
        help="Maximum LLM checkpoints (default: 40)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=1200.0,
        help="Time limit in seconds (default: 1200)",
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
        # Create client
        client = CaptainClient(model=args.alpha_model)

        # Extract short model names for display
        def get_short_model_name(model: str) -> str:
            """Extract short model name from full model path."""
            # e.g., "openai/gpt-4o-mini" -> "GPT-4o-mini"
            # e.g., "anthropic/claude-3-5-haiku-20241022" -> "Claude-Haiku"
            parts = model.split("/")
            name = parts[-1] if parts else model
            # Clean up common patterns
            name = name.replace("-20241022", "").replace("-20240620", "")
            if "claude" in name.lower():
                if "haiku" in name.lower():
                    return "Claude-Haiku"
                elif "sonnet" in name.lower():
                    return "Claude-Sonnet"
                elif "opus" in name.lower():
                    return "Claude-Opus"
                return "Claude"
            elif "gpt-4o" in name.lower():
                return "GPT-4o" if "mini" not in name.lower() else "GPT-4o-mini"
            elif "gpt-4" in name.lower():
                return "GPT-4"
            return name.upper()

        alpha_short = get_short_model_name(args.alpha_model)
        beta_short = get_short_model_name(args.beta_model)

        # Create captain configs with model names
        alpha_config = LLMCaptainConfig(
            name=f"Captain {alpha_short}",
            ship_name=f"TIS {alpha_short}",
            model=args.alpha_model,
            personality=personality_map[args.alpha_personality],
        )

        beta_config = LLMCaptainConfig(
            name=f"Captain {beta_short}",
            ship_name=f"HFS {beta_short}",
            model=args.beta_model,
            personality=personality_map[args.beta_personality],
        )

        # Create battle config
        battle_config = BattleConfig(
            initial_distance_km=args.distance,
            max_checkpoints=args.max_checkpoints,
            time_limit_s=args.time_limit,
            unlimited_mode=args.unlimited,
            verbose=not args.quiet,
            record_sim_trace=args.trace,
            personality_selection=not args.no_personality_selection,
            alpha_ship_type=args.alpha_ship_type,
            beta_ship_type=args.beta_ship_type,
        )

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
        print("Make sure OPENROUTER_API_KEY is set in .env file", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
