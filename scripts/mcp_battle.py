#!/usr/bin/env python3
"""
MCP Battle CLI - Run battles with MCP-controlled fleets.

Usage:
    # Start battle with HTTP API server (recommended)
    python scripts/mcp_battle.py --config data/fleet_config_mcp_example.json

    # Start MCP server for alpha faction in HTTP mode
    python scripts/mcp_battle.py --start-server --faction alpha --http http://localhost:8765

    # Run battle with both fleets MCP-controlled
    python scripts/mcp_battle.py --config data/fleet_config_mcp_vs_mcp.json

The HTTP API server approach allows MCP servers (spawned by Claude Code) to
communicate with the battle runner via HTTP, solving the cross-process
communication problem.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.fleet_config import BattleFleetConfig
from src.llm.battle_runner import LLMBattleRunner, BattleConfig
from src.llm.captain import LLMCaptainConfig
from src.llm.client import CaptainClient


def start_mcp_server(faction: str, http_url: str = None):
    """Start the MCP server for a faction."""
    from src.llm.mcp_server import run_mcp_server

    if http_url:
        print(f"Starting MCP server for {faction} faction (HTTP mode)...")
        print(f"Connecting to battle API at: {http_url}")
    else:
        print(f"Starting MCP server for {faction} faction (shared memory mode)...")

    print("Connect your MCP client (e.g., Claude Code) to this server.")
    print("Press Ctrl+C to stop.")

    try:
        asyncio.run(run_mcp_server(faction, http_url=http_url))
    except KeyboardInterrupt:
        print("\nServer stopped.")


def load_fleet_data() -> dict:
    """Load ship specifications from fleet_ships.json."""
    fleet_data_path = Path(__file__).parent.parent / "data" / "fleet_ships.json"
    if not fleet_data_path.exists():
        raise FileNotFoundError(f"Fleet data not found: {fleet_data_path}")

    with open(fleet_data_path) as f:
        return json.load(f)


async def run_mcp_battle(
    config_path: str,
    verbose: bool = True,
    http_host: str = "localhost",
    http_port: int = 8765,
):
    """Run a battle with MCP configuration using HTTP API."""
    print(f"Loading config from {config_path}...")

    # Load configurations
    fleet_config = BattleFleetConfig.from_json(config_path)
    print(f"Loaded config: {fleet_config.battle_name}")
    fleet_data = load_fleet_data()

    # Create battle config
    battle_config = BattleConfig(
        initial_distance_km=fleet_config.initial_distance_km,
        time_limit_s=fleet_config.time_limit_s,
        decision_interval_s=fleet_config.decision_interval_s,
        unlimited_mode=fleet_config.unlimited_mode,
        verbose=verbose,
        record_battle=fleet_config.record_battle,
        record_sim_trace=fleet_config.record_sim_trace,
        personality_selection=fleet_config.personality_selection,
    )

    # Create client for non-MCP fleets (if any)
    client = CaptainClient()

    # Create dummy captain configs (required by runner but not used for MCP fleets)
    alpha_config = LLMCaptainConfig(
        name="Alpha Captain",
        model="dummy",
        ship_name="Alpha Ship",
    )
    beta_config = LLMCaptainConfig(
        name="Beta Captain",
        model="dummy",
        ship_name="Beta Ship",
    )

    # Create runner
    runner = LLMBattleRunner(
        config=battle_config,
        alpha_config=alpha_config,
        beta_config=beta_config,
        client=client,
        fleet_config=fleet_config,
    )

    # Check if MCP is configured
    if not fleet_config.has_any_mcp():
        print("Warning: No MCP configuration found. Running as standard fleet battle.")
        result = runner.run_fleet_battle(fleet_data)
    else:
        # Get HTTP port from config if specified
        if fleet_config.alpha_fleet.mcp and fleet_config.alpha_fleet.mcp.enabled:
            http_port = fleet_config.alpha_fleet.mcp.http_port
        elif fleet_config.beta_fleet.mcp and fleet_config.beta_fleet.mcp.enabled:
            http_port = fleet_config.beta_fleet.mcp.http_port

        # Import HTTP server module
        from src.llm.mcp_http_server import run_battle_with_http_server

        print(f"\nStarting HTTP API server on http://{http_host}:{http_port}")
        print("MCP clients can now connect using:")
        print(f"  python -m src.llm.mcp_server --faction alpha --http http://{http_host}:{http_port}")
        print()

        # Run battle with HTTP server
        result = await run_battle_with_http_server(
            runner,
            fleet_data,
            host=http_host,
            port=http_port,
        )

    # Print result
    print(f"\n{'='*60}")
    print("BATTLE RESULT")
    print(f"{'='*60}")
    print(f"Outcome: {result.outcome.value}")
    if result.winner:
        print(f"Winner: {result.winner}")
    print(f"Reason: {result.reason}")
    print(f"Duration: {result.duration_s:.1f}s")
    print(f"Checkpoints: {result.checkpoints_used}")
    print(f"{'='*60}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="MCP Battle CLI - Run battles with MCP-controlled fleets"
    )

    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start MCP server mode (for Claude Code to connect)",
    )
    parser.add_argument(
        "--faction",
        choices=["alpha", "beta"],
        default="alpha",
        help="Faction for MCP server (default: alpha)",
    )
    parser.add_argument(
        "--http",
        metavar="URL",
        help="HTTP API URL for MCP server mode (e.g., http://localhost:8765)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to fleet configuration JSON file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP API server port (default: 8765)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="HTTP API server host (default: localhost)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    if args.start_server:
        # Start MCP server mode
        start_mcp_server(args.faction, http_url=args.http)
    elif args.config:
        # Run battle with config
        asyncio.run(run_mcp_battle(
            args.config,
            verbose=not args.quiet,
            http_host=args.host,
            http_port=args.port,
        ))
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Start battle with HTTP API server (recommended)")
        print("  python scripts/mcp_battle.py --config data/fleet_config_mcp_example.json")
        print()
        print("  # Start MCP server for alpha faction in HTTP mode")
        print("  python scripts/mcp_battle.py --start-server --faction alpha --http http://localhost:8765")
        print()
        print("Workflow:")
        print("  1. Start battle: python scripts/mcp_battle.py --config data/fleet_config_mcp_example.json")
        print("  2. In Claude Code, activate the ai-commanders-alpha MCP server")
        print("  3. Use MCP tools to view state and issue commands")
        print("  4. Call ready() to advance the battle")


if __name__ == "__main__":
    main()
