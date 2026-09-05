"""
Fleet configuration for multi-ship battles with optional Admirals.

Simplified design:
- Specify models and ship types only
- Captain names derived from model (e.g., "Captain Haiku-1")
- Ship names derived from model (e.g., "TIS Haiku-1")
- Personalities chosen by models at runtime via select_personality()
- Faction name chosen by Admiral (or defaults to "Alpha Fleet"/"Beta Fleet")
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path

from .client import DEFAULT_ADMIRAL_MODEL, DEFAULT_MODEL


@dataclass
class AdmiralConfig:
    """Configuration for a fleet admiral - just the model."""
    model: str
    enabled: bool = True
    temperature: float = 0.7
    # Attach a rendered tactical-plot image to the admiral's checkpoint
    # prompt (docs/admiral_vision.md). Only takes effect for vision-capable
    # models with matplotlib installed; silently text-only otherwise.
    vision: bool = False
    # These are set at runtime by the Admiral
    name: Optional[str] = None
    personality: Optional[str] = None
    faction_name: Optional[str] = None
    # Accepted commander-notebook lessons (src/llm/notebook.py), injected by
    # the runner when the battle opts in via use_notebooks.
    notebook_text: Optional[str] = None


@dataclass
class MCPConfig:
    """Configuration for MCP-controlled fleet (replaces Admiral when enabled)."""
    enabled: bool = True
    transport: str = "http"  # "http" (recommended), "stdio", or "sse"
    http_port: int = 8765  # Port for HTTP API server
    server_command: Optional[str] = None  # Command to launch MCP server
    name: str = "MCP Commander"
    command_timeout: float = 60.0  # Seconds to wait for MCP client commands (ignored in HTTP mode)


@dataclass
class DraftConfig:
    """
    Pre-battle draft phase: both sides buy their fleets from a point budget
    (src/llm/fleet_draft.py) instead of using the configured ship lists.

    With a draft enabled the per-fleet "ships" lists become optional: MCP
    fleets draft over the battle HTTP API, LLM-admiral fleets draft via
    run_admiral_draft, and fleets with neither get the deterministic
    auto-draft.
    """
    enabled: bool = True
    budget: int = 100
    max_ships: int = 8
    # Captain model flying drafted non-MCP fleets. MCP fleets always get
    # "mcp" placeholder captains (commands are injected directly).
    captain_model: str = "heuristic"


@dataclass
class ShipConfig:
    """Configuration for a ship and its captain - minimal spec."""
    ship_id: str
    ship_type: str  # "frigate", "destroyer", "cruiser", etc.
    model: str  # The LLM model for this captain
    temperature: float = 0.7
    # Optional position/velocity overrides (in km and km/s)
    position: Optional[Dict[str, float]] = None
    velocity: Optional[Dict[str, float]] = None
    # These are derived/set at runtime
    captain_name: Optional[str] = None  # Derived from model
    ship_name: Optional[str] = None  # Derived from model
    captain_personality: Optional[str] = None  # Chosen by model


@dataclass
class FleetDefinition:
    """Definition of a fleet (one side)."""
    ships: List[ShipConfig]
    faction: str  # "alpha" or "beta"
    admiral: Optional[AdmiralConfig] = None
    mcp: Optional[MCPConfig] = None  # MCP control (replaces admiral if enabled)


@dataclass
class BattleFleetConfig:
    """Complete battle configuration with both fleets."""
    battle_name: str
    alpha_fleet: FleetDefinition
    beta_fleet: FleetDefinition
    time_limit_s: float = 1200.0
    decision_interval_s: float = 30.0
    initial_distance_km: float = 500.0
    unlimited_mode: bool = False
    record_battle: bool = True
    record_sim_trace: bool = False
    personality_selection: bool = True  # Let models choose personalities
    # Inject each model's accepted commander-notebook lessons into its prompts.
    # OFF by default so evaluation battles measure the raw model; turn on when
    # the point is cross-battle learning (model + accumulated experience).
    use_notebooks: bool = False
    # Optional pre-battle draft phase (fleets are bought, not configured).
    draft: Optional[DraftConfig] = None

    @classmethod
    def from_json(cls, path: str) -> 'BattleFleetConfig':
        """Load configuration from JSON file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Fleet config not found: {path}")

        with open(config_path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BattleFleetConfig':
        """Create configuration from dictionary."""
        # Parse alpha fleet
        alpha_data = data.get("alpha_fleet", data.get("fleets", {}).get("alpha", {}))
        alpha_fleet = _parse_fleet(alpha_data, "alpha")

        # Parse beta fleet
        beta_data = data.get("beta_fleet", data.get("fleets", {}).get("beta", {}))
        beta_fleet = _parse_fleet(beta_data, "beta")

        # Parse optional draft phase
        draft = None
        draft_data = data.get("draft")
        if draft_data and draft_data.get("enabled", True):
            draft = DraftConfig(
                enabled=True,
                budget=int(draft_data.get("budget", 100)),
                max_ships=int(draft_data.get("max_ships", 8)),
                captain_model=draft_data.get("captain_model", "heuristic"),
            )

        # Validate - with a draft the ship lists are drafted, not configured
        if draft is None:
            if not alpha_fleet.ships:
                raise ValueError("Alpha fleet must have at least one ship")
            if not beta_fleet.ships:
                raise ValueError("Beta fleet must have at least one ship")

        return cls(
            battle_name=data.get("battle_name", "Fleet Battle"),
            alpha_fleet=alpha_fleet,
            beta_fleet=beta_fleet,
            time_limit_s=data.get("time_limit_s", 1200.0),
            decision_interval_s=data.get("decision_interval_s", 30.0),
            initial_distance_km=data.get("initial_distance_km", 500.0),
            unlimited_mode=data.get("unlimited_mode", False),
            record_battle=data.get("record_battle", True),
            record_sim_trace=data.get("record_sim_trace", False),
            personality_selection=data.get("personality_selection", True),
            use_notebooks=data.get("use_notebooks", False),
            draft=draft,
        )

    def get_all_ships(self) -> List[ShipConfig]:
        """Get all ships from both fleets."""
        return self.alpha_fleet.ships + self.beta_fleet.ships

    def get_ship_by_id(self, ship_id: str) -> Optional[ShipConfig]:
        """Find a ship by ID."""
        for ship in self.get_all_ships():
            if ship.ship_id == ship_id:
                return ship
        return None

    def has_any_admiral(self) -> bool:
        """Check if either fleet has an Admiral."""
        return (
            (self.alpha_fleet.admiral is not None and self.alpha_fleet.admiral.enabled) or
            (self.beta_fleet.admiral is not None and self.beta_fleet.admiral.enabled)
        )

    def has_any_mcp(self) -> bool:
        """Check if either fleet is MCP-controlled."""
        return (
            (self.alpha_fleet.mcp is not None and self.alpha_fleet.mcp.enabled) or
            (self.beta_fleet.mcp is not None and self.beta_fleet.mcp.enabled)
        )

    def is_alpha_mcp(self) -> bool:
        """Check if alpha fleet is MCP-controlled."""
        return self.alpha_fleet.mcp is not None and self.alpha_fleet.mcp.enabled

    def is_beta_mcp(self) -> bool:
        """Check if beta fleet is MCP-controlled."""
        return self.beta_fleet.mcp is not None and self.beta_fleet.mcp.enabled


def _get_short_model_name(model: str) -> str:
    """
    Extract a short, version-distinguishing display name from a model id.

    Version matters: a battle between claude-opus-5 and claude-opus-4.8 must not
    label both fleets "Opus", or logs, recordings and the visualizer become
    ambiguous. Examples:
        "openrouter/anthropic/claude-opus-5"   -> "Opus5"
        "anthropic/claude-opus-4.8"            -> "Opus4.8"
        "openai/gpt-5.6-terra"                 -> "GPT5.6"
        "x-ai/grok-4.20"                       -> "Grok4.20"
        "google/gemini-3.5-flash"              -> "Gemini3.5"
    """
    name = model.split("/")[-1] if model else model
    lowered = name.lower()

    if lowered in ("dummy", "mcp"):
        return "MCP"

    # Families in priority order; the first match wins.
    families = [
        ("haiku", "Haiku"), ("sonnet", "Sonnet"), ("opus", "Opus"),
        ("fable", "Fable"), ("mythos", "Mythos"), ("claude", "Claude"),
        ("grok", "Grok"), ("gpt", "GPT"), ("gemini", "Gemini"),
        ("llama", "Llama"), ("mistral", "Mistral"), ("deepseek", "DeepSeek"),
        ("qwen", "Qwen"),
    ]
    label = next((disp for key, disp in families if key in lowered), None)
    if label is None:
        return name.split("-")[0].title()

    # First version-looking token (4, 4.5, 4.20, 5.6). Ignore date stamps.
    version = ""
    for token in re.findall(r"\d+(?:\.\d+)?", re.sub(r"-20\d{6}", "", lowered)):
        version = token
        break

    return f"{label}{version}" if version else label


def _parse_fleet(data: Dict[str, Any], faction: str) -> FleetDefinition:
    """Parse a fleet definition from JSON data."""
    # Parse MCP config (optional - takes precedence over admiral if enabled)
    mcp = None
    mcp_data = data.get("mcp")
    if mcp_data and mcp_data.get("enabled", True):
        mcp = MCPConfig(
            enabled=mcp_data.get("enabled", True),
            transport=mcp_data.get("transport", "http"),
            http_port=mcp_data.get("http_port", 8765),
            server_command=mcp_data.get("server_command"),
            name=mcp_data.get("name", "MCP Commander"),
            command_timeout=mcp_data.get("command_timeout", 60.0),
        )

    # Parse admiral (optional - ignored if MCP is enabled)
    admiral = None
    admiral_data = data.get("admiral")
    if admiral_data and not mcp:  # Only parse admiral if MCP is not enabled
        # Can be just a model string or a dict
        if isinstance(admiral_data, str):
            model = admiral_data
            admiral = AdmiralConfig(
                model=model,
                name=f"Admiral {_get_short_model_name(model)}",
            )
        elif admiral_data.get("enabled", True):
            model = admiral_data.get("model", DEFAULT_ADMIRAL_MODEL)
            admiral = AdmiralConfig(
                model=model,
                enabled=admiral_data.get("enabled", True),
                temperature=admiral_data.get("temperature", 0.7),
                vision=admiral_data.get("vision", False),
                name=admiral_data.get("name", f"Admiral {_get_short_model_name(model)}"),
            )

    # Parse ships
    ships = []
    model_counts = {}  # Track how many ships per model for numbering

    for i, ship_data in enumerate(data.get("ships", [])):
        # Can be just a dict with ship_type and model, or more detailed
        if isinstance(ship_data, dict):
            model = ship_data.get("model", ship_data.get("captain_model", DEFAULT_MODEL))
            ship_type = ship_data.get("ship_type", "destroyer")
            ship_id = ship_data.get("ship_id", f"{faction}_{i + 1}")
            temperature = ship_data.get("temperature", 0.7)

            # Parse optional position override
            position = None
            if "position" in ship_data:
                pos = ship_data["position"]
                position = {
                    "x": float(pos.get("x", 0)),
                    "y": float(pos.get("y", 0)),
                    "z": float(pos.get("z", 0)),
                }

            # Parse optional velocity override
            velocity = None
            if "velocity" in ship_data:
                vel = ship_data["velocity"]
                velocity = {
                    "x": float(vel.get("x", 0)),
                    "y": float(vel.get("y", 0)),
                    "z": float(vel.get("z", 0)),
                }
        else:
            # Shouldn't happen but handle gracefully
            model = DEFAULT_MODEL
            ship_type = "destroyer"
            ship_id = f"{faction}_{i + 1}"
            temperature = 0.7
            position = None
            velocity = None

        # Generate captain name and ship name from model
        short_name = _get_short_model_name(model)
        model_counts[short_name] = model_counts.get(short_name, 0) + 1
        count = model_counts[short_name]

        # Add number suffix if multiple ships with same model
        suffix = f"-{count}" if count > 1 or len([s for s in data.get("ships", []) if _get_short_model_name(s.get("model", s.get("captain_model", ""))) == short_name]) > 1 else ""

        captain_name = f"Captain {short_name}{suffix}"
        prefix = "TIS" if faction == "alpha" else "OCS"
        ship_name = f"{prefix} {short_name}{suffix}"

        ship = ShipConfig(
            ship_id=ship_id,
            ship_type=ship_type,
            model=model,
            temperature=temperature,
            position=position,
            velocity=velocity,
            captain_name=captain_name,
            ship_name=ship_name,
        )
        ships.append(ship)

    return FleetDefinition(
        ships=ships,
        faction=faction,
        admiral=admiral,
        mcp=mcp,
    )


def validate_fleet_config(config: BattleFleetConfig, fleet_data: Dict[str, Any]) -> List[str]:
    """
    Validate fleet configuration against available ship types.

    Args:
        config: Fleet configuration to validate
        fleet_data: Ship specifications from fleet_ships.json

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    available_ships = fleet_data.get("ships", {})

    for ship in config.get_all_ships():
        if ship.ship_type not in available_ships:
            errors.append(
                f"Unknown ship type '{ship.ship_type}' for ship '{ship.ship_name}'. "
                f"Available: {list(available_ships.keys())}"
            )

    # Check for duplicate ship IDs
    ship_ids = [s.ship_id for s in config.get_all_ships()]
    if len(ship_ids) != len(set(ship_ids)):
        errors.append("Duplicate ship IDs found in configuration")

    return errors
