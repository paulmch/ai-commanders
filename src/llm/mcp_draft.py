"""
Pre-battle draft phase for MCP battles.

Bridges the draft machinery (src/llm/fleet_draft.py) to MCP clients: each
MCP-controlled faction buys its fleet and places its formation over the
battle HTTP API BEFORE the simulation is set up, while non-MCP opponents
draft through their admiral LLM (or the deterministic auto-draft). The
battle waits for every faction to commit, then the drafts are materialized
into FleetDefinitions with world positions.

The manager is written to by aiohttp handlers (event loop) and by admiral
drafts running in executor threads, hence the lock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .fleet_draft import (
    FORMATION_MAX_OFFSET_KM,
    MIN_SEPARATION_KM,
    SHIP_POINT_COSTS,
    FleetDraft,
    apply_formation,
    build_catalog_text,
    name_drafted_ships,
    validate_selection,
)


@dataclass
class FactionDraftSlot:
    """One faction's in-progress draft."""
    faction: str
    is_mcp: bool
    draft: FleetDraft
    selected: bool = False
    committed: bool = False


def _roster(draft: FleetDraft) -> List[Dict[str, Any]]:
    """JSON-safe roster of a draft's ships with their formation offsets."""
    return [
        {
            "ship_id": s.ship_id,
            "ship_name": s.ship_name,
            "ship_type": s.ship_type,
            "cost": SHIP_POINT_COSTS[s.ship_type],
            "offset_km": {"x": s.offset_km[0], "y": s.offset_km[1],
                          "z": s.offset_km[2]},
        }
        for s in draft.ships
    ]


class DraftManager:
    """Holds both factions' drafts during the MCP draft phase."""

    def __init__(
        self,
        fleet_data: Dict[str, Any],
        budget: int,
        max_ships: int,
        initial_distance_km: float,
        mcp_factions: Set[str],
    ):
        self.fleet_data = fleet_data
        self.budget = budget
        self.max_ships = max_ships
        self.initial_distance_km = initial_distance_km
        self.active = True
        self._catalog = build_catalog_text(fleet_data)
        self._lock = threading.Lock()
        self._slots: Dict[str, FactionDraftSlot] = {
            faction: FactionDraftSlot(
                faction=faction,
                is_mcp=faction in mcp_factions,
                draft=FleetDraft(faction=faction, budget=budget),
            )
            for faction in ("alpha", "beta")
        }

    def slot(self, faction: str) -> FactionDraftSlot:
        return self._slots[faction]

    def waiting_for(self) -> List[str]:
        """Factions that have not committed a draft yet."""
        with self._lock:
            return [f for f, s in self._slots.items() if not s.committed]

    def all_committed(self) -> bool:
        with self._lock:
            return all(s.committed for s in self._slots.values())

    def finalize(self) -> None:
        """End the draft phase (drafts stay readable for summaries)."""
        self.active = False

    # === MCP-facing operations (called from HTTP handlers) ===

    def state_dict(self, faction: str) -> Dict[str, Any]:
        """Everything an MCP client needs to draft: budget, catalog, picks."""
        with self._lock:
            slot = self._slots[faction]
            opponent = self._slots["beta" if faction == "alpha" else "alpha"]
            return {
                "phase": "draft" if self.active else "battle",
                "faction": faction,
                "budget": self.budget,
                "max_ships": self.max_ships,
                "points_spent": slot.draft.points_spent,
                "points_remaining": self.budget - slot.draft.points_spent,
                "committed": slot.committed,
                "opponent_committed": opponent.committed,
                "initial_distance_km": self.initial_distance_km,
                "ship_costs": dict(SHIP_POINT_COSTS),
                "catalog": self._catalog,
                "your_ships": _roster(slot.draft),
                "formation_limits": {
                    "max_offset_km": FORMATION_MAX_OFFSET_KM,
                    "min_separation_km": MIN_SEPARATION_KM,
                },
                "instructions": (
                    "Buy hulls with select_fleet (replaces any previous "
                    "selection), optionally place them with set_formation "
                    "(offsets in km from your fleet anchor, +x toward the "
                    "enemy), then commit with ready(). Unplaced ships get "
                    "line-abreast default slots."
                ),
            }

    def select(
        self,
        faction: str,
        ships_arg: Any,
        rationale: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Apply a select_fleet payload. Returns (result, error) - exactly one set.

        Replaces any previous selection; formation resets to default slots.
        """
        with self._lock:
            if not self.active:
                return None, "The draft phase is over."
            slot = self._slots[faction]
            if slot.committed:
                return None, "Your draft is already committed."
            flat, spent, error = validate_selection(
                ships_arg, self.budget, self.max_ships)
            if error:
                return None, error
            slot.draft.ships = name_drafted_ships(flat, faction)
            slot.draft.points_spent = spent
            slot.draft.selection_rationale = (rationale or "").strip()
            slot.selected = True
            # Seed default line-abreast offsets so the roster always has
            # concrete positions even if set_formation is never called.
            apply_formation(slot.draft.ships, [])
            return {
                "points_spent": spent,
                "points_remaining": self.budget - spent,
                "your_ships": _roster(slot.draft),
            }, None

    def formation(
        self,
        faction: str,
        placements: Any,
        formation_name: str = "",
        rationale: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Apply a set_formation payload. Returns (result, error)."""
        with self._lock:
            if not self.active:
                return None, "The draft phase is over."
            slot = self._slots[faction]
            if slot.committed:
                return None, "Your draft is already committed."
            if not slot.selected:
                return None, "Select your fleet first (select_fleet)."
            if placements is not None and not isinstance(placements, list):
                return None, "placements must be a list of {ship_name, x_km, y_km, z_km}."
            notes = apply_formation(slot.draft.ships, placements or [])
            if formation_name:
                slot.draft.formation_name = str(formation_name).strip()
            if rationale:
                slot.draft.formation_rationale = str(rationale).strip()
            return {
                "formation_name": slot.draft.formation_name,
                "notes": notes,
                "your_ships": _roster(slot.draft),
            }, None

    def commit(self, faction: str) -> Tuple[bool, Optional[str]]:
        """Commit a faction's draft (ready during the draft phase)."""
        with self._lock:
            if not self.active:
                return False, "The draft phase is over."
            slot = self._slots[faction]
            if not slot.selected:
                return False, (
                    "You have not selected a fleet yet - call select_fleet "
                    "before ready()."
                )
            slot.committed = True
            return True, None

    # === Non-MCP side (admiral / auto drafts, may arrive from a thread) ===

    def set_full_draft(self, faction: str, draft: FleetDraft) -> None:
        """Install a complete draft (admiral LLM or auto-draft) and commit it."""
        with self._lock:
            slot = self._slots[faction]
            slot.draft = draft
            slot.selected = True
            slot.committed = True

    # === Spectator summary (live viewer) ===

    def live_summary(self) -> Dict[str, Any]:
        """Per-faction progress block for the live viewer."""
        with self._lock:
            out = {}
            for faction, slot in self._slots.items():
                out[faction] = {
                    "ready": slot.committed,
                    "is_mcp": slot.is_mcp,
                    "ships": len(slot.draft.ships),
                    "points_spent": slot.draft.points_spent,
                    "ship_names": [s.ship_name for s in slot.draft.ships],
                    "formation_name": slot.draft.formation_name,
                }
            return out
