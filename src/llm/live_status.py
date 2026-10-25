"""Live spectator status shared by the battle HTTP servers.

The live viewer polls ``/live/recording`` while a battle runs. Between
checkpoints the recording itself says what happened; during a checkpoint
deliberation it cannot say who the battle is waiting for, because a model
call only becomes a ``model_call`` event once it returns. ``PendingCalls``
fills that gap: every audited commander call registers here for its
lifetime (see ``replay_audit.AuditedClient``), and the servers serve a
snapshot in the live block as ``pending_calls``.

Nothing here touches the simulation or the battle RNG; it is wall-clock
bookkeeping for spectators only.
"""
from __future__ import annotations

import itertools
import threading
import time
from typing import Any, Callable, Dict, List, Optional


class PendingCalls:
    """Thread-safe registry of commander model calls currently in flight."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._calls: Dict[int, Dict[str, Any]] = {}

    def start(self, actor: str, faction: Optional[str], ship_id: Optional[str] = None,
              model: Optional[str] = None) -> int:
        """Register a call; returns the token to pass to finish()."""
        with self._lock:
            token = next(self._ids)
            self._calls[token] = {'actor': actor, 'faction': faction, 'ship_id': ship_id,
                                  'model': model, 't0': self._clock()}
            return token

    def finish(self, token: int) -> None:
        with self._lock:
            self._calls.pop(token, None)

    def snapshot(self) -> List[Dict[str, Any]]:
        """In-flight calls, oldest first, with their age in wall seconds."""
        now = self._clock()
        with self._lock:
            calls = sorted(self._calls.values(), key=lambda c: c['t0'])
        return [{'actor': c['actor'], 'faction': c['faction'], 'ship_id': c['ship_id'],
                 'model': c['model'], 'age_s': round(max(0.0, now - c['t0']), 1)}
                for c in calls]

    def factions(self) -> List[str]:
        """Factions with at least one call in flight (the live 'waiting_for')."""
        with self._lock:
            return sorted({c['faction'] for c in self._calls.values() if c['faction']})

    def __len__(self) -> int:
        with self._lock:
            return len(self._calls)


def draft_live_entry(draft: Any, *, ready: bool, is_mcp: bool = False,
                     reveal_ships: bool = False, reveal_rationale: bool = False,
                     model: Optional[str] = None, budget: Optional[int] = None,
                     elapsed_s: Optional[float] = None,
                     ready_after_s: Optional[float] = None) -> Dict[str, Any]:
    """One faction's block of the live ``draft`` summary.

    Keeps the original keys (``ships`` is the hull COUNT, ``ship_names``,
    ``points_spent``, ``formation_name``) so older viewers still parse it,
    and adds the draft splash fields. Picks stay sealed: the roster (hull
    types, costs, formation offsets) is only served once ``reveal_ships``,
    the rationales only once ``reveal_rationale`` (both sides ready).
    """
    from .fleet_draft import SHIP_POINT_COSTS
    ships = list(getattr(draft, 'ships', None) or [])
    entry: Dict[str, Any] = {
        'ready': bool(ready),
        'is_mcp': bool(is_mcp),
        'ships': len(ships),
        'points_spent': getattr(draft, 'points_spent', 0) if draft else 0,
        'ship_names': [s.ship_name for s in ships] if reveal_ships else [],
        'formation_name': (getattr(draft, 'formation_name', '') or '') if reveal_ships else '',
        'budget': budget if budget is not None else getattr(draft, 'budget', None),
        'model': model,
        'auto': bool(getattr(draft, 'auto', False)),
        'elapsed_s': None if elapsed_s is None else round(elapsed_s, 1),
        'ready_after_s': None if ready_after_s is None else round(ready_after_s, 1),
        'roster': [{'name': s.ship_name, 'type': s.ship_type,
                    'cost': SHIP_POINT_COSTS.get(s.ship_type),
                    'offset_km': [round(float(v), 2) for v in s.offset_km]}
                   for s in ships] if reveal_ships else [],
        'selection_rationale': (getattr(draft, 'selection_rationale', '') or None) if reveal_rationale else None,
        'formation_rationale': (getattr(draft, 'formation_rationale', '') or None) if reveal_rationale else None,
    }
    return entry
