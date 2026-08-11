"""
Commander notebooks: cross-battle memory for LLM admirals and captains.

Everything an agent knows today is rebuilt from scratch every battle - the same
static doctrine, the same blank slate. A notebook is the per-model exception: a
small file of battle-earned lessons ("amendments") distilled from finished
recordings, injected into that model's system prompt in later battles.

Design rules (borrowed from continual-harness systems and adapted to a game
with ground truth):
- The base doctrine in prompts.py is IMMUTABLE. Notebook entries only append -
  a model can never rewrite the engine-accurate reference its tests pin.
- Entries are small (a few hundred chars), few (bounded per prompt), and carry
  provenance (which battle, what outcome).
- Entries start as "pending" and only reach a prompt once "accepted". The
  honest way to accept is a rematch gate (scripts/refine_commander.py
  validate): the lesson plays A/B battles and must actually win more with the
  entry than without.
- Injection is opt-in per battle (BattleFleetConfig.use_notebooks, default
  off), so evaluation wars stay a clean model-vs-model measurement unless the
  point is to measure learning.
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_NOTEBOOK_DIR = Path("data/notebooks")

# Prompt-side bounds: notebooks ride in the cached system prefix, so keep them
# small enough that a long-lived commander does not bloat every call.
MAX_PROMPT_ENTRIES = 6
MAX_ENTRY_CHARS = 400

VALID_ROLES = ("admiral", "captain", "any")
VALID_STATUS = ("pending", "accepted", "rejected")


def model_slug(model: str) -> str:
    """Filesystem-safe identifier for a model id."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model.strip()) or "unknown"


def notebook_path(model: str, notebook_dir: Optional[Path] = None) -> Path:
    return Path(notebook_dir or DEFAULT_NOTEBOOK_DIR) / f"{model_slug(model)}.json"


def load_notebook(model: str, notebook_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load a model's notebook, or an empty one if none exists."""
    path = notebook_path(model, notebook_dir)
    if path.exists():
        return json.loads(path.read_text())
    return {"model": model, "entries": []}


def save_notebook(notebook: Dict[str, Any], notebook_dir: Optional[Path] = None) -> Path:
    path = notebook_path(notebook["model"], notebook_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=2))
    return path


def add_entry(
    model: str,
    text: str,
    role: str = "any",
    source_battle: str = "",
    source_outcome: str = "",
    status: str = "pending",
    notebook_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Append a lesson to a model's notebook and persist it."""
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")
    if status not in VALID_STATUS:
        raise ValueError(f"status must be one of {VALID_STATUS}, got {status!r}")
    text = text.strip()
    if not text:
        raise ValueError("entry text must be non-empty")
    if len(text) > MAX_ENTRY_CHARS:
        text = text[:MAX_ENTRY_CHARS] + "..."

    notebook = load_notebook(model, notebook_dir)
    entry = {
        "id": uuid.uuid4().hex[:8],
        "role": role,
        "text": text,
        "status": status,
        "source_battle": source_battle,
        "source_outcome": source_outcome,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "validation": {"with_wins": 0, "with_losses": 0,
                       "without_wins": 0, "without_losses": 0},
    }
    notebook["entries"].append(entry)
    save_notebook(notebook, notebook_dir)
    return entry


def get_entry(
    model: str, entry_id: str, notebook_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    for entry in load_notebook(model, notebook_dir)["entries"]:
        if entry["id"] == entry_id:
            return entry
    return None


def set_status(
    model: str,
    entry_id: str,
    status: str,
    notebook_dir: Optional[Path] = None,
    validation: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Change an entry's status (and optionally merge validation tallies)."""
    if status not in VALID_STATUS:
        raise ValueError(f"status must be one of {VALID_STATUS}, got {status!r}")
    notebook = load_notebook(model, notebook_dir)
    for entry in notebook["entries"]:
        if entry["id"] == entry_id:
            entry["status"] = status
            if validation:
                entry.setdefault("validation", {}).update(validation)
            save_notebook(notebook, notebook_dir)
            return entry
    raise KeyError(f"no entry {entry_id!r} in notebook for {notebook['model']}")


def notebook_prompt_text(
    model: str,
    role: str,
    notebook_dir: Optional[Path] = None,
    include_ids: Optional[List[str]] = None,
    max_entries: int = MAX_PROMPT_ENTRIES,
) -> Optional[str]:
    """
    Render a model's accepted lessons for injection into its system prompt.

    Args:
        model: Model id whose notebook to read.
        role: "admiral" or "captain" - selects entries for that role plus
            role-agnostic ("any") ones.
        include_ids: Extra entry ids to include regardless of status. This is
            how the rematch gate plays a still-pending candidate.
        max_entries: Newest-first cap on rendered entries.

    Returns:
        The prompt block, or None when there is nothing to say.
    """
    include_ids = set(include_ids or [])
    entries = [
        e for e in load_notebook(model, notebook_dir)["entries"]
        if e.get("role", "any") in (role, "any")
        and (e.get("status") == "accepted" or e["id"] in include_ids)
    ]
    if not entries:
        return None
    entries = entries[-max_entries:]

    lines = [
        "COMMANDER'S NOTEBOOK (lessons you earned in previous battles - they",
        "supplement your doctrine, they never override the physics above):",
    ]
    for entry in entries:
        outcome = entry.get("source_outcome")
        provenance = f" [{outcome}]" if outcome else ""
        lines.append(f"- {entry['text']}{provenance}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-battle analysis: recording -> digest -> distilled lessons
# ---------------------------------------------------------------------------

# Event types that matter for a post-mortem, in rough narrative order.
_DIGEST_DECISION_EVENTS = ("admiral_plan", "admiral_directive")


def infer_side_for_model(recording: Dict[str, Any], model: str) -> Optional[str]:
    """Which faction's admiral ran on this model, if either."""
    for side in ("alpha", "beta"):
        admiral = (recording.get(f"{side}_fleet") or {}).get("admiral") or {}
        if admiral.get("model") == model:
            return side
    for side in ("alpha", "beta"):
        if recording.get(f"{side}_model") == model:
            return side
    return None


def build_battle_digest(
    recording: Dict[str, Any],
    side: str,
    max_chars: int = 8000,
) -> str:
    """
    Compress a battle recording into a post-mortem digest for one side.

    Full-information on purpose: this is analysis after the battle, so the
    enemy's directives are visible too (labeled). Bounded so the analysis call
    stays cheap even for long recordings.
    """
    events = recording.get("events", [])
    other = "beta" if side == "alpha" else "alpha"

    def fleet_line(faction: str) -> str:
        fleet = recording.get(f"{faction}_fleet") or {}
        admiral = (fleet.get("admiral") or {})
        ships = fleet.get("ships") or []
        types: Dict[str, int] = {}
        for s in ships:
            types[s.get("ship_type", "?")] = types.get(s.get("ship_type", "?"), 0) + 1
        comp = ", ".join(f"{n}x {t}" for t, n in sorted(types.items())) or "?"
        return (f"{faction.upper()}: admiral {admiral.get('model', 'none')}, "
                f"fleet {comp}")

    lines = [
        f"BATTLE POST-MORTEM (you commanded {side.upper()})",
        fleet_line(side),
        fleet_line(other),
        "",
    ]

    # Decision narrative: standing plans and directives per checkpoint,
    # plus combat punctuation (torpedo salvos, impacts, kills).
    by_time: Dict[float, List[str]] = {}

    def note(t: float, text: str) -> None:
        by_time.setdefault(t, []).append(text)

    torp_launches: Dict[float, Dict[str, int]] = {}
    for e in events:
        et = e.get("event_type")
        t = e.get("timestamp", 0.0)
        data = e.get("data", {})
        if et in _DIGEST_DECISION_EVENTS:
            faction = data.get("faction", "?")
            label = "YOU" if faction == side else "ENEMY"
            kind = "plan" if et == "admiral_plan" else "directive"
            text = (data.get("plan") or data.get("directive") or "").strip()
            note(t, f"[{label} {kind}] {text[:300]}")
        elif et == "torpedo_launched":
            src = str(data.get("source_ship_id", e.get("ship_id") or "?"))
            bucket = torp_launches.setdefault(t, {})
            bucket[src] = bucket.get(src, 0) + 1
        elif et == "torpedo_impact":
            note(t, f"[IMPACT] torpedo hit {data.get('target_ship', e.get('ship_id'))} "
                    f"for {data.get('damage_gj', '?')} GJ")
        elif et == "penetration":
            note(t, f"[PENETRATION] {e.get('ship_id')} armor breached")
        elif et == "module_destroyed":
            note(t, f"[KILL] {e.get('ship_id')}: {data.get('module', 'module')} destroyed")
        elif et == "surrender":
            note(t, f"[SURRENDER] {e.get('ship_id')}")
    for t, per_ship in torp_launches.items():
        total = sum(per_ship.values())
        note(t, f"[SALVO] {total} torpedo(es) launched "
                f"({', '.join(f'{k}:{v}' for k, v in sorted(per_ship.items()))})")

    for t in sorted(by_time):
        lines.append(f"T+{t:.0f}s")
        for text in by_time[t]:
            lines.append(f"  {text}")

    winner = recording.get("winner")
    you_won = winner == side
    lines += [
        "",
        f"RESULT: {'YOU WON' if you_won else ('DRAW' if not winner else 'YOU LOST')}"
        f" - {recording.get('result_reason', '')}",
        f"Ships remaining - alpha: {recording.get('alpha_ships_remaining')}, "
        f"beta: {recording.get('beta_ships_remaining')}; "
        f"checkpoints: {recording.get('total_checkpoints')}",
    ]

    digest = "\n".join(lines)
    if len(digest) > max_chars:
        # Keep the header and the tail (endgame usually explains the outcome).
        head, tail = digest[:max_chars // 3], digest[-2 * max_chars // 3:]
        digest = head + "\n[... trimmed ...]\n" + tail
    return digest


DISTILL_TOOL = {
    "type": "function",
    "function": {
        "name": "record_lessons",
        "description": (
            "Record 1-2 durable lessons distilled from this battle. Each must "
            "be a concrete, transferable rule of thumb - something that would "
            "have changed a decision in THIS battle and plausibly generalizes "
            "to future ones. Never restate doctrine the commander already has; "
            "never mention specific ship names from this battle."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lessons": {
                    "type": "array",
                    "maxItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The lesson, imperative voice, <=280 chars."
                            },
                            "role": {
                                "type": "string",
                                "enum": ["admiral", "captain", "any"],
                                "description": "Which role the lesson applies to."
                            },
                        },
                        "required": ["text"],
                    },
                },
            },
            "required": ["lessons"],
        },
    },
}


def build_distill_prompt(model: str, role: str, digest: str) -> List[Dict[str, str]]:
    """Messages for the lesson-distillation call."""
    system = (
        f"You are the post-battle analyst for {model}, which fought the battle "
        f"below as {role}. Your job is continual improvement: extract at most "
        "TWO lessons worth carrying into future battles as standing notebook "
        "entries.\n\n"
        "A good lesson names a decision pattern and when to apply it "
        "(e.g. 'Hold torpedo reserves until enemy PD is measured; my opening "
        "half-salvo was fully blinded'). A bad lesson is vague ('coordinate "
        "better'), battle-specific (names enemy ships), or something doctrine "
        "already says. If the battle teaches nothing new, record zero lessons.\n\n"
        "Call record_lessons with your findings."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": digest},
    ]


def distill_lessons(
    client: Any,
    model: str,
    role: str,
    digest: str,
    analyst_model: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Ask an LLM (by default, the model being refined - it learns its own
    lessons) to distill notebook entries from a battle digest.
    """
    messages = build_distill_prompt(model, role, digest)
    tool_calls = client.decide_with_tools(
        messages, [DISTILL_TOOL], model=analyst_model or model,
    )
    lessons: List[Dict[str, str]] = []
    for tc in tool_calls:
        if tc.name != "record_lessons":
            continue
        for lesson in tc.arguments.get("lessons", []):
            text = str(lesson.get("text", "")).strip()
            if not text:
                continue
            lesson_role = lesson.get("role", role)
            if lesson_role not in VALID_ROLES:
                lesson_role = role
            lessons.append({"text": text, "role": lesson_role})
    return lessons[:2]
