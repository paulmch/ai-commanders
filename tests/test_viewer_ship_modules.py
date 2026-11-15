"""The viewer's module table (visualizer/src/shipModules.js) is the engine's.

The spectator HUD maps a destroyed weapon or engine *module* to the gun
mount, torpedo tube, PD turret or drive it silences (FittingsIndex.js, the
mirror of CombatSimulation._disable_weapon_for_module). That only works if
the viewer lists the same modules, in the same order, with the same types as
src/modules.py builds for every hull - including the variants that reuse a
base hull's layout (cruiser_torpedo -> cruiser, dreadnought_siege ->
dreadnought).
"""

import json
import re
from pathlib import Path

from src.modules import create_module_layout

ROOT = Path(__file__).resolve().parent.parent
FLEET_DATA = ROOT / "data" / "fleet_ships.json"
SHIP_MODULES_JS = ROOT / "visualizer" / "src" / "shipModules.js"


def _viewer_table() -> dict:
    """{ship_type: [(name, type), ...] | None} parsed from SHIP_MODULES."""
    text = SHIP_MODULES_JS.read_text()
    body = text[text.index("export const SHIP_MODULES"):text.index("};")]
    table: dict = {}
    current = None
    for line in body.splitlines():
        head = re.match(r"\s{2}(\w+):\s*(\[|null)", line)
        if head:
            current = head.group(1)
            table[current] = None if head.group(2) == "null" else []
            continue
        entry = re.search(r'\{\s*name:\s*"([^"]+)",\s*type:\s*"([^"]+)"\s*\}', line)
        if entry and current and table[current] is not None:
            table[current].append((entry.group(1), entry.group(2)))
    return table


def _viewer_modules(table: dict, ship_type: str):
    """Same resolution as the viewer: the hull's own list, else its base hull's."""
    return table.get(ship_type) or table.get(ship_type.split("_")[0])


def test_viewer_module_table_matches_engine_layouts():
    table = _viewer_table()
    assert table, "could not parse SHIP_MODULES"
    ships = json.loads(FLEET_DATA.read_text())["ships"]
    for ship_type in ships:
        layout = create_module_layout(ship_type, FLEET_DATA)
        engine = [(m.name, m.module_type.value) for layer in layout.layers for m in layer.modules]
        assert _viewer_modules(table, ship_type) == engine, (
            f"visualizer/src/shipModules.js drifted from src/modules.py for {ship_type}")


def test_torpedo_cruiser_is_listed_as_a_cruiser_variant():
    table = _viewer_table()
    assert "cruiser_torpedo" in table and table["cruiser_torpedo"] is None
    assert _viewer_modules(table, "cruiser_torpedo") == table["cruiser"]
