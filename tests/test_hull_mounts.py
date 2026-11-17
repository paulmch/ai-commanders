"""data/hull_mounts.json: schema, slot coverage and mask sizes.

The file is generated from the viewer's hull kit by
``visualizer/scripts/export_hull_mounts.mjs`` (a node test regenerates and
compares it); Python only reads it, so this pins what the engine relies on.
"""
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MOUNTS = json.loads((ROOT / "data" / "hull_mounts.json").read_text())
FLEET = json.loads((ROOT / "data" / "fleet_ships.json").read_text())
GRID = MOUNTS["grid"]
CELLS = GRID["az_cells"] * GRID["el_cells"]


def _bits(hex_mask):
    raw = bytes.fromhex(hex_mask)
    return [(raw[k >> 3] >> (7 - (k & 7))) & 1 for k in range(CELLS)]


def _cell(az, el):
    i = int(math.floor((az + 180.0) / GRID["az_step_deg"])) % GRID["az_cells"]
    j = max(0, min(GRID["el_cells"] - 1, int(math.floor((el + 90.0) / GRID["el_step_deg"]))))
    return j * GRID["az_cells"] + i


def _open_fraction(bits):
    total = opened = 0.0
    for j in range(GRID["el_cells"]):
        w = math.cos(math.radians(-90 + (j + 0.5) * GRID["el_step_deg"]))
        for i in range(GRID["az_cells"]):
            total += w
            if not bits[j * GRID["az_cells"] + i]:
                opened += w
    return opened / total


def test_header():
    assert MOUNTS["schema"] == 1
    assert MOUNTS["frame"].startswith("x forward, y port, z dorsal")
    assert (GRID["az_step_deg"], GRID["el_step_deg"]) == (5, 5)
    assert (GRID["az_cells"], GRID["el_cells"]) == (72, 36)
    assert set(MOUNTS["hulls"]) == set(FLEET["ships"])


@pytest.mark.parametrize("hull", sorted(FLEET["ships"]))
def test_every_slot_has_a_mount_and_no_extra(hull):
    ship = FLEET["ships"][hull]
    entry = MOUNTS["hulls"][hull]
    assert entry["length_m"] == ship["hull"]["length_m"]
    # same slots, in the data file's weapons order (the `tur` trace field uses it)
    assert [m["slot"] for m in entry["mounts"]] == [w["slot"] for w in ship["weapons"]]
    for m, w in zip(entry["mounts"], ship["weapons"]):
        spec = FLEET["weapon_types"][w["type"]]
        if w["type"] == "pd_laser":
            want = "pd"
        elif w["type"] == "torpedo_launcher":
            want = "launcher"
        elif not spec.get("is_turreted", True):
            want = "spinal"
        else:
            want = "gun"
        assert m["kind"] == want, (hull, m["slot"])
        assert m["type"] == w["type"]


@pytest.mark.parametrize("hull", sorted(FLEET["ships"]))
def test_masks(hull):
    entry = MOUNTS["hulls"][hull]
    for m in entry["mounts"]:
        where = (hull, m["slot"])
        assert len(m["blocked"]) == CELLS // 4 == 648, where
        bits = _bits(m["blocked"])
        assert _open_fraction(bits) == pytest.approx(m["open_fraction"], abs=1e-4), where
        assert len(m["pos"]) == 3 and len(m["axis"]) == 3, where
        assert math.sqrt(sum(a * a for a in m["axis"])) == pytest.approx(1.0, abs=1e-3), where
        assert all(abs(p) <= entry["length_m"] for p in m["pos"]), where
        az = math.degrees(math.atan2(m["axis"][1], m["axis"][0]))
        el = math.degrees(math.asin(max(-1.0, min(1.0, m["axis"][2]))))
        if m["kind"] == "launcher":
            assert not any(bits), where
        elif m["kind"] == "spinal":
            assert m["axis"] == [1, 0, 0]
            gimbal = FLEET["weapon_types"][m["type"]]["pivot_range_deg"]
            assert m["gimbal_deg"] == gimbal
            assert not bits[_cell(0.1, 0.1)], where
            assert bits[_cell(0.0, gimbal + 6)] and bits[_cell(gimbal + 6, 0.0)], where
            assert bits[_cell(180.0, 0.0)], where
        else:
            # open straight out along the mount axis, blocked back through the hull
            assert not bits[_cell(az, el - 0.01)], where
            assert bits[_cell(az + 180.0, -el + 0.01)], where
            assert 0.3 < m["open_fraction"] <= 0.59, where  # 10 deg gun depression: (1 + sin 10 deg) / 2
