"""
Meta-tests: guard the quality of the test suite itself.

An audit found 28 tests that ran a scenario, printed a report, and passed
unconditionally - they exercised code but verified nothing, so a regression in
the exercised path would not fail the build. This file stops that set from
growing and records the ones still outstanding.
"""

import ast
import glob
import pathlib

import pytest

# Tests known to contain no assertion, pending real assertions being written.
# This list must only ever shrink. Adding to it requires a deliberate edit and
# should be justified in review.
KNOWN_ASSERTIONLESS = {
    ("tests/test_combat.py", "test_weapon"),
    ("tests/test_combat.py", "test_armor"),
    ("tests/test_combat.py", "test_ship_armor"),
    ("tests/test_combat_simulation.py", "test_projectile_miss"),
    ("tests/test_combat_simulation.py", "test_rotation_maneuver"),
    ("tests/test_combat_simulation.py", "test_flip_and_burn"),
    ("tests/test_combat_simulation.py", "test_evasive_jink"),
    ("tests/test_combat_simulation.py", "test_damage_resolution"),
    ("tests/test_combat_simulation.py", "test_module_damage"),
    ("tests/test_combat_simulation.py", "test_defensive_script"),
    ("tests/test_combat_simulation.py", "test_destroyer_vs_destroyer"),
    ("tests/test_combat_simulation.py", "test_corvette_vs_destroyer"),
    ("tests/test_combat_simulation.py", "test_battle_ends_on_destruction"),
    ("tests/test_command.py", "test_different_vectors"),
    ("tests/test_command.py", "test_basic_bearing"),
    ("tests/test_command.py", "test_different_bearings"),
    ("tests/test_command.py", "test_diagonal_bearing"),
    ("tests/test_damage.py", "test_spalling_enabled_creates_secondary_damage"),
    ("tests/test_damage.py", "test_angled_hit_different_path"),
    ("tests/test_two_captain_battles.py", "test_crossing_t_engagement"),
    ("tests/test_two_captain_battles.py", "test_pursuit_with_torpedoes"),
    ("tests/test_two_captain_battles.py", "test_long_chase_heat_management"),
    ("tests/test_two_captain_battles.py", "test_evasive_vs_torpedoes"),
    ("tests/test_two_captain_battles.py", "test_damaged_ship_fighting_retreat"),
    ("tests/test_two_captain_battles.py", "test_heat_critical_dogfight"),
    ("tests/test_two_captain_battles.py", "test_torpedo_saturation_attack"),
    ("tests/test_two_captain_battles.py", "test_six_corvettes_overwhelming_pd"),
}


def _assertionless_tests():
    """Every test function in tests/ that contains no assertion of any kind."""
    found = set()
    for path in sorted(glob.glob("tests/*.py")):
        tree = ast.parse(pathlib.Path(path).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test"):
                continue

            has_check = any(isinstance(n, ast.Assert) for n in ast.walk(node))
            if not has_check:
                for n in ast.walk(node):
                    # unittest-style self.assertX(...), or pytest.raises/warns blocks
                    if isinstance(n, ast.Attribute) and n.attr.startswith("assert"):
                        has_check = True
                        break
                    if isinstance(n, ast.withitem):
                        has_check = True
                        break

            if not has_check:
                found.add((path, node.name))
    return found


def test_no_new_assertionless_tests():
    """
    A test that asserts nothing cannot fail, so it provides no regression
    protection no matter how much code it executes.
    """
    current = _assertionless_tests()
    new = current - KNOWN_ASSERTIONLESS

    assert not new, (
        "These tests contain no assertions - they execute code but verify "
        "nothing:\n  "
        + "\n  ".join(f"{path}::{name}" for path, name in sorted(new))
        + "\n\nAdd real assertions rather than adding them to KNOWN_ASSERTIONLESS."
    )


def test_known_assertionless_list_is_accurate():
    """
    Keep the allowlist honest: once a test gains assertions it must be removed
    from the list, so the remaining debt is always the true remaining debt.
    """
    current = _assertionless_tests()
    fixed = KNOWN_ASSERTIONLESS - current

    assert not fixed, (
        "These tests now have assertions and must be removed from "
        "KNOWN_ASSERTIONLESS:\n  "
        + "\n  ".join(f"{path}::{name}" for path, name in sorted(fixed))
    )


def test_recording_filenames_are_filesystem_safe():
    """
    Regression: clean_name() only replaced "-" and ".", so a free-tier model id
    like "google/gemma-4-31b-it:free" produced a filename containing a colon -
    illegal on Windows and awkward on every other platform.
    """
    from src.llm.battle_recorder import create_battle_filename

    for alpha, beta in (
        ("google/gemma-4-31b-it:free", "nvidia/nemotron-3-super-120b-a12b:free"),
        ("anthropic/claude-opus-5", "x-ai/grok-4.20"),
        ("openrouter/openai/gpt-5.6-terra", "deepseek/deepseek-v4-pro"),
    ):
        name = create_battle_filename(alpha, beta)
        illegal = set(name) & set(':*?"<>|/\\')
        assert not illegal, f"{name!r} contains filesystem-hostile characters {illegal}"
        assert name.endswith(".json")
