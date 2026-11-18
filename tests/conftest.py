"""Shared fixtures."""

import os

import pytest


@pytest.fixture
def all_round_fire(monkeypatch):
    """
    Run the scenario without weapon arcs (src/weapon_arcs.py): every mount
    bears in every direction and trains instantly, as before pass 5 wave 4.

    For scripted scenarios that pin a DIFFERENT mechanic (the PD dwell model,
    torpedo triage, ablation recoil, heat coupling) and stage it with a round
    flying straight down the defender's nose - a direction most hulls' own
    structure masks from their point defence. What they pin holds for a turret
    that bears; that a bearing turret under arcs delivers exactly the same is
    pinned by tests/test_weapon_arcs.py. Simulations built inside the test
    read the flag at construction.
    """
    # AIC_ARCS_KEEP_HEAD_ON=1 turns this fixture into a no-op: it re-counts how
    # many of the marked scenarios really fail under arcs on the current mount
    # table (they were classified on the first placement).
    if os.environ.get("AIC_ARCS_KEEP_HEAD_ON") != "1":
        monkeypatch.setenv("AIC_WEAPON_ARCS", "0")


# The suite must leave the repository's data/ as it found it: a test that
# writes recordings or admiral-vision frames there (battle_runner's vision
# directory follows BattleConfig.recording_dir, default data/recordings)
# fails the session here. Point such tests at tmp_path.
def _untracked_data_files():
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "data"],
            cwd=root, capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None      # not a git checkout: nothing to compare
    return set(out.split())


def pytest_sessionstart(session):
    session.config._aic_untracked_data = _untracked_data_files()


def pytest_sessionfinish(session, exitstatus):
    before = getattr(session.config, "_aic_untracked_data", None)
    after = _untracked_data_files()
    if before is None or after is None:
        return
    leaked = sorted(after - before)
    if leaked:
        import sys
        sys.stderr.write("\nTEST SUITE LEAKED FILES INTO data/ (point the test at tmp_path):\n"
                         + "\n".join(f"  {p}" for p in leaked[:20]) + "\n")
        session.exitstatus = 1
