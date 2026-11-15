"""Shared fixtures."""

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
    monkeypatch.setenv("AIC_WEAPON_ARCS", "0")
