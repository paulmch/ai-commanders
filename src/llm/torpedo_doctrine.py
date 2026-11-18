"""
Torpedo launch-profile and datalink facts for commanders (pass 5).

State, not tactics: what each launch profile costs and buys, and what the
fleet datalink does with rounds whose target is lost. Every number is either
an engine constant (src/torpedo.py, src/torpedo_link.py) or a measurement in
tests/fixtures/torpedo_link_facts.json (scripts/torpedo_link_facts.py),
bound by tests/test_doctrine_accuracy.py::TestTorpedoLinkDoctrine.
"""

from __future__ import annotations

try:
    from ..torpedo import RESERVE_CRUISE_CLOSING_KPS, RESERVE_TERMINAL_WINDOW_S
except ImportError:  # pragma: no cover - script-style imports
    from torpedo import RESERVE_CRUISE_CLOSING_KPS, RESERVE_TERMINAL_WINDOW_S

# Measured (tests/fixtures/torpedo_link_facts.json): one round from 500 km at
# a battlecruiser holding course, at a frigate running away at 3 g, and the
# reach of a reassignment 200 km out; seeded rule-based battles with the link.
FACTS = {
    "sprint": {"tof_500_s": 72, "gj_500": 22, "runner_ff_gj": 17, "reach_200_km": 22},
    "reserve": {"tof_500_s": 81, "gj_500": 23, "runner_ff_gj": 15, "reach_200_km": 112},
    "battles": {"pct_orphaned": 59, "pct_reassigned": 72, "pct_reassigned_hit": 51},
}

_S, _R, _B = FACTS["sprint"], FACTS["reserve"], FACTS["battles"]

PROFILE_FACT = (
    f"Launch profile (default sprint). sprint: spends the tank early on closing speed. "
    f"reserve: cruises at {RESERVE_CRUISE_CLOSING_KPS:.0f} km/s closure and holds its "
    f"delta-v until the last {RESERVE_TERMINAL_WINDOW_S:.0f} s, then burns it. Measured "
    f"from 500 km: sprint lands in ~{_S['tof_500_s']} s, reserve ~{_R['tof_500_s']} s, "
    f"both ~{_S['gj_500']}-{_R['gj_500']} GJ on a hull holding course, "
    f"{_S['runner_ff_gj']} vs {_R['runner_ff_gj']} GJ on a 3 g hull running away; a round "
    f"whose target is lost 200 km out can still be swung ~{_S['reach_200_km']} km abeam "
    f"(sprint) or ~{_R['reach_200_km']} km (reserve)."
)

LINK_FACT = (
    "Fleet datalink: rounds whose target dies or becomes a dying hulk are reassigned as a "
    "group to the hulls that still need rounds (PD soak + rounds to kill), but only where "
    "a round's delta-v still reaches; reserve-profile rounds beyond what their target "
    "needs are diverted before it dies, sprint rounds never. Measured in seeded "
    f"rule-based battles (sprint): {_B['pct_orphaned']}% of rounds lost their target in "
    f"flight, {_B['pct_reassigned']}% of those were reassigned, "
    f"{_B['pct_reassigned_hit']}% of the reassigned hit."
)
