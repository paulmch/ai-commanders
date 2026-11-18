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
# a battlecruiser holding course from a launcher at rest / closing 3 km/s /
# opening 1 km/s, at a frigate running away at 3 g, and the gated, flown reach
# of a reassignment 200 km out; seeded rule-based battles (link on).
FACTS = {
    "sprint": {"tof_500_rest_s": 76, "gj_500_rest": 22, "tof_500_closing3_s": 61,
               "gj_500_closing3": 33, "tof_500_opening1_s": 81, "gj_500_opening1": 19,
               "runner_ff_gj": 17, "reach_200_km": 16, "reach_200_closing3_km": 11},
    "reserve": {"tof_500_rest_s": 85, "gj_500_rest": 22, "tof_500_closing3_s": 62,
                "gj_500_closing3": 33, "tof_500_opening1_s": 91, "gj_500_opening1": 19,
                "runner_ff_gj": 14, "reach_200_km": 103, "reach_200_closing3_km": 57},
    "launch": {"ahead_s": 4, "half_round_s": 7},
    "battles": {"pct_orphaned": 55, "pct_reassigned": 65, "pct_reassigned_hit": 55,
                "pct_pd_blinded_sprint": 25, "pct_pd_blinded_reserve": 35,
                "pct_diverted_hit_reserve": 76},
}

_S, _R, _B = FACTS["sprint"], FACTS["reserve"], FACTS["battles"]
_L = FACTS["launch"]

# One fact, no advice: how long a round's torch stays cold after launch
# (src/torpedo_launch.py; measured from a coasting launcher,
# scripts/launch_sequence_facts.py via scripts/torpedo_link_facts.py).
LAUNCH_FACT = (
    f"Launch sequence: a round leaves its rack or cell and drifts clear on its thrusters "
    f"with its torch cold; measured from a coasting launcher the torch lights "
    f"~{_L['ahead_s']} s after launch when the round needs at most a quarter turn onto its "
    f"target and ~{_L['half_round_s']} s when it must turn about first (a corvette rail "
    f"round fired astern, a torpedo-cruiser cell on the far side from the target); the "
    f"flight times quoted here include it."
)

PROFILE_FACT = (
    f"Launch profile (default sprint). sprint: spends the tank early on closing speed. "
    f"reserve: cruises at {RESERVE_CRUISE_CLOSING_KPS:.0f} km/s closure and holds its "
    f"delta-v until the last {RESERVE_TERMINAL_WINDOW_S:.0f} s (longer if the held tank "
    f"needs longer to burn), then burns it. Measured from 500 km on a hull holding course: "
    f"launcher at rest - sprint ~{_S['tof_500_rest_s']} s ~{_S['gj_500_rest']} GJ, reserve "
    f"~{_R['tof_500_rest_s']} s ~{_R['gj_500_rest']} GJ; launcher closing 3 km/s - "
    f"~{_S['tof_500_closing3_s']} s vs ~{_R['tof_500_closing3_s']} s, "
    f"~{_S['gj_500_closing3']} GJ both; launcher opening 1 km/s - ~{_S['tof_500_opening1_s']} s "
    f"vs ~{_R['tof_500_opening1_s']} s, ~{_S['gj_500_opening1']} GJ both; on a 3 g hull "
    f"running away {_S['runner_ff_gj']} vs {_R['runner_ff_gj']} GJ. A round whose target is "
    f"lost 200 km out can still be swung ~{_S['reach_200_km']} km abeam (sprint) or "
    f"~{_R['reach_200_km']} km (reserve) from a launcher at rest, "
    f"~{_S['reach_200_closing3_km']} vs ~{_R['reach_200_closing3_km']} km from one closing "
    f"3 km/s (its tank is already burning). Point defence blinds more of the slower reserve "
    f"rounds: {_B['pct_pd_blinded_reserve']}% of "
    f"those launched in seeded rule-based battles against {_B['pct_pd_blinded_sprint']}% of "
    f"sprint rounds."
)

LINK_FACT = (
    "Fleet datalink: rounds whose target dies, becomes a dying hulk or surrenders are "
    "reassigned as a "
    "group to the hulls that still need rounds (PD soak + rounds to kill + 1, minus what "
    "already arrives first), but only where a round's delta-v and burn time still reach; "
    "reserve-profile rounds beyond what their target needs are diverted before it dies, "
    "sprint rounds never. Measured in seeded rule-based battles (sprint): "
    f"{_B['pct_orphaned']}% of rounds lost their target in flight, {_B['pct_reassigned']}% "
    f"of those were reassigned, {_B['pct_reassigned_hit']}% of the reassigned hit; "
    f"{_B['pct_diverted_hit_reserve']}% of diverted reserve rounds hit."
)
