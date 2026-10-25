"""
Peer coordination (pass-4 follow-up, "orders" workstream).

In 208 captain decisions of a real 150-pt DeepSeek mirror battle nobody
called fleet_signal. Two levers, pinned here:

- the engine datalink announces primary-target SWITCHES (no LLM call): one
  ENGINE line per newly engaged enemy, with who moved, from where and how
  many of the fleet are on it now - so focus fire is visible to peers;
- the captain doctrine says WHEN a signal is worth a call (claim / split a
  target, time-on-target, PD cover, breaking off), one example each, and
  the fleet_signal tool says what the datalink already shares.
"""

from src.llm.fleet_comms import SIGNAL_TARGET, EngineDatalink, budget_inbox
from src.llm.prompts import build_captain_doctrine
from src.llm.tools import FLEET_SIGNAL_TOOL

from tests.test_fleet_comms import FLEET, Stub, make_runner


def _collect(link, runner, checkpoint):
    ids = iter(range(1000))
    return [m for m in link.collect(runner.simulation, runner.fleet_communication,
                                    {"alpha", "beta"}, checkpoint, 30.0 * checkpoint,
                                    lambda s: f"m{checkpoint}-{next(ids)}")
            if m.kind == SIGNAL_TARGET]


class TestTargetDatalink:

    def _setup(self, tmp_path):
        runner = make_runner(Stub(), tmp_path, alpha=(("corvette", "stub/dart"),
                                                      ("corvette", "stub/harpoon"),
                                                      ("corvette", "stub/spear")),
                             beta=(("destroyer", "stub/lancer"), ("destroyer", "stub/ward")))
        runner.setup_fleet_battle(FLEET)
        link = EngineDatalink()
        link.prime(runner.simulation)
        return runner, link

    def test_first_targets_are_not_switches_then_a_switch_is_one_line(self, tmp_path):
        runner, link = self._setup(tmp_path)
        sim = runner.simulation
        for sid in ("alpha_1", "alpha_2", "alpha_3"):
            sim.get_ship(sid).primary_target_id = "beta_1"
        assert _collect(link, runner, 1) == []          # first targets: FRIENDLIES shows them
        assert _collect(link, runner, 2) == []          # unchanged: silent

        for sid in ("alpha_1", "alpha_2"):
            sim.get_ship(sid).primary_target_id = "beta_2"
        out = _collect(link, runner, 3)
        assert len(out) == 1                            # one line per newly engaged enemy
        msg = out[0]
        lancer, ward = sim.get_ship("beta_1").name, sim.get_ship("beta_2").name
        dart, harpoon = sim.get_ship("alpha_1").name, sim.get_ship("alpha_2").name
        assert msg.text == (f"{dart}, {harpoon} switched to {ward} (from {lancer}) - "
                            f"2 of your ships on it now")
        assert msg.recipient_ids == ["alpha_3"]         # the movers know what they did
        assert msg.sender_role == "engine" and msg.faction == "alpha"
        assert msg.structured["ships_on_target"] == 2
        assert msg.structured["target_id"] == "beta_2"

    def test_retarget_after_a_kill_is_announced(self, tmp_path):
        runner, link = self._setup(tmp_path)
        sim = runner.simulation
        sim.get_ship("alpha_1").primary_target_id = "beta_1"
        _collect(link, runner, 1)
        sim.get_ship("alpha_1").primary_target_id = None   # the target died, none picked yet
        sim.get_ship("beta_1").is_destroyed = True
        assert _collect(link, runner, 2) == []
        sim.get_ship("alpha_1").primary_target_id = "beta_2"
        out = _collect(link, runner, 3)
        assert len(out) == 1 and "switched to" in out[0].text
        assert set(out[0].recipient_ids) == {"alpha_2", "alpha_3"}

    def test_switch_lines_rank_last_in_the_inbox(self, tmp_path):
        runner, link = self._setup(tmp_path)
        sim = runner.simulation
        sim.get_ship("alpha_1").primary_target_id = "beta_1"
        _collect(link, runner, 1)
        sim.get_ship("alpha_1").primary_target_id = "beta_2"
        target_msg = _collect(link, runner, 2)[0]
        from dataclasses import replace
        launch = replace(target_msg, msg_id="x", kind="LAUNCH")
        assert budget_inbox([target_msg, launch])[0].kind == "LAUNCH"


class TestWhenToSignal:

    def test_doctrine_gives_a_trigger_and_an_example_per_kind(self):
        system = build_captain_doctrine(
            ship_type="destroyer", fleet_data=FLEET, has_admiral=True,
            tool_names=["fleet_signal", "report_to_admiral", "set_maneuver"])
        block = system.split("COMMUNICATION (", 1)[1].split("OTHER:", 1)[0]
        assert "target\n  switches already reach your fleet as ENGINE lines" in block
        for kind in ("TARGET_CLAIM", "TOT_SYNC", "PD_COVER_REQUEST", "BREAKING_OFF"):
            line = next(l for l in block.splitlines() if l.strip().startswith(kind))
            trigger, example = line.split(":", 1)
            assert len(trigger.split()) > 3 and example.count('"') == 2, kind
        # Without peers there is no fleet_signal tool - and no fleet_signal text.
        solo = build_captain_doctrine(ship_type="destroyer", fleet_data=FLEET,
                                      tool_names=["set_maneuver"])
        assert "TARGET_CLAIM" not in solo

    def test_tool_says_what_is_shared_automatically(self):
        text = FLEET_SIGNAL_TOOL["function"]["description"]
        assert "target switches are shared automatically" in text
        for kind in ("TARGET_CLAIM", "TOT_SYNC", "PD_COVER_REQUEST", "BREAKING_OFF"):
            assert kind in text
