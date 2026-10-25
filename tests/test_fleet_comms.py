"""
Fleet comms: routing, captain reports, the engine datalink and recording.

Pins the pass-4 comms rework (src/llm/communication.py, src/llm/fleet_comms.py,
the comms tools and the runner's shared delivery):
- friendly traffic stays friendly, the enemy channel is separate, recipients
  resolve by ship id or name and dead ships are skipped;
- a captain's DEVIATE at checkpoint N reaches its admiral at N+1 with an
  ordered-vs-executed compliance line, and a silent captain still gets one;
- engine auto-signals (launches with ETA, kills/losses, PD losses) land in
  the peers' inbox; every loop (sync, async, HTTP) delivers;
- comms_message / comms_delivered / captain_report events carry the schema.
No network: every LLM call goes to an in-test stub.
"""

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.llm.battle_runner import BattleConfig, LLMBattleRunner
from src.llm.captain import LLMCaptain, LLMCaptainConfig
from src.llm.client import ToolCall
from src.llm.communication import (
    CHANNEL_DIRECT, CHANNEL_ENEMY, CHANNEL_FLEET_NET, INBOX_MAX_ENEMY_ITEMS, INBOX_MAX_ITEMS,
    MAX_DISCUSSION_EXCHANGES, MAX_ENEMY_MESSAGES_PER_DECISION, MAX_FLEET_SIGNALS_PER_DECISION,
    REPORT_INTENT_CHARS, REPORT_REASON_CHARS, SIGNAL_TEXT_CHARS, CommsMessage,
    FleetCommunicationChannel,
)
from src.llm.fleet_comms import EngineDatalink, budget_inbox, ordered_maneuvers
from src.llm.fleet_config import BattleFleetConfig

FLEET = json.loads((Path(__file__).parent.parent / "data" / "fleet_ships.json").read_text())

COMMS_MESSAGE_KEYS = {"msg_id", "channel", "kind", "sender_id", "sender_role", "sender_name",
                      "faction", "recipient_spec", "recipient_ids", "text", "structured",
                      "checkpoint", "sent_t", "deliver_at_t"}
COMMS_DELIVERED_KEYS = {"msg_id", "recipient_id", "delivered_t", "checkpoint"}
CAPTAIN_REPORT_KEYS = {"report_id", "faction", "checkpoint", "captain_name", "ship_name", "status",
                       "reason", "intent", "contact", "request", "compliance"}
COMPLIANCE_KEYS = {"ordered_maneuver", "executed_maneuver", "ordered_target", "actual_target",
                   "ordered_salvo", "launched"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def channel_fixture():
    channel = FleetCommunicationChannel()
    channel.register_ship("alpha_1", "Cpt A", "TIS Dart-1", "alpha")
    channel.register_ship("alpha_2", "Cpt B", "TIS Dart-2", "alpha")
    channel.register_ship("alpha_3", "Cpt C", "TIS Lancer-1", "alpha")
    channel.register_ship("beta_1", "Cpt D", "OCS Lancer-1", "beta")
    channel.register_ship("beta_2", "Cpt E", "OCS Lancer-2", "beta")
    channel.register_admiral("alpha", "Admiral A")
    return channel


def captain_fixture(has_admiral=True, ship_type="corvette"):
    captain = LLMCaptain(LLMCaptainConfig(name="Cpt", ship_name="TIS Dart-1", ship_type=ship_type,
                                          fleet_data=FLEET), client=Mock())
    captain.has_admiral = has_admiral
    captain.faction = "alpha"
    return captain


def call(captain, name, **args):
    return captain._execute_tool(ToolCall(id="t", name=name, arguments=args),
                                 SimpleNamespace(current_time=30.0), "alpha_1")


class Stub:
    """
    Answers every LLM call of a fleet battle. Captains are told apart by
    their model id; `captain(ctx)` returns [(tool, args), ...] for one turn.
    """
    session_id = None

    def __init__(self, captain=None, order=None):
        self.captain = captain or (lambda ctx: [("set_maneuver", {"maneuver_type": "INTERCEPT",
                                                                  "throttle": 0.3})])
        self.order = order or (lambda ship: {"order_text": "INTERCEPT the nearest enemy.",
                                             "priority": "NORMAL"})
        self.runner = None
        self.reports = {}       # (faction, checkpoint) -> admiral.captain_reports at directive time
        self.inboxes = {}       # (model, checkpoint) -> captain.inbox at decision time
        self.turns = {}         # (model, checkpoint) -> calls this checkpoint
        self.answers = []

    def decide_with_tools(self, messages, tools, model=None, temperature=None, tool_choice="auto"):
        names = {t["function"]["name"] for t in tools}
        text = messages[-1]["content"] if isinstance(messages[-1]["content"], str) else ""
        cp = self.runner.checkpoint_count
        if names == {"issue_order"}:
            ship = re.search(r"Issue tactical order to (.+?)\. You MUST", text).group(1)
            return [ToolCall(id="o", name="issue_order", arguments={"ship_name": ship, **self.order(ship)})]
        if "set_fleet_directive" in names:
            for faction in ("alpha", "beta"):
                admiral = getattr(self.runner, f"{faction}_admiral")
                if admiral is not None and admiral.config.model == model:
                    self.reports[(faction, cp)] = [dict(r) for r in admiral.captain_reports]
            return [ToolCall(id="d", name="set_fleet_directive", arguments={"directive": "Test directive."})]
        captain = next(c for c in self.runner._all_captains() if c.config.model == model)
        self.inboxes.setdefault((model, cp), list(captain.inbox))
        turn = self.turns[(model, cp)] = self.turns.get((model, cp), 0) + 1
        ctx = SimpleNamespace(model=model, cp=cp, turn=turn, names=names, captain=captain,
                              runner=self.runner, messages=messages)
        return [ToolCall(id=f"c{i}", name=n, arguments=a)
                for i, (n, a) in enumerate(self.captain(ctx)) if n in names]

    def complete(self, messages, model=None):
        self.answers.append(messages)
        return SimpleNamespace(content=f"Answer {len(self.answers)}: hold your course.")


def make_runner(stub, tmp_path, *, alpha=(("corvette", "stub/dart"), ("corvette", "stub/harpoon")),
                beta=(("destroyer", "stub/lancer"),), checkpoints=2, admirals=True, distance_km=300.0):
    def fleet(ships, faction):
        data = {"ships": [{"ship_type": t, "model": m} for t, m in ships]}
        if admirals:
            data["admiral"] = {"model": f"stub/adm-{faction}"}
        return data
    fc = BattleFleetConfig.from_dict({
        "battle_name": "comms", "time_limit_s": 600.0, "decision_interval_s": 30.0,
        "initial_distance_km": distance_km,
        "alpha_fleet": fleet(alpha, "alpha"), "beta_fleet": fleet(beta, "beta")})
    config = BattleConfig(initial_distance_km=distance_km, time_limit_s=600.0,
                          max_checkpoints=checkpoints, verbose=False, personality_selection=False,
                          record_battle=True, recording_dir=str(tmp_path), parallel_llm=False, seed=11)
    dummy = LLMCaptainConfig(name="-", ship_name="-", model="stub/x")
    runner = LLMBattleRunner(config, dummy, dummy, stub, fc)
    stub.runner = runner
    return runner


def events(runner, kind):
    return [e for e in runner.recorder.events if e.event_type == kind]


# ---------------------------------------------------------------------------
# Routing (communication.FleetCommunicationChannel)
# ---------------------------------------------------------------------------

class TestRouting:

    def test_friendly_broadcast_stays_friendly(self):
        res = channel_fixture().resolve("alpha_1", "ALL", friendly_only=True)
        assert res.channel == CHANNEL_FLEET_NET
        assert res.recipient_ids == ["alpha_2", "alpha_3"]  # never the sender, never beta

    def test_legacy_all_friendlies_on_send_message_is_not_sent_to_the_enemy(self):
        """battle_runner used to drop the enum: ALL_FRIENDLIES went to the ENEMY."""
        res = channel_fixture().resolve("alpha_1", "ALL_FRIENDLIES", friendly_only=False)
        assert res.channel == CHANNEL_FLEET_NET
        assert all(r.startswith("alpha") for r in res.recipient_ids)

    def test_enemy_channel_is_separate(self):
        channel = channel_fixture()
        for spec in ("ALL_ENEMIES", "ALL"):  # send_message's ALL is the open channel
            res = channel.resolve("alpha_1", spec, friendly_only=False)
            assert res.channel == CHANNEL_ENEMY and res.recipient_ids == ["beta_1", "beta_2"]
        assert channel.resolve("alpha_1", "ALL_ENEMIES", friendly_only=True).error

    @pytest.mark.parametrize("spec", ["alpha_2", "TIS Dart-2", "tis dart-2", "Dart-2", "DART-2"])
    def test_specific_resolves_by_id_full_name_or_short_name(self, spec):
        res = channel_fixture().resolve("alpha_1", spec, friendly_only=True)
        assert (res.channel, res.recipient_ids, res.error) == (CHANNEL_DIRECT, ["alpha_2"], None)

    def test_name_collision_prefers_the_tools_side(self):
        """Battle B had a friendly TIS Lancer-1 and an enemy OCS Lancer-1."""
        channel = channel_fixture()
        assert channel.resolve("alpha_1", "Lancer-1", friendly_only=True).recipient_ids == ["alpha_3"]
        enemy = channel.resolve("alpha_1", "Lancer-1", friendly_only=False)
        assert (enemy.channel, enemy.recipient_ids) == (CHANNEL_ENEMY, ["beta_1"])

    def test_fleet_signal_refuses_an_enemy_ship(self):
        res = channel_fixture().resolve("alpha_1", "OCS Lancer-2", friendly_only=True)
        assert res.recipient_ids == [] and "ENEMY" in res.error

    def test_send_message_to_a_friendly_ship_stays_direct(self):
        res = channel_fixture().resolve("alpha_1", "TIS Dart-2", friendly_only=False)
        assert (res.channel, res.recipient_ids) == (CHANNEL_DIRECT, ["alpha_2"])

    def test_admiral_and_unknown_names(self):
        channel = channel_fixture()
        res = channel.resolve("alpha_1", "ADMIRAL", friendly_only=True)
        assert res.recipient_ids == ["alpha_admiral"]
        assert "no Admiral" in channel.resolve("beta_1", "ADMIRAL", friendly_only=True).error
        error = channel.resolve("alpha_1", "Nonesuch", friendly_only=True).error
        assert "Nonesuch" in error and "TIS Dart-2" in error  # tells the captain the valid names

    def test_dead_recipient_refused_at_send_and_skipped_at_delivery(self):
        channel = channel_fixture()
        dead = {"alpha_2"}
        alive = lambda sid: sid not in dead
        assert "no longer" in channel.resolve("alpha_1", "Dart-2", friendly_only=True, alive=alive).error
        assert channel.resolve("alpha_1", "ALL", friendly_only=True, alive=alive).recipient_ids == ["alpha_3"]

        msg = channel.post(CommsMessage(
            msg_id="", channel=CHANNEL_FLEET_NET, kind="FREE", sender_id="alpha_1",
            sender_role="captain", sender_name="TIS Dart-1", faction="alpha", recipient_spec="ALL",
            recipient_ids=["alpha_2", "alpha_3"], text="x", sent_t=30, deliver_at_t=60))
        assert channel.deliver_due(30.0, ["alpha_3"]) == {"alpha_3": []}  # not due yet
        out = channel.deliver_due(60.0, ["alpha_3"])                      # alpha_2 died meanwhile
        assert out == {"alpha_3": [msg]}
        assert channel.pending_messages == []                              # nothing lingers


# ---------------------------------------------------------------------------
# Captain comms tools
# ---------------------------------------------------------------------------

class TestCaptainTools:

    def test_fleet_signal_cap_and_truncation(self):
        captain = captain_fixture()
        for i in range(MAX_FLEET_SIGNALS_PER_DECISION + 1):
            assert call(captain, "fleet_signal", kind="TOT_SYNC", to="ALL", target="OCS Lancer-1",
                        eta_s=40, text="x" * 400) is None
        assert len(captain.outbox) == MAX_FLEET_SIGNALS_PER_DECISION
        assert all(len(m["text"]) <= SIGNAL_TEXT_CHARS for m in captain.outbox)
        assert captain.outbox[0]["eta_s"] == 40.0
        assert any("only" in e for e in captain.pending_tool_errors)

    def test_send_message_keeps_several_messages(self):
        """The single pending_message slot overwrote all but the last."""
        captain = captain_fixture()
        for text in ("first", "second", "third"):
            call(captain, "send_message", message=text)
        assert [m["text"] for m in captain.outbox] == ["first", "second"][:MAX_ENEMY_MESSAGES_PER_DECISION]
        assert len(captain.pending_tool_errors) == 1

    def test_report_to_admiral_normalises_and_clips(self):
        captain = captain_fixture()
        call(captain, "report_to_admiral", status="deviate", reason="r" * 500, intent="i" * 500,
             contact={"ship": "OCS Lancer-1", "assessment": "pd_down", "note": "n"},
             request={"kind": "PD_COVER", "text": "cover me"})
        report = captain.order_response
        assert report["status"] == "DEVIATE"
        assert len(report["reason"]) <= REPORT_REASON_CHARS and len(report["intent"]) <= REPORT_INTENT_CHARS
        assert report["contact"]["assessment"] == "PD_DOWN"
        assert report["request"] == {"kind": "PD_COVER", "text": "cover me"}

    def test_deviate_without_reason_is_filed_with_a_tool_error(self):
        captain = captain_fixture()
        call(captain, "report_to_admiral", status="UNABLE")
        assert captain.order_response["status"] == "UNABLE"
        assert any("needs a reason" in e for e in captain.pending_tool_errors)

    def test_respond_to_orders_is_an_alias(self):
        captain = captain_fixture()
        call(captain, "respond_to_orders", response_type="DEVIATE", deviation_reason="stale intel",
             acknowledgment_note="evading")
        report = captain.order_response
        assert (report["status"], report["reason"], report["intent"]) == ("DEVIATE", "stale intel", "evading")
        assert report["type"] == "DEVIATE" and report["deviation_reason"] == "stale intel"

    def test_report_without_an_admiral_is_an_error(self):
        captain = captain_fixture(has_admiral=False)
        call(captain, "report_to_admiral", status="ACK")
        assert captain.order_response is None and captain.pending_tool_errors

    def test_tools_offered(self):
        names = lambda c: {t["function"]["name"] for t in c.get_tools_for_context()}
        admiral_fleet = names(captain_fixture())
        assert {"report_to_admiral", "fleet_signal", "discuss_with_admiral"} <= admiral_fleet
        assert "respond_to_orders" not in admiral_fleet
        lone = captain_fixture(has_admiral=False)
        assert not {"report_to_admiral", "fleet_signal"} & names(lone)
        lone.fleet_peer_names = ["TIS Dart-2"]
        assert "fleet_signal" in names(lone) and "report_to_admiral" not in names(lone)

    def test_tool_descriptions_quote_the_enforced_limits(self):
        """Prompt numbers are bound to the constants the code enforces."""
        from src.llm.admiral_tools import DISCUSS_WITH_ADMIRAL_TOOL
        from src.llm.tools import CAPTAIN_TOOLS_BASE, FLEET_SIGNAL_TOOL, REPORT_TO_ADMIRAL_TOOL
        signal = json.dumps(FLEET_SIGNAL_TOOL)
        assert f"Max {MAX_FLEET_SIGNALS_PER_DECISION} per decision" in signal
        assert f"Max {SIGNAL_TEXT_CHARS} chars" in signal
        report = json.dumps(REPORT_TO_ADMIRAL_TOOL)
        assert f"max {REPORT_REASON_CHARS} chars" in report and f"max {REPORT_INTENT_CHARS} chars" in report
        send = json.dumps(next(t for t in CAPTAIN_TOOLS_BASE if t["function"]["name"] == "send_message"))
        assert f"Max {MAX_ENEMY_MESSAGES_PER_DECISION} per decision" in send
        assert f"At most {MAX_DISCUSSION_EXCHANGES} exchanges" in DISCUSS_WITH_ADMIRAL_TOOL["function"]["description"]
        assert captain_fixture().max_discussion_exchanges == MAX_DISCUSSION_EXCHANGES
        # 30 s is an engine setting, never a comms claim: say "next checkpoint".
        assert "30" not in signal and "30" not in report


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_ordered_maneuvers_reads_structure_then_prose():
    order = lambda text, **kw: SimpleNamespace(order_text=text, heading=None, **kw)
    assert ordered_maneuvers(order("INTERCEPT Lancer-3, EVADE if torpedoes inbound")) == ["INTERCEPT", "EVADE"]
    assert ordered_maneuvers(order("Intercept the nearest enemy.")) == ["INTERCEPT"]
    assert ordered_maneuvers(order("Maintain formation and keep heading toward the fleet.")) == []
    assert ordered_maneuvers(SimpleNamespace(order_text="x", heading={"x": 1})) == ["HEADING"]


def test_inbox_budget_puts_llm_traffic_first_and_caps_enemy_chatter():
    def msg(i, channel, role="captain", kind="FREE"):
        return CommsMessage(msg_id=f"m{i}", channel=channel, kind=kind, sender_id="x",
                            sender_role=role, sender_name="x", faction="alpha",
                            recipient_spec="ALL", recipient_ids=["alpha_1"], text=str(i))
    batch = ([msg(i, CHANNEL_FLEET_NET, "engine", "LAUNCH") for i in range(4)]
             + [msg(10 + i, CHANNEL_ENEMY) for i in range(4)]
             + [msg(20, CHANNEL_FLEET_NET), msg(21, CHANNEL_DIRECT),
                msg(22, CHANNEL_FLEET_NET, "engine", "KILL")])
    chosen = budget_inbox(batch)
    assert len(chosen) == INBOX_MAX_ITEMS
    assert [m.msg_id for m in chosen[:2]] == ["m21", "m20"]       # direct, then fleet net
    assert sum(m.channel == CHANNEL_ENEMY for m in chosen) == INBOX_MAX_ENEMY_ITEMS
    assert chosen[4].kind == "KILL"                                # datalink deaths first


# ---------------------------------------------------------------------------
# Runner integration: reports, inbox, datalink, recording
# ---------------------------------------------------------------------------

def deviate_script(ctx):
    if ctx.model == "stub/dart" and ctx.cp == 1:
        return [("set_maneuver", {"maneuver_type": "EVADE", "throttle": 1.0}),
                ("report_to_admiral", {"status": "DEVIATE",
                                       "reason": "OCS Lancer is SEPARATING at 3 km/s - holding "
                                                 "torpedoes, EVADE until it turns."})]
    if ctx.model == "stub/harpoon":
        return []  # says and does nothing: an implicit ACK the engine can audit
    return [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.3})]


def salvo_order(ship):
    return {"order_text": "INTERCEPT OCS Lancer and launch your salvo.", "priority": "HIGH",
            "torpedo_salvo": 2, "torpedo_target": "OCS Lancer"}


class TestReports:

    def test_deviate_at_cp1_reaches_the_admiral_at_cp2_with_compliance(self, tmp_path):
        stub = Stub(captain=deviate_script, order=salvo_order)
        runner = make_runner(stub, tmp_path)
        runner.run_fleet_battle(FLEET)

        assert stub.reports[("alpha", 1)] == []                     # nothing filed before CP1
        reports = {r["ship_id"]: r for r in stub.reports[("alpha", 2)]}
        dart = reports["alpha_1"]
        assert CAPTAIN_REPORT_KEYS <= set(dart)
        assert dart["status"] == "DEVIATE" and "SEPARATING" in dart["reason"]
        assert dart["checkpoint"] == 1 and dart["implicit"] is False
        comp = dart["compliance"]
        assert COMPLIANCE_KEYS <= set(comp) and comp["final"] is True
        assert (comp["ordered_maneuver"], comp["executed_maneuver"]) == ("INTERCEPT", "EVADE")
        assert (comp["ordered_salvo"], comp["launched"]) == (2, 0)
        assert comp["ordered_target"] == "OCS Lancer"
        assert any(d.startswith("maneuver") for d in comp["deviations"])
        assert any(d.startswith("salvo") for d in comp["deviations"])

    def test_silent_captain_still_reports_compliance(self, tmp_path):
        """R8: an admiral was never told its orders were not executed."""
        stub = Stub(captain=deviate_script, order=salvo_order)
        runner = make_runner(stub, tmp_path)
        runner.run_fleet_battle(FLEET)
        harpoon = {r["ship_id"]: r for r in stub.reports[("alpha", 2)]}["alpha_2"]
        assert (harpoon["status"], harpoon["implicit"]) == ("ACK", True)
        assert harpoon["compliance"]["executed_maneuver"] == "NONE"
        assert "salvo: ordered 2, launched 0" in harpoon["compliance"]["deviations"]

    def test_reports_are_recorded_and_acknowledgment_is_not_null(self, tmp_path):
        stub = Stub(captain=deviate_script, order=salvo_order)
        runner = make_runner(stub, tmp_path)
        runner.run_fleet_battle(FLEET)

        decisions = {(e.timestamp, e.ship_id): e for e in events(runner, "captain_decision")}
        assert decisions[(30.0, "alpha_1")].data["acknowledgment"].startswith("DEVIATE")
        assert decisions[(30.0, "alpha_2")].data["acknowledgment"] == "ACK (implicit)"
        assert all(e.data["acknowledgment"] for e in decisions.values())

        recorded = {(e.timestamp, e.ship_id): e for e in events(runner, "captain_report")}
        report = recorded[(30.0, "alpha_1")]
        assert CAPTAIN_REPORT_KEYS <= set(report.data)
        assert report.data["decision_event_id"] == decisions[(30.0, "alpha_1")].event_id
        order = next(e for e in events(runner, "admiral_order") if e.timestamp == 30.0 and e.ship_id == "alpha_1")
        assert report.data["answers_order_event_id"] == order.event_id

        delivered = [e for e in events(runner, "comms_delivered") if e.data["msg_id"] == report.data["report_id"]]
        assert len(delivered) == 1
        assert COMMS_DELIVERED_KEYS <= set(delivered[0].data)
        assert (delivered[0].timestamp, delivered[0].ship_id) == (60.0, "alpha_admiral")
        assert delivered[0].data["compliance"]["final"] is True

    def test_reports_are_replaced_each_checkpoint(self, tmp_path):
        stub = Stub(captain=deviate_script, order=salvo_order)
        runner = make_runner(stub, tmp_path, checkpoints=3)
        runner.run_fleet_battle(FLEET)
        cp3 = {r["ship_id"]: r for r in stub.reports[("alpha", 3)]}
        assert cp3["alpha_1"]["checkpoint"] == 2 and cp3["alpha_1"]["status"] == "ACK"
        assert len(stub.reports[("alpha", 3)]) == 2


class TestFleetNet:

    @staticmethod
    def signal_script(ctx):
        calls = [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.3})]
        if ctx.model == "stub/dart" and ctx.cp == 1:
            calls += [("fleet_signal", {"kind": "TARGET_CLAIM", "to": "ALL", "target": "OCS Lancer",
                                        "text": "I take Lancer - hold your rounds."}),
                      ("send_message", {"message": "Strike your colours, Lancer."})]
        if ctx.model == "stub/dart" and ctx.cp == 1:
            calls.append(("launch_torpedo", {"count": 1}))
        return calls

    def test_fleet_signal_reaches_peers_next_checkpoint_and_never_the_enemy(self, tmp_path):
        stub = Stub(captain=self.signal_script)
        runner = make_runner(stub, tmp_path)
        runner.run_fleet_battle(FLEET)

        assert stub.inboxes[("stub/harpoon", 1)] == []
        harpoon = stub.inboxes[("stub/harpoon", 2)]
        claim = next(i for i in harpoon if i["kind"] == "TARGET_CLAIM")
        assert claim["channel"] == CHANNEL_FLEET_NET and claim["sender_name"] == "TIS Dart"
        assert claim["structured"]["target_id"] == "beta_1"
        assert claim["delivered_t"] == 60.0 and COMMS_MESSAGE_KEYS <= set(claim)
        lancer = stub.inboxes[("stub/lancer", 2)]
        assert not [i for i in lancer if i["channel"] != CHANNEL_ENEMY]   # no alpha fleet traffic
        hail = next(i for i in lancer if i["channel"] == CHANNEL_ENEMY)
        assert hail["text"] == "Strike your colours, Lancer."
        assert not [i for i in harpoon if i["channel"] == CHANNEL_ENEMY]  # own taunt not echoed

    def test_datalink_announces_a_friendly_launch_with_eta(self, tmp_path):
        stub = Stub(captain=self.signal_script)
        runner = make_runner(stub, tmp_path)
        runner.run_fleet_battle(FLEET)
        launch = next(i for i in stub.inboxes[("stub/harpoon", 2)] if i["kind"] == "LAUNCH")
        assert launch["sender_role"] == "engine"
        assert launch["structured"]["subject_id"] == "alpha_1" and launch["structured"]["rounds"] == 1
        assert launch["structured"]["eta_s_min"] > 0 and "ETA" in launch["text"]
        assert not [i for i in stub.inboxes[("stub/dart", 2)] if i["kind"] == "LAUNCH"]  # not to the launcher

    def test_comms_events_carry_the_schema(self, tmp_path):
        stub = Stub(captain=self.signal_script)
        runner = make_runner(stub, tmp_path)
        runner.run_fleet_battle(FLEET)
        messages = events(runner, "comms_message")
        assert messages and all(COMMS_MESSAGE_KEYS <= set(e.data) for e in messages)
        claim = next(e for e in messages if e.data["kind"] == "TARGET_CLAIM")
        assert claim.ship_id == "alpha_1" and claim.data["sender_role"] == "captain"
        assert claim.data["deliver_at_t"] == 60.0 and claim.data["recipient_ids"] == ["alpha_2"]
        delivered = [e for e in events(runner, "comms_delivered") if e.data["msg_id"] == claim.data["msg_id"]]
        assert [(e.ship_id, e.timestamp) for e in delivered] == [("alpha_2", 60.0)]
        # The enemy channel keeps its legacy 'message' event (viewer combat log).
        legacy = events(runner, "message")
        assert [e.data["message"] for e in legacy] == ["Strike your colours, Lancer."]
        assert legacy[0].data["msg_id"] in {e.data["msg_id"] for e in messages}

    def test_async_loop_delivers(self, tmp_path):
        """run_fleet_battle_async (every MCP battle) never delivered captain traffic."""
        stub = Stub(captain=self.signal_script)
        runner = make_runner(stub, tmp_path)
        asyncio.run(runner.run_fleet_battle_async(FLEET))
        assert any(i["kind"] == "TARGET_CLAIM" for i in stub.inboxes[("stub/harpoon", 2)])
        assert any(r["ship_id"] == "alpha_1" for r in stub.reports[("alpha", 2)])

    def test_http_loop_delivers(self, tmp_path):
        from src.llm.mcp_http_server import run_fleet_battle_with_http
        stub = Stub(captain=self.signal_script)
        runner = make_runner(stub, tmp_path)
        asyncio.run(run_fleet_battle_with_http(runner, FLEET, Mock()))
        assert any(i["kind"] == "TARGET_CLAIM" for i in stub.inboxes[("stub/harpoon", 2)])
        assert any(r["ship_id"] == "alpha_1" for r in stub.reports[("alpha", 2)])

    def test_signal_to_admiral_rides_on_the_report(self, tmp_path):
        def script(ctx):
            calls = [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.3})]
            if ctx.model == "stub/dart" and ctx.cp == 1:
                calls.append(("fleet_signal", {"kind": "PD_COVER_REQUEST", "to": "ADMIRAL",
                                               "text": "Two torpedoes inbound, need PD cover."}))
            return calls
        stub = Stub(captain=script)
        runner = make_runner(stub, tmp_path)
        runner.run_fleet_battle(FLEET)
        dart = {r["ship_id"]: r for r in stub.reports[("alpha", 2)]}["alpha_1"]
        assert dart["request"]["kind"] == "PD_COVER"
        assert dart["signals"][0]["text"] == "Two torpedoes inbound, need PD cover."
        assert not any(i["kind"] == "PD_COVER_REQUEST" for i in stub.inboxes[("stub/harpoon", 2)])

    def test_unroutable_recipient_is_a_tool_error_not_a_lost_message(self, tmp_path):
        def script(ctx):
            calls = [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.3})]
            if ctx.model == "stub/dart" and ctx.cp == 1:
                calls.append(("fleet_signal", {"kind": "FREE", "to": "OCS Lancer", "text": "x"}))
            return calls
        stub = Stub(captain=script)
        runner = make_runner(stub, tmp_path)
        runner.run_fleet_battle(FLEET)
        decision = next(e for e in events(runner, "captain_decision")
                        if e.timestamp == 30.0 and e.ship_id == "alpha_1")
        assert any("NOT SENT" in err and "ENEMY" in err for err in decision.data["tool_errors"])
        assert not [e for e in events(runner, "comms_message") if e.data["sender_id"] == "alpha_1"]


class TestDatalink:

    def test_kills_losses_and_pd_losses(self, tmp_path):
        from src.simulation import SimulationEventType
        runner = make_runner(Stub(), tmp_path)
        runner.setup_fleet_battle(FLEET)
        sim, channel = runner.simulation, runner.fleet_communication
        link = EngineDatalink()
        link.prime(sim)
        sim.get_ship("alpha_1").point_defense[0].is_operational = False
        sim._log_event(SimulationEventType.SHIP_DYING, "beta_1", data={"killer_id": "alpha_2"})
        sim.get_ship("beta_1").is_dying = True
        ids = iter(range(100))
        out = link.collect(sim, channel, {"alpha", "beta"}, 3, 90.0, lambda s: f"m{next(ids)}")
        by_kind = {(m.faction, m.kind): m for m in out}

        kill = by_kind[("alpha", "KILL")]
        assert kill.recipient_ids == ["alpha_1", "alpha_2"] and kill.text.endswith("(by TIS Harpoon)")
        assert ("beta", "LOSS") not in by_kind          # its whole fleet is gone: nobody to tell
        pd = by_kind[("alpha", "PD_LOSS")]
        assert pd.recipient_ids == ["alpha_2"]           # peers, not the damaged ship itself
        assert pd.structured["pd_lost"] == 1 and "0/1 left" in pd.text
        assert all(m.sender_role == "engine" and m.channel == CHANNEL_FLEET_NET for m in out)

        # Each death is announced once; an unchanged PD count is silent.
        sim._log_event(SimulationEventType.SHIP_DESTROYED, "beta_1", data={"killer_id": "alpha_2"})
        assert link.collect(sim, channel, {"alpha", "beta"}, 4, 120.0, lambda s: f"m{next(ids)}") == []


# ---------------------------------------------------------------------------
# discuss_with_admiral: context, exchange count, memory
# ---------------------------------------------------------------------------

class TestDiscussion:

    def test_two_exchanges_live_context_and_memory(self, tmp_path):
        def script(ctx):
            if ctx.model == "stub/dart" and ctx.cp == 1 and ctx.turn <= 2:
                return [("discuss_with_admiral", {"question": f"Question {ctx.turn}: is Lancer closing?"})]
            return [("set_maneuver", {"maneuver_type": "INTERCEPT", "throttle": 0.3})]
        stub = Stub(captain=script)
        runner = make_runner(stub, tmp_path)
        runner.run_fleet_battle(FLEET)

        # Both advertised exchanges were answered (the old loop stopped at one).
        assert len(events(runner, "captain_admiral_discussion")) == MAX_DISCUSSION_EXCHANGES
        assert stub.turns[("stub/dart", 1)] == MAX_DISCUSSION_EXCHANGES + 1
        dart = runner.alpha_captains["alpha_1"]
        assert dart.decision_count == 2                   # re-decides did not inflate it
        # The admiral answered from a live picture measured from the asking ship...
        first = stub.answers[0][-1]["content"]
        assert "ENEMIES BY RANGE FROM TIS Dart" in first and "OCS Lancer" in first
        # ...and remembers: the second answer sees the first exchange.
        assert "Question 1" in stub.answers[1][-1]["content"]
        log = runner.alpha_admiral.discussion_log
        assert [q["question"][:10] for q in log] == ["Question 1", "Question 2"]
        report = {r["ship_id"]: r for r in stub.reports[("alpha", 2)]}["alpha_1"]
        assert [d["answer"][:8] for d in report["discussion"]] == ["Answer 1", "Answer 2"]
        kinds = [(e.data["channel"], e.data["kind"]) for e in events(runner, "comms_message")]
        assert kinds.count(("captain_to_admiral", "QUESTION")) == 2
        assert kinds.count(("admiral_to_captain", "ANSWER")) == 2


# ---------------------------------------------------------------------------
# Diplomacy: LLM <-> LLM and MCP bridge
# ---------------------------------------------------------------------------

class TestDiplomacy:

    def test_admiral_messages_are_recorded_and_delivered_at_the_next_decision(self, tmp_path):
        runner = make_runner(Stub(), tmp_path)
        runner.setup_fleet_battle(FLEET)
        runner.checkpoint_count = 1
        runner.alpha_admiral._pending_enemy_message = "Propose a draw."
        runner._exchange_admiral_messages()
        msg = events(runner, "comms_message")[-1]
        assert (msg.data["channel"], msg.data["sender_id"], msg.data["recipient_ids"]) == (
            "admiral_diplomacy", "alpha_admiral", ["beta_admiral"])
        assert not events(runner, "comms_delivered")
        runner.checkpoint_count = 2
        runner._prepare_admiral_inbox(runner.beta_admiral)
        delivered = events(runner, "comms_delivered")
        assert [(e.ship_id, e.data["msg_id"]) for e in delivered] == [("beta_admiral", msg.data["msg_id"])]

    def test_mcp_bridge_delivers_and_records_both_directions(self, tmp_path):
        from src.llm.mcp_chat import AdmiralChat
        runner = make_runner(Stub(), tmp_path)
        runner.setup_fleet_battle(FLEET)
        runner.mcp_chat = AdmiralChat()
        runner.alpha_mcp = SimpleNamespace(name="MCP Alpha")
        runner.alpha_admiral = None
        runner.mcp_chat.send_message("alpha", "Yield the field.", 30.0)
        runner.beta_admiral._pending_enemy_message = "Never."
        runner._bridge_mcp_messages()

        assert runner.beta_admiral._received_enemy_messages == ["Yield the field."]
        recorded = {e.data["sender_id"]: e for e in events(runner, "comms_message")}
        assert recorded["mcp:alpha"].data["sender_role"] == "mcp"
        assert recorded["beta_admiral"].data["recipient_ids"] == ["mcp:alpha"]
        assert [e.ship_id for e in events(runner, "comms_delivered")] == ["mcp:alpha"]
        runner._bridge_mcp_messages()  # nothing new: nothing recorded twice
        assert len(events(runner, "comms_message")) == 2

    def test_mcp_commander_reaches_admiral_less_enemy_captains_and_hears_them(self, tmp_path):
        from src.llm.mcp_chat import AdmiralChat
        runner = make_runner(Stub(), tmp_path, admirals=False)
        runner.setup_fleet_battle(FLEET)
        runner.checkpoint_count = 1
        runner.mcp_chat = AdmiralChat()
        runner.alpha_mcp = SimpleNamespace(name="MCP Alpha")
        runner.mcp_chat.send_message("alpha", "Lancer, stand down.", 30.0)
        runner._bridge_mcp_messages()
        lancer = runner.beta_captains["beta_1"]
        runner._deliver_fleet_messages([("beta_1", lancer, "beta")])
        assert [(i["sender_role"], i["channel"], i["text"]) for i in lancer.inbox] == [
            ("mcp", CHANNEL_ENEMY, "Lancer, stand down.")]

        # ...and the captains' open-channel answer reaches the MCP commander's chat.
        lancer.outbox.append({"tool": "send_message", "to": "ALL_ENEMIES", "kind": "FREE",
                              "text": "Come and take it."})
        runner._collect_captain_message(lancer, "beta_1")
        runner.checkpoint_count, runner.simulation.current_time = 2, 60.0
        runner._deliver_fleet_messages([("beta_1", lancer, "beta")])
        history = runner.mcp_chat.get_recent_history("alpha")
        assert history[-1]["content"] == "[OCS Lancer] Come and take it."
        assert history[-1]["sender"] == "enemy"
        assert [e.ship_id for e in events(runner, "comms_delivered")][-1] == "mcp:alpha"
