"""
The spectator-HUD fixture (scripts/generate_test_battle.py hud_comms):
a heuristic fleet battle that also records a scripted command net. The
viewer is developed against it before the real comms pipeline lands, so
its events must follow the pass-4 comms schema exactly, and scripting
the net must not change the battle itself.
"""
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMS_MESSAGE_KEYS = {"msg_id", "channel", "kind", "sender_id", "sender_role", "sender_name", "faction",
                      "recipient_spec", "recipient_ids", "text", "structured", "checkpoint", "sent_t",
                      "deliver_at_t"}
CHANNELS = {"fleet_net", "direct", "captain_to_admiral", "admiral_to_captain", "enemy_broadcast",
            "admiral_diplomacy"}
ROLES = {"captain", "admiral", "engine", "mcp"}
DELIVERED_KEYS = {"msg_id", "recipient_id", "delivered_t", "checkpoint"}
REPORT_KEYS = {"report_id", "faction", "checkpoint", "captain_name", "ship_name", "status", "reason",
               "intent", "contact", "request", "compliance"}
COMPLIANCE_KEYS = {"ordered_maneuver", "executed_maneuver", "ordered_target", "actual_target",
                   "ordered_salvo", "launched"}
# What the comms runner adds (src/llm/fleet_comms.compliance): the engine's verdict
VERDICT_KEYS = {"ordered_maneuvers", "final", "followed", "deviations"}


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path, runner_cls=None, checkpoints=4):
    gen = _load_script("generate_test_battle")
    if runner_cls is not None:
        gen.HudCommsRunner = runner_cls(gen)
    with redirect_stdout(io.StringIO()):
        _, path = gen.hud_comms_battle(max_checkpoints=checkpoints, recording_dir=str(tmp_path))
    return json.loads(Path(path).read_text())


def _data(event):
    # Recorder-added provenance (sim-event merge) is not part of the schema
    return {k: v for k, v in event["data"].items() if k not in ("command_ids", "decision_ids", "target_id")}


def test_fixture_events_follow_the_comms_schema(tmp_path):
    rec = _run(tmp_path / "net")
    events = rec["events"]
    by_type = {}
    for e in events:
        by_type.setdefault(e["event_type"], []).append(e)

    messages = {e["data"]["msg_id"]: e for e in by_type["comms_message"]}
    assert len(messages) == len(by_type["comms_message"]), "msg_id is unique"
    for e in by_type["comms_message"]:
        d = _data(e)
        assert set(d) == COMMS_MESSAGE_KEYS
        assert d["channel"] in CHANNELS and d["sender_role"] in ROLES
        # ship_id is the sending ship; engine datalink traffic ("engine:alpha")
        # has no ship (battle_recorder.record_comms_message).
        assert e["ship_id"] == d["sender_id"] or (
            e["ship_id"] is None and d["sender_role"] == "engine")
        assert d["deliver_at_t"] >= d["sent_t"] == e["timestamp"]
        assert isinstance(d["recipient_ids"], list) and isinstance(d["structured"], dict)
    assert {e["data"]["channel"] for e in by_type["comms_message"]} == CHANNELS
    assert {"captain", "admiral", "engine"} <= {e["data"]["sender_role"] for e in by_type["comms_message"]}

    reports = {e["data"]["report_id"]: e for e in by_type["captain_report"]}
    delivered = set()
    report_deliveries = []
    for e in by_type["comms_delivered"]:
        d = _data(e)
        assert e["ship_id"] == d["recipient_id"]
        if d["msg_id"] in reports:
            # A report reaches its admiral at the next checkpoint, compliance refreshed (final)
            report_deliveries.append(e)
            filed = reports[d["msg_id"]]
            assert set(d) == DELIVERED_KEYS | {"compliance"}
            assert d["recipient_id"] == f"{filed['data']['faction']}_admiral"
            assert d["delivered_t"] == e["timestamp"] > filed["timestamp"]
            assert set(d["compliance"]) == COMPLIANCE_KEYS | VERDICT_KEYS and d["compliance"]["final"] is True
            continue
        assert set(d) == DELIVERED_KEYS
        msg = messages[d["msg_id"]]["data"]
        assert d["recipient_id"] in msg["recipient_ids"]
        assert d["delivered_t"] >= msg["deliver_at_t"] and d["delivered_t"] == e["timestamp"]
        delivered.add(d["msg_id"])
    due = {m for m, e in messages.items()
           if e["data"]["recipient_ids"] and e["data"]["deliver_at_t"] <= rec["duration_s"]}
    assert due - delivered, "at least one message is never delivered (the viewer shows it)"

    for e in by_type["captain_report"]:
        d = _data(e)
        assert set(d) == REPORT_KEYS
        assert d["status"] in ("ACK", "DEVIATE", "UNABLE")
        assert set(d["compliance"]) == COMPLIANCE_KEYS | VERDICT_KEYS and d["compliance"]["final"] is False
        assert d["compliance"]["followed"] == (not d["compliance"]["deviations"])
        assert d["contact"] is None or set(d["contact"]) == {"ship", "assessment", "note"}
        assert d["request"] is None or set(d["request"]) == {"kind", "text"}
    assert {e["data"]["status"] for e in by_type["captain_report"]} >= {"ACK", "DEVIATE"}
    assert report_deliveries, "reports are delivered to the admiral"
    assert any(e["data"]["compliance"]["deviations"] for e in by_type["captain_report"]), "some filed report is off orders"

    # Every legacy shape the viewer still reads
    for legacy in ("message", "admiral_directive", "admiral_plan", "admiral_order", "captain_log",
                   "captain_admiral_discussion", "captain_decision"):
        assert by_type.get(legacy), legacy
    # One open-channel message carries markup the viewer must show as text
    assert any("<img" in e["data"]["text"] for e in by_type["comms_message"])


def test_scripted_net_does_not_change_the_battle(tmp_path):
    def plain(gen):
        class Plain(gen.LLMBattleRunner):
            pass
        return Plain

    scripted = _run(tmp_path / "scripted", checkpoints=3)
    baseline = _run(tmp_path / "plain", runner_cls=plain, checkpoints=3)
    assert scripted["sim_trace"] == baseline["sim_trace"]
    net = {"comms_message", "comms_delivered", "captain_report", "message", "admiral_directive", "admiral_plan",
           "admiral_order", "captain_log", "captain_admiral_discussion"}
    strip = lambda rec: [(e["timestamp"], e["event_type"], e["ship_id"]) for e in rec["events"] if e["event_type"] not in net]
    assert strip(scripted) == strip(baseline)
