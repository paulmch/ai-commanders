"""scripts/replay_turn.py: select recorded calls and re-send them, offline on the mock."""
import json

import pytest

from scripts import replay_turn as rt
from src.llm.call_policy import call_role


def test_call_role_classifies_every_recorded_call_shape():
    assert call_role("admiral", "draft", ["select_fleet"]) == "draft"
    assert call_role("admiral", None, ["select_fleet"]) == "draft"
    assert call_role("admiral", None, ["set_formation"]) == "draft"
    assert call_role("admiral", None, ["issue_order", "set_fleet_directive", "set_battle_plan"]) == "directive"
    assert call_role("admiral", None, ["issue_order"]) == "order"
    assert call_role("captain", None, ["set_maneuver", "set_heading"]) == "captain"
    assert call_role("captain", None, []) == "other"            # personality selection
    assert call_role("admiral", None, []) == "other"            # discuss_with_admiral answer
    assert call_role(None, None, ["anything"]) == "other"


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """A real recording: a 2-checkpoint mock battle (Sonnet 5.5 vs Haiku 5.5)."""
    from tests.test_anthropic_client import _battle
    tmp = tmp_path_factory.mktemp("rec")
    runner, client, output, data = _battle(tmp, "claude-sonnet-5-5", "claude-haiku-5-5", turns=2)
    runner.run_battle(data)
    return output


def _run(recording, tmp_path, *extra):
    out = tmp_path / "out"
    code = rt.main([str(recording), "--mock", "--out", str(out), *extra])
    rows = [json.loads(line) for line in open(f"{out}.jsonl")] if (tmp_path / "out.jsonl").exists() else []
    return code, rows, out


def test_load_calls_reads_every_role_with_its_verbatim_input(recording):
    calls = rt.load_calls([recording])
    roles = {c["role"] for c in calls}
    assert {"directive", "order", "captain"} <= roles
    for c in calls:
        assert c["messages"][0]["role"] == "system" and c["messages"][-1]["role"] == "user"
        assert c["effort"] == "medium" and c["max_tokens"] == 8192
    orders = [c for c in calls if c["role"] == "order"]
    assert orders and all(c["subject"] for c in orders)     # ordered ship parsed from the text


def test_mock_replay_writes_one_row_per_reply_and_a_comparison_table(recording, tmp_path):
    code, rows, out = _run(recording, tmp_path, "--role", "captain", "--checkpoint", "1",
                           "--variant", "claude-sonnet-5-5:high", "--variant", "claude-haiku-5-5:low",
                           "-n", "2")
    assert code == 0
    sources = {r["source"]["key"] for r in rows}
    assert len(rows) == len(sources) * 2 * 2
    for r in rows:
        assert r["error"] is None and r["tool_calls"] and r["signature"]
        assert r["variant"]["effort"] in ("high", "low")
        assert r["usage"]["output_tokens"] > 0 and r["cost"] > 0
        assert r["source"]["role"] == "captain" and r["source"]["checkpoint"] == 1
    table = open(f"{out}.md").read()
    assert "recorded:" in table and "agree 2/2" in table
    # Replicates after the first of a prefix read the cache the first one wrote.
    calls = [c for c in rt.load_calls([recording]) if c["role"] == "captain" and c["checkpoint"] == 1]
    jobs, _ = rt.build_jobs(calls, [rt.parse_variant("claude-sonnet-5-5:high"),
                                    rt.parse_variant("claude-haiku-5-5:low")], 2)
    prefixes = len({j["prefix"] for j in jobs})
    writes = sum(1 for r in rows if r["usage"]["cache_creation_input_tokens"])
    assert prefixes < len(rows) and writes == prefixes


def test_patch_edits_the_sent_text_and_a_patch_that_matches_nothing_is_refused(recording, tmp_path):
    calls = [c for c in rt.load_calls([recording]) if c["role"] == "captain"]
    user = calls[0]["messages"][-1]["content"]
    needle = user.split("\n")[0][:20]
    good = tmp_path / "good.json"
    good.write_text(json.dumps([{"find": needle, "replace": "PATCHED-LINE"}]))
    edits = rt.load_patch(str(good))
    patched, counts = rt.apply_patch(calls[0]["messages"], edits)
    assert counts == [1] and "PATCHED-LINE" in patched[-1]["content"]
    assert patched[0] == calls[0]["messages"][0]           # system untouched (target user)
    assert calls[0]["messages"][-1]["content"] == user     # source not mutated
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"find": "no such line anywhere", "replace": "x"}]))
    with pytest.raises(SystemExit, match="matches nothing"):
        rt.main([str(recording), "--mock", "--role", "captain", "--limit", "1",
                 "--variant", f"claude-sonnet-5-5:medium:{bad}", "--out", str(tmp_path / "o")])


def test_system_patch_is_a_new_cached_prefix_in_the_estimate(recording):
    calls = [c for c in rt.load_calls([recording]) if c["role"] == "captain"][:4]
    plain = rt.parse_variant("claude-sonnet-5-5:medium")
    jobs, _ = rt.build_jobs(calls, [plain], 3)
    base = rt.estimate(jobs)
    edit = {"target": "system", "find": "a", "replace": "A_PATCHED", "regex": False, "count": 1,
            "pattern": __import__("re").compile("a")}
    patched = dict(plain, label="p", edits=[edit])
    jobs2, stats = rt.build_jobs(calls, [plain, patched], 3)
    est = rt.estimate(jobs2)
    assert est["prefixes"] == 2 * base["prefixes"]
    assert est["expected_usd"] > 2 * base["expected_usd"] * 0.99


def test_dry_run_sends_nothing_and_unknown_models_are_refused(recording, tmp_path, capsys):
    assert rt.main([str(recording), "--role", "order", "--dry-run", "--variant", "claude-opus-5-5:low"]) == 0
    out = capsys.readouterr().out
    assert "Estimate" in out and "order" in out
    assert not (tmp_path / "out.jsonl").exists()
    with pytest.raises(ValueError, match="not in the price table"):
        rt.parse_variant("claude-sonnet-9:low")
    with pytest.raises(SystemExit):
        rt.parse_variant("openai/gpt-5:low")


def test_action_signature_ignores_free_text_and_argument_order():
    a = [{"function": {"name": "set_maneuver", "arguments": json.dumps(
        {"maneuver_type": "PADLOCK", "reason": "because", "throttle": 0.41})}}]
    b = [{"name": "set_maneuver", "args": {"throttle": 0.4, "maneuver_type": "PADLOCK", "reason": "other"}}]
    assert rt.action_signature(a) == rt.action_signature(b)
    assert rt.action_signature([]) == "(no tool call)"


def test_replay_cap_refuses_requests_beyond_it_and_the_ledger_records_the_run(recording, tmp_path):
    from src.llm.spending import SpendLedger
    calls = [c for c in rt.load_calls([recording]) if c["role"] == "captain"]
    one = rt.estimate(rt.build_jobs(calls[:1], [rt.parse_variant("claude-opus-5-5:low")], 1)[0])
    # Room for one request in flight, not all: after the first one lands its
    # cost (about the estimate) plus the next reservation is over the cap.
    # (It was 1.5 x the reservation, which let every request through once the
    # captain doctrine grew a little and the reservation with it.)
    limit = one["reservation_bound_usd"] + 0.5 * one["expected_usd"]
    ledger = tmp_path / "ledger.jsonl"
    code = rt.main([str(recording), "--mock", "--role", "captain", "--variant", "claude-opus-5-5:low",
                    "-n", "3", "--workers", "1", "--spend-limit", f"{limit:.6f}",
                    "--ledger", str(ledger), "--out", str(tmp_path / "cap")])
    rows = [json.loads(x) for x in open(tmp_path / "cap.jsonl")]
    assert code == 3
    assert any(r["error"] and "budget exhausted" in r["error"] for r in rows)
    assert sum(r["cost"] or 0 for r in rows) <= limit
    close = [json.loads(x) for x in open(ledger)][-1]
    assert close["status"] == "stopped" and close["script"] == "replay_turn" and close["mode"] == "mock"
    assert SpendLedger(ledger).total_usd() == 0.0     # mock runs never count


def test_selection_file_round_trip_and_two_battles_in_one_folder_keep_distinct_keys(recording, tmp_path):
    import shutil
    sel = tmp_path / "sel.json"
    assert rt.main([str(recording), "--role", "order", "--checkpoint", "2", "--dry-run",
                    "--variant", "claude-sonnet-5-5:low", "--save-selection", str(sel)]) == 0
    keys = json.loads(sel.read_text())
    args = rt.parse_args([str(recording), "--select", str(sel), "--variant", "claude-sonnet-5-5:low"])
    picked = rt.select_calls(rt.load_calls([recording]), args)
    assert keys and [rt.call_key(c) for c in picked] == keys
    folder = tmp_path / "two"
    folder.mkdir()
    shutil.copy(recording, folder / "battle_a.json")
    shutil.copy(recording, folder / "battle_b.json")
    calls = rt.load_calls(rt.recording_files(folder))
    assert len({rt.call_key(c) for c in calls}) == len(calls)
