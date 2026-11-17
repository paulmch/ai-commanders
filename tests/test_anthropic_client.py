"""The native Anthropic client, the provider factory and their battle wiring.

Everything is offline: the SDK's HTTP layer (httpx2) gets an in-process mock
transport, and the key is a fake.
"""
import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from src.llm import anthropic_client as ac
from src.llm.anthropic_client import (
    ANTHROPIC_MODELS, AnthropicClient, anthropic_effort, cost_usd, mark_cache_breakpoint,
    reservation_usd, translate_messages, translate_tools,
)
from src.llm.client import CaptainClient, LLMCallError
from src.llm.fleet_config import _get_short_model_name
from src.llm.mock_anthropic import message_response, mock_anthropic_transport
from src.llm.providers import LLMRouter, create_llm_client, provider_for_model
from src.llm.spending import RequestBudget

FAKE_KEY = "sk-ant-test-FAKE-0123456789-do-not-leak"
PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"


def tool(name, props=None):
    return {"type": "function", "function": {
        "name": name, "description": f"{name} tool",
        "parameters": {"type": "object", "properties": props or {"x": {"type": "string"}}}}}


def make_client(model="claude-haiku-5-5", record=None, respond=None, **kw):
    transport = mock_anthropic_transport(record=record, respond=respond)
    return AnthropicClient(model, api_key=FAKE_KEY, max_retries=0,
                           http_client=anthropic.DefaultHttpxClient(transport=transport), **kw)


SYSTEM = {"role": "system", "content": "DOCTRINE " * 400}


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def test_tools_translate_to_input_schema_in_order_without_strict():
    tools = [tool("set_maneuver", {"maneuver_type": {"type": "string"}}), tool("log_note")]
    out = translate_tools(tools)
    assert [t["name"] for t in out] == ["set_maneuver", "log_note"]
    assert out[0] == {"name": "set_maneuver", "description": "set_maneuver tool",
                      "input_schema": {"type": "object",
                                       "properties": {"maneuver_type": {"type": "string"}}}}
    assert all("strict" not in t for t in out)


def test_system_and_image_parts_translate():
    system, msgs = translate_messages([
        SYSTEM,
        {"role": "user", "content": [{"type": "text", "text": "plot attached"},
                                     {"type": "image_url", "image_url": {"url": PNG}}]},
    ])
    assert system == [{"type": "text", "text": SYSTEM["content"]}]
    assert msgs == [{"role": "user", "content": [
        {"type": "text", "text": "plot attached"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": PNG.split(",", 1)[1]}}]}]


def test_tool_rounds_translate_with_all_results_in_one_user_message():
    _, msgs = translate_messages([
        {"role": "user", "content": "orders?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "a", "arguments": '{"x": "1"}'}},
            {"id": "t2", "type": "function", "function": {"name": "b", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "ok a"},
        {"role": "tool", "tool_call_id": "t2", "content": "ok b"},
        {"role": "user", "content": "and now?"},
    ])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "a", "input": {"x": "1"}},
        {"type": "tool_use", "id": "t2", "name": "b", "input": {}}]
    assert msgs[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "ok a"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "ok b"},
        {"type": "text", "text": "and now?"}]


def test_trailing_assistant_turn_is_rejected():
    with pytest.raises(ValueError, match="prefill"):
        translate_messages([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])


def test_multi_round_tool_use_replays_thinking_blocks_unchanged():
    record = []
    client = make_client(record=record)
    first = [SYSTEM, {"role": "user", "content": "orders?"}]
    calls = client.decide_with_tools(first, [tool("set_fleet_directive")])
    assert calls and calls[0].name == "set_fleet_directive"
    follow = first + [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in calls]},
        *[{"role": "tool", "tool_call_id": c.id, "content": "accepted"} for c in calls],
    ]
    client.decide_with_tools(follow, [tool("set_fleet_directive")])
    assistant = record[1]["messages"][1]
    assert assistant["role"] == "assistant"
    # The full stored response.content: thinking (with its signature) first.
    assert assistant["content"][0]["type"] == "thinking"
    assert assistant["content"][0]["signature"] == "mock-signature-1"
    assert [b["type"] for b in assistant["content"][1:]] == ["tool_use"] * len(calls)
    results = record[1]["messages"][2]
    assert results["role"] == "user"
    assert [b["type"] for b in results["content"]] == ["tool_result"] * len(calls)


# ---------------------------------------------------------------------------
# Request rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", sorted(ANTHROPIC_MODELS))
def test_every_request_follows_the_current_api_rules(model):
    record = []
    client = make_client(model, record=record, temperature=0.7, max_tokens=32768,
                         reasoning_effort="high")
    client.decide_with_tools([SYSTEM, {"role": "user", "content": "go"}], [tool("a")],
                             temperature=0.2, tool_choice="required")
    client.complete([{"role": "user", "content": "hi"}], temperature=0.3)
    for body in record:
        assert not {"temperature", "top_p", "top_k", "fallbacks"} & set(body)
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "high"}
        assert body["max_tokens"] == 16000
        assert body["model"] == model
    assert record[0]["tool_choice"] == {"type": "auto"}   # never any/tool


def test_effort_mapping_is_explicit():
    assert anthropic_effort(None) == "medium"
    assert anthropic_effort("none") == "low"
    assert anthropic_effort("XHIGH") == "xhigh"
    with pytest.raises(ValueError):
        anthropic_effort("turbo")


def test_cache_breakpoint_on_last_system_block_and_reads_from_second_call():
    record = []
    client = make_client(record=record)
    messages = [SYSTEM, {"role": "user", "content": "checkpoint 1"}]
    client.decide_with_tools(messages, [tool("a")])
    first = client.last_call_meta()["usage"]
    client.decide_with_tools([SYSTEM, {"role": "user", "content": "checkpoint 2"}], [tool("a")])
    second = client.last_call_meta()["usage"]
    body = record[0]
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert all("cache_control" not in t for t in body["tools"])
    assert all("cache_control" not in b for m in body["messages"] for b in m["content"])
    assert first["anthropic"]["cache_creation_input_tokens"] > 0
    assert first["anthropic"]["cache_read_input_tokens"] == 0
    assert second["anthropic"]["cache_read_input_tokens"] == first["anthropic"]["cache_creation_input_tokens"]
    assert client.stats.cache_hits == 1 and client.stats.cache_writes == 1
    assert mark_cache_breakpoint([{"type": "text", "text": "x"}], "5m")[-1]["cache_control"] == {
        "type": "ephemeral"}


# ---------------------------------------------------------------------------
# Stop reasons
# ---------------------------------------------------------------------------

def test_refusal_is_a_counted_no_order_outcome():
    tool_use = {"type": "tool_use", "id": "t1", "name": "a", "input": {"x": "partial"}}
    client = make_client(respond=lambda body: message_response(
        [tool_use], "refusal", stop_details={"type": "refusal", "category": "cyber",
                                             "explanation": "mock"}))
    assert client.decide_with_tools([{"role": "user", "content": "go"}], [tool("a")]) == []
    assert client.last_finish_reason() == "refusal"
    meta = client.last_call_meta()
    assert meta["stop_details"] == {"category": "cyber", "explanation": "mock"}
    assert client.stats.refusals == 1
    assert client.stats.refusals_by_model == {"claude-haiku-5-5": 1}


def test_max_tokens_reports_length_and_drops_the_cut_off_call():
    whole = {"type": "tool_use", "id": "t1", "name": "a", "input": {"x": "1"}}
    cut = {"type": "tool_use", "id": "t2", "name": "b", "input": {"x": "par"}}
    client = make_client(respond=lambda body: message_response([whole, cut], "max_tokens"))
    calls = client.decide_with_tools([{"role": "user", "content": "go"}], [tool("a"), tool("b")])
    assert [c.name for c in calls] == ["a"]
    assert client.last_finish_reason() == "length"
    assert client.stats.truncated == 1 and client.stats.malformed_arguments == 1


def test_end_turn_and_pause_turn():
    replies = iter([message_response([{"type": "text", "text": "thinking..."}], "pause_turn"),
                    message_response([{"type": "text", "text": "done"}], "end_turn")])
    record = []
    client = make_client(record=record, respond=lambda body: next(replies))
    response = client.complete([{"role": "user", "content": "go"}])
    assert (response.content, response.finish_reason) == ("done", "stop")
    assert len(record) == 2 and record[1]["messages"][-1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def _error(status, kind):
    return lambda body: httpx2.Response(status, json={"type": "error",
                                                      "error": {"type": kind, "message": f"{kind} test"}})


@pytest.mark.parametrize("status,kind,retryable", [
    (400, "invalid_request_error", False), (401, "authentication_error", False),
    (404, "not_found_error", False), (429, "rate_limit_error", True),
    (500, "api_error", True), (529, "overloaded_error", True),
])
def test_sdk_errors_translate_to_llm_call_error(status, kind, retryable):
    client = make_client(respond=_error(status, kind))
    with pytest.raises(LLMCallError) as info:
        client.decide_with_tools([{"role": "user", "content": "go"}], [tool("a")])
    assert info.value.status_code == status
    assert info.value.retryable is retryable
    assert FAKE_KEY not in str(info.value)
    assert client.stats.failures == 1


def test_connection_error_is_retryable():
    def fail(request):
        raise httpx2.ConnectError("no route")
    client = AnthropicClient("claude-haiku-5-5", api_key=FAKE_KEY, max_retries=0,
                             http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(fail)))
    with pytest.raises(LLMCallError) as info:
        client.complete([{"role": "user", "content": "go"}])
    assert info.value.retryable and info.value.status_code is None


# ---------------------------------------------------------------------------
# Cost and the spend cap
# ---------------------------------------------------------------------------

def test_cost_per_model_with_cache_writes_and_reads():
    usage = {"input_tokens": 1000, "output_tokens": 2000, "cache_read_input_tokens": 10000,
             "cache_creation_input_tokens": 5000,
             "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 4000}}
    # Sonnet 5.5: $2 in, $10 out, $0.10 read, $2.50 5m write, $4 1h write
    assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(
        (1000 * 2 + 2000 * 10 + 10000 * 0.10 + 1000 * 2.5 + 4000 * 4) / 1e6)
    # Sonnet 5 reads at 0.1x ($0.20)
    assert cost_usd("claude-sonnet-5", usage) == pytest.approx(
        (1000 * 2 + 2000 * 10 + 10000 * 0.20 + 1000 * 2.5 + 4000 * 4) / 1e6)
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(
        (1000 * 4 + 2000 * 20 + 10000 * 0.20 + 1000 * 5 + 4000 * 8) / 1e6)
    assert cost_usd("claude-fable-5-2", usage) == pytest.approx(
        (1000 * 8 + 2000 * 40 + 10000 * 0.20 + 1000 * 10 + 4000 * 16) / 1e6)
    assert cost_usd("claude-haiku-5-5", usage) == pytest.approx(
        (1000 * 0.1 + 2000 * 0.5 + 10000 * 0.01 + 1000 * 0.125 + 4000 * 0.2) / 1e6)
    # Unsplit writes bill at the TTL the request asked for.
    unsplit = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 1000}
    assert cost_usd("claude-sonnet-5-5", unsplit, "1h") == pytest.approx(1000 * 4 / 1e6)
    assert cost_usd("claude-sonnet-5-5", unsplit, "5m") == pytest.approx(1000 * 2.5 / 1e6)


def test_haiku_long_prompts_use_the_second_rate_card():
    long = {"input_tokens": 60_000, "cache_read_input_tokens": 50_000, "output_tokens": 1000}
    assert cost_usd("claude-haiku-5-5", long) == pytest.approx(
        (60_000 * 0.5 + 50_000 * 0.05 + 1000 * 2.5) / 1e6)


def test_unknown_models_raise_instead_of_costing_nothing():
    with pytest.raises(ValueError, match="Unknown Anthropic model"):
        cost_usd("claude-sonnet-9", {"input_tokens": 1})
    with pytest.raises(ValueError, match="Unknown Anthropic model"):
        provider_for_model("claude-sonnet-5-5-20260101")
    with pytest.raises(ValueError, match="Unknown Anthropic model"):
        AnthropicClient("claude-opus-9", api_key=FAKE_KEY)


def test_reservation_bounds_cost_and_settles_to_the_computed_cost():
    client = make_client("claude-sonnet-5-5", max_tokens=4096)
    client.budget = RequestBudget(1.0, {})
    messages = [SYSTEM, {"role": "user", "content": [
        {"type": "text", "text": "plot"}, {"type": "image_url", "image_url": {"url": PNG}}]}]
    bound, images = reservation_usd("claude-sonnet-5-5", {"messages": messages, "tools": [tool("a")]}, 4096)
    client.decide_with_tools(messages, [tool("a")])
    cost = client.last_call_meta()["usage"]["cost"]
    assert images == 1 and 0 < cost < bound
    assert client.budget.reported_usd == pytest.approx(cost)
    assert client.budget.committed_usd == pytest.approx(cost)
    assert client.stats.cost_usd == pytest.approx(cost)


def test_spend_cap_refuses_a_request_it_cannot_cover():
    record = []
    client = make_client("claude-sonnet-5-5", record=record)
    client.budget = RequestBudget(0.001, {})
    with pytest.raises(LLMCallError, match="budget exhausted"):
        client.decide_with_tools([SYSTEM, {"role": "user", "content": "go"}], [tool("a")])
    assert client.budget.exhausted and record == []


# ---------------------------------------------------------------------------
# Provider routing and provenance
# ---------------------------------------------------------------------------

def test_provider_is_chosen_from_the_model_id():
    assert provider_for_model("claude-sonnet-5-5") == "anthropic"
    assert provider_for_model("anthropic/claude-sonnet-5") == "openrouter"
    assert provider_for_model("deepseek/deepseek-v4.1-flash") == "openrouter"
    assert provider_for_model("heuristic") is None


def test_openrouter_only_battles_keep_the_plain_captain_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    client = create_llm_client("anthropic/claude-sonnet-5", ["deepseek/deepseek-v4.1-flash", "heuristic"],
                               session_id="s")
    assert type(client) is CaptainClient and client.session_id == "s"


def test_missing_anthropic_key_fails_at_start_up(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        create_llm_client("claude-sonnet-5-5", ["claude-sonnet-5"])


def test_two_model_ids_get_two_clients_and_each_side_its_own_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    record = []
    router = create_llm_client(
        "claude-sonnet-5-5", ["claude-sonnet-5", "deepseek/deepseek-v4.1-flash"],
        reasoning_effort="high",
        anthropic_options={"api_key": FAKE_KEY, "http_client": anthropic.DefaultHttpxClient(
            transport=mock_anthropic_transport(record=record))})
    assert isinstance(router, LLMRouter)
    a, b = router.client_for("claude-sonnet-5-5"), router.client_for("claude-sonnet-5")
    assert a is not b and (a.model, b.model) == ("claude-sonnet-5-5", "claude-sonnet-5")
    assert isinstance(router.client_for("deepseek/deepseek-v4.1-flash"), CaptainClient)
    for side_model in ("claude-sonnet-5-5", "claude-sonnet-5", "claude-sonnet-5-5"):
        router.decide_with_tools([{"role": "user", "content": side_model}], [tool("a")], model=side_model)
        assert router.last_call_meta()["served_by"] == side_model
    assert [r["model"] for r in record] == ["claude-sonnet-5-5", "claude-sonnet-5", "claude-sonnet-5-5"]
    assert [r["messages"][0]["content"][0]["text"] for r in record] == [r["model"] for r in record]
    assert router.stats.calls == 3   # one shared CallStats
    assert router.provenance("claude-sonnet-5")["model"] == "claude-sonnet-5"
    assert router.provenance("claude-sonnet-5")["effort"] == "high"


def test_short_names_keep_anthropic_versions_apart():
    assert _get_short_model_name("claude-sonnet-5-5") == "Sonnet5.5"
    assert _get_short_model_name("claude-sonnet-5") == "Sonnet5"
    assert _get_short_model_name("claude-haiku-4-5-20251001") == "Haiku4.5"
    assert _get_short_model_name("anthropic/claude-opus-4.8") == "Opus4.8"


def test_key_never_reaches_repr_or_provenance():
    client = make_client()
    assert FAKE_KEY not in repr(client)
    assert FAKE_KEY not in json.dumps(client.provenance())


# ---------------------------------------------------------------------------
# Battles on the mock: caching across checkpoints, refusals, the cap, provenance
# ---------------------------------------------------------------------------

def _battle(tmp_path, alpha, beta, respond=None, record=None, limit=None, turns=3,
            config=None, captain_models=None, admirals=True, anthropic_extra=None,
            transport_options=None):
    from scripts.run_live_draft_battle import LiveDraftRunner, LiveDraftState
    from src.llm.battle_runner import BattleConfig, load_fleet_data
    from src.llm.captain import LLMCaptainConfig
    from src.llm.fleet_config import AdmiralConfig, BattleFleetConfig
    from src.llm.fleet_draft import FleetDraft, draft_to_fleet_definition, name_drafted_ships

    captain_models = captain_models or {}
    extra_models = [m for m in captain_models.values() if m != "heuristic"]
    client = create_llm_client(alpha, [beta, *extra_models], temperature=None, max_tokens=8192,
                               reasoning_effort="medium", anthropic_options={
                                   "api_key": FAKE_KEY, "max_retries": 0, **(anthropic_extra or {}),
                                   "http_client": anthropic.DefaultHttpxClient(
                                       transport=mock_anthropic_transport(
                                           record=record, respond=respond, **(transport_options or {})))})
    if limit is not None:
        client.budget = RequestBudget(limit, {})
    state = LiveDraftState()
    state.drafts = {f: FleetDraft(faction=f, budget=20, points_spent=14,
                                  ships=name_drafted_ships(["frigate", "corvette"], f))
                    for f in ("alpha", "beta")}
    models = {"alpha": alpha, "beta": beta}
    fleets = {f: draft_to_fleet_definition(
        state.drafts[f], 500, captain_model=captain_models.get(f, models[f]), temperature=None,
        admiral_config=AdmiralConfig(model=models[f], name=f"Admiral {f}", temperature=None,
                                     vision=False) if admirals else None) for f in models}
    fc = BattleFleetConfig(
        battle_name="mock", alpha_fleet=fleets["alpha"], beta_fleet=fleets["beta"],
        initial_distance_km=500, decision_interval_s=30, time_limit_s=(turns + 1) * 30,
        max_checkpoints=turns, seed=7, personality_selection=False, record_battle=True,
        record_sim_trace=False, use_notebooks=False)
    runner = LiveDraftRunner(BattleConfig.from_fleet_config(fc, recording_dir=str(tmp_path), verbose=False,
                                                            **(config or {})),
                             LLMCaptainConfig("A", "A"), LLMCaptainConfig("B", "B"), client, fc)
    output = tmp_path / "live.json"
    runner.configure_live(SimpleNamespace(set_battle_status=lambda *a: None), state, {}, output, 0)
    return runner, client, output, load_fleet_data()


def _calls(recording):
    return [e for e in recording["events"] if e["event_type"] == "model_call"]


def test_mock_battle_reads_the_cache_from_checkpoint_two_and_records_provenance(tmp_path):
    runner, client, output, data = _battle(tmp_path, "claude-sonnet-5-5", "claude-sonnet-5")
    runner.run_battle(data)
    raw = output.read_text()
    recording = json.loads(raw)
    calls = _calls(recording)
    later = [c for c in calls if c["timestamp"] > 30]
    assert later and all(c["data"]["usage"]["anthropic"]["cache_creation_input_tokens"] == 0
                         for c in later), "a cached prefix changed between checkpoints"
    assert all(c["data"]["usage"]["anthropic"]["cache_read_input_tokens"] > 0 for c in later)
    # Each side's calls went to its own model, and the provenance says so.
    for c in calls:
        assert c["data"]["model"] == {"alpha": "claude-sonnet-5-5", "beta": "claude-sonnet-5"}[c["data"]["faction"]]
        assert c["data"]["request"]["provider"] == "anthropic"
        assert c["data"]["request"]["output_config"] == {"effort": "medium"}
    llm = recording["provenance"]["llm"]
    assert [e["model"] for e in llm["alpha"]] == ["claude-sonnet-5-5"]
    assert [e["model"] for e in llm["beta"]] == ["claude-sonnet-5"]
    assert llm["alpha"][0]["provider"] == "anthropic" and llm["alpha"][0]["sdk"].startswith("anthropic-python")
    assert llm["alpha"][0]["api_version"] and llm["alpha"][0]["fallbacks"] is False
    # Recorded costs are the price table applied to the recorded usage.
    for c in calls:
        assert c["data"]["usage"]["cost"] == pytest.approx(
            cost_usd(c["data"]["model"], c["data"]["usage"]["anthropic"]))
    assert FAKE_KEY not in raw


def test_refused_turns_issue_no_orders_are_not_re_asked_and_are_logged(tmp_path):
    def refuse(body):
        names = {t["name"] for t in body.get("tools") or []}
        if body["model"] == "claude-sonnet-5" and ("set_maneuver" in names or "set_fleet_directive" in names):
            return message_response([], "refusal", model=body["model"], stop_details={
                "type": "refusal", "category": "general_harms", "explanation": None})
        return None
    record = []
    runner, client, output, data = _battle(tmp_path, "claude-sonnet-5-5", "claude-sonnet-5",
                                           respond=refuse, record=record, turns=2)
    result = runner.run_battle(data)
    recording = json.loads(output.read_text())
    events = recording["events"]
    refusals = [e for e in events if e["event_type"] == "model_refusal"]
    assert refusals and {e["data"]["faction"] for e in refusals} == {"beta"}
    beta_captain_requests = [r for r in record if r["model"] == "claude-sonnet-5"
                             and "set_maneuver" in {t["name"] for t in r.get("tools") or []}]
    beta_directive_requests = [r for r in record if r["model"] == "claude-sonnet-5"
                               and "set_fleet_directive" in {t["name"] for t in r.get("tools") or []}]
    # Two ships x two checkpoints, one call each: no retry, no follow-up round.
    assert len(beta_captain_requests) == 4
    assert len(beta_directive_requests) == 2
    failed = [e for e in events if e["event_type"] == "admiral_directive_failed"]
    assert failed and all("refused" in e["data"]["reason"] for e in failed)
    assert {e["data"]["faction"] for e in failed} == {"beta"}
    beta_decisions = [e for e in events if e["event_type"] == "captain_decision"
                      and e["data"]["faction"] == "beta"]
    assert beta_decisions and all(e["data"]["call_failed"] for e in beta_decisions)
    assert client.stats.refusals == len(refusals)
    assert client.stats.refusals_by_side == {"beta": len(refusals)}
    assert result.checkpoints_used == 2   # the battle went on


def test_spend_cap_stops_an_anthropic_battle_with_the_recorded_abort(tmp_path):
    from scripts.run_live_draft_battle import record_stop
    runner, client, output, data = _battle(tmp_path, "claude-haiku-5-5", "claude-haiku-5-5",
                                           limit=0.03, turns=10)
    with pytest.raises(RuntimeError, match="Spending ceiling") as info:
        runner.run_battle(data)
    record_stop(runner, f"RuntimeError: {info.value}")
    recording = json.loads(output.read_text())
    assert client.budget.exhausted
    assert client.budget.reported_usd == pytest.approx(client.budget.committed_usd)
    assert client.budget.committed_usd <= 0.03
    assert runner.checkpoint_count < 10
    stopped = [e for e in recording["events"] if e["event_type"] == "battle_stopped"]
    assert stopped and "Spending ceiling" in stopped[0]["data"]["reason"]
    assert "Spending ceiling" in recording["result_reason"]
