"""scripts/replay_turn.py plot arms (old / replot / noimage) on a recorded admiral
directive call with a tactical-plot image: offline, on the mock client."""
import base64
import json
from types import SimpleNamespace

import pytest

from scripts import replay_turn as rt

pytest.importorskip("matplotlib")

# The legend an older battle recorded (the pass-5 wave-5 plot), as it sat at
# the end of the cached system message.
OLD_LEGEND = """=== TACTICAL PLOT (attached image, each checkpoint) ===
Left panel top-down (X/Y in km), right panel 3D perspective. Cyan = alpha fleet, orange =
beta fleet. Diamonds are torpedoes (grey = dead seeker), yellow dots are coilgun slugs.
X marks the closest approach of the nearest enemy group if both sides coast."""


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """A 2-checkpoint mock battle with admiral vision and a sim_trace."""
    import anthropic
    from scripts.run_live_draft_battle import LiveDraftRunner, LiveDraftState
    from src.llm.battle_runner import BattleConfig, load_fleet_data
    from src.llm.captain import LLMCaptainConfig
    from src.llm.fleet_config import AdmiralConfig, BattleFleetConfig
    from src.llm.fleet_draft import FleetDraft, draft_to_fleet_definition, name_drafted_ships
    from src.llm.mock_anthropic import mock_anthropic_transport
    from src.llm.providers import create_llm_client
    tmp = tmp_path_factory.mktemp("plotrec")
    models = {"alpha": "claude-sonnet-5-5", "beta": "claude-haiku-5-5"}
    client = create_llm_client(models["alpha"], [models["beta"]], temperature=None, max_tokens=8192,
                               reasoning_effort="medium", anthropic_options={
                                   "api_key": "sk-ant-test-FAKE", "max_retries": 0,
                                   "http_client": anthropic.DefaultHttpxClient(
                                       transport=mock_anthropic_transport())})
    state = LiveDraftState()
    state.drafts = {f: FleetDraft(faction=f, budget=20, points_spent=14,
                                  ships=name_drafted_ships(["frigate", "corvette"], f))
                    for f in ("alpha", "beta")}
    fleets = {f: draft_to_fleet_definition(
        state.drafts[f], 500, captain_model=models[f], temperature=None,
        admiral_config=AdmiralConfig(model=models[f], name=f"Admiral {f}", temperature=None,
                                     vision=True, image_input=True)) for f in models}
    fc = BattleFleetConfig(
        battle_name="mock", alpha_fleet=fleets["alpha"], beta_fleet=fleets["beta"],
        initial_distance_km=500, decision_interval_s=30, time_limit_s=90, max_checkpoints=2,
        seed=7, personality_selection=False, record_battle=True, record_sim_trace=True,
        use_notebooks=False)
    runner = LiveDraftRunner(BattleConfig.from_fleet_config(fc, recording_dir=str(tmp), verbose=False),
                             LLMCaptainConfig("A", "A"), LLMCaptainConfig("B", "B"), client, fc)
    output = tmp / "live.json"
    runner.configure_live(SimpleNamespace(set_battle_status=lambda *a: None), state, {}, output, 0)
    runner._admiral_view_dir = None
    runner.run_battle(load_fleet_data())
    return output


def directive_calls(recording, legend=OLD_LEGEND):
    """The recorded directive calls, with their legend block set to an older text
    (what a replay of an old battle meets)."""
    from src.llm.prompts import ADMIRAL_VISION_LEGEND
    calls = [c for c in rt.load_calls([recording]) if c["role"] == "directive" and rt.has_plot(c)]
    assert calls, "the mock battle recorded no directive call with a plot"
    for c in calls:
        c["messages"] = json.loads(json.dumps(c["messages"]))   # assets are shared between calls
        sysm = c["messages"][0]
        text = rt._content_text(sysm["content"])
        assert ADMIRAL_VISION_LEGEND in text
        new = text.replace(ADMIRAL_VISION_LEGEND, legend) if legend is not None else \
            text.replace("\n\n" + ADMIRAL_VISION_LEGEND, "")
        sysm["content"] = new if isinstance(sysm["content"], str) else [{"type": "text", "text": new}]
    return calls


def _texts(messages):
    """Every text part (system, user) and every image url, in order."""
    out, images = [], []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append((m["role"], content))
        else:
            for p in content:
                if p.get("type") == "text":
                    out.append((m["role"], p["text"]))
                elif p.get("type") == "image_url":
                    images.append(p["image_url"]["url"])
    return out, images


def test_three_arms_change_only_the_image_and_the_legend(recording):
    from src.llm.prompts import ADMIRAL_VISION_LEGEND
    calls = directive_calls(recording)
    arms = [rt.parse_variant(f"claude-sonnet-5-5:high::{m}") for m in ("old", "replot", "noimage")]
    jobs, _ = rt.build_jobs(calls, arms, 1)
    by = {(j["call"]["event_id"], j["variant"]["plot"]): j for j in jobs}
    for call in calls:
        old, new, none = (by[(call["event_id"], m)] for m in ("old", "replot", "noimage"))
        assert old["messages"] == call["messages"] and old["plot"] is None    # verbatim
        (o_text, o_img), (r_text, r_img), (n_text, n_img) = (
            _texts(j["messages"]) for j in (old, new, none))
        # images: recorded / a fresh 1280x720 render / none. This battle was
        # recorded with the current renderer, so the re-render reproduces the
        # image the admiral was sent byte for byte (the same code path).
        assert len(o_img) == len(r_img) == 1 and n_img == [] and r_img == o_img
        png = base64.b64decode(r_img[0].split(",", 1)[1])
        assert rt.png_size(png) == (1280, 720)
        # system: only the legend block differs
        o_sys, r_sys, n_sys = (t[0][1] for t in (o_text, r_text, n_text))
        assert OLD_LEGEND in o_sys and ADMIRAL_VISION_LEGEND in r_sys
        assert r_sys.replace(ADMIRAL_VISION_LEGEND, OLD_LEGEND) == o_sys
        assert n_sys == o_sys.replace("\n\n" + OLD_LEGEND, "") and "TACTICAL PLOT" not in n_sys
        # user text: identical, except that no-image drops the plot pointer line
        assert [t for t in o_text[1:]] == [t for t in r_text[1:]]
        assert [t[1] for t in n_text[1:]] == [rt.PLOT_POINTER.sub("", t[1]) for t in o_text[1:]]
        assert "Tactical plot attached" in o_text[-1][1] and "Tactical plot attached" not in n_text[-1][1]
        # tools and everything else in the request: identical
        assert old["call"]["tools"] is new["call"]["tools"] is none["call"]["tools"]
        assert old["max_tokens"] == new["max_tokens"] == none["max_tokens"]
    # the new legend is a different cached prefix
    assert len({j["prefix"] for j in jobs}) == 3 * len({j["prefix"] for j in jobs if j["variant"] is arms[0]})


def test_replot_renders_what_render_admiral_view_renders(recording):
    from scripts.render_admiral_view import render_at
    from src.recording_io import load_recording
    call = directive_calls(recording)[0]
    messages, info = rt.apply_plot_mode(call, "replot", {})
    url = next(p for p in messages[-1]["content"] if p.get("type") == "image_url")["image_url"]["url"]
    data = load_recording(call["path"])
    assert base64.b64decode(url.split(",", 1)[1]) == render_at(
        data, call["t"], call["faction"], checkpoint=call["checkpoint"], score_text=info["score"])


def test_score_line_goes_into_the_new_header(recording):
    call = directive_calls(recording)[0]
    user = call["messages"][-1]
    for p in user["content"]:
        if p.get("type") == "text":
            p["text"] += "\nSCORE IF THE BATTLE ENDED NOW (time-limit verdict): you 210 / them 153 = you win"
    _, info = rt.apply_plot_mode(call, "replot", {})
    assert info["score"] == "score you 210 / them 153"


def test_a_plot_without_a_legend_block_is_refused(recording):
    call = directive_calls(recording, legend=None)[0]
    with pytest.raises(SystemExit, match="no '=== TACTICAL PLOT"):
        rt.apply_plot_mode(call, "replot", {})
    with pytest.raises(SystemExit, match="no '=== TACTICAL PLOT"):
        rt.apply_plot_mode(call, "noimage", {})


def test_dry_run_shows_the_swap_and_mock_runs_all_three_arms(recording, tmp_path, capsys):
    base = [str(recording), "--role", "directive", "--checkpoint", "1",
            "--variant", "claude-sonnet-5-5:high", "--variant", "claude-sonnet-5-5:high::replot",
            "--variant", "claude-sonnet-5-5:high::noimage"]
    assert rt.main(base + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "image SWAPPED" in out and "-> 1280x720" in out and "changed cached system prefix" in out
    assert "image REMOVED" in out and "legend removed" in out
    prefix = tmp_path / "arms"
    assert rt.main(base + ["--mock", "--out", str(prefix)]) == 0
    rows = [json.loads(line) for line in open(f"{prefix}.jsonl")]
    assert {r["variant"]["plot"] for r in rows} == {"old", "replot", "noimage"}
    assert all(r["error"] is None and r["signature"] for r in rows)


def test_single_variant_flags_and_old_syntax_still_parse():
    assert rt.parse_variant("claude-sonnet-5-5:high")["plot"] == "old"
    assert rt.parse_variant("claude-sonnet-5-5:high:")["plot"] == "old"
    assert rt.parse_variant("claude-sonnet-5-5:high::no-image")["plot"] == "noimage"
    with pytest.raises(SystemExit):
        rt.parse_variant("claude-sonnet-5-5:high::sideways")
    args = rt.parse_args(["rec", "--model", "claude-sonnet-5-5", "--replot"])
    assert args.replot and not args.no_image
