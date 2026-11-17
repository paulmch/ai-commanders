"""
Recording file format (src/recording_io.py): plain .json and gzip .json.gz,
one reader and one writer for every tool, deterministic gzip bytes, the
converter's verify-then-delete, and run_draft_battle's --plain-json /
--no-recording switches.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src import recording_io as rio
from src.llm.battle_recorder import BattleRecorder, create_battle_filename

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _recording(n_events: int = 40) -> dict:
    return {
        "recording_version": "3.0", "battle_name": "t", "winner": "alpha",
        "events": [{"timestamp": i * 0.5, "event_type": "shot_fired",
                    "data": {"distance_km": 100.0 + i / 3, "nan": float("nan") if i == 3 else 1.0}}
                   for i in range(n_events)],
        "sim_trace": [{"t": float(i), "ships": {"alpha_1": {"pos": [i * 1.5, 0.1, -2.0]}}}
                      for i in range(n_events)],
    }


def _draft_battle(out_dir, *flags, seed=3, budget=60, distance=500, time_limit=30):
    cmd = [sys.executable, str(ROOT / "scripts" / "run_draft_battle.py"), "--auto-draft",
           "--no-admirals", "--seed", str(seed), "--budget", str(budget), "--max-ships", "1",
           "--distance", str(distance), "--time-limit", str(time_limit), "--max-checkpoints", "3",
           "--no-predictions", "--quiet", "--recording-dir", str(out_dir), *flags]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)


# ---------------------------------------------------------------- round trip

def test_round_trip_plain_and_gzip(tmp_path):
    rec = _recording()
    plain = rio.dump_recording(rec, tmp_path / "a.json")
    gz = rio.dump_recording(rec, tmp_path / "a.json.gz")
    assert Path(plain).read_bytes()[:2] != rio.GZIP_MAGIC
    assert Path(gz).read_bytes()[:2] == rio.GZIP_MAGIC
    assert rio.canonical(rio.load_recording(plain)) == rio.canonical(rec)
    assert rio.canonical(rio.load_recording(gz)) == rio.canonical(rec)
    # Text payloads are written verbatim.
    rio.dump_recording('{"events": []}', tmp_path / "t.json.gz")
    assert rio.load_recording_text(tmp_path / "t.json.gz") == '{"events": []}'


def test_misnamed_files_load_by_magic_bytes(tmp_path):
    rec = _recording(5)
    rio.dump_recording(rec, tmp_path / "gzip_named_plain.json", compress=True)
    rio.dump_recording(rec, tmp_path / "plain_named_gzip.json.gz", compress=False)
    for name in ("gzip_named_plain.json", "plain_named_gzip.json.gz"):
        assert rio.canonical(rio.load_recording(tmp_path / name)) == rio.canonical(rec)


def test_gzip_is_deterministic_with_a_clean_header(tmp_path):
    rec = _recording()
    a = Path(rio.dump_recording(rec, tmp_path / "a.json.gz")).read_bytes()
    b = Path(rio.dump_recording(rec, tmp_path / "sub" / "other_name.json.gz")).read_bytes()
    assert a == b
    # RFC 1952 header: magic, CM=8, FLG=0 (no FNAME / FCOMMENT / FEXTRA), MTIME=0.
    assert a[:4] == b"\x1f\x8b\x08\x00"
    assert a[4:8] == b"\x00\x00\x00\x00"
    assert b"a.json" not in a and b"other_name" not in a


def test_write_is_atomic_and_leaves_no_temp_files(tmp_path):
    rio.dump_recording(_recording(3), tmp_path / "x.json.gz")
    rio.dump_recording(_recording(4), tmp_path / "x.json.gz")   # overwrite in place
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.json.gz"]
    assert len(rio.load_recording(tmp_path / "x.json.gz")["events"]) == 4


def test_names_sidecars_and_listing(tmp_path):
    assert rio.sidecar_path("d/battle_x.json.gz") == Path("d/battle_x.draft.json")
    assert rio.sidecar_path("d/battle_x.json") == Path("d/battle_x.draft.json")
    assert rio.recording_stem("battle_x.json.gz") == "battle_x"
    assert rio.with_recording_suffix("d/b.json", True) == Path("d/b.json.gz")
    assert rio.with_recording_suffix("d/b.json.gz", False) == Path("d/b.json")
    for name in ("battle_a.json", "battle_b.json.gz", "battle_a.draft.json", "notes.txt"):
        (tmp_path / name).write_text("{}")
    assert [p.name for p in rio.iter_recordings(tmp_path, "battle_*")] == [
        "battle_a.json", "battle_b.json.gz"]
    assert create_battle_filename("a/m1", "b/m2").endswith(".json")
    assert create_battle_filename("a/m1", "b/m2", compress=True).endswith(".json.gz")


# ---------------------------------------------------------------- recorder

def _recorder(n: int = 30) -> BattleRecorder:
    rec = BattleRecorder()
    rec.recording.battle_name = "unit"
    rec.recording.events = [{"timestamp": float(i), "event_type": "x", "data": {"v": i / 7}}
                            for i in range(n)]
    rec.recording.sim_trace = [{"t": float(i), "ships": {}} for i in range(n)]
    return rec


def test_recorder_save_by_extension_and_explicit_choice(tmp_path):
    rec = _recorder()
    gz = rec.save(str(tmp_path / "r.json.gz"))
    plain = rec.save(str(tmp_path / "r.json"))
    forced = rec.save(str(tmp_path / "forced.json"), compress=True)
    # Plain is exactly to_json() (the perf harness hashes that text); the
    # gzip holds compact JSON of the same object.
    assert Path(plain).read_text() == rec.recording.to_json()
    assert gzip.decompress(Path(gz).read_bytes()).decode() == rec.recording.to_compact_json()
    assert Path(forced).read_bytes() == Path(gz).read_bytes()
    want = json.loads(rec.recording.to_json())
    for p in (gz, plain, forced):
        assert rio.load_recording(p) == want


def test_same_battle_gives_the_same_gzip_bytes(tmp_path):
    a = _recorder().save(str(tmp_path / "a.json.gz"))
    b = _recorder().save(str(tmp_path / "b.json.gz"))
    assert Path(a).read_bytes() == Path(b).read_bytes()


# ---------------------------------------------------------------- converter

def test_converter_verifies_then_deletes(tmp_path):
    conv = _script("compress_recordings")
    rec = _recording()
    rio.dump_recording(rec, tmp_path / "battle_a.json", compress=False)
    (tmp_path / "battle_a.draft.json").write_text(json.dumps({"budget": 1, "events": []}))
    (tmp_path / "config.json").write_text(json.dumps({"alpha_fleet": {}}))
    rio.dump_recording(rec, tmp_path / "already.json", compress=True)   # misnamed gzip

    assert conv.main([str(tmp_path), "--dry-run"]) == 0
    assert not (tmp_path / "battle_a.json.gz").exists()

    assert conv.main([str(tmp_path), "--delete-originals"]) == 0
    assert not (tmp_path / "battle_a.json").exists()
    assert rio.canonical(rio.load_recording(tmp_path / "battle_a.json.gz")) == rio.canonical(rec)
    # Sidecars, non-recordings and gzip files are left alone.
    assert (tmp_path / "battle_a.draft.json").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "already.json").exists()


def test_converter_keeps_the_original_when_verification_fails(tmp_path, monkeypatch):
    conv = _script("compress_recordings")
    rio.dump_recording(_recording(), tmp_path / "battle_a.json", compress=False)
    real = conv.load_recording
    monkeypatch.setattr(conv, "load_recording",
                        lambda p: dict(real(p), winner="tampered"))
    assert conv.main([str(tmp_path), "--delete-originals"]) == 1
    assert (tmp_path / "battle_a.json").exists()
    assert not (tmp_path / "battle_a.json.gz").exists()


def test_converter_refuses_a_stale_existing_target(tmp_path):
    conv = _script("compress_recordings")
    rio.dump_recording(_recording(), tmp_path / "battle_a.json", compress=False)
    rio.dump_recording(_recording(3), tmp_path / "battle_a.json.gz")   # different battle
    row = conv.convert_one(tmp_path / "battle_a.json", delete_original=True, dry_run=False)
    assert row["status"] == "skipped-target-exists"
    assert (tmp_path / "battle_a.json").exists()


# ---------------------------------------------------------------- readers

@pytest.fixture(scope="module")
def gz_fixture(tmp_path_factory):
    """A small free battle with gun shots written as an old-style plain
    recording (--plain-json), plus a gzip copy of it."""
    d = tmp_path_factory.mktemp("rec")
    r = _draft_battle(d, "--plain-json", "--trace", seed=1, budget=100, distance=100, time_limit=90)
    assert r.returncode == 0, r.stderr[-2000:]
    plain = next(rio.iter_recordings(d))
    assert plain.name.endswith(".json") and plain.read_bytes()[:1] == b"{"
    gz = rio.dump_recording(rio.load_recording(plain), d / "copy.json.gz")
    return plain, Path(gz)


def test_fire_control_calibration_reads_both(gz_fixture):
    fcc = _script("fire_control_calibration")
    plain, gz = gz_fixture
    a, b = fcc.rows_from_recording(str(plain)), fcc.rows_from_recording(str(gz))
    assert a and len(a) == len(b)
    # The row's recording name is the stem, whatever the suffix.
    for ra, rb in zip(a, b):
        assert {k: v for k, v in ra.items() if v != plain.stem} == \
            {k: v for k, v in rb.items() if v != gz.name[:-len(".json.gz")]}


def test_tactics_eval_analyze_reads_both(gz_fixture):
    ev = _script("tactics_eval")
    plain, gz = gz_fixture
    a, b = ev.analyze(str(plain)), ev.analyze(str(gz))
    a.pop("file"), b.pop("file")
    assert a == b


def test_tactics_eval_caches_rows_and_drops_recordings(tmp_path, gz_fixture, monkeypatch):
    ev = _script("tactics_eval")
    _, gz = gz_fixture
    calls = []

    def fake_run(cmd, cwd, env, stdout, stderr, check):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--recording-dir") + 1])
        (out / "battle_x.json.gz").write_bytes(gz.read_bytes())
        (out / "battle_x.draft.json").write_text("{}")

    monkeypatch.setattr(ev.subprocess, "run", fake_run)
    row = ev.run_one(1, 100, "legacy", "legacy", False, str(tmp_path), 60.0, [])
    tag_dir = tmp_path / row["tag"]
    assert "error" not in row and (tag_dir / "metrics.json").exists()
    assert not list(rio.iter_recordings(tag_dir, "battle_*")) and not (tag_dir / "battle_x.draft.json").exists()
    assert ev.run_one(1, 100, "legacy", "legacy", False, str(tmp_path), 60.0, []) == \
        json.loads(json.dumps(row))
    assert len(calls) == 1
    kept = ev.run_one(2, 100, "legacy", "legacy", False, str(tmp_path), 60.0, [], keep_recordings=True)
    assert list(rio.iter_recordings(tmp_path / kept["tag"], "battle_*"))
    # A recording already in an old --out folder is analysed but never deleted.
    old_dir = tmp_path / "s3_b100_legacy-legacy"
    old_dir.mkdir()
    (old_dir / "battle_old.json.gz").write_bytes(gz.read_bytes())
    ev.run_one(3, 100, "legacy", "legacy", False, str(tmp_path), 60.0, [])
    assert (old_dir / "battle_old.json.gz").exists() and (old_dir / "metrics.json").exists()
    assert len(calls) == 2


def test_fake_live_server_loads_both(gz_fixture):
    fls = _script("fake_live_server")
    plain, gz = gz_fixture
    assert fls.load_recording(str(gz)) == fls.load_recording(str(plain))
    # Its default replay URL keeps the .json.gz name for the viewer.
    assert fls.default_replay_url("visualizer/public/recordings/x.json.gz") == "/recordings/x.json.gz"


def test_generate_test_battle_out_transcodes_by_extension(tmp_path, gz_fixture):
    gtb = _script("generate_test_battle")
    plain, gz = gz_fixture
    out_plain = gtb.copy_recording(str(gz), str(tmp_path / "copy.json"))
    out_gz = gtb.copy_recording(str(plain), str(tmp_path / "copy.json.gz"))
    assert Path(out_plain).read_bytes()[:2] != rio.GZIP_MAGIC
    assert Path(out_gz).read_bytes()[:2] == rio.GZIP_MAGIC
    assert rio.load_recording(out_plain) == rio.load_recording(out_gz) == rio.load_recording(plain)


@pytest.mark.parametrize("script", ["refine_commander", "render_admiral_view", "perf_check",
                                    "run_replay_trials", "compress_recordings"])
def test_script_entry_points_import(script):
    """Every touched script still imports (routes through recording_io)."""
    mod = _script(script)
    assert hasattr(mod, "main")


# ---------------------------------------------------------------- run_draft_battle

def test_run_draft_battle_formats_and_no_recording(tmp_path):
    gz_dir, plain_dir, none_dir = tmp_path / "gz", tmp_path / "plain", tmp_path / "none"
    r = _draft_battle(gz_dir)
    assert r.returncode == 0, r.stderr[-2000:]
    files = sorted(p.name for p in gz_dir.iterdir())
    assert len(files) == 2 and files[0].endswith(".draft.json") and files[1].endswith(".json.gz"), files
    assert files[0][: -len(".draft.json")] == files[1][: -len(".json.gz")]

    r = _draft_battle(plain_dir, "--plain-json")
    assert r.returncode == 0, r.stderr[-2000:]
    rec = next(rio.iter_recordings(plain_dir))
    assert rec.name.endswith(".json") and rec.read_bytes()[:1] == b"{"
    assert rio.load_recording(rec)["winner"] == rio.load_recording(
        next(rio.iter_recordings(gz_dir)))["winner"]

    r = _draft_battle(none_dir, "--no-recording")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Winner:" in r.stdout and "Recording:" not in r.stdout
    assert not none_dir.exists() or not any(none_dir.iterdir())


def test_viewer_scripts_pass_gzip_names_through():
    pytest.importorskip("playwright")
    snap = _script("viewer_snapshot")
    assert "?recording=/recordings/p5_x.json.gz&" in snap.page_url_for(5187, "p5_x.json.gz")
