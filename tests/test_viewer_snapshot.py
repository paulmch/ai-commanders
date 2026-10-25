"""scripts/viewer_snapshot.py: shots must be reproducible run to run."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("playwright")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "viewer_snapshot.py"


def _snapshot_module():
    spec = importlib.util.spec_from_file_location("viewer_snapshot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_shots_keep_the_auto_director_off():
    # The director eases the camera in wall time from page load: left on,
    # a shot without --focus/--camera framed however far it had got.
    snap = _snapshot_module()
    url = snap.page_url_for(5181, "fleet_heur.json", gfx="low")
    assert url == "http://127.0.0.1:5181/?recording=/recordings/fleet_heur.json&gfx=low&director=0"
    assert snap.page_url_for(5181, "http://h/x.json") == "http://127.0.0.1:5181/?recording=http://h/x.json&director=0"
    assert "director" not in snap.page_url_for(5181, "fleet_heur.json", director=True)


def test_extra_query_parameters_reach_the_viewer_url():
    # README captures open viewer state through URL parameters (decision panel, pinned decision)
    snap = _snapshot_module()
    base = "http://127.0.0.1:5181/?recording=/recordings/a.json&gfx=high&director=0"
    assert snap.page_url_for(5181, "a.json", gfx="high", query="decisions=1&decision=e152") == \
        base + "&decisions=1&decision=e152"
    # a pasted leading ? or & is not doubled
    assert snap.page_url_for(5181, "a.json", gfx="high", query="?decisions=1") == base + "&decisions=1"
    assert snap.page_url_for(5181, "a.json", gfx="high", query="&decisions=1") == base + "&decisions=1"
    assert snap.page_url_for(5181, "a.json", gfx="high", query="") == base


def test_clip_is_parsed_in_css_pixels():
    snap = _snapshot_module()
    assert snap.parse_clip(None) is None
    assert snap.parse_clip("392,117,800,450") == {"x": 392.0, "y": 117.0, "width": 800.0, "height": 450.0}
    with pytest.raises(ValueError):
        snap.parse_clip("0,0,0,450")
