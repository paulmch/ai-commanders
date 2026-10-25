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
