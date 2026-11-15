"""scripts/viewer_clip.py: frame timing, camera paths, resume and encode commands."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("playwright")

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _module(name):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def test_frame_times_step_battle_time_at_the_frame_rate():
    clip = _module("viewer_clip")
    times = clip.frame_times(98.0, 6.0, 10)
    assert len(times) == 60
    assert times[0] == 98.0 and times[1] == 98.1 and times[-1] == 103.9
    assert clip.frame_times(0, 0.01, 24) == [0.0]          # never an empty clip
    assert clip.frame_times(5, 1, 24)[-1] == pytest.approx(5 + 23 / 24)


def test_resume_skips_frames_on_disk_and_encodes_the_unbroken_prefix(tmp_path):
    clip = _module("viewer_clip")
    for i in (0, 1, 2, 4):
        (tmp_path / clip.frame_name(i)).write_bytes(b"png")
    (tmp_path / "f_00003.png.tmp.png").write_bytes(b"half")   # an interrupted write is not a frame
    assert clip.missing_frames(tmp_path, 6) == [3, 5]
    assert clip.contiguous_prefix(tmp_path, 6) == 3
    assert clip.contiguous_prefix(tmp_path / "nowhere", 6) == 0


def test_camera_path_is_held_at_the_ends_and_eased_between_keys(tmp_path):
    clip = _module("viewer_clip")
    f = tmp_path / "path.json"
    f.write_text(json.dumps({"relative_to": "alpha_1", "keys": [
        {"t": 20, "camera": [10, 0, 0], "target": [0, 0, 0]},
        {"t": 10, "camera": [0, 0, 0], "target": [0, 0, 0]},
    ]}))
    path = clip.load_camera_path(str(f))
    assert [k["t"] for k in path["keys"]] == [10, 20]         # sorted by time
    assert clip.camera_at(path, 0)["camera"] == [0, 0, 0]
    assert clip.camera_at(path, 99)["camera"] == [10, 0, 0]
    assert clip.camera_at(path, 15)["camera"] == [5, 0, 0]
    assert clip.camera_at(path, 12.5)["camera"][0] == pytest.approx(10 * (0.25 ** 2 * (3 - 0.5)))  # smoothstep
    assert clip.camera_at(path, 15)["rel"] == "alpha_1"
    path["ease"] = "linear"
    assert clip.camera_at(path, 12.5)["camera"][0] == pytest.approx(2.5)
    f.write_text("[]")
    with pytest.raises(ValueError):
        clip.load_camera_path(str(f))
    f.write_text(json.dumps([{"t": 1, "camera": [0, 0], "target": [0, 0, 0]}]))
    with pytest.raises(ValueError):
        clip.load_camera_path(str(f))


def test_encode_commands_are_h264_yuv420p_with_even_dimensions(tmp_path):
    clip = _module("viewer_clip")
    cmd = clip.mp4_command("ffmpeg", tmp_path, 10, tmp_path / "a.mp4", 60)
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert cmd[cmd.index("-framerate") + 1] == "10"
    assert cmd[cmd.index("-frames:v") + 1] == "60"
    assert "trunc(iw/2)*2" in cmd[cmd.index("-vf") + 1]
    assert cmd[cmd.index("-i") + 1].endswith("f_%05d.png")
    gif = clip.gif_command("ffmpeg", tmp_path, 10, tmp_path / "a.gif", 60, 480)
    graph = gif[gif.index("-filter_complex") + 1]
    assert "palettegen" in graph and "paletteuse" in graph and "scale=480:-2" in graph


def test_without_an_h264_encoder_the_frames_are_kept_and_the_command_printed(tmp_path, monkeypatch, capsys):
    clip = _module("viewer_clip")
    (tmp_path / clip.frame_name(0)).write_bytes(b"png")
    monkeypatch.setattr(clip, "find_ffmpeg", lambda: ("/opt/pw/ffmpeg-linux", False))
    assert clip.encode(tmp_path, 1, 10, tmp_path / "a.mp4", False, 640, 18) == 2
    out = capsys.readouterr().out
    assert "no ffmpeg with an H.264 encoder" in out and "libx264" in out and str(tmp_path) in out
    assert not (tmp_path / "a.mp4").exists()
    assert clip.encode(tmp_path, 0, 10, tmp_path / "a.mp4", False, 640, 18) == 1


def test_sequence_seek_pauses_again_before_every_seek():
    # Under load the viewer's auto-play timer can fire after the harness's
    # first pause(); a sequence would then run on the wall clock
    snap = _module("viewer_snapshot")

    class Page:
        def __init__(self):
            self.calls = []
            self.frame = 0

        def evaluate(self, script, arg=None):
            self.calls.append((script, arg))
            if "frameIndex" in script:
                self.frame += 1
                return self.frame
            return None

        def wait_for_timeout(self, ms):
            pass

    session = object.__new__(snap.ViewerSession)
    session.page = Page()
    session.seek(12.5)
    script, arg = session.page.calls[0]
    assert arg == 12.5
    assert script.index("pause()") < script.index("seek(t)")
