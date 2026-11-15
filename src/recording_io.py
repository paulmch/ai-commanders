"""
Battle recording files: one reader and one writer for every tool.

A recording is either plain JSON (``*.json``, the historical format, usually
indented) or gzip-compressed JSON (``*.json.gz``, the default for new
recordings since pass 5; compact JSON inside, nobody reads it by eye).

- ``load_recording(path)`` / ``load_recording_text(path)`` read both. The
  format is decided by the file's first two bytes (gzip magic ``1f 8b``), not
  by its name, so a misnamed file still loads.
- ``dump_recording(obj_or_text, path)`` writes gzip when the path ends in
  ``.gz`` (or ``compress=True``) and plain text otherwise, atomically (temp
  file + ``os.replace``) so a viewer reading a file the live runner rewrites
  never sees half of it. The gzip stream is deterministic: header mtime 0, no
  embedded file name, fixed compression level - the same text always gives
  the same bytes on a given zlib build.

Stdlib only: src/llm/battle_recorder.py imports this module.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

PathLike = Union[str, "os.PathLike[str]"]

GZIP_MAGIC = b"\x1f\x8b"
PLAIN_SUFFIX = ".json"
GZIP_SUFFIX = ".json.gz"
SIDECAR_SUFFIX = ".draft.json"
# Measured on three committed recordings (14-38 MB plain): compact JSON at
# level 9 is 20-37x smaller than the indented plain file and gzips in
# 0.13-0.17 s CPU; level 6 is ~5% larger and barely faster.
GZIP_LEVEL = 9


def is_gzip_path(path: PathLike) -> bool:
    return str(path).endswith(".gz")


def recording_suffix(compress: bool) -> str:
    return GZIP_SUFFIX if compress else PLAIN_SUFFIX


def recording_stem(path: PathLike) -> str:
    """File name without ``.json.gz`` / ``.json`` (``battle_x.json.gz`` -> ``battle_x``)."""
    name = Path(path).name
    for suffix in (GZIP_SUFFIX, PLAIN_SUFFIX, ".gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def with_recording_suffix(path: PathLike, compress: bool) -> Path:
    """``path`` renamed to the plain or gzip recording suffix."""
    p = Path(path)
    return p.with_name(recording_stem(p) + recording_suffix(compress))


def sidecar_path(recording_path: PathLike) -> Path:
    """The draft sidecar next to a recording: ``battle_x.json[.gz]`` -> ``battle_x.draft.json``."""
    p = Path(recording_path)
    return p.with_name(recording_stem(p) + SIDECAR_SUFFIX)


def is_recording_name(path: PathLike) -> bool:
    """By name only: ``*.json`` / ``*.json.gz`` that is not a ``.draft.json`` sidecar."""
    name = Path(path).name
    if name.endswith(SIDECAR_SUFFIX):
        return False
    return name.endswith(PLAIN_SUFFIX) or name.endswith(GZIP_SUFFIX)


def iter_recordings(directory: PathLike, pattern: str = "*") -> Iterator[Path]:
    """Recording files (plain and gzip, no sidecars) in ``directory`` matching
    ``pattern`` + suffix, sorted by name."""
    d = Path(directory)
    found = set(d.glob(pattern + PLAIN_SUFFIX)) | set(d.glob(pattern + GZIP_SUFFIX))
    return iter(sorted(p for p in found if is_recording_name(p)))


def read_recording_bytes(path: PathLike) -> bytes:
    """The file's JSON bytes, inflated when it is gzip (by magic bytes)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] == GZIP_MAGIC:
        return gzip.decompress(data)
    return data


def load_recording_text(path: PathLike) -> str:
    return read_recording_bytes(path).decode("utf-8")


def load_recording(path: PathLike) -> Any:
    """Parse a recording (plain or gzip JSON)."""
    return json.loads(read_recording_bytes(path))


def gzip_bytes(data: bytes, level: int = GZIP_LEVEL) -> bytes:
    """Deterministic gzip: mtime 0 and no FNAME field, written by GzipFile on
    every Python version (gzip.compress hands the header to zlib on 3.11+,
    whose OS byte may differ from 3.10's)."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=level, mtime=0) as gz:
        gz.write(data)
    return buf.getvalue()


def compact_json(obj: Any, default=None) -> str:
    return json.dumps(obj, separators=(",", ":"), default=default)


def dump_recording(obj_or_text: Any, path: PathLike, compress: Optional[bool] = None,
                   default=None) -> str:
    """
    Write a recording. ``obj_or_text`` is a ``str`` (written as is: callers
    pass ``BattleRecording.to_json()`` for plain files and compact text for
    gzip) or a JSON-able object (dumped compact into gzip, indented 2 into a
    plain file). ``compress`` None decides by the ``.gz`` extension. Creates
    the parent directory; atomic. Returns the path written.
    """
    p = Path(path)
    if compress is None:
        compress = is_gzip_path(p)
    if isinstance(obj_or_text, (str, bytes)):
        text = obj_or_text
    elif compress:
        text = compact_json(obj_or_text, default=default)
    else:
        text = json.dumps(obj_or_text, indent=2, default=default)
    data = text.encode("utf-8") if isinstance(text, str) else text
    if compress:
        data = gzip_bytes(data)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o666 & ~_umask())   # mkstemp makes 0600; a recording is not secret
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(p)


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def canonical(obj: Any) -> str:
    """A canonical text of a parsed recording for equality checks (NaN-safe:
    NaN != NaN as floats, but both dump as 'NaN')."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def looks_like_recording(obj: Any) -> bool:
    """A parsed battle recording: a dict with an ``events`` list and a
    recording header field."""
    return (isinstance(obj, dict) and isinstance(obj.get("events"), list)
            and any(k in obj for k in ("recording_version", "sim_trace", "battle_name",
                                       "alpha_fleet", "alpha_model")))


__all__: List[str] = [
    "GZIP_MAGIC", "GZIP_LEVEL", "PLAIN_SUFFIX", "GZIP_SUFFIX", "SIDECAR_SUFFIX",
    "is_gzip_path", "recording_suffix", "recording_stem", "with_recording_suffix",
    "sidecar_path", "is_recording_name", "iter_recordings", "read_recording_bytes",
    "load_recording_text", "load_recording", "gzip_bytes", "compact_json",
    "dump_recording", "canonical", "looks_like_recording",
]
