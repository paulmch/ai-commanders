#!/usr/bin/env python3
"""
Gzip existing plain-JSON battle recordings in place: battle_x.json ->
battle_x.json.gz (compact JSON inside, deterministic bytes; see
src/recording_io.py).

    uv run python scripts/compress_recordings.py data/recordings --dry-run
    uv run python scripts/compress_recordings.py /scratch/sweep --delete-originals

Walks <dir> recursively. Skips draft sidecars (*.draft.json), files that are
already gzip, and any JSON that is not a battle recording (decided by content:
a dict with an `events` list and a recording header field). Every written
.json.gz is loaded back and its parsed object compared (canonical dump, so
NaN-safe) with the original's before the original is deleted; a mismatch
removes the new file and keeps the original. An existing .json.gz next to the
original is reused if it verifies, else the file is skipped. --dry-run writes
nothing and reports the size the gzip files would have.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recording_io import (  # noqa: E402
    GZIP_MAGIC, SIDECAR_SUFFIX, canonical, compact_json, dump_recording, gzip_bytes,
    load_recording, looks_like_recording, with_recording_suffix,
)


def candidates(root: Path) -> List[Path]:
    files = [root] if root.is_file() else sorted(root.rglob("*.json"))
    return [p for p in files if p.is_file() and not p.is_symlink()
            and not p.name.endswith(SIDECAR_SUFFIX)]


def convert_one(path: Path, delete_original: bool, dry_run: bool) -> Dict[str, object]:
    """Returns {status, path, before, after}; status is one of converted,
    would-convert, skipped-<why>, failed-<why>."""
    row: Dict[str, object] = {"path": str(path), "before": path.stat().st_size, "after": 0}
    with open(path, "rb") as f:
        head = f.read(2)
    if head == GZIP_MAGIC:
        row["status"] = "skipped-already-gzip"
        return row
    try:
        obj = json.loads(path.read_bytes())
    except (ValueError, UnicodeDecodeError):
        row["status"] = "skipped-not-json"
        return row
    if not looks_like_recording(obj):
        row["status"] = "skipped-not-a-recording"
        return row
    original = canonical(obj)
    target = with_recording_suffix(path, compress=True)
    if dry_run:
        row["after"] = len(gzip_bytes(compact_json(obj).encode("utf-8")))
        row["status"] = "would-convert"
        return row
    if target.exists():
        try:
            reused = canonical(load_recording(target)) == original
        except (OSError, ValueError, EOFError):
            reused = False
        if not reused:
            row["status"] = "skipped-target-exists"
            return row
    else:
        dump_recording(obj, target, compress=True)
        try:
            ok = canonical(load_recording(target)) == original
        except (OSError, ValueError, EOFError):
            ok = False
        if not ok:
            target.unlink(missing_ok=True)
            row["status"] = "failed-verify"
            return row
    row["after"] = target.stat().st_size
    row["status"] = "converted"
    if delete_original:
        path.unlink()
    return row


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, help="directory (recursive) or a single recording")
    ap.add_argument("--delete-originals", action="store_true",
                    help="delete each .json once its .json.gz verified")
    ap.add_argument("--dry-run", action="store_true",
                    help="write nothing; report the space gzip would save")
    args = ap.parse_args(argv)
    if not args.dir.exists():
        ap.error(f"{args.dir} does not exist")

    rows = []
    for path in candidates(args.dir):
        row = convert_one(path, args.delete_originals, args.dry_run)
        rows.append(row)
        if row["status"] in ("converted", "would-convert"):
            print(f"{row['status']:14s} {row['before'] / 1e6:8.2f} MB -> {row['after'] / 1e6:7.2f} MB  {path}")
        else:
            print(f"{row['status']:14s} {path}")
    done = [r for r in rows if r["status"] in ("converted", "would-convert")]
    before = sum(r["before"] for r in done)
    after = sum(r["after"] for r in done)
    failed = [r for r in rows if str(r["status"]).startswith("failed")]
    verb = "would save" if args.dry_run else ("saved" if args.delete_originals else
                                              "would save once the originals are deleted")
    ratio = f" ({before / after:.1f}x)" if after else ""
    print(f"\n{len(done)} recordings: {before / 1e6:.1f} MB plain -> {after / 1e6:.1f} MB gzip{ratio}; "
          f"{verb} {(before - after) / 1e6:.1f} MB. "
          f"{len(rows) - len(done) - len(failed)} skipped, {len(failed)} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
