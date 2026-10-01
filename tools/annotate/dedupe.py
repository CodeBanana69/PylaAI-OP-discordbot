"""Drop near-duplicate gameplay frames with a perceptual hash.

Consecutive screenshots are almost the same picture. Training and labeling
on all of them wastes money and leaks near-copies into validation.

    python -m tools.annotate.dedupe --input frames --manifest dataset/dedupe.jsonl --output dataset/images

Requires ``imagehash`` (see tools/annotate/requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .boxes import iter_images, rel_posix


class _Node:
    def __init__(self, digest, item):
        self.digest = digest
        self.item = item
        self.children = {}


class BKTree:
    """Hamming-distance index so 50k frames are not compared pairwise."""

    def __init__(self):
        self.root = None

    def add(self, digest, item):
        if self.root is None:
            self.root = _Node(digest, item)
            return
        node = self.root
        while True:
            dist = int(digest - node.digest)
            child = node.children.get(dist)
            if child is None:
                node.children[dist] = _Node(digest, item)
                return
            node = child

    def nearest_within(self, digest, max_dist: int):
        best = None

        def visit(node):
            nonlocal best
            if node is None:
                return
            dist = int(digest - node.digest)
            if dist <= max_dist and (best is None or dist < best[0]):
                best = (dist, node.item)
            for edge, child in node.children.items():
                if abs(edge - dist) <= max_dist:
                    visit(child)

        visit(self.root)
        return best


def phash_image(path: Path, hash_size: int):
    import imagehash

    with Image.open(path) as image:
        return imagehash.phash(image.convert("RGB"), hash_size=hash_size)


def select_kept(records, max_per_folder: int):
    """records are in path order and already marked kept/duplicate."""
    if max_per_folder <= 0:
        return records
    kept_by_folder = defaultdict(list)
    for record in records:
        if record["kept"]:
            folder = str(Path(record["path"]).parent)
            if folder == ".":
                folder = ""
            kept_by_folder[folder].append(record)
    for folder, group in kept_by_folder.items():
        if len(group) <= max_per_folder:
            continue
        if max_per_folder == 1:
            chosen = {group[0]["path"]}
        else:
            last = len(group) - 1
            indexes = {round(i * last / (max_per_folder - 1)) for i in range(max_per_folder)}
            chosen = {group[i]["path"] for i in indexes}
        canonical = sorted(chosen)[0]
        for record in group:
            if record["path"] not in chosen:
                record["kept"] = False
                record["duplicate_of"] = canonical
                record["reason"] = "max_per_folder"
    return records


def dedupe(input_dir: Path, hash_size: int, threshold: int, max_per_folder: int):
    tree = BKTree()
    records = []
    images = list(iter_images(input_dir))
    for index, path in enumerate(images, start=1):
        rel = rel_posix(path, input_dir)
        try:
            digest = phash_image(path, hash_size)
        except Exception as exc:  # unreadable file, keep going
            records.append({
                "path": rel,
                "phash": None,
                "kept": False,
                "duplicate_of": None,
                "distance": None,
                "reason": f"unreadable: {exc}",
            })
            continue
        hit = tree.nearest_within(digest, threshold)
        if hit is None:
            tree.add(digest, rel)
            records.append({
                "path": rel,
                "phash": str(digest),
                "kept": True,
                "duplicate_of": None,
                "distance": None,
                "reason": "kept",
            })
        else:
            distance, canonical = hit
            records.append({
                "path": rel,
                "phash": str(digest),
                "kept": False,
                "duplicate_of": canonical,
                "distance": distance,
                "reason": "near_duplicate",
            })
        if index % 500 == 0 or index == len(images):
            print(f"hashed {index}/{len(images)}", file=sys.stderr)
    select_kept(records, max_per_folder)
    return records


def materialize(records, input_dir: Path, output_dir: Path, copy: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for record in records:
        if not record["kept"]:
            continue
        src = input_dir / record["path"]
        dest = output_dir / record["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            written += 1
            continue
        if copy:
            shutil.copy2(src, dest)
        else:
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)
        written += 1
    return written


def write_manifest(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Directory of frames, searched recursively")
    parser.add_argument("--manifest", required=True, type=Path, help="JSONL report of kept and dropped frames")
    parser.add_argument("--output", type=Path, help="Directory to hardlink kept frames into (copied if hardlink fails)")
    parser.add_argument("--copy", action="store_true", help="Copy kept frames instead of hardlinking")
    parser.add_argument("--hash-size", type=int, default=8, help="phash size. 8 is 64 bits")
    parser.add_argument("--threshold", type=int, default=6, help="Drop a frame when its hamming distance to a kept frame is at or below this")
    parser.add_argument("--max-per-folder", type=int, default=0, help="After dedupe, keep at most this many frames from each folder (0 = no cap)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    input_dir = args.input.resolve()
    if not input_dir.is_dir():
        print(f"input directory not found: {input_dir}", file=sys.stderr)
        return 2
    records = dedupe(input_dir, args.hash_size, args.threshold, args.max_per_folder)
    write_manifest(records, args.manifest)
    kept = sum(1 for record in records if record["kept"])
    print(f"kept {kept} of {len(records)} frames -> {args.manifest}")
    if args.output:
        written = materialize(records, input_dir, args.output.resolve(), args.copy)
        print(f"wrote {written} frames -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
