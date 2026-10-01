"""Convert pipeline JSON labels to a YOLO detection dataset.

Splits are by match folder, not by frame, so near-duplicate screenshots from
one match do not land in both train and val.

    python -m tools.annotate.to_yolo --images dataset/images --labels dataset/labels --out dataset/yolo

Class order is wall, bush, water, projectile.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .boxes import CLASSES, iter_images, label_path_for, load_label_file, match_id_from_rel, rel_posix, xyxy_to_yolo


def assign_splits(image_rels, val_fraction: float, seed: int):
    """Return (split_by_rel, warning). A match folder is never split when there are two or more."""
    groups = defaultdict(list)
    for rel in image_rels:
        groups[match_id_from_rel(rel)].append(rel)
    rng = random.Random(seed)
    if len(groups) <= 1:
        rels = list(image_rels)
        rng.shuffle(rels)
        val_count = int(round(len(rels) * val_fraction))
        val_count = min(max(val_count, 1 if len(rels) > 1 else 0), max(0, len(rels) - 1))
        val = set(rels[:val_count])
        splits = {rel: ("val" if rel in val else "train") for rel in image_rels}
        warning = "only one match folder; frames were split randomly, so similar frames may be in both train and val"
        return splits, warning

    order = list(groups)
    rng.shuffle(order)
    order.sort(key=lambda match: len(groups[match]))
    total = len(image_rels)
    target = total * val_fraction
    val_matches = set()
    count = 0
    for match in order:
        if len(val_matches) == len(groups) - 1:
            break
        size = len(groups[match])
        if val_matches and count >= target:
            break
        if val_matches and count + size > max(target * 1.5, target + 1) and count >= target * 0.5:
            continue
        val_matches.add(match)
        count += size
    if not val_matches:
        smallest = min(groups, key=lambda match: (len(groups[match]), match))
        val_matches.add(smallest)
    splits = {}
    for match, rels in groups.items():
        split = "val" if match in val_matches else "train"
        for rel in rels:
            splits[rel] = split
    return splits, None


def link_image(src: Path, dest: Path, copy: bool):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy:
        import shutil
        shutil.copy2(src, dest)
        return
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        import shutil
        shutil.copy2(src, dest)


def export_dataset(images_dir: Path, labels_dir: Path, out_dir: Path, val_fraction: float, seed: int, copy: bool):
    labeled = []
    missing = 0
    for path in iter_images(images_dir):
        rel = rel_posix(path, images_dir)
        label_file = label_path_for(rel, labels_dir)
        if not label_file.is_file():
            missing += 1
            continue
        labeled.append((rel, path, label_file))
    if not labeled:
        raise SystemExit(f"no labeled images under {images_dir} with json in {labels_dir}")

    splits, warning = assign_splits([rel for rel, _, _ in labeled], val_fraction, seed)
    counts = {split: defaultdict(int) for split in ("train", "val")}
    images_per_split = {"train": 0, "val": 0}

    for rel, path, label_file in labeled:
        split = splits[rel]
        record = load_label_file(label_file)
        with Image.open(path) as image:
            width, height = image.size
        if record["width"] and record["height"] and (record["width"], record["height"]) != (width, height):
            print(f"size mismatch {rel}: label {record['width']}x{record['height']} image {width}x{height}", file=sys.stderr)
        lines = []
        for item in record["boxes"]:
            label = item.get("label")
            if label not in CLASSES:
                continue
            box = item.get("box")
            if not box or len(box) != 4:
                continue
            cx, cy, bw, bh = xyxy_to_yolo(box, width, height)
            if bw <= 0 or bh <= 0:
                continue
            class_id = CLASSES.index(label)
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            counts[split][label] += 1
        dest_image = out_dir / "images" / split / rel
        dest_label = out_dir / "labels" / split / Path(rel).with_suffix(".txt")
        link_image(path, dest_image, copy)
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        dest_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        images_per_split[split] += 1

    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    data_yaml = (
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names}\n"
    )
    (out_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    return {
        "images": images_per_split,
        "boxes": {split: dict(counts[split]) for split in counts},
        "missing_labels": missing,
        "warning": warning,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path, help="JSON labels from gemini_label or hand review")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = export_dataset(
        args.images.resolve(),
        args.labels.resolve(),
        args.out.resolve(),
        args.val_fraction,
        args.seed,
        args.copy,
    )
    print(f"train images: {summary['images']['train']}")
    print(f"val images:   {summary['images']['val']}")
    print(f"missing label files skipped: {summary['missing_labels']}")
    for split in ("train", "val"):
        boxes = summary["boxes"][split]
        rendered = ", ".join(f"{name}={boxes.get(name, 0)}" for name in CLASSES)
        print(f"{split} boxes: {rendered}")
    if summary["warning"]:
        print(f"warning: {summary['warning']}", file=sys.stderr)
    print(f"wrote {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
