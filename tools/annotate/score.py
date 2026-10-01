"""Score predictions against a hand-labeled gold set.

Matching is greedy, per class, at IoU 0.5 by default. Images are paired by
relative path.

    python -m tools.annotate.score --gold gold/labels --pred pred/labels --images gold/images --draw review
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .boxes import CLASSES, greedy_match, iter_images, load_label_file, rel_posix

COLORS = {
    "wall": (255, 140, 0),
    "bush": (40, 180, 40),
    "water": (30, 120, 255),
    "projectile": (255, 60, 180),
}


def boxes_for(record, label):
    return [item["box"] for item in record["boxes"] if item.get("label") == label and item.get("box")]


def score_sets(gold_dir: Path, pred_dir: Path, iou_threshold: float):
    gold_files = {path.relative_to(gold_dir).as_posix(): path for path in gold_dir.rglob("*.json")}
    pred_files = {path.relative_to(pred_dir).as_posix(): path for path in pred_dir.rglob("*.json")}
    shared = sorted(set(gold_files) & set(pred_files))
    per_class = {label: {"tp": 0, "fp": 0, "fn": 0} for label in CLASSES}
    errors = []
    for rel in shared:
        gold = load_label_file(gold_files[rel])
        pred = load_label_file(pred_files[rel])
        image_errors = {"image": gold.get("image") or rel.replace(".json", ""), "label_file": rel, "false_positives": [], "misses": []}
        for label in CLASSES:
            gt = boxes_for(gold, label)
            pr = boxes_for(pred, label)
            matched = greedy_match(gt, pr, iou_threshold)
            used_gt = {i for i, _, _ in matched}
            used_pred = {j for _, j, _ in matched}
            per_class[label]["tp"] += len(matched)
            per_class[label]["fp"] += len(pr) - len(used_pred)
            per_class[label]["fn"] += len(gt) - len(used_gt)
            for j, box in enumerate(pr):
                if j not in used_pred:
                    image_errors["false_positives"].append({"label": label, "box": box})
            for i, box in enumerate(gt):
                if i not in used_gt:
                    image_errors["misses"].append({"label": label, "box": box})
        if image_errors["false_positives"] or image_errors["misses"]:
            errors.append(image_errors)
    return {
        "per_class": per_class,
        "compared": len(shared),
        "gold_only": sorted(set(gold_files) - set(pred_files)),
        "pred_only": sorted(set(pred_files) - set(gold_files)),
        "errors": errors,
    }


def _divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def format_report(result, iou_threshold: float) -> str:
    lines = [f"compared {result['compared']} images at IoU {iou_threshold:.2f}"]
    header = f"{'class':<12} {'gold':>6} {'pred':>6} {'tp':>6} {'fp':>6} {'fn':>6} {'prec':>7} {'rec':>7} {'f1':>7}"
    lines.append(header)
    for label in list(CLASSES) + ["all"]:
        if label == "all":
            stats = {key: sum(result["per_class"][name][key] for name in CLASSES) for key in ("tp", "fp", "fn")}
        else:
            stats = result["per_class"][label]
        gold = stats["tp"] + stats["fn"]
        pred = stats["tp"] + stats["fp"]
        precision = _divide(stats["tp"], pred)
        recall = _divide(stats["tp"], gold)
        f1 = _divide(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        lines.append(
            f"{label:<12} {gold:6d} {pred:6d} {stats['tp']:6d} {stats['fp']:6d} {stats['fn']:6d} "
            f"{precision:7.3f} {recall:7.3f} {f1:7.3f}"
        )
    if result["gold_only"]:
        lines.append(f"gold labels with no prediction: {len(result['gold_only'])}")
    if result["pred_only"]:
        lines.append(f"predictions with no gold label: {len(result['pred_only'])}")
    return "\n".join(lines)


def draw_errors(result, images_dir: Path, draw_dir: Path):
    from PIL import Image, ImageDraw

    draw_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    images = {rel_posix(path, images_dir): path for path in iter_images(images_dir)}
    for error in result["errors"]:
        rel = error["image"]
        path = images.get(rel)
        if path is None:
            candidate = images_dir / rel
            path = candidate if candidate.is_file() else None
        if path is None:
            continue
        with Image.open(path) as image:
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for kind, color_scale in (("misses", 1.0), ("false_positives", 0.55)):
            for item in error[kind]:
                color = tuple(int(channel * color_scale) for channel in COLORS.get(item["label"], (255, 255, 255)))
                x1, y1, x2, y2 = item["box"]
                width = 3 if kind == "misses" else 2
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                tag = ("miss " if kind == "misses" else "fp ") + item["label"]
                draw.text((x1 + 2, max(0, y1 - 12)), tag, fill=color)
        dest = draw_dir / Path(rel).with_suffix(".jpg")
        dest.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest, quality=90)
        written += 1
    return written


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", required=True, type=Path, help="Hand-corrected JSON labels")
    parser.add_argument("--pred", required=True, type=Path, help="Predicted JSON labels in the same format")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--errors", type=Path, help="Write misses and false positives as JSON")
    parser.add_argument("--images", type=Path, help="Image root, required with --draw")
    parser.add_argument("--draw", type=Path, help="Directory of review images. Misses are bright, false positives are dimmer")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.gold.is_dir() or not args.pred.is_dir():
        print("gold and pred must be directories of json labels", file=sys.stderr)
        return 2
    result = score_sets(args.gold.resolve(), args.pred.resolve(), args.iou)
    print(format_report(result, args.iou))
    if args.errors:
        args.errors.parent.mkdir(parents=True, exist_ok=True)
        args.errors.write_text(json.dumps(result["errors"], indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.errors}")
    if args.draw:
        if not args.images:
            print("--draw requires --images", file=sys.stderr)
            return 2
        written = draw_errors(result, args.images.resolve(), args.draw.resolve())
        print(f"drew {written} review images -> {args.draw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
