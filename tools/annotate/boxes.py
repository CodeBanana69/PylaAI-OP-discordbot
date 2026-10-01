"""Box geometry, cleanup, and the on-disk label format shared by the pipeline."""

from __future__ import annotations

import json
from pathlib import Path

CLASSES = ("wall", "bush", "water", "projectile")
TERRAIN_LABELS = ("wall", "water", "bush")
PROJECTILE_LABELS = ("projectile",)
PASS_LABELS = {
    "terrain": TERRAIN_LABELS,
    "projectile": PROJECTILE_LABELS,
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def iter_images(root: Path):
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def label_path_for(image_rel: str, labels_dir: Path) -> Path:
    return Path(labels_dir) / Path(image_rel).with_suffix(".json")


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def spans(length: int, n: int, overlap: float):
    """n tiles along one axis. overlap is the fraction of a tile shared with the next."""
    if n <= 1:
        return [(0, int(length))]
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    tile = length / (n - (n - 1) * overlap)
    out = []
    for i in range(n):
        if i == n - 1:
            end = int(length)
            start = max(0, int(round(end - tile)))
        else:
            start = int(round(i * tile * (1 - overlap)))
            end = min(int(length), int(round(start + tile)))
        if end <= start:
            end = min(int(length), start + 1)
        out.append((start, end))
    return out


def grid_crops(width: int, height: int, rows: int, cols: int, overlap: float):
    xs = spans(width, cols, overlap)
    ys = spans(height, rows, overlap)
    crops = []
    for y0, y1 in ys:
        for x0, x1 in xs:
            crops.append((x0, y0, x1, y1))
    return crops


def _as_floats(values):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


def box_2d_to_xyxy(box_2d, width: int, height: int):
    """[ymin, xmin, ymax, xmax] in 0-1000, 0-1, or pixels -> pixel xyxy."""
    vals = _as_floats(box_2d)
    if vals is None:
        return None
    ymin, xmin, ymax, xmax = vals
    peak = max(abs(v) for v in vals)
    if peak <= 1.0:
        xmin, xmax = xmin * width, xmax * width
        ymin, ymax = ymin * height, ymax * height
    elif peak <= 1000:
        xmin, xmax = xmin / 1000.0 * width, xmax / 1000.0 * width
        ymin, ymax = ymin / 1000.0 * height, ymax / 1000.0 * height
    return _clamp_xyxy([xmin, ymin, xmax, ymax], width, height)


def _clamp_xyxy(box, width: int, height: int):
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def parse_detection_list(payload, width: int, height: int, allowed):
    """Turn a model JSON value into pixel boxes. Raises ValueError on a bad payload."""
    if isinstance(payload, dict):
        if "boxes" in payload:
            payload = payload["boxes"]
        elif "label" in payload:
            payload = [payload]
        else:
            raise ValueError("object has no boxes")
    if not isinstance(payload, list):
        raise ValueError("expected a JSON array")
    boxes = []
    allowed_set = set(allowed)
    for item in payload:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if label not in allowed_set:
            continue
        if "box_2d" in item:
            xyxy = box_2d_to_xyxy(item["box_2d"], width, height)
        elif "box" in item:
            raw = _as_floats(item["box"])
            xyxy = _clamp_xyxy(raw, width, height) if raw is not None else None
        else:
            continue
        if xyxy is None:
            continue
        boxes.append({"label": label, "box": xyxy})
    return boxes


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def loads_model_json(text: str):
    """Parse model output. A truncated array is an error, not a partial label."""
    if text is None or not str(text).strip():
        raise ValueError("empty response")
    cleaned = strip_json_fence(str(text))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            fragment = cleaned[start : end + 1]
            if fragment != cleaned:
                try:
                    return json.loads(fragment)
                except json.JSONDecodeError:
                    pass
        if cleaned.find("[") >= 0 and "]" not in cleaned[cleaned.find("[") :]:
            raise ValueError("truncated json") from exc
        raise ValueError(f"invalid json: {exc}") from exc


def clean_boxes(boxes, width: int, height: int, max_area_fraction=None, min_side: float = 2.0):
    image_area = float(width * height) if width and height else 0.0
    kept = []
    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        if (x2 - x1) < min_side or (y2 - y1) < min_side:
            continue
        if max_area_fraction is not None and image_area > 0:
            if (x2 - x1) * (y2 - y1) > max_area_fraction * image_area:
                continue
        kept.append(item)
    return kept


def merge_duplicates(boxes, iou_threshold: float):
    """Average same-label boxes that overlap above iou_threshold."""
    if iou_threshold <= 0:
        return list(boxes)
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i]["label"] != boxes[j]["label"]:
                continue
            if iou(boxes[i]["box"], boxes[j]["box"]) >= iou_threshold:
                union(i, j)
    groups = {}
    for i, item in enumerate(boxes):
        groups.setdefault(find(i), []).append(item)
    merged = []
    for group in groups.values():
        n = float(len(group))
        acc = [0.0, 0.0, 0.0, 0.0]
        for item in group:
            for k, value in enumerate(item["box"]):
                acc[k] += value
        merged.append({
            "label": group[0]["label"],
            "box": [round(v / n, 1) for v in acc],
        })
    return merged


def offset_box(box, crop):
    x0, y0, _, _ = crop
    x1, y1, x2, y2 = box
    return [round(x1 + x0, 1), round(y1 + y0, 1), round(x2 + x0, 1), round(y2 + y0, 1)]


def xyxy_to_yolo(box, width: int, height: int):
    x1, y1, x2, y2 = box
    bw = max(0.0, x2 - x1) / width
    bh = max(0.0, y2 - y1) / height
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return cx, cy, bw, bh


def match_id_from_rel(image_rel: str) -> str:
    parts = Path(image_rel).parts
    if len(parts) >= 2:
        return parts[0]
    return ""


def load_label_file(path: Path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        boxes = data
        width = height = None
        image = None
    else:
        boxes = data.get("boxes", [])
        width = data.get("width")
        height = data.get("height")
        image = data.get("image")
    return {"image": image, "width": width, "height": height, "boxes": boxes}


def save_label_file(path: Path, image_rel: str, width: int, height: int, boxes, sources=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": image_rel,
        "width": int(width),
        "height": int(height),
        "boxes": boxes,
    }
    if sources is not None:
        payload["sources"] = sources
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def greedy_match(gt_boxes, pred_boxes, iou_threshold: float):
    pairs = []
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            score = iou(gt, pred)
            if score >= iou_threshold:
                pairs.append((score, i, j))
    pairs.sort(reverse=True)
    used_gt, used_pred = set(), set()
    matched = []
    for score, i, j in pairs:
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        matched.append((i, j, score))
    return matched
