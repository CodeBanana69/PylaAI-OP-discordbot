"""Label screenshots with Gemini. Terrain on the full frame, projectiles on crops.

Does not use the bot's existing detector. Reads GEMINI_API_KEY or GOOGLE_API_KEY.

Sync, for a gold set or a smoke test:

    python -m tools.annotate.gemini_label sync --images dataset/images --out dataset/labels --limit 20

Batch, for a few thousand seed images (half the interactive price):

    python -m tools.annotate.gemini_label prepare --images dataset/images --out dataset/labels --workdir dataset/gemini
    python -m tools.annotate.gemini_label submit --workdir dataset/gemini
    python -m tools.annotate.gemini_label collect --workdir dataset/gemini --wait

Labels are JSON: {"image", "width", "height", "boxes": [{"label", "box": [x1, y1, x2, y2]}]}.
box is pixels, origin top-left. Few-shot examples are image/json pairs with that same format.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from .boxes import (
    PASS_LABELS,
    clean_boxes,
    grid_crops,
    label_path_for,
    loads_model_json,
    merge_duplicates,
    offset_box,
    parse_detection_list,
    rel_posix,
    save_label_file,
)
from .boxes import iter_images
from .prompts import PROMPTS

DONE_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def response_schema(labels):
    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "label": {"type": "STRING", "enum": list(labels)},
                "box_2d": {
                    "type": "ARRAY",
                    "items": {"type": "INTEGER"},
                    "minItems": 4,
                    "maxItems": 4,
                },
            },
            "required": ["label", "box_2d"],
        },
    }


def generation_config(task: str, args) -> dict:
    thinking = args.terrain_thinking if task == "terrain" else args.projectile_thinking
    config = {
        "response_mime_type": "application/json",
        "response_schema": response_schema(PASS_LABELS[task]),
        "media_resolution": "MEDIA_RESOLUTION_HIGH",
        "max_output_tokens": args.max_output_tokens,
        "thinking_config": {"thinking_level": thinking},
    }
    if args.temperature is not None:
        config["temperature"] = args.temperature
    return config


def jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def load_examples(path: Path | None):
    if path is None:
        return []
    examples = []
    for image_path in iter_images(path):
        json_path = image_path.with_suffix(".json")
        if not json_path.is_file():
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        boxes = payload["boxes"] if isinstance(payload, dict) else payload
        with Image.open(image_path) as image:
            width, height = image.size
            encoded = jpeg_bytes(image, 92)
        examples.append({
            "name": image_path.name,
            "width": width,
            "height": height,
            "jpeg": encoded,
            "boxes": boxes,
        })
    return examples


def example_answer(example, allowed) -> str:
    rendered = []
    allowed_set = set(allowed)
    for item in example["boxes"]:
        label = item.get("label")
        if label not in allowed_set or "box" not in item:
            continue
        x1, y1, x2, y2 = item["box"]
        width, height = example["width"], example["height"]
        rendered.append({
            "label": label,
            "box_2d": [
                int(round(y1 / height * 1000)),
                int(round(x1 / width * 1000)),
                int(round(y2 / height * 1000)),
                int(round(x2 / width * 1000)),
            ],
        })
    return json.dumps(rendered)


def contents_for(task: str, jpeg: bytes, examples, limit: int):
    allowed = PASS_LABELS[task]
    prompt = PROMPTS[task]
    contents = []
    used = 0
    for example in examples:
        if used >= limit:
            break
        answer = example_answer(example, allowed)
        if answer == "[]":
            continue
        contents.append({
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(example["jpeg"]).decode("ascii")}},
                {"text": prompt},
            ],
        })
        contents.append({"role": "model", "parts": [{"text": answer}]})
        used += 1
    contents.append({
        "role": "user",
        "parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode("ascii")}},
            {"text": prompt},
        ],
    })
    return contents


def build_requests(image_rel, image: Image.Image, args, examples):
    """One terrain request plus one projectile request per crop. Yields (key_suffix, task, crop, request)."""
    width, height = image.size
    full = jpeg_bytes(image, args.jpeg_quality)
    terrain = {
        "contents": contents_for("terrain", full, examples, args.examples_limit),
        "generation_config": generation_config("terrain", args),
    }
    yield ("terrain", "terrain", None, terrain)
    for index, crop in enumerate(grid_crops(width, height, args.rows, args.cols, args.overlap)):
        x0, y0, x1, y1 = crop
        tile = jpeg_bytes(image.crop(crop), args.jpeg_quality)
        request = {
            "contents": contents_for("projectile", tile, examples, args.examples_limit),
            "generation_config": generation_config("projectile", args),
        }
        yield (f"projectile-{index}", "projectile", crop, request)


def images_to_label(images_dir: Path, labels_dir: Path, limit: int, force: bool):
    chosen = []
    skipped = 0
    for path in iter_images(images_dir):
        rel = rel_posix(path, images_dir)
        if not force and label_path_for(rel, labels_dir).is_file():
            skipped += 1
            continue
        chosen.append((rel, path))
        if limit and len(chosen) >= limit:
            break
    return chosen, skipped


class RequestWriter:
    def __init__(self, workdir: Path, images_per_file: int, max_bytes: int):
        self.workdir = workdir
        self.images_per_file = images_per_file
        self.max_bytes = max_bytes
        self.requests_dir = workdir / "requests"
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = workdir / "index.jsonl"
        self.index_handle = self.index_path.open("w", encoding="utf-8")
        self.file_index = 0
        self.current = None
        self.current_bytes = 0
        self.images_in_file = 0
        self.keys = 0
        self.files = []

    def _open_next(self):
        if self.current is not None:
            self.current.close()
        name = f"batch_{self.file_index:04d}.jsonl"
        self.file_index += 1
        path = self.requests_dir / name
        self.current = path.open("w", encoding="utf-8")
        self.current_bytes = 0
        self.images_in_file = 0
        self.files.append(path)

    def add(self, image_rel: str, width: int, height: int, suffix: str, task: str, crop, request: dict):
        line = json.dumps({"key": f"r{self.keys:07d}", "request": request}, separators=(",", ":")) + "\n"
        encoded = len(line.encode("utf-8"))
        if self.current is None or self.images_in_file >= self.images_per_file or self.current_bytes + encoded > self.max_bytes:
            self._open_next()
        key = f"r{self.keys:07d}"
        self.current.write(line)
        self.current_bytes += encoded
        self.keys += 1
        if suffix == "terrain":
            self.images_in_file += 1
        record = {
            "key": key,
            "image": image_rel,
            "task": task,
            "crop": list(crop) if crop else None,
            "width": width,
            "height": height,
            "file": self.files[-1].name,
        }
        self.index_handle.write(json.dumps(record) + "\n")
        return key

    def close(self):
        if self.current is not None:
            self.current.close()
        self.index_handle.close()


def prepare(args):
    images_dir = args.images.resolve()
    labels_dir = args.out.resolve()
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(args.examples.resolve() if args.examples else None)
    chosen, skipped = images_to_label(images_dir, labels_dir, args.limit, args.force)
    writer = RequestWriter(workdir, args.images_per_file, args.max_file_bytes)
    for rel, path in chosen:
        with Image.open(path) as image:
            image.load()
            for suffix, task, crop, request in build_requests(rel, image, args, examples):
                writer.add(rel, image.size[0], image.size[1], suffix, task, crop, request)
    writer.close()
    state = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "model": args.model,
        "jobs": [],
        "files": [path.name for path in writer.files],
    }
    (workdir / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"images queued: {len(chosen)} (skipped existing labels: {skipped})")
    print(f"requests: {writer.keys} across {len(writer.files)} files")
    print(f"workdir: {workdir}")
    return 0


def load_state(workdir: Path) -> dict:
    path = workdir / "state.json"
    if not path.is_file():
        raise SystemExit(f"no state.json in {workdir}. Run prepare first.")
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(workdir: Path, state: dict):
    (workdir / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def client():
    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit("google-genai is not installed. pip install -r tools/annotate/requirements.txt") from exc
    return genai.Client()


def submit(args):
    workdir = args.workdir.resolve()
    state = load_state(workdir)
    from google.genai import types

    api = client()
    known = {job["file"] for job in state["jobs"]}
    for name in state["files"]:
        if name in known and not args.resubmit:
            print(f"already submitted {name}")
            continue
        path = workdir / "requests" / name
        uploaded = api.files.upload(
            file=str(path),
            config=types.UploadFileConfig(display_name=name, mime_type="jsonl"),
        )
        job = api.batches.create(
            model=state["model"],
            src=uploaded.name,
            config={"display_name": f"brawl-{name}"},
        )
        state["jobs"].append({"file": name, "uploaded": uploaded.name, "name": job.name})
        save_state(workdir, state)
        print(f"submitted {name} -> {job.name}")
    return 0


def job_state_name(job) -> str:
    state = getattr(job, "state", None)
    name = getattr(state, "name", None)
    if name:
        return name
    return str(state)


def status(args):
    workdir = args.workdir.resolve()
    state = load_state(workdir)
    if not state["jobs"]:
        print("no jobs submitted")
        return 0
    api = client()
    for job_info in state["jobs"]:
        job = api.batches.get(name=job_info["name"])
        print(f"{job_info['file']}: {job_state_name(job)} ({job_info['name']})")
    return 0


def extract_response_text(record: dict):
    if not isinstance(record, dict):
        return None, "record is not an object"
    if record.get("error") and "response" not in record:
        return None, json.dumps(record["error"])[:500]
    response = record.get("response", record)
    if isinstance(response, dict) and response.get("error") and "candidates" not in response:
        return None, json.dumps(response["error"])[:500]
    candidates = response.get("candidates") or [] if isinstance(response, dict) else []
    texts = []
    finish = None
    for candidate in candidates:
        finish = candidate.get("finishReason") or candidate.get("finish_reason") or finish
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if part.get("thought"):
                continue
            if part.get("text"):
                texts.append(part["text"])
    if finish and "MAX_TOKENS" in str(finish):
        return None, "truncated"
    if texts:
        return "\n".join(texts), None
    return None, "no text in response"


def record_key(record: dict):
    if isinstance(record, dict):
        if record.get("key"):
            return record["key"]
        metadata = record.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("key"):
            return metadata["key"]
    return None


def parse_result_line(line: str):
    record = json.loads(line)
    return record_key(record), extract_response_text(record)


def boxes_from_text(text: str, task: str, width: int, height: int, crop, args):
    payload = loads_model_json(text)
    boxes = parse_detection_list(payload, width if crop is None else (crop[2] - crop[0]), height if crop is None else (crop[3] - crop[1]), PASS_LABELS[task])
    if crop is not None:
        for item in boxes:
            item["box"] = offset_box(item["box"], crop)
    if task == "projectile":
        boxes = clean_boxes(boxes, width, height, max_area_fraction=args.max_projectile_fraction)
    else:
        boxes = clean_boxes(boxes, width, height)
    return merge_duplicates(boxes, args.duplicate_iou)


def assemble_image(records, texts, args):
    """records: index rows for one image. texts: key -> (text or None, error or None)."""
    width = records[0]["width"]
    height = records[0]["height"]
    terrain = []
    projectiles = []
    sources = {}
    failures = []
    for record in records:
        text, error = texts.get(record["key"], (None, "missing response"))
        if error or text is None:
            failures.append({"image": record["image"], "task": record["task"], "key": record["key"], "error": error or "empty"})
            sources[record["task"]] = error or "empty"
            continue
        try:
            crop = tuple(record["crop"]) if record["crop"] else None
            boxes = boxes_from_text(text, record["task"], width, height, crop, args)
        except ValueError as exc:
            failures.append({"image": record["image"], "task": record["task"], "key": record["key"], "error": str(exc)})
            sources[record["task"]] = str(exc)
            continue
        sources[record["task"] if record["task"] == "terrain" else record["key"]] = "ok"
        if record["task"] == "terrain":
            terrain.extend(boxes)
            sources["terrain"] = "ok"
        else:
            projectiles.extend(boxes)
    projectiles = merge_duplicates(projectiles, args.crop_merge_iou)
    ok = not failures
    return terrain + projectiles, sources, failures, ok


def load_index(workdir: Path):
    grouped = defaultdict(list)
    order = []
    with (workdir / "index.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["image"] not in grouped:
                order.append(record["image"])
            grouped[record["image"]].append(record)
    return order, grouped


def write_labels(order, grouped, texts, args, labels_dir: Path, failures_path: Path):
    written = 0
    failed_images = 0
    class_counts = defaultdict(int)
    with failures_path.open("w", encoding="utf-8") as failures_file:
        for image_rel in order:
            boxes, sources, failures, ok = assemble_image(grouped[image_rel], texts, args)
            for failure in failures:
                failures_file.write(json.dumps(failure) + "\n")
            if not ok and not args.allow_partial:
                failed_images += 1
                continue
            width = grouped[image_rel][0]["width"]
            height = grouped[image_rel][0]["height"]
            save_label_file(label_path_for(image_rel, labels_dir), image_rel, width, height, boxes, sources)
            for item in boxes:
                class_counts[item["label"]] += 1
            written += 1
    return written, failed_images, class_counts


def download_job_texts(api, job):
    texts = {}
    dest = getattr(job, "dest", None)
    file_name = getattr(dest, "file_name", None) if dest else None
    if file_name:
        content = api.files.download(file=file_name)
        if isinstance(content, bytes):
            payload = content.decode("utf-8")
        else:
            payload = str(content)
        for line in payload.splitlines():
            if not line.strip():
                continue
            key, extracted = parse_result_line(line)
            if key:
                texts[key] = extracted
        return texts
    inlined = getattr(dest, "inlined_responses", None) if dest else None
    if not inlined:
        return texts
    for item in inlined:
        response = getattr(item, "response", None)
        error = getattr(item, "error", None)
        key = getattr(item, "metadata", None)
        key = getattr(key, "key", None) if key else None
        if response is not None and getattr(response, "text", None):
            texts[key or str(len(texts))] = (response.text, None)
        elif error is not None:
            texts[key or str(len(texts))] = (None, str(error)[:500])
    return texts


def collect(args):
    workdir = args.workdir.resolve()
    state = load_state(workdir)
    if not state["jobs"]:
        print("no jobs submitted", file=sys.stderr)
        return 2
    api = client()
    texts = {}
    pending = True
    while pending:
        pending = False
        for job_info in state["jobs"]:
            job = api.batches.get(name=job_info["name"])
            name = job_state_name(job)
            print(f"{job_info['file']}: {name}")
            if name not in DONE_STATES:
                pending = True
                continue
            if name != "JOB_STATE_SUCCEEDED":
                print(f"{job_info['file']} ended as {name}", file=sys.stderr)
                continue
            texts.update(download_job_texts(api, job))
        if pending and args.wait:
            time.sleep(args.poll_seconds)
        elif pending:
            print("jobs still running. Re-run collect --wait, or collect again later.")
            return 3
    order, grouped = load_index(workdir)
    raw_dir = workdir / "raw"
    raw_dir.mkdir(exist_ok=True)
    for key, (text, error) in texts.items():
        (raw_dir / f"{key}.txt").write_text(text if text is not None else f"ERROR: {error}", encoding="utf-8")
    labels_dir = Path(state["labels_dir"])
    written, failed_images, class_counts = write_labels(
        order, grouped, texts, args, labels_dir, workdir / "failures.jsonl"
    )
    rendered = ", ".join(f"{label}={class_counts[label]}" for label in ("wall", "bush", "water", "projectile"))
    print(f"wrote {written} label files, {failed_images} images failed")
    print(f"boxes: {rendered}")
    print(f"failures: {workdir / 'failures.jsonl'}")
    return 0 if failed_images == 0 else 1


def sync_one(api, rel, path, args, examples, types_module):
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        results = []
        for _suffix, task, crop, request in build_requests(rel, image, args, examples):
            text, error = generate_once(api, args.model, request, types_module, args.retries)
            results.append({
                "key": f"{rel}:{task}:{crop}",
                "image": rel,
                "task": task,
                "crop": list(crop) if crop else None,
                "width": width,
                "height": height,
                "text": text,
                "error": error,
            })
    return results


def generate_once(api, model, request, types_module, retries: int):
    config = dict(request["generation_config"])
    thinking = config.pop("thinking_config", None)
    schema = config.pop("response_schema")
    config_kwargs = {
        "response_mime_type": config["response_mime_type"],
        "response_schema": schema,
        "media_resolution": config["media_resolution"],
        "max_output_tokens": config["max_output_tokens"],
    }
    if thinking:
        config_kwargs["thinking_config"] = types_module.ThinkingConfig(thinking_level=thinking["thinking_level"])
    if config.get("temperature") is not None:
        config_kwargs["temperature"] = config["temperature"]
    gen_config = types_module.GenerateContentConfig(**config_kwargs)
    contents = []
    for turn in request["contents"]:
        parts = []
        for part in turn["parts"]:
            if "text" in part:
                parts.append(types_module.Part(text=part["text"]))
            else:
                raw = base64.b64decode(part["inline_data"]["data"])
                parts.append(types_module.Part.from_bytes(data=raw, mime_type="image/jpeg"))
        contents.append(types_module.Content(role=turn["role"], parts=parts))
    last_error = "request failed"
    for attempt in range(retries + 1):
        try:
            response = api.models.generate_content(model=model, contents=contents, config=gen_config)
        except Exception as exc:  # network and API errors are retried
            last_error = str(exc)[:500]
            time.sleep(min(2 ** attempt, 30))
            continue
        finish = None
        try:
            finish = str(response.candidates[0].finish_reason)
        except (AttributeError, IndexError, TypeError):
            finish = None
        if finish and "MAX_TOKENS" in finish:
            return None, "truncated"
        text = getattr(response, "text", None)
        if text:
            return text, None
        last_error = "no text in response"
        break
    return None, last_error


def sync(args):
    images_dir = args.images.resolve()
    labels_dir = args.out.resolve()
    examples = load_examples(args.examples.resolve() if args.examples else None)
    chosen, skipped = images_to_label(images_dir, labels_dir, args.limit, args.force)
    if not chosen:
        print(f"nothing to label (skipped existing: {skipped})")
        return 0
    from google.genai import types

    api = client()
    print(f"labeling {len(chosen)} images, skipped existing {skipped}")
    failures = []
    class_counts = defaultdict(int)
    lock = threading.Lock()

    def work(item):
        rel, path = item
        rows = sync_one(api, rel, path, args, examples, types)
        texts = {row["key"]: (row["text"], row["error"]) for row in rows}
        boxes, sources, image_failures, ok = assemble_image(rows, texts, args)
        return rel, rows[0]["width"], rows[0]["height"], boxes, sources, image_failures, ok

    written = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(work, item) for item in chosen]
        for done_index, future in enumerate(as_completed(futures), start=1):
            rel, width, height, boxes, sources, image_failures, ok = future.result()
            with lock:
                failures.extend(image_failures)
                if ok or args.allow_partial:
                    save_label_file(label_path_for(rel, labels_dir), rel, width, height, boxes, sources)
                    for item in boxes:
                        class_counts[item["label"]] += 1
                    written += 1
                print(f"{done_index}/{len(chosen)} {rel} boxes={len(boxes)} {'ok' if ok else 'FAILED'}", file=sys.stderr)
    if failures:
        fail_path = labels_dir / "failures.jsonl"
        with fail_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure) + "\n")
        print(f"failures: {fail_path}")
    rendered = ", ".join(f"{label}={class_counts[label]}" for label in ("wall", "bush", "water", "projectile"))
    print(f"wrote {written} label files")
    print(f"boxes: {rendered}")
    return 0 if not failures else 1


def add_common(parser):
    parser.add_argument("--images", type=Path, help="Frame directory")
    parser.add_argument("--out", type=Path, help="Directory for JSON labels")
    parser.add_argument("--workdir", type=Path, help="Batch request and job state directory")
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--examples", type=Path, help="Directory of image + matching json few-shot pairs")
    parser.add_argument("--examples-limit", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=92, help="Quality of the JPEG sent to Gemini. Resolution is unchanged")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.2, help="Fraction of each projectile crop shared with the next")
    parser.add_argument("--limit", type=int, default=0, help="Label at most this many images (0 = all)")
    parser.add_argument("--force", action="store_true", help="Relabel images that already have a json file")
    parser.add_argument("--terrain-thinking", default="MINIMAL", choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--projectile-thinking", default="LOW", choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--temperature", type=float, default=None, help="Omit to use the API default")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--max-projectile-fraction", type=float, default=0.2, help="Drop a projectile box bigger than this fraction of the frame")
    parser.add_argument("--duplicate-iou", type=float, default=0.8, help="Average same-label boxes above this IoU inside one response")
    parser.add_argument("--crop-merge-iou", type=float, default=0.45, help="Average projectile boxes above this IoU across overlapping crops")
    parser.add_argument("--allow-partial", action="store_true", help="Write a label file even when one request failed")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sync_parser = sub.add_parser("sync", help="Label now, a few images at a time")
    add_common(sync_parser)
    sync_parser.add_argument("--workers", type=int, default=4)
    sync_parser.add_argument("--retries", type=int, default=3)

    prepare_parser = sub.add_parser("prepare", help="Write batch JSONL locally, no API calls")
    add_common(prepare_parser)
    prepare_parser.add_argument("--images-per-file", type=int, default=300)
    prepare_parser.add_argument("--max-file-bytes", type=int, default=1_500_000_000)

    submit_parser = sub.add_parser("submit", help="Upload JSONL and create batch jobs")
    submit_parser.add_argument("--workdir", required=True, type=Path)
    submit_parser.add_argument("--resubmit", action="store_true")

    status_parser = sub.add_parser("status", help="Print batch job states")
    status_parser.add_argument("--workdir", required=True, type=Path)

    collect_parser = sub.add_parser("collect", help="Download batch results and write JSON labels")
    add_common(collect_parser)
    collect_parser.add_argument("--wait", action="store_true")
    collect_parser.add_argument("--poll-seconds", type=int, default=30)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        if not args.images or not args.out or not args.workdir:
            print("prepare requires --images, --out, and --workdir", file=sys.stderr)
            return 2
        return prepare(args)
    if args.command == "sync":
        if not args.images or not args.out:
            print("sync requires --images and --out", file=sys.stderr)
            return 2
        return sync(args)
    if args.command in ("submit", "status", "collect") and not args.workdir:
        print(f"{args.command} requires --workdir", file=sys.stderr)
        return 2
    if args.command == "submit":
        return submit(args)
    if args.command == "status":
        return status(args)
    if args.command == "collect":
        return collect(args)
    print(f"unknown command {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
