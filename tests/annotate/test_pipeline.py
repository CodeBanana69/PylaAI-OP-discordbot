import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from tools.annotate.boxes import (
    box_2d_to_xyxy,
    clean_boxes,
    grid_crops,
    greedy_match,
    iou,
    loads_model_json,
    merge_duplicates,
    offset_box,
    parse_detection_list,
    xyxy_to_yolo,
)
from tools.annotate.dedupe import BKTree, dedupe
from tools.annotate.gemini_label import (
    boxes_from_text,
    build_requests,
    extract_response_text,
    prepare,
    write_labels,
)
from tools.annotate.score import score_sets
from tools.annotate.to_yolo import assign_splits, export_dataset


def hash_from_bits(bits: str):
    import imagehash

    size = int(len(bits) ** 0.5)
    array = np.array([[ch == "1" for ch in bits[i : i + size]] for i in range(0, len(bits), size)])
    return imagehash.ImageHash(array)


class BoxTests(unittest.TestCase):
    def test_iou_identical_and_disjoint(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_box_2d_is_ymin_xmin_ymax_xmax(self):
        xyxy = box_2d_to_xyxy([100, 200, 400, 500], 1000, 500)
        self.assertEqual(xyxy, [200.0, 50.0, 500.0, 200.0])

    def test_unit_and_pixel_fallbacks(self):
        self.assertEqual(box_2d_to_xyxy([0.1, 0.2, 0.4, 0.5], 100, 50), [20.0, 5.0, 50.0, 20.0])
        pixels = box_2d_to_xyxy([10, 20, 40, 1500], 2000, 1000)
        self.assertEqual(pixels, [20.0, 10.0, 1500.0, 40.0])

    def test_crops_cover_the_frame_and_overlap(self):
        crops = grid_crops(1000, 500, 2, 2, 0.2)
        self.assertEqual(len(crops), 4)
        self.assertTrue(any(crop[0] == 0 and crop[1] == 0 for crop in crops))
        self.assertTrue(any(crop[2] == 1000 and crop[3] == 500 for crop in crops))
        xs = sorted({(c[0], c[2]) for c in crops})
        self.assertGreater(xs[0][1], xs[1][0])

    def test_crop_box_moves_back_to_the_full_frame(self):
        args = Namespace(max_projectile_fraction=0.2, duplicate_iou=0.8)
        text = json.dumps([{"label": "projectile", "box_2d": [0, 0, 500, 500]}])
        boxes = boxes_from_text(text, "projectile", 100, 100, (40, 10, 100, 60), args)
        self.assertEqual(boxes[0]["box"], [40.0, 10.0, 70.0, 35.0])

    def test_rejects_truncated_json_and_unknown_labels(self):
        with self.assertRaises(ValueError):
            loads_model_json('[{"label": "wall", "box_2d": [0, 0, 10, 10]}')
        parsed = loads_model_json('```json\n[{"label": "wall", "box_2d": [0, 0, 100, 100]}]\n```')
        boxes = parse_detection_list(parsed, 100, 100, ("wall", "water", "bush"))
        self.assertEqual(boxes[0]["label"], "wall")
        extra = parse_detection_list(
            [{"label": "brawler", "box_2d": [0, 0, 100, 100]}, {"label": "water", "box_2d": [0, 0, 10, 10]}],
            100,
            100,
            ("wall", "water", "bush"),
        )
        self.assertEqual([item["label"] for item in extra], ["water"])

    def test_drops_huge_projectiles_and_merges_duplicates(self):
        boxes = clean_boxes(
            [{"label": "projectile", "box": [0, 0, 80, 80]}, {"label": "projectile", "box": [0, 0, 5, 5]}],
            100,
            100,
            max_area_fraction=0.2,
        )
        self.assertEqual(len(boxes), 1)
        merged = merge_duplicates(
            [
                {"label": "projectile", "box": [0, 0, 10, 10]},
                {"label": "projectile", "box": [1, 1, 11, 11]},
                {"label": "wall", "box": [0, 0, 10, 10]},
            ],
            0.5,
        )
        by_label = {item["label"]: item["box"] for item in merged}
        self.assertEqual(by_label["projectile"], [0.5, 0.5, 10.5, 10.5])
        self.assertEqual(by_label["wall"], [0, 0, 10, 10])

    def test_yolo_line_is_normalized_center(self):
        self.assertEqual(xyxy_to_yolo([0, 0, 50, 20], 100, 100), (0.25, 0.1, 0.5, 0.2))

    def test_greedy_match_is_per_call_and_one_to_one(self):
        matched = greedy_match([[0, 0, 10, 10], [50, 50, 60, 60]], [[1, 1, 11, 11], [1, 1, 9, 9]], 0.5)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0][0], 0)


class DedupeTests(unittest.TestCase):
    def test_bk_tree_respects_distance(self):
        tree = BKTree()
        bits = "0" * 64
        tree.add(hash_from_bits(bits), "a")
        near = hash_from_bits("1" + "0" * 63)
        far = hash_from_bits("1" * 20 + "0" * 44)
        self.assertEqual(tree.nearest_within(near, 2)[1], "a")
        self.assertIsNone(tree.nearest_within(far, 6))

    def test_identical_frames_collapse(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            match = root / "match-a"
            match.mkdir()
            frame = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            Image.fromarray(frame).save(match / "0001.png")
            Image.fromarray(frame).save(match / "0002.png")
            other = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            Image.fromarray(other).save(match / "0003.png")
            records = dedupe(root, hash_size=8, threshold=6, max_per_folder=0)
            kept = [record["path"] for record in records if record["kept"]]
            dropped = [record for record in records if not record["kept"]]
            self.assertEqual(len(kept), 2)
            self.assertEqual(len(dropped), 1)
            self.assertEqual(dropped[0]["reason"], "near_duplicate")


class SplitAndScoreTests(unittest.TestCase):
    def test_matches_are_not_split(self):
        rels = [f"big/{i}.png" for i in range(9)] + ["small/0.png"]
        splits, warning = assign_splits(rels, 0.1, seed=1)
        self.assertIsNone(warning)
        self.assertEqual(splits["small/0.png"], "val")
        self.assertTrue(all(splits[rel] == "train" for rel in rels if rel.startswith("big/")))

    def test_export_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            labels = root / "labels"
            for rel, box in (("m1/a.png", [10, 10, 40, 30]), ("m2/b.png", [5, 5, 15, 20])):
                path = images / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (100, 80), (20, 40, 60)).save(path)
                payload = {"image": rel, "width": 100, "height": 80, "boxes": [{"label": "wall", "box": box}]}
                dest = labels / Path(rel).with_suffix(".json")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(payload), encoding="utf-8")
            summary = export_dataset(images, labels, root / "yolo", 0.5, 0, copy=True)
            self.assertEqual(summary["images"]["train"] + summary["images"]["val"], 2)
            train_matches = {p.parts[-2] for p in (root / "yolo" / "images" / "train").rglob("*.png")}
            val_matches = {p.parts[-2] for p in (root / "yolo" / "images" / "val").rglob("*.png")}
            self.assertFalse(train_matches & val_matches)
            yaml_text = (root / "yolo" / "data.yaml").read_text(encoding="utf-8")
            self.assertIn("0: wall", yaml_text)
            self.assertIn("3: projectile", yaml_text)

            pred = root / "pred"
            good = {"image": "m1/a.png", "width": 100, "height": 80, "boxes": [{"label": "wall", "box": [10, 10, 40, 30]}]}
            bad = {"image": "m2/b.png", "width": 100, "height": 80, "boxes": [{"label": "bush", "box": [5, 5, 15, 20]}]}
            for rel, payload in (("m1/a.json", good), ("m2/b.json", bad)):
                path = pred / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")
            result = score_sets(labels, pred, 0.5)
            self.assertEqual(result["per_class"]["wall"]["tp"], 1)
            self.assertEqual(result["per_class"]["wall"]["fn"], 1)
            self.assertEqual(result["per_class"]["bush"]["fp"], 1)


class GeminiPrepareTests(unittest.TestCase):
    def test_prepare_writes_terrain_and_crops_without_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images" / "match"
            images.mkdir(parents=True)
            Image.new("RGB", (32, 24), (10, 20, 30)).save(images / "frame.png")
            args = Namespace(
                images=root / "images",
                out=root / "labels",
                workdir=root / "work",
                examples=None,
                examples_limit=2,
                jpeg_quality=90,
                rows=2,
                cols=2,
                overlap=0.2,
                limit=0,
                force=False,
                terrain_thinking="MINIMAL",
                projectile_thinking="LOW",
                temperature=None,
                max_output_tokens=8192,
                max_projectile_fraction=0.2,
                duplicate_iou=0.8,
                crop_merge_iou=0.45,
                allow_partial=False,
                images_per_file=300,
                max_file_bytes=1_500_000_000,
                model="gemini-3.8-flash",
            )
            self.assertEqual(prepare(args), 0)
            lines = (root / "work" / "requests" / "batch_0000.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 5)
            tasks = []
            for line in lines:
                record = json.loads(line)
                parts = record["request"]["contents"][-1]["parts"]
                self.assertIn("inline_data", parts[0])
                self.assertIn("text", parts[1])
                enum = record["request"]["generation_config"]["response_schema"]["items"]["properties"]["label"]["enum"]
                tasks.append(tuple(enum))
            self.assertIn(("wall", "water", "bush"), tasks)
            self.assertEqual(tasks.count(("projectile",)), 4)

    def test_collect_merges_duplicate_crop_hits(self):
        args = Namespace(max_projectile_fraction=0.2, duplicate_iou=0.8, crop_merge_iou=0.45, allow_partial=False)
        records = [
            {"key": "t", "image": "m/a.png", "task": "terrain", "crop": None, "width": 100, "height": 100},
            {"key": "p0", "image": "m/a.png", "task": "projectile", "crop": [0, 0, 60, 60], "width": 100, "height": 100},
            {"key": "p1", "image": "m/a.png", "task": "projectile", "crop": [40, 40, 100, 100], "width": 100, "height": 100},
        ]
        wall = json.dumps([{"label": "wall", "box_2d": [0, 0, 200, 200]}])
        # Both crops describe the same full-frame box [48, 48, 58, 58].
        shot_a = json.dumps([{"label": "projectile", "box_2d": [800, 800, 967, 967]}])
        shot_b = json.dumps([{"label": "projectile", "box_2d": [133, 133, 300, 300]}])
        texts = {"t": (wall, None), "p0": (shot_a, None), "p1": (shot_b, None)}
        with tempfile.TemporaryDirectory() as tmp:
            written, failed, counts = write_labels(["m/a.png"], {"m/a.png": records}, texts, args, Path(tmp), Path(tmp) / "fail.jsonl")
        self.assertEqual((written, failed), (1, 0))
        self.assertEqual(counts["wall"], 1)
        self.assertEqual(counts["projectile"], 1)

    def test_result_line_skips_thoughts_and_flags_truncation(self):
        text, error = extract_response_text({
            "key": "r",
            "response": {"candidates": [{"content": {"parts": [
                {"text": "hidden", "thought": True},
                {"text": "[]"},
            ]}}]},
        })
        self.assertEqual((text, error), ("[]", None))
        text, error = extract_response_text({
            "response": {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "["}]}}]},
        })
        self.assertEqual(text, None)
        self.assertEqual(error, "truncated")

    def test_requests_put_the_image_before_the_prompt(self):
        image = Image.new("RGB", (16, 16), (1, 2, 3))
        args = Namespace(
            jpeg_quality=90,
            rows=2,
            cols=2,
            overlap=0.2,
            terrain_thinking="MINIMAL",
            projectile_thinking="LOW",
            temperature=None,
            max_output_tokens=100,
            examples_limit=0,
        )
        requests = list(build_requests("a.png", image, args, []))
        self.assertEqual(len(requests), 5)
        _suffix, task, crop, request = requests[0]
        self.assertEqual((task, crop), ("terrain", None))
        self.assertIn("inline_data", request["contents"][0]["parts"][0])


class OffsetTests(unittest.TestCase):
    def test_offset_box(self):
        self.assertEqual(offset_box([1, 2, 3, 4], (10, 20, 30, 40)), [11, 22, 13, 24])


if __name__ == "__main__":
    unittest.main()
