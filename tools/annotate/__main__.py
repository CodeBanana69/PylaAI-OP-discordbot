"""Print the labeling workflow.

    python -m tools.annotate
"""

WORKFLOW = """
Label gameplay frames with Gemini, correct a seed set by hand, then train YOLO.
The existing tile detector is not used.

Install once:
    pip install -r tools/annotate/requirements.txt
    export GEMINI_API_KEY=...

1. Drop near-duplicate frames. Keep the match folder as the parent directory
   so the YOLO split can hold a whole match out of training.
    python -m tools.annotate.dedupe \\
        --input frames --manifest dataset/dedupe.jsonl --output dataset/images

2. Hand-label about 200 varied frames into gold/labels. Same JSON format the
   labeler writes. Do not train on these and do not use them as few-shot examples.
    python -m tools.annotate.gemini_label sync \\
        --images gold/images --out gold/pred --limit 20
    python -m tools.annotate.score \\
        --gold gold/labels --pred gold/pred --images gold/images --draw gold/review

3. Optional few-shot: put 1 or 2 corrected image/json pairs in examples/.
   Then label a seed set of a few thousand frames. prepare writes files only.
    python -m tools.annotate.gemini_label prepare \\
        --images dataset/images --out dataset/labels_raw \\
        --workdir dataset/gemini --examples examples
    python -m tools.annotate.gemini_label submit --workdir dataset/gemini
    python -m tools.annotate.gemini_label collect --workdir dataset/gemini --wait

4. Correct dataset/labels_raw by hand (CVAT, Label Studio, or Roboflow) and
   save the corrected JSON as dataset/labels.

5. Export YOLO. Train with imgsz 1280 so projectiles survive the resize.
    python -m tools.annotate.to_yolo \\
        --images dataset/images --labels dataset/labels --out dataset/yolo
    yolo detect train model=yolo11s.pt data=dataset/yolo/data.yaml imgsz=1280

Class order in data.yaml: wall, bush, water, projectile.
JSON boxes are pixels [x1, y1, x2, y2], origin top-left.
Gemini boxes use [ymin, xmin, ymax, xmax] on a 0-1000 scale and are converted.
""".strip()


def main():
    print(WORKFLOW)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
