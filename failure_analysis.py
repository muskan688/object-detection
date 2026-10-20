"""Find where the fine-tuned model fails on the TEST set, and why.

Run:  python failure_analysis.py
Writes results/failures.csv, results/failure_summary.md and results/failures/*.jpg
(green = ground truth, red = model prediction).
"""
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DATA = Path("data/dawn")
OUT = Path("results")
(OUT / "failures").mkdir(parents=True, exist_ok=True)
CONF = 0.25
IOU_MATCH = 0.5

model = YOLO("runs/detect/runs/finetune/weights/best.pt")
test_rows = [r for r in csv.DictReader(open(DATA / "manifest.csv")) if r["split"] == "test"]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def size_bucket(area_px):
    return "small (<32x32)" if area_px < 32 ** 2 else "medium" if area_px < 96 ** 2 else "large (>96x96)"


def brightness(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


image_stats, object_stats = [], []
for r in test_rows:
    path = DATA / "images" / "test" / r["image"]
    img = cv2.imread(str(path))
    h, w = img.shape[:2]

    gts = []   # ground-truth boxes in pixels: (x1, y1, x2, y2, class)
    for line in (DATA / "labels" / "test" / f"{path.stem}.txt").read_text().splitlines():
        c, x, y, bw, bh = line.split()
        x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
        gts.append((x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2, int(c)))

    res = model(str(path), conf=CONF, verbose=False)[0]
    preds = [(*b.xyxy[0].tolist(), int(b.cls), float(b.conf)) for b in res.boxes]

    # greedy matching: each prediction may claim one ground-truth box of the same class
    used, tp = set(), 0
    for p in sorted(preds, key=lambda p: -p[5]):
        best, best_i = 0, None
        for i, g in enumerate(gts):
            if i not in used and g[4] == p[4] and iou(p, g) > best:
                best, best_i = iou(p, g), i
        if best >= IOU_MATCH:
            used.add(best_i)
            tp += 1
    fp, fn = len(preds) - tp, len(gts) - tp

    for i, g in enumerate(gts):
        object_stats.append({"weather": r["weather"], "found": i in used,
                             "size": size_bucket((g[2] - g[0]) * (g[3] - g[1]))})
    image_stats.append({"image": r["image"], "weather": r["weather"], "gt": len(gts), "tp": tp,
                        "fp": fp, "fn": fn, "errors": fp + fn, "brightness": round(brightness(img), 1)})

    image_stats[-1]["_img"] = (img, gts, preds)

image_stats.sort(key=lambda s: -s["errors"])
for s in image_stats[:12]:
    img, gts, preds = s.pop("_img")
    for g in gts:
        cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 200, 0), 2)
    for p in preds:
        cv2.rectangle(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 255), 2)
    cv2.rectangle(img, (0, 0), (430, 44), (0, 0, 0), -1)
    cv2.putText(img, f"GT {s['gt']}  TP {s['tp']}  FP {s['fp']}  FN {s['fn']}", (10, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(str(OUT / "failures" / s["image"]), img)
for s in image_stats:
    s.pop("_img", None)

with open(OUT / "failures.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=image_stats[0].keys())
    wr.writeheader()
    wr.writerows(image_stats)


def recall_table(key):
    groups = {}
    for o in object_stats:
        g = groups.setdefault(o[key], [0, 0])
        g[0] += o["found"]
        g[1] += 1
    return {k: (v[0], v[1], v[0] / v[1]) for k, v in sorted(groups.items())}


lines = [f"Threshold: confidence >= {CONF}, IoU >= {IOU_MATCH}", ""]
for key, title in (("weather", "Recall by weather"), ("size", "Recall by object size")):
    lines += [f"**{title}**", "", "| Group | Found | Total | Recall |", "|---|---|---|---|"]
    lines += [f"| {k} | {a} | {b} | {r:.2f} |" for k, (a, b, r) in recall_table(key).items()]
    lines.append("")

dark = [s for s in image_stats if s["brightness"] < 90]
bright = [s for s in image_stats if s["brightness"] >= 90]
for name, grp in (("dark images (mean brightness < 90)", dark), ("brighter images", bright)):
    gt, tp = sum(s["gt"] for s in grp), sum(s["tp"] for s in grp)
    lines.append(f"- {name}: {len(grp)} images, recall {tp / gt:.2f}" if gt else f"- {name}: none")
lines += ["", "**Worst 12 images** (see results/failures/):", "",
          "| Image | Weather | GT | TP | FP | FN | Brightness |", "|---|---|---|---|---|---|---|"]
lines += [f"| {s['image']} | {s['weather']} | {s['gt']} | {s['tp']} | {s['fp']} | {s['fn']} | "
          f"{s['brightness']} |" for s in image_stats[:12]]
(OUT / "failure_summary.md").write_text("\n".join(lines))
print("\n".join(lines))
