"""Compare the pretrained YOLOv8n with the fine-tuned model on the held-out TEST set.

Run:  python evaluate.py
Writes results/metrics.csv and results/metrics.md
"""
import csv
from pathlib import Path

from ultralytics import YOLO

DATA = Path("data/dawn")
MODELS = {"pretrained (COCO)": "yolov8n.pt", "fine-tuned (DAWN)": "runs/detect/runs/finetune/weights/best.pt"}
OUT = Path("results")
OUT.mkdir(exist_ok=True)

manifest = list(csv.DictReader(open(DATA / "manifest.csv")))
test_rows = [r for r in manifest if r["split"] == "test"]


def make_subset_yaml(name, rows):
    """A tiny dataset file that points at only some of the test images."""
    txt = OUT / f"test_{name}.txt"
    txt.write_text("\n".join((DATA / "images" / "test" / r["image"]).resolve().as_posix() for r in rows))
    yaml = OUT / f"test_{name}.yaml"
    base = (DATA / "dawn.yaml").read_text().split("train:")[0]   # keeps 'path:'
    names = (DATA / "dawn.yaml").read_text().split("names:")[1]
    yaml.write_text(f"{base}train: {txt.resolve().as_posix()}\nval: {txt.resolve().as_posix()}\n"
                    f"test: {txt.resolve().as_posix()}\nnames:{names}")
    return yaml


subsets = {"all": test_rows}
for w in sorted({r["weather"] for r in test_rows}):
    subsets[w] = [r for r in test_rows if r["weather"] == w]

rows = []
for model_name, weights in MODELS.items():
    model = YOLO(weights)
    for subset, subset_rows in subsets.items():
        r = model.val(data=make_subset_yaml(subset, subset_rows), split="test", imgsz=640,
                      device="cpu", verbose=False, plots=False, conf=0.001)
        rows.append({"model": model_name, "subset": subset, "images": len(subset_rows),
                     "precision": r.box.mp, "recall": r.box.mr,
                     "mAP50": r.box.map50, "mAP50-95": r.box.map})
        print(rows[-1])

with open(OUT / "metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

md = ["| Model | Test subset | Images | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['model']} | {r['subset']} | {r['images']} | {r['precision']:.3f} | "
              f"{r['recall']:.3f} | {r['mAP50']:.3f} | {r['mAP50-95']:.3f} |")
(OUT / "metrics.md").write_text("\n".join(md))
print("\n".join(md))
