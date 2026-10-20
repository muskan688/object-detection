"""Convert the raw DAWN download into a YOLO dataset with train / val / test splits.

DAWN = 1,027 real traffic photos taken in fog, rain, snow and sandstorms.
Run once:  python prepare_data.py
"""
import csv
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

RAW = Path("data/raw/DAWN")
OUT = Path("data/dawn")
SEED = 42
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Class ids follow the COCO numbering used by the pretrained YOLOv8 model.
# That way the pretrained and the fine-tuned model are scored on identical labels.
COCO_IDS = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
# ("train" has a single box in the whole dataset, so it is dropped.)

COCO_NAMES = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant "
    "stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe "
    "backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite "
    "baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup "
    "fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut "
    "cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase scissors "
    "teddy_bear hair_drier toothbrush"
).split()


def read_boxes(xml_path):
    """Return YOLO label lines: 'class x_center y_center width height' (all 0-1)."""
    root = ET.parse(xml_path).getroot()
    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in COCO_IDS:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        lines.append(f"{COCO_IDS[name]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def main():
    random.seed(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)

    rows = []  # (image path, weather, label lines)
    for weather_dir in sorted(RAW.iterdir()):
        if not weather_dir.is_dir():
            continue
        weather = weather_dir.name.lower()
        for img in sorted(weather_dir.rglob("*.jpg")):
            xml = next(weather_dir.rglob(f"{img.stem}.xml"))
            rows.append((img, weather, read_boxes(xml)))

    # Split each weather type separately so every split has fog, rain, snow and sand.
    manifest = []
    for weather in sorted({r[1] for r in rows}):
        group = [r for r in rows if r[1] == weather]
        random.shuffle(group)
        n_train = round(len(group) * SPLITS["train"])
        n_val = round(len(group) * SPLITS["val"])
        parts = {"train": group[:n_train], "val": group[n_train:n_train + n_val],
                 "test": group[n_train + n_val:]}
        for split, items in parts.items():
            (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
            (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
            for img, _, lines in items:
                name = f"{weather}_{img.name}"
                shutil.copy(img, OUT / "images" / split / name)
                (OUT / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines))
                manifest.append((split, name, weather, len(lines)))

    with open(OUT / "manifest.csv", "w", newline="") as f:
        csv.writer(f).writerows([("split", "image", "weather", "n_boxes"), *manifest])

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO_NAMES))
    (OUT / "dawn.yaml").write_text(
        f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\nnames:\n{names}\n")

    for split in SPLITS:
        n = sum(1 for m in manifest if m[0] == split)
        print(f"{split}: {n} images")


if __name__ == "__main__":
    main()
