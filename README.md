# Vehicle Detection in Adverse Weather (YOLOv8 fine-tuning)

## Problem
A model trained on everyday photos (COCO) is not tuned for fog, rain, snow or sandstorms, where contrast drops and
vehicles are hazy, partly hidden or tiny. This project fine-tunes YOLOv8n on real adverse-weather traffic photos and
measures, on images the model never saw, how much that helps, and where it still fails.

## Data
[DAWN](https://data.mendeley.com/datasets/766ygrbt8y/3) (Vehicle Detection in Adverse Weather Nature),
1,027 real traffic photos with hand-drawn boxes. Licence: **CC BY-NC**, so the data is not included in this repo.

| Weather | Images | | Class | Boxes (all splits) |
|---|---|---|---|---|
| fog | 300 | | car | 6,454 |
| rain | 200 | | truck | 646 |
| snow | 204 | | person | 477 |
| sand | 323 | | bus | 161 |
| | | | motorcycle | 81 |
| | | | bicycle | 26 |

- Split 70 / 15 / 15 (train 719, val 154, **test 154**), done separately per weather type with a fixed seed (42).
- The test set is used only for the numbers below. Validation is used by YOLO for checkpoint selection.
- One `train` box in the whole dataset was dropped. Labels use COCO class ids, so the pretrained and the
  fine-tuned model are scored on exactly the same labels.

## Training settings
| Setting | Value |
|---|---|
| Start weights | `yolov8n.pt` (COCO pretrained) |
| Epochs | 20 (early stopping patience 10) |
| Image size / batch | 640 / 16 |
| Optimizer, augmentation | Ultralytics defaults |
| Hardware | CPU only (8 cores), about 70 minutes |
| Seed | 0 (single run) |

## Results (held-out test set, 154 images)
Both models were evaluated with the same script on the same images (`evaluate.py`).

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Pretrained YOLOv8n (COCO) | 0.668 | 0.385 | 0.444 | 0.301 |
| **Fine-tuned on DAWN** | **0.804** | **0.496** | **0.578** | **0.381** |

mAP50 rises by 0.134 and mAP50-95 by 0.080 after fine-tuning.

By weather (mAP50 / mAP50-95):

| Test subset | Images | Pretrained | Fine-tuned |
|---|---|---|---|
| fog | 45 | 0.470 / 0.327 | 0.728 / 0.474 |
| rain | 30 | 0.562 / 0.350 | 0.638 / 0.448 |
| snow | 30 | 0.564 / 0.414 | 0.682 / 0.456 |
| sand | 49 | 0.412 / 0.289 | 0.522 / 0.351 |

Full table: [results/metrics.md](results/metrics.md).

## Failure analysis
`failure_analysis.py` matches fine-tuned predictions (confidence >= 0.25, IoU >= 0.5) to the ground truth.
Details: [results/failure_summary.md](results/failure_summary.md). Annotated worst images are written to
`results/failures/` (green = ground truth, red = prediction; not committed because of the dataset licence).

**Recall by object size (the clearest pattern)**

| Size | Found / total | Recall |
|---|---|---|
| large (>96x96 px) | 465 / 513 | 0.91 |
| medium | 477 / 633 | 0.75 |
| small (<32x32 px) | 76 / 150 | 0.51 |

**Recall by weather**: fog 0.80, rain 0.81, sand 0.78, snow 0.76. At this threshold the weather types are close.
In mAP terms sand is the weakest (0.522 mAP50).

What goes wrong, based on the 12 worst test images:
1. **Small, distant vehicles.** Half of the tiny boxes are missed. In a sandstorm or heavy haze, far vehicles
   fade into the background and only a few pixels remain at 640 px input.
2. **Crowded scenes.** All 12 worst images contain 14 to 53 labelled objects. Queues of overlapping cars in snow
   or in a sandstorm produce both missed cars and duplicate or merged boxes.
3. **Questionable "false positives".** In the worst rain image many red boxes sit on distant, hazy vehicles that
   have no ground-truth box. The model may be right and the labels incomplete, so precision is probably
   understated for far-away objects.
4. **Darkness.** Only 8 test images are dark (mean brightness < 90). Their recall (0.76) is not different from
   the rest (0.79), but 8 images is too few to conclude anything.

## Limits
- **Small test set.** 154 images (30 to 49 per weather type). Differences of a few points between weather types are
  within noise, and rare classes (bicycle 26 boxes, motorcycle 81 in the whole dataset) are too few for reliable per-class scores.
- **One training run, one seed**, no confidence intervals. The improvement is large enough to be believable, but exact
  values will move between runs.
- **Small model, short training.** YOLOv8n for 20 epochs on CPU. Larger models, higher input resolution or longer
  training were not tried; the small-object recall suggests higher resolution is the first thing to test.
- **Label quality.** Boxes are hand-drawn and not exhaustive for far-away vehicles (see failure analysis).
- **Possible leakage.** Images come from web collections and some scenes may be near-duplicates across splits;
  this was not checked.
- **Narrow data.** Mostly cars, still images, no night scenes, no weather-station or camera-specific data. Results
  do not transfer automatically to another camera or region.
- **Licence.** DAWN is non-commercial.

## Reproduce
```
python -m pip install -r requirements.txt
```
Download DAWN from the link above, unzip it (and the four weather zips inside) into `data/raw/DAWN/`, then:
```
python prepare_data.py        # build YOLO dataset and train/val/test split
python train.py               # fine-tune (weights: runs/detect/runs/finetune/weights/best.pt)
python evaluate.py            # pretrained vs fine-tuned on the test set
python failure_analysis.py    # worst images, recall by weather and object size
```

## Demo scripts
- `detect_image.py <image>` runs the pretrained model on one image and saves `output.jpg`.
- `detect_webcam.py` runs live detection on a webcam.
- `app.py` is a small Streamlit page: `python -m streamlit run app.py`.

These use the pretrained COCO model. To try the fine-tuned model, replace `yolov8n.pt` with the path to `best.pt`.
