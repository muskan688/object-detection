"""Fine-tune the pretrained YOLOv8n on the DAWN weather dataset.

Run:  python train.py
Best weights end up in runs/detect/runs/finetune/weights/best.pt
"""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")            # start from the COCO-pretrained model

model.train(
    data="data/dawn/dawn.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    patience=10,                      # stop early if val mAP stops improving
    device="cpu",                     # no GPU on this machine
    workers=4,
    seed=0,
    project="runs",
    name="finetune",
    exist_ok=True,
    plots=True,
)
