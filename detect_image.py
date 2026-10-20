"""Detect objects in ONE image and save the result.

Usage:  python detect_image.py images/street.jpg
"""
import sys

import cv2
from ultralytics import YOLO

# 1. Load a pretrained model. It downloads automatically the first time (~6 MB).
#    "yolov8n" = the smallest and fastest version. It knows 80 everyday objects.
model = YOLO("yolov8n.pt")

# 2. Get the image path from the command line.
if len(sys.argv) < 2:
    sys.exit("Please give an image path, e.g.  python detect_image.py images/street.jpg")
image_path = sys.argv[1]

# 3. Run the model. It returns a list with one result per image.
result = model(image_path)[0]

# 4. Print what was found.
for box in result.boxes:
    name = model.names[int(box.cls)]      # e.g. "person"
    confidence = float(box.conf)          # 0.0 - 1.0, how sure the model is
    print(f"Found {name} ({confidence:.0%} sure)")

# 5. Draw boxes + labels on the image and save it.
annotated = result.plot()                 # image (numpy array) with boxes drawn
cv2.imwrite("output.jpg", annotated)
print("Saved result to output.jpg")
