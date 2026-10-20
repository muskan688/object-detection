# Object Detection (beginner friendly)

Finds everyday objects (people, cars, dogs, phones, ... 80 types) in pictures
and webcam video using **YOLOv8**, a pretrained AI model. No training needed.

## Files
| File | What it does |
|------|--------------|
| `detect_image.py` | Detect objects in one image, save `output.jpg` |
| `detect_webcam.py` | Live detection from your webcam |
| `app.py` | Web page to upload a photo and see results |

## Setup (once)
```
python -m pip install -r requirements.txt
```

## Run
```
python detect_image.py images/your_photo.jpg
python detect_webcam.py
python -m streamlit run app.py
```

## How it works (3 steps)
1. `YOLO("yolov8n.pt")` loads the model (downloads on first run).
2. `model(image)` returns boxes: each has a class (what), confidence (how sure) and position.
3. `result.plot()` draws those boxes on the image.

## Tweaks
- Better accuracy, slower: change `yolov8n.pt` to `yolov8s.pt` or `yolov8m.pt`.
- Ignore weak guesses: `model(image, conf=0.5)`.
