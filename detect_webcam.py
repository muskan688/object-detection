"""Live object detection from your webcam. Press 'q' to quit.

Usage:  python detect_webcam.py
"""
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
camera = cv2.VideoCapture(0)              # 0 = your default webcam

while True:
    ok, frame = camera.read()             # grab one frame
    if not ok:
        print("Could not read from the webcam.")
        break

    result = model(frame, verbose=False)[0]   # detect objects in this frame
    cv2.imshow("Object Detection (press q to quit)", result.plot())

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
