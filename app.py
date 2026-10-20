"""A simple web page: upload a photo, see the detected objects.

Usage:  python -m streamlit run app.py
"""
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


@st.cache_resource                        # load the model only once
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

st.title("Object Detection")
min_conf = st.slider("Minimum confidence", 0.1, 0.9, 0.4)
upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if upload:
    image = Image.open(upload).convert("RGB")
    result = model(np.array(image), conf=min_conf)[0]

    # result.plot() returns BGR colours; [..., ::-1] flips them to RGB for display
    st.image(result.plot()[..., ::-1], caption="Detections")

    names = [model.names[int(c)] for c in result.boxes.cls]
    st.write(f"Found {len(names)} object(s):", {n: names.count(n) for n in set(names)})
