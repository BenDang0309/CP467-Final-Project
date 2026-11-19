import os
import cv2
from ultralytics import YOLO
from sift import load_image, list_images, extract_boxes, remove_overlays

OBJECT_NAMES = {0: "Car", 1: "Bus", 2: "Motorcycle"}

BASE = os.path.dirname(os.path.abspath(__file__))

TRAIN_IMGS = os.path.join(BASE, "Images", "training images")
VAL_IMGS   = os.path.join(BASE, "Images", "validation images")
TEST_IMG   = os.path.join(BASE, "Images", "testing images")
OUTPUT     = os.path.join(BASE, "Images", "YOLOv8s output")
YOLO_DATA = os.path.join(BASE, "YOLO data")
RUNS =
RUN_NAME =

EPOCHS =
BATCH =

def train():
    model = YOLO("yolov8s.pt")
    model.train(
        data=os.path.join(YOLO_DATA, "data.yaml"), epochs=EPOCHS, batch=BATCH, patience=15, workers=2,
        project=RUNS, name=RUN_NAME, exist_ok=True
    )
