import numpy
import cv2

def load_image(path):
    """Load an image and stop with a clear message if it is missing."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img
