import os
import cv2
import numpy as np

RATIO_TEST = 0.70     # Lowe's ratio for keeping a match
MATCH_THRESHOLD = 100   # object is "present" if count >= this value


def load_image(path):
    """Load an image and stop with a clear message if it is missing."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def detect_and_describe(image):
    """
    Detect keypoints and compute descriptors.
    Return (keypoints, descriptors).
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


def match_descriptors(desc_object, desc_scene):
    """
    Match object descriptors to scene descriptors and keep good matches
    using the ratio test (RATIO_TEST). Return a list of cv2.DMatch.
    """

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(desc_object, desc_scene, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < RATIO_TEST * n.distance:
            good_matches.append(m)

    return good_matches


def is_present(good_matches):
    """
    TODO: Decide whether the object is present in the scene.
    Return True or False. Record the counts you see in Results.xlsx (Match Counts).
    """
    return len(good_matches) >= MATCH_THRESHOLD


def locate_object(kp_scene, good_matches):
    """
     Find a bounding box for the object in the scene from the
    scene positions of its good matches.
    1. Collect the scene points: kp_scene[m.trainIdx].pt for each good match.
    2. Remove stray points far from the object.
    3. Return the bounding rectangle (x, y, w, h) of the remaining points, or None
       if too few points remain.
    """
    points = np.array([kp_scene[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    k = min(15, len(points) - 1) # k neighbors to measure distance against, change if needed
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    knn_dist = np.sort(dists, axis=1)[:, k]
    

    keep = knn_dist <= 3.0 * np.median(knn_dist) # cutoff, change if needed
    filtered_points = points[keep]

    if len(filtered_points) < 3:
        return None
    return cv2.boundingRect(filtered_points)


def draw_box(scene_image, box, label=None):
    """Draw an axis-aligned bounding box and its label on the scene."""
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(scene_image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)
    if label:
        cv2.putText(scene_image, label, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return scene_image

def main():
  

