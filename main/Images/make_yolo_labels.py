'''
Writes a YOLO label .txt for every <id>_bb.jpg image in a folder, using the
MIO-TCD ground truth (gt_train.csv). Only car, bus and motorcycle are kept.

Each label file has the same name as its image (00000713_bb.jpg ->
00000713_bb.txt), one line per object:
    <class_id> <x_center> <y_center> <width> <height>   (normalised 0-1)

Usage:
    python make_yolo_labels.py CSV_FILE_NAME IMAGE_DIR LABEL_DIR [IMAGE_DIR LABEL_DIR ...]
'''

import csv
import os
import sys

import cv2

OBJECT_NAMES = {0: "Car", 1: "Bus", 2: "Motorcycle"}
CLASS_IDS = {"car": 0, "bus": 1, "motorcycle": 2}


def load_boxes(csv_path):
    boxes = {}
    with open(csv_path, 'r') as f:
        for row in csv.reader(f):
            if row[1] not in CLASS_IDS:
                continue
            coords = row[2:] if len(row) == 6 else row[3:]
            x1, y1, x2, y2 = (int(v) for v in coords)
            boxes.setdefault(row[0], []).append((CLASS_IDS[row[1]], x1, y1, x2, y2))
    return boxes


def to_yolo(box, w, h):
    cls, x1, y1, x2, y2 = box
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    return "%d %.6f %.6f %.6f %.6f" % (cls, (x1 + bw / 2) / w, (y1 + bh / 2) / h,
                                      bw / w, bh / h)


def label_folder(image_dir, label_dir, boxes):
    os.makedirs(label_dir, exist_ok=True)
    counts = {n: 0 for n in OBJECT_NAMES.values()}
    written = 0
    for f in sorted(os.listdir(image_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() != '.jpg':
            continue
        img_id = stem[:-3] if stem.endswith('_bb') else stem
        img = cv2.imread(os.path.join(image_dir, f))
        if img is None:
            print("Could not read " + f)
            continue
        h, w = img.shape[:2]
        lines = []
        for box in boxes.get(img_id, []):
            line = to_yolo(box, w, h)
            if line:
                lines.append(line)
                counts[OBJECT_NAMES[box[0]]] += 1
        with open(os.path.join(label_dir, stem + '.txt'), 'w') as out:
            out.write("\n".join(lines) + ("\n" if lines else ""))
        written += 1
    print("%s: %d label files, boxes %s" % (label_dir, written, counts))


def main():
    if len(sys.argv) < 4 or len(sys.argv) % 2 != 0:
        print(__doc__)
        return
    boxes = load_boxes(sys.argv[1])
    for i in range(2, len(sys.argv), 2):
        label_folder(sys.argv[i], sys.argv[i + 1], boxes)


if __name__ == '__main__':
    main()
