'''
Makes a validation set of MIO-TCD images with bus, car and motorcycle
bounding boxes, drawn in the same colours as view_bounding_boxes.py.

The MIO-TCD localization dataset has no separate validation split, and the
test set has no real ground truth (your_result_test.csv is random labels).
So the validation images are held out from the labelled training set, and
any image already used in the training output folder is skipped, so the two
sets never overlap.

Usage:
    python make_validation_images.py PATH CSV_FILE_NAME [OUTPUT_DIR] [NUM_IMAGES] [TRAIN_OUTPUT_DIR]

    PATH             : folder containing the training images
    CSV_FILE_NAME    : gt_train.csv
    OUTPUT_DIR       : where to save the validation images (default: ./output_val)
    NUM_IMAGES       : how many images to save (default: 71)
    TRAIN_OUTPUT_DIR : folder of already-saved training images to exclude
                       (default: ./training images)
'''

import os
import sys
import random

import cv2

# Reuse the exact same colour map and drawing code as the training images
from view_bounding_boxes import (TARGET_CLASSES, load_labels,
                                 make_color_map, plot_bboxes)

VAL_SEED = 2027


def select_validation_images(train_label, num_images, exclude):
    '''
        Pick a balanced mix (about a third each for bus, motorcycle and car),
        skipping any image in `exclude`.
    '''
    rng = random.Random(VAL_SEED)

    pools = {}
    for cls in TARGET_CLASSES:
        names = [name for name, boxes in train_label.items()
                 if name not in exclude
                 and any(b['class'] == cls for b in boxes)]
        names.sort()
        rng.shuffle(names)
        pools[cls] = names

    per_class = num_images // len(TARGET_CLASSES)
    quotas = {cls: per_class for cls in TARGET_CLASSES}
    quotas['car'] += num_images - per_class * len(TARGET_CLASSES)

    selected = []
    used = set()
    for cls in sorted(TARGET_CLASSES, key=lambda c: len(pools[c])):
        count = 0
        for name in pools[cls]:
            if count >= quotas[cls]:
                break
            if name not in used:
                used.add(name)
                selected.append((name, cls))
                count += 1

    rng.shuffle(selected)
    return selected


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return

    image_dir = sys.argv[1]
    csv_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'output_val'
    num_images = int(sys.argv[4]) if len(sys.argv) > 4 else 71
    train_out = sys.argv[5] if len(sys.argv) > 5 else 'training images'

    # Images already used for training (saved as <id>_bb.jpg)
    exclude = set()
    if os.path.isdir(train_out):
        for _, _, files in os.walk(train_out):
            for f in files:
                if f.endswith('_bb.jpg'):
                    exclude.add(f[:-len('_bb.jpg')])
    print("Excluding %d images already in %s" % (len(exclude), train_out))

    os.makedirs(output_dir, exist_ok=True)

    train_label = load_labels(csv_path)
    color_map = make_color_map()
    selected = select_validation_images(train_label, num_images, exclude)

    saved = 0
    counts = {cls: 0 for cls in TARGET_CLASSES}
    for img_name, cls in selected:
        path = os.path.join(image_dir, img_name + '.jpg')
        img = cv2.imread(path)
        if img is None:
            print("Could not read " + path)
            continue

        bboxes = [b for b in train_label[img_name]
                  if b['class'] in TARGET_CLASSES]
        out_img = plot_bboxes(img, bboxes, color_map)

        out_path = os.path.join(output_dir, img_name + '_bb.jpg')
        cv2.imwrite(out_path, out_img)
        saved += 1
        counts[cls] += 1
        print("[%d/%d] saved %s" % (saved, num_images, out_path))

    print("\nSaved %d validation images to %s" % (saved, os.path.abspath(output_dir)))
    print("Picked for: " + ", ".join("%s=%d" % (c, n) for c, n in counts.items()))


if __name__ == '__main__':
    main()
