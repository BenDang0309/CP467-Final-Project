'''
Modified version of the MIO-TCD view_bounding_boxes.py script.

Original authors: Zhiming Luo, Pierre-Marc Jodoin (University of Sherbrooke)

Instead of showing images in a window, this version picks a balanced, mixed
set of images containing buses, cars and motorcycles, draws their bounding
boxes, and saves the results to an output folder.

Usage:
    python view_bounding_boxes.py PATH CSV_FILE_NAME [OUTPUT_DIR] [NUM_IMAGES]

    PATH          : folder containing the training images
    CSV_FILE_NAME : csv file with the bounding boxes (typically gt_train.csv)
    OUTPUT_DIR    : where to save the images (default: ./output_bb)
    NUM_IMAGES    : how many images to save (default: 500)
'''

from seaborn import color_palette
import numpy as np
import cv2
import os
import csv
import sys
import random

classes = ['articulated_truck', 'bicycle', 'bus', 'car',
           'motorcycle', 'motorized_vehicle', 'non-motorized_vehicle',
           'pedestrian', 'pickup_truck',
           'single_unit_truck', 'work_van']

# Only these classes are selected and drawn
TARGET_CLASSES = ['bus', 'car', 'motorcycle']
RANDOM_SEED = 42


def make_color_map():
    '''
        Create a color map for each class
    '''
    names = sorted(set(classes))
    n = len(names)

    if n == 0:
        return {}
    cp = color_palette("Paired", n)

    cp[:] = [tuple(int(255*c) for c in rgb) for rgb in cp]

    return dict(zip(names, cp))


def plot_bboxes(img, bboxes, color_map):
    '''
        Plot the  bounding boxes of a given image with a pre-defined colormap
    '''
    show_img = np.copy(img)

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1

    for bbox in bboxes:
        label = bbox['class']
        pts = bbox['bbox']

        pt1 = (int(pts[0]), int(pts[1]))
        pt2 = (int(pts[2]), int(pts[3]))

        cv2.rectangle(show_img, pt1, pt2, color_map[label], 2)

        textSize, baseline = cv2.getTextSize(label, fontFace=fontFace,
                                             fontScale=scale,
                                             thickness=thickness)

        cv2.rectangle(show_img, pt1, (pt1[0]+textSize[0], pt1[1]+textSize[1]),
                      color_map[label], -1)

        cv2.putText(show_img, label, (pt1[0], pt1[1]+baseline*2),
                    fontFace, scale, (255, 255, 255), thickness)

    return show_img


def load_labels(csv_path):
    '''
        Convert the CSV format ground truth into a python dictionary
    '''
    train_label = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            img_name = row[0]

            bbox = {}
            bbox['class'] = row[1]
            if len(row) == 6:
                bbox['bbox'] = np.array(row[2:]).astype('int32')
            else:
                bbox['score'] = float(row[2])
                bbox['bbox'] = np.array(row[3:]).astype('int32')

            train_label.setdefault(img_name, []).append(bbox)
    return train_label


def select_mixed_images(train_label, num_images):
    '''
        Pick a balanced mix of images: roughly a third chosen because they
        contain a bus, a third a motorcycle, and a third a car. Each image is
        only used once.
    '''
    random.seed(RANDOM_SEED)

    pools = {}
    for cls in TARGET_CLASSES:
        names = [name for name, boxes in train_label.items()
                 if any(b['class'] == cls for b in boxes)]
        random.shuffle(names)
        pools[cls] = names

    per_class = num_images // len(TARGET_CLASSES)
    quotas = {cls: per_class for cls in TARGET_CLASSES}
    # Give any remainder to cars (by far the most common class)
    quotas['car'] += num_images - per_class * len(TARGET_CLASSES)

    selected = []
    used = set()
    # Fill the rarer classes first so cars don't use up their images
    for cls in sorted(TARGET_CLASSES, key=lambda c: len(pools[c])):
        count = 0
        for name in pools[cls]:
            if count >= quotas[cls]:
                break
            if name not in used:
                used.add(name)
                selected.append((name, cls))
                count += 1

    # Top up from any target class if a pool ran short
    if len(selected) < num_images:
        for cls in TARGET_CLASSES:
            for name in pools[cls]:
                if len(selected) >= num_images:
                    break
                if name not in used:
                    used.add(name)
                    selected.append((name, cls))

    random.shuffle(selected)
    return selected


def main():

    if len(sys.argv) < 3:
        print("\nUsage : \n\t python view_bounding_boxes.py PATH CSV_FILE_NAME [OUTPUT_DIR] [NUM_IMAGES]\n")
        print("\t PATH : path to the folder containing the training images ")
        print("\t CSV_FILE_NAME : csv file containing the bounding boxes (typically gt_train.csv)")
        print("\t OUTPUT_DIR : folder to save the images in (default ./output_bb)")
        print("\t NUM_IMAGES : number of images to save (default 500) \n")
        return

    image_dir = sys.argv[1]
    csv_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'output_bb'
    num_images = int(sys.argv[4]) if len(sys.argv) > 4 else 500

    os.makedirs(output_dir, exist_ok=True)

    train_label = load_labels(csv_path)
    color_map = make_color_map()
    selected = select_mixed_images(train_label, num_images)

    saved = 0
    counts = {cls: 0 for cls in TARGET_CLASSES}
    for img_name, cls in selected:
        path = os.path.join(image_dir, img_name + '.jpg')
        img = cv2.imread(path)
        if img is None:
            print("Could not read " + path)
            continue

        # Only draw bus, car and motorcycle boxes
        bboxes = [b for b in train_label[img_name]
                  if b['class'] in TARGET_CLASSES]
        show_img = plot_bboxes(img, bboxes, color_map)

        out_path = os.path.join(output_dir, img_name + '_bb.jpg')
        cv2.imwrite(out_path, show_img)
        saved += 1
        counts[cls] += 1
        print("[%d/%d] saved %s" % (saved, num_images, out_path))

    print("\nSaved %d images to %s" % (saved, os.path.abspath(output_dir)))
    print("Picked for: " + ", ".join("%s=%d" % (c, n) for c, n in counts.items()))


if __name__ == '__main__':
    main()
