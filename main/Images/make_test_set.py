'''
Picks NUM unused, labelled MIO-TCD train images (balanced across bus, car and
motorcycle), copies the clean images to OUT_IMG_DIR, and writes YOLO labels
with the same names to OUT_LBL_DIR. Images already in any EXCLUDE_DIR are skipped.

Usage:
    python make_test_set.py TRAIN_DIR GT_CSV OUT_IMG_DIR OUT_LBL_DIR NUM EXCLUDE_DIR [EXCLUDE_DIR ...]
'''
import os, random, shutil, sys
import cv2
from make_yolo_labels import load_boxes, to_yolo, OBJECT_NAMES

SEED = 467


def main():
    if len(sys.argv) < 7:
        print(__doc__); return
    train_dir, csv_path, out_img, out_lbl = sys.argv[1:5]
    num = int(sys.argv[5])
    exclude = set()
    for d in sys.argv[6:]:
        for _, _, files in os.walk(d):
            for f in files:
                stem = os.path.splitext(f)[0]
                exclude.add(stem[:-3] if stem.endswith('_bb') else stem)
    print("Excluding %d images already used" % len(exclude))

    boxes = load_boxes(csv_path)   # only car/bus/motorcycle
    rng = random.Random(SEED)
    pools = {}
    for cid in OBJECT_NAMES:
        ids = sorted(i for i, b in boxes.items() if i not in exclude and any(x[0] == cid for x in b))
        rng.shuffle(ids)
        pools[cid] = ids
    per = num // 3
    quotas = {0: per + num - 3 * per, 1: per, 2: per}
    chosen, used = [], set()
    for cid in sorted(pools, key=lambda c: len(pools[c])):
        n = 0
        for i in pools[cid]:
            if n >= quotas[cid]: break
            if i not in used:
                used.add(i); chosen.append(i); n += 1

    os.makedirs(out_img, exist_ok=True); os.makedirs(out_lbl, exist_ok=True)
    counts = {n: 0 for n in OBJECT_NAMES.values()}
    for i in sorted(chosen):
        src = os.path.join(train_dir, i + '.jpg')
        img = cv2.imread(src)
        if img is None:
            print("Could not read " + src); continue
        h, w = img.shape[:2]
        lines = []
        for b in boxes[i]:
            line = to_yolo(b, w, h)
            if line:
                lines.append(line); counts[OBJECT_NAMES[b[0]]] += 1
        shutil.copy2(src, os.path.join(out_img, i + '.jpg'))
        with open(os.path.join(out_lbl, i + '.txt'), 'w') as f:
            f.write("\n".join(lines) + "\n")
    print("Saved %d images and labels, boxes %s" % (len(chosen), counts))


if __name__ == '__main__':
    main()
