import os
import sys
import csv
import glob
import pickle
import cv2
import numpy as np

OBJECT_NAMES = {0: "Car", 1: "Bus", 2: "Motorcycle"}

RATIO_TEST = 0.50     # Lowe's ratio for keeping a match
THRESHOLDS = {0: 10, 1: 2, 2: 1}  
MAX_TEMPLATES_PER_CLASS = 40 # keep the N templates with the most keypoints

BASE = os.path.dirname(os.path.abspath(__file__))

TRAIN_IMGS = os.path.join(BASE, "Images", "train no bounding box")
TRAIN_TXTS = os.path.join(BASE, "Images", "training labels")
VAL_IMGS = os.path.join(BASE, "Images", "val no bounding box")
VAL_TXTS = os.path.join(BASE, "Images", "validation labels")
TEST_IMG = os.path.join(BASE, "Images", "testing images")
TEST_TXTS = os.path.join(BASE, "Images", "testing labels")
OUTPUT = os.path.join(BASE, "Images", "sift output")
CACHE = os.path.join(BASE, "templates.pkl")


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def list_images(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                   if f.lower().endswith(".jpg"))
    if not files:
        raise FileNotFoundError(f"No .jpg images in: {folder}")
    return files


def read_boxes(img_path, label_dir, W, H):
    """Read the YOLO txt for an image -> [(class_id, (x, y, w, h))] in pixels."""
    name = os.path.splitext(os.path.basename(img_path))[0]
    txt = os.path.join(label_dir, name + ".txt")
    if not os.path.exists(txt):
        return [] # no label file = no vehicles
    boxes = []
    with open(txt) as f:
        for line in f.read().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            cls = int(float(p[0]))
            if cls not in OBJECT_NAMES:
                continue
            xc, yc, bw, bh = (float(v) for v in p[1:])
            x1 = max(int(round((xc - bw / 2) * W)), 0)
            y1 = max(int(round((yc - bh / 2) * H)), 0)
            x2 = min(int(round((xc + bw / 2) * W)), W)
            y2 = min(int(round((yc + bh / 2) * H)), H)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            boxes.append((cls, (x1, y1, x2 - x1, y2 - y1)))
    return boxes


def overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def mask_timestamp(mask, img):
    """Zero out the burned-in timestamp in `mask` (SIFT ignores pixels where mask == 0)."""
    h, w = mask.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = (gray < 20).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if (y > 0.7 * h and 60 < bw < 0.4 * w and 10 < bh < 0.1 * h
                and area > 0.4 * bw * bh):
            mask[max(y - 3, 0):y + bh + 3, max(x - 3, 0):x + bw + 3] = 0
    return mask


def timestamp_corner(mask):
    """Always exclude the bottom-right corner where the date sits."""
    h, w = mask.shape[:2]
    mask[int(0.86 * h):, int(0.70 * w):] = 0
    return mask


def box_mask(img, box, exclude=()):
    """Mask for the interior of an object's box. `exclude` carves out overlapping boxes."""
    x, y, w, h = box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255
    for ex, ey, ew, eh in exclude:
        mask[ey:ey + eh, ex:ex + ew] = 0
    return timestamp_corner(mask_timestamp(mask, img))


def scene_mask(img):
    """Scene mask: everything allowed except the timestamp."""
    mask = mask_timestamp(np.full(img.shape[:2], 255, dtype=np.uint8), img)
    return timestamp_corner(mask)


def detect_and_describe(image, mask=None):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, mask)


def match_descriptors(desc_object, desc_scene):
    """Brute-force match + Lowe's ratio test. Returns a list of cv2.DMatch."""
    if desc_object is None or desc_scene is None or len(desc_scene) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(desc_object, desc_scene, k=2)
    return [p[0] for p in matches
            if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]


def is_present(good_matches, threshold):
    return len(good_matches) >= threshold


def locate_object(kp_scene, good_matches):
    """Bounding rectangle of the good matches after removing stray points (drawing only)."""
    if len(good_matches) < 4:
        return None
    points = np.array([kp_scene[m.trainIdx].pt for m in good_matches], dtype=np.float32)
    k = min(15, len(points) - 1)
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    knn_dist = np.sort(dists, axis=1)[:, k]
    cutoff = 3.0 * max(np.median(knn_dist), 1.0)
    filtered_points = points[knn_dist <= cutoff]
    if len(filtered_points) < 3:
        return None
    return cv2.boundingRect(filtered_points)


def draw_box(scene_image, box, label=None):
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(scene_image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        lx = max(0, min(x, scene_image.shape[1] - tw - 5))
        cv2.putText(scene_image, label, (lx, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return scene_image


def build_templates():
    """Return {class_id: [descriptor arrays]}. Cached on disk because this is slow."""
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    if not os.path.isdir(TRAIN_TXTS):
        raise FileNotFoundError(f"Label folder not found: {TRAIN_TXTS}")

    pool = {cls: [] for cls in OBJECT_NAMES}   # cls -> [(n_keypoints, descriptors)]
    for path in list_images(TRAIN_IMGS):
        img = load_image(path)
        H, W = img.shape[:2]
        boxes = read_boxes(path, TRAIN_TXTS, W, H)
        for idx, (cls, box) in enumerate(boxes):
            others = [b for j, (_, b) in enumerate(boxes) if j != idx and overlaps(box, b)]
            kp, desc = detect_and_describe(img, box_mask(img, box, exclude=others))
            if desc is not None and len(kp) >= 2:
                pool[cls].append((len(kp), desc))

    templates = {}
    for cls, items in pool.items():
        items.sort(key=lambda t: t[0], reverse=True)
        templates[cls] = [d for _, d in items[:MAX_TEMPLATES_PER_CLASS]]
        print(f"{OBJECT_NAMES[cls]}: {len(items)} usable templates, keeping {len(templates[cls])}")

    if all(len(v) == 0 for v in templates.values()):
        raise RuntimeError(f"No templates built. Check '{TRAIN_TXTS}' and that txt names "
                           f"match the image names.")
    with open(CACHE, "wb") as f:
        pickle.dump(templates, f)
    return templates


def best_matches(template_descs, desc_scene):
    """Good matches from whichever template of this class matches the scene best."""
    best = []
    for desc in template_descs:
        good = match_descriptors(desc, desc_scene)
        if len(good) > len(best):
            best = good
    return best


def collect_val_results():
    """Per validation frame and class: (best good-match count, None, [gt boxes of that class])."""
    if not os.path.isdir(VAL_TXTS):
        raise FileNotFoundError(f"Label folder not found: {VAL_TXTS}")
    templates = build_templates()
    results = []
    paths = list_images(VAL_IMGS)
    for i, path in enumerate(paths, 1):
        img = load_image(path)
        H, W = img.shape[:2]
        gt = read_boxes(path, VAL_TXTS, W, H)
        kp_s, desc_s = detect_and_describe(img, scene_mask(img))
        frame = {}
        for cls in OBJECT_NAMES:
            good = best_matches(templates[cls], desc_s)
            frame[cls] = (len(good), None, [b for c, b in gt if c == cls])
        results.append(frame)
        if i % 20 == 0:
            print(f"  processed {i}/{len(paths)} validation frames")
    return results


def evaluate(results, cls, threshold):
    """Per-frame presence counts for one class: (tp, fp, fn, tn)."""
    tp = fp = fn = tn = 0
    for frame in results:
        count, _, gts = frame[cls]
        actual = len(gts) > 0
        predicted = count >= threshold
        if predicted and actual:       tp += 1
        elif predicted and not actual: fp += 1
        elif not predicted and actual: fn += 1
        else:                          tn += 1
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def balanced_acc(tp, fp, fn, tn):
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return (tpr + tnr) / 2


def tune(max_t=300):
    """Pick each class's threshold on all validation frames by balanced accuracy."""
    results = collect_val_results()
    chosen = {}

    for cls, name in OBJECT_NAMES.items():
        pos = sorted(f[cls][0] for f in results if f[cls][2])
        neg = sorted(f[cls][0] for f in results if not f[cls][2])
        print(f"{name}\n  frames WITH it, counts:   {pos}\n  frames WITHOUT it, counts:{neg}")

    print(f"\nTuning on all {len(results)} validation frames "
          f"(RATIO_TEST={RATIO_TEST}, templates={MAX_TEMPLATES_PER_CLASS})\n")
    for cls, name in OBJECT_NAMES.items():
        top = max(f[cls][0] for f in results) + 1
        best_t, best_score = 1, -1.0
        for t in range(1, min(top, max_t) + 1):
            tp, fp, fn, tn = evaluate(results, cls, t)
            score = balanced_acc(tp, fp, fn, tn)
            if score > best_score:
                best_t, best_score = t, score
        chosen[cls] = best_t

        tp, fp, fn, tn = evaluate(results, cls, best_t)
        p, r, f1 = prf(tp, fp, fn)
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)
        print(f"{name:11s} threshold={best_t:4d}  TP={tp} FP={fp} FN={fn} TN={tn}  "
              f"precision={p:.2f} recall={r:.2f} F1={f1:.2f} "
              f"accuracy={acc:.2f} balanced={best_score:.2f}")

    print(f"\nPaste this into the config at the top of the script:\nTHRESHOLDS = {chosen}")


def main():
    missing = [OBJECT_NAMES[c] for c in OBJECT_NAMES if c not in THRESHOLDS]
    if missing:
        raise ValueError(f"THRESHOLDS has no value for: {', '.join(missing)}.")
    if not os.path.isdir(TEST_TXTS):
        raise FileNotFoundError(f"Label folder not found: {TEST_TXTS}")

    os.makedirs(OUTPUT, exist_ok=True)
    templates = build_templates()
    print(f"RATIO_TEST={RATIO_TEST}  thresholds: "
          f"{ {OBJECT_NAMES[c]: t for c, t in THRESHOLDS.items()} }")

    conf = {cls: [0, 0, 0, 0] for cls in OBJECT_NAMES}    # tp, fp, fn, tn
    rows = []
    for path in list_images(TEST_IMG):
        fname = os.path.basename(path)
        scene = load_image(path)
        H, W = scene.shape[:2]
        actual_cls = {c for c, _ in read_boxes(path, TEST_TXTS, W, H)}
        kp_s, desc_s = detect_and_describe(scene, scene_mask(scene))
        output = scene.copy()

        for cls, name in OBJECT_NAMES.items():
            good = best_matches(templates[cls], desc_s)
            present = is_present(good, THRESHOLDS[cls])
            actual = cls in actual_cls
            idx = (0 if actual else 1) if present else (2 if actual else 3)
            conf[cls][idx] += 1
            rows.append((fname, name, len(good), THRESHOLDS[cls], present, actual))

            if present:
                box = locate_object(kp_s, good)
                if box is not None:
                    draw_box(output, box, f"{name} ({len(good)})")
        cv2.imwrite(os.path.join(OUTPUT, fname), output)

    with open(os.path.join(OUTPUT, "match_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "good_matches", "threshold", "present", "actual"])
        w.writerows(rows)

    print("\nTEST RESULTS (presence per frame)")
    for cls, name in OBJECT_NAMES.items():
        tp, fp, fn, tn = conf[cls]
        p, r, f1 = prf(tp, fp, fn)
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)
        print(f"{name:11s} TP={tp} FP={fp} FN={fn} TN={tn}  precision={p:.2f} "
              f"recall={r:.2f} F1={f1:.2f} accuracy={acc:.2f} "
              f"balanced={balanced_acc(tp, fp, fn, tn):.2f}")
    print(f"Done. Saved to '{OUTPUT}'.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "tune":
        tune()
    else:
        main()