import os
import sys
import csv
import glob
import pickle
import cv2
import numpy as np

OBJECT_NAMES = {0: "Car", 1: "Bus", 2: "Motorcycle"}
OBJECT_COLORS = {0: (64, 156, 50), 1: (175, 221, 144), 2: (248, 155, 154)} # BGR colors of the respective bounding boxes, in the training data
COLOR_TOLERANCE = 30 # Tolerance because of JPG compression

RATIO_TEST = 0.80     # Lowe's ratio for keeping a match

THRESHOLDS = {0: 23, 1: 97, 2: 5}
MAX_TEMPLATES_PER_CLASS = 40   # keep the N templates with the most keypoints (speed)

BASE = os.path.dirname(os.path.abspath(__file__))

TRAIN_IMGS = os.path.join(BASE, "Images", "training images")
VAL_IMGS = os.path.join(BASE, "Images", "validation images")
TEST_IMG = os.path.join(BASE, "Images", "testing images")
OUTPUT = os.path.join(BASE, "Images", "sift output")
CACHE = os.path.join(BASE, "templates.pkl")

def load_image(path):
    """Load an image and stop with a clear message if it is missing."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def list_images(folder):
    return sorted(f for f in glob.glob(os.path.join(folder, "*"))
                  if f.lower().endswith(".jpg"))

def mask_timestamp(mask, img):
    """
    Zero out the burned-in timestamp in `mask` (SIFT ignores pixels where mask == 0).
    Keeping the timestamp could mess with training. The timestamp is a near-black label 
    in the lower part of the image, so we look for a small dark rectangle there.
    """
    h, w = mask.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = (gray < 20).astype(np.uint8) * 255
    # close the holes left by the white digits so the label becomes one solid blob
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        # position / size / solidity checks
        if (y > 0.7 * h and 60 < bw < 0.4 * w and 10 < bh < 0.1 * h
                and area > 0.4 * bw * bh):
            mask[max(y - 3, 0):y + bh + 3, max(x - 3, 0):x + bw + 3] = 0
    return mask

def color_mask(img, cls):
    dist = np.linalg.norm(img.astype(np.int32) - np.array(OBJECT_COLORS[cls]), axis=2)
    return (dist < COLOR_TOLERANCE).astype(np.uint8) * 255

def extract_boxes(img):
    """
    Find each drawn box by its class color.
    Returns [(class_id, (x, y, w, h))] in pixel coordinates.
    """
    boxes = []
    for cls in OBJECT_COLORS:
        mask = color_mask(img, cls)
        # close the small holes left by the white label text
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if w < 20 or h < 20:
                continue                      # specks (cheap early filter)
            if area > 0.6 * w * h:
                continue                      # solid blob, not an outline
            comp = lab[y:y + h, x:x + w] == i
            # the label tab sticks up above the box: the true top edge is the first mostly-filled row
            top = int(np.argmax(comp.sum(axis=1) >= 0.8 * w))
            bw, bh = int(w), int(h - top)
            if bw < 30 or bh < 30:
                continue                      # too small once the label tab is trimmed off
            boxes.append((cls, (int(x), int(y + top), bw, bh)))
    return boxes

def remove_overlays(img):
    """Remove the drawn boxes and labels away so SIFT doesn't see them."""
    mask = np.zeros(img.shape[:2], np.uint8)
    for cls in OBJECT_COLORS:
        mask |= color_mask(img, cls)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

def check_boxes(folder=TRAIN_IMGS, n=10):
    """Debug: save extracted boxes, cleaned copies and the timestamp mask for n frames."""
    os.makedirs(OUTPUT, exist_ok=True)
    for path in list_images(folder)[:n]:
        name = os.path.basename(path)
        img = load_image(path)
        vis = img.copy()
        for cls, (x, y, w, h) in extract_boxes(img):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis, OBJECT_NAMES[cls], (x, y + h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(OUTPUT, "check_" + name), vis)
        cv2.imwrite(os.path.join(OUTPUT, "clean_" + name), remove_overlays(img))

        # masked-out pixels are shown darkened and tinted red
        m = mask_timestamp(np.full(img.shape[:2], 255, np.uint8), img)
        ts = img.copy()
        ts[m == 0] = (ts[m == 0] * 0.3).astype(np.uint8) + np.array([0, 0, 120], np.uint8)
        cv2.imwrite(os.path.join(OUTPUT, "timestamp_" + name), ts)
    print(f"Saved check images to '{OUTPUT}/' (check_*, clean_*, timestamp_*). Open and confirm.")

def box_mask(img, box, inset=5, top_inset=20, exclude=()):
    """
    Mask for the interior of an object's box. The inset skips the drawn outline,
    top_inset also skips the label tab, and `exclude` carves out overlapping boxes.
    """
    x, y, w, h = box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y + top_inset:y + h - inset, x + inset:x + w - inset] = 255
    for ex, ey, ew, eh in exclude:
        mask[ey:ey + eh, ex:ex + ew] = 0
    return mask_timestamp(mask, img)


def scene_mask(img):
    """Scene mask: everything allowed except the timestamp."""
    return mask_timestamp(np.full(img.shape[:2], 255, dtype=np.uint8), img)


def detect_and_describe(image, mask=None):
    """
    Detect keypoints and compute descriptors.
    Return (keypoints, descriptors).
    """
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, mask)

def match_descriptors(desc_object, desc_scene):
    """
    Match object descriptors to scene descriptors and keep good matches
    using the ratio test (RATIO_TEST). Return a list of cv2.DMatch.
    """
    if desc_object is None or desc_scene is None or len(desc_scene) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(desc_object, desc_scene, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance:
            good_matches.append(pair[0])
    return good_matches


def is_present(good_matches, threshold):
    """
    TODO: Decide whether the object is present in the scene.
    Return True or False. Record the counts you see in Results.xlsx (Match Counts).
    """
    return len(good_matches) >= threshold


def locate_object(kp_scene, good_matches):
    """
    Bounding rectangle (x, y, w, h) of the scene positions of the good matches,
    after removing stray points far from the rest. Returns None if too few remain.
    """
    if len(good_matches) < 4:
        return None

    points = np.array([kp_scene[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    k = min(15, len(points) - 1)  # k neighbors to measure distance against, change if needed
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    knn_dist = np.sort(dists, axis=1)[:, k]

    cutoff = 3.0 * max(np.median(knn_dist), 1.0)  # floor avoids a zero cutoff, change if needed
    filtered_points = points[knn_dist <= cutoff]

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

def overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah

def build_templates():
    """Return {class_id: [descriptor arrays]}. Cached on disk because this is slow."""
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)

    pool = {cls: [] for cls in OBJECT_NAMES}   # cls -> [(n_keypoints, descriptors)]
    for path in list_images(TRAIN_IMGS):
        img = load_image(path)
        boxes = extract_boxes(img)
        for idx, (cls, box) in enumerate(boxes):
            # carve out any other box overlapping this one (e.g. the car inside the bus box)
            others = [b for j, (_, b) in enumerate(boxes) if j != idx and overlaps(box, b)]
            kp, desc = detect_and_describe(img, box_mask(img, box, exclude=others))
            if desc is not None and len(kp) >= 2:   # skip objects with no usable keypoints
                pool[cls].append((len(kp), desc))

    templates = {}
    for cls, items in pool.items():
        items.sort(key=lambda t: t[0], reverse=True)
        templates[cls] = [d for _, d in items[:MAX_TEMPLATES_PER_CLASS]]
        print(f"{OBJECT_NAMES[cls]}: {len(items)} usable templates, keeping {len(templates[cls])}")

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

def load_val_ground_truth(path):
    """Return (ground-truth boxes, clean image) for one validation frame."""
    img = load_image(path)
    boxes = extract_boxes(img)
    return boxes, remove_overlays(img)


def collect_val_results():
    """
    For every validation frame and class, store (best good-match count, None,
    [ground-truth boxes of that class]). The middle slot is unused now, since
    tuning is on presence only, but it keeps the tuple shape consistent.
    """
    templates = build_templates()
    results = []   # one dict per frame: {cls: (count, None, [gt boxes])}
    paths = list_images(VAL_IMGS)
    for i, path in enumerate(paths, 1):
        gt, clean = load_val_ground_truth(path)
        kp_s, desc_s = detect_and_describe(clean, scene_mask(clean))
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
        actual = len(gts) > 0            # is this class really in the frame?
        predicted = count >= threshold   # good matches >= threshold
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
    tpr = tp / (tp + fn) if tp + fn else 0.0   # recall on frames WITH the object
    tnr = tn / (tn + fp) if tn + fp else 0.0   # recall on frames WITHOUT it
    return (tpr + tnr) / 2

# ignore, its for finding thresholds
def tune(max_t=300):
    """Pick each class's threshold on all validation frames by balanced accuracy."""
    results = collect_val_results()
    chosen = {}

    for cls, name in OBJECT_NAMES.items():
        pos = sorted(f[cls][0] for f in results if f[cls][2])
        neg = sorted(f[cls][0] for f in results if not f[cls][2])
        print(f"{name}\n  frames WITH it, counts:   {pos}\n  frames WITHOUT it, counts:{neg}")

    print(f"\nTuning on all {len(results)} validation frames (RATIO_TEST={RATIO_TEST})\n")
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
        raise ValueError(f"THRESHOLDS has no value for: {', '.join(missing)}. "
                         f"Run `python script.py tune` and paste the line it prints.")

    os.makedirs(OUTPUT, exist_ok=True)
    templates = build_templates()
    print(f"Using thresholds: { {OBJECT_NAMES[c]: t for c, t in THRESHOLDS.items()} }")

    rows = []
    for path in list_images(TEST_IMG):
        fname = os.path.basename(path)
        scene = load_image(path)
        kp_s, desc_s = detect_and_describe(scene, scene_mask(scene))
        output = scene.copy()

        for cls, name in OBJECT_NAMES.items():
            good = best_matches(templates[cls], desc_s)
            present = is_present(good, THRESHOLDS[cls])

            # These counts are what is recorded in Results.xlsx (Match Counts)
            print(f"{fname:30s} {name:11s} good_matches={len(good):4d} "
                  f"threshold={THRESHOLDS[cls]:4d} present={present}")
            rows.append((fname, name, len(good), THRESHOLDS[cls], present))

            if present:
                box = locate_object(kp_s, good)
                if box is not None:
                    draw_box(output, box, f"{name} ({len(good)})")

        cv2.imwrite(os.path.join(OUTPUT, fname), output)

    with open(os.path.join(OUTPUT, "match_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "class", "good_matches", "threshold", "present"])
        w.writerows(rows)
    print(f"Done. Annotated images and match_counts.csv saved to '{OUTPUT}/'.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "check":
        check_boxes(sys.argv[2] if len(sys.argv) > 2 else TRAIN_IMGS)
    elif cmd == "tune":
        tune() # For finding thresholds, ignore  
    else:
        main()
