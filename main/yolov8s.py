import os
import sys
import csv
import json
import time
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from sift import load_image, list_images, prf, balanced_acc

OBJECT_NAMES = {0: "Car", 1: "Bus", 2: "Motorcycle"}

BASE = os.path.dirname(os.path.abspath(__file__))

TRAIN_IMG = os.path.join(BASE, "Images", "train no bounding box")
VAL_IMG   = os.path.join(BASE, "Images", "val no bounding box")
TEST_IMG   = os.path.join(BASE, "Images", "testing images")
TRAIN_LBL = os.path.join(BASE, "Images", "training labels")
VAL_LBL   = os.path.join(BASE, "Images", "validation labels")
TEST_LBL  = os.path.join(BASE, "Images", "testing labels")   # assumed name, change if yours differs
OUTPUT     = os.path.join(BASE, "Images", "YOLOv8s output")
YOLO_DATA  = os.path.join(BASE, "YOLO data")
DATA_YAML  = os.path.join(YOLO_DATA, "data.yaml")
RUNS       = os.path.join(BASE, "runs")
RUN_NAME   = "yolov8s_vehicles"

EPOCHS   = 50
BATCH    = 16          # drop to 8 if you run out of memory
PATIENCE = 15          # early stopping: epochs without improvement before stopping
IMGSZ    = 640
SEED     = 0
DEVICE   = None        # None = Ultralytics picks CUDA, then Apple MPS, then CPU

CONF      = 0.25       # confidence cut-off used for the test-set boxes, precision/recall and F1
IOU_MATCH = 0.5        # a predicted box is correct if IoU with a same-class ground-truth box >= this


# ---------------------------------------------------------------- small helpers

def hardware_description():
    """One line describing the machine, for the 'comparable hardware' note in the report."""
    import platform
    gpu = "none (CPU)"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            gpu = "Apple MPS"
    except Exception:
        pass
    return f"{platform.platform()}; {platform.processor() or platform.machine()}; GPU: {gpu}"


def _sync():
    """Wait for the GPU to finish so wall-clock timings are honest."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass


def read_label_file(path):
    """
    Read a YOLO .txt file. Returns ([(cls, cx, cy, w, h)], number_of_bad_lines),
    or (None, 0) if the file does not exist.
    """
    if not os.path.exists(path):
        return None, 0
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                if len(parts) != 5:
                    raise ValueError
                cls = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:])
                if cls not in OBJECT_NAMES or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError
            except ValueError:
                bad += 1
                continue
            rows.append((cls, cx, cy, w, h))
    return rows, bad


def write_label_file(path, rows):
    with open(path, "w") as f:
        for cls, cx, cy, w, h in rows:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def yolo_to_xyxy(rows, img_w, img_h):
    """[(cls, cx, cy, w, h)] normalised -> [(cls, (x1, y1, x2, y2))] in pixels."""
    out = []
    for cls, cx, cy, w, h in rows:
        out.append((cls, ((cx - w / 2) * img_w, (cy - h / 2) * img_h,
                          (cx + w / 2) * img_w, (cy + h / 2) * img_h)))
    return out


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def draw_boxes(img, boxes, color, thickness=2, labels=True):
    """boxes: [(cls, (x1, y1, x2, y2), conf_or_None)]"""
    for cls, (x1, y1, x2, y2), conf in boxes:
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, p1, p2, color, thickness)
        if labels:
            text = OBJECT_NAMES[cls] + (f" {conf:.2f}" if conf is not None else "")
            cv2.putText(img, text, (p1[0], max(p1[1] - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- dataset preparation

def prepare_split(split, img_dir, lbl_dir, write_labels=True):
    """
    Copy one split into the YOLO folder layout. The images are used as they are (clean
    originals) and each label is the existing .txt file with the same name as its image (<id>.jpg -> <id>.txt).
    """
    out_img = os.path.join(YOLO_DATA, "images", split)
    out_lbl = os.path.join(YOLO_DATA, "labels", split)
    os.makedirs(out_img, exist_ok=True)
    if write_labels:
        os.makedirs(out_lbl, exist_ok=True)

    stats = {"ids": [], "no_label": 0, "empty_label": 0, "bad_lines": 0,
             "objects": {c: 0 for c in OBJECT_NAMES}}

    for path in list_images(img_dir):
        stem = os.path.splitext(os.path.basename(path))[0]
        stats["ids"].append(stem)
        shutil.copy2(path, os.path.join(out_img, stem + ".jpg"))

        if not write_labels:
            continue
        rows, bad = read_label_file(os.path.join(lbl_dir, stem + ".txt"))
        stats["bad_lines"] += bad
        if rows is None:
            rows = []
            stats["no_label"] += 1
        elif not rows:
            stats["empty_label"] += 1
        write_label_file(os.path.join(out_lbl, stem + ".txt"), rows)
        for r in rows:
            stats["objects"][r[0]] += 1

    print(f"[{split}] {len(stats['ids'])} images")
    if write_labels:
        objs = ", ".join(f"{OBJECT_NAMES[c]}={n}" for c, n in stats["objects"].items())
        print(f"        objects: {objs} | {stats['bad_lines']} bad label lines skipped")
    if stats["no_label"]:
        print(f"        WARNING: {stats['no_label']} images have no .txt file in '{lbl_dir}'.")
    if stats["empty_label"]:
        # every image was picked because it contains a car, bus or motorcycle
        print(f"        WARNING: {stats['empty_label']} label files are empty, but every image should contain "
              f"at least one vehicle. Are those ids in gt_train.csv?")
    return stats


def prepare():
    """Build 'YOLO data/' (images, labels, data.yaml) from the Images/ folders."""
    for sub in ("images", "labels"):
        shutil.rmtree(os.path.join(YOLO_DATA, sub), ignore_errors=True)
    os.makedirs(YOLO_DATA, exist_ok=True)

    ids = {}
    ids["train"] = prepare_split("train", TRAIN_IMG, TRAIN_LBL)["ids"]
    ids["val"] = prepare_split("val", VAL_IMG, VAL_LBL)["ids"]
    test_has_labels = os.path.isdir(TEST_LBL)
    if not test_has_labels:
        print(f"WARNING: '{TEST_LBL}' not found, so mAP / precision / recall / F1 cannot be computed on the "
              f"test images. Create it with:\n  python make_yolo_labels.py gt_train.csv "
              f"\"Images/testing images\" \"Images/testing labels\"")
    ids["test"] = prepare_split("test", TEST_IMG, TEST_LBL, write_labels=test_has_labels)["ids"]

    # The splits must never share an image
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sorted(set(ids[a]) & set(ids[b]))
        if shared:
            raise ValueError(f"{len(shared)} images appear in both {a} and {b} (e.g. {shared[:3]}). "
                             f"Rebuild the splits so they do not overlap.")
    print(f"Splits are disjoint: train={len(ids['train'])}, val={len(ids['val'])}, test={len(ids['test'])}")

    root = os.path.abspath(YOLO_DATA).replace("\\", "/")
    with open(DATA_YAML, "w") as f:
        f.write(f'path: "{root}"\n')
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n")
        for c, n in OBJECT_NAMES.items():
            f.write(f"  {c}: {n}\n")
    print(f"Wrote {DATA_YAML}")


def check(n=10, split="train"):
    """Debug: draw the converted labels on a few prepared images. Open them and confirm the boxes line up."""
    os.makedirs(OUTPUT, exist_ok=True)
    for path in list_images(os.path.join(YOLO_DATA, "images", split))[:n]:
        name = os.path.basename(path)
        img = load_image(path)
        h, w = img.shape[:2]
        rows, _ = read_label_file(os.path.join(YOLO_DATA, "labels", split,
                                               os.path.splitext(name)[0] + ".txt"))
        boxes = [(c, b, None) for c, b in yolo_to_xyxy(rows or [], w, h)]
        cv2.imwrite(os.path.join(OUTPUT, f"check_yolo_{split}_{name}"),
                    draw_boxes(img, boxes, (0, 0, 255)))
    print(f"Saved check_yolo_{split}_* to '{OUTPUT}/'. Open and confirm the boxes sit on the vehicles.")


# ---------------------------------------------------------------- training

def train():
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError("data.yaml not found. Run `python yolov8s.py prepare` first.")
    model = YOLO("yolov8s.pt")   # COCO-pretrained; the head is re-built for our 3 classes

    start = time.perf_counter()
    model.train(
        data=DATA_YAML, epochs=EPOCHS, batch=BATCH, patience=PATIENCE, workers=2,
        imgsz=IMGSZ, seed=SEED, device=DEVICE,
        project=RUNS, name=RUN_NAME, exist_ok=True
    )
    elapsed = time.perf_counter() - start

    # Training time is stored separately from inference time (see test())
    run_dir = os.path.join(RUNS, RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "training_time.json"), "w") as f:
        json.dump({"training_time_s": round(elapsed, 1), "epochs_max": EPOCHS, "batch": BATCH,
                   "imgsz": IMGSZ, "patience": PATIENCE, "hardware": hardware_description()}, f, indent=2)
    print(f"Training took {elapsed / 60:.1f} min. Best weights: {os.path.join(run_dir, 'weights', 'best.pt')}")


# ---------------------------------------------------------------- testing

def match_class(preds, gts, cls):
    """Greedy one-to-one matching by confidence for one class. Returns (tp, fp, fn)."""
    p = sorted((x for x in preds if x[0] == cls), key=lambda t: -t[1])
    g = [b for c, b in gts if c == cls]
    used = [False] * len(g)
    tp = 0
    for _, _, pb in p:
        best, best_iou = -1, IOU_MATCH
        for j, gb in enumerate(g):
            if used[j]:
                continue
            iou = box_iou(pb, gb)
            if iou >= best_iou:
                best, best_iou = j, iou
        if best >= 0:
            used[best] = True
            tp += 1
    return tp, len(p) - tp, len(g) - tp


def test():
    weights = os.path.join(RUNS, RUN_NAME, "weights", "best.pt")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"{weights} not found. Run `python yolov8s.py train` first.")
    model = YOLO(weights)

    os.makedirs(os.path.join(OUTPUT, "gt_vs_pred"), exist_ok=True)
    test_lbl_dir = os.path.join(YOLO_DATA, "labels", "test")
    paths = list_images(os.path.join(YOLO_DATA, "images", "test"))
    if not paths:
        raise FileNotFoundError("No prepared test images. Run `python yolov8s.py prepare` first.")
    have_gt = os.path.isdir(test_lbl_dir) and any(f.endswith(".txt") for f in os.listdir(test_lbl_dir))
    if not have_gt:
        print("WARNING: no test labels were prepared. Only annotated images and timing will be produced.")

    # The first call includes one-off start-up cost, so it is not timed
    model.predict(load_image(paths[0]), imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)

    det = {c: {"tp": 0, "fp": 0, "fn": 0} for c in OBJECT_NAMES}
    pres = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in OBJECT_NAMES}
    times, per_image, det_rows, missing_labels = [], [], [], 0

    for path in paths:
        fname = os.path.basename(path)
        img = load_image(path)
        h, w = img.shape[:2]

        # Timed: preprocessing + network + NMS, on the decoded image (no file reading, drawing or saving)
        t0 = time.perf_counter()
        result = model.predict(img, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)[0]
        _sync()
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        preds = [(int(c), float(s), tuple(float(v) for v in b)) for c, s, b in zip(clss, confs, xyxy)]
        for c, s, (x1, y1, x2, y2) in preds:
            det_rows.append((fname, OBJECT_NAMES[c], round(s, 4), round(x1), round(y1),
                             round(x2 - x1), round(y2 - y1)))

        out = draw_boxes(img.copy(), [(c, b, s) for c, s, b in preds], (0, 255, 0))
        cv2.imwrite(os.path.join(OUTPUT, fname), out)

        row = {"image": fname, "n_pred": len(preds), "n_gt": "", "tp": "", "fp": "", "fn": "", "ms": round(ms, 2)}
        if have_gt:
            rows, _ = read_label_file(os.path.join(test_lbl_dir, os.path.splitext(fname)[0] + ".txt"))
            if rows is None:
                rows, missing_labels = [], missing_labels + 1
            gts = yolo_to_xyxy(rows, w, h)
            tp_i = fp_i = fn_i = 0
            for cls in OBJECT_NAMES:
                tp, fp, fn = match_class(preds, gts, cls)
                det[cls]["tp"] += tp; det[cls]["fp"] += fp; det[cls]["fn"] += fn
                tp_i += tp; fp_i += fp; fn_i += fn
                actual = any(c == cls for c, _ in gts)
                predicted = any(c == cls for c, _, _ in preds)
                key = ("tp" if actual else "fp") if predicted else ("fn" if actual else "tn")
                pres[cls][key] += 1
            row.update(n_gt=len(gts), tp=tp_i, fp=fp_i, fn=fn_i)

            cmp_img = draw_boxes(img.copy(), [(c, b, None) for c, b in gts], (255, 0, 0), 1, labels=False)
            draw_boxes(cmp_img, [(c, b, s) for c, s, b in preds], (0, 255, 0))
            cv2.imwrite(os.path.join(OUTPUT, "gt_vs_pred", fname), cmp_img)
        per_image.append(row)

    mean_ms, std_ms = float(np.mean(times)), float(np.std(times))
    hw = hardware_description()
    print(f"\nTest images: {len(paths)} | mean inference {mean_ms:.1f} ms/image (std {std_ms:.1f}) | {hw}")

    with open(os.path.join(OUTPUT, "detections.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "class", "confidence", "x", "y", "w", "h"])
        wr.writerows(det_rows)
    with open(os.path.join(OUTPUT, "per_image.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["image", "n_gt", "n_pred", "tp", "fp", "fn", "ms"])
        wr.writeheader()
        wr.writerows(per_image)

    summary = [("images", len(paths)), ("conf_threshold", CONF), ("iou_match", IOU_MATCH),
               ("mean_inference_ms_per_image", round(mean_ms, 2)), ("std_inference_ms", round(std_ms, 2)),
               ("hardware_test", hw)]
    t_path = os.path.join(RUNS, RUN_NAME, "training_time.json")
    if os.path.exists(t_path):
        with open(t_path) as f:
            tt = json.load(f)
        summary += [("training_time_s", tt["training_time_s"]), ("hardware_train", tt["hardware"])]

    if have_gt:
        if missing_labels:
            print(f"WARNING: {missing_labels} test images had no label file (treated as no vehicles).")

        # mAP@0.5 from Ultralytics' own evaluator (confidence-ranked, all confidences)
        metrics = model.val(data=DATA_YAML, split="test", imgsz=IMGSZ, device=DEVICE,
                            project=RUNS, name=RUN_NAME + "_test", exist_ok=True, verbose=False)
        map50 = float(metrics.box.map50)
        ap50 = {int(c): float(a) for c, a in zip(metrics.box.ap_class_index, metrics.box.ap50)}

        n = len(paths)
        print(f"\n{'Class':11s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AP@.5':>6s}"
              f" | presence: {'TP':>3s} {'FP':>3s} {'FN':>3s} {'TN':>3s} {'Acc':>5s} {'Bal':>5s}")
        class_rows, f1s, accs = [], [], []
        for cls, name in OBJECT_NAMES.items():
            d, pr = det[cls], pres[cls]
            p, r, f1 = prf(d["tp"], d["fp"], d["fn"])
            acc = (pr["tp"] + pr["tn"]) / n
            bal = balanced_acc(pr["tp"], pr["fp"], pr["fn"], pr["tn"])
            ap = ap50.get(cls, float("nan"))
            f1s.append(f1); accs.append(acc)
            print(f"{name:11s} {d['tp']:4d} {d['fp']:4d} {d['fn']:4d} {p:6.2f} {r:6.2f} {f1:6.2f} {ap:6.2f}"
                  f" | {'':9s} {pr['tp']:3d} {pr['fp']:3d} {pr['fn']:3d} {pr['tn']:3d} {acc:5.2f} {bal:5.2f}")
            class_rows.append([name, d["tp"], d["fp"], d["fn"], round(p, 4), round(r, 4), round(f1, 4),
                               round(ap, 4), pr["tp"], pr["fp"], pr["fn"], pr["tn"], round(acc, 4), round(bal, 4)])

        tp = sum(d["tp"] for d in det.values()); fp = sum(d["fp"] for d in det.values())
        fn = sum(d["fn"] for d in det.values())
        micro_p, micro_r, _ = prf(tp, fp, fn)
        macro_f1, presence_acc = float(np.mean(f1s)), float(np.mean(accs))
        print(f"\nmAP@0.5 = {map50:.3f} | macro-F1 = {macro_f1:.3f} | overall box precision = {micro_p:.3f}, "
              f"recall = {micro_r:.3f} | mean presence accuracy = {presence_acc:.3f}")
        summary += [("mAP@0.5", round(map50, 4)), ("macro_F1", round(macro_f1, 4)),
                    ("overall_precision", round(micro_p, 4)), ("overall_recall", round(micro_r, 4)),
                    ("mean_presence_accuracy", round(presence_acc, 4))]

        with open(os.path.join(OUTPUT, "class_metrics.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["class", "TP", "FP", "FN", "precision", "recall", "F1", "AP@0.5",
                         "presence_TP", "presence_FP", "presence_FN", "presence_TN",
                         "presence_accuracy", "presence_balanced_acc"])
            wr.writerows(class_rows)

        scored = sorted(per_image, key=lambda r: (r["fp"] + r["fn"], -r["tp"]))   # clean images that contain vehicles come first
        print("\nCleanest images (use for 'where it works'):", [r["image"] for r in scored[:5]])
        print("Most errors   (use for 'where it fails'): ", [r["image"] for r in scored[-5:][::-1]])

    with open(os.path.join(OUTPUT, "summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["metric", "value"])
        wr.writerows(summary)
    print(f"\nDone. Annotated images, gt_vs_pred/, detections.csv, per_image.csv, class_metrics.csv "
          f"and summary.csv saved to '{OUTPUT}/'.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "prepare":
        prepare()
    elif cmd == "check":
        check(split=sys.argv[2] if len(sys.argv) > 2 else "train")
    elif cmd == "train":
        train()
    elif cmd == "all":
        prepare(); train(); test()
    else:
        test()