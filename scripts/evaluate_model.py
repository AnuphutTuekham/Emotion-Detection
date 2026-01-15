"""Evaluate ONNX model on prepared FER2013 dataset and save confusion matrices.

Produces:
- confusion_matrix.png
- confusion_matrix_normalized.png
- results.png (per-class accuracy bar chart)

Usage:
  python scripts/evaluate_model.py --model public/models/onnx_model.fixed.onnx --data data/fer2013 --out results

"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(path, size=48):
    img = Image.open(path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # shape to (1,1,H,W)
    arr = arr.reshape(1, 1, size, size)
    return arr


def plot_confusion(cm, classes, outpath, title="Confusion matrix", cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="public/models/onnx_model.fixed.onnx")
    p.add_argument("--data", default="data/fer2013")
    p.add_argument("--labels", default="data/fer2013/labels.json")
    p.add_argument("--out", default="results")
    p.add_argument("--split", default="test", help="Which split to evaluate: test, val, train")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # load labels.json if provided, otherwise try to auto-discover dataset under args.data
    if os.path.exists(args.labels):
        labels = load_labels(args.labels)
    else:
        # auto-scan directory structure: expect <data>/<split>/<class>/*.jpg
        labels = []
        classes_json = Path("public/models/classes.json")
        if classes_json.exists():
            classes = json.loads(classes_json.read_text(encoding="utf-8"))
            class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            classes = None
            class_to_idx = {}

        split_dir = Path(args.data) / args.split
        if not split_dir.exists():
            print(f"Data split folder not found: {split_dir}")
            return

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            cls_name = class_dir.name
            if cls_name in class_to_idx:
                label_idx = class_to_idx[cls_name]
            else:
                # assign incremental indices if classes.json missing
                if cls_name not in class_to_idx:
                    class_to_idx[cls_name] = len(class_to_idx)
                label_idx = class_to_idx[cls_name]
            for img in sorted(class_dir.iterdir()):
                if img.is_file() and img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    rel = os.path.join(args.split, cls_name, img.name).replace("\\", "/")
                    labels.append({"file": rel, "label": label_idx, "split": args.split})

        # finalize classes list if using discovered mapping
        if classes is None:
            # produce classes list ordered by assigned indices
            classes = [None] * len(class_to_idx)
            for name, idx in class_to_idx.items():
                classes[idx] = name

    # collect items for the desired split
    items = [it for it in labels if it.get("split", "train") == args.split]
    if not items:
        print(f"No items found for split={args.split}")
        return

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print("Model inputs:", [i.name for i in session.get_inputs()])
    print("Model outputs:", [o.name for o in session.get_outputs()])

    y_true = []
    y_pred = []

    for it in items:
        img_path = Path(args.data) / it["file"]
        if not img_path.exists():
            print("Missing image, skipping:", img_path)
            continue
        arr = preprocess_image(img_path)
        feeds = {input_name: arr}
        out = session.run(None, feeds)
        logits = out[0].ravel()
        pred = int(np.argmax(logits))
        y_pred.append(pred)
        y_true.append(int(it["label"]))

    labels_unique = sorted(list(set(y_true + y_pred)))
    num_classes = max(labels_unique) + 1

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    # map class indices to names if classes is list of names
    class_names = classes if isinstance(classes, list) and len(classes) >= num_classes else [str(i) for i in range(num_classes)]

    plot_confusion(cm, class_names, os.path.join(args.out, "confusion_matrix.png"), title="Confusion matrix")
    plot_confusion(cm_norm, class_names, os.path.join(args.out, "confusion_matrix_normalized.png"), title="Normalized confusion matrix")

    # per-class accuracy (diagonal / support)
    support = cm.sum(axis=1)
    acc = np.zeros(num_classes)
    for i in range(num_classes):
        acc[i] = cm[i, i] / (support[i] + 1e-12)

    plt.figure(figsize=(10, 5))
    xs = np.arange(num_classes)
    plt.bar(xs, acc * 100)
    plt.xticks(xs, class_names, rotation=45, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-class accuracy")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "results.png"), bbox_inches="tight")
    plt.close()

    print("Saved:", os.path.join(args.out, "confusion_matrix.png"))
    print("Saved:", os.path.join(args.out, "confusion_matrix_normalized.png"))
    print("Saved:", os.path.join(args.out, "results.png"))


if __name__ == "__main__":
    main()
