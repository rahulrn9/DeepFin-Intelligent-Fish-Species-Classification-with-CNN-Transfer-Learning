import os, json, argparse, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import tensorflow as tf

def expected_calibration_error(probs, y_true, n_bins=15):
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0: 
            continue
        bin_acc = acc[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)

def reliability_plot(probs, y_true, out_path, n_bins=15):
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0: 
            continue
        xs.append((lo+hi)/2)
        ys.append(acc[mask].mean())
    plt.figure()
    plt.plot([0,1],[0,1])
    if len(xs) > 0:
        plt.plot(xs, ys, marker="o")
    plt.title("Reliability Diagram")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/best_model.h5")
    parser.add_argument("--class_map", default="models/class_indices.json")
    parser.add_argument("--val_dir", default="data/processed/val")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", default="reports")
    args = parser.parse_args()

    from utils import datagens
    val_aug = datagens(args.img_size, args.batch_size, fast=False)[1]
    val_flow = val_aug.flow_from_directory(args.val_dir, target_size=(args.img_size,args.img_size), batch_size=args.batch_size, class_mode="categorical", shuffle=False)
    model = tf.keras.models.load_model(args.model_path, compile=False)
    probs = model.predict(val_flow, verbose=0)
    y_true = val_flow.classes
    ece = expected_calibration_error(probs, y_true)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "calibration_metrics.json"), "w") as f:
        json.dump({"ECE": ece}, f, indent=2)
    reliability_plot(probs, y_true, os.path.join(args.out_dir, "reliability_diagram.png"))
    print("ECE:", ece)
