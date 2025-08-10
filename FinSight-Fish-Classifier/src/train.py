
import os, json, argparse, pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
from utils import (
    seed_everything, make_flows,
    build_cnn_scratch, build_tl, compile_model, train_model, evaluate_model
)

TL_MODELS = ["VGG16","ResNet50","MobileNetV2","InceptionV3","EfficientNetB0"]

def main(args):
    # FAST MODE tweaks
    if args.fast:
        if args.img_size == 224: args.img_size = 160
        if args.batch_size == 16: args.batch_size = 32
        if args.epochs == 5: args.epochs = 3
        try: tf.config.optimizer.set_jit(True)
        except Exception: pass
        try: mixed_precision.set_global_policy('mixed_float16')
        except Exception: pass

    out_models = args.models_dir
    out_reports = args.reports_dir
    os.makedirs(out_models, exist_ok=True)
    os.makedirs(out_reports, exist_ok=True)

    train_dir = os.path.join(args.processed_dir, "train")
    val_dir = os.path.join(args.processed_dir, "val")
    classes = sorted(os.listdir(train_dir)) if os.path.exists(train_dir) else []

    train_flow, val_flow = make_flows(train_dir, val_dir, args.img_size, args.batch_size, fast=args.fast)
    class_names = list(train_flow.class_indices.keys())
    with open(os.path.join(out_models, "class_indices.json"), "w") as f:
        json.dump(train_flow.class_indices, f, indent=2)

    # Baseline scratch
    scratch = build_cnn_scratch(num_classes=len(class_names), img_size=args.img_size)
    scratch = compile_model(scratch)
    _ = train_model(scratch, train_flow, val_flow, args.epochs, out_models, "cnn_scratch", fast=args.fast)
    df, cm = evaluate_model(scratch, val_flow, class_names, out_reports, "cnn_scratch")

    # Narrow list in fast mode
    global TL_MODELS
    if args.fast:
        TL_MODELS = ["MobileNetV2", "EfficientNetB0"]

    # Transfer learning
    for m in TL_MODELS:
        try:
            model = build_tl(m, num_classes=len(class_names), img_size=args.img_size)
            model = compile_model(model)
            _ = train_model(model, train_flow, val_flow, args.epochs, out_models, m, fast=args.fast)
            df, cm = evaluate_model(model, val_flow, class_names, out_reports, m)
        except Exception as e:
            print(f"Skipping {m} due to error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--reports_dir", type=str, default="reports")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fast", action="store_true", help="Fast mode for quick baseline")
    args = parser.parse_args()
    main(args)
