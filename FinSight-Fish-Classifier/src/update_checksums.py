import os, hashlib, json, argparse

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--out", default="models/checksums.json")
    args = parser.parse_args()

    checks = {}
    if os.path.isdir(args.models_dir):
        for f in os.listdir(args.models_dir):
            if f.endswith((".h5",".tflite",".onnx")):
                p = os.path.join(args.models_dir, f)
                checks[f] = sha256(p)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(checks, fp, indent=2)
    print("Wrote", args.out)
