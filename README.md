DeepFin — Intelligent Fish Species Classification with CNN & Transfer Learning
DeepFin is an end-to-end project that classifies fish images into species using a baseline CNN and multiple transfer-learning backbones. It ships with a Streamlit app for real-time predictions, a FastAPI microservice with security guardrails, calibration & explainability tools, and optional fast mode for quick iteration.

Highlights
Models: CNN (scratch) + TL backbones (MobileNetV2, EfficientNetB0, …).

Fast mode: one flag for quick, accurate baselines (smaller size, fewer epochs, mixed precision).

Deployment: Streamlit UI (single-image & batch ZIP), FastAPI endpoint, Dockerfile.

Security: API key auth, rate limiting, file safety checks, model checksum verification.

Robustness: “Unknown fish” via confidence + entropy, optional JPEG re-encode defense, ensemble toggle.

Explainability: Grad-CAM overlay.

Calibration: ECE + reliability diagram.

Artifacts: metrics CSVs, confusion matrices, calibration plots.

Project Structure
perl
Copy
Edit
DeepFin/
├─ app/
│  └─ app.py                   # Streamlit app (UI, Grad-CAM, ensemble, OOD, batch)
├─ api/
│  └─ main.py                  # FastAPI service with auth, rate limit, file safety
├─ data/
│  ├─ raw/                     # (optional) sample/raw images
│  └─ processed/
│     ├─ train/<species>/
│     └─ val/<species>/        # expected training layout
├─ models/
│  ├─ best_model.h5            # saved best model (after training)
│  ├─ class_indices.json       # label mapping
│  └─ checksums.json           # model integrity hashes (optional)
├─ reports/
│  ├─ model_comparison.csv     # (if generated)
│  ├─ calibration_metrics.json # (if generated)
│  └─ reliability_diagram.png  # (if generated)
├─ src/
│  ├─ train.py                 # training loop (+ --fast mode)
│  ├─ utils.py                 # data loaders, builders, training helpers
│  ├─ adv_utils.py             # TTA, Grad-CAM, misclass CSVs (advanced utils)
│  ├─ calibration.py           # ECE + reliability diagram
│  ├─ export_tflite.py         # Keras -> TFLite export
│  └─ update_checksums.py      # generate model checksums
├─ Dockerfile
├─ requirements.txt
└─ .gitignore
Note on data: This repo expects a train/val folder split. If you have a single folder per class, use any splitter script (or quickly copy 80% to train/ and 20% to val/) before running.

Dataset Format
Place your dataset as:

kotlin
Copy
Edit
data/processed/
├─ train/
│  ├─ Salmon/
│  ├─ Tuna/
│  └─ Carp/
└─ val/
   ├─ Salmon/
   ├─ Tuna/
   └─ Carp/
Images can be .jpg/.jpeg/.png/.bmp/.webp.

Class names are folder names.

Setup
bash
Copy
Edit
# 1) Create environment and install deps
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
Optional (large files):
git lfs install && git lfs track "*.h5" "*.tflite" "*.onnx" "data/*.zip"

Quickstart
A) Fast baseline (recommended first run)
Fast mode narrows models (MobileNetV2 + EfficientNetB0), uses img_size=160, epochs=3, batch=32, mixed precision + XLA, lighter augmentations.

bash
Copy
Edit
python src/train.py --fast \
  --processed_dir data/processed \
  --models_dir models \
  --reports_dir reports
B) Standard training
bash
Copy
Edit
python src/train.py \
  --processed_dir data/processed \
  --models_dir models \
  --reports_dir reports \
  --img_size 224 --batch_size 16 --epochs 10
Outputs

Saved models: models/<model_name>.h5

Label map: models/class_indices.json

Metrics: reports/<model_name>_metrics.csv

(If present) comparison table: reports/model_comparison.csv

Tip: After training, pick or copy your best weights to models/best_model.h5 (some flows already do this).

Streamlit App (UI)
Run the web app:

bash
Copy
Edit
streamlit run app/app.py
Features

Upload an image → predicted species + confidence

Unknown detection (confidence & entropy thresholds)

TTA toggle

Grad-CAM overlay

Ensemble toggle (averages MobileNetV2 + EfficientNetB0 if present)

JPEG re-encode defense (mitigates some noisy/adversarial inputs)

Batch ZIP inference with downloadable CSV

The app prefers MobileNetV2.h5 when available; otherwise best_model.h5.

FastAPI (Secure Inference API)
bash
Copy
Edit
# Generate checksums after you have model files (optional but recommended)
python src/update_checksums.py --models_dir models --out models/checksums.json

# Run the API (set an API key)
API_KEY=your-strong-key uvicorn api.main:app --host 0.0.0.0 --port 8000
Auth: x-api-key: your-strong-key (if API_KEY is set)

Rate limits: 60/min (global), 30/min on /predict

File safety: content-type allowlist (jpeg/png/webp), 5 MB cap, decode verification, EXIF strip

Model integrity: verifies SHA-256 against models/checksums.json (if present)

Example cURL
bash
Copy
Edit
curl -X POST "http://localhost:8000/predict" \
  -H "x-api-key: your-strong-key" \
  -F "file=@sample.jpg"
Robustness & Calibration
“Unknown fish” / OOD
Confidence threshold (max softmax)

Entropy threshold (higher entropy → more uncertain)
Configure both in the Streamlit sidebar.

Calibration (ECE + Reliability Diagram)
bash
Copy
Edit
python src/calibration.py \
  --model_path models/best_model.h5 \
  --class_map models/class_indices.json \
  --val_dir data/processed/val \
  --img_size 224
Outputs to reports/: calibration_metrics.json and reliability_diagram.png.

Export for Edge (TFLite)
bash
Copy
Edit
python src/export_tflite.py \
  --model_path models/best_model.h5 \
  --out_path models/best_model.tflite
(Optional) Wire the Streamlit app to prefer .tflite for even faster CPU inference.

Explainability (Grad-CAM)
In the app, enable “Show Grad-CAM” to visualize salient regions (works for conv backbones). Useful for debugging misclassifications and building trust.

Security Notes
API Key (env API_KEY) & rate limiting (SlowAPI).

File validation: size/type checks, safe decode, EXIF strip, and optional JPEG re-encode defense.

Checksum verification for models (supply chain integrity).

Run Docker with least privilege (non-root, read-only FS, --cap-drop ALL).

Docker
A simple production Dockerfile is included. Example:

bash
Copy
Edit
docker build -t deepfin:latest .
docker run -p 8501:8501 -p 8000:8000 \
  -e API_KEY=your-strong-key \
  --read-only --cap-drop ALL \
  deepfin:latest
(Default CMD runs Streamlit; override to run FastAPI:
docker run ... uvicorn api.main:app --host 0.0.0.0 --port 8000)

Tips & Best Practices
Imbalance: consider class weights or focal loss if some species are rare.

Data quality: crop/segment fish to reduce background bias; consider a “fish vs not-fish” gate.

Few-shot: add a metric-learning head and a small k-NN for new species onboarding.

Monitoring: log unknown/OOD rate, per-class accuracy, latency; alert on drift.

Troubleshooting
No classes found: ensure data/processed/train/<class>/ and val/<class>/ exist with images.

Out of memory (GPU): reduce --batch_size, use --img_size 160, or --fast.

Grad-CAM blank: some non-conv tops may not expose a final conv layer; switch to a supported backbone.

API 401: set API_KEY in env and include x-api-key header.

Roadmap (Nice-to-haves)
TFLite execution in Streamlit (auto-select .tflite when present)

Active learning loop in the app (review & correct low-confidence images)

Automated data split utility (single-folder dataset → train/val)
