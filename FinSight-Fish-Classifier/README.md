

### Fast mode (quick baseline)
```bash
python src/train.py --fast
```


## Security & Robustness Additions
- **API key** auth and **rate limiting** (SlowAPI) on `/predict`
- **Model checksums** verification (`src/update_checksums.py` → `models/checksums.json`)
- **File safety**: content-type allowlist, size cap, decode verify, EXIF strip
- **JPEG re-encode defense** toggle in the app
- **OOD detection**: confidence threshold + entropy threshold
- **Ensemble** toggle (MobileNetV2 + EfficientNetB0)
- **Calibration**: `src/calibration.py` generates ECE + reliability diagram

### Commands
```bash
# Generate checksums after training
python src/update_checksums.py --models_dir models --out models/checksums.json

# Run secure API
API_KEY=yourkey uvicorn api.main:app --host 0.0.0.0 --port 8000

# Calibration plots & ECE
python src/calibration.py --model_path models/best_model.h5 --class_map models/class_indices.json --val_dir data/processed/val --img_size 224
```
