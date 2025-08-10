from fastapi import FastAPI, UploadFile, File, Security, HTTPException
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from PIL import Image, UnidentifiedImageError
import numpy as np, io, json, os, tensorflow as tf, hashlib

app = FastAPI(title="FinSight Fish API (Secure)")

# Rate limiting: 60 requests/min per IP
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda req, exc: HTTPException(status_code=429, detail="Too Many Requests"))
app.add_middleware(SlowAPIMiddleware)

# API Key
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def require_key(key=Security(api_key_header)):
    if not API_KEY:
        return  # open mode if not configured
    if not key or key != API_KEY:
        raise HTTPException(401, "Unauthorized")

# Model + class map with checksum verify
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.h5")
CLASS_MAP = os.getenv("CLASS_MAP", "models/class_indices.json")
CHECKSUMS = os.getenv("CHECKSUMS", "models/checksums.json")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

if os.path.exists(CHECKSUMS):
    try:
        checks = json.load(open(CHECKSUMS))
        if os.path.exists(MODEL_PATH):
            h = sha256(MODEL_PATH)
            expected = checks.get(os.path.basename(MODEL_PATH))
            if expected and h != expected:
                raise RuntimeError("Model checksum mismatch")
    except Exception as e:
        print("Checksum warning:", e)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(CLASS_MAP, "r") as f:
    class_indices = json.load(f)
classes = [None]*len(class_indices)
for k,v in class_indices.items():
    classes[v]=k

MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5*1024*1024)))
ALLOWED_TYPES = {"image/jpeg","image/png","image/webp"}

def preprocess(image: Image.Image, size=(224,224)):
    # JPEG re-encode to mitigate some adversarial noise and strip metadata
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image = image.resize(size)
    x = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(x, 0)

@app.post("/predict")
@limiter.limit("30/minute")
async def predict(file: UploadFile = File(...), _: str = Security(require_key)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Unsupported file type")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()   # verify first
        img = Image.open(io.BytesIO(data))  # reopen clean
    except UnidentifiedImageError:
        raise HTTPException(400, "Invalid image")

    size = (model.input_shape[1], model.input_shape[2])
    x = preprocess(img, size)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs)); conf = float(np.max(probs))
    label = classes[idx]
    return {"label": label, "confidence": conf}
