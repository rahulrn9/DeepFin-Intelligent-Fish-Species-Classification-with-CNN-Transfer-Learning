
import os, json, numpy as np, streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="FinSight: Fish Classifier", page_icon="🐟", layout="centered")
st.title("🐟 FinSight: Multiclass Fish Classifier")
st.caption("Upload a fish image and get the predicted species with confidence.")

@st.cache_resource
def jpeg_reencode(img: Image.Image, quality=90):
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def softmax_entropy(p):
    p = np.clip(p, 1e-8, 1.0)
    return -np.sum(p * np.log(p))

def load_extra_model(name):
    path = os.path.join("models", name)
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path, compile=False)
        except Exception:
            return None
    return None


def load_model_and_classes():
    models_dir = "models"
    model_path = os.path.join(models_dir, "best_model.h5")
    class_map_path = os.path.join(models_dir, "class_indices.json")
    model = None
    class_names = None

    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            st.warning(f"Could not load model: {e}")
    else:
        st.info("No `models/best_model.h5` found yet. Train models with `python src/train.py ...`")

    if os.path.exists(class_map_path):
        with open(class_map_path, "r") as f:
            class_indices = json.load(f)
        class_names = [None] * len(class_indices)
        for k, v in class_indices.items():
            class_names[v] = k
    return model, class_names

def infer_img_size(model):
    try:
        shape = model.input_shape
        h, w = shape[1], shape[2]
        return (w if w else 224), (h if h else 224)
    except Exception:
        return 224, 224

def preprocess(img: Image.Image, size):
    img = img.convert("RGB").resize(size)
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def predict(model, class_names, img: Image.Image):
    size = infer_img_size(model)
    x = preprocess(img, size)
    probs = model.predict(x)[0]
    topk_idx = probs.argsort()[-3:][::-1]
    results = [(class_names[i], float(probs[i])) for i in topk_idx]
    return results, probs

model, class_names = load_model_and_classes()

uploaded = st.file_uploader("Upload a fish image", type=["jpg","jpeg","png","bmp","webp"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model is not None and class_names is not None:
        with st.spinner("Predicting..."):
            top3, probs = predict(model, class_names, image)

        st.subheader("Prediction")
        best_name, best_conf = top3[0]
        st.write(f"**{best_name}**  •  Confidence: **{best_conf:.2%}**")

        st.write("Top-3 classes:")
        for name, p in top3:
            st.write(f"- {name}: {p:.2%}")
    else:
        st.warning("Model and/or class mapping not found. Please run training first.")
else:
    st.write("👆 Upload an image to begin.")
