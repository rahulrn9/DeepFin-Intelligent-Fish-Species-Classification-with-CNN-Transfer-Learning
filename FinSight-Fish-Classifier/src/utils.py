
import os, json, random, shutil, pathlib
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def datagens(img_size: int, batch_size: int, fast: bool=False):
    aug_kwargs = dict(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True
    )
    if fast:
        aug_kwargs.update(rotation_range=10, width_shift_range=0.05, height_shift_range=0.05, zoom_range=0.10)
    train_aug = ImageDataGenerator(**aug_kwargs)
    val_aug = ImageDataGenerator(rescale=1./255)
    return train_aug, val_aug

def make_flows(train_dir: str, val_dir: str, img_size: int, batch_size: int, fast: bool=False):
    train_aug, val_aug = datagens(img_size, batch_size, fast=fast)
    train_flow = train_aug.flow_from_directory(
        train_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical"
    )
    val_flow = val_aug.flow_from_directory(
        val_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode="categorical", shuffle=False
    )
    return train_flow, val_flow

def build_cnn_scratch(num_classes: int, img_size: int):
    from tensorflow.keras import layers, models
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="cnn_scratch")
    return model

def build_tl(model_name: str, num_classes: int, img_size: int):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    base = None
    if model_name == "VGG16":
        base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
        preprocess = tf.keras.applications.vgg16.preprocess_input
    elif model_name == "ResNet50":
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
        preprocess = tf.keras.applications.resnet50.preprocess_input
    elif model_name == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif model_name == "InceptionV3":
        base = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
        preprocess = tf.keras.applications.inception_v3.preprocess_input
    elif model_name == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Unknown model {model_name}")

    base.trainable = False
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = preprocess(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=f"{model_name}_finetune")
    return model

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_model(model, train_flow, val_flow, epochs: int, out_dir: str, model_name: str, fast: bool=False):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=(1 if fast else 3), restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, f"{model_name}.h5"), monitor="val_accuracy", save_best_only=True)
    ]
    hist = model.fit(train_flow, validation_data=val_flow, epochs=epochs, callbacks=callbacks)
    return hist

def evaluate_model(model, val_flow, class_names, out_dir: str, model_name: str):
    import numpy as np, pandas as pd, os
    val_flow.reset()
    preds = model.predict(val_flow, verbose=0)
    y_true = val_flow.classes
    y_pred = np.argmax(preds, axis=1)
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(out_dir, f"{model_name}_metrics.csv"))
    cm = confusion_matrix(y_true, y_pred)
    return df, cm
