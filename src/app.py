import argparse
from pathlib import Path
from functools import lru_cache

import streamlit as st
from PIL import Image
import tensorflow as tf

from src.config import TrainConfig
from src.predict import load_and_preprocess
from src.evaluate import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Streamlit app for TB detection")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to .keras weights")
    parser.add_argument("--model", choices=["efficientnetb0", "resnet50", "vgg16"], default=TrainConfig.model_name)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    return parser.parse_args()


@lru_cache(maxsize=1)
def get_model(model_path: Path, model_name: str, image_size: int):
    cfg = TrainConfig(model_name=model_name, image_size=image_size)
    return load_model(model_path, cfg.model_name, cfg.image_size), cfg


def main_cli():
    args = parse_args()
    run_app(args.model_path, args.model, args.image_size)


def run_app(model_path: Path, model_name: str, image_size: int):
    st.set_page_config(page_title="TB Detector", page_icon="🫁", layout="centered")
    st.title("Tuberculosis Detection on Chest X-rays")
    st.caption("Upload an X-ray to classify as TB or Normal. This is a demo, not medical advice.")

    model, cfg = get_model(model_path, model_name, image_size)

    uploaded = st.file_uploader("Upload chest X-ray", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        tensor = tf.keras.utils.img_to_array(image)
        tensor = tf.convert_to_tensor(tensor)
        tensor = tf.image.resize(tensor, [cfg.image_size, cfg.image_size])
        tensor = tf.cast(tensor, tf.float32) / 255.0
        tensor = tf.expand_dims(tensor, axis=0)

        preds = model.predict(tensor, verbose=0)[0]
        prob_normal, prob_tb = float(preds[0]), float(preds[1])
        label = "TB" if prob_tb >= prob_normal else "Normal"

        st.subheader(f"Prediction: {label}")
        st.write({"Normal": prob_normal, "TB": prob_tb})

    st.sidebar.markdown("## Settings")
    st.sidebar.write(f"Model: {model_name}")
    st.sidebar.write(f"Image size: {image_size}")
    st.sidebar.write(f"Model path: {model_path}")


if __name__ == "__main__":
    main_cli()
