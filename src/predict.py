import argparse
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import TrainConfig
from src.evaluate import load_model


def load_and_preprocess(path: Path, image_size: int) -> tf.Tensor:
    image = tf.io.read_file(str(path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def predict(paths: List[Path], model_path: Path, cfg: TrainConfig):
    model = load_model(model_path, cfg.model_name, cfg.image_size)
    images = tf.stack([load_and_preprocess(p, cfg.image_size) for p in paths])
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    idx_to_label = {i: name for i, name in enumerate(cfg.class_names)}
    results = []
    for p, pred_vec, pred_idx in zip(paths, probs, preds):
        results.append({
            "path": str(p),
            "label": idx_to_label[int(pred_idx)],
            "prob_TB": float(pred_vec[1]),
            "prob_Normal": float(pred_vec[0]),
        })
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Predict TB vs Normal for images")
    parser.add_argument("images", nargs="+", type=Path, help="Image file paths")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model", choices=["efficientnetb0", "resnet50", "vgg16"], default=TrainConfig.model_name)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(model_name=args.model, image_size=args.image_size)
    results = predict(args.images, args.model_path, cfg)
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
