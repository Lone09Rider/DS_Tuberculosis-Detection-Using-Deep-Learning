import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras

from src import data_prep
from src.config import TrainConfig
from src.training import MODEL_FACTORIES


def load_model(model_path: Path, model_name: str, image_size: int) -> keras.Model:
    # Rebuild backbone to ensure custom objects not needed
    input_shape = (image_size, image_size, 3)
    base = MODEL_FACTORIES[model_name](input_shape)
    base.trainable = False
    dummy = keras.Input(shape=input_shape)
    x = base(dummy, training=False)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(2, activation="softmax")(x)
    model = keras.Model(dummy, outputs)
    model.load_weights(model_path)
    return model


def evaluate(cfg: TrainConfig, model_path: Path) -> Tuple[float, float, float, float, float]:
    df = data_prep.load_metadata(cfg.csv)
    test_ds = data_prep.build_tf_dataset(df, cfg, cfg.test_split_name, shuffle=False)
    model = load_model(model_path, cfg.model_name, cfg.image_size)

    y_true = []
    y_prob = []
    for batch_images, batch_labels in test_ds:
        preds = model.predict(batch_images, verbose=0)
        y_true.extend(batch_labels.numpy().tolist())
        y_prob.extend(preds[:, 1].tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred, zero_division=0)
    rec = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    auc = metrics.roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    print({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc})
    return acc, prec, rec, f1, auc


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TB classifier")
    parser.add_argument("--csv", type=Path, default=TrainConfig.csv)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODEL_FACTORIES.keys()), default=TrainConfig.model_name)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(csv=args.csv, image_size=args.image_size, model_name=args.model)
    evaluate(cfg, args.model_path)


if __name__ == "__main__":
    main()
