from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"patient_id", "split", "label", "image_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def _augment(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image


def _load_image(path: tf.Tensor, image_size: int) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def _process_row(path: tf.Tensor, label: tf.Tensor, image_size: int, augment: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    image = _load_image(path, image_size)
    if augment:
        image = _augment(image)
    return image, label


def make_label_map(class_names) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}


def compute_class_weights(df: pd.DataFrame, class_names) -> Dict[int, float]:
    label_map = make_label_map(class_names)
    counts = df["label"].map(label_map).value_counts().to_dict()
    total = sum(counts.values())
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    return {int(k): float(v) for k, v in weights.items()}


def build_tf_dataset(df: pd.DataFrame, cfg, split: str, shuffle: bool = True) -> tf.data.Dataset:
    label_map = make_label_map(cfg.class_names)
    df_split = df[df["split"] == split].copy()
    paths = df_split["image_path"].values
    labels = df_split["label"].map(label_map).values

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    ds = ds.map(lambda p, l: _process_row(p, l, cfg.image_size, cfg.augment and shuffle), num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df_split))
    ds = ds.batch(cfg.batch_size).prefetch(AUTOTUNE)
    return ds
