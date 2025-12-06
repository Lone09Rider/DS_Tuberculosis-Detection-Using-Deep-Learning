import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src import data_prep
from src.config import TrainConfig, ensure_output_dir


MODEL_FACTORIES = {
    "efficientnetb0": lambda input_shape: keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    ),
    "resnet50": lambda input_shape: keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    ),
    "vgg16": lambda input_shape: keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    ),
}


def build_model(cfg: TrainConfig) -> keras.Model:
    input_shape = (cfg.image_size, cfg.image_size, 3)
    base = MODEL_FACTORIES[cfg.model_name](input_shape)
    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(cfg.class_names), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


def compile_model(model: keras.Model, cfg: TrainConfig) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(cfg: TrainConfig):
    if cfg.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    df = data_prep.load_metadata(cfg.csv)
    train_ds = data_prep.build_tf_dataset(df, cfg, cfg.train_split_name, shuffle=True)
    val_ds = data_prep.build_tf_dataset(df, cfg, cfg.val_split_name, shuffle=False)

    class_weights = None
    if cfg.class_weight:
        class_weights = data_prep.compute_class_weights(df[df["split"].isin([cfg.train_split_name])], cfg.class_names)

    model = build_model(cfg)
    model = compile_model(model, cfg)

    out_dir = ensure_output_dir(cfg.output_dir)
    ckpt_path = out_dir / "best.keras"
    callbacks = [
        # Cast to str because keras callbacks expect string paths, not pathlib objects on Windows
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_auc", mode="max", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    final_path = out_dir / "final.keras"
    model.save(str(final_path))
    return history, ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TB classifier")
    parser.add_argument("--csv", type=Path, default=TrainConfig.csv, help="Metadata CSV path")
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--model", choices=list(MODEL_FACTORIES.keys()), default=TrainConfig.model_name)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--no-class-weight", action="store_true", help="Disable class weighting")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        csv=args.csv,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_name=args.model,
        output_dir=args.output_dir,
        augment=not args.no_augment,
        class_weight=not args.no_class_weight,
        mixed_precision=args.mixed_precision,
    )
    train(cfg)


if __name__ == "__main__":
    main()
