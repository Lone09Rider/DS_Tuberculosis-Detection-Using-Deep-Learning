from pathlib import Path
from dataclasses import dataclass


@dataclass
class TrainConfig:
    csv: Path = Path("data/synthetic_metadata.csv")
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 1e-4
    model_name: str = "efficientnetb0"  # one of: efficientnetb0, resnet50, vgg16
    output_dir: Path = Path("models/efficientnetb0")
    val_split_name: str = "val"
    test_split_name: str = "test"
    train_split_name: str = "train"
    class_names: tuple = ("Normal", "TB")
    augment: bool = True
    mixed_precision: bool = False
    class_weight: bool = True


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
