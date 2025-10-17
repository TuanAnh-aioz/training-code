import logging
import os
import random

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from ..enums import Tasks
from .metrics import get_transforms

logger = logging.getLogger(__name__)


class ClassificationDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row["image_path"]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.class_to_idx[row["label"]], dtype=torch.long)
        return img, label, image_path


class DetectionDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        required_cols = {"image_path", "label", "xmin", "ymin", "xmax", "ymax"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV detection requires columns: {required_cols}")

        self.image_groups = self.df.groupby("image_path")
        self.image_paths = list(self.image_groups.groups.keys())

        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c: i + 1 for i, c in enumerate(self.classes)}
        self.class_to_idx["__background__"] = 0

        self.transform = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        group = self.image_groups.get_group(img_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        if self.transform:
            img = self.transform(img)

        _, new_h, new_w = img.shape

        # Scale bbox theo resize
        x_scale = new_w / orig_w
        y_scale = new_h / orig_h

        boxes, labels = [], []
        for _, row in group.iterrows():
            xmin, ymin, xmax, ymax = row[["xmin", "ymin", "xmax", "ymax"]].astype(float)
            boxes.append([xmin * x_scale, ymin * y_scale, xmax * x_scale, ymax * y_scale])
            labels.append(self.class_to_idx[row["label"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return img, target, img_path


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(task_type: str, config: dict, seed: int = 42):
    csv_file = config["csv_file"]
    batch_size = config.get("batch_size", 4)
    num_workers = config.get("num_workers", 4)
    val_ratio = config.get("val_ratio", 0.3)

    torch.manual_seed(seed)
    random.seed(seed)

    train_transform = get_transforms(config, task_type=task_type, mode="train")
    val_transform = get_transforms(config, task_type=task_type, mode="val")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    if task_type == Tasks.CLASSIFICATION.value:
        dataset = ClassificationDataset(csv_file)
        dataset.transform = train_transform

        # Stratified split
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_ratio, stratify=dataset.df["label"], random_state=seed)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        val_dataset.dataset.transform = val_transform

        collate = None
        total_len = len(dataset)
        train_len = len(train_dataset)
        val_len = len(val_dataset)

    elif task_type == Tasks.DETECTION.value:
        dataset = DetectionDataset(csv_file)
        dataset.transform = train_transform

        total_len = len(dataset)
        val_len = int(total_len * val_ratio)
        train_len = total_len - val_len

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))
        val_dataset.dataset.transform = val_transform
        collate = collate_fn

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate, pin_memory=True)

    logger.info(f"Dataset loaded: {total_len} samples ({train_len} train / {val_len} val)")
    return train_loader, val_loader, dataset
