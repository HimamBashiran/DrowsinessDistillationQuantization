"""PyTorch Datasets, metadata builders, and leak-free per-fold Z-score scaler."""

from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import CONFIG


def build_point_feature_df(feature_dir: Path = CONFIG.point_feature_dir) -> pd.DataFrame:
    """Index all Track A .pkl files with subject and label."""
    records = []
    for pkl_path in sorted(Path(feature_dir).glob("*.pkl")):
        fname = pkl_path.name
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "features" in data:
            label = int(data.get("label", 1 if fname.startswith("drowsiness_") else 0))
            subject = data.get("subject")
        else:
            label = 0 if fname.startswith("non-drowsiness") else 1
            subject = None
        if not subject:
            m = re.search(r"(subject_\d+)", fname)
            subject = m.group(1) if m else "unknown"
        records.append({"path": str(pkl_path), "filename": fname, "subject": subject, "label": label})
    return pd.DataFrame(records)


def build_pixel_clip_df(clip_dir: Path = CONFIG.pixel_cache_dir) -> pd.DataFrame:
    """Index all Track B .npy face-cropped clips with subject and label."""
    records = []
    for npy_path in sorted(Path(clip_dir).glob("*.npy")):
        fname = npy_path.name
        label = 0 if fname.startswith("non-drowsiness") else 1
        m = re.search(r"(subject_\d+)", fname)
        subject = m.group(1) if m else "unknown"
        records.append({"path": str(npy_path), "filename": fname, "subject": subject, "label": label})
    return pd.DataFrame(records)


def split_subject_independent(df: pd.DataFrame, test_subjects: list[str] = CONFIG.test_subjects):
    """Split DataFrame into train pool (Subjects 01-15) and hold-out test set (Subjects 16-20)."""
    df_train = df[~df["subject"].isin(test_subjects)].reset_index(drop=True)
    df_test = df[df["subject"].isin(test_subjects)].reset_index(drop=True)
    return df_train, df_test


def compute_point_scaler(feature_paths) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean and std strictly from the given training fold paths."""
    seqs = []
    for p in feature_paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        feats = d["features"] if isinstance(d, dict) and "features" in d else d
        seqs.append(np.asarray(feats, dtype=np.float32))
    concat = np.concatenate(seqs, axis=0)
    mean = torch.tensor(concat.mean(axis=0, keepdims=True), dtype=torch.float32)
    std = torch.tensor(concat.std(axis=0, keepdims=True) + 1e-6, dtype=torch.float32)
    return mean, std


class PointSequenceDataset(Dataset):
    """Track A Dataset: (120, 109) sequences with fold Z-score normalization and training augmentation."""

    def __init__(self, paths, labels, scaler_mean=None, scaler_std=None, augment: bool = False):
        self.paths = list(paths)
        self.labels = list(labels)
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with open(self.paths[idx], "rb") as f:
            data = pickle.load(f)
        feats = data["features"] if isinstance(data, dict) and "features" in data else data
        x = torch.tensor(feats, dtype=torch.float32)

        if self.scaler_mean is not None and self.scaler_std is not None:
            x = (x - self.scaler_mean) / self.scaler_std

        if self.augment:
            if CONFIG.aug_scale > 0:
                scale = torch.empty(1, x.size(1)).uniform_(1.0 - CONFIG.aug_scale, 1.0 + CONFIG.aug_scale)
                x = x * scale
            if CONFIG.aug_noise > 0:
                x = x + torch.randn_like(x) * CONFIG.aug_noise
            if CONFIG.aug_frame_drop > 0:
                mask = (torch.rand(x.size(0), 1) >= CONFIG.aug_frame_drop).float()
                x = x * mask

        return x, torch.tensor(self.labels[idx], dtype=torch.long)


class PixelClipDataset(Dataset):
    """Track B Dataset: (3, 120, 112, 112) Kinetics-normalized face clips."""

    def __init__(self, paths, labels, augment: bool = False):
        self.paths = list(paths)
        self.labels = list(labels)
        self.augment = augment
        self.mean = torch.tensor(CONFIG.kinetics_mean, dtype=torch.float32).view(3, 1, 1, 1)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        clip = np.load(self.paths[idx])  # (T, H, W, C) uint8
        x = torch.from_numpy(clip).permute(3, 0, 1, 2).float()  # (C, T, H, W), 0..255

        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[3])
            brightness = torch.empty(1).uniform_(0.9, 1.1).item()
            x = torch.clamp(x * brightness, 0.0, 255.0)

        x = x - self.mean
        return x, torch.tensor(self.labels[idx], dtype=torch.long)
