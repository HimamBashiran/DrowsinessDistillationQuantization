"""Centralized configuration for Track A (Point/Landmark) and Track B (Pixel/3D CNN)."""

from dataclasses import dataclass, field
from pathlib import Path
import os
import random
import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent.parent

# MediaPipe 52 landmark indices (left eye, right eye, mouth)
LEFT_EYE_LM = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_LM = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_LM = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LANDMARK_INDICES = LEFT_EYE_LM + RIGHT_EYE_LM + MOUTH_LM
NUM_SCALAR_FEATURES = 5  # [EAR, MAR, Pitch, Yaw, Roll]
POINT_INPUT_DIM = NUM_SCALAR_FEATURES + len(LANDMARK_INDICES) * 2  # 5 + 104 = 109


@dataclass
class Config:
    dataset_name: str = "telup"
    base_dir: Path = BASE_DIR
    raw_dataset_dir: Path = BASE_DIR / "drowsiness dataset"
    point_feature_dir: Path = BASE_DIR / "hasil extract 120 frame"
    pretrained_dir: Path = BASE_DIR / "pretrained_models"
    face_landmarker_path: Path = BASE_DIR / "pretrained_models" / "face_landmarker.task"
    kinetics_dir: Path = BASE_DIR / "pretrained_models" / "kinetics_3dcnn"
    pixel_cache_root: Path = BASE_DIR / "hasil-extract-120-frame"
    results_root: Path = BASE_DIR / "results"

    # Shared experiment protocol
    seq_len: int = 120
    num_classes: int = 2
    folds: int = 5
    seed: int = 42
    test_subjects: list[str] = field(
        default_factory=lambda: [f"subject_{i:02d}" for i in range(16, 21)]
    )
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Track A (Point / Landmark) — Scenario S4
    point_input_dim: int = POINT_INPUT_DIM
    teacher_cfg: dict = field(
        default_factory=lambda: {"hidden": 32, "layers": 3, "heads": 2, "ffn": 128, "cls": 32}
    )
    student_cfg: dict = field(
        default_factory=lambda: {"hidden": 16, "layers": 2, "heads": 2, "ffn": 64, "cls": 16}
    )
    dropout: float = 0.35
    batch_size: int = 16
    epochs: int = 50
    warmup_epochs: int = 5
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 5e-2
    grad_clip: float = 1.0
    label_smoothing: float = 0.10
    patience: int = 15

    # Track A training-only temporal augmentation
    aug_noise: float = 0.08
    aug_scale: float = 0.10
    aug_frame_drop: float = 0.10

    # Knowledge Distillation & QAT
    kd_temperature: float = 3.0
    kd_alpha: float = 0.5
    qat_bits: int = 8

    # Track B (Pixel / 3D CNN)
    sample_size: int = 112
    face_crop_margin: float = 0.25
    kinetics_mean: tuple[float, float, float] = (114.7748, 107.7354, 99.4750)
    pixel_finetune_mode: str = "last_block"  # "last_block" (CPU-friendly) or "full"
    pixel_batch_size: int = 4
    pixel_epochs: int = 20
    pixel_warmup_epochs: int = 2
    pixel_lr: float = 1e-4
    pixel_weight_decay: float = 1e-2
    pixel_patience: int = 7

    # XAI
    shap_bg_samples: int = 16
    scorecam_top_k: int = 32

    @property
    def pixel_cache_dir(self) -> Path:
        return self.pixel_cache_root / self.dataset_name

    @property
    def result_dir(self) -> Path:
        return self.results_root / self.dataset_name

    @property
    def ckpt_track_a_dir(self) -> Path:
        return self.result_dir / "checkpoints" / "track_a"

    @property
    def ckpt_track_b_dir(self) -> Path:
        return self.result_dir / "checkpoints" / "track_b"

    @property
    def tables_dir(self) -> Path:
        return self.result_dir / "tables"

    @property
    def per_model_dir(self) -> Path:
        return self.tables_dir / "per_model"

    @property
    def figures_dir(self) -> Path:
        return self.result_dir / "figures"

    def ensure_dirs(self) -> None:
        dirs = [
            self.pretrained_dir,
            self.kinetics_dir,
            self.pixel_cache_dir,
            self.ckpt_track_a_dir,
            self.ckpt_track_b_dir,
            self.per_model_dir,
            self.figures_dir / "eda",
            self.figures_dir / "training_curves",
            self.figures_dir / "comparison",
            self.figures_dir / "xai_shap",
            self.figures_dir / "xai_scorecam",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


CONFIG = Config()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
