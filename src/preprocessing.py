"""Preprocessing for Track A (120x109 landmarks) and Track B (120x112x112 face-cropped RGB)."""

from pathlib import Path
import pickle
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CONFIG, LANDMARK_INDICES


LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LIPS_IDX = [61, 291, 39, 181, 0, 17, 269, 405]

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def scan_telup_videos(dataset_dir: Path = CONFIG.raw_dataset_dir) -> pd.DataFrame:
    """Scan Telup dataset folder into a standardized metadata DataFrame."""
    records = []
    label_map = {"non-drowsiness": 0, "drowsiness": 1}
    for class_name, label in label_map.items():
        class_dir = Path(dataset_dir) / class_name
        if not class_dir.exists():
            continue
        for video_path in sorted(class_dir.rglob("*")):
            if video_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                rel_parts = video_path.relative_to(class_dir).parts
                subject = rel_parts[0] if len(rel_parts) > 1 else "unknown"
                records.append({
                    "video_path": str(video_path),
                    "filename": video_path.name,
                    "subject": subject,
                    "class_name": class_name,
                    "label": label,
                })
    return pd.DataFrame(records)


def sample_uniform_frames(video_path: str, num_frames: int = CONFIG.seq_len) -> list[np.ndarray]:
    """Uniformly sample `num_frames` RGB frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif frames:
            frames.append(frames[-1].copy())
    cap.release()
    while 0 < len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames]


def apply_clahe(image_rgb: np.ndarray, clip_limit: float = 2.0, grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid).apply(l_ch)
    return cv2.cvtColor(cv2.merge((cl, a_ch, b_ch)), cv2.COLOR_LAB2RGB)


def create_face_landmarker(model_path: Path = CONFIG.face_landmarker_path) -> vision.FaceLandmarker:
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _calc_ear(landmarks, indices, w: int, h: int) -> float:
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices], dtype=np.float64)
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    return float((v1 + v2) / (2.0 * horiz + 1e-6))


def _calc_mar(landmarks, indices, w: int, h: int) -> float:
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices], dtype=np.float64)
    v1 = np.linalg.norm(pts[2] - pts[3])
    v2 = np.linalg.norm(pts[4] - pts[5])
    v3 = np.linalg.norm(pts[6] - pts[7])
    horiz = np.linalg.norm(pts[0] - pts[1])
    return float((v1 + v2 + v3) / (3.0 * horiz + 1e-6))


def _estimate_head_pose(landmarks, w: int, h: int) -> tuple[float, float, float]:
    pose_idx = [1, 152, 33, 263, 61, 291]
    img_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in pose_idx], dtype=np.float64)
    cam = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rot_vec, _ = cv2.solvePnP(
        MODEL_POINTS_3D, img_pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return float(angles[0]), float(angles[1]), float(angles[2])


def _extract_point_vector(landmarks, w: int, h: int) -> np.ndarray:
    ear = (_calc_ear(landmarks, LEFT_EYE_IDX, w, h) + _calc_ear(landmarks, RIGHT_EYE_IDX, w, h)) / 2.0
    mar = _calc_mar(landmarks, LIPS_IDX, w, h)
    pitch, yaw, roll = _estimate_head_pose(landmarks, w, h)
    scalars = np.array([ear, mar, pitch, yaw, roll], dtype=np.float32)

    coords = np.array([[landmarks[i].x, landmarks[i].y] for i in LANDMARK_INDICES], dtype=np.float32)
    centered = coords - coords.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered, axis=1).max() + 1e-6
    return np.concatenate([scalars, (centered / scale).flatten()], axis=0)


def _crop_face_square(frame_rgb: np.ndarray, landmarks, margin: float, out_size: int) -> np.ndarray:
    h, w, _ = frame_rgb.shape
    if landmarks is None:
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = frame_rgb[y0:y0 + side, x0:x0 + side]
    else:
        xs = np.array([lm.x * w for lm in landmarks])
        ys = np.array([lm.y * h for lm in landmarks])
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        side = max(x_max - x_min, y_max - y_min) * (1.0 + 2.0 * margin)
        cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        x1 = max(0, int(cx - side / 2.0))
        y1 = max(0, int(cy - side / 2.0))
        x2 = min(w, int(cx + side / 2.0))
        y2 = min(h, int(cy + side / 2.0))
        crop = frame_rgb[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame_rgb
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def extract_point_features_cache(df_videos: pd.DataFrame, out_dir: Path = CONFIG.point_feature_dir) -> None:
    """Extract Track A (120, 109) .pkl files for any missing videos."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.pkl"))
    if len(existing) >= len(df_videos) and len(df_videos) > 0:
        print(f"[SKIP] Track A features already cached ({len(existing)} .pkl files).")
        return

    detector = create_face_landmarker()
    for _, row in tqdm(df_videos.iterrows(), total=len(df_videos), desc="Extracting Track A (.pkl)"):
        save_name = f"{row['class_name']}_{row['subject']}_{Path(row['filename']).stem}.pkl"
        save_path = out_dir / save_name
        if save_path.exists():
            continue
        frames = sample_uniform_frames(row["video_path"], CONFIG.seq_len)
        seq = []
        last_valid = np.zeros(CONFIG.point_input_dim, dtype=np.float32)
        for frame in frames:
            enhanced = apply_clahe(frame)
            h, w, _ = enhanced.shape
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced))
            if res.face_landmarks:
                last_valid = _extract_point_vector(res.face_landmarks[0], w, h)
            seq.append(last_valid.copy())
        feats = np.stack(seq, axis=0).astype(np.float32) if seq else np.zeros(
            (CONFIG.seq_len, CONFIG.point_input_dim), dtype=np.float32
        )
        with open(save_path, "wb") as f:
            pickle.dump({"features": feats, "label": row["label"], "subject": row["subject"]}, f)


def extract_pixel_clips_cache(df_videos: pd.DataFrame, out_dir: Path = CONFIG.pixel_cache_dir) -> None:
    """Extract Track B (120, 112, 112, 3) uint8 face-cropped clips (.npy) for any missing videos."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.npy"))
    if len(existing) >= len(df_videos) and len(df_videos) > 0:
        print(f"[SKIP] Track B face clips already cached ({len(existing)} .npy files).")
        return

    detector = create_face_landmarker()
    for _, row in tqdm(df_videos.iterrows(), total=len(df_videos), desc="Extracting Track B face clips (.npy)"):
        save_name = f"{row['class_name']}_{row['subject']}_{Path(row['filename']).stem}.npy"
        save_path = out_dir / save_name
        if save_path.exists():
            continue
        frames = sample_uniform_frames(row["video_path"], CONFIG.seq_len)
        crops = []
        last_landmarks = None
        for frame in frames:
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
            if res.face_landmarks:
                last_landmarks = res.face_landmarks[0]
            crops.append(_crop_face_square(frame, last_landmarks, CONFIG.face_crop_margin, CONFIG.sample_size))
        clip = np.stack(crops, axis=0).astype(np.uint8) if crops else np.zeros(
            (CONFIG.seq_len, CONFIG.sample_size, CONFIG.sample_size, 3), dtype=np.uint8
        )
        np.save(save_path, clip)
