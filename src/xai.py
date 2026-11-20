"""Explainable AI (XAI): SHAP for Track A (Point) and Top-K 3D Score-CAM for Track B (Pixel)."""

import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F

from src.config import CONFIG
from src.datasets import PixelClipDataset, PointSequenceDataset
from src.trainer import _safe_slug


def _get_shap_group_names() -> list[tuple[str, slice]]:
    return [
        ("EAR", slice(0, 1)),
        ("MAR", slice(1, 2)),
        ("Head Pitch", slice(2, 3)),
        ("Head Yaw", slice(3, 4)),
        ("Head Roll", slice(4, 5)),
        ("Left Eye Landmarks", slice(5, 37)),
        ("Right Eye Landmarks", slice(37, 69)),
        ("Mouth Landmarks", slice(69, 109)),
    ]


def run_shap_analysis(
    model_fn,
    ckpt_path: str,
    model_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> pd.DataFrame:
    """Compute SHAP values for a Track A model and save English global & temporal figures."""
    ckpt = torch.load(ckpt_path, map_location=CONFIG.device, weights_only=False)
    s_mean, s_std = ckpt.get("scaler_mean"), ckpt.get("scaler_std")

    model = model_fn().to(CONFIG.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bg_ds = PointSequenceDataset(df_train["path"][:CONFIG.shap_bg_samples], df_train["label"][:CONFIG.shap_bg_samples], s_mean, s_std)
    te_ds = PointSequenceDataset(df_test["path"], df_test["label"], s_mean, s_std)

    bg_x = torch.stack([bg_ds[i][0] for i in range(len(bg_ds))]).to(CONFIG.device)
    te_x = torch.stack([te_ds[i][0] for i in range(len(te_ds))]).to(CONFIG.device)

    explainer = shap.GradientExplainer(model, bg_x)
    shap_vals = explainer.shap_values(te_x)
    # Handle both list-of-arrays (class 0, class 1) and 4D ndarray (N, T, F, C)
    if isinstance(shap_vals, list):
        sv = np.asarray(shap_vals[1])
    elif shap_vals.ndim == 4:
        sv = shap_vals[..., 1]
    else:
        sv = np.asarray(shap_vals)

    te_np = te_x.detach().cpu().numpy()
    groups = _get_shap_group_names()
    g_names = [g[0] for g in groups]
    sv_mat = np.stack([sv[:, :, g[1]].mean(axis=2).reshape(-1) for g in groups], axis=1)
    fv_mat = np.stack([
        (te_np[:, :, g[1]].mean(axis=2) if (g[1].stop - g[1].start) == 1
         else np.linalg.norm(te_np[:, :, g[1]], axis=2)).reshape(-1)
        for g in groups
    ], axis=1)

    group_rows = [
        {"Feature_Group": name, "Mean_Abs_SHAP": round(float(np.abs(sv_mat[:, i]).mean()), 6)}
        for i, name in enumerate(g_names)
    ]
    df_shap = pd.DataFrame(group_rows).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    CONFIG.tables_dir.mkdir(parents=True, exist_ok=True)
    df_shap.to_csv(CONFIG.tables_dir / "shap_feature_importance.csv", index=False)

    out_dir = CONFIG.figures_dir / "xai_shap"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Global Feature Group Importance (matched 1:1 with Violin Plot)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    ax.barh(df_shap["Feature_Group"], df_shap["Mean_Abs_SHAP"], color="#0072B2", alpha=0.9)
    ax.invert_yaxis()
    ax.set(
        xlabel="Mean |SHAP Value| (Impact on Drowsiness Prediction)",
        title=f"Global Feature Importance (SHAP) — {model_name}",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.savefig(out_dir / "shap_global_importance.png", dpi=180)
    plt.show()
    plt.close(fig)

    # 1b. SHAP Violin Summary Plot (Signed impact on Drowsy class)
    plt.figure(figsize=(9.5, 5.2))
    shap.summary_plot(sv_mat, fv_mat, feature_names=g_names, plot_type="violin", show=False)
    plt.title(f"SHAP Summary Plot (Impact on Drowsiness) — {model_name}", fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_violin_summary.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()

    # 2. Temporal SHAP Profile (1 Drowsy vs 1 Non-Drowsy)
    drowsy_idx = int(np.where(df_test["label"].values == 1)[0][0])
    nondrowsy_idx = int(np.where(df_test["label"].values == 0)[0][0])
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, layout="constrained")
    frames = np.arange(1, CONFIG.seq_len + 1)

    for ax, idx, label_str in [(axes[0], drowsy_idx, "Drowsy (1)"), (axes[1], nondrowsy_idx, "Non-Drowsy (0)")]:
        with open(df_test.iloc[idx]["path"], "rb") as f:
            loaded = pickle.load(f)
        raw = loaded["features"] if isinstance(loaded, dict) and "features" in loaded else np.asarray(loaded)
        temp_imp = np.abs(sv[idx]).mean(axis=1)
        temp_imp = (temp_imp - temp_imp.min()) / (temp_imp.max() - temp_imp.min() + 1e-6)
        ax.plot(frames, raw[:, 0], label="EAR (Eye Aspect Ratio)", color="#0072B2", lw=1.6)
        ax.plot(frames, raw[:, 1], label="MAR (Mouth Aspect Ratio)", color="#009E73", lw=1.6)
        ax.fill_between(frames, 0, temp_imp, color="#D55E00", alpha=0.28, label="Normalized Temporal |SHAP|")
        ax.set(ylabel="Ratio / Normalized |SHAP|", title=f"Temporal SHAP Profile — {df_test.iloc[idx]['filename']} ({label_str})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame Index (1..120)")
    fig.savefig(out_dir / "shap_temporal_profile.png", dpi=180)
    plt.show()
    plt.close(fig)
    return df_shap


@torch.no_grad()
def compute_topk_scorecam_3d(
    model: torch.nn.Module,
    x_clip: torch.Tensor,
    target_class: int = 1,
    top_k: int = CONFIG.scorecam_top_k,
    chunk_size: int = 8,
) -> tuple[np.ndarray, float]:
    """Gradient-free Top-K 3D Score-CAM returning (120, 112, 112) heatmap in [0, 1] and class prob."""
    model.eval()
    x_clip = x_clip.to(CONFIG.device)
    base_probs = torch.softmax(model(x_clip), dim=1)[0]
    target_prob = float(base_probs[target_class].item())

    fmap = model.forward_features(x_clip)  # (1, C, Tf, Hf, Wf)
    channel_scores = fmap.abs().mean(dim=(0, 2, 3, 4))
    k = min(top_k, fmap.size(1))
    top_indices = torch.topk(channel_scores, k=k).indices

    t_in, h_in, w_in = x_clip.shape[2], x_clip.shape[3], x_clip.shape[4]
    masks = []
    for ch in top_indices:
        m = fmap[:, ch:ch + 1, :, :, :]
        m_up = F.interpolate(m, size=(t_in, h_in, w_in), mode="trilinear", align_corners=False)
        m_min, m_max = m_up.min(), m_up.max()
        if (m_max - m_min) > 1e-6:
            m_up = (m_up - m_min) / (m_max - m_min)
        else:
            m_up = torch.zeros_like(m_up)
        masks.append(m_up)

    masks_tensor = torch.cat(masks, dim=0)  # (K, 1, T, H, W)
    weights = []
    for start in range(0, k, chunk_size):
        batch_masks = masks_tensor[start:start + chunk_size]
        masked_inputs = x_clip * batch_masks
        probs = torch.softmax(model(masked_inputs), dim=1)[:, target_class]
        weights.append(probs)
    weights_tensor = torch.cat(weights, dim=0).view(k, 1, 1, 1, 1)

    cam = F.relu((weights_tensor * masks_tensor).sum(dim=0)).squeeze(0).squeeze(0)  # (T, H, W)
    cam_np = cam.cpu().numpy()
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-6)
    return cam_np, target_prob


def visualize_scorecam_3d(model_fn, ckpt_path: str, model_name: str, df_test: pd.DataFrame) -> None:
    """Visualize Top-K 3D Score-CAM overlays on 6 key frames for Drowsy and Non-Drowsy samples."""
    ckpt = torch.load(ckpt_path, map_location=CONFIG.device, weights_only=False)
    model = model_fn().to(CONFIG.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = CONFIG.figures_dir / "xai_scorecam"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = PixelClipDataset(df_test["path"], df_test["label"], augment=False)
    sample_indices = [
        ("drowsy", int(np.where(df_test["label"].values == 1)[0][0])),
        ("nondrowsy", int(np.where(df_test["label"].values == 0)[0][0])),
    ]
    key_frames = np.linspace(0, CONFIG.seq_len - 1, 6, dtype=int)

    for tag, idx in sample_indices:
        x_tensor, label = ds[idx]
        raw_clip = np.load(df_test.iloc[idx]["path"])  # (120, 112, 112, 3) uint8
        cam, prob = compute_topk_scorecam_3d(model, x_tensor.unsqueeze(0), target_class=int(label))

        fig, axes = plt.subplots(1, 6, figsize=(15, 3.1), layout="constrained")
        for ax, f_idx in zip(axes, key_frames):
            img = raw_clip[f_idx]
            hm = cv2.applyColorMap(np.uint8(255 * cam[f_idx]), cv2.COLORMAP_JET)
            hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img, 0.58, hm, 0.42, 0)
            ax.imshow(overlay)
            ax.set_title(f"Frame {f_idx + 1}", fontsize=9)
            ax.axis("off")

        lbl_str = "Drowsy (1)" if int(label) == 1 else "Non-Drowsy (0)"
        fig.suptitle(
            f"3D Score-CAM ({model_name}) — {df_test.iloc[idx]['filename']} | "
            f"True: {lbl_str} | Target Prob: {prob*100:.1f}%",
            fontweight="bold",
        )
        fig.savefig(out_dir / f"{_safe_slug(model_name)}_scorecam_{tag}.png", dpi=180)
        plt.show()
        plt.close(fig)
