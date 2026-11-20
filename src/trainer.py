"""Shared 5-Fold GroupKFold trainer for Track A (Point) and Track B (Pixel 3D CNN)."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import CONFIG, seed_everything
from src.datasets import PixelClipDataset, PointSequenceDataset, compute_point_scaler


class DistillationLoss(nn.Module):
    """Hard-label CrossEntropy (with smoothing) + Soft-label KLDivLoss."""

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5, label_smoothing: float = 0.10):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.hard_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.soft_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        loss_hard = self.hard_loss(student_logits, labels)
        s_log_probs = F.log_softmax(student_logits / self.T, dim=1)
        t_probs = F.softmax(teacher_logits / self.T, dim=1)
        loss_soft = self.soft_loss(s_log_probs, t_probs) * (self.T ** 2)
        return self.alpha * loss_hard + (1.0 - self.alpha) * loss_soft


def _build_optimizer_and_scheduler(model: nn.Module, lr: float, weight_decay: float, epochs: int, warmup_epochs: int):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )
    warmup = min(warmup_epochs, max(1, epochs - 1))
    sched = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup),
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=CONFIG.min_lr),
        ],
        milestones=[warmup],
    )
    return optimizer, sched


def _safe_slug(name: str) -> str:
    return name.replace(" ", "_").replace("+", "_")


def _train_one_fold(
    fold: int,
    track: str,
    model_name: str,
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ckpt_dir: Path,
    teacher_model: nn.Module | None = None,
):
    is_point = track == "A"
    epochs = CONFIG.epochs if is_point else CONFIG.pixel_epochs
    batch_size = CONFIG.batch_size if is_point else CONFIG.pixel_batch_size
    lr = CONFIG.lr if is_point else CONFIG.pixel_lr
    wd = CONFIG.weight_decay if is_point else CONFIG.pixel_weight_decay
    warmup = CONFIG.warmup_epochs if is_point else CONFIG.pixel_warmup_epochs
    patience = CONFIG.patience if is_point else CONFIG.pixel_patience

    s_mean, s_std = (None, None)
    if is_point:
        s_mean, s_std = compute_point_scaler(train_df["path"].values)
        train_ds = PointSequenceDataset(train_df["path"], train_df["label"], s_mean, s_std, augment=True)
        val_ds = PointSequenceDataset(val_df["path"], val_df["label"], s_mean, s_std, augment=False)
    else:
        train_ds = PixelClipDataset(train_df["path"], train_df["label"], augment=True)
        val_ds = PixelClipDataset(val_df["path"], val_df["label"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model.to(CONFIG.device)
    if teacher_model is not None:
        teacher_model = teacher_model.to(CONFIG.device).eval()
        train_crit = DistillationLoss(CONFIG.kd_temperature, CONFIG.kd_alpha, CONFIG.label_smoothing)
    else:
        train_crit = nn.CrossEntropyLoss(label_smoothing=CONFIG.label_smoothing)
    val_crit = nn.CrossEntropyLoss(label_smoothing=CONFIG.label_smoothing)

    optimizer, scheduler = _build_optimizer_and_scheduler(model, lr, wd, epochs, warmup)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_val_acc = float("inf"), 0.0
    best_preds, best_targets = [], []
    patience_cnt = 0

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{_safe_slug(model_name)}_fold{fold}_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(CONFIG.device), y_b.to(CONFIG.device)
            optimizer.zero_grad()
            logits = model(x_b)
            if teacher_model is not None:
                with torch.no_grad():
                    t_logits = teacher_model(x_b)
                loss = train_crit(logits, t_logits, y_b)
                with torch.no_grad():
                    hard_val = val_crit(logits, y_b).item()
            else:
                loss = train_crit(logits, y_b)
                hard_val = loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip)
            optimizer.step()

            t_loss += hard_val * y_b.size(0)
            t_correct += (logits.argmax(dim=1) == y_b).sum().item()
            t_total += y_b.size(0)
        scheduler.step()

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        f_preds, f_targets = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(CONFIG.device), y_b.to(CONFIG.device)
                logits = model(x_b)
                v_loss += val_crit(logits, y_b).item() * y_b.size(0)
                preds = logits.argmax(dim=1)
                v_correct += (preds == y_b).sum().item()
                v_total += y_b.size(0)
                f_preds.extend(preds.cpu().numpy())
                f_targets.extend(y_b.cpu().numpy())

        ep_tl, ep_vl = t_loss / max(1, t_total), v_loss / max(1, v_total)
        ep_ta, ep_va = t_correct / max(1, t_total), v_correct / max(1, v_total)
        history["train_loss"].append(ep_tl)
        history["val_loss"].append(ep_vl)
        history["train_acc"].append(ep_ta)
        history["val_acc"].append(ep_va)

        improved = (ep_vl < best_val_loss - 1e-4) or (abs(ep_vl - best_val_loss) <= 1e-4 and ep_va >= best_val_acc)
        if improved:
            best_val_loss, best_val_acc = ep_vl, ep_va
            best_preds, best_targets = f_preds.copy(), f_targets.copy()
            patience_cnt = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_mean": s_mean,
                "scaler_std": s_std,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "epoch": epoch,
            }, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                for _ in range(epochs - epoch):
                    history["train_loss"].append(ep_tl)
                    history["val_loss"].append(best_val_loss)
                    history["train_acc"].append(ep_ta)
                    history["val_acc"].append(best_val_acc)
                break

    metrics = {
        "fold": fold,
        "val_loss": float(best_val_loss),
        "accuracy": float(accuracy_score(best_targets, best_preds)),
        "precision": float(precision_score(best_targets, best_preds, zero_division=0)),
        "recall": float(recall_score(best_targets, best_preds, zero_division=0)),
        "f1": float(f1_score(best_targets, best_preds, zero_division=0)),
        "ckpt_path": str(ckpt_path),
    }
    return metrics, history, best_targets, best_preds


def _plot_cv_curves(model_name: str, histories: list[dict], all_targets: list, all_preds: list, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    epochs_range = np.arange(1, len(histories[0]["train_loss"]) + 1)

    tl = np.array([h["train_loss"] for h in histories])
    vl = np.array([h["val_loss"] for h in histories])
    ta = np.array([h["train_acc"] for h in histories])
    va = np.array([h["val_acc"] for h in histories])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")

    for arr, label, color, ls in [
        (tl, "Train Loss (Mean ± SD)", "#0072B2", "-"),
        (vl, "Validation Loss (Mean ± SD)", "#D55E00", "--"),
    ]:
        m, s = arr.mean(axis=0), arr.std(axis=0)
        axes[0].plot(epochs_range, m, label=label, color=color, linestyle=ls, lw=2)
        axes[0].fill_between(epochs_range, m - s, m + s, color=color, alpha=0.15)
    axes[0].set(title=f"{model_name} — 5-Fold Loss Curve", xlabel="Epoch", ylabel="Cross-Entropy Loss")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    for arr, label, color, ls in [
        (ta, "Train Accuracy (Mean ± SD)", "#009E73", "-"),
        (va, "Validation Accuracy (Mean ± SD)", "#E69F00", "--"),
    ]:
        m, s = arr.mean(axis=0), arr.std(axis=0)
        axes[1].plot(epochs_range, m, label=label, color=color, linestyle=ls, lw=2)
        axes[1].fill_between(epochs_range, m - s, m + s, color=color, alpha=0.15)
    axes[1].set(title=f"{model_name} — 5-Fold Accuracy Curve", xlabel="Epoch", ylabel="Accuracy", ylim=(0.0, 1.05))
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[2],
        xticklabels=["Non-Drowsy (0)", "Drowsy (1)"],
        yticklabels=["Non-Drowsy (0)", "Drowsy (1)"],
    )
    axes[2].set(title=f"{model_name} — CV Confusion Matrix", xlabel="Predicted Label", ylabel="True Label")

    out_path = fig_dir / f"{_safe_slug(model_name)}_cv_curves.png"
    fig.savefig(out_path, dpi=180)
    plt.show()
    plt.close(fig)


def run_cv_training(
    model_fn,
    model_name: str,
    track: str,
    df_train_pool: pd.DataFrame,
    teacher_fn=None,
    teacher_name: str | None = None,
    show_plot: bool = True,
) -> dict:
    """Run 5-Fold GroupKFold CV for either Track A or Track B."""
    ckpt_dir = CONFIG.ckpt_track_a_dir if track == "A" else CONFIG.ckpt_track_b_dir
    fig_dir = CONFIG.figures_dir / "training_curves"
    gkf = GroupKFold(n_splits=CONFIG.folds)

    fold_metrics, histories, all_targets, all_preds = [], [], [], []
    print(f"\n{'=' * 75}")
    print(f"  5-FOLD GROUP-CV [Track {track}] : {model_name}")
    print(f"{'=' * 75}")

    for fold, (tr_idx, val_idx) in enumerate(
        gkf.split(df_train_pool, df_train_pool["label"], groups=df_train_pool["subject"]), start=1
    ):
        seed_everything(CONFIG.seed + fold)
        tr_df = df_train_pool.iloc[tr_idx].reset_index(drop=True)
        val_df = df_train_pool.iloc[val_idx].reset_index(drop=True)

        model = model_fn()
        teacher = None
        if teacher_fn is not None and teacher_name is not None:
            teacher = teacher_fn()
            t_ckpt = ckpt_dir / f"{_safe_slug(teacher_name)}_fold{fold}_best.pth"
            if not t_ckpt.exists():
                raise FileNotFoundError(
                    f"Teacher checkpoint not found: {t_ckpt}. Train '{teacher_name}' first."
                )
            ckpt = torch.load(t_ckpt, map_location=CONFIG.device, weights_only=False)
            teacher.load_state_dict(ckpt["model_state_dict"])

        m, h, f_t, f_p = _train_one_fold(fold, track, model_name, model, tr_df, val_df, ckpt_dir, teacher)
        fold_metrics.append(m)
        histories.append(h)
        all_targets.extend(f_t)
        all_preds.extend(f_p)
        print(f"  Fold {fold} -> Val Loss: {m['val_loss']:.4f} | Acc: {m['accuracy']*100:.2f}% | F1: {m['f1']*100:.2f}%")

    df_m = pd.DataFrame(fold_metrics)
    print(
        f"\n  [CV Summary] {model_name} | "
        f"Acc: {df_m['accuracy'].mean()*100:.2f}% ± {df_m['accuracy'].std()*100:.2f}% | "
        f"F1: {df_m['f1'].mean()*100:.2f}% ± {df_m['f1'].std()*100:.2f}%"
    )

    if show_plot:
        _plot_cv_curves(model_name, histories, all_targets, all_preds, fig_dir)

    return {
        "track": f"Track {track}",
        "model": model_name,
        "fold_metrics": fold_metrics,
        "cv_val_loss": float(df_m["val_loss"].mean()),
        "cv_acc_mean": float(df_m["accuracy"].mean()),
        "cv_acc_std": float(df_m["accuracy"].std()),
        "cv_prec_mean": float(df_m["precision"].mean()),
        "cv_rec_mean": float(df_m["recall"].mean()),
        "cv_f1_mean": float(df_m["f1"].mean()),
        "cv_f1_std": float(df_m["f1"].std()),
    }
