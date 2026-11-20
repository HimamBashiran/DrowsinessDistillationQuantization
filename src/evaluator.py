"""Hold-out Test evaluation, per-model JSON persistence, and comparison tables/charts."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.datasets import PixelClipDataset, PointSequenceDataset
from src.trainer import _safe_slug


def evaluate_holdout_ensemble(model_fn, cv_result: dict, df_test: pd.DataFrame, track: str) -> dict:
    """Evaluate 5-fold soft-voting ensemble on the Hold-out Test Set (Subjects 16-20)."""
    batch_size = CONFIG.batch_size if track == "A" else CONFIG.pixel_batch_size
    all_probs = []
    y_true = None

    for fold_row in cv_result["fold_metrics"]:
        ckpt = torch.load(fold_row["ckpt_path"], map_location=CONFIG.device, weights_only=False)
        if track == "A":
            ds = PointSequenceDataset(
                df_test["path"], df_test["label"],
                scaler_mean=ckpt.get("scaler_mean"), scaler_std=ckpt.get("scaler_std"), augment=False
            )
        else:
            ds = PixelClipDataset(df_test["path"], df_test["label"], augment=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        model = model_fn().to(CONFIG.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        fold_probs, labels = [], []
        with torch.no_grad():
            for x_b, y_b in loader:
                logits = model(x_b.to(CONFIG.device))
                fold_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                labels.extend(y_b.numpy())
        all_probs.append(np.concatenate(fold_probs, axis=0))
        y_true = np.array(labels)

    avg_probs = np.mean(all_probs, axis=0)
    y_pred = np.argmax(avg_probs, axis=1)

    cv_result["test_acc"] = float(accuracy_score(y_true, y_pred))
    cv_result["test_prec"] = float(precision_score(y_true, y_pred, zero_division=0))
    cv_result["test_rec"] = float(recall_score(y_true, y_pred, zero_division=0))
    cv_result["test_f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    print(
        f"  [Hold-Out Test] {cv_result['model']} -> "
        f"Acc: {cv_result['test_acc']*100:.2f}% | Prec: {cv_result['test_prec']*100:.2f}% | "
        f"Rec: {cv_result['test_rec']*100:.2f}% | F1: {cv_result['test_f1']*100:.2f}%"
    )
    return cv_result


def save_per_model_result(res: dict, eff: dict) -> Path:
    """Persist a single model's CV, Hold-out, and efficiency metrics to JSON."""
    CONFIG.per_model_dir.mkdir(parents=True, exist_ok=True)
    payload = {**res, **eff}
    out_path = CONFIG.per_model_dir / f"{_safe_slug(res['model'])}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def compile_all_saved_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all per-model JSONs from disk and build summary + per-fold CSV tables."""
    json_files = sorted(CONFIG.per_model_dir.glob("*.json"))
    if not json_files:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows, fold_rows = [], []
    for jp in json_files:
        with open(jp, encoding="utf-8") as f:
            d = json.load(f)
        summary_rows.append({
            "Track": d["track"],
            "Model": d["model"],
            "Total_Params": d["Total_Params"],
            "Trainable_Params": d["Trainable_Params"],
            "GFLOPs": d["GFLOPs"],
            "Size_KB": d["Size_KB"],
            "Latency_ms": d["Latency_ms"],
            "CV_Val_Loss": round(d["cv_val_loss"], 4),
            "CV_Acc (%)": round(d["cv_acc_mean"] * 100, 2),
            "CV_Acc_SD (%)": round(d["cv_acc_std"] * 100, 2),
            "CV_F1 (%)": round(d["cv_f1_mean"] * 100, 2),
            "CV_F1_SD (%)": round(d["cv_f1_std"] * 100, 2),
            "Test_Acc (%)": round(d["test_acc"] * 100, 2),
            "Test_Prec (%)": round(d["test_prec"] * 100, 2),
            "Test_Rec (%)": round(d["test_rec"] * 100, 2),
            "Test_F1 (%)": round(d["test_f1"] * 100, 2),
        })
        for fm in d["fold_metrics"]:
            fold_rows.append({
                "Track": d["track"],
                "Model": d["model"],
                "Fold": fm["fold"],
                "Val_Loss": round(fm["val_loss"], 4),
                "Acc (%)": round(fm["accuracy"] * 100, 2),
                "Prec (%)": round(fm["precision"] * 100, 2),
                "Rec (%)": round(fm["recall"] * 100, 2),
                "F1 (%)": round(fm["f1"] * 100, 2),
            })

    df_summary = pd.DataFrame(summary_rows)
    df_folds = pd.DataFrame(fold_rows)
    CONFIG.tables_dir.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(CONFIG.tables_dir / "summary_comparison.csv", index=False)
    df_folds.to_csv(CONFIG.tables_dir / "cv_fold_details.csv", index=False)
    return df_summary, df_folds


def plot_overall_comparison(df_summary: pd.DataFrame) -> None:
    """Generate English publication-ready comparison charts (F1 & GFLOPs trade-off)."""
    if df_summary.empty:
        return
    comp_dir = CONFIG.figures_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # 1. F1-Score: 5-Fold CV vs Hold-out Test
    fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(df_summary))), layout="constrained")
    y = np.arange(len(df_summary))
    h = 0.38
    ax.barh(y - h / 2, df_summary["CV_F1 (%)"], xerr=df_summary["CV_F1_SD (%)"],
            height=h, label="5-Fold CV F1 (Mean ± SD)", color="#0072B2", alpha=0.9, capsize=3)
    ax.barh(y + h / 2, df_summary["Test_F1 (%)"],
            height=h, label="Hold-out Test F1", color="#009E73", alpha=0.9)
    ax.set_yticks(y, labels=df_summary["Model"])
    ax.invert_yaxis()
    ax.set(xlim=(0, 105), xlabel="F1-Score (%)",
           title="Drowsiness Detection Performance: Landmark (Track A) vs. 3D CNN Pixel (Track B)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.savefig(comp_dir / "f1_cv_vs_test_all_models.png", dpi=200)
    plt.show()
    plt.close(fig)

    # 2. Efficiency Trade-off: GFLOPs (log scale) vs Hold-out Test F1
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for track, marker, color in [("Track A", "o", "#0072B2"), ("Track B", "s", "#D55E00")]:
        sub = df_summary[df_summary["Track"] == track]
        if sub.empty:
            continue
        ax.scatter(
            sub["GFLOPs"].clip(lower=1e-6), sub["Test_F1 (%)"],
            s=90, marker=marker, color=color, edgecolors="black", label=track, zorder=3,
        )
        for _, r in sub.iterrows():
            ax.annotate(r["Model"], (max(r["GFLOPs"], 1e-6), r["Test_F1 (%)"]),
                        xytext=(6, 4), textcoords="offset points", fontsize=8.5)
    ax.set_xscale("log")
    ax.set(xlabel="Computational Cost (GFLOPs, log scale)", ylabel="Hold-out Test F1-Score (%)",
           ylim=(0, 105), title="Accuracy vs. Computational Cost Trade-Off (Telup Dataset)")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.savefig(comp_dir / "efficiency_tradeoff_gflops_vs_f1.png", dpi=200)
    plt.show()
    plt.close(fig)
