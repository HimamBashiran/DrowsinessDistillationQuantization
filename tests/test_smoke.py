"""Fast smoke checks for Track A, Track B, GFLOPs profiler, Score-CAM, and SHAP."""

import sys
from pathlib import Path
import numpy as np
import shap
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import profile_model
from src.config import CONFIG
from src.models_pixel import build_pixel_model
from src.models_point import build_point_model
from src.xai import compute_topk_scorecam_3d


def test_track_a_models():
    x = torch.randn(2, CONFIG.seq_len, CONFIG.point_input_dim)
    names = [
        "RNN", "CNN", "LSTM",
        "TRANSFORMER TEACHER", "TRANSFORMER STUDENT", "TRANSFORMER KD", "TRANSFORMER KD+QAT",
    ]
    for name in names:
        m = build_point_model(name)
        out = m(x)
        assert out.shape == (2, CONFIG.num_classes), f"{name} bad shape: {out.shape}"
        eff = profile_model(m, (1, CONFIG.seq_len, CONFIG.point_input_dim), name,
                            is_int8_qat=("QAT" in name), num_runs=3, warmup_runs=1)
        assert eff["Total_Params"] > 0 and eff["GFLOPs"] > 0, f"{name} bad profile: {eff}"
    print("[PASS] Track A all 7 models + profiler")


def test_track_b_models():
    x = torch.randn(1, 3, CONFIG.seq_len, CONFIG.sample_size, CONFIG.sample_size)
    for name in ["3D ResNet", "3D WideResNet", "3D ResNeXt", "3D DenseNet"]:
        m = build_pixel_model(name, finetune_mode="last_block", load_pretrained=False)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        assert 0 < trainable < total, f"{name} partial finetune failed: {trainable}/{total}"
        out = m(x)
        assert out.shape == (1, CONFIG.num_classes), f"{name} bad shape: {out.shape}"
        fmap = m.forward_features(x)
        assert fmap.ndim == 5, f"{name} bad fmap ndim: {fmap.ndim}"
    print("[PASS] Track B all 4 3D CNN models + partial fine-tune")


def test_xai_smoke():
    # SHAP on QAT Student
    m_pt = build_point_model("TRANSFORMER KD+QAT").eval()
    bg = torch.randn(4, CONFIG.seq_len, CONFIG.point_input_dim)
    te = torch.randn(2, CONFIG.seq_len, CONFIG.point_input_dim)
    sv = shap.GradientExplainer(m_pt, bg).shap_values(te)
    assert np.asarray(sv).size > 0, "SHAP returned empty"

    # Top-K 3D Score-CAM on 3D ResNet (k=2 for fast smoke check)
    m_px = build_pixel_model("3D ResNet", load_pretrained=False).eval()
    clip = torch.randn(1, 3, CONFIG.seq_len, CONFIG.sample_size, CONFIG.sample_size)
    cam, prob = compute_topk_scorecam_3d(m_px, clip, target_class=1, top_k=2, chunk_size=2)
    assert cam.shape == (CONFIG.seq_len, CONFIG.sample_size, CONFIG.sample_size), f"Bad CAM shape: {cam.shape}"
    assert 0.0 <= prob <= 1.0, f"Bad prob: {prob}"
    print("[PASS] XAI smoke (SHAP + Top-K 3D Score-CAM)")


if __name__ == "__main__":
    test_track_a_models()
    test_track_b_models()
    test_xai_smoke()
    print("\nALL SMOKE CHECKS PASSED!")
