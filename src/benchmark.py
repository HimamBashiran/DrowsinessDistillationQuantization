"""Model efficiency benchmarking: Parameters, GFLOPs, Memory Size, and CPU Latency."""

import time
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from src.config import CONFIG


def _estimate_rnn_lstm_flops(model: nn.Module, dummy_input: torch.Tensor) -> int:
    """Analytical FLOPs (2 * MACs) for fused nn.RNN/nn.LSTM ops not tracked by FlopCounterMode."""
    total_flops = 0
    hooks = []

    def make_hook(mod):
        def hook(_m, inp, _out):
            nonlocal total_flops
            x = inp[0]
            bsz, seq_len, _ = x.shape if mod.batch_first else (x.shape[1], x.shape[0], x.shape[2])
            num_dirs = 2 if mod.bidirectional else 1
            gate_mult = 4 if isinstance(mod, nn.LSTM) else 1
            for layer in range(mod.num_layers):
                in_dim = mod.input_size if layer == 0 else mod.hidden_size * num_dirs
                macs_per_step = gate_mult * (in_dim * mod.hidden_size + mod.hidden_size * mod.hidden_size)
                total_flops += 2 * bsz * seq_len * num_dirs * macs_per_step
        return hook

    for m in model.modules():
        if isinstance(m, (nn.RNN, nn.LSTM)):
            hooks.append(m.register_forward_hook(make_hook(m)))

    if hooks:
        with torch.no_grad():
            _ = model(dummy_input)
        for h in hooks:
            h.remove()
    return total_flops


def profile_model(
    model: nn.Module,
    input_shape: tuple[int, ...],
    model_name: str,
    is_int8_qat: bool = False,
    num_runs: int = 50,
    warmup_runs: int = 10,
) -> dict:
    """Measure Total/Trainable Params, GFLOPs, Size (KB), and Inference Latency (ms)."""
    model = model.to(CONFIG.device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_per_param = 1 if is_int8_qat else 4
    size_kb = (total_params * bytes_per_param) / 1024.0

    dummy = torch.randn(*input_shape, device=CONFIG.device)
    with torch.no_grad():
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            _ = model(dummy)
        total_flops = flop_counter.get_total_flops() + _estimate_rnn_lstm_flops(model, dummy)
        gflops = total_flops / 1e9

        for _ in range(warmup_runs):
            _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0

    return {
        "Model": model_name,
        "Total_Params": int(total_params),
        "Trainable_Params": int(trainable_params),
        "GFLOPs": round(float(gflops), 6),
        "Size_KB": round(float(size_kb), 2),
        "Latency_ms": round(float(latency_ms), 3),
    }
