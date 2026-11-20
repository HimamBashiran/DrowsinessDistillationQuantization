"""Track A models: Point/Landmark sequence classifiers (RNN, CNN, LSTM, Transformer S4)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CONFIG


class PointRNN(nn.Module):
    def __init__(self, input_dim=CONFIG.point_input_dim, hidden_dim=32, num_layers=1,
                 num_classes=CONFIG.num_classes, dropout=CONFIG.dropout):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.drop(self.norm(out.mean(dim=1))))


class PointLSTM(nn.Module):
    def __init__(self, input_dim=CONFIG.point_input_dim, hidden_dim=32, num_layers=1,
                 num_classes=CONFIG.num_classes, dropout=CONFIG.dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(self.norm(out.mean(dim=1))))


class PointCNN1D(nn.Module):
    def __init__(self, input_dim=CONFIG.point_input_dim, hidden_dim=32,
                 num_classes=CONFIG.num_classes, dropout=CONFIG.dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        feat = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.fc(feat)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class FakeQuantizeSTE(torch.autograd.Function):
    """Symmetric INT8 fake quantization with Straight-Through Estimator."""

    @staticmethod
    def forward(ctx, x, num_bits=8):
        qmax = (1 << (num_bits - 1)) - 1
        scale = x.abs().max() / max(qmax, 1)
        if scale == 0:
            return x
        return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def fake_quantize(x, num_bits=8):
    return FakeQuantizeSTE.apply(x, num_bits)


class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, num_bits: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.num_bits = num_bits

    def forward(self, x):
        return F.linear(
            fake_quantize(x, self.num_bits),
            fake_quantize(self.linear.weight, self.num_bits),
            self.linear.bias,
        )


class PointTransformer(nn.Module):
    """Unified Transformer for Teacher, Student, and Student QAT (Scenario S4)."""

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, ffn_dim: int,
                 cls_hidden: int, input_dim: int = CONFIG.point_input_dim,
                 num_classes: int = CONFIG.num_classes, seq_len: int = CONFIG.seq_len,
                 dropout: float = CONFIG.dropout, use_qat: bool = False, num_bits: int = CONFIG.qat_bits):
        super().__init__()
        self.use_qat = use_qat
        self.num_bits = num_bits
        linear_cls = (lambda i, o: QuantizedLinear(i, o, num_bits=num_bits)) if use_qat else nn.Linear

        self.project = nn.Sequential(
            linear_cls(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_enc = SinusoidalPositionalEncoding(hidden_dim, max_len=seq_len + 10)
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            linear_cls(hidden_dim, cls_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            linear_cls(cls_hidden, num_classes),
        )

    def forward(self, x):
        bsz = x.size(0)
        x = self.pos_drop(self.pos_enc(self.project(x)))
        cls = self.cls_token.expand(bsz, -1, -1)
        if self.use_qat:
            cls = fake_quantize(cls, self.num_bits)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        pooled = 0.5 * x[:, 0, :] + 0.5 * x[:, 1:, :].mean(dim=1)
        normed = self.final_norm(pooled)
        if self.use_qat:
            normed = fake_quantize(normed, self.num_bits)
        return self.classifier(normed)


def build_point_model(model_name: str) -> nn.Module:
    """Factory for all 7 Track A models using exact PRD names."""
    t, s = CONFIG.teacher_cfg, CONFIG.student_cfg
    builders = {
        "RNN": lambda: PointRNN(),
        "CNN": lambda: PointCNN1D(),
        "LSTM": lambda: PointLSTM(),
        "TRANSFORMER TEACHER": lambda: PointTransformer(
            t["hidden"], t["layers"], t["heads"], t["ffn"], t["cls"], use_qat=False
        ),
        "TRANSFORMER STUDENT": lambda: PointTransformer(
            s["hidden"], s["layers"], s["heads"], s["ffn"], s["cls"], use_qat=False
        ),
        "TRANSFORMER KD": lambda: PointTransformer(
            s["hidden"], s["layers"], s["heads"], s["ffn"], s["cls"], use_qat=False
        ),
        "TRANSFORMER KD+QAT": lambda: PointTransformer(
            s["hidden"], s["layers"], s["heads"], s["ffn"], s["cls"], use_qat=True
        ),
    }
    if model_name not in builders:
        raise ValueError(f"Unknown Track A model: {model_name}. Valid: {list(builders)}")
    return builders[model_name]()
