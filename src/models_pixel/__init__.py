"""Track B 3D CNN model factory, Kinetics weight loader, and Partial Fine-Tuning."""

from pathlib import Path
import torch
import torch.nn as nn

from src.config import CONFIG
from src.models_pixel.resnet import resnet18
from src.models_pixel.wide_resnet import wideresnet50
from src.models_pixel.resnext import resnext101
from src.models_pixel.densenet import densenet121


KINETICS_CKPT_MAP = {
    "3D ResNet": "resnet-18-kinetics.pth",
    "3D WideResNet": "wideresnet-50-kinetics.pth",
    "3D ResNeXt": "resnext-101-kinetics.pth",
}


def load_kinetics_weights(model: nn.Module, ckpt_path: Path) -> None:
    """Load Kinetics-400 weights cleanly, stripping DataParallel 'module.' and skipping classifier head."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    model_state = model.state_dict()
    for k, v in state.items():
        name = k[7:] if k.startswith("module.") else k
        for old, new in ((".norm.1.", ".norm1."), (".relu.1.", ".relu1."), (".conv.1.", ".conv1."),
                         (".norm.2.", ".norm2."), (".relu.2.", ".relu2."), (".conv.2.", ".conv2.")):
            name = name.replace(old, new)
        if name.startswith(("fc.", "classifier.")):
            continue
        if name in model_state and model_state[name].shape == v.shape:
            cleaned[name] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[PRETRAINED] Loaded {len(cleaned)}/{len(model_state)} tensors from {ckpt_path.name} "
          f"(head/missing={len(missing)}, skipped={len(unexpected)})")


def apply_finetune_mode(model: nn.Module, model_name: str, mode: str) -> None:
    """Freeze early layers for 'last_block' partial fine-tuning or unfreeze all for 'full'."""
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return
    if mode != "last_block":
        raise ValueError(f"Unsupported finetune mode: {mode}")

    for p in model.parameters():
        p.requires_grad = False

    trainable_prefixes = (
        ("features.denseblock4.", "features.norm5.", "classifier.")
        if model_name == "3D DenseNet"
        else ("layer4.", "fc.")
    )
    for name, p in model.named_parameters():
        if name.startswith(trainable_prefixes):
            p.requires_grad = True


def build_pixel_model(
    model_name: str,
    finetune_mode: str = CONFIG.pixel_finetune_mode,
    load_pretrained: bool = True,
) -> nn.Module:
    """Factory for all 4 Track B 3D CNN models."""
    kwargs = dict(
        sample_size=CONFIG.sample_size,
        sample_duration=CONFIG.seq_len,
        num_classes=CONFIG.num_classes,
    )
    builders = {
        "3D ResNet": lambda: resnet18(shortcut_type="A", **kwargs),
        "3D WideResNet": lambda: wideresnet50(k=2, shortcut_type="B", **kwargs),
        "3D ResNeXt": lambda: resnext101(cardinality=32, shortcut_type="B", **kwargs),
        "3D DenseNet": lambda: densenet121(**kwargs),
    }
    if model_name not in builders:
        raise ValueError(f"Unknown Track B model: {model_name}. Valid: {list(builders)}")

    model = builders[model_name]()
    effective_mode = finetune_mode
    if load_pretrained:
        ckpt_path = CONFIG.kinetics_dir / KINETICS_CKPT_MAP[model_name]
        if ckpt_path.exists():
            load_kinetics_weights(model, ckpt_path)
        else:
            print(f"[WARNING] Pretrained weights not found at {ckpt_path}. Switching to full training.")
            effective_mode = "full"

    apply_finetune_mode(model, model_name, effective_mode)
    return model
