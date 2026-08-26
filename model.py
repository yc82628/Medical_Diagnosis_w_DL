from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import PreprocessConfig, NUM_GRADES


# Backbones
_BACKBONES = {
    "resnet18":         (models.resnet18,        models.ResNet18_Weights.DEFAULT),
    "resnet50":         (models.resnet50,        models.ResNet50_Weights.DEFAULT),
    "efficientnet_b0":  (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    "efficientnet_b3":  (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
    "convnext_tiny":    (models.convnext_tiny,   models.ConvNeXt_Tiny_Weights.DEFAULT),
}


def _strip_classifier(name: str, net: nn.Module) -> tuple[nn.Module, int]:
    """Return (feature extractor producing B x C x H x W, channel count)."""
    if name.startswith("resnet"):
        dim = net.fc.in_features
        return nn.Sequential(*list(net.children())[:-2]), dim
    if name.startswith("efficientnet"):
        dim = net.classifier[-1].in_features
        return net.features, dim
    if name.startswith("convnext"):
        dim = net.classifier[-1].in_features
        return net.features, dim
    raise ValueError(f"Unsupported backbone: {name}")


# Ordinal head

class CoralHead(nn.Module):
    """
    CORAL-style cumulative-link head.

    ONE shared weight vector and K-1 independent biases. Sharing the weights is
    what makes the K-1 thresholds behave like thresholds on a common severity
    axis rather than K-1 unrelated binary classifiers -- it strongly encourages
    (though does not strictly guarantee) monotonic outputs. We still enforce
    monotonicity at decode time; see splits.enforce_monotonic.
    """

    def __init__(self, in_features: int, num_grades: int = NUM_GRADES, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_grades - 1))

    def forward(self, x):
        return self.fc(self.dropout(x)) + self.biases      # B x (K-1)


class DRModel(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b3", num_grades: int = NUM_GRADES,
                 dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}. Options: {list(_BACKBONES)}")
        ctor, weights = _BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)

        self.backbone_name = backbone
        self.features, feat_dim = _strip_classifier(backbone, net)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = CoralHead(feat_dim, num_grades, dropout)
        self.num_grades = num_grades

    def target_layer(self) -> nn.Module:
        """
        The layer Grad-CAM hooks. Exposed as a method so explanation code never
        has to reach into module internals with a string path that silently
        breaks when the backbone changes.
        """
        return self.features[-1] if hasattr(self.features, "__getitem__") else self.features

    def forward(self, x):
        fmap = self.features(x)                    # B x C x H x W
        pooled = self.pool(fmap).flatten(1)        # B x C
        return self.head(pooled)                   # B x (K-1) cumulative logits

    # -- decoding helpers -------------------------------------------------

    @staticmethod
    def cumulative_probs(logits: torch.Tensor) -> torch.Tensor:
        """P(grade > k) for each k."""
        return torch.sigmoid(logits)

    @staticmethod
    def referable_logit(logits: torch.Tensor) -> torch.Tensor:
        """
        Logit of P(grade >= 2) -- the screening decision, as a raw logit so it
        can be temperature-scaled directly. Index 1 is the k=1 threshold.
        """
        return logits[:, 1]


# Loss

class OrdinalLoss(nn.Module):
    """
    BCE over the K-1 cumulative targets, with per-threshold positive weighting.

    Referable DR is ~7-25% prevalent depending on the population, and grade 4
    is often under 2%. Without pos_weight the model learns the base rate and
    the sensitivity you actually care about never materialises.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None,
                 threshold_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        # Optionally upweight the k=1 threshold: it IS the clinical decision.
        self.register_buffer("threshold_weight", threshold_weight)

    def forward(self, logits, targets):
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none",
        )
        if self.threshold_weight is not None:
            loss = loss * self.threshold_weight
        return loss.mean()


# Checkpointing

def save_checkpoint(path, model: DRModel, preprocess: PreprocessConfig,
                    class_names=None, extra: dict | None = None):
    """
    The preprocessing config travels WITH the weights. At serving time the
    pipeline is rebuilt from the checkpoint, never from whatever config.py
    happens to say that day.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "backbone": model.backbone_name,
        "num_grades": model.num_grades,
        "preprocess": asdict(preprocess),
        "preprocess_fingerprint": preprocess.fingerprint(),
        "class_names": class_names,
        **(extra or {}),
    }, path)


def load_checkpoint(path, device="cpu", strict_preprocess: bool = True):
    """Returns (model, PreprocessConfig, payload)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = PreprocessConfig(**ck["preprocess"])

    if strict_preprocess and pc.fingerprint() != ck.get("preprocess_fingerprint"):
        raise RuntimeError(
            "Checkpoint preprocessing fingerprint does not match its own stored "
            "config. The checkpoint is corrupt or was written by a different "
            "version of config.py."
        )

    model = DRModel(ck["backbone"], ck["num_grades"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, pc, ck


def param_groups(model: DRModel, lr: float, backbone_mult: float = 0.1,
                 weight_decay: float = 1e-4):
    """Lower LR for pretrained features, full LR for the fresh head."""
    return [
        {"params": model.features.parameters(), "lr": lr * backbone_mult,
         "weight_decay": weight_decay},
        {"params": model.head.parameters(), "lr": lr, "weight_decay": weight_decay},
    ]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("  Apple Silicon GPU (MPS)")
    else:
        d = torch.device("cpu")
        print("  CPU -- 512px training will be impractically slow. Use Colab/cloud.")
    return d


def count_parameters(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable "
          f"({100*trainable/max(total,1):.1f}%)")
