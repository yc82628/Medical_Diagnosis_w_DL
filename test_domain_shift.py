"""
Training augmentation and test-time augmentation used against the camera shift
between DeepDRiD and IDRiD (MODEL_CARD.md, "Domain-shift experiments").
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("albumentations")

from config import PreprocessConfig  # noqa: E402
from dataset import build_transforms, FieldOfViewBandCrop  # noqa: E402
from model import TTA_VIEWS, tta_logits  # noqa: E402

PC = PreprocessConfig(image_size=128)


def _fundus(size=128):
    """Grey disc on black: the shape a preprocessed cache image has."""
    yy, xx = np.mgrid[:size, :size]
    inside = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (0.45 * size) ** 2
    img = np.zeros((size, size, 3), np.uint8)
    img[inside] = (120, 128, 135)
    return img, inside


@pytest.mark.parametrize("augment", ["mild", "strong"])
def test_train_transform_shapes(augment):
    img, _ = _fundus()
    x = build_transforms("train", PC, augment)(image=img)["image"]
    assert tuple(x.shape) == (3, 128, 128)


def test_unknown_augmentation_is_refused():
    with pytest.raises(ValueError):
        build_transforms("train", PC, "extreme")


def test_strong_augmentation_keeps_the_background_black():
    """Grey backgrounds never occur in real preprocessed images: a shortcut cue."""
    img, inside = _fundus()
    t = build_transforms("train", PC, "strong")
    norm_black = (0 - 0.5 * 255) / (0.5 * 255)        # Normalize(0.5, 0.5) of 0
    for _ in range(20):
        x = t(image=img)["image"].numpy()
        corners = x[:, :6, :6]
        assert np.allclose(corners, norm_black, atol=1e-4)


def test_band_crop_blacks_out_top_and_bottom_only():
    img, _ = _fundus()
    full = np.full_like(img, 100)
    out = FieldOfViewBandCrop(max_frac=0.2, rim=False, p=1.0)(image=full)["image"]
    assert out[0].max() == 0 and out[-1].max() == 0
    assert out[64].min() == 100


class _Spy(torch.nn.Module):
    """Logit = mean of the top-left quarter: orientation-dependent on purpose."""

    def forward(self, x):
        q = x[:, :, : x.shape[2] // 2, : x.shape[3] // 2].mean(dim=(1, 2, 3))
        return torch.stack([q, q, q, q], dim=1)


def test_tta_none_is_the_plain_model():
    x = torch.rand(2, 3, 16, 16)
    assert torch.allclose(tta_logits(_Spy(), x, "none"), _Spy()(x))


def test_d8_tta_averages_all_quarters():
    """Over the 8 dihedral views each quarter is the top-left one twice."""
    x = torch.rand(2, 3, 16, 16)
    got = tta_logits(_Spy(), x, "d8")[:, 0]
    assert torch.allclose(got, x.mean(dim=(1, 2, 3)), atol=1e-6)
    assert len(TTA_VIEWS["d8"]) == 8 and len(TTA_VIEWS["flip4"]) == 4


def test_unknown_tta_is_refused():
    with pytest.raises(ValueError):
        tta_logits(_Spy(), torch.rand(1, 3, 8, 8), "d16")
