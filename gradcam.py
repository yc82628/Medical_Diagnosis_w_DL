"""
Grad-CAM for the DR model.

Read this before you put a heatmap in front of anyone:

**Grad-CAM is a debugging tool, not an explanation.** It shows which spatial
regions most influenced the score. It does not show why, it does not show what
feature was detected, and a plausible-looking map is not evidence the model is
reasoning correctly. Published work has repeatedly found saliency maps that look
convincing while being largely insensitive to the model's actual parameters.

Which is why `sanity_check` here is not optional decoration. It re-runs Grad-CAM
with progressively randomised weights. If the map barely changes, it is
responding to image structure rather than to anything the model learned, and it
must not be shown to a clinician. Run it once per trained model and record the
number.

What Grad-CAM IS good for: catching shortcut learning. If the map lights up on a
laterality marker, a lens artefact, a scanner watermark, or the black border
rather than on retinal tissue, you have found a real bug.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import PreprocessConfig, REFERABLE_THRESHOLD
from preprocessing import preprocess_image


class GradCAM:
    """
    Hooks the last convolutional block and weights its channels by the gradient
    of the target score.

    Targets the REFERABLE logit by default -- the cumulative threshold that is
    the actual clinical decision -- rather than an arbitrary class index. A
    heatmap for a quantity nobody acts on explains nothing useful.
    """

    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        self.layer = target_layer if target_layer is not None else model.target_layer()
        self._acts = None
        self._grads = None
        self._handles = [
            self.layer.register_forward_hook(self._save_acts),
            self.layer.register_full_backward_hook(self._save_grads),
        ]

    def _save_acts(self, _m, _i, out):
        self._acts = out.detach()

    def _save_grads(self, _m, _gi, gout):
        self._grads = gout[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, x: torch.Tensor, target_index: int | None = None) -> np.ndarray:
        """
        x: 1 x 3 x H x W, already preprocessed and normalised.
        Returns an H x W map in [0, 1].
        """
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(f"expected a single image of shape 1x3xHxW, got {tuple(x.shape)}")

        k = REFERABLE_THRESHOLD - 1 if target_index is None else target_index

        # No no_grad here: Grad-CAM needs the backward pass.
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[0, k]
        score.backward()

        if self._acts is None or self._grads is None:
            raise RuntimeError(
                "No activations captured. The target layer did not run during "
                "forward. Check model.target_layer()."
            )

        # Detect degenerate activations BEFORE producing a map. A silently flat
        # heatmap is the worst possible failure here: it renders as a uniform
        # overlay that looks like "the model sees nothing suspicious" rather
        # than "this computation is broken".
        #
        # The usual cause is an untrained backbone. EfficientNet in particular
        # collapses when randomly initialised -- its squeeze-excite gates sit at
        # ~0.5 and compound across stages, measured at roughly x0.03 per stage,
        # giving feature magnitudes near 1e-13 by the final block. That is a
        # property of the forward pass, not of Grad-CAM.
        act_scale = float(self._acts.abs().mean())
        if act_scale < 1e-8:
            raise RuntimeError(
                f"Activations at the target layer are degenerate "
                f"(mean magnitude {act_scale:.2e}). Grad-CAM cannot produce a "
                f"meaningful map.\n"
                f"  Most likely the backbone is untrained (pretrained=False) or "
                f"the checkpoint failed to load.\n"
                f"  Check:  model.features(x).abs().mean()  should be order 0.1-10."
            )

        # Channel weights = spatially averaged gradient.
        weights = self._grads.mean(dim=(2, 3), keepdim=True)      # 1 x C x 1 x 1
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))

        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        lo, hi = float(cam.min()), float(cam.max())
        # Relative, not absolute. Grad-CAM magnitudes depend on gradient scale,
        # which varies by orders of magnitude across architectures, so a fixed
        # epsilon silently misclassifies a valid small-magnitude map as flat.
        if hi - lo <= max(abs(hi), abs(lo), 1e-30) * 1e-6:
            raise RuntimeError(
                f"Grad-CAM produced a flat map (range {hi - lo:.2e}). Every "
                f"location contributed equally, which is not a usable "
                f"explanation. Usually an untrained or unconverged model."
            )
        return (cam - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def overlay(img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45,
            mask_outside_retina: bool = True) -> np.ndarray:
    """
    Heatmap over the fundus image.

    The retinal border is masked by default. Grad-CAM often puts weight on the
    black surround, which is meaningless and looks alarming to a reader.
    """
    h, w = img_bgr.shape[:2]
    cam = cv2.resize(cam.astype(np.float32), (w, h))

    if mask_outside_retina:
        m = np.zeros((h, w), np.float32)
        cv2.circle(m, (w // 2, h // 2), int(0.48 * min(h, w)), 1.0, -1)
        cam = cam * cv2.GaussianBlur(m, (0, 0), 9)

    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # Weight the blend by intensity so cold regions stay readable as retina.
    a = (alpha * np.clip(cam, 0, 1))[..., None]
    return np.clip(img_bgr * (1 - a) + heat * a, 0, 255).astype(np.uint8)


def cam_statistics(cam: np.ndarray) -> dict:
    """
    Numbers that catch shortcut learning without a human staring at every map.

    `border_mass` is the one to watch: attention on the black surround or the
    FOV rim means the model is keying on the camera, not the retina.
    """
    h, w = cam.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (0.5 * min(h, w))

    total = cam.sum() + 1e-9
    return {
        "peak": float(cam.max()),
        "mean": float(cam.mean()),
        "border_mass": float(cam[radius > 0.92].sum() / total),
        "central_mass": float(cam[radius < 0.35].sum() / total),
        "concentration": float((cam > 0.5).mean()),     # fraction strongly lit
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(model, x: torch.Tensor, target_index: int | None = None,
                 seed: int = 0) -> dict:
    """
    Cascading weight randomisation.

    Randomise the model's layers progressively and recompute the map. If the
    saliency is genuinely explaining the trained model, it should degrade toward
    noise. If it survives randomisation, it is an edge detector wearing a lab
    coat, and showing it to a clinician would be actively misleading.

    Returns correlations against the original map. Rough reading:
        < 0.3   good -- the map depends on what the model learned
        0.3-0.6 weak -- treat the map with suspicion
        > 0.6   FAIL -- do not display this map as an explanation
    """
    import copy

    try:
        with GradCAM(model) as cam_fn:
            base = cam_fn(x.clone(), target_index)
    except RuntimeError as e:
        return {"status": "degenerate", "note": str(e), "correlations": {}}

    results = {}
    randomised = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(seed)

    # Randomise from the output end backwards -- the standard cascade.
    named = [(n, m) for n, m in randomised.named_modules()
             if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    for name, module in reversed(named[-4:]):
        with torch.no_grad():
            for p in module.parameters(recurse=False):
                p.copy_(torch.empty_like(p).normal_(0, 0.05, generator=gen))
        try:
            with GradCAM(randomised) as cam_fn:
                m = cam_fn(x.clone(), target_index)
        except RuntimeError:
            # Randomisation destroyed the map entirely, which is the ideal
            # outcome: the saliency depended completely on the trained weights.
            results[name] = 0.0
            continue
        a, b = base.ravel(), m.ravel()
        corr = 0.0 if (a.std() < 1e-9 or b.std() < 1e-9) else float(np.corrcoef(a, b)[0, 1])
        results[name] = corr

    worst = max((abs(v) for v in results.values()), default=0.0)
    status = "pass" if worst < 0.3 else ("weak" if worst < 0.6 else "FAIL")
    return {"status": status, "max_abs_correlation": worst, "correlations": results}


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def explain(model, img_bgr: np.ndarray, pc: PreprocessConfig,
            device="cpu", target_index: int | None = None) -> dict:
    """Preprocess -> Grad-CAM -> overlay + statistics, from a raw BGR image."""
    proc = preprocess_image(img_bgr, pc)
    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(pc.norm_mean)) / np.array(pc.norm_std)
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with GradCAM(model) as cam_fn:
        cam = cam_fn(x, target_index)

    return {
        "cam": cam,
        "overlay": overlay(proc, cam),
        "processed": proc,
        "stats": cam_statistics(cam),
    }
