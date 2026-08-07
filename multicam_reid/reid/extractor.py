"""
Vehicle ReID feature extractor.

Wraps the IBN-ResNet backbone with GeM pooling + a BNNeck and L2 normalization
to turn a vehicle crop into a compact appearance embedding. This mirrors the
AI City / fast-reid "Bag of Tricks" (BoT) inference head so we can load
*vehicle-ReID-trained* weights (VeRi / VeRi-Wild / VehicleID), which are vastly
more discriminative for cross-camera vehicle matching than ImageNet weights.

Two weight formats are supported automatically:
  * fast-reid checkpoints  (keys prefixed 'backbone.' + 'heads.*') -- preferred,
    because they are trained on real vehicle-ReID data. Any extra non-local
    ('NL_*') keys are transparently ignored since the plain backbone omits them.
  * plain IBN-Net ImageNet checkpoints (bare 'conv1'/'layer1'... keys) -- used
    only as a fallback; produces weak features and is not recommended.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .ibn_resnet import resnet50_ibn_a, load_backbone_weights

_CACHE = Path.home() / ".cache" / "multicam_reid"

# Vehicle-ReID weights (fast-reid BoT, ResNet50-IBN-a) trained on VeRi-Wild --
# a large, camera-diverse vehicle dataset -> best cross-camera generalization.
VEHICLE_WEIGHTS = _CACHE / "veriwild_bot_R50-ibn.pth"
VEHICLE_WEIGHTS_URL = (
    "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/"
    "veriwild_bot_R50-ibn.pth"
)

# ImageNet fallback (weak; not vehicle-specific).
IMAGENET_WEIGHTS = _CACHE / "resnet50_ibn_a-d9d0bb7b.pth"

# Default fast-reid preprocessing (0-255 scale, RGB). Overridden by the
# pixel_mean / pixel_std stored inside the checkpoint when present.
_FASTREID_MEAN = [123.675, 116.28, 103.53]
_FASTREID_STD = [58.395, 57.12, 57.225]


class GeM(nn.Module):
    """Generalized-Mean pooling with a (loadable) power parameter p."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (1, 1)
        ).pow(1.0 / self.p)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Downloading ReID weights -> {dest}")
    torch.hub.download_url_to_file(url, str(dest))
    return dest


class ReIDExtractor:
    """Turns BGR vehicle crops into L2-normalized appearance embeddings."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        image_size: tuple[int, int] = (256, 256),
        device: str | None = None,
        batch_size: int = 64,
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Resolve which checkpoint to use.
        if weights_path is None:
            if not VEHICLE_WEIGHTS.exists():
                _download(VEHICLE_WEIGHTS_URL, VEHICLE_WEIGHTS)
            weights_path = VEHICLE_WEIGHTS
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"ReID weights not found: {weights_path}")

        self.backbone = resnet50_ibn_a(last_stride=1)
        self.pool = GeM()
        self.bnneck: nn.Module = nn.BatchNorm1d(self.feature_dim)
        self._mean = torch.tensor(_FASTREID_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(_FASTREID_STD).view(1, 3, 1, 1)

        kind = self._load_weights(weights_path)
        logger.info(f"  ReID extractor ready ({kind}, {Path(weights_path).name})")

        self.backbone.eval().to(self.device)
        self.pool.eval().to(self.device)
        self.bnneck.eval().to(self.device)
        self._mean = self._mean.to(self.device)
        self._std = self._std.to(self.device)

    @property
    def feature_dim(self) -> int:
        return 2048

    # ------------------------------------------------------------------ #
    def _load_weights(self, path: Path) -> str:
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        state = {k.replace("module.", ""): v for k, v in state.items()}

        is_fastreid = any(k.startswith("backbone.") for k in state)
        if not is_fastreid:
            # ImageNet fallback: load backbone only, default GeM(p=3), no BNNeck.
            loaded, total = load_backbone_weights(self.backbone, str(path))
            self.bnneck = nn.Identity()
            logger.info(f"  Loaded {loaded}/{total} backbone tensors (ImageNet fallback, weak)")
            return "ImageNet-IBN fallback"

        # fast-reid vehicle-ReID checkpoint.
        # 1) pixel mean/std (already in 0-255 scale, RGB).
        if "pixel_mean" in state:
            self._mean = state["pixel_mean"].view(1, 3, 1, 1).clone()
        if "pixel_std" in state:
            self._std = state["pixel_std"].view(1, 3, 1, 1).clone()

        # 2) backbone (strip prefix, keep only keys the plain backbone has;
        #    this transparently drops any extra non-local NL_* keys).
        bb_state = self.backbone.state_dict()
        loaded = 0
        for key, value in state.items():
            if not key.startswith("backbone."):
                continue
            bkey = key[len("backbone."):]
            if bkey in bb_state and bb_state[bkey].shape == value.shape:
                bb_state[bkey] = value
                loaded += 1
        self.backbone.load_state_dict(bb_state)

        # 3) GeM power.
        if "heads.pool_layer.p" in state:
            self.pool.p.data.copy_(state["heads.pool_layer.p"].view(-1)[:1])

        # 4) BNNeck (fast-reid stores it as heads.bottleneck.0.*).
        bn_map = {
            "weight": "heads.bottleneck.0.weight",
            "bias": "heads.bottleneck.0.bias",
            "running_mean": "heads.bottleneck.0.running_mean",
            "running_var": "heads.bottleneck.0.running_var",
        }
        if all(v in state for v in bn_map.values()):
            bn = self.bnneck
            assert isinstance(bn, nn.BatchNorm1d)
            bn.weight.data.copy_(state[bn_map["weight"]])
            bn.bias.data.copy_(state[bn_map["bias"]])
            bn.running_mean.data.copy_(state[bn_map["running_mean"]])
            bn.running_var.data.copy_(state[bn_map["running_var"]])
        else:
            logger.warning("  BNNeck stats missing in checkpoint; using identity BNNeck")
            self.bnneck = nn.Identity()

        logger.info(
            f"  Loaded {loaded} backbone tensors + GeM(p={float(self.pool.p):.2f}) + BNNeck "
            f"(vehicle-ReID weights)"
        )
        return "vehicle-ReID (fast-reid BoT)"

    # ------------------------------------------------------------------ #
    def _preprocess(self, crops: list[np.ndarray]) -> torch.Tensor:
        """BGR uint8 crops -> normalized NCHW float tensor on device (0-255 scale)."""
        import cv2

        h, w = self.image_size
        batch = np.empty((len(crops), h, w, 3), dtype=np.float32)
        for i, crop in enumerate(crops):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            batch[i] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2)
        tensor = (tensor - self._mean) / self._std
        return tensor

    @torch.no_grad()
    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract (N, 2048) L2-normalized embeddings for a list of BGR crops."""
        if not crops:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        features: list[np.ndarray] = []
        for start in range(0, len(crops), self.batch_size):
            chunk = crops[start:start + self.batch_size]
            x = self._preprocess(chunk)
            pooled = self.pool(self.backbone(x)).flatten(1)
            feat = self.bnneck(pooled)
            feat = F.normalize(feat, dim=1)
            features.append(feat.cpu().numpy().astype(np.float32))
        return np.concatenate(features, axis=0)
