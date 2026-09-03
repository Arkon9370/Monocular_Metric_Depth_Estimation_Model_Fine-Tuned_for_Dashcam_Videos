"""
kitti_dataset.py
────────────────
Standalone module for the KITTI Eigen-split Dataset loader.

WHY THIS FILE EXISTS
--------------------
DataLoader uses multiprocessing with start_method="spawn".  Spawned worker
processes start a *fresh* Python interpreter whose __main__ is the frozen
built-in importer — they cannot see anything defined inside a Jupyter cell.
Moving the Dataset class here lets workers do `from kitti_dataset import ...`
and successfully unpickle the dataset object.

USAGE (in the notebook, replace Cell 3 body with):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # if needed
    from kitti_dataset import KITTIEigenDepthDataset, collate_fn
"""
from __future__ import annotations

import glob
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

# Module-level logger — workers get their own copy; that's fine.
log = logging.getLogger("da3_kitti.dataset")

# ── ImageNet normalisation (same constants as notebook) ──────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_train_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# ── Default hyperparameter values (must match notebook config) ────────────────
_DEFAULT_TARGET_H = 378
_DEFAULT_TARGET_W = 1246
_DEFAULT_MIN_DEPTH = 1e-3
_DEFAULT_MAX_DEPTH = 80.0


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _read_kitti_depth(path: str) -> np.ndarray:
    """Load 16-bit KITTI depth PNG → float32 metres (0 = invalid)."""
    raw = np.array(Image.open(path), dtype=np.uint16)
    return raw.astype(np.float32) / 256.0


def _parse_kitti_calib(date_dir: Path, drive_dir: Path) -> Optional[np.ndarray]:
    """
    Parse KITTI calib_cam_to_cam.txt → 3×3 K matrix for cam2.
    Returns None if calibration file is missing.
    """
    calib: Dict[str, np.ndarray] = {}
    for fname in ["calib_cam_to_cam.txt"]:
        for search in [date_dir, drive_dir]:
            p = search / fname
            if p.exists():
                for line in p.read_text().splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        try:
                            calib[k.strip()] = np.array(v.split(), dtype=np.float64)
                        except Exception:
                            pass
    K = None
    try:
        if "P_rect_02" in calib:
            P = calib["P_rect_02"].reshape(3, 4)
            K = P[:3, :3].copy()
        elif "P2" in calib:
            P = calib["P2"].reshape(3, 4)
            K = P[:3, :3].copy()
    except Exception:
        pass
    return K

# Change _augment signature to also accept and return valid:
def _augment(
    rgb: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,          # ← add this
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:   # ← returns valid too
    """Random horizontal flip + mild colour jitter."""
    if rng.rand() < 0.5:
        rgb   = np.fliplr(rgb).copy()
        depth = np.fliplr(depth).copy()
        valid = np.fliplr(valid).copy()            # ← flip valid too
    f  = rgb.astype(np.float32) / 255.0
    br = rng.uniform(0.8, 1.2)
    co = rng.uniform(0.8, 1.2)
    f  = np.clip((f - f.mean()) * co + f.mean() * br, 0.0, 1.0)
    return (f * 255.0).astype(np.uint8), depth, valid   # ← return valid

# def _augment(
#     rgb: np.ndarray,
#     depth: np.ndarray,
#     rng: np.random.RandomState,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Random horizontal flip + mild colour jitter."""
#     if rng.rand() < 0.5:
#         rgb   = np.fliplr(rgb).copy()
#         depth = np.fliplr(depth).copy()
#     f  = rgb.astype(np.float32) / 255.0
#     br = rng.uniform(0.8, 1.2)
#     co = rng.uniform(0.8, 1.2)
#     f  = np.clip((f - f.mean()) * co + f.mean() * br, 0.0, 1.0)
#     return (f * 255.0).astype(np.uint8), depth


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class KITTIEigenDepthDataset(Dataset):
    """
    Loads KITTI Eigen-Split frames directly from pre-projected depth PNGs.
    No velodyne processing needed — depth maps are ready to use.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_hw: Tuple[int, int] = (_DEFAULT_TARGET_H, _DEFAULT_TARGET_W),
        min_depth: float = _DEFAULT_MIN_DEPTH,
        max_depth: float = _DEFAULT_MAX_DEPTH,
        augment: bool = True,
        max_samples: int = 0,
        seed: int = 42,
    ):
        self.root      = Path(root)
        self.split     = split
        self.target_hw = target_hw
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.augment   = augment and (split == "train")
        self.rng       = np.random.RandomState(seed)
        self._index: List[Dict] = []
        self._build_index()

        if max_samples > 0 and len(self._index) > max_samples:
            idx = self.rng.choice(len(self._index), max_samples, replace=False)
            self._index = [self._index[i] for i in sorted(idx)]

        log.info("[KITTI-%s]  %d samples  (target %dx%d)",
                 split, len(self._index), target_hw[1], target_hw[0])

    # ─── Index builder ────────────────────────────────────────────────────

    def _build_index(self):
        """
        Walk the split root and collect (rgb_path, depth_path, K) triples.
        Expected layout:
          <root>/<date>/<drive>/image_02/data/*.png
          <root>/<date>/<drive>/proj_depth/groundtruth/image_02/*.png
        """
        depth_glob = str(self.root / "*" / "*_sync" /
                         "proj_depth" / "groundtruth" / "image_02" / "*.png")
        depth_paths = sorted(glob.glob(depth_glob))

        # Also try without the *_sync pattern (some layouts differ)
        if not depth_paths:
            depth_glob = str(self.root / "*" / "*" /
                             "proj_depth" / "groundtruth" / "image_02" / "*.png")
            depth_paths = sorted(glob.glob(depth_glob))

        for depth_path in depth_paths:
            parts = Path(depth_path).parts
            try:
                gt_idx    = [p for p in parts].index("proj_depth")
                drive_dir = Path(*parts[:gt_idx])
                date_dir  = drive_dir.parent
                fname     = parts[-1]
            except (ValueError, IndexError):
                continue

            rgb_path = drive_dir / "image_02" / "data" / fname
            if not rgb_path.exists():
                continue

            K = _parse_kitti_calib(date_dir, drive_dir)

            self._index.append(dict(
                rgb_path   = str(rgb_path),
                depth_path = depth_path,
                K          = K,
                sequence   = drive_dir.name,
                frame_id   = Path(fname).stem,
                date       = date_dir.name,
            ))

    # ─── __getitem__ ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        rec      = self._index[idx]
        rgb      = cv2.cvtColor(cv2.imread(rec["rgb_path"]), cv2.COLOR_BGR2RGB)
        depth_gt = _read_kitti_depth(rec["depth_path"])
        orig_hw  = rgb.shape[:2]

        # Resize
        H, W = self.target_hw
        # rgb      = cv2.resize(rgb,      (W, H), interpolation=cv2.INTER_LINEAR)
        # depth_gt = cv2.resize(depth_gt, (W, H), interpolation=cv2.INTER_NEAREST)

        # # Augment
        # if self.augment:
        #     rgb, depth_gt = _augment(rgb, depth_gt, self.rng)

        # depth_gt = np.clip(depth_gt, 0.0, self.max_depth)
        # valid    = (depth_gt > self.min_depth) & (depth_gt < self.max_depth)

        # ── Fix New-A: compute valid mask at ORIGINAL resolution before resize ──
        valid_orig = (depth_gt > self.min_depth) & (depth_gt < self.max_depth)

        rgb      = cv2.resize(rgb,      (W, H), interpolation=cv2.INTER_LINEAR)
        depth_gt = cv2.resize(depth_gt, (W, H), interpolation=cv2.INTER_NEAREST)

        # Remap original valid mask (binary) to resized resolution with INTER_NEAREST
        # so valid pixels track actual LiDAR hit locations before the resize shift
        valid = cv2.resize(
            valid_orig.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # ── Fix ①: add Eigen crop to valid mask ──────────────────────────────────
        eigen_crop = np.zeros((H, W), dtype=bool)
        eigen_crop[44:min(44 + 352, H), :] = True
        valid = valid & eigen_crop

        # ── Fix New-B: per-worker RNG seeding ────────────────────────────────────
        worker_info = torch.utils.data.get_worker_info()
        if self.augment:
            if worker_info is not None:
                rng = np.random.RandomState((worker_info.seed + idx) % (2**31))
            else:
                rng = self.rng
            rgb, depth_gt, valid = _augment(rgb, depth_gt, valid, rng)  # ← pass+receive valid
        # if self.augment:
        #     if worker_info is not None:
        #         rng = np.random.RandomState((worker_info.seed + idx) % (2**31))
        #     else:
        #         rng = self.rng
        #     rgb, depth_gt = _augment(rgb, depth_gt, rng)

        depth_gt = np.clip(depth_gt, 0.0, self.max_depth)

        # Rescale intrinsics
        K = rec["K"]
        if K is not None:
            K = K.copy()
            K[0] *= W / orig_hw[1]
            K[1] *= H / orig_hw[0]

        # To tensors
        rgb_t   = _train_transform(Image.fromarray(rgb))          # [3,H,W]
        depth_t = torch.from_numpy(depth_gt).unsqueeze(0).float() # [1,H,W]
        valid_t = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)

        K_t   = torch.from_numpy(K).float() if K is not None else torch.zeros(3, 3)
        has_K = torch.tensor(K is not None, dtype=torch.bool)

        return {
            "rgb":        rgb_t,
            "depth_gt":   depth_t,
            "valid_mask": valid_t,
            "K":          K_t,
            "has_K":      has_K,
            "rgb_path":   rec["rgb_path"],
            "depth_path": rec["depth_path"],
            "sequence":   rec["sequence"],
            "frame_id":   rec["frame_id"],
            "date":       rec["date"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """Collate with string passthrough."""
    from torch.utils.data.dataloader import default_collate
    tensor_keys = [k for k in batch[0] if isinstance(batch[0][k], torch.Tensor)]
    str_keys    = [k for k in batch[0] if isinstance(batch[0][k], str)]
    out = {k: default_collate([b[k] for b in batch]) for k in tensor_keys}
    out.update({k: [b[k] for b in batch] for k in str_keys})
    return out
