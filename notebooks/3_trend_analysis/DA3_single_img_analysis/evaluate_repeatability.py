"""
=============================================================================
  DA3NESTED-GIANT-LARGE-1.1 – Depth Prediction Repeatability Evaluator
=============================================================================
  Purpose
  ───────
  Run DA3NESTED-GIANT-LARGE-1.1 on the SAME front-camera KITTI image
  N times (default 10) and quantify how much the predicted depth changes
  between runs.  SAM3 is used ONCE to produce the vehicle mask (the
  image never changes, so the mask is fixed).

  Key questions answered
  ──────────────────────
  • Does DA3 produce identical depth maps for the same input?
  • If not, how large is the variation (pixel std, RMSE, CV)?
  • Does variation concentrate on vehicles or background?
  • Do any standard depth metrics (AbsRel, RMSE, δ₁…) fluctuate?

  Install
  ───────
    pip install xformers "torch>=2" torchvision
    pip install -e /path/to/Depth-Anything-3
    pip install -e /path/to/sam3
    pip install opencv-python matplotlib numpy pillow

  Usage
  ─────
    # Full run with real models
    python evaluate_repeatability.py \
        --image  /data/kitti/image_02/data/0000000010.png \
        --depth  /data/kitti/proj_depth/groundtruth/image_02/0000000010.png \
        --n_runs 10 \
        --output_dir ./repeat_outputs
    python evaluate_repeatability.py \
        --image /STORAGE/Arka/Downloads/KITTI/kitti_eigen_split_dataset/test/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/0000002725.png \
        --depth /STORAGE/Arka/Downloads/KITTI/kitti_eigen_split_dataset/test/2011_10_03/2011_10_03_drive_0027_sync/proj_depth/groundtruth/image_02/0000002725.png \
        --n_runs 10 \
        --output_dir ./repeat_outputs

    # Dry-run demo (no models required)
    python evaluate_repeatability.py --demo --n_runs 10
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

# Pure numpy/matplotlib module (no torch)
from repeatability_core import (
    KITTI_MAX_DEPTH, KITTI_MIN_DEPTH, VEHICLE_PROMPTS, DEFAULT_N_RUNS,
    DepthMetrics, RunResult, RepeatabilityStudy,
    compute_metrics, build_study,
    generate_all_figures, print_repeatability_summary,
)

DEPTH_SCALE = 256.0   # KITTI uint16 → metres

# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────
def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def load_kitti_depth(path: str) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.float32) / DEPTH_SCALE

def align_to(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    h, w = ref.shape
    if pred.shape != ref.shape:
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    return pred

# ─────────────────────────────────────────────────────────────────────────────
# Lazy model singletons
# ─────────────────────────────────────────────────────────────────────────────
_MODELS: Dict = {}

def get_da3_model(device):
    if "da3" not in _MODELS:
        import torch
        from depth_anything_3.api import DepthAnything3   # type: ignore
        print("[DA3]  Loading DA3NESTED-GIANT-LARGE-1.1 …")
        model = DepthAnything3.from_pretrained(
            "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
        ).to(device=device).eval()
        _MODELS["da3"] = model
        print("[DA3]  Ready.")
    return _MODELS["da3"]

def get_sam3_processor(device):
    if "sam3" not in _MODELS:
        import torch
        from sam3.model_builder import build_sam3_image_model           # type: ignore
        from sam3.model.sam3_image_processor import Sam3Processor       # type: ignore
        print("[SAM3] Loading …")
        sam_model = build_sam3_image_model().to(device).eval()
        _MODELS["sam3"] = Sam3Processor(sam_model)
        print("[SAM3] Ready.")
    return _MODELS["sam3"]

# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def run_da3_once(image_path: str, device) -> np.ndarray:
    """Single forward pass of DA3NESTED → float32 [H,W] metres."""
    import torch
    model = get_da3_model(device)
    with torch.no_grad():
        pred = model.inference([image_path])
    depth = pred.depth[0]
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    return depth.astype(np.float32)

def get_vehicle_mask(rgb: np.ndarray, gt_shape, device) -> np.ndarray:
    """
    Run SAM3 ONCE and return a fixed bool vehicle mask [H,W].
    The mask is computed once and reused for all N runs.
    """
    from PIL import Image as PILImage
    import torch
    processor = get_sam3_processor(device)
    pil_img   = PILImage.fromarray(rgb)
    combined  = np.zeros(rgb.shape[:2], dtype=bool)

    for prompt in VEHICLE_PROMPTS:
        try:
            state  = processor.set_image(pil_img)
            output = processor.set_text_prompt(state=state, prompt=prompt)
            masks  = output.get("masks", None)
            if isinstance(masks, torch.Tensor):
                arr = masks.detach().cpu().numpy()  # safe conversion first
                masks_np = [arr[i, 0] for i in range(arr.shape[0])]  # then iterate
            scores = output.get("scores", None)
            for m, s in zip(masks, scores):
                if float(s) < 0.5:
                    continue
                if isinstance(m, torch.Tensor):
                    m = m.cpu().numpy()
                m = np.squeeze(m).astype(bool)
                if m.shape == rgb.shape[:2]:
                    combined |= m
        except Exception as e:
            print(f"  [SAM3] '{prompt}' skipped: {e}")

    # Resize to GT resolution
    combined = cv2.resize(
        combined.astype(np.uint8),
        (gt_shape[1], gt_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    n_veh = combined.sum()
    print(f"  [SAM3] Vehicle mask: {n_veh:,} pixels  "
          f"({n_veh / combined.size * 100:.1f}% of frame)")
    return combined

# ─────────────────────────────────────────────────────────────────────────────
# Main repeatability loop
# ─────────────────────────────────────────────────────────────────────────────
def run_repeatability_study(
    image_path: str,
    depth_path: str,
    n_runs:     int,
    out_dir:    str,
    device,
) -> RepeatabilityStudy:
    """
    Run DA3 N times on the same image, collect all depth maps,
    compute metrics per run and return a complete RepeatabilityStudy.
    """
    print(f"\n  Image : {image_path}")
    print(f"  Depth : {depth_path}")
    print(f"  Runs  : {n_runs}")

    # ── Load data once ──────────────────────────────────────────────────────
    rgb    = load_rgb(image_path)
    gt_dep = load_kitti_depth(depth_path)
    print(f"  Frame : {rgb.shape[1]} × {rgb.shape[0]} px  "
          f"|  GT depth shape: {gt_dep.shape}")

    # ── SAM3 vehicle mask (computed ONCE, fixed for all runs) ───────────────
    print("\n  [Step 1/2]  Computing SAM3 vehicle mask (once) …")
    vehicle_mask = get_vehicle_mask(rgb, gt_dep.shape, device)

    # ── DA3 repeated inference ──────────────────────────────────────────────
    print(f"\n  [Step 2/2]  Running DA3 {n_runs} times on the same image …")
    run_results: List[RunResult] = []
    run_times:   List[float]     = []

    for i in range(n_runs):
        t0    = time.perf_counter()
        pred  = run_da3_once(image_path, device)
        pred  = align_to(pred, gt_dep)
        t_ms  = (time.perf_counter() - t0) * 1000
        run_times.append(t_ms)

        m_all = compute_metrics(pred, gt_dep)
        m_veh = compute_metrics(pred, gt_dep, vehicle_mask)
        m_non = compute_metrics(pred, gt_dep, ~vehicle_mask)

        run_results.append(RunResult(
            run_idx=i, pred_depth=pred,
            metrics_all=m_all, metrics_veh=m_veh, metrics_non=m_non,
        ))

        print(f"    Run {i+1:>2}/{n_runs}  {t_ms:>7.1f}ms  "
              f"AbsRel(all)={m_all.abs_rel:.4f}  "
              f"RMSE(all)={m_all.rmse:.4f}m  "
              f"δ₁(all)={m_all.delta1:.4f}  "
              f"| RMSE(veh)={m_veh.rmse:.4f}m  "
              f"RMSE(non)={m_non.rmse:.4f}m")

    print(f"\n  Avg inference time: {np.mean(run_times):.1f} ms  "
          f"(±{np.std(run_times):.1f} ms)")

    # ── Assemble and return study ────────────────────────────────────────────
    study = RepeatabilityStudy(
        image_path   = image_path,
        gt_depth     = gt_dep,
        rgb_image    = rgb,
        vehicle_mask = vehicle_mask,
        runs         = run_results,
    )
    return build_study(study)


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode (no models — synthetic depth maps with controlled noise)
# ─────────────────────────────────────────────────────────────────────────────
def run_demo(n_runs: int, out_dir: str):
    """
    Dry-run without any models.
    Generates synthetic depth maps where the 'noise' between runs simulates
    real DA3 non-determinism (dropout, GPU floating-point ordering, etc.).
    """
    print("\n[Demo] Running without models (synthetic data) …")
    H, W  = 375, 1242
    rng   = np.random.default_rng(0)

    # Synthetic ground truth: road receding with distance
    y, x  = np.mgrid[0:H, 0:W]
    cam_fy, cam_cy, cam_h = 721.5, 172.85, 1.65
    gt_raw = np.where(
        y > H * 0.40,
        np.clip(cam_h * cam_fy / np.maximum(y - cam_cy, 1), 0.5, 70),
        50.0,
    ).astype(np.float32)
    # Sparse GT (simulate LiDAR)
    gt_dep = gt_raw.copy()
    gt_dep[rng.random((H, W)) > 0.30] = 0.0

    # Synthetic RGB
    rgb = rng.integers(60, 200, (H, W, 3), dtype=np.uint8)

    # Synthetic vehicle mask
    vehicle_mask = np.zeros((H, W), dtype=bool)
    for _ in range(rng.integers(1, 4)):
        r0 = rng.integers(H//2, H*3//4); c0 = rng.integers(W//6, W//2)
        vehicle_mask[r0:r0+80, c0:c0+160] = True

    # Simulate DA3 non-determinism:
    #   • A small per-run global scale jitter (simulates floating-point order variance)
    #   • Per-pixel additive noise with spatial correlation
    #   • Occasional "mode shift" in a small region (simulates attention map variation)
    print(f"  Generating {n_runs} synthetic depth maps with simulated jitter …")
    run_results: List[RunResult] = []

    for i in range(n_runs):
        # Base: near-perfect prediction with small global scale jitter
        global_jitter = rng.normal(0, 0.003)           # ~0.3% scale std
        pred = gt_raw * (1.0 + global_jitter)

        # Spatially correlated additive noise (simulate feature-level variation)
        noise_coarse = rng.normal(0, 0.15, (H//8, W//8))
        noise_coarse = cv2.resize(noise_coarse.astype(np.float32), (W, H),
                                  interpolation=cv2.INTER_LINEAR)
        pred = pred + noise_coarse

        # Occasional local mode shift (1–2 patches per run)
        n_patches = rng.integers(1, 3)
        for _ in range(n_patches):
            pr = rng.integers(0, H - 40); pc = rng.integers(0, W - 80)
            ph = rng.integers(20, 40);    pw = rng.integers(40, 80)
            shift = rng.uniform(-1.5, 1.5)
            pred[pr:pr+ph, pc:pc+pw] += shift

        pred = np.clip(pred, KITTI_MIN_DEPTH, KITTI_MAX_DEPTH).astype(np.float32)

        m_all = compute_metrics(pred, gt_dep)
        m_veh = compute_metrics(pred, gt_dep, vehicle_mask)
        m_non = compute_metrics(pred, gt_dep, ~vehicle_mask)

        run_results.append(RunResult(
            run_idx=i, pred_depth=pred,
            metrics_all=m_all, metrics_veh=m_veh, metrics_non=m_non,
        ))

        print(f"  Run {i+1:>2}/{n_runs}  "
              f"AbsRel(all)={m_all.abs_rel:.4f}  RMSE={m_all.rmse:.4f}m  "
              f"δ₁={m_all.delta1:.4f}")

    study = RepeatabilityStudy(
        image_path   = "synthetic_kitti_frame.png",
        gt_depth     = gt_dep,
        rgb_image    = rgb,
        vehicle_mask = vehicle_mask,
        runs         = run_results,
    )
    return build_study(study)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description="DA3NESTED Depth Prediction Repeatability Study on a Single KITTI Image",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--image",      type=str, default=None,
                   help="Path to KITTI front-camera RGB image (.png)")
    p.add_argument("--depth",      type=str, default=None,
                   help="Path to KITTI LiDAR depth PNG (uint16, /256 → metres)")
    p.add_argument("--n_runs",     type=int, default=DEFAULT_N_RUNS,
                   help=f"Number of times to run DA3 on the same image (default: {DEFAULT_N_RUNS})")
    p.add_argument("--output_dir", type=str, default="repeat_outputs",
                   help="Directory for output figures and summary")
    p.add_argument("--device",     type=str, default="cuda",
                   help="Torch device: 'cuda' or 'cpu'")
    p.add_argument("--demo",       action="store_true",
                   help="Run with synthetic data (no models required)")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"

    if args.demo:
        study = run_demo(n_runs=args.n_runs, out_dir=str(out_dir))
    else:
        if not args.image or not args.depth:
            print("[ERROR] Please provide --image and --depth (or use --demo)")
            sys.exit(1)
        if not os.path.exists(args.image):
            print(f"[ERROR] Image not found: {args.image}"); sys.exit(1)
        if not os.path.exists(args.depth):
            print(f"[ERROR] Depth not found: {args.depth}"); sys.exit(1)

        import torch
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"\n  Device: {device}")

        study = run_repeatability_study(
            image_path = args.image,
            depth_path = args.depth,
            n_runs     = args.n_runs,
            out_dir    = str(out_dir),
            device     = device,
        )

    # ── Output ──────────────────────────────────────────────────────────────
    print_repeatability_summary(study)
    generate_all_figures(study, str(fig_dir))
    print(f"\n  All outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
