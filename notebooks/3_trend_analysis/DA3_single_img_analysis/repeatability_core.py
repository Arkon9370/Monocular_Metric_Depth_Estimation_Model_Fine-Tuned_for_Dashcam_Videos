"""
repeatability_core.py
─────────────────────
Pure NumPy / Matplotlib module — no PyTorch dependency.

Data structures
───────────────
  RunResult          one inference run on the same image
  RepeatabilityStudy collection of N runs + all derived statistics

Metrics (per run, vs LiDAR GT)
───────────────────────────────
  AbsRel, SqRel, RMSE, RMSE-log, δ₁ δ₂ δ₃   (standard KITTI set)

Repeatability / variance metrics (across runs)
───────────────────────────────────────────────
  pixel_std_map      per-pixel std of predicted depth across N runs
  pixel_mean_map     per-pixel mean
  pixel_cv_map       coefficient of variation  (std / mean)
  max_pixel_diff_map max − min across runs, per pixel
  run_mean_depth     scalar per run
  run_metrics        DepthMetrics per run, per region

Figures
───────
  Fig 1  per-run metric lines  (ALL / VEH / NON)  — 7 subplots
  Fig 2  depth map gallery     (mean ± std per region visualised)
  Fig 3  pixel-wise std heatmap + histogram
  Fig 4  max-pixel-difference map (worst-case instability)
  Fig 5  run-to-run diff maps  (depth[run_k] − depth[run_1])
  Fig 6  scatter consistency   (run k vs run 1, coloured by region)
  Fig 7  metric stability band (mean ± std across runs, per metric)
  Fig 8  coefficient of variation map + regional CV box-plots
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
KITTI_MAX_DEPTH: float = 80.0
KITTI_MIN_DEPTH: float = 1e-3
VEHICLE_PROMPTS        = ["car", "truck", "bus", "van", "vehicle"]
DEFAULT_N_RUNS: int    = 10

_REGIONS     = ["all", "veh", "non"]
_LABELS_REG  = {"all": "All Pixels", "veh": "Vehicle",  "non": "Non-Vehicle"}
_COLORS_REG  = {"all": "#2196F3",    "veh": "#E91E63",  "non": "#4CAF50"}
_METRIC_KEYS = ["abs_rel", "sq_rel", "rmse", "rmse_log",
                "delta1",  "delta2", "delta3"]
_METRIC_LBLS = {
    "abs_rel":  "AbsRel ↓",
    "sq_rel":   "SqRel ↓",
    "rmse":     "RMSE (m) ↓",
    "rmse_log": "RMSE-log ↓",
    "delta1":   "δ₁ ↑",
    "delta2":   "δ₂ ↑",
    "delta3":   "δ₃ ↑",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
})

# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DepthMetrics:
    abs_rel:  float = 0.0
    sq_rel:   float = 0.0
    rmse:     float = 0.0
    rmse_log: float = 0.0
    delta1:   float = 0.0
    delta2:   float = 0.0
    delta3:   float = 0.0
    n_pixels: int   = 0

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class RunResult:
    """Everything produced by a single inference run."""
    run_idx:      int              # 0-based
    pred_depth:   np.ndarray      # [H, W] float32 metres
    metrics_all:  DepthMetrics
    metrics_veh:  DepthMetrics
    metrics_non:  DepthMetrics


@dataclass
class RepeatabilityStudy:
    """Aggregated analysis across N runs on the same image."""
    image_path:   str
    gt_depth:     np.ndarray      # [H, W] float32 metres
    rgb_image:    np.ndarray      # [H, W, 3] uint8
    vehicle_mask: np.ndarray      # [H, W] bool
    runs:         List[RunResult]

    # ── Derived (populated by build_study) ──────────────────────────────
    depth_stack:      np.ndarray = field(default=None)  # [N, H, W]
    pixel_mean_map:   np.ndarray = field(default=None)  # [H, W]
    pixel_std_map:    np.ndarray = field(default=None)  # [H, W]
    pixel_cv_map:     np.ndarray = field(default=None)  # [H, W]
    max_pixel_diff:   np.ndarray = field(default=None)  # [H, W]
    run_mean_depths:  np.ndarray = field(default=None)  # [N]
    run_mean_depths_veh: np.ndarray = field(default=None)
    run_mean_depths_non: np.ndarray = field(default=None)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(pred: np.ndarray,
                    gt:   np.ndarray,
                    mask: Optional[np.ndarray] = None) -> DepthMetrics:
    valid = (gt > KITTI_MIN_DEPTH) & (gt <= KITTI_MAX_DEPTH)
    if mask is not None:
        valid = valid & mask
    if valid.sum() < 10:
        return DepthMetrics()

    p = np.clip(pred[valid].astype(np.float64), KITTI_MIN_DEPTH, KITTI_MAX_DEPTH)
    g = gt[valid].astype(np.float64)

    thresh   = np.maximum(p / g, g / p)
    diff     = p - g
    return DepthMetrics(
        abs_rel  = float((np.abs(diff) / g).mean()),
        sq_rel   = float(((diff**2)    / g).mean()),
        rmse     = float(np.sqrt((diff**2).mean())),
        rmse_log = float(np.sqrt(((np.log(p) - np.log(g))**2).mean())),
        delta1   = float((thresh < 1.25   ).mean()),
        delta2   = float((thresh < 1.25**2).mean()),
        delta3   = float((thresh < 1.25**3).mean()),
        n_pixels = int(valid.sum()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Study builder
# ─────────────────────────────────────────────────────────────────────────────
def build_study(study: RepeatabilityStudy) -> RepeatabilityStudy:
    """Compute all derived pixel-level statistics from the run stack."""
    N = len(study.runs)
    stack = np.stack([r.pred_depth for r in study.runs], axis=0)  # [N,H,W]
    study.depth_stack    = stack
    study.pixel_mean_map = stack.mean(axis=0)
    study.pixel_std_map  = stack.std(axis=0)
    study.max_pixel_diff = stack.max(axis=0) - stack.min(axis=0)

    # Coefficient of variation (avoid /0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = study.pixel_std_map / (study.pixel_mean_map + 1e-6)
    study.pixel_cv_map = np.clip(cv, 0, 10)

    # Per-run scalar mean depths per region
    valid_gt  = (study.gt_depth > KITTI_MIN_DEPTH) & (study.gt_depth <= KITTI_MAX_DEPTH)
    veh_mask  = study.vehicle_mask & valid_gt
    non_mask  = ~study.vehicle_mask & valid_gt

    def _mean_depth(pred, mask):
        return float(pred[mask].mean()) if mask.sum() > 0 else float("nan")

    study.run_mean_depths     = np.array([_mean_depth(r.pred_depth, valid_gt)  for r in study.runs])
    study.run_mean_depths_veh = np.array([_mean_depth(r.pred_depth, veh_mask)  for r in study.runs])
    study.run_mean_depths_non = np.array([_mean_depth(r.pred_depth, non_mask)  for r in study.runs])
    return study


# ─────────────────────────────────────────────────────────────────────────────
# Print console summary
# ─────────────────────────────────────────────────────────────────────────────
def print_repeatability_summary(study: RepeatabilityStudy):
    N  = len(study.runs)
    sep = "─" * 100

    print(f"\n{sep}")
    print(f"  DA3NESTED-GIANT-LARGE-1.1  ·  Repeatability Study")
    print(f"  Image  : {study.image_path}")
    print(f"  Runs   : {N}   |   GT depth capped at {KITTI_MAX_DEPTH} m")
    print(sep)

    # Per-run metrics
    header = (f"  {'Run':>4}  {'AbsRel':>8}  {'SqRel':>8}  {'RMSE':>8}  "
              f"{'RMSElog':>8}  {'δ₁':>8}  {'δ₂':>8}  {'δ₃':>8}  {'MeanD':>8}")
    print(f"\n  ── Per-Run Metrics (ALL pixels) ──")
    print(header); print("  " + "·" * 98)
    for r in study.runs:
        m = r.metrics_all
        md = study.run_mean_depths[r.run_idx] if study.run_mean_depths is not None else float("nan")
        print(f"  {r.run_idx+1:>4}  {m.abs_rel:>8.4f}  {m.sq_rel:>8.4f}  "
              f"{m.rmse:>8.3f}  {m.rmse_log:>8.4f}  "
              f"{m.delta1:>8.4f}  {m.delta2:>8.4f}  {m.delta3:>8.4f}  {md:>8.3f}")

    # Across-run statistics
    print(f"\n  ── Across-Run Statistics (ALL pixels) ──")
    print(f"  {'Stat':>8}  {'AbsRel':>8}  {'SqRel':>8}  {'RMSE':>8}  "
          f"{'RMSElog':>8}  {'δ₁':>8}  {'δ₂':>8}  {'δ₃':>8}")
    print("  " + "·" * 98)

    for stat_name, fn in [("mean", np.mean), ("std", np.std),
                           ("min",  np.min),  ("max", np.max)]:
        vals = {k: fn([getattr(r.metrics_all, k) for r in study.runs])
                for k in _METRIC_KEYS}
        print(f"  {stat_name:>8}  "
              f"{vals['abs_rel']:>8.4f}  {vals['sq_rel']:>8.4f}  "
              f"{vals['rmse']:>8.3f}  {vals['rmse_log']:>8.4f}  "
              f"{vals['delta1']:>8.4f}  {vals['delta2']:>8.4f}  "
              f"{vals['delta3']:>8.4f}")

    # Pixel variance summary
    if study.pixel_std_map is not None:
        ps = study.pixel_std_map
        print(f"\n  ── Pixel-wise Depth Variation (across {N} runs) ──")
        print(f"  Mean pixel std  : {ps.mean():.4f} m")
        print(f"  Max  pixel std  : {ps.max():.4f} m")
        print(f"  95th-pct std    : {np.percentile(ps, 95):.4f} m")
        print(f"  Max depth range : {study.max_pixel_diff.max():.4f} m  "
              f"(max − min across runs, worst pixel)")
        md = study.run_mean_depths
        print(f"\n  Run mean depths : {md.mean():.3f} ± {md.std():.4f} m  "
              f"(min={md.min():.3f}  max={md.max():.3f})")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Shared plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def _run_labels(runs): return [f"Run {r.run_idx+1}" for r in runs]


def _get_metric_series(runs: List[RunResult], region: str, key: str) -> np.ndarray:
    return np.array([getattr(getattr(r, f"metrics_{region}"), key) for r in runs])


def _annotate_band(ax, xs, vals, color, label):
    mean = vals.mean(); std = vals.std()
    ax.plot(xs, vals, "o-", color=color, lw=1.8, ms=6, label=label)
    ax.axhline(mean, color=color, lw=1.0, ls="--", alpha=0.6)
    ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.10)


def _depth_imshow(ax, arr, cmap="turbo", vmin=None, vmax=None,
                  title="", colorbar=True):
    vis = np.where(arr > 0, arr, np.nan) if vmin == 0 else arr
    im  = ax.imshow(vis, cmap=cmap,
                    vmin=vmin if vmin is not None else np.nanmin(vis),
                    vmax=vmax if vmax is not None else np.nanmax(vis))
    ax.set_title(title, fontsize=8); ax.axis("off")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 – Per-run metric lines for all 7 metrics × 3 regions
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig1_per_run_metrics(study: RepeatabilityStudy, out_dir: str):
    runs = study.runs
    xs   = np.arange(1, len(runs) + 1)

    fig, axes = plt.subplots(4, 2, figsize=(18, 20), sharex=True)
    axes = axes.flatten()

    for ai, key in enumerate(_METRIC_KEYS):
        ax = axes[ai]
        for reg in _REGIONS:
            vals = _get_metric_series(runs, reg, key)
            _annotate_band(ax, xs, vals, _COLORS_REG[reg], _LABELS_REG[reg])
        ax.set_title(_METRIC_LBLS[key], fontsize=11, fontweight="bold")
        ax.set_ylabel(key, fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"R{i}" for i in xs], fontsize=8)
        ax.legend(fontsize=8, loc="best")

    # 8th panel: per-run mean predicted depth (all / veh / non)
    ax8 = axes[7]
    ax8.plot(xs, study.run_mean_depths,     "o-", color=_COLORS_REG["all"], lw=1.8, ms=6, label="All")
    ax8.plot(xs, study.run_mean_depths_veh, "s-", color=_COLORS_REG["veh"], lw=1.8, ms=6, label="Vehicle")
    ax8.plot(xs, study.run_mean_depths_non, "^-", color=_COLORS_REG["non"], lw=1.8, ms=6, label="Non-Veh")
    ax8.set_title("Mean Predicted Depth per Run (m)", fontsize=11, fontweight="bold")
    ax8.set_ylabel("Depth (m)"); ax8.set_xticks(xs)
    ax8.set_xticklabels([f"R{i}" for i in xs], fontsize=8)
    ax8.legend(fontsize=8)

    fig.suptitle(
        f"Per-Run Metric Trends  ·  {Path(study.image_path).name}\n"
        f"({len(runs)} runs of the same image — dashed line = mean, band = ±1 std)",
        fontsize=13, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    p = os.path.join(out_dir, "fig1_per_run_metrics.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Depth map gallery: RGB / GT / Mean / Std (per region overlay)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig2_depth_gallery(study: RepeatabilityStudy, out_dir: str):
    cmap = plt.cm.turbo
    H, W = study.gt_depth.shape

    fig = plt.figure(figsize=(24, 10))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.25)

    # Row 0: RGB, GT, Mean pred, Std map, Max-diff map
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_rgb.imshow(study.rgb_image); ax_rgb.axis("off")
    ax_rgb.set_title(f"RGB Input\n{Path(study.image_path).name}", fontsize=8)

    ax_gt = fig.add_subplot(gs[0, 1])
    gt_vis = np.where(study.gt_depth > 0, study.gt_depth, np.nan)
    im_gt = ax_gt.imshow(gt_vis, cmap=cmap, vmin=0, vmax=KITTI_MAX_DEPTH)
    ax_gt.set_title("LiDAR GT Depth (m)", fontsize=8); ax_gt.axis("off")
    plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

    ax_mn = fig.add_subplot(gs[0, 2])
    im_mn = ax_mn.imshow(study.pixel_mean_map, cmap=cmap, vmin=0, vmax=KITTI_MAX_DEPTH)
    ax_mn.set_title(f"Pixel Mean Depth ({len(study.runs)} runs)", fontsize=8); ax_mn.axis("off")
    plt.colorbar(im_mn, ax=ax_mn, fraction=0.046, pad=0.04)

    ax_st = fig.add_subplot(gs[0, 3])
    std_vmax = max(study.pixel_std_map.max(), 0.01)
    im_st = ax_st.imshow(study.pixel_std_map, cmap="hot", vmin=0, vmax=std_vmax)
    ax_st.set_title("Pixel Std Dev (m)  ← instability", fontsize=8); ax_st.axis("off")
    plt.colorbar(im_st, ax=ax_st, fraction=0.046, pad=0.04)

    ax_mx = fig.add_subplot(gs[0, 4])
    diff_vmax = max(study.max_pixel_diff.max(), 0.01)
    im_mx = ax_mx.imshow(study.max_pixel_diff, cmap="plasma", vmin=0, vmax=diff_vmax)
    ax_mx.set_title("Max−Min Depth Range per Pixel (m)", fontsize=8); ax_mx.axis("off")
    plt.colorbar(im_mx, ax=ax_mx, fraction=0.046, pad=0.04)

    # Row 1: vehicle mask | std on vehicles | std on non-vehicles | GT error | CV
    ax_vm = fig.add_subplot(gs[1, 0])
    overlay = study.rgb_image.copy().astype(np.float32)
    overlay[study.vehicle_mask] = (overlay[study.vehicle_mask]*0.35 +
                                    np.array([255,60,60])*0.65).clip(0,255)
    ax_vm.imshow(overlay.astype(np.uint8)); ax_vm.axis("off")
    ax_vm.set_title("SAM3 Vehicle Mask (red)", fontsize=8)

    ax_vs = fig.add_subplot(gs[1, 1])
    veh_std = study.pixel_std_map.copy(); veh_std[~study.vehicle_mask] = np.nan
    im_vs = ax_vs.imshow(veh_std, cmap="hot", vmin=0, vmax=std_vmax)
    ax_vs.set_title("Pixel Std — Vehicle only", fontsize=8); ax_vs.axis("off")
    plt.colorbar(im_vs, ax=ax_vs, fraction=0.046, pad=0.04)

    ax_ns = fig.add_subplot(gs[1, 2])
    non_std = study.pixel_std_map.copy(); non_std[study.vehicle_mask] = np.nan
    im_ns = ax_ns.imshow(non_std, cmap="hot", vmin=0, vmax=std_vmax)
    ax_ns.set_title("Pixel Std — Non-Vehicle", fontsize=8); ax_ns.axis("off")
    plt.colorbar(im_ns, ax=ax_ns, fraction=0.046, pad=0.04)

    ax_er = fig.add_subplot(gs[1, 3])
    err = study.pixel_mean_map - np.where(study.gt_depth > 0, study.gt_depth, study.pixel_mean_map)
    elim = max(abs(err).max(), 0.5)
    im_er = ax_er.imshow(err, cmap="RdBu_r", vmin=-elim, vmax=elim)
    ax_er.set_title("Mean Pred − GT (m)\n(red=over, blue=under)", fontsize=8); ax_er.axis("off")
    plt.colorbar(im_er, ax=ax_er, fraction=0.046, pad=0.04)

    ax_cv = fig.add_subplot(gs[1, 4])
    cv_vmax = np.percentile(study.pixel_cv_map, 98)
    im_cv = ax_cv.imshow(study.pixel_cv_map, cmap="YlOrRd", vmin=0, vmax=cv_vmax)
    ax_cv.set_title("Coefficient of Variation (std/mean)", fontsize=8); ax_cv.axis("off")
    plt.colorbar(im_cv, ax=ax_cv, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Depth Map Gallery – Repeatability Study  ·  {Path(study.image_path).name}",
        fontsize=13, fontweight="bold",
    )
    p = os.path.join(out_dir, "fig2_depth_gallery.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 – Pixel std histogram + sorted-pixel std plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig3_std_analysis(study: RepeatabilityStudy, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Panel A: histogram of pixel std across three regions
    for reg, mask_fn in [
        ("all", lambda: np.ones(study.gt_depth.shape, bool)),
        ("veh", lambda: study.vehicle_mask),
        ("non", lambda: ~study.vehicle_mask),
    ]:
        m = mask_fn()
        vals = study.pixel_std_map[m]
        axes[0].hist(vals, bins=100, alpha=0.55, density=True,
                     color=_COLORS_REG[reg], label=f"{_LABELS_REG[reg]}"
                     f"  μ={vals.mean():.3f}m  σ={vals.std():.3f}m")
    axes[0].set_xlabel("Per-pixel Depth Std Dev (m)", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].set_title("Distribution of Pixel-wise Std Dev\nacross Regions", fontsize=11)
    axes[0].legend(fontsize=8)

    # Panel B: sorted pixel std (shows how many pixels are stable)
    sorted_std = np.sort(study.pixel_std_map.ravel())
    pct        = np.linspace(0, 100, len(sorted_std))
    axes[1].plot(pct, sorted_std, lw=1.8, color="#5C6BC0")
    for thr in [0.05, 0.10, 0.20, 0.50]:
        frac = (sorted_std <= thr).mean() * 100
        axes[1].axhline(thr, color="grey", lw=0.8, ls="--")
        axes[1].text(2, thr + 0.005, f"{frac:.1f}% < {thr}m", fontsize=7.5,
                     color="grey")
    axes[1].set_xlabel("Pixel percentile (%)", fontsize=11)
    axes[1].set_ylabel("Pixel Std Dev (m)", fontsize=11)
    axes[1].set_title("Sorted Pixel Std Dev\n(shows fraction of stable pixels)", fontsize=11)

    # Panel C: max-diff histogram
    diff_vals = study.max_pixel_diff.ravel()
    for reg, mask_fn in [
        ("all", lambda: np.ones(study.gt_depth.shape, bool)),
        ("veh", lambda: study.vehicle_mask),
        ("non", lambda: ~study.vehicle_mask),
    ]:
        m = mask_fn()
        vals = study.max_pixel_diff[m]
        axes[2].hist(vals, bins=100, alpha=0.55, density=True,
                     color=_COLORS_REG[reg], label=f"{_LABELS_REG[reg]}"
                     f"  max={vals.max():.3f}m")
    axes[2].set_xlabel("Max − Min Depth per Pixel (m)", fontsize=11)
    axes[2].set_ylabel("Density", fontsize=11)
    axes[2].set_title("Worst-case Depth Swing per Pixel\n(max − min across all runs)", fontsize=11)
    axes[2].legend(fontsize=8)

    fig.suptitle(
        f"Pixel-wise Std Dev & Depth Range Analysis  ·  N={len(study.runs)} runs",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    p = os.path.join(out_dir, "fig3_std_analysis.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – Run-to-run difference maps  (depth[run_k] - depth[run_0])
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig4_run_diff_maps(study: RepeatabilityStudy, out_dir: str):
    runs  = study.runs
    N     = len(runs)
    base  = runs[0].pred_depth   # reference = first run
    cols  = min(N - 1, 9)        # show up to 9 diff maps
    if cols == 0:
        return

    fig, axes = plt.subplots(2, cols, figsize=(cols * 3.2, 7.5),
                              squeeze=False)

    elim_global = 0.0
    diffs = []
    for k in range(1, cols + 1):
        d = runs[k].pred_depth - base
        diffs.append(d)
        elim_global = max(elim_global, np.abs(d).max())
    elim_global = max(elim_global, 0.1)

    for col, k in enumerate(range(1, cols + 1)):
        d = diffs[col]
        # Top row: full diff map
        im = axes[0][col].imshow(d, cmap="RdBu_r",
                                  vmin=-elim_global, vmax=elim_global)
        axes[0][col].set_title(f"Run {k+1} − Run 1\nmax|Δ|={np.abs(d).max():.3f}m",
                                fontsize=8)
        axes[0][col].axis("off")
        plt.colorbar(im, ax=axes[0][col], fraction=0.05, pad=0.03,
                     format="%.2f")

        # Bottom row: abs diff, coloured by magnitude
        im2 = axes[1][col].imshow(np.abs(d), cmap="hot", vmin=0, vmax=elim_global)
        axes[1][col].set_title(f"|Δ| map  RMSE={np.sqrt((d**2).mean()):.4f}m",
                                fontsize=8)
        axes[1][col].axis("off")
        plt.colorbar(im2, ax=axes[1][col], fraction=0.05, pad=0.03,
                     format="%.2f")

    fig.suptitle(
        f"Run-to-Run Difference Maps  (each run vs Run 1)  ·  {Path(study.image_path).name}\n"
        "Top: signed Δ depth  |  Bottom: |Δ| depth  (metres)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    p = os.path.join(out_dir, "fig4_run_diff_maps.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 – Scatter consistency: run k vs run 1 depth values
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig5_scatter_consistency(study: RepeatabilityStudy, out_dir: str):
    runs  = study.runs
    N     = len(runs)
    base  = runs[0].pred_depth.ravel()
    cols  = min(N - 1, 9)
    if cols == 0:
        return

    fig, axes = plt.subplots(2, cols, figsize=(cols * 3.0, 8), squeeze=False)
    max_pts   = 5_000
    rng       = np.random.default_rng(0)

    # colour by vehicle / non-vehicle
    veh_flat = study.vehicle_mask.ravel()
    colors   = np.where(veh_flat, _COLORS_REG["veh"], _COLORS_REG["non"])

    lim = (0, KITTI_MAX_DEPTH)

    for col, k in enumerate(range(1, cols + 1)):
        comp = runs[k].pred_depth.ravel()
        diff = comp - base

        # Subsample
        ix = rng.choice(len(base), min(max_pts, len(base)), replace=False)

        # Row 0: scatter (run k vs run 1)
        axes[0][col].scatter(base[ix], comp[ix], s=1.5, alpha=0.35,
                              c=colors[ix], rasterized=True)
        axes[0][col].plot(lim, lim, "k--", lw=1.0)
        axes[0][col].set_xlim(*lim); axes[0][col].set_ylim(*lim)
        axes[0][col].set_title(f"Run {k+1} vs Run 1\nPearsonR={np.corrcoef(base[ix],comp[ix])[0,1]:.5f}",
                                fontsize=8)
        axes[0][col].set_xlabel("Run 1 (m)", fontsize=7)
        axes[0][col].set_ylabel(f"Run {k+1} (m)", fontsize=7)
        axes[0][col].tick_params(labelsize=7)

        # Row 1: diff histogram
        axes[1][col].hist(diff, bins=80, color="#5C6BC0", alpha=0.7, density=True)
        axes[1][col].axvline(0, color="black", lw=1.2, ls="--")
        axes[1][col].axvline(diff.mean(), color="red", lw=1.2, ls="-.",
                              label=f"μ={diff.mean():.4f}m")
        axes[1][col].set_title(f"Δ histogram  σ={diff.std():.4f}m", fontsize=8)
        axes[1][col].set_xlabel("Δ depth (m)", fontsize=7)
        axes[1][col].tick_params(labelsize=7)
        axes[1][col].legend(fontsize=7)

    fig.suptitle(
        f"Run Consistency Scatter & Diff Histograms  ·  {Path(study.image_path).name}\n"
        f"(blue = non-vehicle  |  pink = vehicle)  —  N={N} runs total",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    p = os.path.join(out_dir, "fig5_scatter_consistency.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 – Metric stability band: mean ± std across runs, per metric
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig6_metric_stability(study: RepeatabilityStudy, out_dir: str):
    runs = study.runs
    xs   = np.arange(1, len(runs) + 1)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for ai, key in enumerate(_METRIC_KEYS):
        ax = axes[ai]
        for reg in _REGIONS:
            vals = _get_metric_series(runs, reg, key)
            mean = vals.mean(); std = vals.std()
            ax.plot(xs, vals, "o-", color=_COLORS_REG[reg], lw=1.8, ms=6,
                    label=f"{_LABELS_REG[reg]}  μ={mean:.4f}  σ={std:.4f}")
            ax.axhline(mean, color=_COLORS_REG[reg], lw=1.0, ls="--", alpha=0.5)
            ax.fill_between(xs, mean - std, mean + std,
                            color=_COLORS_REG[reg], alpha=0.12)
        ax.set_title(_METRIC_LBLS[key], fontsize=11, fontweight="bold")
        ax.set_ylabel(key, fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"R{i}" for i in xs], fontsize=8)
        ax.legend(fontsize=7, loc="best")

    # Last panel: RMSE vs run index all three regions, larger
    axes[7].axis("off")
    # Re-use for inset table of std values
    axes[7].text(0.05, 0.95, "Metric σ (std across runs)", transform=axes[7].transAxes,
                 fontsize=11, fontweight="bold", va="top")
    lines = []
    for reg in _REGIONS:
        lines.append(f"\n  {_LABELS_REG[reg]}")
        for key in _METRIC_KEYS:
            vals = _get_metric_series(runs, reg, key)
            lines.append(f"    {key:<12} σ={vals.std():.5f}  range={vals.max()-vals.min():.5f}")
    axes[7].text(0.05, 0.88, "\n".join(lines), transform=axes[7].transAxes,
                 fontsize=7.5, va="top", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#F5F5F5", alpha=0.9))

    fig.suptitle(
        f"Metric Stability Across {len(runs)} Runs  ·  {Path(study.image_path).name}\n"
        "Dashed = mean  |  Band = ±1 std",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    p = os.path.join(out_dir, "fig6_metric_stability.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 – Coefficient of variation map + regional CV box-plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig7_cv_analysis(study: RepeatabilityStudy, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Panel A: CV map
    cv_vmax = np.percentile(study.pixel_cv_map, 98)
    im = axes[0].imshow(study.pixel_cv_map, cmap="YlOrRd", vmin=0, vmax=cv_vmax)
    axes[0].axis("off")
    axes[0].set_title("Coefficient of Variation Map\n(std / mean per pixel)", fontsize=11)
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="CV")

    # Panel B: CV by region box-plot
    region_data = []
    region_labels = []
    for reg in _REGIONS:
        mask = study.vehicle_mask if reg == "veh" else \
               (~study.vehicle_mask if reg == "non" else
                np.ones(study.gt_depth.shape, bool))
        vals = study.pixel_cv_map[mask].ravel()
        region_data.append(vals)
        region_labels.append(_LABELS_REG[reg])

    bp = axes[1].boxplot(region_data, patch_artist=True, notch=False,
                         medianprops=dict(color="black", lw=2))
    for patch, reg in zip(bp["boxes"], _REGIONS):
        patch.set_facecolor(_COLORS_REG[reg])
        patch.set_alpha(0.6)
    axes[1].set_xticklabels(region_labels, fontsize=10)
    axes[1].set_ylabel("Coefficient of Variation (std/mean)")
    axes[1].set_title("CV Distribution by Region", fontsize=11)
    axes[1].set_ylim(0, cv_vmax * 2)

    # Panel C: std vs mean scatter (per pixel, sampled)
    rng  = np.random.default_rng(1)
    H, W = study.gt_depth.shape
    all_idx = rng.choice(H * W, min(15000, H * W), replace=False)
    means_s = study.pixel_mean_map.ravel()[all_idx]
    stds_s  = study.pixel_std_map.ravel()[all_idx]
    veh_s   = study.vehicle_mask.ravel()[all_idx]
    cols_s  = np.where(veh_s, _COLORS_REG["veh"], _COLORS_REG["non"])
    axes[2].scatter(means_s, stds_s, s=1.5, alpha=0.3, c=cols_s, rasterized=True)
    axes[2].set_xlabel("Pixel Mean Depth (m)", fontsize=11)
    axes[2].set_ylabel("Pixel Std Dev (m)", fontsize=11)
    axes[2].set_title("Std Dev vs Mean Depth per Pixel\n(pink=vehicle, blue=non-vehicle)", fontsize=11)
    # ideal proportional line
    x_line = np.linspace(0, KITTI_MAX_DEPTH, 100)
    for frac, ls in [(0.01, ":"), (0.02, "--"), (0.05, "-.")]:
        axes[2].plot(x_line, frac * x_line, lw=1.2, ls=ls, color="grey",
                     label=f"{frac*100:.0f}% of depth")
    axes[2].legend(fontsize=8); axes[2].set_xlim(0, KITTI_MAX_DEPTH)
    axes[2].set_ylim(0, stds_s.max() * 1.1 + 0.01)

    fig.suptitle(
        f"Coefficient of Variation Analysis  ·  N={len(study.runs)} runs",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    p = os.path.join(out_dir, "fig7_cv_analysis.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 – Depth heatmap: run × depth-bin  (shows if any run shifts the histogram)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig8_depth_histogram_evolution(study: RepeatabilityStudy, out_dir: str):
    runs   = study.runs
    N      = len(runs)
    gt_dep = study.gt_depth
    valid  = (gt_dep > KITTI_MIN_DEPTH) & (gt_dep <= KITTI_MAX_DEPTH)

    bins    = np.linspace(0, KITTI_MAX_DEPTH, 81)
    bin_ctr = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    regions = [
        ("all", np.ones(gt_dep.shape, bool), "All pixels"),
        ("veh", study.vehicle_mask,           "Vehicle pixels"),
        ("non", ~study.vehicle_mask,           "Non-Vehicle pixels"),
    ]

    for ax, (reg, mask, title) in zip(axes, regions):
        hist_mat = np.zeros((N, len(bins) - 1))
        for i, r in enumerate(runs):
            combined = mask & valid
            if combined.sum() < 5:
                continue
            h, _ = np.histogram(
                np.clip(r.pred_depth[combined], KITTI_MIN_DEPTH, KITTI_MAX_DEPTH),
                bins=bins, density=True)
            hist_mat[i] = h

        im = ax.imshow(hist_mat, aspect="auto", origin="upper",
                       extent=[0, KITTI_MAX_DEPTH, N + 0.5, 0.5],
                       cmap="viridis", vmin=0)
        ax.set_xlabel("Predicted Depth (m)", fontsize=11)
        ax.set_ylabel("Run index", fontsize=11)
        ax.set_yticks(range(1, N + 1))
        ax.set_yticklabels([f"R{i+1}" for i in range(N)], fontsize=8)
        ax.set_title(f"Depth Distribution per Run\n{title}", fontsize=11)
        plt.colorbar(im, ax=ax, label="Density", fraction=0.046, pad=0.04)

        # Overlay the GT histogram as a line
        gt_h, _ = np.histogram(
            gt_dep[valid & mask], bins=bins, density=True)
        ax_twin = ax.twiny()
        ax_twin.plot(gt_h, bin_ctr, color="white", lw=1.5, ls="--", alpha=0.8,
                     label="GT (LiDAR)")
        ax_twin.set_xlabel("GT density", fontsize=8, color="white")
        ax_twin.tick_params(axis="x", colors="white", labelsize=7)
        ax_twin.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Depth Histogram Evolution Across {N} Runs  ·  {Path(study.image_path).name}\n"
        "Each row = one run's depth histogram  |  dashed white = LiDAR GT",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    p = os.path.join(out_dir, "fig8_histogram_evolution.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Master call
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_figures(study: RepeatabilityStudy, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print("\n[Graphs] Generating repeatability figures …")
    plot_fig1_per_run_metrics         (study, out_dir)
    plot_fig2_depth_gallery           (study, out_dir)
    plot_fig3_std_analysis            (study, out_dir)
    plot_fig4_run_diff_maps           (study, out_dir)
    plot_fig5_scatter_consistency     (study, out_dir)
    plot_fig6_metric_stability        (study, out_dir)
    plot_fig7_cv_analysis             (study, out_dir)
    plot_fig8_depth_histogram_evolution(study, out_dir)
    print(f"[Graphs] All figures → {out_dir}")
