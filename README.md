# Monocular Metric Depth Estimation Model Fine-Tuned for Dashcam Videos

This repository contains the full experimental pipeline for my M.Tech thesis, **"Monocular Metric Depth Estimation Model Fine-Tuned for Dashcam Videos"**, which is to be submitted to the Department of Civil Engineering (GeoInformatics), Indian Institute of Technology Kanpur, under the supervision of **Dr. Bharat Lohani**.

The core goal of this work is simple: take a single RGB frame from a dashcam, and predict how far away everything in it is, **in actual metres**, without using LiDAR or any other sensor at inference time. The code here covers everything from generating LiDAR ground truth for training/evaluation, to benchmarking existing depth foundation models, to fine-tuning the best one on real driving data.

---

## Table of Contents

- [Why this Thesis exists](#why-this-thesis-exists)
- [Pipeline overview](#pipeline-overview)
- [Repository structure](#repository-structure)
- [Environment setup](#environment-setup)
- [Dataset preparation](#dataset-preparation)
- [Stage-by-stage usage guide](#stage-by-stage-usage-guide)
  - [Stage 0 — Instance Segmentation (SAM3)](#stage-0--instance-segmentation-sam3)
  - [Stage 0.5 — LiDAR to Depth Map Generation](#stage-05--lidar--depth-map-generation)
  - [Stage 1 — Zero-Shot Model Benchmarking](#stage-1--zero-shot-model-benchmarking)
  - [Stage 2 — Error Trend Analysis](#stage-2--error-trend-analysis)
  - [Stage 3 — Vehicle-Anchored Scale/Shift Correction](#stage-3--vehicle-anchored-scaleshift-correction)
  - [Stage 4 — Fully Supervised Fine-Tuning](#stage-4--fully-supervised-fine-tuning)
  - [Stage 5 — Self-Supervised Fine-Tuning](#stage-5--self-supervised-fine-tuning)
- [Configuration — before you run anything](#configuration--before-you-run-anything)
- [Outputs for each Stages](#outputs-for-each-stages)
- [Summary of findings](#summary-of-findings)
- [Thesis document](#thesis-document)
- [Acknowledgements](#acknowledgements)

---

## Why this Thesis exists

LiDAR gives accurate, metric-scale depth, but it's expensive and the point clouds it produces are sparse — most of an image ends up with no depth value at all. Dashcams, on the other hand, are already sitting in millions of vehicles, recording continuously, and producing dense RGB video that nobody is using for depth. If a single dashcam frame could be turned into a dense, metrically-correct depth map, that unlocks a lot of downstream use cases — ADAS, scenario generation for simulation, video analytics — without needing extra hardware.

The problem is that most monocular depth models are either:
- **Relative-depth only** (they get the shape of the scene right but have no idea about actual metres), or
- **Metric but not tuned for dashcam-style driving footage** (trained on different camera setups, different scenarios).

This thesis evaluates the current best metric depth models on real driving data, figures out where and how they fail, and then fine-tunes the best-performing one specifically for this task — first with scale & shift correction using dimensions of vehicles, then with LiDAR supervision and then with a self-supervised pass that uses only consecutive video frames and oxts data.

## Pipeline overview

The work follows five sequential stages, and the notebooks in this repo map directly onto them:

```
┌─────────────────────┐     ┌──────────────────────┐     ┌───────────────────┐
│  1. (SAM3) Instance │     │  2. Zero-shot        │     │  3. Error Trend   │
│     Segmentation    │────▶│    benchmarking of  |────▶│     analysis      │
│     & LiDAR map     │     │     6 SOTA depth     │     │     for the best  │
│     Generation      │     │     models           │     │     metric model  │
└─────────────────────┘     └──────────────────────┘     └───────────────────┘
                                                                        │
                                                                        ▼
┌─────────────────────┐     ┌───────────────────────┐     ┌────────────────────────┐
│  6. Self-supervised │◀────│  5. Fully supervised │◀────│  4. vehicle-anchored   │
│     fine-tuning     │     │     fine-tuning       │     │     scale/shift        │
│     (video-frame    │     │     (LiDAR-supervised,│     │     correction         │
│      consistency)   │     │     ablation study)   │     │                        │
└─────────────────────┘     └───────────────────────┘     └────────────────────────┘
```

Running the whole thing end-to-end in order (Stage 0 → Stage 5) reproduces the full experimental chain. Each stage can also be run independently as long as its input artifacts (masks, depth maps, checkpoints) already exist and the notebooks check for these and tell you what's missing.

## Environment setup

The code was developed and run on Ubuntu with an NVIDIA TESLA V100 GPU (CUDA 12.8). A GPU with at least 32 GB of VRAM is recommended for fine-tuning while the benchmarking notebooks are lighter and can run on smaller GPUs.

Environment management for this project is done with **[uv](https://docs.astral.sh/uv/)** instead of conda/pip directly. Everything the project needs — PyTorch (CUDA 12.8 build), torchvision/torchaudio, xformers, OpenCV, Open3D, and even the two git-based packages (SAM3 and Depth Anything V3) — is already pinned in `pyproject.toml` and locked in `uv.lock`. In practice this means setup is a single command rather than a long chain of manual installs.

1. **Install uv** (one-time, if you don't already have it):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   (See the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/) for the Windows/macOS equivalents.)

2. **Clone the repo and sync the environment**:

   ```bash
   git clone <this-repo-url>
   cd <this-repo>
   uv sync
   ```

   `uv sync` reads `.python-version` (pinned to **3.12**) and automatically downloads that interpreter if you don't have it, then creates a `.venv/` and installs every dependency from `uv.lock` at the exact pinned version — including the two git dependencies (but not other models):

   - `depth_anything_3` from `github.com/ByteDance-Seed/Depth-Anything-3`
   - `sam3` from `github.com/facebookresearch/sam3`

   This step also pulls in the CUDA 12.8 NVIDIA wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc.), so you generally don't need a separate system-wide CUDA toolkit install to run PyTorch — just a driver new enough to support CUDA 12.8.

3. **Hugging Face authentication**: SAM3 and some of the depth checkpoints (UniDepth V2, DepthPro, DA3-Metric) are gated on Hugging Face. Authenticate once inside the environment and request access to the relevant model cards before running the notebooks:

   ```bash
   uv run hf auth login
   ```

4. **Register the environment as a Jupyter kernel and launch**:

   ```bash
   uv run python -m ipykernel install --user --name mmde --display-name "mmde (uv)"
   uv run jupyter lab
   ```

   Open whichever notebook you need and make sure it's using the **`mmde (uv)`** kernel. Alternatively, run notebooks non-interactively with `uv run jupyter nbconvert --to notebook --execute <notebook>.ipynb`.

5. **Adding or changing a dependency** (if you extend a notebook and need a new package):

   ```bash
   uv add <package-name>
   ```

   This updates `pyproject.toml` and `uv.lock` together — commit both so the environment stays reproducible for anyone else who clones the repo.

**Not using uv?** A plain `requirements.txt` is included as a fallback (`pip install -r requirements.txt` inside a Python 3.12 virtualenv), but it was exported from the uv environment and isn't actively maintained in parallel — treat `pyproject.toml` / `uv.lock` as the source of truth and the requirements file as a convenience snapshot.

## Dataset preparation

Only KITTI Eigen Split dataset is strictly required to reproduce the core results; but can be extended to nuScenes, IDD, and TIAND to widen geographic and sensor diversity.

| Dataset | Region | Format | Links |
|---|---|---|---|
| **KITTI** (Eigen Split) | Germany | `.png` point clouds + calibration files | [link](https://www.kaggle.com/datasets/awsaf49/kitti-eigen-split-dataset) [link](https://www.cvlibs.net/datasets/kitti/raw_data.php) |
| **nuScenes** | USA / Singapore | `.bin` point clouds | [link](https://www.kaggle.com/datasets/mitanshuchakrawarty/nuscenes) [link](https://www.nuscenes.org/nuscenes) |
| **IDD** (India Driving Dataset) | India | `.npy` point clouds | [link](https://idd.insaan.iiit.ac.in/) |
| **TIAND** | India | `.pcd` point clouds | [link](https://tihan.iith.ac.in/TiAND.html) |

Download each dataset from its official source (KITTI raw + Eigen split file list, nuScenes devkit + data, IDD, TIAND) and note the root directory which you will have to point each notebook's `*_ROOT` config variable to it.

### Generating LiDAR ground-truth depth maps 

> Not required for KITTI eigen split dataset

`generate_depth_maps.ipynb` projects raw LiDAR point clouds onto the front-camera image plane to produce dense-enough, orthonormal ground-truth depth maps for nuScenes, TIAND, and IDD. It expects the following layout per dataset:

```
<TIAND_ROOT>/<sequence_name>/image_2/       # camera frames
<TIAND_ROOT>/<sequence_name>/velodyne/      # LiDAR .pcd files
<TIAND_ROOT>/<sequence_name>/calib/calib.txt

<IDD_ROOT>/<split>/<segment_id>/leftImg8bit/   # camera frames
<IDD_ROOT>/<split>/<segment_id>/lidar/          # .npy point clouds
<IDD_ROOT>/calib/idd_calib.yaml
```

Depth is encoded as 8-bit PNGs: `0` = invalid pixel (no LiDAR return), `1–255` linearly maps to `MIN_DEPTH_M`–`MAX_DEPTH_M` (defaults: 0.1 m–80 m). 
KITTI ground truth is handled directly inside each evaluation notebook (it ships with usable LiDAR projections and is not routed through the generator above).

### Optional: reflection removal pre-processing (DC-Gaussian)

**Notebook:** `dc_gaussian_pipeline.ipynb`

Dashcam footage often has windshield glare and reflections that can corrupt depth predictions near the top of the frame. This notebook is an optional pre-processing step that runs [DC-Gaussian](https://arxiv.org/abs/2405.17705) (Wang et al., NeurIPS 2024) — COLMAP for pose estimation, followed by a 3D Gaussian Splatting reconstruction trained per-sequence — to produce a reflection-free version of a frame sequence. It's not wired into the main five-stage pipeline by default; it's here for anyone who wants to test whether cleaning up reflections first changes downstream depth accuracy.

## Stage-by-stage usage guide

Every notebook follows the same convention: a **Configuration cell near the top** collects all paths, GPU index, and hyperparameters in one place — edit that cell, then run all cells top to bottom. Logging is set up early in each notebook so you get progress bars and per-step timing in the console. All the notebooks have resume capability.

### Stage 0 — Instance Segmentation (SAM3)

**Notebook:** `inst_seg_notebook_final.ipynb`

Runs Meta's Segment Anything Model 3 (SAM3) over the KITTI Eigen-Split test set to produce per-frame, per-category binary instance masks (vehicles, pedestrians, and stationary infrastructure). These masks are the backbone of every category-level evaluation later in the pipeline — without them, all downstream notebooks can only report whole-image (per-frame) metrics.

- **Input:** KITTI Eigen-Split RGB frames.
- **Output:** A "pre-generated mask store" (per-frame, per-category binary PNG/npy masks) that later notebooks load directly instead of re-running segmentation.
- **Requires:** SAM3 installed and Hugging Face access approved (see [Environment setup](#environment-setup)).

Run this **first** — every "premasked" notebook downstream depends on its output.

### Stage 0.5 — LiDAR → Depth Map Generation

**Notebook:** `generate_depth_maps.ipynb`

Converts raw LiDAR point clouds from nuScenes, IDD, and TIAND into dense, uint8-encoded, camera-aligned depth-map PNGs (see [Dataset preparation](#dataset-preparation) above for the encoding scheme). Includes a built-in QC cell that plots a sample depth map, its colourised version, and a depth histogram so you can sanity-check calibration before moving on — this step is worth not skipping, since a bad extrinsic/intrinsic calibration here silently corrupts every metric computed later.

### Stage 1 — Zero-Shot Model Benchmarking

**Notebooks:** `DA_2_metric_analysis_premasked_updated.ipynb`, `DA_3_giant_analysis_premasked_updated.ipynb`, `depthpro_premasked_analysis_v2.ipynb`, `unidepth_v2_premasked_eval_updated.ipynb`

Each of these notebooks loads one pretrained depth model, runs it on the full KITTI Eigen-Split test set, and computes per-frame **and** per-instance-category metrics (using the SAM3 masks from Stage 0) under four scale-alignment strategies (unscaled, median-scaling, least-squares scale, least-squares scale+shift). They all share the same structure, so once you've run one, the others follow the same recipe:

1. **Config cell** — set `KITTI_ROOT`, cache directory, GPU index.
2. **Model loading** — pulls the checkpoint from Hugging Face (cached locally after the first run).
3. **Inference loop** — runs the model over every test frame, applies the pre-generated masks, computes metrics (AbsRel, RMSE, δ₁/δ₂/δ₃, etc.) at the frame and category level.
4. **Plotting / summary cells** — writes out CSVs and comparison plots per scale-alignment strategy.

Run all four (or however many you're comparing) before moving to Stage 2 — Stage 2 assumes the DA3-Metric results already exist.

### Stage 2 — Error Trend Analysis

**Notebook:** `DA_3_metric_trend_analysis_FINAL.ipynb`

Takes the best model from Stage 1 (Depth Anything V3 Metric-Large, "DA3-M") and digs into *how* it's wrong, not just *how wrong* it is. It bins errors by ground-truth depth range, plots residual trends, and flags systematic bias.

- **Input:** KITTI Eigen-Split, raw (unscaled) DA3-Metric predictions, valid-LiDAR masks.
- **Output:** Trend plots (error vs. depth, error vs. object category, error vs. scene position) and a CSV of per-frame residuals used again in Stage 3.

### Stage 3 — Vehicle-Anchored Scale/Shift Correction

**Notebook:** `da3_vehicle_anchored_v6.ipynb`

This is the "can we fix the bias without touching the network weights?" experiment. The idea is deliberately lightweight: since retraining a foundation model is expensive, first check whether a geometric correction gets you most of the way there.

It runs in two passes:

| Pass | What happens |
|---|---|
| **Pass 1** | Load SAM3 vehicle masks → run DA3 depth inference → predict per-sequence camera intrinsics (using DA3-Nested-Giant) → extract vehicle bounding-box height/width → fit a global scale and shift using **RANSAC** (for outlier-robust rejection of bad detections) followed by **Weighted Least Squares** against known average vehicle dimensions (height ≈ 1.5 m, width ≈ 1.8 m, with per-category reliability weights) |
| **Pass 2** | Apply the fitted (scale, shift) to raw DA3 depth and re-evaluate against LiDAR ground truth across all scale variants |

Crucially, this correction uses **no LiDAR at inference time** — it only needs vehicle detections and known physical vehicle sizes, so it's a realistic option for a deployed dashcam system that has no sensor to fall back on. 

Set `SKIP_PASS1 = True` if Pass 1 has already been run and you just want to re-use its cached scale/shift values, or `SKIP_INFERENCE = True` to jump straight to the analysis cells from an existing results CSV.

### Stage 4 — Fully Supervised Fine-Tuning

**Notebooks:** `DA3_supervised_FT_Ablation_HPT_v6.ipynb` → `DA3_Super_Ablation_Analysis_-_Copy.ipynb` → `kitti_da3_finetune_v4_copy.ipynb`

Here it runs a structured ablation study first, and only then commits to a final configuration:

1. **`DA3_supervised_FT_Ablation_HPT_v6.ipynb`** — fine-tunes DA3-Metric-Large on the KITTI Eigen train split under a grid of:
   - 13 loss function variants (classic losses like SILog and BerHu, plus several from recent depth-estimation papers),
   - 3 architecture variants (decoder-only, partial fine-tune, full fine-tune),
   - 4 augmentation strategies,
   - 3 optimiser/scheduler combinations.

   Each configuration is trained and logged (train/val loss curves, periodic checkpoints) to a summary CSV.

2. **`DA3_Super_Ablation_Analysis_-_Copy.ipynb`** — loads that summary CSV and the per-epoch history for every configuration, and produces the comparison plots (validation RMSE and test RMSE vs. epoch, per configuration) used to pick a winner.

3. **`kitti_da3_finetune_v4_copy.ipynb`** — takes the best-performing recipe from the ablation and re-trains it as the final model, using seven additional loss-combination variants (tagged A1–A4, B1–B2, C in the notebook) on top of the winning architecture/augmentation/optimiser settings. It fine-tunes with a 90/10 train-val split on KITTI Eigen train, checkpoints every epoch (`latest.pth`, resumable), and evaluates the best checkpoint on the held-out test split.

### Stage 5 — Self-Supervised Fine-Tuning

**Notebooks:** `DA3_SelfSupervised_FT_Ablation_HPT_v5 (1).ipynb`→ `da3_litti-finetune_fixed4.ipynb`

A companion study to Stage 4, asking a different question: rather than "which loss best uses sparse LiDAR labels", this asks "which geometric signal from raw video can still improve the model, without using the LiDAR labels?" This matters practically, since most dashcam footage in the wild has no LiDAR alongside it — if fine-tuning can be done from video (and GPS data), the approach scales to far more data than KITTI or any other dataset provides.

> WORK IN PROGRESS

The notebook implements (for now):
- A 6-DoF pose network (ResNet-18 backbone predicting axis-angle rotation + translation between consecutive frames),
- Differentiable inverse warping (projecting one frame into another using the predicted depth and pose),
- 13 self-supervised / hybrid loss variants spanning the Monodepth2 → PackNet-SfM → ManyDepth → FeatDepth → VideoDA → UniDepthV2 lineage, ranging from photometric reprojection (best geometric grounding) down to a simple temporal depth-consistency regulariser,
- The same 3 architecture variants and hyperparameter sweep structure as the supervised ablation, so results are directly comparable.
- And further more Algorithms to check and improve the accuracy.

Initial results from `da3_litti-finetune_fixed4.ipynb` gave only a modest further improvement, and mostly within the 40–50 m depth range but further study is still ongoing.

## Configuration — before you run anything

Every notebook centralises its editable settings in a config cell at the top. Please check and update (based on your scenario):

- `*_ROOT` paths (`KITTI_ROOT`) — point these at wherever you stored the dataset.
- `CACHE_DIR` — where Hugging Face checkpoints get cached.
- `CUDA_VISIBLE_DEVICES` (set at the very top of most notebooks, before `torch` is imported) — pick which GPU to use.
- Output directories — these are created automatically if they don't exist, but double-check you have disk space; depth-map generation and checkpoint saving can both be storage-heavy.

## Outputs for each Stages

- **Stage 0:** a mask store of per-frame, per-category binary masks.
- **Stage 0.5:** uint8 PNG depth maps per dataset.
- **Stage 1:** per-model CSVs of frame-level and category-level accuracy metrics under 4 scale-alignment strategies, with comparison plots.
- **Stage 2:** error-trend plots and a per-frame residual CSV for the best model.
- **Stage 3:** fitted global (scale, shift) parameters, a corrected-depth evaluation CSV, and analysis plots.
- **Stage 4:** an ablation summary CSV, per-configuration training histories, and a final fine-tuned checkpoint (`latest.pth` + best checkpoint).
- **Stage 5:** an ablation summary CSV and a final self-supervised-refined checkpoint.
an ablation summary CSV, per-configuration training histories, per-strategy training histories, and a final fine-tuned checkpoint (`latest.pth` + best checkpoint).

## Summary of findings

(Full detail will be in the thesis; this is just a short pointer)

- Among the zero-shot models evaluated, **Depth Anything V3 Metric-Large** performed best on KITTI Eigen-Split Dataset.
- It shows a systematic depth **underestimation within roughly 40–50 m** compared to LiDAR ground truth.
- A geometry-only fix (vehicle-size-anchored RANSAC + WLS scale/shift correction, no LiDAR at inference) gives only a **marginal** improvement — and scale-only correction actually makes things worse — which motivated moving to fine-tuning.
- The best **Fully supervised fine-tuning** on LiDAR-labelled KITTI Eigen Split Dataset gave the largest accuracy gain and is able to percieve vehicles with about 1m accuracy.
- The initial results of **Self-supervised fine-tuning** gave a smaller improvement, mostly in the 40–50 m to below range.
- Further work in the **Self-supervised fine-tuning** is still ongoing, which include about 5 new SSFT algorithms, which are still being processed and evaluated.

## Thesis document

The full write-up, including all formulae, dataset descriptions, and result tables, will be included later when it is officially published.


## Acknowledgements

This work builds on several open-source models and datasets: Segment Anything Model 3 (SAM3), UniDepth V2, Apple DepthPro, Depth Anything V2/V3, and the KITTI dataset. Thanks to Prof. Bharat Lohani for supervision throughout this thesis. And Thanks to my Dear Friends and Family of thier invaluable support.
