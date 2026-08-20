# AUTOMATED LUMBAR SPINAL STENOSIS GRADING
A three-stage deep learning pipeline for automated severity grading of lumbar spinal canal stenosis from 3D Sagittal T2/STIR MRI volumes, at all five lumbar intervertebral levels (L1–L2 through L5–S1), built on the RSNA 2024 Lumbar Spine Degenerative Classification benchmark dataset.\
The system uses a three-stage pipeline to identify clinically relevant slices, localize lumbar anatomical landmarks, and classify stenosis severity at each intervertebral level.

## Overview

Lumbar spinal stenosis grading traditionally requires manual radiological review across multiple vertebral levels. This project automates the process through a three-stage pipeline:

1. **Slice Selection** — an EfficientNet-B0 classifier identifies the most diagnostically relevant MRI slice from each 3D volume.
2. **Keypoint Detection** — a lightweight U-Net localizes 5 vertebral levels via Gaussian heatmap regression.
3. **Severity Classification** — a 2.5D EfficientNet-B0 classifier grades stenosis severity (Normal/Mild, Moderate, Severe) using multi-slice ROI patches centered on the detected keypoints.

## Key Results

| Stage | Metric | Value |
|---|---|---|
| Stage 1 — Slice Selector | Validation BCE loss | 0.165 |
| Stage 2 — Keypoint Detector | Validation MSE (heatmap) | 0.058 |
| Stage 3 — Severity Classifier | Macro AUC-ROC | **0.649** |

## Tech Stack

- **Framework:** PyTorch, torchvision
- **Models:** EfficientNet-B0, U-Net
- **Data handling:** pydicom, NumPy, Pandas
- **Image processing:** OpenCV
- **Visualization:** Matplotlib
- **Training infra:** Kaggle Notebooks (NVIDIA T4/P100 GPU)

## Key Engineering Details

- **Class imbalance handling** — inverse-frequency weighted cross-entropy loss for the severity classifier.
- **I/O optimization** — float16 in-memory RAM caching of DICOM volumes, reducing disk I/O by ~80% during training.
- **Training setup** — AdamW optimizer with cosine annealing LR schedule, batch size 16.

## Dataset

Trained and evaluated on the [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) dataset (1,975 studies, Sagittal T2/STIR MRI).

## Repository Structure

```
config_and_utils.py          # constants, hyperparameters, DICOM I/O, normalization, RAM caching
stage1_slice_selector.py     # SliceSelectorDataset, EfficientNet-B0 model, train/infer
stage2_keypoint_detector.py  # KeypointDataset, U-Net, Gaussian heatmaps, train/infer
stage3_severity_classifier.py# SeverityDataset, 2.5D EfficientNet-B0, ROI extraction, train
```

## Setup
 
```bash
git clone https://github.com/yash-gupta05/LUMBAR-SPINAL-STENOSIS-GRADING-.git
cd lumbar-scs-severity-grading
pip install -r requirements.txt
```

## Usage
 
### Training
 
Each stage script is fully self-contained — it builds its own volume cache, does its own train/val split, and saves its own checkpoints. `train.py` just runs them in order:
 
```bash
# Run all three stages end-to-end
python train.py
 
# Run specific stages only (e.g. re-run Stage 3 after tweaking it,
# reusing existing Stage 1/2 checkpoints)
python train.py --stages 3
 
# Or run a stage script directly
python stage1_slice_selector.py
```
 
### Inference
 
`inference.py` runs the full three-stage pipeline on a single study and prints (or saves) structured per-level severity predictions:
 
```bash
# Sagittal T2/STIR — Spinal Canal Stenosis (the scope of this project)
python inference.py --study_id 12345678
 
# Save results to a JSON file
python inference.py --study_id 12345678 --out_json results.json
 
# Run every series branch that has trained checkpoints available
python inference.py --study_id 12345678 --series_key all
```
