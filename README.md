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
├── config_and_utils.py          # Shared config, constants, and utility functions
├── stage1_slice_selector.py     # EfficientNet-B0 slice selection model
├── stage2_keypoint_detector.py  # U-Net keypoint detector (Gaussian heatmap regression)
├── stage3_severity_classifier.py # 2.5D EfficientNet-B0 severity classifier
└── README.md
```

## Setup
 
```bash
git clone https://github.com/<your-username>/lumbar-scs-severity-grading.git
cd lumbar-scs-severity-grading
pip install -r requirements.txt
```

## Usage
 
```bash
# Stage 1: train the slice selector
python stage1_slice_selector.py --train
 
# Stage 2: train the keypoint detector
python stage2_keypoint_detector.py --train
 
# Stage 3: train the severity classifier
python stage3_severity_classifier.py --train
```
