# AUTOMATED LUMBAR SPINAL STENOSIS GRADING
A three-stage deep learning pipeline for automated severity grading of lumbar spinal canal stenosis from 3D Sagittal T2/STIR MRI volumes, built on the RSNA 2024 Lumbar Spine Degenerative Classification benchmark dataset.

## Overview

Lumbar spinal stenosis grading traditionally requires manual radiological review across multiple vertebral levels. This project automates the process through a three-stage pipeline:

1. **Slice Selection** — an EfficientNet-B0 classifier identifies the most diagnostically relevant MRI slice from each 3D volume.
2. **Keypoint Detection** — a lightweight U-Net localizes 5 vertebral levels via Gaussian heatmap regression.
3. **Severity Classification** — a 2.5D EfficientNet-B0 classifier grades stenosis severity (Normal/Mild, Moderate, Severe) using multi-slice ROI patches centered on the detected keypoints.

## Results

| Stage | Task | Metric |
|---|---|---|
| 1 | Slice selection | Validation loss: 0.165 |
| 2 | Keypoint localization | Validation loss: 0.058 |
| 3 | Severity classification | Macro AUC-ROC: 0.649 |

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
pip install torch torchvision pydicom numpy pandas opencv-python matplotlib
```

## Usage

```bash
python stage1_slice_selector.py      # Train/run slice selection
python stage2_keypoint_detector.py   # Train/run keypoint detection
python stage3_severity_classifier.py # Train/run severity classification
```
