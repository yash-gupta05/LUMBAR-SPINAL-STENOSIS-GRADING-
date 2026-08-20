# ============================================================
#  RSNA 2024 Lumbar Spine  |  stage3_severity_classifier.py
#
#  Two models, one per sagittal disease group:
#      sag_t2  →  spinal_canal_stenosis
#      sag_t1  →  left_subarticular_stenosis
#                 right_subarticular_stenosis
#
#  Design
#  ------
#  A single model handles ALL levels (L1-L2 … L5-S1) and, for
#  sag_t1, both left and right sides.  Each call to the model
#  takes one ROI crop and returns one severity prediction.
#
#  ROI crops (2.5-D)
#  -----------------
#  Centre coordinate from Stage 2 → square patch (2*roi_pad) from the
#  original-resolution slice.  Central slice ± n_extra neighbours are
#  stacked to give a 3-channel input (2.5-D).
#
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2

from config_and_utils import (
    CFG, TRAIN_DIR, OUTPUT_DIR,
    SAGITTAL_SERIES, CONDITIONS_BY_SERIES,
    LEVELS, NUM_LEVELS, NUM_SEVERITY_CLASSES,
    SEVERITY_MAP,
    load_dataframes, filter_coords_by_series_key,
    load_dicom_volume, resize_slice,
    build_volume_cache, cache_get,
    build_instance_index_map, inst_to_idx,
    get_tidy_labels, train_val_split,
    seed_everything,
)

seed_everything()

CKPT_NAMES = {
    "sag_t2": "stage3_sag_t2.pt",
    "sag_t1": "stage3_sag_t1.pt",
}


# ── 1.  ROI cropping (2.5-D) ─────────────────────────────────
def crop_roi_25d(volume: np.ndarray,
                  slice_idx: int,
                  cx_px: float,
                  cy_px: float,
                  pad:    int = CFG.roi_pad,
                  n_extra: int = CFG.n_extra) -> np.ndarray:
    """
    Extract a 2.5-D patch centred on (cx_px, cy_px).

    For n_extra=1 three slices are stacked:
        slice_idx - 1,  slice_idx,  slice_idx + 1
    Boundary indices are clamped; edge crops are zero-padded so the
    output is always (2*n_extra+1, 2*pad, 2*pad).

    Parameters
    ----------
    volume    : (D, H, W) float32 array in [0, 1]
    slice_idx : central slice index
    cx_px     : x centre (original-res pixels)
    cy_px     : y centre (original-res pixels)
    pad       : half-side of the square crop
    n_extra   : neighbours on each side

    Returns
    -------
    np.ndarray  shape (2*n_extra+1, 2*pad, 2*pad), dtype float32
    """
    D, H, W = volume.shape
    cx, cy  = int(round(cx_px)), int(round(cy_px))

    channels = []
    for offset in range(-n_extra, n_extra + 1):
        si  = min(max(slice_idx + offset, 0), D - 1)
        img = volume[si]

        x1, x2 = max(cx - pad, 0), min(cx + pad, W)
        y1, y2 = max(cy - pad, 0), min(cy + pad, H)
        crop    = img[y1:y2, x1:x2]

        top    = max(0, pad - cy)
        bottom = max(0, cy + pad - H)
        left   = max(0, pad - cx)
        right  = max(0, cx + pad - W)
        if any([top, bottom, left, right]):
            crop = cv2.copyMakeBorder(
                crop, top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT, value=0,
            )

        crop = cv2.resize(crop, (2 * pad, 2 * pad),
                           interpolation=cv2.INTER_LINEAR)
        channels.append(crop.astype(np.float32))

    return np.stack(channels, axis=0)   # (2*n_extra+1, 2*pad, 2*pad)


# ── 2.  Pre-compute keypoint map ──────────────────────────────
def build_keypoint_map(pairs_df:       pd.DataFrame,
                        s2_model,
                        cache:          dict,
                        best_slice_map: dict) -> dict:
    """
    Run Stage-2 inference on every (study_id, series_id) pair and
    store the keypoint dict so __getitem__ never calls predict_keypoints.

    Parameters
    ----------
    pairs_df       : DataFrame [study_id, series_id]
    s2_model       : trained KeypointUNet, or None (fallback: GT coords)
    cache          : volume cache
    best_slice_map : { (study_id, series_id): best_slice_idx }

    Returns
    -------
    dict { (study_id, series_id): { level: (cx_px, cy_px) } }
    """
    from stage2_keypoint_detector import predict_keypoints

    kp_map = {}
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df),
                       desc="Stage-2 inference (building keypoint map)"):
        study_id  = int(row["study_id"])
        series_id = int(row["series_id"])
        best_idx  = best_slice_map.get((study_id, series_id), None)

        if best_idx is None:
            vol      = cache_get(cache, study_id, series_id)
            best_idx = vol.shape[0] // 2

        if s2_model is not None:
            vol      = cache_get(cache, study_id, series_id)
            slice_2d = vol[best_idx]
            kp       = predict_keypoints(s2_model, slice_2d)
        else:
            kp = None   # handled in build_sample_list via GT fallback

        kp_map[(study_id, series_id)] = kp

    return kp_map


# ── 3.  Build sample list ─────────────────────────────────────
def build_sample_list(
        coords_df:      pd.DataFrame,
        labels_df:      pd.DataFrame,
        cache:          dict,
        best_slice_map: dict,
        keypoint_map:   dict,
        series_key:     str,
) -> list:
    """
    Build a flat list of per-ROI samples, one per
    (study_id, series_id, level, condition) tuple.

    Each entry is a dict:
        study_id, series_id, slice_idx,
        cx_px, cy_px,        ← ROI centre from Stage 2 (or GT)
        condition, level,
        severity             ← 0 / 1 / 2

    Parameters
    ----------
    coords_df      : coordinate rows filtered for series_key
    labels_df      : raw labels DataFrame (train.csv)
    cache          : volume cache
    best_slice_map : { (study_id, series_id): slice_idx }
    keypoint_map   : { (study_id, series_id): { level: (cx, cy) } or None }
    series_key     : 'sag_t2' or 'sag_t1'
    """
    tidy       = get_tidy_labels(labels_df)
    conditions = CONDITIONS_BY_SERIES[series_key]
    samples    = []

    for (study_id, series_id), grp in tqdm(
            coords_df.groupby(["study_id", "series_id"]),
            desc=f"Building samples [{series_key}]"):

        study_id  = int(study_id)
        series_id = int(series_id)

        folder = TRAIN_DIR / str(study_id) / str(series_id)
        if not folder.exists():
            continue

        best_idx = best_slice_map.get((study_id, series_id), None)
        if best_idx is None:
            vol      = cache_get(cache, study_id, series_id)
            best_idx = vol.shape[0] // 2

        # ── Stage-2 keypoints (already computed, no model call) ──
        kp = keypoint_map.get((study_id, series_id))

        if kp is None:
            # Fallback: use annotated GT coordinates from the CSV
            vol    = cache_get(cache, study_id, series_id)
            _, H, W = vol.shape
            kp = {}
            for _, row in grp.iterrows():
                lvl = row["level"].lower().replace(" ", "_")
                kp[lvl] = (float(row["x"]), float(row["y"]))

        # ── One sample per level × condition ─────────────────
        for lvl, (cx, cy) in kp.items():
            for cond in conditions:
                sev_rows = tidy[
                    (tidy["study_id"]  == study_id) &
                    (tidy["condition"] == cond) &
                    (tidy["level"]     == lvl)
                ]
                if sev_rows.empty:
                    continue
                severity = int(sev_rows.iloc[0]["severity"])
                if severity < 0:
                    continue

                samples.append({
                    "study_id":  study_id,
                    "series_id": series_id,
                    "slice_idx": best_idx,
                    "cx_px":     cx,
                    "cy_px":     cy,
                    "condition": cond,
                    "level":     lvl,
                    "severity":  severity,
                })

    return samples


# ── 4.  Dataset ───────────────────────────────────────────────
class SeverityDataset(Dataset):
    """
    One sample = one 2.5-D ROI crop + severity label (0 / 1 / 2).

    The ROI is (3, 2*roi_pad, 2*roi_pad).
    All volume data comes from `cache` — no DICOM I/O in __getitem__.
    """

    def __init__(self, sample_list: list,
                 cache: dict,
                 augment: bool = False):
        self.samples = sample_list
        self.cache   = cache
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        vol   = cache_get(self.cache, s["study_id"], s["series_id"])
        patch = crop_roi_25d(vol, s["slice_idx"], s["cx_px"], s["cy_px"])
        # patch: (3, 2*roi_pad, 2*roi_pad)

        if self.augment:
            if np.random.rand() > 0.5:
                patch = patch[:, :, ::-1].copy()   # horizontal flip
            if np.random.rand() > 0.5:
                patch = patch[:, ::-1, :].copy()   # vertical flip

        img_t = torch.tensor(patch, dtype=torch.float32)
        lbl_t = torch.tensor(s["severity"], dtype=torch.long)
        return img_t, lbl_t


# ── 5.  Model ─────────────────────────────────────────────────
class SeverityClassifier(nn.Module):
    """
    EfficientNet-B0 for 3-channel 2.5-D ROI input.
    Output: logits over NUM_SEVERITY_CLASSES = 3.
    One checkpoint per series type; applied to every level/side.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = (models.EfficientNet_B0_Weights.IMAGENET1K_V1
                   if pretrained else None)
        base = models.efficientnet_b0(weights=weights)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, NUM_SEVERITY_CLASSES),
        )
        self.net = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 3)


# ── 6.  Training ──────────────────────────────────────────────
def compute_class_weights(sample_list: list) -> torch.Tensor:
    """
    Inverse-frequency class weights to mitigate Normal/Mild imbalance.
    Returns a (NUM_SEVERITY_CLASSES,) float tensor on CFG.device.
    """
    counts = np.zeros(NUM_SEVERITY_CLASSES, dtype=np.float64)
    for s in sample_list:
        counts[s["severity"]] += 1
    counts  = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32).to(CFG.device)


def train_severity_classifier(
        train_samples: list,
        val_samples:   list,
        cache:         dict,
        series_key:    str,
) -> "SeverityClassifier":
    """
    Train a SeverityClassifier for one sagittal disease group.

    Parameters
    ----------
    train_samples : list of sample dicts from build_sample_list
    val_samples   : list of sample dicts for validation
    cache         : volume cache from build_volume_cache()
    series_key    : 'sag_t2' or 'sag_t1'
    """
    tag       = f"S3/{series_key}"
    save_path = CFG.output_dir / CKPT_NAMES[series_key]

    train_ds = SeverityDataset(train_samples, cache, augment=True)
    val_ds   = SeverityDataset(val_samples,   cache, augment=False)

    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG.batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)

    class_weights = compute_class_weights(train_samples)
    print(f"  [{tag}] class weights: {class_weights.cpu().numpy()}")

    model     = SeverityClassifier(pretrained=True).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=CFG.epochs_s3)
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")

    for epoch in range(1, CFG.epochs_s3 + 1):
        model.train()
        train_loss = 0.0
        for patches, labels in tqdm(train_dl,
                                     desc=f"[{tag}] epoch {epoch:02d} train",
                                     leave=False):
            patches, labels = patches.to(CFG.device), labels.to(CFG.device)
            optimizer.zero_grad()
            logits = model(patches)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for patches, labels in val_dl:
                patches, labels = patches.to(CFG.device), labels.to(CFG.device)
                logits   = model(patches)
                val_loss += loss_fn(logits, labels).item()
                preds    = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        scheduler.step()

        avg_tr = train_loss / max(len(train_dl), 1)
        avg_vl = val_loss   / max(len(val_dl),   1)
        acc    = correct / max(total, 1) * 100
        print(f"  [{tag}] epoch {epoch:02d} | "
              f"train {avg_tr:.4f}  val {avg_vl:.4f}  acc {acc:.1f}%")

        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ checkpoint saved  (val={best_val_loss:.4f})")

    model.load_state_dict(torch.load(save_path, map_location=CFG.device))
    model.eval()
    return model


# ── 7.  Inference ─────────────────────────────────────────────
@torch.no_grad()
def predict_severity(model: SeverityClassifier,
                      roi_patch: np.ndarray) -> tuple:
    """
    Predict severity class for a single 2.5-D ROI patch.

    Parameters
    ----------
    roi_patch : (3, H, W) float32 array

    Returns
    -------
    (pred_class: int, probabilities: np.ndarray shape (3,))
    """
    model.eval()
    t      = torch.tensor(roi_patch, dtype=torch.float32).unsqueeze(0).to(CFG.device)
    logits = model(t).squeeze(0)
    probs  = torch.softmax(logits, dim=0).cpu().numpy()
    return int(probs.argmax()), probs


def run_full_inference(study_id, series_id,
                        s1_model, s2_model, s3_model,
                        cache: dict = None) -> dict:
    """
    Run the complete 3-stage pipeline on one (study_id, series_id) pair.

    Parameters
    ----------
    cache : if provided, volumes are read from RAM; otherwise read from disk.

    Returns
    -------
    dict { level: { 'pred_class': int, 'probs': np.ndarray } }
    """
    from stage1_slice_selector    import select_best_slice
    from stage2_keypoint_detector import predict_keypoints

    vol      = cache_get(cache, study_id, series_id) if cache \
               else load_dicom_volume(study_id, series_id)
    best_idx = select_best_slice(s1_model, study_id, series_id, cache=cache)
    slice_2d = vol[best_idx]

    keypoints = predict_keypoints(s2_model, slice_2d)

    results = {}
    for level, (cx, cy) in keypoints.items():
        patch = crop_roi_25d(vol, best_idx, cx, cy)
        pred_class, probs = predict_severity(s3_model, patch)
        results[level] = {"pred_class": pred_class, "probs": probs}

    return results


# ── 8.  Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    from stage1_slice_selector    import SliceSelector, select_best_slice
    from stage2_keypoint_detector import (KeypointUNet,
                                           build_best_slice_map)

    labels, coords, descs = load_dataframes()

    s1_ckpt = {"sag_t2": "stage1_sag_t2.pt", "sag_t1": "stage1_sag_t1.pt"}
    s2_ckpt = {"sag_t2": "stage2_sag_t2.pt", "sag_t1": "stage2_sag_t1.pt"}

    def _load_s1(key):
        p = CFG.output_dir / s1_ckpt[key]
        if not p.exists():
            print(f"  S1 checkpoint missing: {p}")
            return None
        m = SliceSelector(pretrained=False).to(CFG.device)
        m.load_state_dict(torch.load(p, map_location=CFG.device))
        m.eval()
        return m

    def _load_s2(key):
        p = CFG.output_dir / s2_ckpt[key]
        if not p.exists():
            print(f"  S2 checkpoint missing: {p}")
            return None
        m = KeypointUNet().to(CFG.device)
        m.load_state_dict(torch.load(p, map_location=CFG.device))
        m.eval()
        return m

    trained_s3 = {}

    for series_key in SAGITTAL_SERIES:
        series_name = SAGITTAL_SERIES[series_key]
        print(f"\n{'='*60}")
        print(f"  Stage 3  |  {series_key}  ({series_name})")
        print(f"  Conditions : {CONDITIONS_BY_SERIES[series_key]}")
        print(f"{'='*60}")

        s1_model = _load_s1(series_key)
        s2_model = _load_s2(series_key)

        coords_sub = filter_coords_by_series_key(coords, series_key)
        if coords_sub.empty:
            print("  No coordinate rows found — skipping.")
            continue

        # ── Pre-build helpers (done ONCE) ────────────────────
        print("  Building instance-index map …")
        inst_map = build_instance_index_map(coords_sub)

        print("  Building/restoring volume cache …")
        cache = build_volume_cache(coords_sub, series_key)

        pairs_df = (coords_sub[["study_id", "series_id"]]
                    .drop_duplicates()
                    .reset_index(drop=True))

        print("  Running Stage-1 inference (best-slice map) …")
        best_slice_map = build_best_slice_map(pairs_df, s1_model, cache)

        print("  Running Stage-2 inference (keypoint map) …")
        keypoint_map = build_keypoint_map(pairs_df, s2_model,
                                           cache, best_slice_map)

        print("  Building sample list …")
        all_samples = build_sample_list(
            coords_sub, labels,
            cache,
            best_slice_map,
            keypoint_map,
            series_key,
        )
        print(f"  Total samples : {len(all_samples)}")
        if not all_samples:
            print("  No samples built — check Stage 1 & 2 outputs.")
            continue

        study_ids = list({s["study_id"] for s in all_samples})
        rng       = np.random.default_rng(CFG.seed)
        rng.shuffle(study_ids)
        cut       = int(len(study_ids) * 0.8)
        train_ids = set(study_ids[:cut])
        val_ids   = set(study_ids[cut:])

        train_samples = [s for s in all_samples if s["study_id"] in train_ids]
        val_samples   = [s for s in all_samples if s["study_id"] in val_ids]

        print(f"  Train samples : {len(train_samples)}  "
              f"(studies {len(train_ids)})")
        print(f"  Val   samples : {len(val_samples)}  "
              f"(studies {len(val_ids)})")

        model = train_severity_classifier(
            train_samples, val_samples, cache, series_key
        )
        trained_s3[series_key] = model

    print("\n✓  Stage 3 complete.  Checkpoints in", CFG.output_dir)
