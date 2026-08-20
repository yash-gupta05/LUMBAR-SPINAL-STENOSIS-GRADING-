# ============================================================
#  RSNA 2024 Lumbar Spine  |  stage1_slice_selector.py
#
#  Two models, one per sagittal series type:
#      sag_t2  →  Sagittal T2/STIR  (Spinal Canal Stenosis)
#      sag_t1  →  Sagittal T1       (Subarticular Stenosis)
#
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config_and_utils import (
    CFG, TRAIN_DIR, OUTPUT_DIR,
    SAGITTAL_SERIES,
    load_dataframes, filter_coords_by_series_key,
    load_dicom_volume, resize_slice,
    build_volume_cache, cache_get,
    build_instance_index_map, inst_to_idx,
    train_val_split,
    seed_everything,
)

seed_everything()

CKPT_NAMES = {
    "sag_t2": "stage1_sag_t2.pt",
    "sag_t1": "stage1_sag_t1.pt",
}


# ── 1.  Build ground-truth slice table ───────────────────────
def build_slice_gt(coords_df: pd.DataFrame,
                   inst_map:  dict,
                   base_dir:  Path = TRAIN_DIR) -> pd.DataFrame:
    """
    For every (study_id, series_id) in coords_df, find the 0-based
    index of each annotated DICOM instance.

    Parameters
    ----------
    coords_df : coordinate rows for one series key
    inst_map  : pre-built { (study_id, series_id, instance_number): slice_idx }
                from build_instance_index_map()

    Returns
    -------
    DataFrame with columns:
        study_id | series_id | gt_slice_idx | total_slices
    """
    records = []
    grouped = coords_df.groupby(["study_id", "series_id"])

    for (study_id, series_id), grp in tqdm(
            grouped, desc="Building slice GT", total=len(grouped)):

        folder = base_dir / str(study_id) / str(series_id)
        if not folder.exists():
            continue

        # Total slices = number of entries in inst_map for this series
        total_slices = sum(
            1 for (sid, serid, _) in inst_map
            if sid == int(study_id) and serid == int(series_id)
        )
        if total_slices == 0:
            continue

        seen_instances = set()
        for _, row in grp.iterrows():
            inst = int(row["instance_number"])
            if inst in seen_instances:
                continue
            seen_instances.add(inst)

            gt_idx = inst_to_idx(inst_map, study_id, series_id, inst)
            if gt_idx < 0:
                continue

            records.append({
                "study_id":     int(study_id),
                "series_id":    int(series_id),
                "gt_slice_idx": gt_idx,
                "total_slices": total_slices,
            })

    return pd.DataFrame(records)


# ── 2.  Dataset ───────────────────────────────────────────────
class SliceSelectorDataset(Dataset):
    """
    One sample = one 2-D slice image + a binary label.

    Positives  (label = 1.0) : radiologist-annotated slices.
    Negatives  (label = 0.0) : randomly sampled other slices
                               from the same series.

    All volume data is read from `cache` (in-RAM float16 arrays)
    — no DICOM I/O happens inside __getitem__.
    """

    def __init__(self,
                 slice_gt_df: pd.DataFrame,
                 cache: dict,
                 augment: bool = False,
                 neg_per_pos: int = 3):
        self.cache   = cache
        self.augment = augment
        # Each entry: (study_id, series_id, slice_idx, label)
        self.samples: list = []
        self._build(slice_gt_df, neg_per_pos)

        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]) if augment else None

    def _build(self, df: pd.DataFrame, neg_per_pos: int) -> None:
        rng = np.random.default_rng(CFG.seed)
        for _, row in df.iterrows():
            total = int(row["total_slices"])
            gt    = int(row["gt_slice_idx"])
            key   = (int(row["study_id"]), int(row["series_id"]))

            self.samples.append((*key, gt, 1.0))

            pool = [i for i in range(total) if i != gt]
            if pool:
                chosen = rng.choice(
                    pool,
                    size=min(neg_per_pos, len(pool)),
                    replace=False,
                )
                for n in chosen:
                    self.samples.append((*key, int(n), 0.0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        study_id, series_id, slice_idx, label = self.samples[idx]

        # Read from in-RAM cache (float16 → float32 cast is cheap)
        vol = cache_get(self.cache, study_id, series_id)  # (D, H, W) f32
        img = vol[slice_idx]                               # (H, W)
        img = resize_slice(img, CFG.s1_img_size)           # (S, S)
        img = np.stack([img, img, img], axis=0)            # (3, S, S)

        img_t = torch.tensor(img, dtype=torch.float32)
        if self.aug is not None:
            img_t = self.aug(img_t)

        return img_t, torch.tensor(label, dtype=torch.float32)


# ── 3.  Model ─────────────────────────────────────────────────
class SliceSelector(nn.Module):
    """
    EfficientNet-B0 with a single sigmoid output.
    One checkpoint per series type.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = (models.EfficientNet_B0_Weights.IMAGENET1K_V1
                   if pretrained else None)
        base = models.efficientnet_b0(weights=weights)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )
        self.net = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)   # (B,)


# ── 4.  Training ──────────────────────────────────────────────
def train_slice_selector(
        train_df:   pd.DataFrame,
        val_df:     pd.DataFrame,
        cache:      dict,
        series_key: str,
) -> "SliceSelector":
    """
    Train a SliceSelector for one series type and return the model
    loaded with the best (lowest validation loss) weights.

    Parameters
    ----------
    train_df   : output of build_slice_gt(), training portion
    val_df     : output of build_slice_gt(), validation portion
    cache      : volume cache from build_volume_cache()
    series_key : 'sag_t2' or 'sag_t1'
    """
    tag       = f"S1/{series_key}"
    save_path = CFG.output_dir / CKPT_NAMES[series_key]

    # num_workers=0 is fine now: all data is already in RAM, multiprocess
    # forking would just duplicate the cache in each worker unnecessarily.
    train_ds = SliceSelectorDataset(train_df, cache, augment=True)
    val_ds   = SliceSelectorDataset(val_df,   cache, augment=False)

    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG.batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)

    model     = SliceSelector(pretrained=True).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=CFG.epochs_s1)
    loss_fn   = nn.BCELoss()

    best_val_loss = float("inf")

    for epoch in range(1, CFG.epochs_s1 + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_dl,
                                  desc=f"[{tag}] epoch {epoch:02d} train",
                                  leave=False):
            imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)
                val_loss += loss_fn(model(imgs), labels).item()

        scheduler.step()

        avg_tr = train_loss / max(len(train_dl), 1)
        avg_vl = val_loss   / max(len(val_dl),   1)
        print(f"  [{tag}] epoch {epoch:02d} | "
              f"train {avg_tr:.4f}  val {avg_vl:.4f}")

        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ checkpoint saved  (val={best_val_loss:.4f})")

    model.load_state_dict(torch.load(save_path, map_location=CFG.device))
    model.eval()
    return model


# ── 5.  Inference ─────────────────────────────────────────────
@torch.no_grad()
def select_best_slice(model: SliceSelector,
                      study_id, series_id,
                      cache: dict = None) -> int:
    """
    Score every slice in the series and return the index of the
    highest-scoring one.

    Parameters
    ----------
    cache : if provided, volumes are read from RAM; otherwise the
            DICOM folder is read from disk (slow fallback for test set).

    Returns
    -------
    int  0-based slice index
    """
    model.eval()

    if cache is not None:
        vol = cache_get(cache, study_id, series_id)
    else:
        vol = load_dicom_volume(study_id, series_id)

    scores = []
    for i in range(vol.shape[0]):
        img = resize_slice(vol[i], CFG.s1_img_size)
        img = np.stack([img, img, img], axis=0)
        t   = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(CFG.device)
        scores.append(model(t).item())

    return int(np.argmax(scores))


# ── 6.  Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    labels, coords, descs = load_dataframes()

    trained = {}   # series_key → model

    for series_key in SAGITTAL_SERIES:
        series_name = SAGITTAL_SERIES[series_key]
        print(f"\n{'='*60}")
        print(f"  Stage 1  |  {series_key}  ({series_name})")
        print(f"{'='*60}")

        coords_sub = filter_coords_by_series_key(coords, series_key)
        if coords_sub.empty:
            print("  No coordinate rows found — skipping.")
            continue

        # ── Pre-build helpers (done ONCE per series key) ──────
        print("  Building instance-index map …")
        inst_map = build_instance_index_map(coords_sub)

        print("  Building volume cache …")
        cache = build_volume_cache(coords_sub, series_key)

        print("  Building slice GT table …")
        slice_gt = build_slice_gt(coords_sub, inst_map)
        print(f"  GT rows : {len(slice_gt)}")

        train_df, val_df = train_val_split(slice_gt)
        print(f"  Train studies : {train_df['study_id'].nunique()}")
        print(f"  Val   studies : {val_df['study_id'].nunique()}")

        model = train_slice_selector(train_df, val_df, cache, series_key)
        trained[series_key] = model

    print("\n✓  Stage 1 complete.  Checkpoints in", CFG.output_dir)
