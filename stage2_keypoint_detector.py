# ============================================================
#  RSNA 2024 Lumbar Spine  |  stage2_keypoint_detector.py
#
#  Two models, one per sagittal series type:
#      sag_t2  →  Sagittal T2/STIR  (Spinal Canal Stenosis)
#      sag_t1  →  Sagittal T1       (Subarticular Stenosis)
#
#  KEY CHANGES vs original
#  -----------------------
#  1. `build_gt_heatmaps_for_sample` previously called
#     `load_dicom_volume` again just to get (H, W) shape.
#     It now receives the volume shape directly — no extra I/O.
#
#  2. `KeypointDataset.__getitem__` reads from `cache` instead of
#     calling `load_dicom_volume`.  A pre-computed `best_slice_map`
#     dict (study_id, series_id) → best_slice_idx is also injected
#     so Stage-1 inference is not re-run inside the DataLoader loop.
#
#  3. `train_keypoint_detector` accepts `cache` and `best_slice_map`
#     and passes them through.
#
#  4. num_workers set to 0 because all data is in RAM; forking would
#     copy the entire cache into each worker process.
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config_and_utils import (
    CFG, TRAIN_DIR, OUTPUT_DIR,
    SAGITTAL_SERIES, LEVELS, NUM_LEVELS,
    load_dataframes, filter_coords_by_series_key,
    load_dicom_volume, resize_slice,
    build_volume_cache, cache_get,
    build_instance_index_map, inst_to_idx,
    train_val_split,
    seed_everything,
)

seed_everything()

CKPT_NAMES = {
    "sag_t2": "stage2_sag_t2.pt",
    "sag_t1": "stage2_sag_t1.pt",
}


# ── 1.  Heatmap helpers ───────────────────────────────────────
def make_gaussian_heatmap(H: int, W: int,
                           cx: float, cy: float,
                           sigma: float = CFG.heatmap_sigma) -> np.ndarray:
    """
    Return a (H, W) float32 array with a 2-D Gaussian peak at (cx, cy).
    Values are in [0, 1].
    """
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    heatmap = np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


def build_gt_heatmaps_for_sample(
        study_id, series_id,
        best_slice_idx: int,
        orig_H: int,
        orig_W: int,
        coords_df: pd.DataFrame,
        inst_map:  dict,
        size: int = CFG.s2_img_size,
) -> np.ndarray:
    """
    Build ground-truth heatmaps for all 5 vertebral levels.

    Parameters
    ----------
    orig_H, orig_W : height and width of the original DICOM slice.
                     Passed in to avoid re-loading the volume just for shape.
    inst_map       : pre-built instance→index lookup dict.

    Returns
    -------
    np.ndarray  shape (NUM_LEVELS, size, size), dtype float32.
    """
    mask = (
        (coords_df["study_id"]  == study_id) &
        (coords_df["series_id"] == series_id)
    )
    sub = coords_df[mask]

    heatmaps = np.zeros((NUM_LEVELS, size, size), dtype=np.float32)

    for li, level in enumerate(LEVELS):
        level_rows = sub[
            sub["level"].str.lower().str.replace(" ", "_") == level
        ]
        if level_rows.empty:
            continue

        best_row  = None
        best_dist = float("inf")
        for _, row in level_rows.iterrows():
            idx  = inst_to_idx(inst_map, study_id, series_id,
                                int(row["instance_number"]))
            dist = abs(idx - best_slice_idx)
            if dist < best_dist:
                best_dist = dist
                best_row  = row

        if best_row is None:
            continue

        cx = best_row["x"] / orig_W * size
        cy = best_row["y"] / orig_H * size
        heatmaps[li] = make_gaussian_heatmap(size, size, cx, cy)

    return heatmaps


# ── 2.  Pre-compute best-slice map ────────────────────────────
def build_best_slice_map(pairs_df:    pd.DataFrame,
                          s1_model,
                          cache:       dict) -> dict:
    """
    Run Stage-1 inference on every (study_id, series_id) pair and
    store the result in a dict so __getitem__ never calls the model.

    Parameters
    ----------
    pairs_df : DataFrame with columns [study_id, series_id]
    s1_model : trained SliceSelector, or None (falls back to median)
    cache    : volume cache

    Returns
    -------
    dict { (study_id, series_id): best_slice_idx }
    """
    from stage1_slice_selector import select_best_slice

    best_map = {}
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df),
                       desc="Stage-1 inference (building best-slice map)"):
        study_id  = int(row["study_id"])
        series_id = int(row["series_id"])
        if s1_model is not None:
            idx = select_best_slice(s1_model, study_id, series_id,
                                    cache=cache)
        else:
            vol = cache_get(cache, study_id, series_id)
            idx = vol.shape[0] // 2
        best_map[(study_id, series_id)] = idx
    return best_map


# ── 3.  Dataset ───────────────────────────────────────────────
class KeypointDataset(Dataset):
    """
    One sample:
        image    : (3, s2_img_size, s2_img_size)              float32
        heatmaps : (NUM_LEVELS, s2_img_size, s2_img_size)     float32

    All heavy data comes from `cache` and `best_slice_map` —
    no DICOM I/O or Stage-1 model calls inside __getitem__.
    """

    def __init__(self,
                 study_series_df:  pd.DataFrame,
                 coords_df:        pd.DataFrame,
                 cache:            dict,
                 best_slice_map:   dict,
                 inst_map:         dict,
                 augment:          bool = False):
        self.coords_df      = coords_df
        self.cache          = cache
        self.best_slice_map = best_slice_map
        self.inst_map       = inst_map
        self.augment        = augment

        self.pairs = (
            study_series_df[["study_id", "series_id"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        row       = self.pairs.iloc[idx]
        study_id  = int(row["study_id"])
        series_id = int(row["series_id"])

        vol      = cache_get(self.cache, study_id, series_id)  # (D, H, W)
        _, H, W  = vol.shape
        best_idx = self.best_slice_map.get((study_id, series_id),
                                            vol.shape[0] // 2)

        # ── Image tensor ────────────────────────────────────
        img   = resize_slice(vol[best_idx], CFG.s2_img_size)
        img3  = np.stack([img, img, img], axis=0)
        img_t = torch.tensor(img3, dtype=torch.float32)

        # ── Ground-truth heatmaps ────────────────────────────
        hm = build_gt_heatmaps_for_sample(
            study_id, series_id,
            best_idx,
            orig_H=H, orig_W=W,
            coords_df=self.coords_df,
            inst_map=self.inst_map,
            size=CFG.s2_img_size,
        )
        hm_t = torch.tensor(hm, dtype=torch.float32)

        # ── Augmentation (same flip applied to image and heatmaps) ──
        if self.augment and torch.rand(1).item() > 0.5:
            img_t = torch.flip(img_t, dims=[2])
            hm_t  = torch.flip(hm_t,  dims=[2])

        return img_t, hm_t


# ── 4.  Model: lightweight U-Net ─────────────────────────────
class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class KeypointUNet(nn.Module):
    """
    Lightweight U-Net (≈4 M parameters) that maps a
    (3, H, W) sagittal slice to (NUM_LEVELS, H, W) heatmaps.

    Encoder : 4 stages, each doubles channels and halves spatial size.
    Bottleneck : one DoubleConv at the deepest level.
    Decoder : 4 stages of ConvTranspose2d + skip-concat + DoubleConv.
    Head    : 1×1 conv + Sigmoid → output in [0, 1].
    """

    def __init__(self,
                 in_channels:  int  = 3,
                 out_channels: int  = NUM_LEVELS,
                 features:     list = None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.enc1 = _DoubleConv(in_channels,  features[0])
        self.enc2 = _DoubleConv(features[0],  features[1])
        self.enc3 = _DoubleConv(features[1],  features[2])
        self.enc4 = _DoubleConv(features[2],  features[3])
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _DoubleConv(features[3], features[3] * 2)

        self.up4  = nn.ConvTranspose2d(features[3] * 2, features[3], 2, stride=2)
        self.dec4 = _DoubleConv(features[3] * 2, features[3])

        self.up3  = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.dec3 = _DoubleConv(features[2] * 2, features[2])

        self.up2  = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.dec2 = _DoubleConv(features[1] * 2, features[1])

        self.up1  = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.dec1 = _DoubleConv(features[0] * 2, features[0])

        self.head = nn.Sequential(
            nn.Conv2d(features[0], out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d  = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d  = self.dec3(torch.cat([self.up3(d),  e3], dim=1))
        d  = self.dec2(torch.cat([self.up2(d),  e2], dim=1))
        d  = self.dec1(torch.cat([self.up1(d),  e1], dim=1))
        return self.head(d)   # (B, NUM_LEVELS, H, W)


# ── 5.  Training ──────────────────────────────────────────────
def train_keypoint_detector(
        train_pairs:    pd.DataFrame,
        val_pairs:      pd.DataFrame,
        coords_df:      pd.DataFrame,
        cache:          dict,
        best_slice_map: dict,
        inst_map:       dict,
        series_key:     str,
) -> KeypointUNet:
    """
    Train a KeypointUNet for one series type.

    Parameters
    ----------
    train_pairs     : DataFrame [study_id, series_id] — train set
    val_pairs       : same structure — validation set
    coords_df       : coordinate rows for this series key
    cache           : volume cache from build_volume_cache()
    best_slice_map  : pre-computed { (study_id, series_id): slice_idx }
    inst_map        : pre-built instance→index lookup
    series_key      : 'sag_t2' or 'sag_t1'
    """
    tag       = f"S2/{series_key}"
    save_path = CFG.output_dir / CKPT_NAMES[series_key]

    train_ds = KeypointDataset(train_pairs, coords_df, cache,
                                best_slice_map, inst_map, augment=True)
    val_ds   = KeypointDataset(val_pairs,   coords_df, cache,
                                best_slice_map, inst_map, augment=False)

    # num_workers=0: data is in RAM, no benefit from multiprocessing here
    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG.batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)

    model     = KeypointUNet().to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=CFG.epochs_s2)
    loss_fn   = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, CFG.epochs_s2 + 1):
        model.train()
        train_loss = 0.0
        for imgs, heatmaps in tqdm(train_dl,
                                    desc=f"[{tag}] epoch {epoch:02d} train",
                                    leave=False):
            imgs, heatmaps = imgs.to(CFG.device), heatmaps.to(CFG.device)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), heatmaps)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps in val_dl:
                imgs, heatmaps = imgs.to(CFG.device), heatmaps.to(CFG.device)
                val_loss += loss_fn(model(imgs), heatmaps).item()

        scheduler.step()

        avg_tr = train_loss / max(len(train_dl), 1)
        avg_vl = val_loss   / max(len(val_dl),   1)
        print(f"  [{tag}] epoch {epoch:02d} | "
              f"train {avg_tr:.6f}  val {avg_vl:.6f}")

        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ checkpoint saved  (val={best_val_loss:.6f})")

    model.load_state_dict(torch.load(save_path, map_location=CFG.device))
    model.eval()
    return model


# ── 6.  Inference ─────────────────────────────────────────────
@torch.no_grad()
def predict_keypoints(model: KeypointUNet,
                       slice_img: np.ndarray) -> dict:
    """
    Run the keypoint model on a single 2-D slice and return the
    predicted centre coordinate for each vertebral level.

    Parameters
    ----------
    slice_img : 2-D float32 numpy array (H, W) in [0, 1]

    Returns
    -------
    dict  { level_str: (x_px, y_px) }
    Coordinates are in original-slice pixel space.
    """
    model.eval()
    H_orig, W_orig = slice_img.shape
    S = CFG.s2_img_size

    img   = resize_slice(slice_img, S)
    img3  = np.stack([img, img, img], axis=0)
    img_t = torch.tensor(img3, dtype=torch.float32).unsqueeze(0).to(CFG.device)

    pred_hm = model(img_t).squeeze(0).cpu().numpy()   # (NUM_LEVELS, S, S)

    coords_out = {}
    for li, level in enumerate(LEVELS):
        flat   = pred_hm[li].argmax()
        cy_s   = int(flat // S)
        cx_s   = int(flat  % S)
        cx_orig = cx_s / S * W_orig
        cy_orig = cy_s / S * H_orig
        coords_out[level] = (float(cx_orig), float(cy_orig))

    return coords_out


# ── 7.  Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    from stage1_slice_selector import SliceSelector, select_best_slice

    labels, coords, descs = load_dataframes()

    s1_ckpt_names = {
        "sag_t2": "stage1_sag_t2.pt",
        "sag_t1": "stage1_sag_t1.pt",
    }

    trained_s2 = {}

    for series_key in SAGITTAL_SERIES:
        series_name = SAGITTAL_SERIES[series_key]
        print(f"\n{'='*60}")
        print(f"  Stage 2  |  {series_key}  ({series_name})")
        print(f"{'='*60}")

        # ── Load Stage-1 model ──────────────────────────────
        s1_path = CFG.output_dir / s1_ckpt_names[series_key]
        if s1_path.exists():
            s1_model = SliceSelector(pretrained=False).to(CFG.device)
            s1_model.load_state_dict(
                torch.load(s1_path, map_location=CFG.device)
            )
            s1_model.eval()
            print(f"  Loaded S1 checkpoint : {s1_path}")
        else:
            print(f"  S1 checkpoint not found ({s1_path})")
            print("  Falling back to median slice.")
            s1_model = None

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

        print("  Running Stage-1 inference to build best-slice map …")
        best_slice_map = build_best_slice_map(pairs_df, s1_model, cache)

        train_pairs, val_pairs = train_val_split(pairs_df)
        print(f"  Train series : {len(train_pairs)}")
        print(f"  Val   series : {len(val_pairs)}")

        model = train_keypoint_detector(
            train_pairs, val_pairs,
            coords_sub,
            cache,
            best_slice_map,
            inst_map,
            series_key,
        )
        trained_s2[series_key] = model

    print("\n✓  Stage 2 complete.  Checkpoints in", CFG.output_dir)
