# ============================================================
#  RSNA 2024 Lumbar Spine  |  config_and_utils.py
#
#  Shared constants, paths, hyperparameters, and helper
#  functions used across all three stages.

import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from pathlib import Path
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = Path("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification")
TRAIN_DIR  = BASE_DIR / "train_images"
TEST_DIR   = BASE_DIR / "test_images"
TRAIN_CSV  = BASE_DIR / "train.csv"
COORDS_CSV = BASE_DIR / "train_label_coordinates.csv"
DESC_CSV   = BASE_DIR / "train_series_descriptions.csv"
OUTPUT_DIR = Path("/kaggle/working")

# ── Severity ──────────────────────────────────────────────────
SEVERITY_MAP         = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
SEV_LABELS           = ["Normal/Mild", "Moderate", "Severe"]
NUM_SEVERITY_CLASSES = 3

# ── Sagittal series types ─────────────────────────────────────
SAGITTAL_SERIES = {
    "sag_t2": "Sagittal T2/STIR",
    "sag_t1": "Sagittal T1",
}

CONDITIONS_BY_SERIES = {
    "sag_t2": ["spinal_canal_stenosis"],
    "sag_t1": ["left_subarticular_stenosis",
                "right_subarticular_stenosis"],
}

ALL_SAGITTAL_CONDITIONS = [
    "spinal_canal_stenosis",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]

LEVELS     = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
NUM_LEVELS = len(LEVELS)   # 5


# ── Hyperparameters ───────────────────────────────────────────
class CFG:
    seed = 42

    # Stage 1 — slice selector
    s1_img_size = 224

    # Stage 2 — keypoint / heatmap detector
    s2_img_size   = 256
    heatmap_sigma = 8

    # Stage 3 — severity classifier
    roi_pad = 32
    n_extra = 1              # 2.5D stack: central ± n_extra slices → 3 ch

    # Shared training
    batch_size  = 16
    num_workers = 2
    lr          = 1e-4

    epochs_s1 = 10
    epochs_s2 = 15
    epochs_s3 = 20

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = OUTPUT_DIR

    # ── Cache settings ────────────────────────────────────────
    # float16 halves RAM vs float32.
    # Rough worst-case estimate for the RSNA-2024 training set:
    #   ~1975 studies × ~3 series × ~20 slices × 512×512 px × 2 bytes (f16)
    #   ≈ 1975 × 3 × 20 × 512 × 512 × 2  ≈ 31 GB  (absolute worst case)
    # In practice most series have fewer slices and smaller H×W,
    # so the actual peak is typically 10–15 GB.  Kaggle gives 19.5 GB,
    # so loading BOTH sagittal series types together is borderline.
    #
    # Strategy: cache one series key at a time (sag_t2, then sag_t1).
    # Each partial cache is saved to disk so it can be reloaded cheaply
    # if you need to restart.  Call build_volume_cache(series_key=...).
    cache_dtype   = np.float16   # halves memory; cast back to f32 on access
    cache_dir     = OUTPUT_DIR / "vol_cache"   # where .npz files land


# ── Reproducibility ───────────────────────────────────────────
def seed_everything(seed: int = CFG.seed) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything()
print(f"Device : {CFG.device}")


# ── DICOM / volume helpers ────────────────────────────────────
def load_dicom_volume(study_id, series_id,
                      base_dir: Path = TRAIN_DIR) -> np.ndarray:
    """
    Load all DICOM slices for one series.

    Returns
    -------
    np.ndarray  shape (D, H, W), dtype float32, values in [0, 1].
    Slices are in ascending InstanceNumber order.
    """
    folder = base_dir / str(study_id) / str(series_id)
    dcm_files = sorted(
        folder.glob("*.dcm"),
        key=lambda f: int(
            pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber
        ),
    )
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {folder}")

    slices = []
    for f in dcm_files:
        ds  = pydicom.dcmread(f)
        img = ds.pixel_array.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        slices.append(img)
    return np.stack(slices, axis=0)   # (D, H, W)  float32


def resize_slice(img_2d: np.ndarray, size: int) -> np.ndarray:
    """Resize a single 2-D float slice to (size × size)."""
    return cv2.resize(img_2d, (size, size),
                      interpolation=cv2.INTER_LINEAR)


def get_slice_index_by_instance(study_id, series_id,
                                 instance_number: int,
                                 base_dir: Path = TRAIN_DIR) -> int:
    """
    Return the 0-based position of the DICOM file whose InstanceNumber
    equals `instance_number` within the sorted series.
    Returns -1 if not found.
    """
    folder = base_dir / str(study_id) / str(series_id)
    dcm_files = sorted(
        folder.glob("*.dcm"),
        key=lambda f: int(
            pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber
        ),
    )
    for idx, f in enumerate(dcm_files):
        if int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber) \
                == int(instance_number):
            return idx
    return -1


# ── Volume Cache ──────────────────────────────────────────────
# The cache is a plain Python dict:
#   { (study_id, series_id): np.ndarray (D, H, W) float16 }
#
# It is built once before training, optionally saved to disk, and
# then passed into every Dataset and helper function.  Inside
# __getitem__ the float16 array is cast to float32 on the fly;
# that cast is cheap compared to DICOM I/O.

def _cache_npz_path(series_key: str) -> Path:
    CFG.cache_dir.mkdir(parents=True, exist_ok=True)
    return CFG.cache_dir / f"vol_cache_{series_key}.npz"


def build_volume_cache(coords_df: pd.DataFrame,
                       series_key: str,
                       base_dir: Path = TRAIN_DIR,
                       force_rebuild: bool = False) -> dict:
    """
    Pre-load every DICOM volume needed for `series_key` into RAM.

    The result is stored as float16 to halve memory usage.

    If a matching .npz file already exists in CFG.cache_dir and
    force_rebuild is False, the cache is reloaded from disk instead
    of re-reading all DICOMs — useful after a kernel restart.

    Parameters
    ----------
    coords_df    : coordinate DataFrame already filtered for series_key
    series_key   : 'sag_t2' or 'sag_t1'
    base_dir     : root directory containing study folders
    force_rebuild: ignore any existing .npz and re-read from DICOM

    Returns
    -------
    dict  { (study_id, series_id) : np.ndarray float16 (D, H, W) }
    """
    npz_path = _cache_npz_path(series_key)

    # ── Try to restore from disk ──────────────────────────────
    if npz_path.exists() and not force_rebuild:
        print(f"[Cache] Loading {series_key} cache from {npz_path} …")
        data = np.load(npz_path, allow_pickle=True)
        cache = {}
        # Keys are stored as "studyid_seriesid" strings
        for k in data.files:
            sid, serid = k.split("__")
            cache[(int(sid), int(serid))] = data[k]
        print(f"[Cache] Restored {len(cache)} volumes from disk.")
        return cache

    # ── Build from DICOM ──────────────────────────────────────
    pairs = (coords_df[["study_id", "series_id"]]
             .drop_duplicates()
             .reset_index(drop=True))

    cache    = {}
    npz_data = {}

    total_bytes = 0
    print(f"[Cache] Building {series_key} volume cache "
          f"({len(pairs)} series) …")

    for _, row in tqdm(pairs.iterrows(), total=len(pairs),
                       desc=f"Loading {series_key} DICOMs"):
        study_id  = int(row["study_id"])
        series_id = int(row["series_id"])
        folder    = base_dir / str(study_id) / str(series_id)
        if not folder.exists():
            continue
        try:
            vol = load_dicom_volume(study_id, series_id,
                                    base_dir=base_dir)          # float32
            vol16 = vol.astype(np.float16)                      # halve size
            cache[(study_id, series_id)] = vol16
            npz_data[f"{study_id}__{series_id}"] = vol16
            total_bytes += vol16.nbytes
        except Exception as exc:
            print(f"  [Cache] Skipping ({study_id}, {series_id}): {exc}")

    gb = total_bytes / 1024 ** 3
    print(f"[Cache] {len(cache)} volumes loaded — "
          f"{gb:.2f} GB in RAM (float16).")

    # ── Persist to disk ───────────────────────────────────────
    print(f"[Cache] Saving to {npz_path} …")
    np.savez_compressed(npz_path, **npz_data)
    print(f"[Cache] Saved.")

    return cache


def cache_get(cache: dict, study_id, series_id) -> np.ndarray:
    """
    Retrieve a volume from the cache as float32.

    Falls back to reading from disk if the key is missing (e.g., a
    test-set volume that was not pre-cached).
    """
    key = (int(study_id), int(series_id))
    if key in cache:
        return cache[key].astype(np.float32)
    # Fallback — should not happen during training if cache was built
    print(f"[Cache] MISS for {key} — reading from DICOM (slow path)")
    return load_dicom_volume(study_id, series_id)


# ── InstanceNumber → slice-index lookup ──────────────────────
# Pre-building this mapping avoids re-scanning folder on every call.

def build_instance_index_map(coords_df: pd.DataFrame,
                              base_dir: Path = TRAIN_DIR) -> dict:
    """
    Build a dict   { (study_id, series_id, instance_number): slice_idx }
    for every (study_id, series_id) pair that appears in coords_df.

    This is computed once and then passed to helpers that previously
    called get_slice_index_by_instance() in a hot loop.
    """
    pairs = (coords_df[["study_id", "series_id"]]
             .drop_duplicates()
             .reset_index(drop=True))

    inst_map = {}
    for _, row in tqdm(pairs.iterrows(), total=len(pairs),
                       desc="Building instance-index map"):
        study_id  = int(row["study_id"])
        series_id = int(row["series_id"])
        folder    = base_dir / str(study_id) / str(series_id)
        if not folder.exists():
            continue
        dcm_files = sorted(
            folder.glob("*.dcm"),
            key=lambda f: int(
                pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber
            ),
        )
        for idx, f in enumerate(dcm_files):
            inst = int(
                pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber
            )
            inst_map[(study_id, series_id, inst)] = idx
    return inst_map


def inst_to_idx(inst_map: dict,
                study_id, series_id, instance_number: int) -> int:
    """Look up slice index from the pre-built map. Returns -1 if missing."""
    return inst_map.get((int(study_id), int(series_id),
                         int(instance_number)), -1)


# ── CSV helpers ───────────────────────────────────────────────
def load_dataframes():
    """
    Load the three core CSVs.

    Returns
    -------
    labels_df : one row per study, severity columns for every condition/level
    coords_df : one row per annotation point, with series_description merged in
    descs_df  : mapping of (study_id, series_id) → series_description
    """
    labels = pd.read_csv(TRAIN_CSV)
    coords = pd.read_csv(COORDS_CSV)
    descs  = pd.read_csv(DESC_CSV)
    coords = coords.merge(descs, on=["study_id", "series_id"], how="left")
    return labels, coords, descs


def filter_coords_by_series_key(coords_df: pd.DataFrame,
                                 series_key: str) -> pd.DataFrame:
    pattern = SAGITTAL_SERIES[series_key]
    mask = coords_df["series_description"].str.contains(
        pattern, case=False, na=False
    )
    return coords_df[mask].copy().reset_index(drop=True)


def get_tidy_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot labels_df into one row per (study_id, condition, level)
    with a numeric 'severity' column (0 / 1 / 2).
    Rows with unknown severity are kept with severity = -1.
    Only sagittal conditions are included.
    """
    rows = []
    for _, row in labels_df.iterrows():
        for cond in ALL_SAGITTAL_CONDITIONS:
            for lvl in LEVELS:
                col = f"{cond}_{lvl}"
                if col in labels_df.columns:
                    rows.append({
                        "study_id": row["study_id"],
                        "condition": cond,
                        "level":    lvl,
                        "severity": SEVERITY_MAP.get(row[col], -1),
                    })
    return pd.DataFrame(rows)


def train_val_split(df: pd.DataFrame,
                    val_frac: float = 0.2,
                    seed: int = CFG.seed):
    """
    Split df into (train_df, val_df) stratified by study_id so no
    study appears in both sets.
    """
    study_ids = df["study_id"].unique().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(study_ids)
    cut = int(len(study_ids) * (1 - val_frac))
    return (
        df[df["study_id"].isin(study_ids[:cut])].reset_index(drop=True),
        df[df["study_id"].isin(study_ids[cut:])].reset_index(drop=True),
    )


# ── Quick sanity check ────────────────────────────────────────
if __name__ == "__main__":
    labels, coords, descs = load_dataframes()
    print(f"labels : {labels.shape}")
    print(f"coords : {coords.shape}")
    print("\nSeries description counts:")
    print(descs["series_description"].value_counts().to_string())
    print()
    for key in SAGITTAL_SERIES:
        sub = filter_coords_by_series_key(coords, key)
        print(f"  {key!r} ({SAGITTAL_SERIES[key]!r}) → {len(sub)} coord rows")
