# ============================================================
#  RSNA 2024 Lumbar Spine  |  inference.py
#
#  Runs the complete three-stage pipeline on a single study:
#      Stage 1 → pick the most informative sagittal slice
#      Stage 2 → locate the 5 vertebral-level keypoints on it
#      Stage 3 → classify severity at each keypoint's ROI
#
#  Loads pre-trained checkpoints for all three stages (per series
#  type) and prints/saves structured per-level severity predictions.
#
#  Usage
#  -----
#      python inference.py --study_id 12345678
#      python inference.py --study_id 12345678 --series_key sag_t2
#      python inference.py --study_id 12345678 --out_json results.json
#
#  Note
#  -----
#  - Looks up the series_id for the requested study from
#    train_series_descriptions.csv (or test equivalent). Pass
#    --series_id directly to skip this lookup.
# ============================================================

import argparse
import json
import sys

import torch

from config_and_utils import (
    CFG, SAGITTAL_SERIES, LEVELS, SEV_LABELS,
    load_dataframes, filter_coords_by_series_key,
)


CKPT_NAMES = {
    "s1": {"sag_t2": "stage1_sag_t2.pt", "sag_t1": "stage1_sag_t1.pt"},
    "s2": {"sag_t2": "stage2_sag_t2.pt", "sag_t1": "stage2_sag_t1.pt"},
    "s3": {"sag_t2": "stage3_sag_t2.pt", "sag_t1": "stage3_sag_t1.pt"},
}


def _load_checkpoint(model, ckpt_path):
    if not ckpt_path.exists():
        return None
    model.load_state_dict(torch.load(ckpt_path, map_location=CFG.device))
    model.eval()
    return model


def load_stage_models(series_key: str):
    """
    Load the Stage 1, 2, 3 checkpoints for one series key.
    Returns (s1_model, s2_model, s3_model); any of them may be None
    if the corresponding checkpoint file is missing.
    """
    from stage1_slice_selector import SliceSelector
    from stage2_keypoint_detector import KeypointUNet
    from stage3_severity_classifier import SeverityClassifier

    s1 = _load_checkpoint(
        SliceSelector(pretrained=False).to(CFG.device),
        CFG.output_dir / CKPT_NAMES["s1"][series_key],
    )
    s2 = _load_checkpoint(
        KeypointUNet().to(CFG.device),
        CFG.output_dir / CKPT_NAMES["s2"][series_key],
    )
    s3 = _load_checkpoint(
        SeverityClassifier(pretrained=False).to(CFG.device),
        CFG.output_dir / CKPT_NAMES["s3"][series_key],
    )
    return s1, s2, s3


def find_series_id(descs_df, study_id: int, series_key: str):
    """
    Look up the series_id for a given study and series type
    (e.g. 'sag_t2' -> the Sagittal T2/STIR series_id for that study)
    using the series-description table. Returns None if not found.
    """
    pattern = SAGITTAL_SERIES[series_key]
    sub = descs_df[
        (descs_df["study_id"] == study_id) &
        (descs_df["series_description"].str.contains(
            pattern, case=False, na=False))
    ]
    if sub.empty:
        return None
    return int(sub.iloc[0]["series_id"])


def run_for_series_key(study_id: int, series_id: int, series_key: str) -> dict:
    """
    Run the full 3-stage pipeline for one (study_id, series_id) pair
    and return {level: {"severity_label": str, "severity_idx": int,
                         "probs": [p0, p1, p2]}}.
    """
    from stage3_severity_classifier import run_full_inference

    s1_model, s2_model, s3_model = load_stage_models(series_key)
    missing = [name for name, m in
               [("Stage 1", s1_model), ("Stage 2", s2_model), ("Stage 3", s3_model)]
               if m is None]
    if missing:
        print(f"  ✗ Missing checkpoint(s) for '{series_key}': {', '.join(missing)}. "
              f"Skipping this series. Train it first with train.py.")
        return {}

    print(f"  Running pipeline for series_id={series_id} ({series_key}) …")
    # cache=None → volumes are read directly from disk (single-study inference,
    # not worth building a full in-RAM cache for one study)
    raw = run_full_inference(study_id, series_id, s1_model, s2_model, s3_model,
                              cache=None)

    results = {}
    for level in LEVELS:
        if level not in raw:
            continue
        pred = raw[level]
        results[level] = {
            "severity_label": SEV_LABELS[pred["pred_class"]],
            "severity_idx":   pred["pred_class"],
            "probs":          [round(float(p), 4) for p in pred["probs"]],
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end lumbar SCS grading pipeline on one study."
    )
    parser.add_argument("--study_id", type=int, required=True,
                        help="study_id to run inference on.")
    parser.add_argument("--series_id", type=int, default=None,
                        help="series_id to use directly, skipping the lookup "
                             "in train_series_descriptions.csv. Only valid "
                             "when a single --series_key is requested.")
    parser.add_argument("--series_key", type=str, default="sag_t2",
                        choices=list(SAGITTAL_SERIES.keys()) + ["all"],
                        help="Which series branch(es) to run. Default: sag_t2 "
                             "(Spinal Canal Stenosis — the scope of this project). "
                             "'all' runs every series key with available checkpoints.")
    parser.add_argument("--out_json", type=str, default=None,
                        help="Optional path to save results as JSON.")
    args = parser.parse_args()

    series_keys = (list(SAGITTAL_SERIES.keys()) if args.series_key == "all"
                    else [args.series_key])

    if args.series_id is not None and len(series_keys) > 1:
        print("✗ --series_id can only be used with a single --series_key.")
        sys.exit(1)

    print(f"Study ID: {args.study_id}")

    _, coords, descs = load_dataframes()

    all_results = {}
    for series_key in series_keys:
        series_id = args.series_id
        if series_id is None:
            series_id = find_series_id(descs, args.study_id, series_key)
            if series_id is None:
                print(f"  No {series_key} series found for study "
                      f"{args.study_id} — skipping.")
                continue

        result = run_for_series_key(args.study_id, series_id, series_key)
        if result:
            all_results[series_key] = {
                "series_id": series_id,
                "levels": result,
            }

    if not all_results:
        print("\nNo predictions produced. Check that checkpoints exist and "
              "the study/series was found.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Results for study {args.study_id}")
    print(f"{'=' * 60}")
    for series_key, payload in all_results.items():
        print(f"\n[{series_key}]  series_id={payload['series_id']}")
        for level, pred in payload["levels"].items():
            print(f"  {level:8s} → {pred['severity_label']:12s} "
                  f"(probs: Normal/Mild={pred['probs'][0]:.2f}  "
                  f"Moderate={pred['probs'][1]:.2f}  "
                  f"Severe={pred['probs'][2]:.2f})")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to {args.out_json}")


if __name__ == "__main__":
    main()
