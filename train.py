# ============================================================
#  RSNA 2024 Lumbar Spine  |  train.py
#
#  Orchestrates the full three-stage training pipeline:
#      Stage 1 → stage1_slice_selector.py   (slice selection)
#      Stage 2 → stage2_keypoint_detector.py (keypoint detection)
#      Stage 3 → stage3_severity_classifier.py (severity grading)
#
#  Each stage script is fully self-contained (builds its own
#  volume cache, does its own train/val split, saves its own
#  checkpoints) — this script just runs them in the correct
#  order as subprocesses, so a failure or restart at any stage
#  doesn't require re-running the ones before it.
#
#  Usage
#  -----
#      python train.py                 # run all three stages
#      python train.py --stages 1 2    # run only stages 1 and 2
#      python train.py --stages 3      # run only stage 3
#                                       # (assumes stage 1 & 2 checkpoints exist)
# ============================================================

import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

STAGE_SCRIPTS = {
    1: THIS_DIR / "stage1_slice_selector.py",
    2: THIS_DIR / "stage2_keypoint_detector.py",
    3: THIS_DIR / "stage3_severity_classifier.py",
}

STAGE_NAMES = {
    1: "Slice Selector",
    2: "Keypoint Detector",
    3: "Severity Classifier",
}


def run_stage(stage_num: int) -> None:
    script = STAGE_SCRIPTS[stage_num]
    if not script.exists():
        raise FileNotFoundError(f"Stage {stage_num} script not found: {script}")

    print(f"\n{'#' * 60}")
    print(f"#  STAGE {stage_num}  —  {STAGE_NAMES[stage_num]}")
    print(f"{'#' * 60}\n")

    result = subprocess.run([sys.executable, str(script)], cwd=THIS_DIR)
    if result.returncode != 0:
        print(f"\n✗ Stage {stage_num} exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the three-stage lumbar SCS training pipeline."
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which stages to run, in order (default: 1 2 3).",
    )
    args = parser.parse_args()

    stages = sorted(set(args.stages))
    print(f"Running stages: {stages}")

    for stage_num in stages:
        run_stage(stage_num)

    print(f"\n{'=' * 60}")
    print("✓  Training pipeline complete.")
    print("   Checkpoints saved under CFG.output_dir "
          "(see config_and_utils.py).")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
