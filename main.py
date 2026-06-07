from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

PIPELINE_STEPS = {
    "check": [
        "scripts/01_data_integrity.py",
    ],
    "prepare": [
        "scripts/02_split_data.py",
        "scripts/03_preprocess_metadata.py",
    ],
    "verify": [
        "scripts/04_test_dataloader.py",
        "scripts/05_test_model.py",
    ],
    "train": [
        "scripts/06_train_demo.py",
    ],
    "evaluate": [
        "scripts/07_evaluate_holdout.py",
    ],
    "infer": [
        "scripts/08_infer_submission.py",
    ],
}

def print_header(title: str) -> None:
    line = "=" * 80
    print("\n" + line)
    print(title)
    print(line)


def run_script(script_path: str) -> None:
    full_path = ROOT_DIR / script_path

    if not full_path.exists():
        raise FileNotFoundError(f"Script not found: {full_path}")

    print_header(f"Running: {script_path}")

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=ROOT_DIR,
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        raise RuntimeError(
            f"Script failed: {script_path}\n"
            f"Exit code: {result.returncode}"
        )

    print(f"\n[DONE] {script_path}")
    print(f"Time elapsed: {elapsed:.2f} seconds")


def get_scripts_to_run(stage: str, skip_verify: bool, skip_infer: bool) -> list[str]:
    if stage == "all":
        scripts = []

        scripts.extend(PIPELINE_STEPS["check"])
        scripts.extend(PIPELINE_STEPS["prepare"])

        if not skip_verify:
            scripts.extend(PIPELINE_STEPS["verify"])

        scripts.extend(PIPELINE_STEPS["train"])
        scripts.extend(PIPELINE_STEPS["evaluate"])

        if not skip_infer:
            scripts.extend(PIPELINE_STEPS["infer"])

        return scripts

    if stage == "demo":
        scripts = []

        scripts.extend(PIPELINE_STEPS["check"])
        scripts.extend(PIPELINE_STEPS["prepare"])

        if not skip_verify:
            scripts.extend(PIPELINE_STEPS["verify"])

        scripts.extend(PIPELINE_STEPS["train"])
        scripts.extend(PIPELINE_STEPS["evaluate"])

        return scripts

    scripts = PIPELINE_STEPS[stage]

    if stage == "verify" and skip_verify:
        return []

    if stage == "infer" and skip_infer:
        return []

    return scripts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ISIC 2024 multimodal classification pipeline."
    )

    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "demo",
            "check",
            "prepare",
            "verify",
            "train",
            "evaluate",
            "infer",
        ],
        help=(
            "Pipeline stage to run. "
            "'all' runs everything including inference. "
            "'demo' runs local CPU demo without inference."
        ),
    )

    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip dataloader and model test scripts.",
    )

    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Skip submission inference step.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show scripts that would be executed.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    scripts = get_scripts_to_run(
        stage=args.stage,
        skip_verify=args.skip_verify,
        skip_infer=args.skip_infer,
    )

    print_header("ISIC 2024 Multimodal Pipeline")
    print(f"Project root : {ROOT_DIR}")
    print(f"Stage        : {args.stage}")
    print(f"Skip verify  : {args.skip_verify}")
    print(f"Skip infer   : {args.skip_infer}")

    if not scripts:
        print("\nNo script to run.")
        return

    print("\nScripts:")
    for idx, script in enumerate(scripts, start=1):
        print(f"{idx}. {script}")

    if args.dry_run:
        print("\n[DRY RUN] No script was executed.")
        return

    start_time = time.time()

    try:
        for script in scripts:
            run_script(script)

    except Exception as error:
        print_header("PIPELINE FAILED")
        print(error)
        sys.exit(1)

    total_time = time.time() - start_time

    print_header("PIPELINE COMPLETED")
    print(f"Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()