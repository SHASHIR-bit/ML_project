"""
Master Pipeline Runner for ML Challenge 2026: Entity Resolution
Runs all steps sequentially, with timing and error handling.

Usage:
    python scripts/run_pipeline.py              # Run all steps
    python scripts/run_pipeline.py --from 3     # Start from step 3
    python scripts/run_pipeline.py --only 4     # Run only step 4
"""
import subprocess
import sys
import time
import argparse
from pathlib import Path

import os

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Reroute all temporary files to X drive to prevent C drive from filling up
TEMP_DIR = PROJECT_DIR / "temp"
os.environ["TMPDIR"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)
os.environ["TMP"] = str(TEMP_DIR)
os.environ["JOBLIB_TEMP_FOLDER"] = str(TEMP_DIR)

STEPS = [
    # (step_num, script, description)
    (1, "scripts/01_data_forensics.py", "Data Forensics (EDA)"),
    (2, "scripts/02_preprocess.py", "Text Preprocessing"),
    (3, "scripts/03_blocking.py", "Blocking / Candidate Generation"),
    (4, "scripts/04_features.py", "Feature Engineering"),
    (5, "scripts/05_train_model.py", "Model Training"),
    (6, "scripts/06_generate_submission.py", "Generate Submission"),
    (7, "scripts/07_package_submission.py", "Package Final Submission"),
]


def run_step(step_num, script, description):
    """Run a single pipeline step."""
    print(f"\n{'#'*80}")
    print(f"# STEP {step_num}: {description}")
    print(f"# Script: {script}")
    print(f"{'#'*80}\n")
    
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / script)],
        cwd=str(PROJECT_DIR),
    )
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"\n!!! STEP {step_num} FAILED (exit code {result.returncode}) after {elapsed:.1f}s !!!")
        sys.exit(1)
    
    print(f"\n>>> Step {step_num} completed in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Run entity resolution pipeline")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this step number")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only this step number")
    parser.add_argument("--skip-forensics", action="store_true",
                        help="Skip step 1 (data forensics)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("ML CHALLENGE 2026: ENTITY RESOLUTION PIPELINE")
    print("=" * 80)
    
    total_t0 = time.time()
    step_times = {}
    
    for step_num, script, description in STEPS:
        if args.only is not None and step_num != args.only:
            continue
        if step_num < args.from_step:
            print(f"  Skipping step {step_num}: {description}")
            continue
        if args.skip_forensics and step_num == 1:
            print(f"  Skipping step 1: {description}")
            continue
        
        elapsed = run_step(step_num, script, description)
        step_times[step_num] = elapsed
    
    total_elapsed = time.time() - total_t0
    
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"\nStep timings:")
    for step_num, elapsed in step_times.items():
        desc = [d for n, s, d in STEPS if n == step_num][0]
        print(f"  Step {step_num} ({desc}): {elapsed:.1f}s")
    print(f"\n  Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    
    # Run validator if output exists
    val_script = PROJECT_DIR / "utils" / "validate_submission.py"
    matching = PROJECT_DIR / "output" / "matching_results.tsv"
    candidate = PROJECT_DIR / "output" / "candidate_pairs.tsv"
    
    if matching.exists() and candidate.exists() and val_script.exists():
        print(f"\n{'='*80}")
        print("RUNNING VALIDATOR")
        print(f"{'='*80}")
        subprocess.run([
            sys.executable, str(val_script),
            "--matching", str(matching),
            "--candidate", str(candidate),
            "--test-dir", str(PROJECT_DIR / "dataset" / "test"),
        ])


if __name__ == "__main__":
    main()
