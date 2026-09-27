"""
Step 6: Generate Submission (LightGBM) — Memory-Safe Chunked Version
Applies the trained LightGBM model to test features in chunks to avoid OOM.
Formats the output as matching_results.tsv and candidate_pairs.tsv.
"""
import pandas as pd
import numpy as np
import os, sys, io, gc, time, pickle, json
from pathlib import Path
import lightgbm as lgb
from collections import defaultdict
import pyarrow.parquet as pq

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', write_through=True)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "processed"
FEATURES_DIR = PROJECT_DIR / "features"
CANDIDATES_DIR = PROJECT_DIR / "candidates"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 80)
    print("STEP 6: GENERATE SUBMISSION (Memory-Safe Chunked)")
    print("=" * 80)

    t_total = time.time()

    # Load model and metadata
    print(f"\n  Loading model...")
    model = lgb.Booster(model_file=str(MODELS_DIR / "lgb_matcher_full.txt"))

    with open(MODELS_DIR / "model_metadata.json") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_cols"]
    threshold = metadata["best_threshold"]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Threshold: {threshold:.3f}")

    # ── Chunked prediction over Parquet row groups ────────────────────────
    print(f"\n  Reading test features in chunks (row groups)...")
    parquet_path = FEATURES_DIR / "test_features.parquet"
    pf = pq.ParquetFile(parquet_path)
    n_row_groups = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    print(f"  Parquet: {total_rows:,} rows in {n_row_groups} row groups")

    matches = defaultdict(list)   # s1_id -> [sx_ids above threshold]
    n_positive = 0
    processed = 0

    # Process one row group at a time (each is ~100-500 MB uncompressed)
    for rg_idx in range(n_row_groups):
        t_rg = time.time()
        table = pf.read_row_group(rg_idx, columns=["s1_id", "sx_id"] + feature_cols)
        df = table.to_pandas()
        del table
        rg_size = len(df)

        X = df[feature_cols].values.astype(np.float32)
        s1_ids = df["s1_id"].values
        sx_ids = df["sx_id"].values
        del df
        gc.collect()

        proba = model.predict(X)
        del X
        gc.collect()

        # Accumulate matches
        mask = proba >= threshold
        n_pos_rg = int(mask.sum())
        n_positive += n_pos_rg

        for idx in np.where(mask)[0]:
            matches[s1_ids[idx]].append(sx_ids[idx])

        processed += rg_size
        elapsed = time.time() - t_rg
        print(f"    Row group {rg_idx+1}/{n_row_groups}: {rg_size:,} pairs, "
              f"{n_pos_rg:,} matches, {elapsed:.1f}s  "
              f"[{processed:,}/{total_rows:,} = {processed/total_rows*100:.1f}%]")

        del s1_ids, sx_ids, proba, mask
        gc.collect()

    print(f"\n  Total positive matches: {n_positive:,} / {total_rows:,}")

    # Get all S1 test IDs
    print(f"\n  Loading all test S1 IDs...")
    s1_test = pd.read_parquet(PROCESSED_DIR / "test_s1.parquet", columns=["entity_id"])
    all_s1_ids = sorted(s1_test["entity_id"].tolist())
    del s1_test
    gc.collect()
    print(f"  Total test S1 entities: {len(all_s1_ids):,}")

    # === Write matching_results.tsv ===
    print(f"\n  Writing matching_results.tsv...")
    results_path = OUTPUT_DIR / "matching_results.tsv"

    n_with_matches = 0
    n_singletons = 0

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            if s1_id in matches:
                matched_str = ",".join(sorted(set(matches[s1_id])))
                f.write(f"{s1_id}\t{matched_str}\n")
                n_with_matches += 1
            else:
                f.write(f"{s1_id}\t\n")
                n_singletons += 1

    print(f"  S1 with matches: {n_with_matches:,}")
    print(f"  Singletons: {n_singletons:,}")
    print(f"  Saved to {results_path}")

    del matches
    gc.collect()

    # === Write candidate_pairs.tsv (stream the pickle) ===
    print(f"\n  Writing candidate_pairs.tsv...")
    cand_path = OUTPUT_DIR / "candidate_pairs.tsv"

    with open(CANDIDATES_DIR / "test_candidates.pkl", "rb") as f:
        candidates = pickle.load(f)

    n_with_cands = 0

    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates.get(s1_id, [])
            if cands:
                cand_str = ",".join(sorted(cands))
                f.write(f"{s1_id}\t{cand_str}\n")
                n_with_cands += 1
            else:
                f.write(f"{s1_id}\t\n")

    del candidates
    gc.collect()

    print(f"  S1 with candidates: {n_with_cands:,}")
    print(f"  Saved to {cand_path}")

    print(f"\n>>> SUBMISSION COMPLETE in {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
