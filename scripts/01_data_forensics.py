"""
Step 1: Memory-efficient Data Forensics for ML Challenge 2026
Process one file at a time to stay within 16GB RAM
"""
import pandas as pd
import numpy as np
import os, sys, gc, io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', write_through=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

print("=" * 80)
print("STEP 1: DATA FORENSICS (MEMORY-EFFICIENT)")
print("=" * 80)

files = {
    "train_s1": os.path.join(TRAIN_DIR, "train_source1.tsv"),
    "train_s2": os.path.join(TRAIN_DIR, "train_source2.tsv"),
    "train_s3": os.path.join(TRAIN_DIR, "train_source3.tsv"),
    "train_gt": os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
    "test_s1": os.path.join(TEST_DIR, "test_source1.tsv"),
    "test_s2": os.path.join(TEST_DIR, "test_source2.tsv"),
    "test_s3": os.path.join(TEST_DIR, "test_source3.tsv"),
}

file_stats = {}
for name, path in files.items():
    print(f"\n>>> Processing {name}...")
    df = pd.read_csv(path, sep="\t")
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    
    print(f"  Shape: {df.shape[0]:>10,} rows x {df.shape[1]:>3} cols  |  {mem_mb:>8.1f} MB")
    print(f"  Columns: {list(df.columns)}")
    
    for col in df.columns:
        nuniq = df[col].nunique()
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) * 100
        print(f"    {col:30s}  dtype={str(df[col].dtype):10s}  nunique={nuniq:>10,}  nulls={null_count} ({null_pct:.2f}%)")
    
    # Safe print for sample rows
    try:
        sample = df.head(3).to_string()
        print(f"\n  First 3 rows:")
        print(sample)
    except Exception:
        print(f"\n  First 3 rows (ASCII safe):")
        for i, row in df.head(3).iterrows():
            row_str = " | ".join(str(v).encode('ascii', 'replace').decode('ascii') for v in row.values)
            print(f"    {row_str}")
    
    if name != "train_gt":
        # Text stats
        for col in ["business_name", "business_address"]:
            lengths = df[col].fillna("").str.len()
            print(f"\n  {col} char length: mean={lengths.mean():.1f} std={lengths.std():.1f} min={lengths.min()} max={lengths.max()}")
        
        # Country distribution
        print(f"\n  Country distribution:")
        for country, count in df["country"].value_counts().items():
            print(f"    {country}: {count:,}")
        
        # Duplicates
        dup_id = df["entity_id"].duplicated().sum()
        dup_name_addr = df.duplicated(subset=["business_name", "business_address"]).sum()
        print(f"\n  Duplicate entity_ids: {dup_id}")
        print(f"  Duplicate (name+addr): {dup_name_addr}")
        
        # Missingness
        for col in df.columns:
            n_null = df[col].isna().sum()
            n_empty = (df[col].fillna("").astype(str).str.strip() == "").sum()
            if n_null > 0 or n_empty > 0:
                print(f"  Missing/empty in {col}: null={n_null}  empty={n_empty}")
        
        file_stats[name] = {
            "shape": df.shape,
            "entity_ids": set(df["entity_id"]),
            "countries": set(df["country"].dropna()),
        }
    else:
        # Ground truth analysis
        print(f"\n  Ground truth analysis:")
        df["n_matches"] = df["matched_entity_ids"].apply(
            lambda x: 0 if pd.isna(x) or str(x).strip() == "" else len(str(x).split(","))
        )
        singletons = (df["n_matches"] == 0).sum()
        matched = (df["n_matches"] > 0).sum()
        print(f"  Singletons (no matches): {singletons:,} ({singletons/len(df)*100:.1f}%)")
        print(f"  Matched (>=1 match):     {matched:,} ({matched/len(df)*100:.1f}%)")
        print(f"\n  Match count distribution:")
        for stat, val in df["n_matches"].describe().items():
            print(f"    {stat}: {val}")
        print(f"\n  Match count value_counts (top 20):")
        for val, cnt in df["n_matches"].value_counts().head(20).items():
            print(f"    {val}: {cnt:,}")
        
        # S2 vs S3 match breakdown (vectorized)
        valid = df["matched_entity_ids"].dropna()
        valid = valid[valid.str.strip() != ""]
        all_ids = valid.str.split(",").explode().str.strip()
        s2_count = all_ids.str.startswith("S2-").sum()
        s3_count = all_ids.str.startswith("S3-").sum()
        print(f"\n  Total match links: {len(all_ids):,}")
        print(f"  S2 matches: {s2_count:,}")
        print(f"  S3 matches: {s3_count:,}")
        
        # Distribution of matches per S1 entity
        # How many S1 entities match only S2, only S3, or both?
        def classify_matches(x):
            if pd.isna(x) or str(x).strip() == "":
                return "singleton"
            ids = [m.strip() for m in str(x).split(",")]
            has_s2 = any(m.startswith("S2-") for m in ids)
            has_s3 = any(m.startswith("S3-") for m in ids)
            if has_s2 and has_s3:
                return "both"
            elif has_s2:
                return "s2_only"
            else:
                return "s3_only"
        
        df["match_type"] = df["matched_entity_ids"].apply(classify_matches)
        print(f"\n  Match type distribution:")
        for mtype, cnt in df["match_type"].value_counts().items():
            print(f"    {mtype}: {cnt:,}")
        
        file_stats[name] = {
            "shape": df.shape,
            "s1_ids": set(df["source1_entity_id"]),
            "singletons": singletons,
            "matched": matched,
        }
    
    del df
    gc.collect()

# --- Cross-file analysis ---
print("\n" + "=" * 80)
print("CROSS-FILE ANALYSIS")
print("=" * 80)

# Train/Test ID overlap
train_ids = file_stats["train_s1"]["entity_ids"] | file_stats["train_s2"]["entity_ids"] | file_stats["train_s3"]["entity_ids"]
test_ids = file_stats["test_s1"]["entity_ids"] | file_stats["test_s2"]["entity_ids"] | file_stats["test_s3"]["entity_ids"]
overlap = train_ids & test_ids
print(f"\nTrain entity IDs: {len(train_ids):,}")
print(f"Test entity IDs:  {len(test_ids):,}")
print(f"ID overlap:       {len(overlap):,}")

# GT consistency
gt_s1 = file_stats["train_gt"]["s1_ids"]
s1_ids = file_stats["train_s1"]["entity_ids"]
print(f"\nGT S1 count: {len(gt_s1):,}")
print(f"train_s1 count: {len(s1_ids):,}")
print(f"GT subset train_s1: {gt_s1 <= s1_ids}")
print(f"train_s1 subset GT: {s1_ids <= gt_s1}")

# Country analysis
print("\nCountries by dataset:")
for name in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
    if name in file_stats:
        print(f"  {name}: {file_stats[name]['countries']}")

# Scale
print("\nScale analysis:")
for name in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
    print(f"  {name}: {file_stats[name]['shape'][0]:,} rows")

# Comparison space
train_s1_n = file_stats["train_s1"]["shape"][0]
train_s23_n = file_stats["train_s2"]["shape"][0] + file_stats["train_s3"]["shape"][0]
test_s1_n = file_stats["test_s1"]["shape"][0]
test_s23_n = file_stats["test_s2"]["shape"][0] + file_stats["test_s3"]["shape"][0]
print(f"\n  Naive comparison space (train): {train_s1_n * train_s23_n:,.0f}")
print(f"  Naive comparison space (test):  {test_s1_n * test_s23_n:,.0f}")
print(f"  -> BLOCKING IS ESSENTIAL")

print("\n>>> FORENSICS COMPLETE")
