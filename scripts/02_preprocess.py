"""
Step 2: PyArrow Compute Text Preprocessing
Blazing fast text processing using C++ backend.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import os, sys, io, gc, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', write_through=True)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
PROCESSED_DIR = PROJECT_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

def fast_normalize(series):
    arr = pa.array(series.fillna("").astype(str))
    arr = pc.utf8_lower(arr)
    arr = pc.replace_substring(arr, "&", " and ")
    # Replace non-alphanumeric with space (ASCII-only for safety/speed)
    arr = pc.replace_substring_regex(arr, r"[^a-z0-9\s]", " ")
    arr = pc.replace_substring_regex(arr, r"\s+", " ")
    arr = pc.utf8_trim_whitespace(arr)
    return arr.to_pandas()

def extract_pincode(series):
    arr = pa.array(series.fillna("").astype(str))
    # Extract 5 or 6 digit number. Pyarrow regex extract returns struct.
    # A simpler way is to use pandas just for this since addresses might not be 100% full.
    # Actually, pandas str.extract on normalized string is fine.
    return pd.Series(arr.to_pandas()).str.extract(r'\b(\d{5,6})\b', expand=False).fillna("")

def process_source(filepath, label):
    print(f"\n>>> Processing {label}...")
    t0 = time.time()

    filepath_str = str(filepath)
    df = pd.read_parquet(filepath_str.replace(".tsv", ".parquet")) if filepath_str.endswith('.parquet') else pd.read_csv(filepath, sep="\t", dtype=str)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    t1 = time.time()
    df["name_norm"] = fast_normalize(df["business_name"])
    df["addr_norm"] = fast_normalize(df["business_address"])
    print(f"  Normalized text in {time.time()-t1:.1f}s")

    t2 = time.time()
    # Extract blocking keys
    df["pincode"] = extract_pincode(df["addr_norm"])
    
    # Prefix (remove 'the ', remove space, take 5)
    name_no_the = pc.replace_substring_regex(pa.array(df["name_norm"]), r"^the\s+", "")
    name_no_space = pc.replace_substring(name_no_the, " ", "")
    df["name_prefix"] = pc.utf8_slice_codeunits(name_no_space, 0, 5).to_pandas()
    
    # First word
    df["first_word"] = df["name_norm"].str.split().str[0].fillna("")
    
    # Country
    country_arr = pc.utf8_lower(pa.array(df["country"].fillna("").astype(str)))
    df["country_lower"] = country_arr.to_pandas()

    # Blocking key combos
    df["bk_prefix_country"] = df["name_prefix"] + "|" + df["country_lower"]
    df["bk_firstword_country"] = df["first_word"] + "|" + df["country_lower"]
    df["bk_pincode_country"] = df["pincode"] + "|" + df["country_lower"]
    print(f"  Extracted blocking keys in {time.time()-t2:.1f}s")

    # Keep only needed columns
    cols_to_keep = [
        "entity_id", "name_norm", "addr_norm", "pincode", "name_prefix", "first_word",
        "country_lower", "bk_prefix_country", "bk_firstword_country", "bk_pincode_country"
    ]
    df = df[cols_to_keep]

    out_path = PROCESSED_DIR / f"{label}.parquet"
    df.to_parquet(out_path, index=False, engine='pyarrow')
    print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Total: {time.time()-t0:.1f}s")

    return len(df)


def main():
    print("=" * 80)
    print("STEP 2: PREPROCESSING (PYARROW COMPUTE)")
    print("=" * 80)

    total_t0 = time.time()

    sources = [
        ("train_s1", TRAIN_DIR / "train_source1.tsv"),
        ("train_s2", TRAIN_DIR / "train_source2.tsv"),
        ("train_s3", TRAIN_DIR / "train_source3.tsv"),
        ("test_s1", TEST_DIR / "test_source1.tsv"),
        ("test_s2", TEST_DIR / "test_source2.tsv"),
        ("test_s3", TEST_DIR / "test_source3.tsv"),
    ]

    for label, filepath in sources:
        process_source(str(filepath), label)
        gc.collect()

    print("\n>>> Saving ground truth...")
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str)
    gt.to_parquet(PROCESSED_DIR / "train_gt.parquet", index=False)
    print(f"  Saved {len(gt):,} rows")

    print(f"\n>>> PREPROCESSING COMPLETE in {time.time()-total_t0:.1f}s")


if __name__ == "__main__":
    main()
