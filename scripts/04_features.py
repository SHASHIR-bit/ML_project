"""
Step 4: Feature Engineering (Batch-Vectorized)
Computes similarity features for all candidate pairs.
Uses pre-computed token sets for fast Jaccard computation.
Processes in batches to manage memory.
"""
import pandas as pd
import numpy as np
import os, sys, io, gc, time, pickle, re
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', write_through=True)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "processed"
CANDIDATES_DIR = PROJECT_DIR / "candidates"
FEATURES_DIR = PROJECT_DIR / "features"
FEATURES_DIR.mkdir(exist_ok=True)


# ─── Fast Similarity Functions ────────────────────────────────────────────────

def token_jaccard(set1, set2):
    """Jaccard similarity between two token sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def token_overlap(set1, set2):
    """Fraction of set1 tokens that appear in set2."""
    if not set1:
        return 0.0
    if not set2:
        return 0.0
    return len(set1 & set2) / len(set1)


def char_ngram_set(s, n=3):
    """Generate character n-gram set from a string."""
    if not s or len(s) < n:
        return frozenset()
    return frozenset(s[i:i+n] for i in range(len(s) - n + 1))


def char_ngram_jaccard(s1, s2, n=3):
    """Character n-gram Jaccard similarity."""
    ng1 = char_ngram_set(s1, n)
    ng2 = char_ngram_set(s2, n)
    if not ng1 and not ng2:
        return 1.0
    if not ng1 or not ng2:
        return 0.0
    inter = len(ng1 & ng2)
    union = len(ng1 | ng2)
    return inter / union if union > 0 else 0.0


def number_set(s):
    """Extract set of numbers from string."""
    if not s:
        return frozenset()
    return frozenset(re.findall(r'\d+', s))


def compute_features_batch(s1_rows, sx_rows):
    """Compute features for a batch of (S1, SX) pairs.
    
    s1_rows: list of dicts with keys: name_norm, addr_norm, pincode, first_word, name_tokens, addr_tokens, name_3grams
    sx_rows: list of dicts (same keys)
    
    Returns: list of feature dicts
    """
    results = []
    
    for s1_r, sx_r in zip(s1_rows, sx_rows):
        n1 = s1_r["name_norm"]
        n2 = sx_r["name_norm"]
        a1 = s1_r["addr_norm"]
        a2 = sx_r["addr_norm"]
        
        nt1 = s1_r["name_tokens"]
        nt2 = sx_r["name_tokens"]
        at1 = s1_r["addr_tokens"]
        at2 = sx_r["addr_tokens"]
        
        feats = {}
        
        # Name features
        feats["name_jaccard"] = token_jaccard(nt1, nt2)
        feats["name_overlap_s1"] = token_overlap(nt1, nt2)
        feats["name_overlap_sx"] = token_overlap(nt2, nt1)
        # HACKATHON SPEEDUP: Disable slow char ngrams
        feats["name_char3gram"] = 0.0 # char_ngram_jaccard(n1, n2, 3)
        feats["name_char4gram"] = 0.0 # char_ngram_jaccard(n1, n2, 4)
        
        # Name length features
        len_n1 = len(n1) if n1 else 0
        len_n2 = len(n2) if n2 else 0
        feats["name_len_ratio"] = min(len_n1, len_n2) / max(len_n1, len_n2) if max(len_n1, len_n2) > 0 else 1.0
        feats["name_len_diff"] = abs(len_n1 - len_n2)
        feats["name_exact"] = 1.0 if n1 == n2 and n1 else 0.0
        
        # Token count features
        tc1 = len(nt1) if nt1 else 0
        tc2 = len(nt2) if nt2 else 0
        feats["name_token_count_diff"] = abs(tc1 - tc2)
        feats["name_token_count_ratio"] = min(tc1, tc2) / max(tc1, tc2) if max(tc1, tc2) > 0 else 1.0
        
        # First word match
        feats["first_word_match"] = 1.0 if s1_r["first_word"] == sx_r["first_word"] and s1_r["first_word"] else 0.0
        
        # Address features
        feats["addr_jaccard"] = token_jaccard(at1, at2)
        feats["addr_overlap_s1"] = token_overlap(at1, at2)
        feats["addr_overlap_sx"] = token_overlap(at2, at1)
        feats["addr_char3gram"] = 0.0 # char_ngram_jaccard(a1, a2, 3)
        
        len_a1 = len(a1) if a1 else 0
        len_a2 = len(a2) if a2 else 0
        feats["addr_len_ratio"] = min(len_a1, len_a2) / max(len_a1, len_a2) if max(len_a1, len_a2) > 0 else 1.0
        feats["addr_exact"] = 1.0 if a1 == a2 and a1 else 0.0
        
        # Address number overlap
        nums1 = number_set(a1)
        nums2 = number_set(a2)
        if nums1 and nums2:
            feats["addr_num_overlap"] = len(nums1 & nums2) / len(nums1 | nums2)
        elif not nums1 and not nums2:
            feats["addr_num_overlap"] = 1.0
        else:
            feats["addr_num_overlap"] = 0.0
        
        # Pincode
        p1 = s1_r["pincode"]
        p2 = sx_r["pincode"]
        feats["pincode_match"] = 1.0 if p1 == p2 and p1 else 0.0
        feats["pincode_both_missing"] = 1.0 if not p1 and not p2 else 0.0
        feats["pincode_one_missing"] = 1.0 if bool(p1) != bool(p2) else 0.0
        
        # Source (S2 vs S3)
        sx_id = sx_r["entity_id"]
        feats["is_s2"] = 1.0 if sx_id.startswith("S2-") else 0.0
        
        # Combined scores
        feats["combined_name_addr"] = 0.6 * feats["name_jaccard"] + 0.4 * feats["addr_jaccard"]
        feats["combined_char3gram"] = 0.0 # 0.6 * feats["name_char3gram"] + 0.4 * feats["addr_char3gram"]
        
        # Address has data
        feats["addr_empty_s1"] = 1.0 if not a1 else 0.0
        feats["addr_empty_sx"] = 1.0 if not a2 else 0.0
        
        results.append(feats)
    
    return results


def build_entity_lookup(df):
    """Build a dict of entity_id -> record dict for fast lookup."""
    lookup = {}
    for _, row in df.iterrows():
        eid = row["entity_id"]
        name_norm = row["name_norm"] if pd.notna(row["name_norm"]) else ""
        addr_norm = row["addr_norm"] if pd.notna(row["addr_norm"]) else ""
        pincode = row["pincode"] if pd.notna(row["pincode"]) else ""
        first_word = row["first_word"] if pd.notna(row["first_word"]) else ""
        
        lookup[eid] = {
            "entity_id": eid,
            "name_norm": name_norm,
            "addr_norm": addr_norm,
            "pincode": pincode,
            "first_word": first_word,
            "name_tokens": frozenset(name_norm.split()) if name_norm else frozenset(),
            "addr_tokens": frozenset(addr_norm.split()) if addr_norm else frozenset(),
        }
    return lookup


def build_features_for_split(split="train"):
    """Build feature matrix for all candidate pairs in a split."""
    print(f"\n{'='*80}")
    print(f"FEATURE ENGINEERING: {split.upper()}")
    print(f"{'='*80}")
    
    t_total = time.time()
    
    # Load candidates
    print(f"\n  Loading candidates...")
    with open(CANDIDATES_DIR / f"{split}_candidates.pkl", "rb") as f:
        candidates = pickle.load(f)
    
    total_pairs = sum(len(v) for v in candidates.values())
    print(f"  Total candidate pairs: {total_pairs:,}")
    
    # Load S1 data
    print(f"  Loading S1...")
    s1_df = pd.read_parquet(PROCESSED_DIR / f"{split}_s1.parquet",
                            columns=["entity_id", "name_norm", "addr_norm", "pincode", "first_word"])
    print(f"  Building S1 lookup ({len(s1_df):,} entities)...")
    s1_lookup = build_entity_lookup(s1_df)
    del s1_df
    gc.collect()
    
    # Load S2 + S3 data
    print(f"  Loading S2 + S3...")
    s2_df = pd.read_parquet(PROCESSED_DIR / f"{split}_s2.parquet",
                            columns=["entity_id", "name_norm", "addr_norm", "pincode", "first_word"])
    s3_df = pd.read_parquet(PROCESSED_DIR / f"{split}_s3.parquet",
                            columns=["entity_id", "name_norm", "addr_norm", "pincode", "first_word"])
    sx_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    # HACKATHON SPEEDUP: Filter sx_df to only keep needed candidates
    needed_sx_ids = set()
    for cands in candidates.values():
        needed_sx_ids.update(cands)
    
    n_before = len(sx_df)
    sx_df = sx_df[sx_df["entity_id"].isin(needed_sx_ids)].copy()
    print(f"  Filtered SX entities from {n_before:,} to {len(sx_df):,} (huge memory savings!)")
    
    print(f"  Building SX lookup ({len(sx_df):,} entities)...")
    sx_lookup = build_entity_lookup(sx_df)
    del sx_df
    gc.collect()
    
    # Load ground truth for labels (train only)
    gt_labels = {}
    if split == "train":
        gt = pd.read_parquet(PROCESSED_DIR / "train_gt.parquet")
        for _, row in gt.iterrows():
            s1_id = row["source1_entity_id"]
            matched = row["matched_entity_ids"]
            if pd.isna(matched) or str(matched).strip() == "":
                gt_labels[s1_id] = set()
            else:
                gt_labels[s1_id] = set(m.strip() for m in str(matched).split(",") if m.strip())
        del gt
        gc.collect()
    
    # Compute features in batches
    print(f"\n  Computing features for {total_pairs:,} pairs...")
    
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    out_path = FEATURES_DIR / f"{split}_features.parquet"
    writer = None
    
    batch_s1_rows = []
    batch_sx_rows = []
    batch_s1_ids = []
    batch_sx_ids = []
    batch_labels = []
    batch_size = 50000
    processed = 0
    
    t_feat = time.time()
    
    for s1_id, cand_ids in candidates.items():
        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            continue
        
        true_matches = gt_labels.get(s1_id, set()) if split == "train" else set()
        
        cand_ids_to_process = cand_ids
        if split == "train":
            import random
            pos_cands = [c for c in cand_ids if c in true_matches]
            neg_cands = [c for c in cand_ids if c not in true_matches]
            # Subsample negatives to vastly speed up training data creation
            if len(neg_cands) > 5:
                neg_cands = random.sample(neg_cands, 5)
            cand_ids_to_process = pos_cands + neg_cands

        for sx_id in cand_ids_to_process:
            sx_row = sx_lookup.get(sx_id)
            if sx_row is None:
                continue
            
            batch_s1_rows.append(s1_row)
            batch_sx_rows.append(sx_row)
            batch_s1_ids.append(s1_id)
            batch_sx_ids.append(sx_id)
            if split == "train":
                batch_labels.append(1 if sx_id in true_matches else 0)
            
            if len(batch_s1_rows) >= batch_size:
                feats = compute_features_batch(batch_s1_rows, batch_sx_rows)
                df_batch = pd.DataFrame(feats)
                df_batch["s1_id"] = batch_s1_ids
                df_batch["sx_id"] = batch_sx_ids
                if split == "train":
                    df_batch["label"] = batch_labels
                feat_cols = [c for c in df_batch.columns if c not in ["s1_id", "sx_id", "label"]]
                df_batch[feat_cols] = df_batch[feat_cols].astype(np.float32)
                
                table = pa.Table.from_pandas(df_batch)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
                
                processed += len(batch_s1_rows)
                if processed % 500000 == 0:
                    elapsed = time.time() - t_feat
                    rate = processed / elapsed
                    eta = (total_pairs - processed) / rate if rate > 0 else 0
                    print(f"    {processed:,}/{total_pairs:,} pairs ({processed/total_pairs*100:.1f}%) {rate:.0f} pairs/s ETA {eta:.0f}s")
                
                batch_s1_rows = []
                batch_sx_rows = []
                batch_s1_ids = []
                batch_sx_ids = []
                batch_labels = []
    
    # Process remaining batch
    if batch_s1_rows:
        feats = compute_features_batch(batch_s1_rows, batch_sx_rows)
        df_batch = pd.DataFrame(feats)
        df_batch["s1_id"] = batch_s1_ids
        df_batch["sx_id"] = batch_sx_ids
        if split == "train":
            df_batch["label"] = batch_labels
            
        feat_cols = [c for c in df_batch.columns if c not in ["s1_id", "sx_id", "label"]]
        df_batch[feat_cols] = df_batch[feat_cols].astype(np.float32)
        
        table = pa.Table.from_pandas(df_batch)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        processed += len(batch_s1_rows)
        
    if writer:
        writer.close()
    
    print(f"    Total processed: {processed:,} pairs in {time.time()-t_feat:.1f}s")
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved to {out_path} ({size_mb:.1f} MB)")
    print(f"  Total time: {time.time()-t_total:.1f}s")
    
    gc.collect()


def main():
    print("=" * 80)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 80)
    
    # train is already done and saved! 
    # build_features_for_split("train")
    # gc.collect()
    
    build_features_for_split("test")
    
    print("\n>>> FEATURE ENGINEERING COMPLETE")


if __name__ == "__main__":
    main()
