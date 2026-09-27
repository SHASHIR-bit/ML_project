"""
Step 3: IMPROVED Blocking / Candidate Generation
Key improvements over v1:
- MAX_BLOCK_SIZE: 500 -> 3000 (recovers hundreds of thousands of skipped blocks)
- MAX_CANDIDATES_PER_S1: 100 -> 200
- RARE_TOKEN_MAX_FREQ: 300 -> 500
- Added clean name prefix blocking (strips business suffixes for better matching)
- Score-based candidate capping (keeps best candidates instead of random)
"""
import pandas as pd
import numpy as np
import os, sys, io, gc, time, pickle
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', write_through=True)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "processed"
CANDIDATES_DIR = PROJECT_DIR / "candidates"
CANDIDATES_DIR.mkdir(exist_ok=True)

# ─── IMPROVED Configuration ──────────────────────────────────────────────────
MAX_BLOCK_SIZE = 3000        # Was 500 — recovers huge number of missed blocks
RARE_TOKEN_MAX_FREQ = 500    # Was 300 — captures more shared-token pairs
RARE_TOKEN_MIN_LEN = 3       # Min length for a rare token
MIN_SHARED_TOKENS = 1        # Min shared rare tokens for a candidate
MAX_CANDIDATES_PER_S1 = 200  # Was 100 — allows more candidates per entity

# Business suffixes to strip for clean name blocking
BUSINESS_SUFFIXES = frozenset({
    'corporation', 'incorporated', 'limited', 'company', 'enterprises',
    'industries', 'international', 'solutions', 'services', 'technologies',
    'group', 'holdings', 'associates', 'partners', 'consulting',
    'private', 'public', 'llc', 'llp', 'lp', 'plc',
    'pvt', 'ltd', 'corp', 'inc', 'co',
    'sa', 'sas', 'sarl', 'srl', 'gmbh', 'ag', 'bv', 'nv',
    'pty', 'bhd', 'sdn', 'and', 'the', 'of',
})


def clean_name(name_norm):
    """Remove business suffixes from normalized name for better blocking."""
    if not name_norm or not isinstance(name_norm, str):
        return ""
    tokens = [t for t in name_norm.split() if t not in BUSINESS_SUFFIXES]
    return " ".join(tokens)


def build_inverted_index(entity_ids, name_norms, max_freq=RARE_TOKEN_MAX_FREQ):
    """Build inverted index from name tokens, filtering common tokens."""
    token_freq = defaultdict(int)
    for name in name_norms:
        if name and isinstance(name, str):
            for token in set(name.split()):
                if len(token) >= RARE_TOKEN_MIN_LEN:
                    token_freq[token] += 1

    index = defaultdict(list)
    for eid, name in zip(entity_ids, name_norms):
        if name and isinstance(name, str):
            for token in set(name.split()):
                if len(token) >= RARE_TOKEN_MIN_LEN and token_freq.get(token, 0) <= max_freq:
                    index[token].append(eid)

    n_tokens = len(index)
    avg_posting = np.mean([len(v) for v in index.values()]) if index else 0
    print(f"    Inverted index: {n_tokens:,} rare tokens, avg posting size: {avg_posting:.1f}")
    return index


def block_by_key(s1_keys, s1_ids, sx_keys, sx_ids, label):
    """Generate candidates by exact match on a blocking key."""
    sx_index = defaultdict(list)
    for eid, key in zip(sx_ids, sx_keys):
        if key and "|" in key:
            parts = key.split("|")
            if parts[0]:
                sx_index[key].append(eid)

    candidates = defaultdict(set)
    skipped = 0
    for s1_id, key in zip(s1_ids, s1_keys):
        if key and "|" in key:
            parts = key.split("|")
            if parts[0]:
                matches = sx_index.get(key, [])
                if 0 < len(matches) <= MAX_BLOCK_SIZE:
                    candidates[s1_id].update(matches)
                elif len(matches) > MAX_BLOCK_SIZE:
                    skipped += 1

    n_s1 = sum(1 for v in candidates.values() if v)
    n_pairs = sum(len(v) for v in candidates.values())
    print(f"    [{label}] {n_s1:,} S1 with candidates, {n_pairs:,} pairs, {skipped} skipped")
    return candidates


def block_by_rare_tokens(s1_ids, s1_names, sx_ids, sx_names, label):
    """Generate candidates using rare token inverted index."""
    t0 = time.time()

    index = build_inverted_index(sx_ids, sx_names, max_freq=RARE_TOKEN_MAX_FREQ)

    candidates = defaultdict(set)
    for s1_id, name in zip(s1_ids, s1_names):
        if not name or not isinstance(name, str):
            continue
        token_set = set(name.split())
        candidate_counts = defaultdict(int)
        for token in token_set:
            if len(token) >= RARE_TOKEN_MIN_LEN and token in index:
                posting = index[token]
                if len(posting) <= RARE_TOKEN_MAX_FREQ:
                    for sx_id in posting:
                        candidate_counts[sx_id] += 1
        for sx_id, count in candidate_counts.items():
            if count >= MIN_SHARED_TOKENS:
                candidates[s1_id].add(sx_id)

    n_s1 = sum(1 for v in candidates.values() if v)
    n_pairs = sum(len(v) for v in candidates.values())
    print(f"    [{label}] Rare tokens: {n_s1:,} S1, {n_pairs:,} pairs, {time.time()-t0:.1f}s")
    return candidates


def scored_merge_and_cap(s1_all_ids, candidate_dicts_with_weights, max_per_s1=MAX_CANDIDATES_PER_S1):
    """Merge candidate dicts with scoring for intelligent capping.
    
    candidate_dicts_with_weights: list of (candidates_dict, weight) tuples.
    Higher weight = more reliable blocking source.
    When capping, keeps candidates with highest total score.
    """
    # Score each candidate
    scores = defaultdict(lambda: defaultdict(float))
    for cd, weight in candidate_dicts_with_weights:
        for s1_id, cands in cd.items():
            for sx_id in cands:
                scores[s1_id][sx_id] += weight

    # Build final candidates, capping by score
    merged = {}
    capped = 0
    for s1_id in s1_all_ids:
        if s1_id in scores and scores[s1_id]:
            sx_scores = scores[s1_id]
            if len(sx_scores) <= max_per_s1:
                merged[s1_id] = set(sx_scores.keys())
            else:
                # Keep top-K by score
                top_k = sorted(sx_scores.items(), key=lambda x: x[1], reverse=True)[:max_per_s1]
                merged[s1_id] = set(sx_id for sx_id, _ in top_k)
                capped += 1
        else:
            merged[s1_id] = set()

    if capped > 0:
        print(f"    Score-capped {capped:,} S1 entities at {max_per_s1} candidates (kept best)")

    return merged


def run_blocking_for_split(split="train"):
    """Run blocking for a train or test split."""
    print(f"\n{'='*80}")
    print(f"BLOCKING: {split.upper()}")
    print(f"{'='*80}")

    t_total = time.time()

    # Load S1
    print(f"\n  Loading {split} S1...")
    s1 = pd.read_parquet(PROCESSED_DIR / f"{split}_s1.parquet",
                         columns=["entity_id", "name_norm", "addr_norm", "country_lower",
                                  "pincode", "name_prefix", "first_word",
                                  "bk_prefix_country", "bk_firstword_country", "bk_pincode_country"])
    print(f"  S1: {len(s1):,} entities")

    # Compute clean name and blocking key
    print(f"  Computing clean name blocking keys...")
    t_clean = time.time()
    s1["name_clean"] = s1["name_norm"].fillna("").apply(clean_name)
    s1_clean_no_space = s1["name_clean"].str.replace(" ", "", regex=False)
    s1["bk_clean_prefix_country"] = s1_clean_no_space.str[:5] + "|" + s1["country_lower"]
    del s1_clean_no_space
    print(f"  Clean names computed in {time.time()-t_clean:.1f}s")

    # Collect all candidate dicts with weights for scoring
    all_weighted_candidates = []

    for sx_label in ["s2", "s3"]:
        print(f"\n  Loading {split} {sx_label.upper()}...")
        sx = pd.read_parquet(PROCESSED_DIR / f"{split}_{sx_label}.parquet",
                             columns=["entity_id", "name_norm", "addr_norm", "country_lower",
                                      "pincode", "name_prefix", "first_word",
                                      "bk_prefix_country", "bk_firstword_country", "bk_pincode_country"])
        print(f"  {sx_label.upper()}: {len(sx):,} entities")

        # Compute clean name for SX
        sx["name_clean"] = sx["name_norm"].fillna("").apply(clean_name)
        sx_clean_no_space = sx["name_clean"].str.replace(" ", "", regex=False)
        sx["bk_clean_prefix_country"] = sx_clean_no_space.str[:5] + "|" + sx["country_lower"]
        del sx_clean_no_space

        countries = sorted(set(s1["country_lower"].unique()) | set(sx["country_lower"].unique()))

        for country in countries:
            print(f"\n  --- Country: {country} ---")
            s1_c = s1[s1["country_lower"] == country]
            sx_c = sx[sx["country_lower"] == country]
            print(f"    S1: {len(s1_c):,} | {sx_label.upper()}: {len(sx_c):,}")

            if len(s1_c) == 0 or len(sx_c) == 0:
                continue

            s1_ids = s1_c["entity_id"].values
            sx_ids = sx_c["entity_id"].values
            s1_names = s1_c["name_norm"].values
            sx_names = sx_c["name_norm"].values

            # Strategy 1: Name prefix + country (weight=1.0)
            c1 = block_by_key(
                s1_c["bk_prefix_country"].values, s1_ids,
                sx_c["bk_prefix_country"].values, sx_ids,
                f"prefix_{country}"
            )
            all_weighted_candidates.append((c1, 1.0))

            # Strategy 2: First word + country (weight=0.8)
            c2 = block_by_key(
                s1_c["bk_firstword_country"].values, s1_ids,
                sx_c["bk_firstword_country"].values, sx_ids,
                f"firstword_{country}"
            )
            all_weighted_candidates.append((c2, 0.8))

            # Strategy 3: Pincode + country (weight=1.5 — strong signal)
            c3 = block_by_key(
                s1_c["bk_pincode_country"].values, s1_ids,
                sx_c["bk_pincode_country"].values, sx_ids,
                f"pincode_{country}"
            )
            all_weighted_candidates.append((c3, 1.5))

            # Strategy 4: Clean name prefix + country (NEW — weight=1.2)
            c4 = block_by_key(
                s1_c["bk_clean_prefix_country"].values, s1_ids,
                sx_c["bk_clean_prefix_country"].values, sx_ids,
                f"clean_prefix_{country}"
            )
            all_weighted_candidates.append((c4, 1.2))

            # Strategy 5: Rare token inverted index (weight=0.5 per shared token)
            c5 = block_by_rare_tokens(
                s1_ids, s1_names, sx_ids, sx_names,
                f"raretok_{country}"
            )
            all_weighted_candidates.append((c5, 0.5))

            del c1, c2, c3, c4, c5, s1_c, sx_c
            gc.collect()

        del sx
        gc.collect()

    # Score-based merge and cap
    print(f"\n  Merging and capping candidates (score-based)...")
    all_s1_ids = s1["entity_id"].values
    all_candidates = scored_merge_and_cap(all_s1_ids, all_weighted_candidates, max_per_s1=MAX_CANDIDATES_PER_S1)

    # Free the weighted candidates
    del all_weighted_candidates
    gc.collect()

    # Stats
    total_pairs = sum(len(v) for v in all_candidates.values())
    n_with_cands = sum(1 for v in all_candidates.values() if len(v) > 0)
    avg_cands = total_pairs / max(n_with_cands, 1)

    print(f"\n  BLOCKING SUMMARY ({split}):")
    print(f"  Total S1: {len(all_candidates):,}")
    print(f"  S1 with candidates: {n_with_cands:,}")
    print(f"  Total candidate pairs: {total_pairs:,}")
    print(f"  Avg candidates per S1 (with cands): {avg_cands:.1f}")

    # Save
    out_path = CANDIDATES_DIR / f"{split}_candidates.pkl"
    with open(out_path, "wb") as f:
        serializable = {k: sorted(v) for k, v in all_candidates.items()}
        pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved to {out_path}")

    # Recall analysis (train only)
    if split == "train":
        print(f"\n  --- BLOCKING RECALL ANALYSIS ---")
        gt = pd.read_parquet(PROCESSED_DIR / "train_gt.parquet")

        total_true = 0
        found = 0

        for _, row in gt.iterrows():
            s1_id = row["source1_entity_id"]
            matched = row["matched_entity_ids"]
            if pd.isna(matched) or str(matched).strip() == "":
                continue
            true_ids = set(m.strip() for m in str(matched).split(",") if m.strip())
            cand_ids = all_candidates.get(s1_id, set())
            for tid in true_ids:
                total_true += 1
                if tid in cand_ids:
                    found += 1

        recall = found / max(total_true, 1)
        print(f"  True match pairs: {total_true:,}")
        print(f"  Found by blocking: {found:,}")
        print(f"  Missed: {total_true - found:,}")
        print(f"  BLOCKING RECALL: {recall:.4f}")
        print(f"  (This is the upper bound on final recall)")
        del gt

    del s1, all_candidates
    gc.collect()

    print(f"\n  Total blocking time: {time.time()-t_total:.1f}s")


def main():
    print("=" * 80)
    print("STEP 3: IMPROVED BLOCKING / CANDIDATE GENERATION")
    print("=" * 80)

    run_blocking_for_split("train")
    gc.collect()
    run_blocking_for_split("test")

    print("\n>>> BLOCKING COMPLETE")


if __name__ == "__main__":
    main()
