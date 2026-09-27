"""
ML Challenge 2026 - Entity Resolution Pipeline (0.99+ Target Edition)
=====================================================================
Paste this entire script into a single Kaggle notebook cell.
Requires: dataset "er-challenge-data" added to notebook.
Runtime: ~80-100 minutes on Kaggle GPU/CPU.
Changes for 0.99+:
- Ensembled LightGBM + XGBoost + CatBoost
- Expanded candidate blocking limits (TF-IDF Top-K 100, Max Cands 500)
- Added address-based TF-IDF blocking
- Added word-level TF-IDF similarity features
- Added token_sort_ratio and token_set_ratio features
- Optimized threshold finding mechanism
"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "rapidfuzz", "catboost", "xgboost", "-q"], check=True)

import pandas as pd, numpy as np, gc, time, os, re, json
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import GroupKFold
from scipy.sparse import csr_matrix
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from rapidfuzz.distance import JaroWinkler, Levenshtein as RFLevenshtein
from rapidfuzz.fuzz import token_sort_ratio, token_set_ratio
import warnings; warnings.filterwarnings('ignore')

# ─── Auto-detect data path ───────────────────────────────────────────────────
INPUT_ROOT = Path("/kaggle/input")
DATA_DIR = None

# Recursively search for the data since Kaggle sometimes nests uploaded folders
for path in INPUT_ROOT.rglob("train_s1.parquet"):
    DATA_DIR = path.parent
    break

assert DATA_DIR is not None, f"Cannot find train_s1.parquet anywhere in {INPUT_ROOT}. Make sure your Kaggle Dataset is attached to this notebook."
OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
print(f"✅ Data directory: {DATA_DIR}")

# ─── Constants ────────────────────────────────────────────────────────────────
TFIDF_TOP_K = 100       # Increased for ~100% blocking recall
MAX_BLOCK_SIZE = 8000
MAX_CANDIDATES = 500
NEG_RATIO = 12

SUFFIXES = frozenset({
    'corporation','incorporated','limited','company','enterprises','industries',
    'international','solutions','services','technologies','group','holdings',
    'associates','partners','consulting','private','public','llc','llp','lp',
    'plc','pvt','ltd','corp','inc','co','sa','sas','sarl','srl','gmbh','ag',
    'bv','nv','pty','bhd','sdn','and','the','of',
})
ABBREVS = {
    'corp':'corporation','inc':'incorporated','ltd':'limited','pvt':'private',
    'co':'company','intl':'international','natl':'national','mfg':'manufacturing',
    'tech':'technology','sys':'systems','svcs':'services','svc':'service',
    'grp':'group','hldgs':'holdings','engg':'engineering','inds':'industries',
    'bros':'brothers','assoc':'associates','mgmt':'management','dept':'department',
    'govt':'government','hosp':'hospital','pharma':'pharmaceutical',
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def clean(name):
    if not name: return ""
    return " ".join(t for t in name.split() if t not in SUFFIXES)

def expand(name):
    if not name: return ""
    return " ".join(ABBREVS.get(t, t) for t in name.split())

def ngram_set(s, n=3):
    if not s or len(s) < n: return frozenset()
    return frozenset(s[i:i+n] for i in range(len(s)-n+1))

def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    i = len(a & b); u = len(a | b)
    return i / u if u else 0.0

def overlap(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a)

def fbeta(p, r, beta=0.5):
    if p + r == 0: return 0.0
    return (1+beta**2)*p*r / (beta**2*p + r)

T0 = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("="*80); print("SECTION 1: LOAD DATA"); print("="*80)
t0 = time.time()

train_s1 = pd.read_parquet(DATA_DIR/"train_s1.parquet")
train_s2 = pd.read_parquet(DATA_DIR/"train_s2.parquet")
train_s3 = pd.read_parquet(DATA_DIR/"train_s3.parquet")
train_gt = pd.read_parquet(DATA_DIR/"train_gt.parquet")
test_s1  = pd.read_parquet(DATA_DIR/"test_s1.parquet")
test_s2  = pd.read_parquet(DATA_DIR/"test_s2.parquet")
test_s3  = pd.read_parquet(DATA_DIR/"test_s3.parquet")

print(f"Train: S1={len(train_s1):,} S2={len(train_s2):,} S3={len(train_s3):,}")
print(f"Test:  S1={len(test_s1):,}  S2={len(test_s2):,}  S3={len(test_s3):,}")

# Ground truth dict
gt_dict = {}
for _, r in train_gt.iterrows():
    s1 = r["source1_entity_id"]; m = r["matched_entity_ids"]
    gt_dict[s1] = set() if pd.isna(m) or str(m).strip()=="" else set(x.strip() for x in str(m).split(",") if x.strip())
del train_gt; gc.collect()

# Add expanded/clean names
for df in [train_s1, train_s2, train_s3, test_s1, test_s2, test_s3]:
    df["name_norm"] = df["name_norm"].fillna("")
    df["addr_norm"] = df["addr_norm"].fillna("")
    df["name_exp"]  = df["name_norm"].apply(expand)
    df["name_cln"]  = df["name_norm"].apply(clean)

print(f"Loaded in {time.time()-t0:.1f}s\n")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: BLOCKING (TF-IDF + TRADITIONAL)
# ══════════════════════════════════════════════════════════════════════════════
print("="*80); print("SECTION 2: BLOCKING"); print("="*80)

def tfidf_block(s1_df, sx_df, col="name_exp", top_k=TFIDF_TOP_K):
    """TF-IDF blocking per country."""
    cands = defaultdict(set)
    countries = sorted(set(s1_df["country_lower"].unique()) | set(sx_df["country_lower"].unique()))
    for country in countries:
        s1c = s1_df[s1_df["country_lower"]==country]
        sxc = sx_df[sx_df["country_lower"]==country]
        if len(s1c)==0 or len(sxc)==0: continue
        
        # Fill empty with 'unknown' for vectorizer
        sx_vals = sxc[col].replace("", "unknown").values
        s1_vals = s1c[col].replace("", "unknown").values
        
        s1_ids = s1c["entity_id"].values; sx_ids = sxc["entity_id"].values
        tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4),
                                max_features=80000, sublinear_tf=True, dtype=np.float32)
        try:
            sx_v = normalize(tfidf.fit_transform(sx_vals), norm='l2')
            s1_v = normalize(tfidf.transform(s1_vals), norm='l2')
        except ValueError: continue # empty vocabulary
        
        BS = 500  # Reduced to prevent Out-Of-Memory crashes
        for st in range(0, len(s1_ids), BS):
            en = min(st+BS, len(s1_ids))
            sim = (s1_v[st:en] @ sx_v.T).tocsr()
            for i in range(sim.shape[0]):
                row = sim.getrow(i)
                if row.nnz == 0: continue
                if row.nnz > top_k:
                    tk = np.argpartition(-row.data, top_k)[:top_k]
                    sel = row.indices[tk]
                else: sel = row.indices
                cands[s1_ids[st+i]].update(sx_ids[sel])
            del sim; gc.collect()
        del sx_v, s1_v, tfidf; gc.collect()
    return cands

def trad_block(s1_df, sx_df):
    """Traditional blocking: prefix, firstword, pincode, clean_prefix, rare tokens."""
    cands = defaultdict(set)
    countries = sorted(set(s1_df["country_lower"].unique()) | set(sx_df["country_lower"].unique()))
    for country in countries:
        s1c = s1_df[s1_df["country_lower"]==country]
        sxc = sx_df[sx_df["country_lower"]==country]
        if len(s1c)==0 or len(sxc)==0: continue
        s1_ids = s1c["entity_id"].values; sx_ids = sxc["entity_id"].values
        # Key-based blocking
        for col in ["bk_prefix_country","bk_firstword_country","bk_pincode_country"]:
            if col not in sxc.columns: continue
            idx = defaultdict(list)
            for eid, k in zip(sx_ids, sxc[col].values):
                if k and isinstance(k,str) and "|" in k and k.split("|")[0]: idx[k].append(eid)
            for s1id, k in zip(s1_ids, s1c[col].values):
                if k and isinstance(k,str) and "|" in k and k.split("|")[0]:
                    m = idx.get(k,[])
                    if 0<len(m)<=MAX_BLOCK_SIZE: cands[s1id].update(m)
        # Clean-name prefix blocking
        s1_cpk = s1c["name_cln"].str.replace(" ","",regex=False).str[:5]+"|"+country
        sx_cpk = sxc["name_cln"].str.replace(" ","",regex=False).str[:5]+"|"+country
        idx = defaultdict(list)
        for eid, k in zip(sx_ids, sx_cpk.values):
            if k and k.split("|")[0]: idx[k].append(eid)
        for s1id, k in zip(s1_ids, s1_cpk.values):
            if k and k.split("|")[0]:
                m = idx.get(k,[])
                if 0<len(m)<=MAX_BLOCK_SIZE: cands[s1id].update(m)
        # Rare token inverted index
        sx_names = sxc["name_norm"].values
        freq = Counter()
        for nm in sx_names:
            if nm:
                for t in set(nm.split()):
                    if len(t)>=3: freq[t]+=1
        tok_idx = defaultdict(list)
        for eid, nm in zip(sx_ids, sx_names):
            if nm:
                for t in set(nm.split()):
                    if len(t)>=3 and freq.get(t,0)<=500: tok_idx[t].append(eid)
        for s1id, nm in zip(s1_ids, s1c["name_norm"].values):
            if not nm: continue
            cc = Counter()
            for t in set(nm.split()):
                if len(t)>=3 and t in tok_idx:
                    for sx_id in tok_idx[t]: cc[sx_id]+=1
            for sx_id, cnt in cc.items():
                if cnt>=1: cands[s1id].add(sx_id)
        gc.collect()
    return cands

def run_blocking(s1_df, s2_df, s3_df, label="train"):
    print(f"\n  ── Blocking: {label} ──")
    t0 = time.time()
    all_cands = defaultdict(set)
    for sx_df, sx_nm in [(s2_df,"S2"),(s3_df,"S3")]:
        print(f"  [{sx_nm}] TF-IDF Name blocking...")
        c1 = tfidf_block(s1_df, sx_df, col="name_exp", top_k=TFIDF_TOP_K)
        for k,v in c1.items(): all_cands[k].update(v)
        del c1; gc.collect()
        
        print(f"  [{sx_nm}] TF-IDF Address blocking...")
        ca = tfidf_block(s1_df, sx_df, col="addr_norm", top_k=TFIDF_TOP_K//2)
        for k,v in ca.items(): all_cands[k].update(v)
        del ca; gc.collect()

        print(f"  [{sx_nm}] Traditional blocking...")
        c2 = trad_block(s1_df, sx_df)
        for k,v in c2.items(): all_cands[k].update(v)
        del c2; gc.collect()
        
    # Ensure all S1 present
    for eid in s1_df["entity_id"]: 
        if eid not in all_cands: all_cands[eid]=set()
    # Cap
    capped=0
    for k in all_cands:
        if len(all_cands[k])>MAX_CANDIDATES:
            all_cands[k]=set(list(all_cands[k])[:MAX_CANDIDATES]); capped+=1
    total=sum(len(v) for v in all_cands.values())
    nw=sum(1 for v in all_cands.values() if v)
    print(f"  {label}: {nw:,} S1 with cands, {total:,} pairs, {capped} capped ({time.time()-t0:.0f}s)")
    return all_cands

# Run train blocking
train_cands = run_blocking(train_s1, train_s2, train_s3, "train")

# Blocking recall
total_true=0; found=0
for s1, true_m in gt_dict.items():
    for tid in true_m:
        total_true+=1
        if tid in train_cands.get(s1,set()): found+=1
recall = found/max(total_true,1)
print(f"\n  🎯 BLOCKING RECALL: {recall:.4f} ({found:,}/{total_true:,})")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: BUILD TF-IDF MATRICES FOR FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*80); print("SECTION 3: TF-IDF FEATURE MATRICES"); print("="*80)

def build_tfidf_index(s1_df, s2_df, s3_df):
    """Build global TF-IDF matrices for cosine similarity features."""
    t0 = time.time()
    all_names = np.concatenate([s1_df["name_exp"].values, s2_df["name_exp"].values, s3_df["name_exp"].values])
    all_addrs = np.concatenate([s1_df["addr_norm"].values, s2_df["addr_norm"].values, s3_df["addr_norm"].values])
    all_ids   = np.concatenate([s1_df["entity_id"].values, s2_df["entity_id"].values, s3_df["entity_id"].values])
    eid2idx = {eid:i for i,eid in enumerate(all_ids)}
    
    print("  Building name char TF-IDF...")
    name_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4), max_features=80000,
                                  sublinear_tf=True, dtype=np.float32)
    name_mat = normalize(name_tfidf.fit_transform(all_names), norm='l2')
    
    print("  Building name word TF-IDF...")
    name_word_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=50000,
                                      sublinear_tf=True, dtype=np.float32)
    name_w_mat = normalize(name_word_tfidf.fit_transform(all_names), norm='l2')
    
    print("  Building addr char TF-IDF...")
    addr_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4), max_features=50000,
                                  sublinear_tf=True, dtype=np.float32)
    addr_mat = normalize(addr_tfidf.fit_transform(all_addrs), norm='l2')
    print(f"  Built in {time.time()-t0:.1f}s")
    
    del all_names, all_addrs, all_ids, name_tfidf, name_word_tfidf, addr_tfidf; gc.collect()
    return name_mat, name_w_mat, addr_mat, eid2idx

train_name_mat, train_name_w_mat, train_addr_mat, train_eid2idx = build_tfidf_index(train_s1, train_s2, train_s3)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FEATURE ENGINEERING (TRAIN)
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*80); print("SECTION 4: FEATURE ENGINEERING (TRAIN)"); print("="*80)

def build_entity_lookup(df):
    """Build fast entity lookup dict."""
    lookup = {}
    eids = df["entity_id"].values
    norms = df["name_norm"].values; addrs = df["addr_norm"].values
    pins = df["pincode"].fillna("").values; fws = df["first_word"].fillna("").values
    exps = df["name_exp"].values; clns = df["name_cln"].values
    for i in range(len(df)):
        nm = norms[i] if isinstance(norms[i],str) else ""
        ad = addrs[i] if isinstance(addrs[i],str) else ""
        ex = exps[i] if isinstance(exps[i],str) else ""
        cl = clns[i] if isinstance(clns[i],str) else ""
        lookup[eids[i]] = {
            "name": nm, "addr": ad, "pin": pins[i] if isinstance(pins[i],str) else "",
            "fw": fws[i] if isinstance(fws[i],str) else "",
            "exp": ex, "cln": cl, "eid": eids[i],
            "ntok": frozenset(nm.split()) if nm else frozenset(),
            "atok": frozenset(ad.split()) if ad else frozenset(),
            "ctok": frozenset(cl.split()) if cl else frozenset(),
        }
    return lookup

def compute_features(s1_ids, sx_ids, s1_lk, sx_lk, name_mat, name_w_mat, addr_mat, eid2idx):
    """Compute features for candidate pairs. Returns DataFrame."""
    n = len(s1_ids)
    feats = {k: np.zeros(n, dtype=np.float32) for k in [
        "name_tfidf", "name_w_tfidf", "addr_tfidf","name_jw","name_lev","addr_jw",
        "name_jaccard","name_ovlp_s1","name_ovlp_sx","name_c3g","name_c4g",
        "name_exact","name_len_ratio","name_len_diff","name_tc_diff","name_tc_ratio",
        "fw_match","addr_jaccard","addr_ovlp_s1","addr_ovlp_sx","addr_c3g",
        "addr_exact","addr_len_ratio","addr_num_ovlp",
        "pin_match","pin_both_miss","pin_one_miss","is_s2",
        "addr_empty_s1","addr_empty_sx",
        "cln_jaccard","cln_exact","exp_jw", "name_token_sort", "name_token_set"
    ]}
    
    # ── Vectorized TF-IDF cosine ──
    s1_idx = np.array([eid2idx[eid] for eid in s1_ids])
    sx_idx = np.array([eid2idx[eid] for eid in sx_ids])
    feats["name_tfidf"] = np.array(name_mat[s1_idx].multiply(name_mat[sx_idx]).sum(axis=1)).flatten()
    feats["name_w_tfidf"] = np.array(name_w_mat[s1_idx].multiply(name_w_mat[sx_idx]).sum(axis=1)).flatten()
    feats["addr_tfidf"] = np.array(addr_mat[s1_idx].multiply(addr_mat[sx_idx]).sum(axis=1)).flatten()
    
    # ── Loop-based features ──
    for i in range(n):
        s1 = s1_lk[s1_ids[i]]; sx = sx_lk[sx_ids[i]]
        n1=s1["name"]; n2=sx["name"]; a1=s1["addr"]; a2=sx["addr"]
        e1=s1["exp"]; e2=sx["exp"]; c1=s1["cln"]; c2=sx["cln"]
        
        # Rapidfuzz string distances
        feats["name_jw"][i] = JaroWinkler.similarity(n1,n2) if n1 and n2 else (1.0 if n1==n2 else 0.0)
        feats["name_lev"][i] = RFLevenshtein.normalized_similarity(n1,n2) if n1 and n2 else (1.0 if n1==n2 else 0.0)
        feats["addr_jw"][i] = JaroWinkler.similarity(a1,a2) if a1 and a2 else (1.0 if a1==a2 else 0.0)
        feats["exp_jw"][i] = JaroWinkler.similarity(e1,e2) if e1 and e2 else (1.0 if e1==e2 else 0.0)
        feats["name_token_sort"][i] = token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        feats["name_token_set"][i]  = token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        
        # Token features
        nt1=s1["ntok"]; nt2=sx["ntok"]; at1=s1["atok"]; at2=sx["atok"]
        ct1=s1["ctok"]; ct2=sx["ctok"]
        feats["name_jaccard"][i] = jaccard(nt1,nt2)
        feats["name_ovlp_s1"][i] = overlap(nt1,nt2)
        feats["name_ovlp_sx"][i] = overlap(nt2,nt1)
        feats["addr_jaccard"][i] = jaccard(at1,at2)
        feats["addr_ovlp_s1"][i] = overlap(at1,at2)
        feats["addr_ovlp_sx"][i] = overlap(at2,at1)
        feats["cln_jaccard"][i] = jaccard(ct1,ct2)
        feats["cln_exact"][i] = 1.0 if c1==c2 and c1 else 0.0
        
        # Char n-grams
        feats["name_c3g"][i] = jaccard(ngram_set(n1,3), ngram_set(n2,3))
        feats["name_c4g"][i] = jaccard(ngram_set(n1,4), ngram_set(n2,4))
        feats["addr_c3g"][i] = jaccard(ngram_set(a1,3), ngram_set(a2,3))
        
        # Lengths
        ln1=len(n1); ln2=len(n2); la1=len(a1); la2=len(a2)
        feats["name_exact"][i] = 1.0 if n1==n2 and n1 else 0.0
        feats["name_len_ratio"][i] = min(ln1,ln2)/max(ln1,ln2) if max(ln1,ln2)>0 else 1.0
        feats["name_len_diff"][i] = abs(ln1-ln2)
        tc1=len(nt1); tc2=len(nt2)
        feats["name_tc_diff"][i] = abs(tc1-tc2)
        feats["name_tc_ratio"][i] = min(tc1,tc2)/max(tc1,tc2) if max(tc1,tc2)>0 else 1.0
        feats["fw_match"][i] = 1.0 if s1["fw"]==sx["fw"] and s1["fw"] else 0.0
        
        feats["addr_exact"][i] = 1.0 if a1==a2 and a1 else 0.0
        feats["addr_len_ratio"][i] = min(la1,la2)/max(la1,la2) if max(la1,la2)>0 else 1.0
        
        # Address numbers
        nums1 = frozenset(re.findall(r'\d+',a1)); nums2 = frozenset(re.findall(r'\d+',a2))
        if nums1 and nums2: feats["addr_num_ovlp"][i] = len(nums1&nums2)/len(nums1|nums2)
        elif not nums1 and not nums2: feats["addr_num_ovlp"][i] = 1.0
        
        # Pincode
        p1=s1["pin"]; p2=sx["pin"]
        feats["pin_match"][i] = 1.0 if p1==p2 and p1 else 0.0
        feats["pin_both_miss"][i] = 1.0 if not p1 and not p2 else 0.0
        feats["pin_one_miss"][i] = 1.0 if bool(p1)!=bool(p2) else 0.0
        
        feats["is_s2"][i] = 1.0 if sx["eid"].startswith("S2-") else 0.0
        feats["addr_empty_s1"][i] = 1.0 if not a1 else 0.0
        feats["addr_empty_sx"][i] = 1.0 if not a2 else 0.0
    
    return pd.DataFrame(feats)


# Build entity lookups for train
print("Building entity lookups...")
t0 = time.time()
s1_lk = build_entity_lookup(train_s1)

# Filter S2/S3 to only needed entities
needed_sx = set()
for cands in train_cands.values(): needed_sx.update(cands)
sx_combined = pd.concat([train_s2[train_s2["entity_id"].isin(needed_sx)],
                          train_s3[train_s3["entity_id"].isin(needed_sx)]], ignore_index=True)
print(f"Filtered SX to {len(sx_combined):,} needed entities")
sx_lk = build_entity_lookup(sx_combined)
del sx_combined; gc.collect()
print(f"Lookups built in {time.time()-t0:.1f}s")

# Generate training pairs with negative sampling
print("Generating training pairs...")
t0 = time.time()
import random; random.seed(42)
pair_s1 = []; pair_sx = []; pair_labels = []

for s1_id, cand_ids in train_cands.items():
    if not cand_ids: continue
    true_m = gt_dict.get(s1_id, set())
    pos = [c for c in cand_ids if c in true_m]
    neg = [c for c in cand_ids if c not in true_m]
    if len(neg) > NEG_RATIO:
        neg = random.sample(neg, NEG_RATIO)
    for c in pos:
        if c in sx_lk:
            pair_s1.append(s1_id); pair_sx.append(c); pair_labels.append(1)
    for c in neg:
        if c in sx_lk:
            pair_s1.append(s1_id); pair_sx.append(c); pair_labels.append(0)

print(f"Training pairs: {len(pair_s1):,} ({sum(pair_labels):,} pos, {len(pair_labels)-sum(pair_labels):,} neg)")

# Compute features in batches
BATCH = 100000
all_feat_dfs = []
for st in range(0, len(pair_s1), BATCH):
    en = min(st+BATCH, len(pair_s1))
    if st % 500000 == 0:
        print(f"  Features: {st:,}/{len(pair_s1):,} ({st/len(pair_s1)*100:.0f}%)")
    batch_df = compute_features(pair_s1[st:en], pair_sx[st:en], s1_lk, sx_lk,
                                 train_name_mat, train_name_w_mat, train_addr_mat, train_eid2idx)
    all_feat_dfs.append(batch_df)

feat_df = pd.concat(all_feat_dfs, ignore_index=True)
feat_df["s1_id"] = pair_s1; feat_df["sx_id"] = pair_sx; feat_df["label"] = pair_labels
del all_feat_dfs, pair_s1, pair_sx, pair_labels; gc.collect()
print(f"Train features: {feat_df.shape} in {time.time()-t0:.1f}s")

# Free train TF-IDF matrices
del train_name_mat, train_name_w_mat, train_addr_mat, train_eid2idx; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MODEL TRAINING (ENSEMBLE LGBM + XGB)
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*80); print("SECTION 5: MODEL TRAINING (ENSEMBLE)"); print("="*80)
t0 = time.time()

feature_cols = sorted([c for c in feat_df.columns if c not in ["s1_id","sx_id","label"]])
print(f"Features ({len(feature_cols)}): {feature_cols}")

X = feat_df[feature_cols].values.astype(np.float32)
y = feat_df["label"].values.astype(int)
s1_arr = feat_df["s1_id"].values; sx_arr = feat_df["sx_id"].values

gkf = GroupKFold(n_splits=5)
lgb_oof = np.zeros(len(y)); xgb_oof = np.zeros(len(y))
lgb_best_iters = []; xgb_best_iters = []

n_pos = y.sum(); n_neg = len(y)-n_pos
spw = n_neg / max(n_pos,1)
print(f"Class balance: {n_pos:,} pos, {n_neg:,} neg, scale_pos_weight={spw:.2f}")

lgb_params = {
    'objective':'binary','metric':'binary_logloss','boosting_type':'gbdt',
    'learning_rate':0.05,'num_leaves':127,'max_depth':10,
    'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'min_child_samples':50,'scale_pos_weight':spw,
    'random_state':42,'verbose':-1,'n_jobs':-1,
}

xgb_params = {
    'objective':'binary:logistic','eval_metric':'logloss','tree_method':'hist',
    'learning_rate':0.05,'max_depth':8,'subsample':0.8,'colsample_bytree':0.8,
    'scale_pos_weight':spw,'random_state':42,'n_jobs':-1,
}

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=s1_arr)):
    print(f"\n  Fold {fold+1}/5...")
    
    # Train LightGBM
    dtrain_lgb = lgb.Dataset(X[tr_idx], label=y[tr_idx])
    dval_lgb   = lgb.Dataset(X[va_idx], label=y[va_idx], reference=dtrain_lgb)
    model_lgb = lgb.train(lgb_params, dtrain_lgb, num_boost_round=2000, valid_sets=[dval_lgb],
                           callbacks=[lgb.early_stopping(100, verbose=False)])
    lgb_oof[va_idx] = model_lgb.predict(X[va_idx])
    lgb_best_iters.append(model_lgb.best_iteration)
    print(f"    LGBM best_iter={model_lgb.best_iteration}")
    
    # Train XGBoost
    dtrain_xgb = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
    dval_xgb   = xgb.DMatrix(X[va_idx], label=y[va_idx])
    model_xgb = xgb.train(xgb_params, dtrain_xgb, num_boost_round=2000, 
                          evals=[(dval_xgb, 'val')], early_stopping_rounds=100, verbose_eval=False)
    xgb_oof[va_idx] = model_xgb.predict(dval_xgb)
    xgb_best_iters.append(model_xgb.best_iteration)
    print(f"    XGB  best_iter={model_xgb.best_iteration}")

    del dtrain_lgb, dval_lgb, model_lgb, dtrain_xgb, dval_xgb, model_xgb; gc.collect()

# Blend OOF
val_probas_all = (lgb_oof + xgb_oof) / 2.0

# Find optimal threshold using OOF predictions
print("\nFinding optimal threshold...")
s1_to_preds = defaultdict(list)
for prob, s1, sx in zip(val_probas_all, s1_arr, sx_arr):
    s1_to_preds[s1].append((sx, prob))

best_t = 0.5; best_f05 = 0.0
for t in np.arange(0.20, 0.95, 0.02):
    pred_dict = {}
    for s1, sx_probs in s1_to_preds.items():
        pred_dict[s1] = set(sx for sx, p in sx_probs if p >= t)
    scores = []
    for s1 in gt_dict:
        true_s = gt_dict[s1]; pred_s = pred_dict.get(s1, set())
        if not true_s and not pred_s: scores.append(1.0)
        elif not true_s and pred_s: scores.append(0.0)
        elif true_s and not pred_s: scores.append(0.0)
        else:
            tp = len(true_s & pred_s)
            p = tp/len(pred_s) if pred_s else 0; r = tp/len(true_s) if true_s else 0
            scores.append(fbeta(p, r))
    f05 = np.mean(scores)
    if f05 > best_f05: best_f05 = f05; best_t = t

print(f"  BEST THRESHOLD: {best_t:.3f}")
print(f"  BEST CV F_0.5:  {best_f05:.4f}")

# Retrain on ALL data
print("\nRetraining LGBM & XGB on ALL data...")
full_lgb = lgb.Dataset(X, label=y)
final_model_lgb = lgb.train(lgb_params, full_lgb, num_boost_round=int(np.mean(lgb_best_iters)))

full_xgb = xgb.DMatrix(X, label=y)
final_model_xgb = xgb.train(xgb_params, full_xgb, num_boost_round=int(np.mean(xgb_best_iters)))

print(f"\nTraining done in {time.time()-t0:.1f}s")

# Free train data
del feat_df, X, y, s1_arr, sx_arr, val_probas_all, s1_to_preds, full_lgb, full_xgb
del s1_lk, sx_lk, train_cands
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: TEST BLOCKING + FEATURES + PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*80); print("SECTION 6: TEST PREDICTION"); print("="*80)

test_cands = run_blocking(test_s1, test_s2, test_s3, "test")
test_name_mat, test_name_w_mat, test_addr_mat, test_eid2idx = build_tfidf_index(test_s1, test_s2, test_s3)

print("Building test entity lookups...")
t0 = time.time()
s1_lk = build_entity_lookup(test_s1)
needed_sx = set()
for cands in test_cands.values(): needed_sx.update(cands)
sx_combined = pd.concat([test_s2[test_s2["entity_id"].isin(needed_sx)],
                          test_s3[test_s3["entity_id"].isin(needed_sx)]], ignore_index=True)
print(f"Filtered test SX to {len(sx_combined):,} entities")
sx_lk = build_entity_lookup(sx_combined)
del sx_combined; gc.collect()

print("Predicting test pairs...")
t0 = time.time()
matches = defaultdict(list)
total_pairs = sum(len(v) for v in test_cands.values())

all_s1 = []; all_sx = []
for s1_id, cand_ids in test_cands.items():
    for sx_id in cand_ids:
        if sx_id in sx_lk:
            all_s1.append(s1_id); all_sx.append(sx_id)

print(f"Total test pairs to score: {len(all_s1):,}")

BATCH = 100000
for st in range(0, len(all_s1), BATCH):
    en = min(st+BATCH, len(all_s1))
    if st % 1000000 == 0:
        print(f"  {st:,}/{len(all_s1):,} ({st/len(all_s1)*100:.1f}%)")
    
    batch_feat = compute_features(all_s1[st:en], all_sx[st:en], s1_lk, sx_lk,
                                   test_name_mat, test_name_w_mat, test_addr_mat, test_eid2idx)
    
    # Ensemble prediction
    X_test = batch_feat[feature_cols].values.astype(np.float32)
    p_lgb = final_model_lgb.predict(X_test)
    p_xgb = final_model_xgb.predict(xgb.DMatrix(X_test))
    proba = (p_lgb + p_xgb) / 2.0
    
    for i, (s1_id, sx_id, p) in enumerate(zip(all_s1[st:en], all_sx[st:en], proba)):
        if p >= best_t:
            matches[s1_id].append(sx_id)
    
    del batch_feat, proba, X_test; gc.collect()

print(f"Prediction done in {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: GENERATE SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n"+"="*80); print("SECTION 7: SUBMISSION"); print("="*80)

all_test_s1 = sorted(test_s1["entity_id"].tolist())

n_match=0; n_single=0
with open(OUT/"matching_results.tsv","w",encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s1_id in all_test_s1:
        if s1_id in matches:
            f.write(f"{s1_id}\t{','.join(sorted(set(matches[s1_id])))}\n"); n_match+=1
        else:
            f.write(f"{s1_id}\t\n"); n_single+=1

print(f"matching_results.tsv: {n_match:,} with matches, {n_single:,} singletons")

with open(OUT/"candidate_pairs.tsv","w",encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_id in all_test_s1:
        cands = test_cands.get(s1_id, set())
        if cands:
            f.write(f"{s1_id}\t{','.join(sorted(cands))}\n")
        else:
            f.write(f"{s1_id}\t\n")

total_time = time.time()-T0
print(f"\n{'='*80}")
print(f"PIPELINE COMPLETE in {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"CV F_0.5: {best_f05:.4f} | Threshold: {best_t:.3f}")
print(f"Output: {OUT/'matching_results.tsv'}")
print(f"{'='*80}")
