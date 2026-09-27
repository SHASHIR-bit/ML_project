"""
Step 5: Model Training (LightGBM)
Trains a LightGBM model to predict matches based on computed features.
Uses custom Group K-Fold validation grouped by S1 entity to prevent leakage.
Optimizes for the F_0.5 metric.
"""
import pandas as pd
import numpy as np
import os, sys, io, gc, time, pickle, json
from pathlib import Path
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', write_through=True)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "processed"
FEATURES_DIR = PROJECT_DIR / "features"
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def f_beta_score(precision, recall, beta=0.5):
    """Compute F_beta score."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def compute_macro_f05(gt_dict, pred_dict):
    """Compute macro-averaged F_0.5 score."""
    scores = []
    for s1_id in gt_dict:
        true_set = gt_dict[s1_id]
        pred_set = pred_dict.get(s1_id, set())
        
        if not true_set and not pred_set:
            scores.append(1.0)
        elif not true_set and pred_set:
            scores.append(0.0)
        elif true_set and not pred_set:
            scores.append(0.0)
        else:
            tp = len(true_set & pred_set)
            precision = tp / len(pred_set) if pred_set else 0
            recall = tp / len(true_set) if true_set else 0
            scores.append(f_beta_score(precision, recall, beta=0.5))
    
    return np.mean(scores)


def find_optimal_threshold(y_true_proba, y_true_labels, s1_ids, sx_ids, gt_dict, thresholds=None):
    """Find the threshold that maximizes macro F_0.5."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.025)
    
    best_t = 0.5
    best_f05 = 0.0
    
    # Pre-group predictions by S1 to speed up threshold search
    s1_to_sx_probs = defaultdict(list)
    for prob, s1, sx in zip(y_true_proba, s1_ids, sx_ids):
        s1_to_sx_probs[s1].append((sx, prob))
        
    for t in thresholds:
        pred_dict = {}
        for s1, sx_probs in s1_to_sx_probs.items():
            pred_dict[s1] = set(sx for sx, prob in sx_probs if prob >= t)
            
        # Ensure all GT entities are in pred_dict
        for s1_id in gt_dict:
            if s1_id not in pred_dict:
                pred_dict[s1_id] = set()
                
        f05 = compute_macro_f05(gt_dict, pred_dict)
        n_pos = sum(1 for p in y_true_proba if p >= t)
        
        if f05 > best_f05:
            best_f05 = f05
            best_t = t
            
    return best_t, best_f05

from collections import defaultdict

def main():
    print("=" * 80)
    print("STEP 5: MODEL TRAINING (LightGBM)")
    print("=" * 80)
    
    t_total = time.time()
    
    # Load features
    print(f"\n  Loading training features...")
    df = pd.read_parquet(FEATURES_DIR / "train_features.parquet")
    print(f"  Loaded {len(df):,} pairs")
    
    id_cols = ["s1_id", "sx_id", "label"]
    feature_cols = sorted([c for c in df.columns if c not in id_cols])
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    
    # Downcast features to float32
    df[feature_cols] = df[feature_cols].astype(np.float32)
    
    X = df[feature_cols].values
    y = df["label"].values.astype(int)
    s1_ids = df["s1_id"].values
    sx_ids = df["sx_id"].values
    
    # Build ground truth dict
    print(f"  Loading ground truth...")
    gt = pd.read_parquet(PROCESSED_DIR / "train_gt.parquet")
    gt_dict = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if pd.isna(matched) or str(matched).strip() == "":
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(m.strip() for m in str(matched).split(",") if m.strip())
    del gt
    
    # Train/Val Split (GroupKFold on S1_ID)
    print(f"\n  Splitting data...")
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(X, y, groups=s1_ids))
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    s1_val, sx_val = s1_ids[val_idx], sx_ids[val_idx]
    
    # Clean up some memory
    del df, X, y, train_idx, val_idx
    gc.collect()
    
    print(f"  Train: {len(X_train):,} pairs ({y_train.sum():,} pos)")
    print(f"  Val:   {len(X_val):,} pairs ({y_val.sum():,} pos)")
    
    # Calculate scale_pos_weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    # LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train
    print(f"\n  Training LightGBM on NVIDIA RTX 4050 GPU...")
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'feature_fraction': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbose': -1,
        'device': 'cpu',
        'n_jobs': -1
    }
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=50)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    # Feature importance
    print("\n  Feature Importance (top 15):")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(15).iterrows():
        print(f"    {row['feature']:25s} {row['importance']:.1f}")
        
    # Validate and find threshold
    print(f"\n  Validating and finding threshold...")
    val_proba = model.predict(X_val)
    
    # Filter GT dict to validation S1s
    val_s1_set = set(s1_val)
    val_gt_dict = {k: v for k, v in gt_dict.items() if k in val_s1_set}
    
    best_t, best_f05 = find_optimal_threshold(
        val_proba, y_val, s1_val, sx_val, val_gt_dict,
        thresholds=np.arange(0.1, 0.95, 0.025)
    )
    
    print(f"\n  BEST THRESHOLD: {best_t:.3f}")
    print(f"  BEST MACRO F_0.5: {best_f05:.4f}")
    
    # Save model and metadata
    model_path = MODELS_DIR / "lgb_matcher.txt"
    model.save_model(str(model_path))
    
    metadata = {
        "feature_cols": feature_cols,
        "best_threshold": float(best_t),
        "best_f05": float(best_f05),
        "n_train_pairs": int(len(X_train)),
        "n_val_pairs": int(len(X_val)),
        "pos_train": int(y_train.sum()),
        "pos_val": int(y_val.sum()),
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(model.best_iteration)
    }
    
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    # Retrain on ALL data
    print(f"\n  Retraining on ALL data ({model.best_iteration} rounds)...")
    
    # Reload full X, y since we deleted them
    df = pd.read_parquet(FEATURES_DIR / "train_features.parquet", columns=feature_cols + ["label"])
    X_full = df[feature_cols].values.astype(np.float32)
    y_full = df["label"].values.astype(int)
    del df
    gc.collect()
    
    full_data = lgb.Dataset(X_full, label=y_full)
    model_full = lgb.train(
        params,
        full_data,
        num_boost_round=model.best_iteration
    )
    
    full_model_path = MODELS_DIR / "lgb_matcher_full.txt"
    model_full.save_model(str(full_model_path))
    
    print(f"\n>>> MODEL TRAINING COMPLETE in {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
